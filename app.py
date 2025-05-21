#!/usr/bin/env python3
'''
Web-based Crypto Investment Strategy service - Render Fixed with Improved Rate Limit Handling
'''

import os
import sys
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import logging
import time
import random
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
import threading
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- Improved Caching Setup ---
api_cache = {}
CACHE_DURATION = datetime.timedelta(hours=6)  # Increased from 30 minutes to 6 hours
CACHE_FILE = 'data/api_cache.json'
# --- End Caching Setup ---

# --- Rate Limiting Setup ---
RATE_LIMIT_RESET = datetime.timedelta(minutes=60)  # Time to wait after hitting rate limit
last_rate_limit_hit = None
api_call_timestamps = []
MAX_CALLS_PER_MINUTE = 10  # Conservative limit for CoinGecko free tier
BACKOFF_FACTOR = 1.5  # Exponential backoff factor
MAX_RETRIES = 3  # Maximum number of retries for API calls
# --- End Rate Limiting Setup ---

# --- Improved Configuration Section --- 

def get_env_var(name, default=None, required=False):
    value = os.environ.get(name, default)
    if name in ['SECRET_KEY', 'DATABASE_URL'] and value:
        log_value = '[SET]'
    elif value:
        log_value = value
    else:
        log_value = '[NOT SET]'
        
    logger.info(f'Environment variable {name}: {log_value}')
    if required and not value:
        logger.error(f'Required environment variable {name} is not set. Exiting.')
        sys.exit(f'Error: Missing required environment variable {name}')
    return value

database_url = get_env_var('DATABASE_URL', required=True)
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
    logger.info('Modified DATABASE_URL to use postgresql:// prefix')

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = get_env_var('SECRET_KEY', required=True)

# --- End Improved Configuration Section ---

try:
    db = SQLAlchemy(app)
    bcrypt = Bcrypt(app)
    login_manager = LoginManager(app)
    login_manager.login_view = 'login'
    logger.info('Flask extensions initialized successfully')
except Exception as e:
    logger.error(f'Error initializing Flask extensions: {str(e)}', exc_info=True)
    sys.exit('Failed to initialize Flask extensions.')

os.makedirs('data', exist_ok=True)
os.makedirs('static/img', exist_ok=True)

# --- Cache Management Functions ---
def load_cache_from_disk():
    global api_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                
            # Convert string timestamps back to datetime objects
            restored_cache = {}
            for key_str, (data, timestamp_str) in cache_data.items():
                # Handle tuple keys (stored as strings in JSON)
                if key_str.startswith('(') and key_str.endswith(')'):
                    # Parse the tuple key
                    parts = key_str.strip('()').split(',')
                    if len(parts) == 2:
                        key = (parts[0].strip(' "\''), int(parts[1]))
                    else:
                        key = key_str
                else:
                    key = key_str
                    
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                restored_cache[key] = (data, timestamp)
                
            api_cache = restored_cache
            logger.info(f'Loaded {len(api_cache)} items from cache file')
    except Exception as e:
        logger.error(f'Error loading cache from disk: {e}', exc_info=True)
        api_cache = {}

def save_cache_to_disk():
    try:
        # Convert datetime objects to strings for JSON serialization
        serializable_cache = {}
        for key, (data, timestamp) in api_cache.items():
            # Convert tuple keys to strings
            if isinstance(key, tuple):
                key_str = str(key)
            else:
                key_str = key
                
            serializable_cache[key_str] = (data, timestamp.isoformat())
            
        with open(CACHE_FILE, 'w') as f:
            json.dump(serializable_cache, f)
        logger.info(f'Saved {len(api_cache)} items to cache file')
    except Exception as e:
        logger.error(f'Error saving cache to disk: {e}', exc_info=True)

# Load cache at startup
load_cache_from_disk()

# Schedule periodic cache saving
def schedule_cache_save():
    save_cache_to_disk()
    # Schedule next save in 15 minutes
    threading.Timer(900, schedule_cache_save).start()

# Start the cache saving schedule
schedule_cache_save()

# --- Rate Limit Management Functions ---
def can_make_api_call():
    global api_call_timestamps, last_rate_limit_hit
    
    # If we hit a rate limit recently, enforce a cooling period
    if last_rate_limit_hit is not None:
        time_since_limit = datetime.datetime.utcnow() - last_rate_limit_hit
        if time_since_limit < RATE_LIMIT_RESET:
            logger.warning(f'Still in rate limit cooling period. {(RATE_LIMIT_RESET - time_since_limit).total_seconds():.0f} seconds remaining.')
            return False
        else:
            # Reset after cooling period
            last_rate_limit_hit = None
            api_call_timestamps = []
    
    # Clean up old timestamps
    now = datetime.datetime.utcnow()
    one_minute_ago = now - datetime.timedelta(minutes=1)
    api_call_timestamps = [ts for ts in api_call_timestamps if ts > one_minute_ago]
    
    # Check if we're under the rate limit
    return len(api_call_timestamps) < MAX_CALLS_PER_MINUTE

def record_api_call():
    global api_call_timestamps
    api_call_timestamps.append(datetime.datetime.utcnow())

def record_rate_limit_hit():
    global last_rate_limit_hit
    last_rate_limit_hit = datetime.datetime.utcnow()
    logger.warning(f'Rate limit hit. Cooling period started for {RATE_LIMIT_RESET.total_seconds():.0f} seconds.')

def api_request_with_backoff(url, params=None, timeout=10):
    """Make API request with exponential backoff for rate limiting"""
    if not can_make_api_call():
        logger.warning('Rate limit prevention: Skipping API call')
        raise Exception('Rate limit prevention active')
        
    for attempt in range(MAX_RETRIES):
        try:
            # Record this attempt
            record_api_call()
            
            # Add a small random delay to prevent concurrent requests
            time.sleep(random.uniform(0.1, 0.5) * attempt)
            
            # Make the request
            response = requests.get(url, params=params, timeout=timeout)
            
            # Check for rate limiting response
            if response.status_code == 429:
                record_rate_limit_hit()
                wait_time = (BACKOFF_FACTOR ** attempt) * 2
                logger.warning(f'Rate limit hit. Backing off for {wait_time:.1f} seconds (attempt {attempt+1}/{MAX_RETRIES})')
                time.sleep(wait_time)
                continue
                
            # Raise exception for other errors
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            # For network errors, apply backoff
            wait_time = (BACKOFF_FACTOR ** attempt) * 2
            if attempt < MAX_RETRIES - 1:
                logger.warning(f'Request failed: {e}. Retrying in {wait_time:.1f} seconds (attempt {attempt+1}/{MAX_RETRIES})')
                time.sleep(wait_time)
            else:
                logger.error(f'Request failed after {MAX_RETRIES} attempts: {e}')
                raise
    
    raise Exception(f'API request failed after {MAX_RETRIES} attempts')

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    subscription_type = db.Column(db.String(20), default='free')
    subscription_start = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    subscription_end = db.Column(db.DateTime, default=lambda: datetime.datetime.utcnow() + datetime.timedelta(days=30))
    preferences = db.relationship('UserPreference', backref='user', uselist=False)

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    email_frequency = db.Column(db.String(20), default='daily')
    preferred_coins = db.Column(db.String(200), default='BTC,ETH,SOL')
    preferred_indicators = db.Column(db.String(200), default='price,volume,rsi,macd')
    theme = db.Column(db.String(20), default='light')

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f'Error loading user {user_id}: {str(e)}', exc_info=True)
        return None

# --- Crypto data functions with Improved Caching and Rate Limiting ---
def get_crypto_data(symbol, days=30):
    cache_key = (symbol, days)
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if datetime.datetime.utcnow() - timestamp < CACHE_DURATION:
            logger.info(f'Returning cached crypto data for {symbol} ({days} days)')
            return cached_data.copy()
        else:
            logger.info(f'Cache expired for {symbol} ({days} days)')

    logger.info(f'Fetching crypto data for {symbol} ({days} days) from API')
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
        params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
        
        data = api_request_with_backoff(url, params=params)
        
        if not data or 'prices' not in data or 'total_volumes' not in data:
             logger.warning(f'Incomplete data received from CoinGecko for {symbol}')
             raise ValueError('Incomplete data from API')

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('timestamp', inplace=True)
        df['volume'] = volume_df['volume']
        
        api_cache[cache_key] = (df.copy(), datetime.datetime.utcnow())
        logger.info(f'Successfully fetched and cached data for {symbol}')
        save_cache_to_disk()  # Save cache after successful update
        return df
    except Exception as e:
        logger.error(f'Error fetching data for {symbol}: {e}', exc_info=True)
        
        # Check if we have expired cached data we can still use
        if cache_key in api_cache:
            cached_data, timestamp = api_cache[cache_key]
            age = datetime.datetime.utcnow() - timestamp
            if age < datetime.timedelta(days=7):  # Use expired cache for up to a week
                logger.warning(f'Using expired cached data for {symbol} (age: {age.total_seconds()/3600:.1f} hours)')
                return cached_data.copy()
        
        logger.warning(f'Returning dummy data for {symbol} due to API error')
        dates = pd.date_range(end=datetime.datetime.now(datetime.timezone.utc), periods=days)
        df = pd.DataFrame(index=dates)
        df['price'] = [50000 - i * 100 + i * i * 5 for i in range(days)]
        df['volume'] = [2000000000 - i * 10000000 + i * i * 500000 for i in range(days)]
        return df

def get_trending_coins():
    cache_key = 'trending_coins'
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if datetime.datetime.utcnow() - timestamp < CACHE_DURATION:
            logger.info('Returning cached trending coins data')
            return list(cached_data)
        else:
            logger.info('Cache expired for trending coins')

    logger.info('Fetching trending coins from API')
    try:
        url = 'https://api.coingecko.com/api/v3/search/trending'
        
        data = api_request_with_backoff(url)
        
        trending = []
        if data and 'coins' in data:
            for coin_data in data['coins'][:5]:
                item = coin_data.get('item', {})
                if not item: continue
                trending.append({
                    'id': item.get('id', 'N/A'), 'name': item.get('name', 'N/A'),
                    'symbol': item.get('symbol', 'N/A'), 'market_cap_rank': item.get('market_cap_rank', 'N/A'),
                    'price_btc': item.get('price_btc', 0), 'score': item.get('score', 0),
                    'thumb': item.get('thumb', '')
                })
            api_cache[cache_key] = (list(trending), datetime.datetime.utcnow())
            logger.info(f'Successfully fetched and cached {len(trending)} trending coins')
            save_cache_to_disk()  # Save cache after successful update
            return trending
        else:
            logger.warning('Incomplete data received from CoinGecko for trending coins')
            raise ValueError('Incomplete data from API')
    except Exception as e:
        logger.error(f'Error fetching trending coins: {e}', exc_info=True)
        
        # Check if we have expired cached data we can still use
        if cache_key in api_cache:
            cached_data, timestamp = api_cache[cache_key]
            age = datetime.datetime.utcnow() - timestamp
            if age < datetime.timedelta(days=7):  # Use expired cache for up to a week
                logger.warning(f'Using expired cached trending coins data (age: {age.total_seconds()/3600:.1f} hours)')
                return list(cached_data)
        
        logger.warning('Returning dummy trending coins data due to API error')
        return [
            {'id': 'solana', 'name': 'Solana', 'symbol': 'SOL', 'market_cap_rank': 5, 'price_btc': 0.00123, 'score': 100, 'thumb': ''},
            {'id': 'cardano', 'name': 'Cardano', 'symbol': 'ADA', 'market_cap_rank': 8, 'price_btc': 0.00002, 'score': 98, 'thumb': ''},
            {'id': 'polkadot', 'name': 'Polkadot', 'symbol': 'DOT', 'market_cap_rank': 12, 'price_btc': 0.00015, 'score': 95, 'thumb': ''},
            {'id': 'avalanche', 'name': 'Avalanche', 'symbol': 'AVAX', 'market_cap_rank': 10, 'price_btc': 0.00045, 'score': 93, 'thumb': ''},
            {'id': 'chainlink', 'name': 'Chainlink', 'symbol': 'LINK', 'market_cap_rank': 15, 'price_btc': 0.00018, 'score': 90, 'thumb': ''}
        ]

def get_economic_indicators():
    logger.info('Fetching economic indicators (using dummy data)')
    return {
        'usd_index': 104.2, 'usd_trend': 'rising',
        'inflation_rate': 3.5, 'inflation_trend': 'stable',
        'interest_rate': 5.25, 'interest_trend': 'stable',
        'global_liquidity': 'moderate', 'liquidity_trend': 'tightening',
        'stock_market': 'bullish', 'bond_market': 'bearish',
        'gold_price': 2342.50, 'gold_trend': 'rising'
    }

def calculate_technical_indicators(df):
    logger.info('Calculating technical indicators')
    if df is None or df.empty: return pd.DataFrame()
    try:
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, 1e-6) 
        rs = avg_gain / avg_loss
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
        df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean() # Renamed 'signal' to 'signal_line' to avoid conflict
        df['macd_histogram'] = df['macd'] - df['signal_line']
        df['sma20'] = df['price'].rolling(window=20, min_periods=1).mean()
        df['std20'] = df['price'].rolling(window=20, min_periods=1).std().fillna(0)
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        df['sma50'] = df['price'].rolling(window=50, min_periods=1).mean()
        df['sma200'] = df['price'].rolling(window=200, min_periods=1).mean()
        logger.info('Technical indicators calculated successfully')
        return df.fillna(0)
    except Exception as e:
        logger.error(f'Error calculating technical indicators: {e}', exc_info=True)
        return df

def generate_chart(df, symbol, indicator_type):
    logger.info(f'Generating chart for {symbol} - {indicator_type}')
    if df is None or df.empty or indicator_type not in ['price', 'rsi', 'macd', 'volume']:
        logger.warning(f'Cannot generate chart for {symbol} - {indicator_type}. Invalid input.')
        return None
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        if indicator_type == 'price':
            ax.plot(df.index, df.get('price'), label='Price')
            ax.plot(df.index, df.get('sma20'), label='SMA 20', linestyle='--')
            ax.plot(df.index, df.get('upper_band'), label='Upper Band', linestyle=':', color='grey')
            ax.plot(df.index, df.get('lower_band'), label='Lower Band', linestyle=':', color='grey')
            ax.set_title(f'{symbol.upper()} Price & Bollinger Bands')
            ax.set_ylabel('Price (USD)')
        elif indicator_type == 'rsi':
            ax.plot(df.index, df.get('rsi'), label='RSI')
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax.set_title(f'{symbol.upper()} RSI')
            ax.set_ylabel('RSI'); ax.set_ylim(0, 100)
        elif indicator_type == 'macd':
            ax.plot(df.index, df.get('macd'), label='MACD')
            ax.plot(df.index, df.get('signal_line'), label='Signal Line', linestyle='--') # Use 'signal_line'
            hist_values = pd.to_numeric(df.get('macd_histogram'), errors='coerce').fillna(0)
            ax.bar(df.index, hist_values, label='Histogram', alpha=0.3, width=0.8)
            ax.set_title(f'{symbol.upper()} MACD')
            ax.set_ylabel('MACD Value')
        elif indicator_type == 'volume':
            ax.bar(df.index, df.get('volume'), label='Volume', alpha=0.6)
            ax.set_title(f'{symbol.upper()} Volume')
            ax.set_ylabel('Volume')
        ax.set_xlabel('Date'); ax.legend(); ax.grid(True)
        plt.xticks(rotation=45); plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png'); buffer.seek(0)
        plt.close(fig)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        logger.info(f'Chart generated successfully for {symbol} - {indicator_type}')
        return f'data:image/png;base64,{img_str}'
    except Exception as e:
        logger.error(f'Error generating chart for {symbol} - {indicator_type}: {e}', exc_info=True)
        return None

def get_asset_rotation_strategy():
    logger.info('Generating asset rotation strategy')
    try:
        btc_data = get_crypto_data('bitcoin', days=30)
        eth_data = get_crypto_data('ethereum', days=30)
        sol_data = get_crypto_data('solana', days=30)
        btc_perf = (btc_data['price'].iloc[-1] / btc_data['price'].iloc[0] - 1) * 100 if not btc_data.empty and len(btc_data['price']) > 1 else 0
        eth_perf = (eth_data['price'].iloc[-1] / eth_data['price'].iloc[0] - 1) * 100 if not eth_data.empty and len(eth_data['price']) > 1 else 0
        sol_perf = (sol_data['price'].iloc[-1] / sol_data['price'].iloc[0] - 1) * 100 if not sol_data.empty and len(sol_data['price']) > 1 else 0
        trending = get_trending_coins()
        econ = get_economic_indicators()
        allocations = []
        btc_alloc, eth_alloc, sol_alloc = 25, 20, 15 
        btc_signal_val, eth_signal_val, sol_signal_val = 'HOLD', 'HOLD', 'HOLD' # Renamed signal variables
        if btc_perf > 10 and econ.get('usd_trend') == 'falling': btc_alloc = 40; btc_signal_val = 'BUY'
        elif btc_perf < -10 and econ.get('usd_trend') == 'rising': btc_alloc = 10; btc_signal_val = 'SELL'
        allocations.append({'coin': 'Bitcoin (BTC)', 'allocation': btc_alloc, 'signal': btc_signal_val, 'performance': f'{btc_perf:.2f}%'})
        if eth_perf > 15: eth_alloc = 30; eth_signal_val = 'BUY'
        elif eth_perf < -15: eth_alloc = 10; eth_signal_val = 'SELL'
        allocations.append({'coin': 'Ethereum (ETH)', 'allocation': eth_alloc, 'signal': eth_signal_val, 'performance': f'{eth_perf:.2f}%'})
        if sol_perf > 20: sol_alloc = 20; sol_signal_val = 'BUY'
        elif sol_perf < -20: sol_alloc = 5; sol_signal_val = 'SELL'
        allocations.append({'coin': 'Solana (SOL)', 'allocation': sol_alloc, 'signal': sol_signal_val, 'performance': f'{sol_perf:.2f}%'})
        remaining = 100 - (btc_alloc + eth_alloc + sol_alloc)
        if trending and remaining > 0:
             per_trending = remaining / len(trending)
             for coin_item in trending:
                 allocations.append({'coin': f'{coin_item.get("name", "N/A")} ({coin_item.get("symbol", "N/A")})', 'allocation': round(per_trending), 'signal': 'BUY' if coin_item.get('score', 0) > 95 else 'RESEARCH', 'performance': 'Trending'})
        elif remaining > 0:
             allocations.append({'coin': 'Cash/Stablecoin', 'allocation': remaining, 'signal': 'HOLD', 'performance': 'N/A'})
        logger.info('Asset rotation strategy generated successfully')
        return allocations
    except Exception as e:
        logger.error(f'Error generating asset rotation strategy: {e}', exc_info=True)
        return []

def generate_buy_sell_recommendations():
    logger.info('Generating buy/sell recommendations')
    try:
        btc_data = calculate_technical_indicators(get_crypto_data('bitcoin', days=90))
        eth_data = calculate_technical_indicators(get_crypto_data('ethereum', days=90))
        sol_data = calculate_technical_indicators(get_crypto_data('solana', days=90))
        recommendations = []
        
        # Bitcoin analysis
        btc_rec = {'coin': 'Bitcoin (BTC)', 'price': f'${btc_data["price"].iloc[-1]:,.2f}', 'signals': []}
        if btc_data['rsi'].iloc[-1] > 70: btc_rec['signals'].append({'indicator': 'RSI', 'signal': 'SELL', 'value': f'{btc_data["rsi"].iloc[-1]:.2f}'})
        elif btc_data['rsi'].iloc[-1] < 30: btc_rec['signals'].append({'indicator': 'RSI', 'signal': 'BUY', 'value': f'{btc_data["rsi"].iloc[-1]:.2f}'})
        else: btc_rec['signals'].append({'indicator': 'RSI', 'signal': 'NEUTRAL', 'value': f'{btc_data["rsi"].iloc[-1]:.2f}'})
        
        if btc_data['macd'].iloc[-1] > btc_data['signal_line'].iloc[-1]: btc_rec['signals'].append({'indicator': 'MACD', 'signal': 'BUY', 'value': f'{btc_data["macd"].iloc[-1]:.2f}'})
        else: btc_rec['signals'].append({'indicator': 'MACD', 'signal': 'SELL', 'value': f'{btc_data["macd"].iloc[-1]:.2f}'})
        
        if btc_data['price'].iloc[-1] > btc_data['upper_band'].iloc[-1]: btc_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'SELL', 'value': 'Upper band crossed'})
        elif btc_data['price'].iloc[-1] < btc_data['lower_band'].iloc[-1]: btc_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'BUY', 'value': 'Lower band crossed'})
        else: btc_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'NEUTRAL', 'value': 'Within bands'})
        
        if btc_data['sma50'].iloc[-1] > btc_data['sma200'].iloc[-1]: btc_rec['signals'].append({'indicator': 'Golden Cross', 'signal': 'BUY', 'value': 'SMA50 > SMA200'})
        else: btc_rec['signals'].append({'indicator': 'Death Cross', 'signal': 'SELL', 'value': 'SMA50 < SMA200'})
        
        buy_signals = sum(1 for s in btc_rec['signals'] if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in btc_rec['signals'] if s['signal'] == 'SELL')
        if buy_signals > sell_signals: btc_rec['overall'] = 'BUY'
        elif sell_signals > buy_signals: btc_rec['overall'] = 'SELL'
        else: btc_rec['overall'] = 'HOLD'
        recommendations.append(btc_rec)
        
        # Ethereum analysis
        eth_rec = {'coin': 'Ethereum (ETH)', 'price': f'${eth_data["price"].iloc[-1]:,.2f}', 'signals': []}
        if eth_data['rsi'].iloc[-1] > 70: eth_rec['signals'].append({'indicator': 'RSI', 'signal': 'SELL', 'value': f'{eth_data["rsi"].iloc[-1]:.2f}'})
        elif eth_data['rsi'].iloc[-1] < 30: eth_rec['signals'].append({'indicator': 'RSI', 'signal': 'BUY', 'value': f'{eth_data["rsi"].iloc[-1]:.2f}'})
        else: eth_rec['signals'].append({'indicator': 'RSI', 'signal': 'NEUTRAL', 'value': f'{eth_data["rsi"].iloc[-1]:.2f}'})
        
        if eth_data['macd'].iloc[-1] > eth_data['signal_line'].iloc[-1]: eth_rec['signals'].append({'indicator': 'MACD', 'signal': 'BUY', 'value': f'{eth_data["macd"].iloc[-1]:.2f}'})
        else: eth_rec['signals'].append({'indicator': 'MACD', 'signal': 'SELL', 'value': f'{eth_data["macd"].iloc[-1]:.2f}'})
        
        if eth_data['price'].iloc[-1] > eth_data['upper_band'].iloc[-1]: eth_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'SELL', 'value': 'Upper band crossed'})
        elif eth_data['price'].iloc[-1] < eth_data['lower_band'].iloc[-1]: eth_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'BUY', 'value': 'Lower band crossed'})
        else: eth_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'NEUTRAL', 'value': 'Within bands'})
        
        if eth_data['sma50'].iloc[-1] > eth_data['sma200'].iloc[-1]: eth_rec['signals'].append({'indicator': 'Golden Cross', 'signal': 'BUY', 'value': 'SMA50 > SMA200'})
        else: eth_rec['signals'].append({'indicator': 'Death Cross', 'signal': 'SELL', 'value': 'SMA50 < SMA200'})
        
        buy_signals = sum(1 for s in eth_rec['signals'] if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in eth_rec['signals'] if s['signal'] == 'SELL')
        if buy_signals > sell_signals: eth_rec['overall'] = 'BUY'
        elif sell_signals > buy_signals: eth_rec['overall'] = 'SELL'
        else: eth_rec['overall'] = 'HOLD'
        recommendations.append(eth_rec)
        
        # Solana analysis
        sol_rec = {'coin': 'Solana (SOL)', 'price': f'${sol_data["price"].iloc[-1]:,.2f}', 'signals': []}
        if sol_data['rsi'].iloc[-1] > 70: sol_rec['signals'].append({'indicator': 'RSI', 'signal': 'SELL', 'value': f'{sol_data["rsi"].iloc[-1]:.2f}'})
        elif sol_data['rsi'].iloc[-1] < 30: sol_rec['signals'].append({'indicator': 'RSI', 'signal': 'BUY', 'value': f'{sol_data["rsi"].iloc[-1]:.2f}'})
        else: sol_rec['signals'].append({'indicator': 'RSI', 'signal': 'NEUTRAL', 'value': f'{sol_data["rsi"].iloc[-1]:.2f}'})
        
        if sol_data['macd'].iloc[-1] > sol_data['signal_line'].iloc[-1]: sol_rec['signals'].append({'indicator': 'MACD', 'signal': 'BUY', 'value': f'{sol_data["macd"].iloc[-1]:.2f}'})
        else: sol_rec['signals'].append({'indicator': 'MACD', 'signal': 'SELL', 'value': f'{sol_data["macd"].iloc[-1]:.2f}'})
        
        if sol_data['price'].iloc[-1] > sol_data['upper_band'].iloc[-1]: sol_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'SELL', 'value': 'Upper band crossed'})
        elif sol_data['price'].iloc[-1] < sol_data['lower_band'].iloc[-1]: sol_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'BUY', 'value': 'Lower band crossed'})
        else: sol_rec['signals'].append({'indicator': 'Bollinger', 'signal': 'NEUTRAL', 'value': 'Within bands'})
        
        if sol_data['sma50'].iloc[-1] > sol_data['sma200'].iloc[-1]: sol_rec['signals'].append({'indicator': 'Golden Cross', 'signal': 'BUY', 'value': 'SMA50 > SMA200'})
        else: sol_rec['signals'].append({'indicator': 'Death Cross', 'signal': 'SELL', 'value': 'SMA50 < SMA200'})
        
        buy_signals = sum(1 for s in sol_rec['signals'] if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in sol_rec['signals'] if s['signal'] == 'SELL')
        if buy_signals > sell_signals: sol_rec['overall'] = 'BUY'
        elif sell_signals > buy_signals: sol_rec['overall'] = 'SELL'
        else: sol_rec['overall'] = 'HOLD'
        recommendations.append(sol_rec)
        
        logger.info('Buy/sell recommendations generated successfully')
        return recommendations
    except Exception as e:
        logger.error(f'Error generating buy/sell recommendations: {e}', exc_info=True)
        return []

def send_email_alert(user_email, subject, content, charts=None):
    logger.info(f'Preparing to send email alert to {user_email}')
    try:
        # This is a placeholder for actual email sending logic
        # In a production environment, you would use a proper email service
        logger.info(f'Email would be sent to {user_email} with subject: {subject}')
        logger.info(f'Email content: {content[:100]}...')
        if charts:
            logger.info(f'Email would include {len(charts)} charts')
        return True
    except Exception as e:
        logger.error(f'Error sending email alert: {e}', exc_info=True)
        return False

# --- Lazy Loading for Home Page ---
def get_minimal_home_data():
    """Get minimal data needed for initial home page load"""
    try:
        # Use cached data if available, otherwise use dummy data
        trending_coins = []
        economic_indicators = get_economic_indicators()
        
        if 'trending_coins' in api_cache:
            cached_data, _ = api_cache['trending_coins']
            trending_coins = list(cached_data)
        else:
            trending_coins = [
                {'id': 'solana', 'name': 'Solana', 'symbol': 'SOL', 'market_cap_rank': 5, 'price_btc': 0.00123, 'score': 100, 'thumb': ''},
                {'id': 'cardano', 'name': 'Cardano', 'symbol': 'ADA', 'market_cap_rank': 8, 'price_btc': 0.00002, 'score': 98, 'thumb': ''}
            ]
            
        return {
            'trending_coins': trending_coins,
            'economic_indicators': economic_indicators
        }
    except Exception as e:
        logger.error(f'Error getting minimal home data: {e}', exc_info=True)
        return {
            'trending_coins': [],
            'economic_indicators': get_economic_indicators()
        }

# --- Routes ---
@app.route('/')
@app.route('/home')
def home():
    try:
        # Start with minimal data for fast initial load
        minimal_data = get_minimal_home_data()
        
        # Schedule background data loading
        def load_charts_in_background():
            try:
                # These API calls are now protected by rate limiting and caching
                btc_data = calculate_technical_indicators(get_crypto_data('bitcoin', days=30))
                eth_data = calculate_technical_indicators(get_crypto_data('ethereum', days=30))
                sol_data = calculate_technical_indicators(get_crypto_data('solana', days=30))
                
                # Generate and cache charts
                generate_chart(btc_data, 'bitcoin', 'price')
                generate_chart(eth_data, 'ethereum', 'price')
                generate_chart(sol_data, 'solana', 'price')
                generate_chart(btc_data, 'bitcoin', 'rsi')
                generate_chart(eth_data, 'ethereum', 'rsi')
                generate_chart(sol_data, 'solana', 'rsi')
                
                logger.info('Background chart generation completed')
            except Exception as e:
                logger.error(f'Error in background chart generation: {e}', exc_info=True)
        
        # Start background thread if not in rate limit cooling period
        if can_make_api_call():
            threading.Thread(target=load_charts_in_background).start()
        
        # Try to get cached chart data if available
        btc_price_chart = None
        eth_price_chart = None
        sol_price_chart = None
        btc_rsi_chart = None
        eth_rsi_chart = None
        sol_rsi_chart = None
        
        # Use cached data if available
        try:
            btc_data = calculate_technical_indicators(get_crypto_data('bitcoin', days=30))
            eth_data = calculate_technical_indicators(get_crypto_data('ethereum', days=30))
            sol_data = calculate_technical_indicators(get_crypto_data('solana', days=30))
            
            btc_price_chart = generate_chart(btc_data, 'bitcoin', 'price')
            eth_price_chart = generate_chart(eth_data, 'ethereum', 'price')
            sol_price_chart = generate_chart(sol_data, 'solana', 'price')
            
            btc_rsi_chart = generate_chart(btc_data, 'bitcoin', 'rsi')
            eth_rsi_chart = generate_chart(eth_data, 'ethereum', 'rsi')
            sol_rsi_chart = generate_chart(sol_data, 'solana', 'rsi')
        except Exception as e:
            logger.warning(f'Could not generate all charts for home page: {e}')
            # Continue with whatever charts we have
        
        return render_template('home.html', 
                              trending_coins=minimal_data['trending_coins'],
                              btc_price_chart=btc_price_chart,
                              eth_price_chart=eth_price_chart,
                              sol_price_chart=sol_price_chart,
                              btc_rsi_chart=btc_rsi_chart,
                              eth_rsi_chart=eth_rsi_chart,
                              sol_rsi_chart=sol_rsi_chart,
                              economic_indicators=minimal_data['economic_indicators'],
                              rate_limited=(last_rate_limit_hit is not None))
    except Exception as e:
        logger.error(f'Error in home route: {e}', exc_info=True)
        return render_template('500.html', error_message=str(e)), 500

@app.route('/strategy')
def strategy():
    try:
        # Start with minimal data
        recommendations = []
        asset_rotation = []
        
        # Try to get data with rate limit protection
        try:
            asset_rotation = get_asset_rotation_strategy()
            recommendations = generate_buy_sell_recommendations()
        except Exception as e:
            logger.warning(f'Error getting strategy data: {e}')
            # Continue with empty data
        
        # Try to get charts if possible
        btc_macd_chart = None
        eth_macd_chart = None
        sol_macd_chart = None
        
        try:
            btc_data = calculate_technical_indicators(get_crypto_data('bitcoin', days=90))
            eth_data = calculate_technical_indicators(get_crypto_data('ethereum', days=90))
            sol_data = calculate_technical_indicators(get_crypto_data('solana', days=90))
            
            btc_macd_chart = generate_chart(btc_data, 'bitcoin', 'macd')
            eth_macd_chart = generate_chart(eth_data, 'ethereum', 'macd')
            sol_macd_chart = generate_chart(sol_data, 'solana', 'macd')
        except Exception as e:
            logger.warning(f'Could not generate all charts for strategy page: {e}')
            # Continue with whatever charts we have
        
        return render_template('strategy.html',
                              asset_rotation=asset_rotation,
                              recommendations=recommendations,
                              btc_macd_chart=btc_macd_chart,
                              eth_macd_chart=eth_macd_chart,
                              sol_macd_chart=sol_macd_chart,
                              rate_limited=(last_rate_limit_hit is not None))
    except Exception as e:
        logger.error(f'Error in strategy route: {e}', exc_info=True)
        return render_template('500.html', error_message=str(e)), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if not all([username, email, password, confirm_password]):
                flash('All fields are required', 'danger')
                return render_template('register.html')
                
            if password != confirm_password:
                flash('Passwords do not match', 'danger')
                return render_template('register.html')
                
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'danger')
                return render_template('register.html')
                
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'danger')
                return render_template('register.html')
            
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            
            # Create default preferences for the new user
            user_pref = UserPreference(user_id=user.id)
            db.session.add(user_pref)
            db.session.commit()
            
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f'Error in register route: {e}', exc_info=True)
            flash('An error occurred during registration. Please try again.', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            remember = 'remember' in request.form
            
            user = User.query.filter_by(email=email).first()
            
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user, remember=remember)
                next_page = request.args.get('next')
                flash('Login successful!', 'success')
                return redirect(next_page) if next_page else redirect(url_for('home'))
            else:
                flash('Login unsuccessful. Please check email and password.', 'danger')
        except Exception as e:
            logger.error(f'Error in login route: {e}', exc_info=True)
            flash('An error occurred during login. Please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    try:
        user_pref = current_user.preferences or UserPreference(user_id=current_user.id)
        
        if request.method == 'POST':
            user_pref.email_frequency = request.form.get('email_frequency', 'daily')
            user_pref.preferred_coins = request.form.get('preferred_coins', 'BTC,ETH,SOL')
            user_pref.preferred_indicators = request.form.get('preferred_indicators', 'price,volume,rsi,macd')
            user_pref.theme = request.form.get('theme', 'light')
            
            if not current_user.preferences:
                db.session.add(user_pref)
            db.session.commit()
            flash('Your preferences have been updated!', 'success')
            
        return render_template('profile.html', user=current_user, preferences=user_pref)
    except Exception as e:
        db.session.rollback()
        logger.error(f'Error in profile route: {e}', exc_info=True)
        flash('An error occurred while updating your profile. Please try again.', 'danger')
        return render_template('profile.html', user=current_user, preferences=user_pref)

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home'))
    
    try:
        users = User.query.all()
        return render_template('admin_dashboard.html', users=users)
    except Exception as e:
        logger.error(f'Error in admin_dashboard route: {e}', exc_info=True)
        return render_template('500.html', error_message=str(e)), 500

@app.route('/initialize-db')
def initialize_db_route():
    key = request.args.get('key')
    # Use the get_env_var function to fetch the DB_INIT_KEY
    expected_key = get_env_var('DB_INIT_KEY', 'default_secret_key_for_init') 
    if key != expected_key:
        logger.warning(f'Unauthorized attempt to initialize DB with key: {key}. Expected key starts with: {expected_key[:5]}...')
        return 'Unauthorized', 403
    try:
        with app.app_context():
            logger.info('Attempting to drop and create all database tables.')
            db.drop_all()
            db.create_all()
            logger.info('Database tables dropped and created.')
            
            if not User.query.filter_by(username='admin').first():
                logger.info('Admin user not found, creating new admin user.')
                hashed_password = bcrypt.generate_password_hash('defaultadmin').decode('utf-8')
                admin_user = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
                db.session.add(admin_user)
                try:
                    db.session.commit() # Commit the admin_user to get an ID
                    logger.info(f'Admin user {admin_user.username} committed with ID: {admin_user.id}')
                except Exception as e_commit_user:
                    db.session.rollback()
                    logger.error(f'Error committing admin user: {e_commit_user}', exc_info=True)
                    return f'Error committing admin user: {e_commit_user}', 500

                if admin_user.id is None:
                    logger.error('Admin user ID is None after commit. Attempting to fetch.')
                    # Fallback: try to fetch the user again if ID is still None (should not happen)
                    admin_user_reloaded = User.query.filter_by(username='admin').first()
                    if admin_user_reloaded and admin_user_reloaded.id:
                        admin_user_id_for_prefs = admin_user_reloaded.id
                        logger.info(f'Admin user ID reloaded: {admin_user_id_for_prefs}')
                    else:
                        logger.error('Failed to obtain admin user ID even after reloading.')
                        return 'Failed to obtain admin user ID for preferences.', 500
                else:
                    admin_user_id_for_prefs = admin_user.id

                admin_prefs = UserPreference(user_id=admin_user_id_for_prefs)
                db.session.add(admin_prefs)
                try:
                    db.session.commit() # Commit the preferences
                    logger.info('Admin preferences created and committed.')
                except Exception as e_commit_prefs:
                    db.session.rollback()
                    logger.error(f'Error committing admin preferences: {e_commit_prefs}', exc_info=True)
                    return f'Error committing admin preferences: {e_commit_prefs}', 500
                logger.info('Admin user and preferences fully created.')
            else:
                logger.info('Admin user already exists.')
            
        logger.info('Database initialized successfully.')
        return 'Database initialized!'
    except Exception as e:
        db.session.rollback()
        logger.error(f'Error initializing database: {e}', exc_info=True)
        # Ensure 500.html is rendered or a simple string if it fails
        try:
            return render_template('500.html', error_message=str(e)), 500
        except:
            return f'Error initializing database: {e}. Additionally, 500.html template might be missing or broken.', 500

@app.route('/api/crypto-data/<symbol>')
def api_crypto_data(symbol):
    try:
        days = int(request.args.get('days', 30))
        data = get_crypto_data(symbol, days)
        return jsonify({
            'dates': [d.strftime('%Y-%m-%d') for d in data.index],
            'prices': data['price'].tolist(),
            'volumes': data['volume'].tolist()
        })
    except Exception as e:
        logger.error(f'Error in api_crypto_data route: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations')
def api_recommendations():
    try:
        recommendations = generate_buy_sell_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f'Error in api_recommendations route: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/asset-rotation')
def api_asset_rotation():
    try:
        asset_rotation = get_asset_rotation_strategy()
        return jsonify(asset_rotation)
    except Exception as e:
        logger.error(f'Error in api_asset_rotation route: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/trending-coins')
def api_trending_coins():
    try:
        trending = get_trending_coins()
        return jsonify(trending)
    except Exception as e:
        logger.error(f'Error in api_trending_coins route: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/economic-indicators')
def api_economic_indicators():
    try:
        indicators = get_economic_indicators()
        return jsonify(indicators)
    except Exception as e:
        logger.error(f'Error in api_economic_indicators route: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/rate-limit-status')
def api_rate_limit_status():
    """API endpoint to check rate limit status"""
    global last_rate_limit_hit, api_call_timestamps
    
    now = datetime.datetime.utcnow()
    cooling_period_remaining = None
    
    if last_rate_limit_hit is not None:
        time_since_limit = now - last_rate_limit_hit
        if time_since_limit < RATE_LIMIT_RESET:
            cooling_period_remaining = (RATE_LIMIT_RESET - time_since_limit).total_seconds()
    
    # Clean up old timestamps
    one_minute_ago = now - datetime.timedelta(minutes=1)
    api_call_timestamps = [ts for ts in api_call_timestamps if ts > one_minute_ago]
    
    return jsonify({
        'rate_limited': last_rate_limit_hit is not None and cooling_period_remaining is not None,
        'cooling_period_remaining_seconds': cooling_period_remaining,
        'api_calls_last_minute': len(api_call_timestamps),
        'max_calls_per_minute': MAX_CALLS_PER_MINUTE
    })

@app.route('/status')
def status_page():
    """Status page showing application health and rate limit information"""
    try:
        # Get rate limit status
        now = datetime.datetime.utcnow()
        cooling_period_remaining = None
        
        if last_rate_limit_hit is not None:
            time_since_limit = now - last_rate_limit_hit
            if time_since_limit < RATE_LIMIT_RESET:
                cooling_period_remaining = (RATE_LIMIT_RESET - time_since_limit).total_seconds()
        
        # Clean up old timestamps
        one_minute_ago = now - datetime.timedelta(minutes=1)
        current_api_calls = [ts for ts in api_call_timestamps if ts > one_minute_ago]
        
        # Get cache stats
        cache_stats = {
            'total_items': len(api_cache),
            'cache_size_kb': sum(len(str(data)) for data, _ in api_cache.values()) / 1024 if api_cache else 0,
            'oldest_item_age': max((now - timestamp).total_seconds() / 3600 for _, timestamp in api_cache.values()) if api_cache else 0,
            'newest_item_age': min((now - timestamp).total_seconds() / 3600 for _, timestamp in api_cache.values()) if api_cache else 0
        }
        
        return render_template('status.html',
                              rate_limited=last_rate_limit_hit is not None and cooling_period_remaining is not None,
                              cooling_period_remaining=cooling_period_remaining,
                              api_calls_last_minute=len(current_api_calls),
                              max_calls_per_minute=MAX_CALLS_PER_MINUTE,
                              cache_stats=cache_stats)
    except Exception as e:
        logger.error(f'Error in status page: {e}', exc_info=True)
        return render_template('500.html', error_message=str(e)), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', error_message=str(e)), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
