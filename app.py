#!/usr/bin/env python3
"""
Web-based Crypto Investment Strategy service - Render Fixed
"""

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
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
import threading
import time
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout for Render
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- Improved Configuration Section --- 

# Get environment variables with fallbacks and logging
def get_env_var(name, default=None, required=False):
    value = os.environ.get(name, default)
    log_value = "[SET]" if value else "[NOT SET]" 
    # Avoid logging sensitive values like SECRET_KEY or DATABASE_URL directly
    if name in ["SECRET_KEY", "DATABASE_URL"] and value:
        log_value = "[SET]"
    elif value and name not in ["SECRET_KEY", "DATABASE_URL"]:
        log_value = value # Log non-sensitive values if needed for debugging
        
    logger.info(f"Environment variable {name}: {log_value}")
    if required and not value:
        logger.error(f"Required environment variable {name} is not set. Exiting.")
        sys.exit(f"Error: Missing required environment variable {name}")
    return value

# Configure database with Render compatibility
database_url = get_env_var('DATABASE_URL', required=True)
if database_url.startswith("postgres://"):
    # Render provides postgres://, SQLAlchemy needs postgresql://
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    logger.info("Modified DATABASE_URL to use postgresql:// prefix")

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Secret Key (REQUIRED for sessions)
app.config['SECRET_KEY'] = get_env_var('SECRET_KEY', required=True)

# --- End Improved Configuration Section ---

# Initialize extensions
try:
    db = SQLAlchemy(app)
    bcrypt = Bcrypt(app)
    login_manager = LoginManager(app)
    login_manager.login_view = 'login'
    logger.info("Flask extensions initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Flask extensions: {str(e)}", exc_info=True)
    sys.exit("Failed to initialize Flask extensions.")

# Create data directory if it doesn't exist (less relevant for Render, but harmless)
os.makedirs('data', exist_ok=True)
os.makedirs('static/img', exist_ok=True)

# User models
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
        logger.error(f"Error loading user {user_id}: {str(e)}", exc_info=True)
        return None

# --- Crypto data functions (Keep as is, but add logging) ---
def get_crypto_data(symbol, days=30):
    logger.info(f"Fetching crypto data for {symbol} ({days} days)")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        if not data or 'prices' not in data or 'total_volumes' not in data:
             logger.warning(f"Incomplete data received from CoinGecko for {symbol}")
             raise ValueError("Incomplete data from API")

        # Create DataFrame
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add volume
        volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('timestamp', inplace=True)
        df['volume'] = volume_df['volume']
        
        logger.info(f"Successfully fetched data for {symbol}")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {e}", exc_info=True)
        
    # Return dummy data if API fails
    logger.warning(f"Returning dummy data for {symbol}")
    dates = pd.date_range(end=datetime.datetime.now(datetime.timezone.utc), periods=days)
    df = pd.DataFrame(index=dates)
    df['price'] = [50000 - i * 100 + i * i * 5 for i in range(days)]
    df['volume'] = [2000000000 - i * 10000000 + i * i * 500000 for i in range(days)]
    return df

def get_trending_coins():
    logger.info("Fetching trending coins")
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'coins' not in data:
            logger.warning("Incomplete data received from CoinGecko for trending coins")
            raise ValueError("Incomplete data from API")
            
        trending = []
        for coin in data['coins'][:5]:  # Get top 5
            item = coin.get('item')
            if not item:
                continue
            trending.append({
                'id': item.get('id', 'N/A'),
                'name': item.get('name', 'N/A'),
                'symbol': item.get('symbol', 'N/A'),
                'market_cap_rank': item.get('market_cap_rank', 'N/A'),
                'price_btc': item.get('price_btc', 0),
                'score': item.get('score', 0),
                'thumb': item.get('thumb', '')
            })
        logger.info(f"Successfully fetched {len(trending)} trending coins")
        return trending
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching trending coins: {e}")
    except Exception as e:
        logger.error(f"Error processing trending coins: {e}", exc_info=True)
        
    # Return dummy data if API fails
    logger.warning("Returning dummy trending coins data")
    return [
        {'id': 'solana', 'name': 'Solana', 'symbol': 'SOL', 'market_cap_rank': 5, 'price_btc': 0.00123, 'score': 100, 'thumb': ''},
        {'id': 'cardano', 'name': 'Cardano', 'symbol': 'ADA', 'market_cap_rank': 8, 'price_btc': 0.00002, 'score': 98, 'thumb': ''},
        {'id': 'polkadot', 'name': 'Polkadot', 'symbol': 'DOT', 'market_cap_rank': 12, 'price_btc': 0.00034, 'score': 95, 'thumb': ''},
        {'id': 'avalanche-2', 'name': 'Avalanche', 'symbol': 'AVAX', 'market_cap_rank': 11, 'price_btc': 0.00045, 'score': 92, 'thumb': ''},
        {'id': 'chainlink', 'name': 'Chainlink', 'symbol': 'LINK', 'market_cap_rank': 15, 'price_btc': 0.00028, 'score': 90, 'thumb': ''}
    ]

def get_economic_indicators():
    logger.info("Fetching economic indicators (using dummy data)")
    # In a real implementation, this would fetch from economic data APIs
    return {
        'usd_index': 104.2,
        'usd_trend': 'rising',
        'inflation_rate': 3.5,
        'inflation_trend': 'stable',
        'interest_rate': 5.25,
        'interest_trend': 'stable',
        'global_liquidity': 'moderate',
        'liquidity_trend': 'tightening',
        'stock_market': 'bullish',
        'bond_market': 'bearish',
        'gold_price': 2342.50,
        'gold_trend': 'rising'
    }

def calculate_technical_indicators(df):
    logger.info("Calculating technical indicators")
    if df is None or df.empty:
        logger.warning("Cannot calculate indicators on empty DataFrame")
        return pd.DataFrame() # Return empty df
    try:
        # Calculate RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean() # Use min_periods
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 1e-6) 
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50) # Fill initial NaNs
        
        # Calculate MACD
        df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['signal']
        
        # Calculate Bollinger Bands
        df['sma20'] = df['price'].rolling(window=20, min_periods=1).mean()
        df['std20'] = df['price'].rolling(window=20, min_periods=1).std().fillna(0)
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        
        # Calculate Moving Averages
        df['sma50'] = df['price'].rolling(window=50, min_periods=1).mean()
        df['sma200'] = df['price'].rolling(window=200, min_periods=1).mean()
        
        logger.info("Technical indicators calculated successfully")
        return df.fillna(0) # Fill any remaining NaNs
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        return df # Return original df or empty df?

def generate_chart(df, symbol, indicator_type):
    logger.info(f"Generating chart for {symbol} - {indicator_type}")
    if df is None or df.empty or indicator_type not in ['price', 'rsi', 'macd', 'volume']:
        logger.warning(f"Cannot generate chart for {symbol} - {indicator_type}. Invalid input.")
        return None

    try:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)

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
            ax.set_ylabel('RSI')
            ax.set_ylim(0, 100)
        elif indicator_type == 'macd':
            ax.plot(df.index, df.get('macd'), label='MACD')
            ax.plot(df.index, df.get('signal'), label='Signal Line', linestyle='--')
            # Ensure histogram values are numeric before plotting
            hist_values = pd.to_numeric(df.get('macd_histogram'), errors='coerce').fillna(0)
            ax.bar(df.index, hist_values, label='Histogram', alpha=0.3, width=0.8)
            ax.set_title(f'{symbol.upper()} MACD')
            ax.set_ylabel('MACD Value')
        elif indicator_type == 'volume':
            ax.bar(df.index, df.get('volume'), label='Volume', alpha=0.6)
            ax.set_title(f'{symbol.upper()} Volume')
            ax.set_ylabel('Volume')

        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig) # Close the figure

        # Convert to base64 for embedding in HTML
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        logger.info(f"Chart generated successfully for {symbol} - {indicator_type}")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error generating chart for {symbol} - {indicator_type}: {e}", exc_info=True)
        return None

# --- Asset Rotation and Recommendations (Keep as is, but ensure data is handled) ---
def get_asset_rotation_strategy():
    logger.info("Generating asset rotation strategy")
    try:
        # Get data for main coins
        btc_data = get_crypto_data('bitcoin', days=30)
        eth_data = get_crypto_data('ethereum', days=30)
        sol_data = get_crypto_data('solana', days=30)
        
        # Calculate performance (handle potential empty dataframes)
        btc_perf = (btc_data['price'].iloc[-1] / btc_data['price'].iloc[0] - 1) * 100 if not btc_data.empty else 0
        eth_perf = (eth_data['price'].iloc[-1] / eth_data['price'].iloc[0] - 1) * 100 if not eth_data.empty else 0
        sol_perf = (sol_data['price'].iloc[-1] / sol_data['price'].iloc[0] - 1) * 100 if not sol_data.empty else 0
        
        # Get trending coins
        trending = get_trending_coins()
        
        # Get economic indicators
        econ = get_economic_indicators()
        
        # Determine allocation based on performance and indicators
        allocations = []
        
        # Bitcoin allocation
        if btc_perf > 10 and econ.get('usd_trend') == 'falling':
            btc_alloc = 40
            btc_signal = 'BUY'
        elif btc_perf < -10 and econ.get('usd_trend') == 'rising':
            btc_alloc = 10
            btc_signal = 'SELL'
        else:
            btc_alloc = 25
            btc_signal = 'HOLD'
        
        allocations.append({
            'coin': 'Bitcoin (BTC)',
            'allocation': btc_alloc,
            'signal': btc_signal,
            'performance': f"{btc_perf:.2f}%"
        })
        
        # Ethereum allocation
        if eth_perf > 15:
            eth_alloc = 30
            eth_signal = 'BUY'
        elif eth_perf < -15:
            eth_alloc = 10
            eth_signal = 'SELL'
        else:
            eth_alloc = 20
            eth_signal = 'HOLD'
        
        allocations.append({
            'coin': 'Ethereum (ETH)',
            'allocation': eth_alloc,
            'signal': eth_signal,
            'performance': f"{eth_perf:.2f}%"
        })
        
        # Solana allocation
        if sol_perf > 20:
            sol_alloc = 20
            sol_signal = 'BUY'
        elif sol_perf < -20:
            sol_alloc = 5
            sol_signal = 'SELL'
        else:
            sol_alloc = 15
            sol_signal = 'HOLD'
        
        allocations.append({
            'coin': 'Solana (SOL)',
            'allocation': sol_alloc,
            'signal': sol_signal,
            'performance': f"{sol_perf:.2f}%"
        })
        
        # Allocate remaining to trending coins
        remaining = 100 - (btc_alloc + eth_alloc + sol_alloc)
        if trending and remaining > 0:
             per_trending = remaining / len(trending)
             for coin in trending:
                 allocations.append({
                     'coin': f"{coin.get('name', 'N/A')} ({coin.get('symbol', 'N/A')})",
                     'allocation': round(per_trending),
                     'signal': 'BUY' if coin.get('score', 0) > 95 else 'RESEARCH',
                     'performance': 'Trending'
                 })
        elif remaining > 0: # If no trending coins, allocate remaining to cash/stable
             allocations.append({
                 'coin': 'Cash/Stablecoin',
                 'allocation': remaining,
                 'signal': 'HOLD',
                 'performance': 'N/A'
             })

        logger.info("Asset rotation strategy generated successfully")
        return allocations
    except Exception as e:
        logger.error(f"Error generating asset rotation strategy: {e}", exc_info=True)
        return [] # Return empty list on error

def generate_buy_sell_recommendations():
    logger.info("Generating buy/sell recommendations")
    try:
        # Get data for main coins
        btc_data = get_crypto_data('bitcoin', days=90) # Use longer period for MAs
        eth_data = get_crypto_data('ethereum', days=90)
        sol_data = get_crypto_data('solana', days=90)
        
        # Calculate indicators
        btc_data = calculate_technical_indicators(btc_data)
        eth_data = calculate_technical_indicators(eth_data)
        sol_data = calculate_technical_indicators(sol_data)
        
        # Get economic indicators
        econ = get_economic_indicators()
        
        recommendations = []
        
        # Bitcoin recommendation
        if not btc_data.empty:
            btc_rsi = btc_data['rsi'].iloc[-1]
            btc_macd_hist = btc_data['macd_histogram'].iloc[-1]
            btc_price = btc_data['price'].iloc[-1]
            btc_sma50 = btc_data['sma50'].iloc[-1]
            
            if btc_rsi < 30 and btc_macd_hist > 0 and econ.get('usd_trend') == 'falling':
                btc_rec = 'STRONG BUY'
                btc_reason = 'Oversold RSI, positive MACD momentum, weakening USD'
            elif btc_rsi > 70 and btc_macd_hist < 0 and econ.get('usd_trend') == 'rising':
                btc_rec = 'STRONG SELL'
                btc_reason = 'Overbought RSI, negative MACD momentum, strengthening USD'
            elif btc_price > btc_sma50 and btc_macd_hist > 0:
                btc_rec = 'BUY'
                btc_reason = 'Price > 50-day MA, positive momentum'
            elif btc_price < btc_sma50 and btc_macd_hist < 0:
                btc_rec = 'SELL'
                btc_reason = 'Price < 50-day MA, negative momentum'
            else:
                btc_rec = 'HOLD'
                btc_reason = 'Mixed signals'
            
            recommendations.append({
                'coin': 'Bitcoin (BTC)',
                'price': f"${btc_price:,.2f}",
                'recommendation': btc_rec,
                'reason': btc_reason
            })
        else:
             logger.warning("Skipping Bitcoin recommendation due to missing data")

        # Ethereum recommendation
        if not eth_data.empty:
            eth_rsi = eth_data['rsi'].iloc[-1]
            eth_macd_hist = eth_data['macd_histogram'].iloc[-1]
            eth_price = eth_data['price'].iloc[-1]
            eth_sma50 = eth_data['sma50'].iloc[-1]
            
            if eth_rsi < 30 and eth_macd_hist > 0:
                eth_rec = 'STRONG BUY'
                eth_reason = 'Oversold RSI, positive MACD momentum'
            elif eth_rsi > 70 and eth_macd_hist < 0:
                eth_rec = 'STRONG SELL'
                eth_reason = 'Overbought RSI, negative MACD momentum'
            elif eth_price > eth_sma50 and eth_macd_hist > 0:
                eth_rec = 'BUY'
                eth_reason = 'Price > 50-day MA, positive momentum'
            elif eth_price < eth_sma50 and eth_macd_hist < 0:
                eth_rec = 'SELL'
                eth_reason = 'Price < 50-day MA, negative momentum'
            else:
                eth_rec = 'HOLD'
                eth_reason = 'Mixed signals'
            
            recommendations.append({
                'coin': 'Ethereum (ETH)',
                'price': f"${eth_price:,.2f}",
                'recommendation': eth_rec,
                'reason': eth_reason
            })
        else:
             logger.warning("Skipping Ethereum recommendation due to missing data")

        # Solana recommendation
        if not sol_data.empty:
            sol_rsi = sol_data['rsi'].iloc[-1]
            sol_macd_hist = sol_data['macd_histogram'].iloc[-1]
            sol_price = sol_data['price'].iloc[-1]
            sol_sma50 = sol_data['sma50'].iloc[-1]
            
            if sol_rsi < 30 and sol_macd_hist > 0:
                sol_rec = 'STRONG BUY'
                sol_reason = 'Oversold RSI, positive MACD momentum'
            elif sol_rsi > 70 and sol_macd_hist < 0:
                sol_rec = 'STRONG SELL'
                sol_reason = 'Overbought RSI, negative MACD momentum'
            elif sol_price > sol_sma50 and sol_macd_hist > 0:
                sol_rec = 'BUY'
                sol_reason = 'Price > 50-day MA, positive momentum'
            elif sol_price < sol_sma50 and sol_macd_hist < 0:
                sol_rec = 'SELL'
                sol_reason = 'Price < 50-day MA, negative momentum'
            else:
                sol_rec = 'HOLD'
                sol_reason = 'Mixed signals'
            
            recommendations.append({
                'coin': 'Solana (SOL)',
                'price': f"${sol_price:,.2f}",
                'recommendation': sol_rec,
                'reason': sol_reason
            })
        else:
             logger.warning("Skipping Solana recommendation due to missing data")

        # Add trending coins
        trending = get_trending_coins()
        for coin in trending:
            # Simple recommendation based on score
            rec = 'RESEARCH' if coin.get('score', 0) < 95 else 'POTENTIAL BUY'
            reason = f"Trending coin (Score: {coin.get('score', 0)})"
            recommendations.append({
                'coin': f"{coin.get('name', 'N/A')} ({coin.get('symbol', 'N/A')})",
                'price': f"{coin.get('price_btc', 0):.8f} BTC",
                'recommendation': rec,
                'reason': reason
            })

        logger.info("Buy/sell recommendations generated successfully")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating buy/sell recommendations: {e}", exc_info=True)
        return [] # Return empty list on error

# --- Email Service (Keep as is, but add logging) ---
def send_daily_email_report():
    logger.info("Starting daily email report generation")
    try:
        # Get user preferences (assuming one user for now)
        # In a multi-user system, loop through users with daily preference
        user_email = "keithtrmurray@aol.com" # Hardcoded for now
        
        # Generate content
        recommendations = generate_buy_sell_recommendations()
        allocations = get_asset_rotation_strategy()
        econ_indicators = get_economic_indicators()
        trending_coins = get_trending_coins()
        
        # Generate charts
        btc_data = calculate_technical_indicators(get_crypto_data('bitcoin', days=90))
        eth_data = calculate_technical_indicators(get_crypto_data('ethereum', days=90))
        sol_data = calculate_technical_indicators(get_crypto_data('solana', days=90))
        
        btc_price_chart = generate_chart(btc_data, 'bitcoin', 'price')
        eth_price_chart = generate_chart(eth_data, 'ethereum', 'price')
        sol_price_chart = generate_chart(sol_data, 'solana', 'price')
        btc_rsi_chart = generate_chart(btc_data, 'bitcoin', 'rsi')
        eth_macd_chart = generate_chart(eth_data, 'ethereum', 'macd')
        
        # Construct email body (HTML)
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; }}
                h2 {{ color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ margin-bottom: 20px; }}
                .chart img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Crypto Investment Strategy - Daily Report</h1>
            <p>Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Buy/Sell Recommendations</h2>
            <table>
                <tr><th>Coin</th><th>Price</th><th>Recommendation</th><th>Reason</th></tr>
                {''.join([f"<tr><td>{rec['coin']}</td><td>{rec['price']}</td><td>{rec['recommendation']}</td><td>{rec['reason']}</td></tr>" for rec in recommendations])}
            </table>

            <h2>Asset Rotation Strategy</h2>
            <table>
                <tr><th>Asset</th><th>Allocation %</th><th>Signal</th><th>30d Performance</th></tr>
                {''.join([f"<tr><td>{alloc['coin']}</td><td>{alloc['allocation']}%</td><td>{alloc['signal']}</td><td>{alloc['performance']}</td></tr>" for alloc in allocations])}
            </table>

            <h2>Economic Indicators</h2>
            <ul>
                <li>USD Index: {econ_indicators.get('usd_index')} ({econ_indicators.get('usd_trend')})</li>
                <li>Inflation Rate: {econ_indicators.get('inflation_rate')}% ({econ_indicators.get('inflation_trend')})</li>
                <li>Interest Rate: {econ_indicators.get('interest_rate')}% ({econ_indicators.get('interest_trend')})</li>
                <li>Global Liquidity: {econ_indicators.get('global_liquidity')} ({econ_indicators.get('liquidity_trend')})</li>
                <li>Gold Price: ${econ_indicators.get('gold_price'):,.2f} ({econ_indicators.get('gold_trend')})</li>
            </ul>

            <h2>Trending Coins (Top 5)</h2>
            <ul>
                {''.join([f"<li>{coin.get('name', 'N/A')} ({coin.get('symbol', 'N/A')}) - Rank: {coin.get('market_cap_rank', 'N/A')}, Score: {coin.get('score', 0)}</li>" for coin in trending_coins])}
            </ul>

            <h2>Charts</h2>
            <div class="chart">
                <h3>Bitcoin Price & Bollinger Bands</h3>
                {f'<img src="{btc_price_chart}">' if btc_price_chart else '<p>Chart unavailable</p>'}
            </div>
            <div class="chart">
                <h3>Ethereum Price & Bollinger Bands</h3>
                {f'<img src="{eth_price_chart}">' if eth_price_chart else '<p>Chart unavailable</p>'}
            </div>
            <div class="chart">
                <h3>Solana Price & Bollinger Bands</h3>
                {f'<img src="{sol_price_chart}">' if sol_price_chart else '<p>Chart unavailable</p>'}
            </div>
            <div class="chart">
                <h3>Bitcoin RSI</h3>
                {f'<img src="{btc_rsi_chart}">' if btc_rsi_chart else '<p>Chart unavailable</p>'}
            </div>
            <div class="chart">
                <h3>Ethereum MACD</h3>
                {f'<img src="{eth_macd_chart}">' if eth_macd_chart else '<p>Chart unavailable</p>'}
            </div>

        </body>
        </html>
        """
        
        # Send email (using environment variables for credentials)
        sender_email = get_env_var("EMAIL_USER")
        sender_password = get_env_var("EMAIL_PASS")
        smtp_server = get_env_var("EMAIL_SMTP_SERVER", "smtp.aol.com") # Default to AOL SMTP
        smtp_port = int(get_env_var("EMAIL_SMTP_PORT", 587)) # Default to 587 (TLS)

        if not sender_email or not sender_password:
            logger.warning("Email credentials (EMAIL_USER, EMAIL_PASS) not set. Skipping email.")
            # Save report locally as fallback
            try:
                with open(f"data/daily_report_{datetime.date.today()}.html", "w") as f:
                    f.write(html_body)
                logger.info(f"Email report saved locally to data/daily_report_{datetime.date.today()}.html")
            except Exception as e:
                logger.error(f"Failed to save email report locally: {e}")
            return

        message = MIMEMultipart("related")
        message["Subject"] = "Crypto Investment Strategy - Daily Report"
        message["From"] = sender_email
        message["To"] = user_email
        
        # Attach HTML body
        message.attach(MIMEText(html_body, "html"))
        
        # Note: Embedding images directly in email is complex and not fully shown here.
        # The base64 strings are already in the HTML body.

        try:
            logger.info(f"Connecting to SMTP server {smtp_server}:{smtp_port}")
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls() # Secure the connection
                logger.info("Logging into SMTP server")
                server.login(sender_email, sender_password)
                logger.info(f"Sending email to {user_email}")
                server.sendmail(sender_email, user_email, message.as_string())
                logger.info("Email sent successfully")
        except smtplib.SMTPAuthenticationError as e:
             logger.error(f"SMTP Authentication Error: {e}. Check EMAIL_USER and EMAIL_PASS.")
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error generating daily email report: {e}", exc_info=True)

def run_daily_email_scheduler():
    logger.info("Email scheduler thread started")
    while True:
        now = datetime.datetime.now()
        # Send at a specific time, e.g., 8:00 AM UTC
        if now.hour == 8 and now.minute == 0:
            send_daily_email_report()
            # Sleep for slightly more than a minute to avoid sending multiple times
            time.sleep(65) 
        # Check every minute
        time.sleep(60)

# --- Flask Routes --- 

@app.route('/')
def home():
    logger.info("Accessed home page")
    try:
        trending_coins = get_trending_coins()
        econ_indicators = get_economic_indicators()
        # Get brief data for overview
        btc_data = get_crypto_data('bitcoin', days=2)
        eth_data = get_crypto_data('ethereum', days=2)
        sol_data = get_crypto_data('solana', days=2)
        
        market_overview = []
        if not btc_data.empty:
            market_overview.append({'name': 'Bitcoin', 'price': btc_data['price'].iloc[-1], 'change': (btc_data['price'].iloc[-1]/btc_data['price'].iloc[0]-1)*100 if len(btc_data['price']) > 1 else 0})
        if not eth_data.empty:
             market_overview.append({'name': 'Ethereum', 'price': eth_data['price'].iloc[-1], 'change': (eth_data['price'].iloc[-1]/eth_data['price'].iloc[0]-1)*100 if len(eth_data['price']) > 1 else 0})
        if not sol_data.empty:
             market_overview.append({'name': 'Solana', 'price': sol_data['price'].iloc[-1], 'change': (sol_data['price'].iloc[-1]/sol_data['price'].iloc[0]-1)*100 if len(sol_data['price']) > 1 else 0})

        return render_template('home.html', 
                               trending_coins=trending_coins, 
                               econ_indicators=econ_indicators,
                               market_overview=market_overview)
    except Exception as e:
        logger.error(f"Error loading home page: {e}", exc_info=True)
        flash('Error loading page data. Please try again later.', 'danger')
        return render_template('home.html', trending_coins=[], econ_indicators={}, market_overview=[])

@app.route('/dashboard')
@login_required
def dashboard():
    logger.info(f"Accessed dashboard page by user {current_user.username}")
    try:
        recommendations = generate_buy_sell_recommendations()
        allocations = get_asset_rotation_strategy()
        econ_indicators = get_economic_indicators()
        
        # Generate charts
        btc_data = calculate_technical_indicators(get_crypto_data('bitcoin', days=90))
        eth_data = calculate_technical_indicators(get_crypto_data('ethereum', days=90))
        sol_data = calculate_technical_indicators(get_crypto_data('solana', days=90))
        
        charts = {
            'btc_price': generate_chart(btc_data, 'bitcoin', 'price'),
            'eth_price': generate_chart(eth_data, 'ethereum', 'price'),
            'sol_price': generate_chart(sol_data, 'solana', 'price'),
            'btc_rsi': generate_chart(btc_data, 'bitcoin', 'rsi'),
            'eth_macd': generate_chart(eth_data, 'ethereum', 'macd'),
            'sol_volume': generate_chart(sol_data, 'solana', 'volume')
        }
        
        return render_template('dashboard.html', 
                               recommendations=recommendations, 
                               allocations=allocations, 
                               econ_indicators=econ_indicators,
                               charts=charts)
    except Exception as e:
        logger.error(f"Error loading dashboard page: {e}", exc_info=True)
        flash('Error loading dashboard data. Please try again later.', 'danger')
        return render_template('dashboard.html', recommendations=[], allocations=[], econ_indicators={}, charts={})

@app.route('/education')
def education():
    logger.info("Accessed education page")
    return render_template('education.html')

@app.route('/sources')
def sources():
    logger.info("Accessed sources page")
    return render_template('information_sources.html')

@app.route('/altcoins')
def altcoins():
    logger.info("Accessed altcoin analysis page")
    try:
        trending = get_trending_coins()
        altcoin_data = []
        for coin in trending:
            data = calculate_technical_indicators(get_crypto_data(coin['id'], days=30))
            if not data.empty:
                recommendation = 'HOLD' # Simplified logic for altcoins
                if data['rsi'].iloc[-1] < 40 and data['macd_histogram'].iloc[-1] > 0:
                    recommendation = 'POTENTIAL BUY'
                elif data['rsi'].iloc[-1] > 60 and data['macd_histogram'].iloc[-1] < 0:
                     recommendation = 'POTENTIAL SELL'
                altcoin_data.append({
                    'name': coin['name'],
                    'symbol': coin['symbol'],
                    'rank': coin['market_cap_rank'],
                    'price': data['price'].iloc[-1],
                    'rsi': data['rsi'].iloc[-1],
                    'macd_hist': data['macd_histogram'].iloc[-1],
                    'recommendation': recommendation,
                    'chart': generate_chart(data, coin['id'], 'price')
                })
            else:
                 logger.warning(f"Could not get data for altcoin {coin['id']}")
        return render_template('altcoin_analysis.html', altcoins=altcoin_data)
    except Exception as e:
        logger.error(f"Error loading altcoin page: {e}", exc_info=True)
        flash('Error loading altcoin data. Please try again later.', 'danger')
        return render_template('altcoin_analysis.html', altcoins=[])

@app.route('/market-data')
def market_data():
    logger.info("Accessed market data page")
    # This route might be for the live JS charts page
    # Ensure the template exists
    try:
        return render_template('market_data.html')
    except Exception as e:
        logger.error(f"Error loading market_data template: {e}")
        return "Market data page not found or error loading template.", 404

@app.route('/api/market_data')
def api_market_data():
    logger.info("API request for market data")
    try:
        symbol = request.args.get('symbol', 'bitcoin')
        days = int(request.args.get('days', 30))
        
        df = get_crypto_data(symbol, days=days)
        df = calculate_technical_indicators(df)
        
        # Prepare data for JSON response (e.g., for Chart.js)
        chart_data = {
            'labels': df.index.strftime('%Y-%m-%d').tolist(),
            'price': df['price'].tolist(),
            'volume': df['volume'].tolist(),
            'rsi': df['rsi'].tolist(),
            'macd': df['macd'].tolist(),
            'signal': df['signal'].tolist(),
            'macd_histogram': df['macd_histogram'].tolist(),
            'upper_band': df['upper_band'].tolist(),
            'lower_band': df['lower_band'].tolist(),
            'sma20': df['sma20'].tolist(),
            'sma50': df['sma50'].tolist(),
            'sma200': df['sma200'].tolist()
        }
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error in /api/market_data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch market data'}), 500

# --- User Authentication Routes --- 

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            
            # Basic validation
            if not username or not email or not password:
                 flash('All fields are required.', 'danger')
                 return redirect(url_for('register'))
                 
            existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                flash('Username or email already exists.', 'danger')
                return redirect(url_for('register'))

            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            # Create default preferences
            prefs = UserPreference(user=user)
            db.session.add(prefs)
            db.session.commit()
            flash('Your account has been created! You can now log in.', 'success')
            logger.info(f"New user registered: {username}")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during registration: {e}", exc_info=True)
            flash('An error occurred during registration. Please try again.', 'danger')
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            remember = request.form.get('remember') == 'on'
            
            user = User.query.filter_by(email=email).first()
            
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user, remember=remember)
                logger.info(f"User logged in: {user.username}")
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('dashboard'))
            else:
                flash('Login Unsuccessful. Please check email and password.', 'danger')
        except Exception as e:
             logger.error(f"Error during login: {e}", exc_info=True)
             flash('An error occurred during login. Please try again.', 'danger')
             
    return render_template('login.html')

@app.route('/logout')
def logout():
    if current_user.is_authenticated:
        logger.info(f"User logged out: {current_user.username}")
        logout_user()
    return redirect(url_for('home'))

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    if request.method == 'POST':
        try:
            # Update preferences
            prefs = current_user.preferences
            if not prefs:
                 prefs = UserPreference(user=current_user)
                 db.session.add(prefs)
                 
            prefs.email_frequency = request.form.get('email_frequency', 'daily')
            prefs.preferred_coins = request.form.get('preferred_coins', 'BTC,ETH,SOL')
            prefs.preferred_indicators = request.form.get('preferred_indicators', 'price,volume,rsi,macd')
            prefs.theme = request.form.get('theme', 'light')
            
            db.session.commit()
            flash('Your preferences have been updated!', 'success')
            logger.info(f"User preferences updated for {current_user.username}")
            return redirect(url_for('account'))
        except Exception as e:
             db.session.rollback()
             logger.error(f"Error updating preferences for {current_user.username}: {e}", exc_info=True)
             flash('An error occurred while updating preferences.', 'danger')
             
    prefs = current_user.preferences
    if not prefs:
        # Create default preferences if they don't exist
        prefs = UserPreference(user=current_user)
        db.session.add(prefs)
        db.session.commit()
        logger.info(f"Created default preferences for user {current_user.username}")
        
    return render_template('account.html', preferences=prefs)

# --- Initialization Command --- 

def initialize_database():
    """Create database tables and default admin user"""
    try:
        logger.info("Initializing database...")
        with app.app_context():
            db.create_all()
            logger.info("Database tables created (if they didn't exist).")
            
            # Check if admin user exists
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                logger.info("Admin user not found, creating default admin user...")
                hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
                admin = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
                db.session.add(admin)
                # Create default preferences for admin
                prefs = UserPreference(user=admin)
                db.session.add(prefs)
                db.session.commit()
                logger.info("Default admin user created (username: admin, password: admin123)")
            else:
                logger.info("Admin user already exists.")
        logger.info("Database initialization complete.")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        sys.exit("Failed to initialize database.")

# --- Main Execution --- 

@app.route('/initialize-db')
def init_db_route():
    try:
        # Only allow this in development or with a special key
        init_key = request.args.get('key')
        if init_key != 'setup123':
            return "Unauthorized", 401
        
        logger.info("Initializing database from web route...")
        db.create_all()
        
        # Check if admin user exists
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            logger.info("Creating default admin user...")
            hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
            db.session.add(admin)
            # Create default preferences for admin
            prefs = UserPreference(user=admin)
            db.session.add(prefs)
            db.session.commit()
            return "Database initialized and admin user created successfully!"
        else:
            return "Database initialized. Admin user already exists."
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        return f"Error initializing database: {str(e)}", 500


if __name__ == '__main__':
    # Check for initialization flag
    if '--initialize-db' in sys.argv:
        initialize_database()
    else:
        # Start email scheduler in a separate thread
        # Check if email env vars are set before starting thread
        if get_env_var("EMAIL_USER") and get_env_var("EMAIL_PASS"):
            scheduler_thread = threading.Thread(target=run_daily_email_scheduler, daemon=True)
            scheduler_thread.start()
        else:
            logger.warning("Email credentials not set. Email scheduler thread not started.")
            
        # Run Flask app using Gunicorn recommended approach for production
        # The actual command is run by Render based on Procfile or Start Command
        # For local testing (if needed):
        # app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
        logger.info("Starting Flask application...")
        # Gunicorn will run this script, so this block might not execute in Render context
        # Ensure Gunicorn command is correct in Render settings: gunicorn app:app
        pass # Placeholder, Gunicorn runs the app instance

