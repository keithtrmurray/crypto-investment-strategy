#!/usr/bin/env python3
"""
Web-based Crypto Investment Strategy service - Render Fixed with Caching and Diagnostics
"""

import os
import sys
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Corrected: No backslashes
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Corrected: No backslashes
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout for Render
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- Caching Setup ---
api_cache = {}
CACHE_DURATION = datetime.timedelta(minutes=30) # Cache data for 30 minutes
# --- End Caching Setup ---

# --- Improved Configuration Section --- 

def get_env_var(name, default=None, required=False):
    value = os.environ.get(name, default)
    # Simplified logging for sensitive vars, log actual value for others if not None
    if name in ["SECRET_KEY", "DATABASE_URL"]:
        log_value = "[SET]" if value else "[NOT SET]"
    elif value:
        log_value = value # Log actual value for non-sensitive, non-empty vars
    else:
        log_value = "[NOT SET]"
        
    logger.info(f"Environment variable {name}: {log_value}")
    if required and not value:
        logger.error(f"Required environment variable {name} is not set. Exiting.")
        sys.exit(f"Error: Missing required environment variable {name}")
    return value

database_url = get_env_var('DATABASE_URL', required=True)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    logger.info("Modified DATABASE_URL to use postgresql:// prefix")

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = get_env_var('SECRET_KEY', required=True)

# --- End Improved Configuration Section ---

try:
    db = SQLAlchemy(app)
    bcrypt = Bcrypt(app)
    login_manager = LoginManager(app)
    login_manager.login_view = 'login'
    logger.info("Flask extensions initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Flask extensions: {str(e)}", exc_info=True)
    sys.exit("Failed to initialize Flask extensions.")

os.makedirs('data', exist_ok=True)
os.makedirs('static/img', exist_ok=True)

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

# --- Crypto data functions with Caching ---
def get_crypto_data(symbol, days=30):
    cache_key = (symbol, days)
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if datetime.datetime.utcnow() - timestamp < CACHE_DURATION:
            logger.info(f"Returning cached crypto data for {symbol} ({days} days)")
            return cached_data.copy()
        else:
            logger.info(f"Cache expired for {symbol} ({days} days)")

    logger.info(f"Fetching crypto data for {symbol} ({days} days) from API")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'prices' not in data or 'total_volumes' not in data:
             logger.warning(f"Incomplete data received from CoinGecko for {symbol}")
             raise ValueError("Incomplete data from API")

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('timestamp', inplace=True)
        df['volume'] = volume_df['volume']
        
        api_cache[cache_key] = (df.copy(), datetime.datetime.utcnow())
        logger.info(f"Successfully fetched and cached data for {symbol}")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {e}", exc_info=True)
        
    logger.warning(f"Returning dummy data for {symbol} due to API error")
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
            logger.info("Returning cached trending coins data")
            return list(cached_data)
        else:
            logger.info("Cache expired for trending coins")

    logger.info("Fetching trending coins from API")
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'coins' not in data:
            logger.warning("Incomplete data received from CoinGecko for trending coins")
            raise ValueError("Incomplete data from API")
            
        trending = []
        for coin_data in data['coins'][:5]: # Get top 5 trending
            item = coin_data.get('item')
            if not item: continue
            trending.append({
                'id': item.get('id', 'N/A'), 'name': item.get('name', 'N/A'),
                'symbol': item.get('symbol', 'N/A'), 'market_cap_rank': item.get('market_cap_rank', 'N/A'),
                'price_btc': item.get('price_btc', 0), 'score': item.get('score', 0),
                'thumb': item.get('thumb', '')
            })
        api_cache[cache_key] = (list(trending), datetime.datetime.utcnow())
        logger.info(f"Successfully fetched and cached {len(trending)} trending coins")
        return trending
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching trending coins: {e}")
    except Exception as e:
        logger.error(f"Error processing trending coins: {e}", exc_info=True)
        
    logger.warning("Returning dummy trending coins data due to API error")
    return [
        {'id': 'solana', 'name': 'Solana', 'symbol': 'SOL', 'market_cap_rank': 5, 'price_btc': 0.00123, 'score': 100, 'thumb': ''},
        {'id': 'cardano', 'name': 'Cardano', 'symbol': 'ADA', 'market_cap_rank': 8, 'price_btc': 0.00002, 'score': 98, 'thumb': ''}
    ]

def get_economic_indicators():
    logger.info("Fetching economic indicators (using dummy data)")
    return {
        'usd_index': 104.2, 'usd_trend': 'rising',
        'inflation_rate': 3.5, 'inflation_trend': 'stable',
        'interest_rate': 5.25, 'interest_trend': 'stable',
        'global_liquidity': 'moderate', 'liquidity_trend': 'tightening',
        'stock_market': 'bullish', 'bond_market': 'bearish',
        'gold_price': 2342.50, 'gold_trend': 'rising'
    }

def calculate_technical_indicators(df):
    logger.info("Calculating technical indicators")
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
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['signal']
        df['sma20'] = df['price'].rolling(window=20, min_periods=1).mean()
        df['std20'] = df['price'].rolling(window=20, min_periods=1).std().fillna(0)
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        df['sma50'] = df['price'].rolling(window=50, min_periods=1).mean()
        df['sma200'] = df['price'].rolling(window=200, min_periods=1).mean()
        logger.info("Technical indicators calculated successfully")
        return df.fillna(0)
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        return df # Return original df on error, or an empty one if preferred

def generate_chart(df, symbol, indicator_type):
    logger.info(f"Generating chart for {symbol} - {indicator_type}")
    if df is None or df.empty or indicator_type not in ['price', 'rsi', 'macd', 'volume']:
        logger.warning(f"Cannot generate chart for {symbol} - {indicator_type}. Invalid input.")
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
            ax.plot(df.index, df.get('signal'), label='Signal Line', linestyle='--')
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
        logger.info(f"Chart generated successfully for {symbol} - {indicator_type}")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error generating chart for {symbol} - {indicator_type}: {e}", exc_info=True)
        return None

def get_asset_rotation_strategy():
    logger.info("Generating asset rotation strategy")
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
        btc_alloc, eth_alloc, sol_alloc = 25, 20, 15 # Defaults
        if btc_perf > 10 and econ.get('usd_trend') == 'falling': btc_alloc = 40; btc_signal = 'BUY'
        elif btc_perf < -10 and econ.get('usd_trend') == 'rising': btc_alloc = 10; btc_signal = 'SELL'
        else: btc_signal = 'HOLD'
        allocations.append({'coin': 'Bitcoin (BTC)', 'allocation': btc_alloc, 'signal': btc_signal, 'performance': f"{btc_perf:.2f}%"})
        if eth_perf > 15: eth_alloc = 30; eth_signal = 'BUY'
        elif eth_perf < -15: eth_alloc = 10; eth_signal = 'SELL'
        else: eth_signal = 'HOLD'
        allocations.append({'coin': 'Ethereum (ETH)', 'allocation': eth_alloc, 'signal': eth_signal, 'performance': f"{eth_perf:.2f}%"})
        if sol_perf > 20: sol_alloc = 20; sol_signal = 'BUY'
        elif sol_perf < -20: sol_alloc = 5; sol_signal = 'SELL'
        else: sol_signal = 'HOLD'
        allocations.append({'coin': 'Solana (SOL)', 'allocation': sol_alloc, 'signal': sol_signal, 'performance': f"{sol_perf:.2f}%"})
        remaining = 100 - (btc_alloc + eth_alloc + sol_alloc)
        if trending and remaining > 0:
             per_trending = remaining / len(trending)
             for coin_item in trending: # Renamed to avoid conflict with outer 'coin'
                 allocations.append({'coin': f"{coin_item.get('name', 'N/A')} ({coin_item.get('symbol', 'N/A')})", 'allocation': round(per_trending), 'signal': 'BUY' if coin_item.get('score', 0) > 95 else 'RESEARCH', 'performance': 'Trending'})
        elif remaining > 0:
             allocations.append({'coin': 'Cash/Stablecoin', 'allocation': remaining, 'signal': 'HOLD', 'performance': 'N/A'})
        logger.info("Asset rotation strategy generated successfully")
        return allocations
    except Exception as e:
        logger.error(f"Error generating asset rotation strategy: {e}", exc_info=True)
        return []

def generate_buy_sell_recommendations():
    logger.info("Generating buy/sell recommendations")
    try:
        btc_data = calculate_technical_indicators(get_crypto_data('bitcoin', days=90))
        eth_data = calculate_technical_indicators(get_crypto_data('ethereum', days=90))
        sol_data = calculate_technical_indicators(get_crypto_data('solana', days=90))
        recommendations = []
        for name, data_df in [('Bitcoin', btc_data), ('Ethereum', eth_data), ('Solana', sol_data)]: # Renamed data to data_df
            if data_df.empty: continue
            last_row = data_df.iloc[-1]
            signal = 'HOLD'
            if last_row["macd"] > last_row["signal"] and last_row["rsi"] < 70: signal = 'BUY'
            elif last_row["macd"] < last_row["signal"] and last_row["rsi"] > 30: signal = 'SELL'
            if last_row["price"] < last_row["lower_band"]: signal = 'STRONG BUY'
            elif last_row["price"] > last_row["upper_band"]: signal = 'STRONG SELL'
            recommendations.append({
                'coin': name,
                'signal': signal,
                'current_price': f"{last_row['price']:.2f}",
                'rsi': f"{last_row['rsi']:.2f}",
                'macd': f"{last_row['macd']:.4f}",
                'signal_line': f"{last_row['signal']:.4f}",
                'bollinger_lower': f"{last_row['lower_band']:.2f}",
                'bollinger_upper': f"{last_row['upper_band']:.2f}"
            })
        logger.info("Buy/sell recommendations generated successfully")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating buy/sell recommendations: {e}", exc_info=True)
        return []

# --- Flask Routes --- 
@app.route('/')
def home():
    logger.info("Accessed home page")
    try:
        trending_coins = get_trending_coins()
        econ_indicators = get_economic_indicators()
        btc_data = get_crypto_data('bitcoin', days=2)
        eth_data = get_crypto_data('ethereum', days=2)
        sol_data = get_crypto_data('solana', days=2)
        market_overview = []
        if not btc_data.empty: market_overview.append({'name': 'Bitcoin', 'price': btc_data['price'].iloc[-1], 'change': (btc_data['price'].iloc[-1]/btc_data['price'].iloc[0]-1)*100 if len(btc_data['price']) > 1 else 0})
        if not eth_data.empty: market_overview.append({'name': 'Ethereum', 'price': eth_data['price'].iloc[-1], 'change': (eth_data['price'].iloc[-1]/eth_data['price'].iloc[0]-1)*100 if len(eth_data['price']) > 1 else 0})
        if not sol_data.empty: market_overview.append({'name': 'Solana', 'price': sol_data['price'].iloc[-1], 'change': (sol_data['price'].iloc[-1]/sol_data['price'].iloc[0]-1)*100 if len(sol_data['price']) > 1 else 0})
        return render_template('home.html', market_overview=market_overview, trending_coins=trending_coins, econ_indicators=econ_indicators)
    except Exception as e:
        logger.error(f"Error on home page: {e}", exc_info=True)
        return render_template('home.html', market_overview=[], trending_coins=[], econ_indicators={}) # Fallback for home page

@app.route('/initialize-db')
def initialize_db_route():
    key = request.args.get('key')
    if key != get_env_var('DB_INIT_KEY', 'default_secret_key_for_init'): # Use env var for key
        logger.warning(f"Unauthorized attempt to initialize DB with key: {key}")
        return "Unauthorized", 403
    try:
        with app.app_context():
            logger.info("Attempting to drop and create all database tables.")
            db.drop_all()
            db.create_all()
            # Create a default admin user if one doesn't exist
            if not User.query.filter_by(username='admin').first():
                hashed_password = bcrypt.generate_password_hash('defaultmin').decode('utf-8') # Ensure this is changed
                admin_user = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
                db.session.add(admin_user)
                # Create default preferences for admin
                admin_prefs = UserPreference(user_id=admin_user.id) # This will be set after admin_user is flushed
                db.session.add(admin_prefs)
                logger.info("Admin user and preferences created.")
            else:
                logger.info("Admin user already exists.")
            db.session.commit()
        logger.info("Database initialized successfully.")
        return "Database initialized!"
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        return f"Error initializing database: {e}", 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        try:
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.flush() # Flush to get user.id for preferences
            prefs = UserPreference(user_id=user.id)
            db.session.add(prefs)
            db.session.commit()
            flash('Your account has been created! You are now able to log in', 'success')
            logger.info(f"User {username} registered successfully.")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during registration for {username}: {e}", exc_info=True)
            flash('Registration failed. Username or email may already exist.', 'danger')
    return render_template('register.html', title='Register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=request.form.get('remember'))
            next_page = request.args.get('next')
            logger.info(f"User {email} logged in successfully.")
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
            logger.warning(f"Failed login attempt for {email}.")
    return render_template('login.html', title='Login')

@app.route('/logout')
def logout():
    if current_user.is_authenticated:
        logger.info(f"User {current_user.username} logged out.")
        logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    logger.info(f"User {current_user.username} accessed dashboard.")
    try:
        user_prefs = current_user.preferences if current_user.preferences else UserPreference(user_id=current_user.id) # Ensure prefs exist
        preferred_coins_list = user_prefs.preferred_coins.split(',') if user_prefs.preferred_coins else ['bitcoin', 'ethereum', 'solana']
        preferred_indicators_list = user_prefs.preferred_indicators.split(',') if user_prefs.preferred_indicators else ['price', 'volume', 'rsi', 'macd']
        
        charts = {}
        coin_data_dict = {}
        for coin_symbol_pref in preferred_coins_list:
            coin_symbol = coin_symbol_pref.strip().lower()
            if not coin_symbol: continue
            coin_data = calculate_technical_indicators(get_crypto_data(coin_symbol, days=90))
            coin_data_dict[coin_symbol] = coin_data
            charts[coin_symbol] = {}
            for indicator in preferred_indicators_list:
                chart_img = generate_chart(coin_data, coin_symbol, indicator.strip().lower())
                if chart_img:
                    charts[coin_symbol][indicator.strip().lower()] = chart_img
        
        asset_strategy = get_asset_rotation_strategy()
        buy_sell_recs = generate_buy_sell_recommendations()
        
        return render_template('dashboard.html', title='Dashboard', 
                               charts=charts, asset_strategy=asset_strategy, 
                               buy_sell_recs=buy_sell_recs, user_prefs=user_prefs,
                               preferred_coins_list=preferred_coins_list, # Pass this for the form
                               preferred_indicators_list=preferred_indicators_list # Pass this for the form
                               )
    except Exception as e:
        logger.error(f"Error on dashboard for user {current_user.username}: {e}", exc_info=True)
        flash('Error loading dashboard data.', 'danger')
        return render_template('dashboard.html', title='Dashboard', charts={}, asset_strategy=[], buy_sell_recs=[], user_prefs=current_user.preferences or UserPreference(user_id=current_user.id))

@app.route('/preferences', methods=['GET', 'POST'])
@login_required
def preferences():
    user_prefs = current_user.preferences
    if not user_prefs: # Should not happen if prefs are created at registration/login
        user_prefs = UserPreference(user_id=current_user.id)
        db.session.add(user_prefs)
        # db.session.commit() # Commit might be better done after form submission

    if request.method == 'POST':
        try:
            user_prefs.email_frequency = request.form.get('email_frequency', 'daily')
            user_prefs.preferred_coins = request.form.get('preferred_coins', 'BTC,ETH,SOL')
            user_prefs.preferred_indicators = request.form.get('preferred_indicators', 'price,volume,rsi,macd')
            user_prefs.theme = request.form.get('theme', 'light')
            db.session.commit()
            flash('Preferences updated successfully!', 'success')
            logger.info(f"User {current_user.username} updated preferences.")
            return redirect(url_for('preferences'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating preferences for {current_user.username}: {e}", exc_info=True)
            flash('Error updating preferences.', 'danger')
            
    return render_template('preferences.html', title='Preferences', user_prefs=user_prefs)

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('You do not have access to this page.', 'danger')
        return redirect(url_for('home'))
    try:
        users = User.query.all()
        return render_template('admin.html', title='Admin Dashboard', users=users)
    except Exception as e:
        logger.error(f"Error accessing admin dashboard: {e}", exc_info=True)
        flash('Error loading admin dashboard.', 'danger')
        return redirect(url_for('home'))

# --- Background Tasks (Example: Email Sending) ---
def send_email_async(subject, recipient, html_content):
    # This function would contain your actual email sending logic
    # For now, it just logs
    logger.info(f"Attempting to send email: Subject='{subject}', To='{recipient}'")
    # Example: using smtplib (ensure email server details are in env vars)
    # try:
    #     msg = MIMEMultipart()
    #     msg['From'] = get_env_var('SMTP_FROM_EMAIL')
    #     msg['To'] = recipient
    #     msg['Subject'] = subject
    #     msg.attach(MIMEText(html_content, 'html'))
    #     # If you have images, attach them like so:
    #     # with open('path/to/image.png', 'rb') as fp:
    #     #     img = MIMEImage(fp.read())
    #     #     img.add_header('Content-ID', '<image1>') # Match cid in HTML
    #     #     msg.attach(img)
    #     with smtplib.SMTP(get_env_var('SMTP_SERVER'), get_env_var('SMTP_PORT', 587)) as server:
    #         server.starttls()
    #         server.login(get_env_var('SMTP_USERNAME'), get_env_var('SMTP_PASSWORD'))
    #         server.send_message(msg)
    #     logger.info(f"Email sent successfully to {recipient}")
    # except Exception as e:
    #     logger.error(f"Failed to send email to {recipient}: {e}", exc_info=True)
    pass # Placeholder

@app.route('/send-test-email')
@login_required
def send_test_email_route():
    if not current_user.is_admin:
        return "Unauthorized", 403
    # Example usage
    subject = "Test Email from Crypto App"
    recipient = current_user.email
    html_content = "<h1>Hello!</h1><p>This is a test email from your Crypto Investment Strategy app.</p>"
    # To send it in the background:
    thread = threading.Thread(target=send_email_async, args=(subject, recipient, html_content))
    thread.start()
    flash(f"Test email dispatch initiated to {recipient}. Check logs for status.", 'info')
    return redirect(url_for('admin_dashboard'))

# --- Error Handling ---
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 Not Found: {request.url} (Error: {error})")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    # It's good practice to rollback the session in case of an internal error
    try:
        db.session.rollback()
        logger.error(f"500 Internal Server Error: {request.url} (Error: {error})", exc_info=True)
    except Exception as e:
        logger.error(f"Error during 500 error handler rollback: {e}", exc_info=True)
    return render_template('500.html'), 500

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    # Ensure the app context is available for db operations if run directly
    with app.app_context():
        try:
            # db.drop_all() # Uncomment if you want to clear DB on every local start
            db.create_all() # Ensure tables are created
            logger.info("Database tables checked/created.")
            # Optional: Create default admin if it doesn't exist, useful for first run
            if not User.query.filter_by(username='admin').first():
                hashed_password = bcrypt.generate_password_hash('defaultadmin').decode('utf-8')
                admin_user = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
                db.session.add(admin_user)
                admin_prefs = UserPreference(user_id=admin_user.id)
                db.session.add(admin_prefs)
                db.session.commit()
                logger.info("Default admin user and preferences created on startup.")
        except Exception as e:
            logger.error(f"Error during initial DB setup on startup: {e}", exc_info=True)
            
    logger.info(f"Application starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) # Debug should be False for Render

