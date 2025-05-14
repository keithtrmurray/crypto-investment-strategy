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

# --- Caching Setup ---
api_cache = {}
CACHE_DURATION = datetime.timedelta(minutes=30) # Cache data for 30 minutes
# --- End Caching Setup ---

# --- Improved Configuration Section --- 

def get_env_var(name, default=None, required=False):
    value = os.environ.get(name, default)
    log_value = '[SET]' if value else \'[NOT SET]' 
    if name in [\"SECRET_KEY\", \"DATABASE_URL\"] and value:
        log_value = \"[SET]\"
    elif value and name not in [\"SECRET_KEY\", \"DATABASE_URL\"]:
        log_value = value
        
    logger.info(f\"Environment variable {name}: {log_value}\")
    if required and not value:
        logger.error(f\"Required environment variable {name} is not set. Exiting.\")
        sys.exit(f\"Error: Missing required environment variable {name}\")
    return value

database_url = get_env_var(\'DATABASE_URL\', required=True)
if database_url.startswith(\"postgres://\"):
    database_url = database_url.replace(\"postgres://\", \"postgresql://\", 1)
    logger.info(\"Modified DATABASE_URL to use postgresql:// prefix\")

app.config[\"SQLALCHEMY_DATABASE_URI\"] = database_url
app.config[\"SQLALCHEMY_TRACK_MODIFICATIONS\"] = False
app.config[\"SECRET_KEY\"] = get_env_var(\'SECRET_KEY\', required=True)

# --- End Improved Configuration Section ---

try:
    db = SQLAlchemy(app)
    bcrypt = Bcrypt(app)
    login_manager = LoginManager(app)
    login_manager.login_view = \'login\'
    logger.info(\"Flask extensions initialized successfully\")
except Exception as e:
    logger.error(f\"Error initializing Flask extensions: {str(e)}\", exc_info=True)
    sys.exit(\"Failed to initialize Flask extensions.\")

os.makedirs(\'data\', exist_ok=True)
os.makedirs(\'static/img\', exist_ok=True)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    subscription_type = db.Column(db.String(20), default=\'free\')
    subscription_start = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    subscription_end = db.Column(db.DateTime, default=lambda: datetime.datetime.utcnow() + datetime.timedelta(days=30))
    preferences = db.relationship(\'UserPreference\', backref=\'user\', uselist=False)

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(\'user.id\'), nullable=False)
    email_frequency = db.Column(db.String(20), default=\'daily\')
    preferred_coins = db.Column(db.String(200), default=\'BTC,ETH,SOL\')
    preferred_indicators = db.Column(db.String(200), default=\'price,volume,rsi,macd\')
    theme = db.Column(db.String(20), default=\'light\')

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f\"Error loading user {user_id}: {str(e)}\", exc_info=True)
        return None

# --- Crypto data functions with Caching ---
def get_crypto_data(symbol, days=30):
    cache_key = (symbol, days)
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if datetime.datetime.utcnow() - timestamp < CACHE_DURATION:
            logger.info(f\"Returning cached crypto data for {symbol} ({days} days)\")
            return cached_data.copy() # Return a copy to prevent modification of cached DataFrame
        else:
            logger.info(f\"Cache expired for {symbol} ({days} days)\")

    logger.info(f\"Fetching crypto data for {symbol} ({days} days) from API\")
    try:
        url = f\"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart\"
        params = {\'vs_currency\': \'usd\', \'days\': days, \'interval\': \'daily\'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or \'prices\' not in data or \'total_volumes\' not in data:
             logger.warning(f\"Incomplete data received from CoinGecko for {symbol}\")
             raise ValueError(\"Incomplete data from API\")

        df = pd.DataFrame(data[\'prices\'], columns=[\'timestamp\', \'price\'])
        df[\'timestamp\'] = pd.to_datetime(df[\'timestamp\'], unit=\'ms\')
        df.set_index(\'timestamp\', inplace=True)
        
        volume_df = pd.DataFrame(data[\'total_volumes\'], columns=[\'timestamp\', \'volume\'])
        volume_df[\'timestamp\'] = pd.to_datetime(volume_df[\'timestamp\'], unit=\'ms\')
        volume_df.set_index(\'timestamp\', inplace=True)
        df[\'volume\'] = volume_df[\'volume\']
        
        api_cache[cache_key] = (df.copy(), datetime.datetime.utcnow()) # Store a copy
        logger.info(f\"Successfully fetched and cached data for {symbol}\")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f\"Network error fetching data for {symbol}: {e}\")
    except Exception as e:
        logger.error(f\"Error processing data for {symbol}: {e}\", exc_info=True)
        
    logger.warning(f\"Returning dummy data for {symbol} due to API error\")
    dates = pd.date_range(end=datetime.datetime.now(datetime.timezone.utc), periods=days)
    df = pd.DataFrame(index=dates)
    df[\'price\'] = [50000 - i * 100 + i * i * 5 for i in range(days)]
    df[\'volume\'] = [2000000000 - i * 10000000 + i * i * 500000 for i in range(days)]
    return df

def get_trending_coins():
    cache_key = \'trending_coins\'
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if datetime.datetime.utcnow() - timestamp < CACHE_DURATION:
            logger.info(\"Returning cached trending coins data\")
            return list(cached_data) # Return a copy
        else:
            logger.info(\"Cache expired for trending coins\")

    logger.info(\"Fetching trending coins from API\")
    try:
        url = \"https://api.coingecko.com/api/v3/search/trending\"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or \'coins\' not in data:
            logger.warning(\"Incomplete data received from CoinGecko for trending coins\")
            raise ValueError(\"Incomplete data from API\")
            
        trending = []
        for coin_data in data[\'coins\'][:5]:
            item = coin_data.get(\'item\')
            if not item: continue
            trending.append({
                \'id\': item.get(\'id\', \'N/A\'), \'name\': item.get(\'name\', \'N/A\'),
                \'symbol\': item.get(\'symbol\', \'N/A\'), \'market_cap_rank\': item.get(\'market_cap_rank\', \'N/A\'),
                \'price_btc\': item.get(\'price_btc\', 0), \'score\': item.get(\'score\', 0),
                \'thumb\': item.get(\'thumb\', \'\')
            })
        api_cache[cache_key] = (list(trending), datetime.datetime.utcnow()) # Store a copy
        logger.info(f\"Successfully fetched and cached {len(trending)} trending coins\")
        return trending
    except requests.exceptions.RequestException as e:
        logger.error(f\"Network error fetching trending coins: {e}\")
    except Exception as e:
        logger.error(f\"Error processing trending coins: {e}\", exc_info=True)
        
    logger.warning(\"Returning dummy trending coins data due to API error\")
    return [
        {\'id\': \'solana\', \'name\': \'Solana\', \'symbol\': \'SOL\', \'market_cap_rank\': 5, \'price_btc\': 0.00123, \'score\': 100, \'thumb\': \'\'},
        {\'id\': \'cardano\', \'name\': \'Cardano\', \'symbol\': \'ADA\', \'market_cap_rank\': 8, \'price_btc\': 0.00002, \'score\': 98, \'thumb\': \'\'}
    ]

def get_economic_indicators():
    logger.info(\"Fetching economic indicators (using dummy data)\")
    return {
        \'usd_index\': 104.2, \'usd_trend\': \'rising\',
        \'inflation_rate\': 3.5, \'inflation_trend\': \'stable\',
        \'interest_rate\': 5.25, \'interest_trend\': \'stable\',
        \'global_liquidity\': \'moderate\', \'liquidity_trend\': \'tightening\',
        \'stock_market\': \'bullish\', \'bond_market\': \'bearish\',
        \'gold_price\': 2342.50, \'gold_trend\': \'rising\'
    }

def calculate_technical_indicators(df):
    logger.info(\"Calculating technical indicators\")
    if df is None or df.empty: return pd.DataFrame()
    try:
        delta = df[\'price\'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, 1e-6) 
        rs = avg_gain / avg_loss
        df[\'rsi\'] = (100 - (100 / (1 + rs))).fillna(50)
        df[\'ema12\'] = df[\'price\'].ewm(span=12, adjust=False).mean()
        df[\'ema26\'] = df[\'price\'].ewm(span=26, adjust=False).mean()
        df[\'macd\'] = df[\'ema12\'] - df[\'ema26\']
        df[\'signal\'] = df[\'macd\'].ewm(span=9, adjust=False).mean()
        df[\'macd_histogram\'] = df[\'macd\'] - df[\'signal\']
        df[\'sma20\'] = df[\'price\'].rolling(window=20, min_periods=1).mean()
        df[\'std20\'] = df[\'price\'].rolling(window=20, min_periods=1).std().fillna(0)
        df[\'upper_band\'] = df[\'sma20\'] + (df[\'std20\'] * 2)
        df[\'lower_band\'] = df[\'sma20\'] - (df[\'std20\'] * 2)
        df[\'sma50\'] = df[\'price\'].rolling(window=50, min_periods=1).mean()
        df[\'sma200\'] = df[\'price\'].rolling(window=200, min_periods=1).mean()
        logger.info(\"Technical indicators calculated successfully\")
        return df.fillna(0)
    except Exception as e:
        logger.error(f\"Error calculating technical indicators: {e}\", exc_info=True)
        return df

def generate_chart(df, symbol, indicator_type):
    logger.info(f\"Generating chart for {symbol} - {indicator_type}\")
    if df is None or df.empty or indicator_type not in [\'price\', \'rsi\', \'macd\', \'volume\']:
        logger.warning(f\"Cannot generate chart for {symbol} - {indicator_type}. Invalid input.\")
        return None
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        if indicator_type == \'price\':
            ax.plot(df.index, df.get(\'price\'), label=\'Price\')
            ax.plot(df.index, df.get(\'sma20\'), label=\'SMA 20\', linestyle=\'--\')
            ax.plot(df.index, df.get(\'upper_band\'), label=\'Upper Band\', linestyle=\':\', color=\'grey\')
            ax.plot(df.index, df.get(\'lower_band\'), label=\'Lower Band\', linestyle=\':\', color=\'grey\')
            ax.set_title(f\'{symbol.upper()} Price & Bollinger Bands\')
            ax.set_ylabel(\'Price (USD)\')
        elif indicator_type == \'rsi\':
            ax.plot(df.index, df.get(\'rsi\'), label=\'RSI\')
            ax.axhline(y=70, color=\'r\', linestyle=\'--\', alpha=0.5)
            ax.axhline(y=30, color=\'g\', linestyle=\'--\', alpha=0.5)
            ax.set_title(f\'{symbol.upper()} RSI\')
            ax.set_ylabel(\'RSI\'); ax.set_ylim(0, 100)
        elif indicator_type == \'macd\':
            ax.plot(df.index, df.get(\'macd\'), label=\'MACD\')
            ax.plot(df.index, df.get(\'signal\'), label=\'Signal Line\', linestyle=\'--\')
            hist_values = pd.to_numeric(df.get(\'macd_histogram\'), errors=\'coerce\').fillna(0)
            ax.bar(df.index, hist_values, label=\'Histogram\', alpha=0.3, width=0.8)
            ax.set_title(f\'{symbol.upper()} MACD\')
            ax.set_ylabel(\'MACD Value\')
        elif indicator_type == \'volume\':
            ax.bar(df.index, df.get(\'volume\'), label=\'Volume\', alpha=0.6)
            ax.set_title(f\'{symbol.upper()} Volume\')
            ax.set_ylabel(\'Volume\')
        ax.set_xlabel(\'Date\'); ax.legend(); ax.grid(True)
        plt.xticks(rotation=45); plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format=\'png\'); buffer.seek(0)
        plt.close(fig)
        img_str = base64.b64encode(buffer.read()).decode(\'utf-8\')
        logger.info(f\"Chart generated successfully for {symbol} - {indicator_type}\")
        return f\"data:image/png;base64,{img_str}\"
    except Exception as e:
        logger.error(f\"Error generating chart for {symbol} - {indicator_type}: {e}\", exc_info=True)
        return None

def get_asset_rotation_strategy():
    logger.info(\"Generating asset rotation strategy\")
    try:
        btc_data = get_crypto_data(\'bitcoin\', days=30)
        eth_data = get_crypto_data(\'ethereum\', days=30)
        sol_data = get_crypto_data(\'solana\', days=30)
        btc_perf = (btc_data[\'price\'].iloc[-1] / btc_data[\'price\'].iloc[0] - 1) * 100 if not btc_data.empty and len(btc_data[\'price\']) > 1 else 0
        eth_perf = (eth_data[\'price\'].iloc[-1] / eth_data[\'price\'].iloc[0] - 1) * 100 if not eth_data.empty and len(eth_data[\'price\']) > 1 else 0
        sol_perf = (sol_data[\'price\'].iloc[-1] / sol_data[\'price\'].iloc[0] - 1) * 100 if not sol_data.empty and len(sol_data[\'price\']) > 1 else 0
        trending = get_trending_coins()
        econ = get_economic_indicators()
        allocations = []
        btc_alloc, eth_alloc, sol_alloc = 25, 20, 15 # Defaults
        if btc_perf > 10 and econ.get(\'usd_trend\') == \'falling\': btc_alloc = 40; btc_signal = \'BUY\'
        elif btc_perf < -10 and econ.get(\'usd_trend\') == \'rising\': btc_alloc = 10; btc_signal = \'SELL\'
        else: btc_signal = \'HOLD\'
        allocations.append({\'coin\': \'Bitcoin (BTC)\', \'allocation\': btc_alloc, \'signal\': btc_signal, \'performance\': f\"{btc_perf:.2f}%\"})
        if eth_perf > 15: eth_alloc = 30; eth_signal = \'BUY\'
        elif eth_perf < -15: eth_alloc = 10; eth_signal = \'SELL\'
        else: eth_signal = \'HOLD\'
        allocations.append({\'coin\': \'Ethereum (ETH)\', \'allocation\': eth_alloc, \'signal\': eth_signal, \'performance\': f\"{eth_perf:.2f}%\"})
        if sol_perf > 20: sol_alloc = 20; sol_signal = \'BUY\'
        elif sol_perf < -20: sol_alloc = 5; sol_signal = \'SELL\'
        else: sol_signal = \'HOLD\'
        allocations.append({\'coin\': \'Solana (SOL)\', \'allocation\': sol_alloc, \'signal\': sol_signal, \'performance\': f\"{sol_perf:.2f}%\"})
        remaining = 100 - (btc_alloc + eth_alloc + sol_alloc)
        if trending and remaining > 0:
             per_trending = remaining / len(trending)
             for coin in trending:
                 allocations.append({\'coin\': f\"{coin.get(\'name\', \'N/A\')} ({coin.get(\'symbol\', \'N/A\')})\", \'allocation\': round(per_trending), \'signal\': \'BUY\' if coin.get(\'score\', 0) > 95 else \'RESEARCH\', \'performance\': \'Trending\\'})
        elif remaining > 0:
             allocations.append({\'coin\': \'Cash/Stablecoin\', \'allocation\': remaining, \'signal\': \'HOLD\', \'performance\': \'N/A\\'})
        logger.info(\"Asset rotation strategy generated successfully\")
        return allocations
    except Exception as e:
        logger.error(f\"Error generating asset rotation strategy: {e}\", exc_info=True)
        return []

def generate_buy_sell_recommendations():
    logger.info(\"Generating buy/sell recommendations\")
    try:
        btc_data = calculate_technical_indicators(get_crypto_data(\'bitcoin\', days=90))
        eth_data = calculate_technical_indicators(get_crypto_data(\'ethereum\', days=90))
        sol_data = calculate_technical_indicators(get_crypto_data(\'solana\', days=90))
        recommendations = []
        for name, data in [(\'Bitcoin\', btc_data), (\'Ethereum\', eth_data), (\'Solana\', sol_data)]:
            if data.empty: continue
            last_row = data.iloc[-1]
            signal = \'HOLD\'
            if last_row["macd"] > last_row["signal"] and last_row["rsi"] < 70: signal = \'BUY\'
            elif last_row["macd"] < last_row["signal"] and last_row["rsi"] > 30: signal = \'SELL\'
            if last_row["price"] < last_row["lower_band"]: signal = \'STRONG BUY\'
            elif last_row["price"] > last_row["upper_band"]: signal = \'STRONG SELL\'
            recommendations.append({
                \'coin

# --- Flask Routes --- 
@app.route(\'/\')
def home():
    logger.info(\"Accessed home page\")
    try:
        trending_coins = get_trending_coins()
        econ_indicators = get_economic_indicators() # This was the fix for undefined econ_indicators
        btc_data = get_crypto_data(\'bitcoin\', days=2)
        eth_data = get_crypto_data(\'ethereum\', days=2)
        sol_data = get_crypto_data(\'solana\', days=2)
        market_overview = []
        if not btc_data.empty: market_overview.append({\'name\': \'Bitcoin\', \'price\': btc_data[\'price\'].iloc[-1], \'change\': (btc_data[\'price\'].iloc[-1]/btc_data[\'price\'].iloc[0]-1)*100 if len(btc_data[\'price\']) > 1 else 0})
        if not eth_data.empty: market_overview.append({\'name\': \'Ethereum\', \'price\': eth_data[\'price\'].iloc[-1], \'change\': (eth_data[\'price\'].iloc[-1]/eth_data[\'price\'].iloc[0]-1)*100 if len(eth_data[\'price\']) > 1 else 0})
        if not sol_data.empty: market_overview.append({\'name\': \'Solana\', \'price\': sol_data[\'price\'].iloc[-1], \'change\': (sol_data[\'price\'].iloc[-1]/sol_data[\'price\'].iloc[0]-1)*100 if len(sol_data[\'price\']) > 1 else 0})
        return render_template(\'home.html\', market_overview=market_overview, trending_coins=trending_coins, econ_indicators=econ_indicators)
    except Exception as e:
        logger.error(f\"Error on home page: {e}\", exc_info=True)
        return render_template(\'home.html\', market_overview=[], trending_coins=[], econ_indicators={}) # Fallback for home page

@app.route(\'/register\', methods=[\'GET\', \'POST\'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for(\'home\'))
    if request.method == \'POST\':
        username = request.form.get(\'username\')
        email = request.form.get(\'email\')
        password = request.form.get(\'password\')
        hashed_password = bcrypt.generate_password_hash(password).decode(\'utf-8\')
        try:
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash(\'Your account has been created! You are now able to log in\', \'success\')
            logger.info(f\"User {email} registered successfully.\")
            return redirect(url_for(\'login\'))
        except Exception as e:
            db.session.rollback()
            logger.error(f\"Error registering user {email}: {e}\", exc_info=True)
            flash(\'Registration failed. Email or username may already exist.\', \'danger\')
    return render_template(\'register.html\', title=\'Register\')

@app.route(\'/login\', methods=[\'GET\', \'POST\'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for(\'home\'))
    if request.method == \'POST\':
        email = request.form.get(\'email\')
        password = request.form.get(\'password\')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            logger.info(f\"User {email} logged in successfully.\")
            next_page = request.args.get(\'next\')
            return redirect(next_page) if next_page else redirect(url_for(\'home\'))
        else:
            logger.warning(f\"Failed login attempt for user {email}.\")
            flash(\'Login Unsuccessful. Please check email and password\', \'danger\')
    return render_template(\'login.html\', title=\'Login\')

@app.route(\'/logout\')
def logout():
    if current_user.is_authenticated:
        logger.info(f\"User {current_user.email} logged out.\")
        logout_user()
    return redirect(url_for(\'home\'))

@app.route(\'/dashboard\')
@login_required
def dashboard():
    logger.info(f\"User {current_user.email} accessed dashboard\")
    try:
        btc_data = calculate_technical_indicators(get_crypto_data(\'bitcoin\'))
        eth_data = calculate_technical_indicators(get_crypto_data(\'ethereum\'))
        sol_data = calculate_technical_indicators(get_crypto_data(\'solana\'))
        
        charts = {
            \'btc_price_chart\': generate_chart(btc_data, \'bitcoin\', \'price\'),
            \'btc_rsi_chart\': generate_chart(btc_data, \'bitcoin\', \'rsi\'),
            \'btc_macd_chart\': generate_chart(btc_data, \'bitcoin\', \'macd\'),
            \'eth_price_chart\': generate_chart(eth_data, \'ethereum\', \'price\'),
            \'eth_rsi_chart\': generate_chart(eth_data, \'ethereum\', \'rsi\'),
            \'eth_macd_chart\': generate_chart(eth_data, \'ethereum\', \'macd\'),
            \'sol_price_chart\': generate_chart(sol_data, \'solana\', \'price\'),
            \'sol_rsi_chart\': generate_chart(sol_data, \'solana\', \'rsi\'),
            \'sol_macd_chart\': generate_chart(sol_data, \'solana\', \'macd\'),
        }
        recommendations = generate_buy_sell_recommendations()
        rotation_strategy = get_asset_rotation_strategy()
        return render_template(\'dashboard.html\', title=\'Dashboard\', charts=charts, recommendations=recommendations, rotation_strategy=rotation_strategy)
    except Exception as e:
        logger.error(f\"Error on dashboard for user {current_user.email}: {e}\", exc_info=True)
        flash(\'Error loading dashboard data. Please try again later.\', \'danger\')
        return render_template(\'dashboard.html\', title=\'Dashboard\', charts={}, recommendations=[], rotation_strategy=[])

@app.route(\'/information_sources\')
def information_sources():
    logger.info(\"Accessed information sources page\")
    return render_template(\'information_sources.html\', title=\'Information Sources\')

@app.route(\'/initialize-db\')
def initialize_db():
    retrieved_env_key = os.environ.get(\'DB_INIT_KEY\')
    key_from_url = request.args.get(\'key\')
    logger.info(f\"DIAGNOSTIC: DB_INIT_KEY from environment is: \'{retrieved_env_key}\")
    logger.info(f\"DIAGNOSTIC: Key provided in URL is: \'{key_from_url}\")
    
    expected_key = get_env_var(\'DB_INIT_KEY\')
    if not expected_key or key_from_url != expected_key:
        logger.warning(f\"Unauthorized attempt to initialize DB. URL key: \'{key_from_url}\', Expected key (from env): \'{expected_key}\")
        return \"Unauthorized\", 401
    try:
        with app.app_context():
            db.drop_all() # Use with caution in production
            db.create_all()
            logger.info(\"Database tables dropped and recreated.\")
            if not User.query.filter_by(email=\'admin@example.com\').first():
                hashed_password = bcrypt.generate_password_hash(\'admin123\').decode(\'utf-8\')
                admin_user = User(username=\'admin\', email=\'admin@example.com\', password=hashed_password, is_admin=True)
                db.session.add(admin_user)
                db.session.commit()
                logger.info(\"Admin user created successfully.\")
            else:
                logger.info(\"Admin user already exists.\")
        return \"Database initialized and admin user created/verified successfully!\"
    except Exception as e:
        logger.error(f\"Error initializing database: {e}\", exc_info=True)
        return f\"Error initializing database: {e}\", 500

# --- Email Sending Functionality (Placeholder) ---
def send_email_report(user_email, subject, body_html, image_paths=None):
    logger.info(f\"Attempting to send email report to {user_email}\")
    # This is a placeholder. Actual email sending requires SMTP server setup.
    # For Render, you might use a service like SendGrid and their API.
    logger.warning(\"Email sending is currently a placeholder and not fully implemented.\")
    return True # Simulate success

# --- Background Task for Email Reports (Placeholder) ---
def daily_email_task():
    while True:
        logger.info(\"Daily email task running (placeholder).\")
        # Query users who want daily reports
        # Generate reports
        # Send emails
        time.sleep(24 * 60 * 60) # Sleep for 24 hours

if __name__ == \'__main__\':
    # For local development, you might run this directly.
    # For Render, Gunicorn will be used as specified in render.yaml.
    # Ensure DB_INIT_KEY is set as an environment variable for local testing of /initialize-db
    logger.info(\"Starting Flask app directly (for local development or testing)\")
    # threading.Thread(target=daily_email_task, daemon=True).start()
    app.run(host=\'0.0.0.0\', port=int(os.environ.get(\'PORT\', 5000)), debug=False)

