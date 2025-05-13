#!/usr/bin/env python3
"""
Web-based Crypto Investment Strategy service - Render Fixed with Caching
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
    log_value = "[SET]" if value else "[NOT SET]" 
    if name in ["SECRET_KEY", "DATABASE_URL"] and value:
        log_value = "[SET]"
    elif value and name not in ["SECRET_KEY", "DATABASE_URL"]:
        log_value = value
        
    logger.info(f"Environment variable {name}: {log_value}")
    if required and not value:
        logger.error(f"Required environment variable {name} is not set. Exiting.")
        sys.exit(f"Error: Missing required environment variable {name}")
    return value

database_url = get_env_var('DATABASE_URL', required=True)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    logger.info("Modified DATABASE_URL to use postgresql:// prefix")

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = get_env_var('SECRET_KEY', required=True)

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
            return cached_data.copy() # Return a copy to prevent modification of cached DataFrame
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
        
        api_cache[cache_key] = (df.copy(), datetime.datetime.utcnow()) # Store a copy
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
            return list(cached_data) # Return a copy
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
        for coin_data in data['coins'][:5]:
            item = coin_data.get('item')
            if not item: continue
            trending.append({
                'id': item.get('id', 'N/A'), 'name': item.get('name', 'N/A'),
                'symbol': item.get('symbol', 'N/A'), 'market_cap_rank': item.get('market_cap_rank', 'N/A'),
                'price_btc': item.get('price_btc', 0), 'score': item.get('score', 0),
                'thumb': item.get('thumb', '')
            })
        api_cache[cache_key] = (list(trending), datetime.datetime.utcnow()) # Store a copy
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
        return df

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
             for coin in trending:
                 allocations.append({'coin': f"{coin.get('name', 'N/A')} ({coin.get('symbol', 'N/A')})", 'allocation': round(per_trending), 'signal': 'BUY' if coin.get('score', 0) > 95 else 'RESEARCH', 'performance': 'Trending'})
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
        econ = get_economic_indicators()
        recommendations = []
        for coin_name, data in [('Bitcoin (BTC)', btc_data), ('Ethereum (ETH)', eth_data), ('Solana (SOL)', sol_data)]:
            if not data.empty:
                rsi = data['rsi'].iloc[-1]; macd_hist = data['macd_histogram'].iloc[-1]
                price = data['price'].iloc[-1]; sma50 = data['sma50'].iloc[-1]
                rec, reason = 'HOLD', 'Mixed signals'
                if rsi < 30 and macd_hist > 0 and (coin_name.startswith('Bitcoin') and econ.get('usd_trend') == 'falling'): rec, reason = 'STRONG BUY', 'Oversold, +MACD, weakening USD'
                elif rsi < 30 and macd_hist > 0 : rec, reason = 'STRONG BUY', 'Oversold, +MACD'
                elif rsi > 70 and macd_hist < 0 and (coin_name.startswith('Bitcoin') and econ.get('usd_trend') == 'rising'): rec, reason = 'STRONG SELL', 'Overbought, -MACD, strengthening USD'
                elif rsi > 70 and macd_hist < 0 : rec, reason = 'STRONG SELL', 'Overbought, -MACD'
                elif price > sma50 and macd_hist > 0: rec, reason = 'BUY', 'Price > 50-MA, +MACD'
                elif price < sma50 and macd_hist < 0: rec, reason = 'SELL', 'Price < 50-MA, -MACD'
                recommendations.append({'coin': coin_name, 'price': f"${price:,.2f}", 'recommendation': rec, 'reason': reason})
            else: logger.warning(f"Skipping {coin_name} recommendation due to missing data")
        trending = get_trending_coins()
        for coin in trending:
            rec = 'RESEARCH' if coin.get('score', 0) < 95 else 'POTENTIAL BUY'
            recommendations.append({'coin': f"{coin.get('name', 'N/A')} ({coin.get('symbol', 'N/A')})", 'price': f"{coin.get('price_btc', 0):.8f} BTC", 'recommendation': rec, 'reason': f"Trending (Score: {coin.get('score',0)})"})
        logger.info("Buy/sell recommendations generated successfully")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating buy/sell recommendations: {e}", exc_info=True)
        return []

def send_daily_email_report(): # Simplified, more to do for robust email
    logger.info("Starting daily email report generation")
    try:
        user_email = get_env_var("PRIMARY_USER_EMAIL", "keithtrmurray@aol.com")
        recommendations = generate_buy_sell_recommendations()
        allocations = get_asset_rotation_strategy()
        econ_indicators = get_economic_indicators()
        trending_coins = get_trending_coins()
        # Charts are intensive, consider generating them only if email creds are set
        html_body = render_template("email_report_template.html", 
                                    recommendations=recommendations, 
                                    allocations=allocations, 
                                    econ_indicators=econ_indicators, 
                                    trending_coins=trending_coins,
                                    # Charts can be added here if generated
                                    current_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sender_email = get_env_var("EMAIL_USER")
        sender_password = get_env_var("EMAIL_PASS")
        if not sender_email or not sender_password:
            logger.warning("Email credentials not set. Saving report locally.")
            with open(f"data/daily_report_{datetime.date.today()}.html", "w") as f: f.write(html_body)
            return
        message = MIMEMultipart("related")
        message["Subject"] = "Crypto Investment Strategy - Daily Report"
        message["From"] = sender_email; message["To"] = user_email
        message.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(get_env_var("EMAIL_SMTP_SERVER", "smtp.aol.com"), int(get_env_var("EMAIL_SMTP_PORT", 587))) as server:
            server.starttls(); server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, message.as_string())
            logger.info("Email sent successfully")
    except Exception as e:
        logger.error(f"Error in send_daily_email_report: {e}", exc_info=True)

def run_daily_email_scheduler():
    logger.info("Email scheduler thread started")
    while True:
        now = datetime.datetime.now()
        if now.hour == 8 and now.minute == 0: # Send at 8:00 AM UTC
            send_daily_email_report()
            time.sleep(65) 
        time.sleep(60)

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
        return render_template('home.html', market_overview=[], trending_coins=[], econ_indicators={})
        flash('Error loading homepage data. Some information may be missing.', 'danger')
        return render_template('home.html', market_overview=[], trending_coins=[], econ_indicators={})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password: flash('Passwords do not match!', 'danger'); return redirect(url_for('register'))
        try:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            # Create default preferences
            prefs = UserPreference(user=user)
            db.session.add(prefs)
            db.session.commit()
            flash('Your account has been created! You are now able to log in', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during registration: {e}", exc_info=True)
            flash(f'Registration failed: An error occurred. Please try again. Details: {str(e)}', 'danger')
    return render_template('register.html', title='Register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        try:
            user = User.query.filter_by(email=email).first()
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user, remember=remember)
                next_page = request.args.get('next')
                flash('Login Successful!', 'success')
                return redirect(next_page) if next_page else redirect(url_for('home'))
            else:
                flash('Login Unsuccessful. Please check email and password', 'danger')
        except Exception as e:
            logger.error(f"Error during login: {e}", exc_info=True)
            flash(f'Login failed: An error occurred. {str(e)}', 'danger')
    return render_template('login.html', title='Login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    prefs = current_user.preferences if current_user.preferences else UserPreference(user=current_user)
    if request.method == 'POST':
        try:
            current_user.username = request.form.get('username', current_user.username)
            current_user.email = request.form.get('email', current_user.email)
            prefs.email_frequency = request.form.get('email_frequency', prefs.email_frequency)
            prefs.preferred_coins = request.form.get('preferred_coins', prefs.preferred_coins)
            prefs.preferred_indicators = request.form.get('preferred_indicators', prefs.preferred_indicators)
            prefs.theme = request.form.get('theme', prefs.theme)
            if not current_user.preferences: db.session.add(prefs) # Add if new
            db.session.commit()
            flash('Your account has been updated!', 'success')
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating account: {e}", exc_info=True)
            flash(f'Account update failed: {str(e)}', 'danger')
        return redirect(url_for('account'))
    return render_template('account.html', title='Account', prefs=prefs)

@app.route('/market_data')
def market_data():
    logger.info("Accessed market data page")
    try:
        coins_to_display = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot']
        charts = {}
        for coin in coins_to_display:
            data = calculate_technical_indicators(get_crypto_data(coin, days=90))
            if not data.empty:
                charts[coin] = {
                    'price': generate_chart(data, coin, 'price'),
                    'rsi': generate_chart(data, coin, 'rsi'),
                    'macd': generate_chart(data, coin, 'macd'),
                    'volume': generate_chart(data, coin, 'volume')
                }
            else:
                charts[coin] = {'price': None, 'rsi': None, 'macd': None, 'volume': None}
                logger.warning(f"No data to generate charts for {coin} on market_data page")
        return render_template('market_data.html', title='Market Data', charts=charts, coins=coins_to_display)
    except Exception as e:
        logger.error(f"Error on market_data page: {e}", exc_info=True)
        flash('Error loading market data. Some information may be missing.', 'danger')
        return render_template('market_data.html', title='Market Data', charts={}, coins=[])

@app.route('/education')
def education():
    logger.info("Accessed education page")
    # Placeholder for educational content
    return render_template('education.html', title='Education')

@app.route('/information_sources')
def information_sources():
    logger.info("Accessed information sources page")
    return render_template('information_sources.html', title='Information Sources')

@app.route('/api/recommendations')
@login_required # Or remove if public API
def api_recommendations():
    try:
        recommendations = generate_buy_sell_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error in /api/recommendations: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/asset_rotation')
@login_required # Or remove if public API
def api_asset_rotation():
    try:
        strategy = get_asset_rotation_strategy()
        return jsonify(strategy)
    except Exception as e:
        logger.error(f"Error in /api/asset_rotation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Database Initialization Route (for Render or environments without shell access)
@app.route('/initialize-db')
def init_db_route():
    try:
        init_key = request.args.get('key')
        # IMPORTANT: Change this key or protect this route in a production environment!
        if init_key != get_env_var("DB_INIT_KEY", "setup123"):
            logger.warning(f"Unauthorized attempt to initialize DB with key: {init_key}")
            return "Unauthorized", 401
        
        logger.info("Initializing database from web route...")
        with app.app_context(): # Ensure we are within application context
            db.create_all()
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                logger.info("Creating default admin user...")
                hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
                admin = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
                db.session.add(admin)
                prefs = UserPreference(user=admin)
                db.session.add(prefs)
                db.session.commit()
                return "Database initialized and admin user created successfully!"
            else:
                return "Database initialized. Admin user already exists."
    except Exception as e:
        logger.error(f"Error initializing database via web route: {e}", exc_info=True)
        return f"Error initializing database: {str(e)}<br><pre>{traceback.format_exc()}</pre>", 500

if __name__ == '__main__':
    # Check for command-line argument to initialize DB (for local dev)
    if len(sys.argv) > 1 and sys.argv[1] == '--initialize-db':
        with app.app_context(): # Ensure we are within application context
            logger.info("Initializing database from command line...")
            db.create_all()
            # Create a default admin user if it doesn't exist
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                logger.info("Creating default admin user...")
                hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
                admin = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
                db.session.add(admin)
                prefs = UserPreference(user=admin)
                db.session.add(prefs)
                db.session.commit()
                logger.info("Database initialized and admin user created.")
            else:
                logger.info("Admin user already exists.")
            sys.exit(0)
    
    # Start email scheduler in a separate thread if not in debug mode (Render runs in prod)
    if not app.debug:
        email_thread = threading.Thread(target=run_daily_email_scheduler, daemon=True)
        email_thread.start()
        logger.info("Email scheduler thread initiated.")

    # Gunicorn will be used by Render, this is for local dev
    app.run(host='0.0.0.0', port=int(get_env_var("PORT", 5000)), debug=False) # debug=False for Render like env

