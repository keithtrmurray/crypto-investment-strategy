#!/usr/bin/env python3
"""
Web-based Crypto Investment Strategy service - Optimized for Free Tier

Includes:
- Reduced API call frequency (target 4 times/day)
- Enhanced caching (12 hours)
- Database storage for historical data
- Legal disclaimer integration
"""

import os
import sys
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
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
from sqlalchemy import desc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- Optimized Caching & Rate Limiting Setup ---
api_cache = {}
CACHE_DURATION = datetime.timedelta(hours=12)  # Increased to 12 hours
CACHE_FILE = "data/api_cache.json"
API_REFRESH_INTERVAL = datetime.timedelta(hours=6) # Target 4 times per day

RATE_LIMIT_RESET = datetime.timedelta(minutes=60)
last_rate_limit_hit = None
api_call_timestamps = []
MAX_CALLS_PER_MINUTE = 10
BACKOFF_FACTOR = 1.5
MAX_RETRIES = 3
# --- End Setup ---

# --- Configuration Section ---
def get_env_var(name, default=None, required=False):
    value = os.environ.get(name, default)
    log_value = "[SET]" if value and name in ["SECRET_KEY", "DATABASE_URL"] else value if value else "[NOT SET]"
    logger.info(f"Environment variable {name}: {log_value}")
    if required and not value:
        logger.error(f"Required environment variable {name} is not set. Exiting.")
        sys.exit(f"Error: Missing required environment variable {name}")
    return value

database_url = get_env_var("DATABASE_URL", required=True)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    logger.info("Modified DATABASE_URL to use postgresql:// prefix")

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = get_env_var("SECRET_KEY", required=True)
# --- End Configuration ---

try:
    db = SQLAlchemy(app)
    bcrypt = Bcrypt(app)
    login_manager = LoginManager(app)
    login_manager.login_view = "login"
    logger.info("Flask extensions initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Flask extensions: {str(e)}", exc_info=True)
    sys.exit("Failed to initialize Flask extensions.")

os.makedirs("data", exist_ok=True)
os.makedirs("static/img", exist_ok=True)

# --- Cache Management ---
def load_cache_from_disk():
    global api_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache_data = json.load(f)
            restored_cache = {}
            for key_str, (data, timestamp_str) in cache_data.items():
                key = tuple(json.loads(key_str)) if key_str.startswith("[") else key_str
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                restored_cache[key] = (data, timestamp)
            api_cache = restored_cache
            logger.info(f"Loaded {len(api_cache)} items from cache file")
    except Exception as e:
        logger.error(f"Error loading cache from disk: {e}", exc_info=True)
        api_cache = {}

def save_cache_to_disk():
    try:
        serializable_cache = {}
        for key, (data, timestamp) in api_cache.items():
            key_str = json.dumps(key) if isinstance(key, tuple) else key
            serializable_cache[key_str] = (data, timestamp.isoformat())
        with open(CACHE_FILE, "w") as f:
            json.dump(serializable_cache, f)
        logger.info(f"Saved {len(api_cache)} items to cache file")
    except Exception as e:
        logger.error(f"Error saving cache to disk: {e}", exc_info=True)

load_cache_from_disk()

def schedule_cache_save():
    save_cache_to_disk()
    threading.Timer(900, schedule_cache_save).start()

schedule_cache_save()
# --- End Cache Management ---

# --- Rate Limit Management ---
def can_make_api_call():
    global api_call_timestamps, last_rate_limit_hit
    now = datetime.datetime.utcnow()
    if last_rate_limit_hit is not None:
        time_since_limit = now - last_rate_limit_hit
        if time_since_limit < RATE_LIMIT_RESET:
            logger.warning(f"Still in rate limit cooling period. {(RATE_LIMIT_RESET - time_since_limit).total_seconds():.0f} seconds remaining.")
            return False
        else:
            last_rate_limit_hit = None
            api_call_timestamps = []
    one_minute_ago = now - datetime.timedelta(minutes=1)
    api_call_timestamps = [ts for ts in api_call_timestamps if ts > one_minute_ago]
    return len(api_call_timestamps) < MAX_CALLS_PER_MINUTE

def record_api_call():
    global api_call_timestamps
    api_call_timestamps.append(datetime.datetime.utcnow())

def record_rate_limit_hit():
    global last_rate_limit_hit
    last_rate_limit_hit = datetime.datetime.utcnow()
    logger.warning(f"Rate limit hit. Cooling period started for {RATE_LIMIT_RESET.total_seconds():.0f} seconds.")

def api_request_with_backoff(url, params=None, timeout=10):
    if not can_make_api_call():
        logger.warning("Rate limit prevention: Skipping API call")
        raise Exception("Rate limit prevention active")
    for attempt in range(MAX_RETRIES):
        try:
            record_api_call()
            time.sleep(random.uniform(0.1, 0.5) * attempt)
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                record_rate_limit_hit()
                wait_time = (BACKOFF_FACTOR ** attempt) * 2
                logger.warning(f"Rate limit hit. Backing off for {wait_time:.1f} seconds (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            wait_time = (BACKOFF_FACTOR ** attempt) * 2
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Request failed: {e}. Retrying in {wait_time:.1f} seconds (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                logger.error(f"Request failed after {MAX_RETRIES} attempts: {e}")
                raise
    raise Exception(f"API request failed after {MAX_RETRIES} attempts")
# --- End Rate Limit Management ---

# --- Database Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    subscription_type = db.Column(db.String(20), default="free")
    subscription_start = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    subscription_end = db.Column(db.DateTime, default=lambda: datetime.datetime.utcnow() + datetime.timedelta(days=30))
    preferences = db.relationship("UserPreference", backref="user", uselist=False)
    email_preferences = db.relationship("EmailPreference", backref="user_email_prefs", uselist=False) # Renamed backref

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    email_frequency = db.Column(db.String(20), default="daily")
    preferred_coins = db.Column(db.String(200), default="BTC,ETH,SOL")
    preferred_indicators = db.Column(db.String(200), default="price,volume,rsi,macd")
    theme = db.Column(db.String(20), default="light")

class EmailPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    receive_daily = db.Column(db.Boolean, default=True)
    receive_weekly = db.Column(db.Boolean, default=True)
    receive_monthly = db.Column(db.Boolean, default=True)
    include_market_summary = db.Column(db.Boolean, default=True)
    include_portfolio = db.Column(db.Boolean, default=True)
    include_recommendations = db.Column(db.Boolean, default=True)
    include_trending = db.Column(db.Boolean, default=True)
    price_alert_threshold = db.Column(db.Float, default=5.0)
    rsi_overbought = db.Column(db.Float, default=70.0)
    rsi_oversold = db.Column(db.Float, default=30.0)
    preferred_coins = db.Column(db.String(500), default="BTC,ETH,SOL")
    html_format = db.Column(db.Boolean, default=True)

class HistoricalPrice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    price = db.Column(db.Float)
    volume = db.Column(db.Float)
    source = db.Column(db.String(50), default="coingecko")
    fetched_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class ApiFetchLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    endpoint = db.Column(db.String(100), nullable=False, index=True)
    symbol = db.Column(db.String(20), index=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, index=True)
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.String(200))
# --- End Database Models ---

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {str(e)}", exc_info=True)
        return None

# --- Data Fetching & Processing ---
def get_last_fetch_time(endpoint, symbol=None):
    query = ApiFetchLog.query.filter_by(endpoint=endpoint, success=True)
    if symbol:
        query = query.filter_by(symbol=symbol)
    last_log = query.order_by(desc(ApiFetchLog.timestamp)).first()
    return last_log.timestamp if last_log else None

def log_api_fetch(endpoint, symbol=None, success=True, error_message=None):
    try:
        log_entry = ApiFetchLog(
            endpoint=endpoint,
            symbol=symbol,
            success=success,
            error_message=error_message
        )
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error logging API fetch: {e}")

def get_crypto_data_from_db(symbol, days=30):
    try:
        start_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        data = HistoricalPrice.query.filter(
            HistoricalPrice.symbol == symbol,
            HistoricalPrice.timestamp >= start_date
        ).order_by(HistoricalPrice.timestamp).all()
        
        if not data:
            return None
            
        df = pd.DataFrame([(d.timestamp, d.price, d.volume) for d in data], columns=["timestamp", "price", "volume"])
        df.set_index("timestamp", inplace=True)
        logger.info(f"Retrieved {len(df)} records for {symbol} from DB")
        return df
    except Exception as e:
        logger.error(f"Error fetching data from DB for {symbol}: {e}")
        return None

def save_crypto_data_to_db(symbol, df):
    try:
        records = []
        for timestamp, row in df.iterrows():
            # Check if record already exists for this timestamp and symbol
            exists = HistoricalPrice.query.filter_by(symbol=symbol, timestamp=timestamp).first()
            if not exists:
                record = HistoricalPrice(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=row.get("price"),
                    volume=row.get("volume")
                )
                records.append(record)
        
        if records:
            db.session.bulk_save_objects(records)
            db.session.commit()
            logger.info(f"Saved {len(records)} new records for {symbol} to DB")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving data to DB for {symbol}: {e}")

def get_crypto_data(symbol, days=30):
    endpoint = "market_chart"
    cache_key = (symbol, days)
    now = datetime.datetime.utcnow()
    
    # 1. Check DB first
    df_db = get_crypto_data_from_db(symbol, days)
    if df_db is not None and not df_db.empty:
        # Check how recent the latest data point is
        latest_db_time = df_db.index.max().replace(tzinfo=None) # Ensure timezone naive for comparison
        if now - latest_db_time < API_REFRESH_INTERVAL:
             logger.info(f"Using recent data for {symbol} from DB (latest: {latest_db_time})")
             return df_db.copy()
        else:
            logger.info(f"DB data for {symbol} is older than refresh interval ({latest_db_time}). Checking cache/API.")

    # 2. Check Cache
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if now - timestamp < CACHE_DURATION:
            logger.info(f"Returning cached crypto data for {symbol} ({days} days)")
            # Convert cached list of lists back to DataFrame
            df_cache = pd.DataFrame(cached_data, columns=["timestamp", "price", "volume"])
            df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"], unit="ms")
            df_cache.set_index("timestamp", inplace=True)
            return df_cache.copy()
        else:
            logger.info(f"Cache expired for {symbol} ({days} days)")

    # 3. Check if API call is allowed (based on last fetch time)
    last_fetch = get_last_fetch_time(endpoint, symbol)
    if last_fetch and (now - last_fetch < API_REFRESH_INTERVAL):
        logger.warning(f"Skipping API call for {symbol}. Last fetch was at {last_fetch} (within {API_REFRESH_INTERVAL}). Using DB/dummy data.")
        if df_db is not None: return df_db.copy()
        # Fallback to dummy data if DB also failed
        logger.warning(f"Returning dummy data for {symbol} due to API refresh interval.")
        dates = pd.date_range(end=now, periods=days)
        df_dummy = pd.DataFrame(index=dates)
        df_dummy["price"] = [50000 - i * 100 + i * i * 5 for i in range(days)]
        df_dummy["volume"] = [2000000000 - i * 10000000 + i * i * 500000 for i in range(days)]
        return df_dummy

    # 4. Fetch from API
    logger.info(f"Fetching crypto data for {symbol} ({days} days) from API")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        data = api_request_with_backoff(url, params=params)
        log_api_fetch(endpoint, symbol, success=True)
        
        if not data or "prices" not in data or "total_volumes" not in data:
             logger.warning(f"Incomplete data received from CoinGecko for {symbol}")
             raise ValueError("Incomplete data from API")

        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        volume_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms")
        volume_df.set_index("timestamp", inplace=True)
        df["volume"] = volume_df["volume"]
        
        # Save to cache (store as list of lists for JSON compatibility)
        cache_data_list = df.reset_index().values.tolist()
        api_cache[cache_key] = (cache_data_list, now)
        logger.info(f"Successfully fetched and cached data for {symbol}")
        save_cache_to_disk()
        
        # Save to DB
        save_crypto_data_to_db(symbol, df)
        
        return df.copy()
    except Exception as e:
        log_api_fetch(endpoint, symbol, success=False, error_message=str(e))
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        
        # Use DB data if available, even if old
        if df_db is not None: 
            logger.warning(f"Using stale DB data for {symbol} due to API error.")
            return df_db.copy()
        
        logger.warning(f"Returning dummy data for {symbol} due to API error")
        dates = pd.date_range(end=now, periods=days)
        df_dummy = pd.DataFrame(index=dates)
        df_dummy["price"] = [50000 - i * 100 + i * i * 5 for i in range(days)]
        df_dummy["volume"] = [2000000000 - i * 10000000 + i * i * 500000 for i in range(days)]
        return df_dummy

def get_trending_coins():
    endpoint = "trending"
    cache_key = "trending_coins"
    now = datetime.datetime.utcnow()

    # 1. Check Cache
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        if now - timestamp < CACHE_DURATION:
            logger.info("Returning cached trending coins data")
            return list(cached_data)
        else:
            logger.info("Cache expired for trending coins")

    # 2. Check if API call is allowed
    last_fetch = get_last_fetch_time(endpoint)
    if last_fetch and (now - last_fetch < API_REFRESH_INTERVAL):
        logger.warning(f"Skipping API call for trending coins. Last fetch was at {last_fetch}. Using dummy data.")
        # Fallback to dummy data
        return [
            {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC", "market_cap_rank": 1, "price_btc": 1.0, "score": 100, "thumb": ""},
            {"id": "ethereum", "name": "Ethereum", "symbol": "ETH", "market_cap_rank": 2, "price_btc": 0.05, "score": 99, "thumb": ""},
            {"id": "solana", "name": "Solana", "symbol": "SOL", "market_cap_rank": 5, "price_btc": 0.00123, "score": 98, "thumb": ""}
        ]

    # 3. Fetch from API
    logger.info("Fetching trending coins from API")
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        data = api_request_with_backoff(url)
        log_api_fetch(endpoint, success=True)
        
        trending = []
        if data and "coins" in data:
            for coin_data in data["coins"][:5]:
                item = coin_data.get("item", {})
                if not item: continue
                trending.append({
                    "id": item.get("id", "N/A"), "name": item.get("name", "N/A"),
                    "symbol": item.get("symbol", "N/A"), "market_cap_rank": item.get("market_cap_rank", "N/A"),
                    "price_btc": item.get("price_btc", 0), "score": item.get("score", 0),
                    "thumb": item.get("thumb", "")
                })
            api_cache[cache_key] = (list(trending), now)
            logger.info(f"Successfully fetched and cached {len(trending)} trending coins")
            save_cache_to_disk()
            return trending
        else:
            logger.warning("Incomplete data received from CoinGecko for trending coins")
            raise ValueError("Incomplete data from API")
    except Exception as e:
        log_api_fetch(endpoint, success=False, error_message=str(e))
        logger.error(f"Error fetching trending coins: {e}", exc_info=True)
        
        # Use expired cache if available
        if cache_key in api_cache:
            cached_data, timestamp = api_cache[cache_key]
            age = now - timestamp
            if age < datetime.timedelta(days=7):
                logger.warning(f"Using expired cached trending coins data (age: {age.total_seconds()/3600:.1f} hours)")
                return list(cached_data)
        
        logger.warning("Returning dummy trending coins data due to API error")
        return [
            {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC", "market_cap_rank": 1, "price_btc": 1.0, "score": 100, "thumb": ""},
            {"id": "ethereum", "name": "Ethereum", "symbol": "ETH", "market_cap_rank": 2, "price_btc": 0.05, "score": 99, "thumb": ""},
            {"id": "solana", "name": "Solana", "symbol": "SOL", "market_cap_rank": 5, "price_btc": 0.00123, "score": 98, "thumb": ""}
        ]

def get_economic_indicators(): # Dummy data, no API calls needed
    logger.info("Fetching economic indicators (using dummy data)")
    return {
        "usd_index": 104.2, "usd_trend": "rising",
        "inflation_rate": 3.5, "inflation_trend": "stable",
        "interest_rate": 5.25, "interest_trend": "stable",
        "global_liquidity": "moderate", "liquidity_trend": "tightening",
        "stock_market": "bullish", "bond_market": "bearish",
        "gold_price": 2342.50, "gold_trend": "rising"
    }

def calculate_technical_indicators(df):
    # ... (keep existing implementation, ensure it handles potential NaNs gracefully) ...
    logger.info("Calculating technical indicators")
    if df is None or df.empty: return pd.DataFrame()
    try:
        delta = df["price"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, 1e-6) 
        rs = avg_gain / avg_loss
        df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
        df["ema12"] = df["price"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["price"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["signal_line"]
        df["sma20"] = df["price"].rolling(window=20, min_periods=1).mean()
        df["std20"] = df["price"].rolling(window=20, min_periods=1).std().fillna(0)
        df["upper_band"] = df["sma20"] + (df["std20"] * 2)
        df["lower_band"] = df["sma20"] - (df["std20"] * 2)
        df["sma50"] = df["price"].rolling(window=50, min_periods=1).mean()
        df["sma200"] = df["price"].rolling(window=200, min_periods=1).mean()
        logger.info("Technical indicators calculated successfully")
        return df.fillna(0)
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        return df # Return original df if calculation fails

def generate_chart(df, symbol, indicator_type):
    # ... (keep existing implementation) ...
    logger.info(f"Generating chart for {symbol} - {indicator_type}")
    if df is None or df.empty or indicator_type not in ["price", "rsi", "macd", "volume"]:
        logger.warning(f"Cannot generate chart for {symbol} - {indicator_type}. Invalid input.")
        return None
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        if indicator_type == "price":
            ax.plot(df.index, df.get("price"), label="Price")
            ax.plot(df.index, df.get("sma20"), label="SMA 20", linestyle="--")
            ax.plot(df.index, df.get("upper_band"), label="Upper Band", linestyle=":", color="grey")
            ax.plot(df.index, df.get("lower_band"), label="Lower Band", linestyle=":", color="grey")
            ax.set_title(f"{symbol.upper()} Price & Bollinger Bands")
            ax.set_ylabel("Price (USD)")
        elif indicator_type == "rsi":
            ax.plot(df.index, df.get("rsi"), label="RSI")
            ax.axhline(y=70, color="r", linestyle="--", alpha=0.5)
            ax.axhline(y=30, color="g", linestyle="--", alpha=0.5)
            ax.set_title(f"{symbol.upper()} RSI")
            ax.set_ylabel("RSI"); ax.set_ylim(0, 100)
        elif indicator_type == "macd":
            ax.plot(df.index, df.get("macd"), label="MACD")
            ax.plot(df.index, df.get("signal_line"), label="Signal Line", linestyle="--")
            hist_values = pd.to_numeric(df.get("macd_histogram"), errors="coerce").fillna(0)
            ax.bar(df.index, hist_values, label="Histogram", alpha=0.3, width=0.8)
            ax.set_title(f"{symbol.upper()} MACD")
            ax.set_ylabel("MACD Value")
        elif indicator_type == "volume":
            ax.bar(df.index, df.get("volume"), label="Volume", alpha=0.6)
            ax.set_title(f"{symbol.upper()} Volume")
            ax.set_ylabel("Volume")
        ax.set_xlabel("Date"); ax.legend(); ax.grid(True)
        plt.xticks(rotation=45); plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png"); buffer.seek(0)
        plt.close(fig)
        img_str = base64.b64encode(buffer.read()).decode("utf-8")
        logger.info(f"Chart generated successfully for {symbol} - {indicator_type}")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error generating chart for {symbol} - {indicator_type}: {e}", exc_info=True)
        return None

def get_asset_rotation_strategy():
    # ... (keep existing implementation, relies on get_crypto_data which is now optimized) ...
    logger.info("Generating asset rotation strategy")
    try:
        btc_data = get_crypto_data("bitcoin", days=30)
        eth_data = get_crypto_data("ethereum", days=30)
        sol_data = get_crypto_data("solana", days=30)
        btc_perf = (btc_data["price"].iloc[-1] / btc_data["price"].iloc[0] - 1) * 100 if not btc_data.empty and len(btc_data["price"]) > 1 else 0
        eth_perf = (eth_data["price"].iloc[-1] / eth_data["price"].iloc[0] - 1) * 100 if not eth_data.empty and len(eth_data["price"]) > 1 else 0
        sol_perf = (sol_data["price"].iloc[-1] / sol_data["price"].iloc[0] - 1) * 100 if not sol_data.empty and len(sol_data["price"]) > 1 else 0
        trending = get_trending_coins()
        econ = get_economic_indicators()
        allocations = []
        btc_alloc, eth_alloc, sol_alloc = 25, 20, 15 
        btc_signal_val, eth_signal_val, sol_signal_val = "HOLD", "HOLD", "HOLD"
        if btc_perf > 10 and econ.get("usd_trend") == "falling": btc_alloc = 40; btc_signal_val = "BUY"
        elif btc_perf < -10 and econ.get("usd_trend") == "rising": btc_alloc = 10; btc_signal_val = "SELL"
        allocations.append({"coin": "Bitcoin (BTC)", "allocation": btc_alloc, "signal": btc_signal_val, "performance": f"{btc_perf:.2f}%"})
        if eth_perf > 15: eth_alloc = 30; eth_signal_val = "BUY"
        elif eth_perf < -15: eth_alloc = 10; eth_signal_val = "SELL"
        allocations.append({"coin": "Ethereum (ETH)", "allocation": eth_alloc, "signal": eth_signal_val, "performance": f"{eth_perf:.2f}%"})
        if sol_perf > 20: sol_alloc = 20; sol_signal_val = "BUY"
        elif sol_perf < -20: sol_alloc = 5; sol_signal_val = "SELL"
        allocations.append({"coin": "Solana (SOL)", "allocation": sol_alloc, "signal": sol_signal_val, "performance": f"{sol_perf:.2f}%"})
        remaining = 100 - (btc_alloc + eth_alloc + sol_alloc)
        if trending and remaining > 0:
             per_trending = remaining / len(trending)
             for coin_item in trending:
                 allocations.append({"coin": f"{coin_item.get(\"name\", \"N/A\")} ({coin_item.get(\"symbol\", \"N/A\")})", "allocation": round(per_trending), "signal": "BUY" if coin_item.get("score", 0) > 95 else "RESEARCH", "performance": "Trending"})
        elif remaining > 0:
             allocations.append({"coin": "Cash/Stablecoin", "allocation": remaining, "signal": "HOLD", "performance": "N/A"})
        logger.info("Asset rotation strategy generated successfully")
        return allocations
    except Exception as e:
        logger.error(f"Error generating asset rotation strategy: {e}", exc_info=True)
        return []

def generate_buy_sell_recommendations():
    # ... (keep existing implementation, relies on get_crypto_data which is now optimized) ...
    logger.info("Generating buy/sell recommendations")
    try:
        btc_data = calculate_technical_indicators(get_crypto_data("bitcoin", days=90))
        eth_data = calculate_technical_indicators(get_crypto_data("ethereum", days=90))
        sol_data = calculate_technical_indicators(get_crypto_data("solana", days=90))
        recommendations = []
        
        # Bitcoin analysis
        btc_rec = {"coin": "Bitcoin (BTC)", "price": f"${btc_data[\"price\"].iloc[-1]:,.2f}" if not btc_data.empty else "N/A", "signals": []}
        if not btc_data.empty:
            if btc_data["rsi"].iloc[-1] > 70: btc_rec["signals"].append({"indicator": "RSI", "signal": "SELL", "value": f"{btc_data[\"rsi\"].iloc[-1]:.2f}"})
            elif btc_data["rsi"].iloc[-1] < 30: btc_rec["signals"].append({"indicator": "RSI", "signal": "BUY", "value": f"{btc_data[\"rsi\"].iloc[-1]:.2f}"})
            else: btc_rec["signals"].append({"indicator": "RSI", "signal": "NEUTRAL", "value": f"{btc_data[\"rsi\"].iloc[-1]:.2f}"})
            if btc_data["macd"].iloc[-1] > btc_data["signal_line"].iloc[-1]: btc_rec["signals"].append({"indicator": "MACD", "signal": "BUY", "value": f"{btc_data[\"macd\"].iloc[-1]:.2f}"})
            else: btc_rec["signals"].append({"indicator": "MACD", "signal": "SELL", "value": f"{btc_data[\"macd\"].iloc[-1]:.2f}"})
            if btc_data["price"].iloc[-1] > btc_data["upper_band"].iloc[-1]: btc_rec["signals"].append({"indicator": "Bollinger", "signal": "SELL", "value": "Upper band crossed"})
            elif btc_data["price"].iloc[-1] < btc_data["lower_band"].iloc[-1]: btc_rec["signals"].append({"indicator": "Bollinger", "signal": "BUY", "value": "Lower band crossed"})
            else: btc_rec["signals"].append({"indicator": "Bollinger", "signal": "NEUTRAL", "value": "Within bands"})
            if btc_data["sma50"].iloc[-1] > btc_data["sma200"].iloc[-1]: btc_rec["signals"].append({"indicator": "Golden Cross", "signal": "BUY", "value": "SMA50 > SMA200"})
            else: btc_rec["signals"].append({"indicator": "Death Cross", "signal": "SELL", "value": "SMA50 < SMA200"})
        buy_signals = sum(1 for s in btc_rec["signals"] if s["signal"] == "BUY")
        sell_signals = sum(1 for s in btc_rec["signals"] if s["signal"] == "SELL")
        if buy_signals > sell_signals: btc_rec["overall"] = "BUY"
        elif sell_signals > buy_signals: btc_rec["overall"] = "SELL"
        else: btc_rec["overall"] = "HOLD"
        recommendations.append(btc_rec)
        
        # Ethereum analysis
        eth_rec = {"coin": "Ethereum (ETH)", "price": f"${eth_data[\"price\"].iloc[-1]:,.2f}" if not eth_data.empty else "N/A", "signals": []}
        if not eth_data.empty:
            if eth_data["rsi"].iloc[-1] > 70: eth_rec["signals"].append({"indicator": "RSI", "signal": "SELL", "value": f"{eth_data[\"rsi\"].iloc[-1]:.2f}"})
            elif eth_data["rsi"].iloc[-1] < 30: eth_rec["signals"].append({"indicator": "RSI", "signal": "BUY", "value": f"{eth_data[\"rsi\"].iloc[-1]:.2f}"})
            else: eth_rec["signals"].append({"indicator": "RSI", "signal": "NEUTRAL", "value": f"{eth_data[\"rsi\"].iloc[-1]:.2f}"})
            if eth_data["macd"].iloc[-1] > eth_data["signal_line"].iloc[-1]: eth_rec["signals"].append({"indicator": "MACD", "signal": "BUY", "value": f"{eth_data[\"macd\"].iloc[-1]:.2f}"})
            else: eth_rec["signals"].append({"indicator": "MACD", "signal": "SELL", "value": f"{eth_data[\"macd\"].iloc[-1]:.2f}"})
            if eth_data["price"].iloc[-1] > eth_data["upper_band"].iloc[-1]: eth_rec["signals"].append({"indicator": "Bollinger", "signal": "SELL", "value": "Upper band crossed"})
            elif eth_data["price"].iloc[-1] < eth_data["lower_band"].iloc[-1]: eth_rec["signals"].append({"indicator": "Bollinger", "signal": "BUY", "value": "Lower band crossed"})
            else: eth_rec["signals"].append({"indicator": "Bollinger", "signal": "NEUTRAL", "value": "Within bands"})
            if eth_data["sma50"].iloc[-1] > eth_data["sma200"].iloc[-1]: eth_rec["signals"].append({"indicator": "Golden Cross", "signal": "BUY", "value": "SMA50 > SMA200"})
            else: eth_rec["signals"].append({"indicator": "Death Cross", "signal": "SELL", "value": "SMA50 < SMA200"})
        buy_signals = sum(1 for s in eth_rec["signals"] if s["signal"] == "BUY")
        sell_signals = sum(1 for s in eth_rec["signals"] if s["signal"] == "SELL")
        if buy_signals > sell_signals: eth_rec["overall"] = "BUY"
        elif sell_signals > buy_signals: eth_rec["overall"] = "SELL"
        else: eth_rec["overall"] = "HOLD"
        recommendations.append(eth_rec)
        
        # Solana analysis
        sol_rec = {"coin": "Solana (SOL)", "price": f"${sol_data[\"price\"].iloc[-1]:,.2f}" if not sol_data.empty else "N/A", "signals": []}
        if not sol_data.empty:
            if sol_data["rsi"].iloc[-1] > 70: sol_rec["signals"].append({"indicator": "RSI", "signal": "SELL", "value": f"{sol_data[\"rsi\"].iloc[-1]:.2f}"})
            elif sol_data["rsi"].iloc[-1] < 30: sol_rec["signals"].append({"indicator": "RSI", "signal": "BUY", "value": f"{sol_data[\"rsi\"].iloc[-1]:.2f}"})
            else: sol_rec["signals"].append({"indicator": "RSI", "signal": "NEUTRAL", "value": f"{sol_data[\"rsi\"].iloc[-1]:.2f}"})
            if sol_data["macd"].iloc[-1] > sol_data["signal_line"].iloc[-1]: sol_rec["signals"].append({"indicator": "MACD", "signal": "BUY", "value": f"{sol_data[\"macd\"].iloc[-1]:.2f}"})
            else: sol_rec["signals"].append({"indicator": "MACD", "signal": "SELL", "value": f"{sol_data[\"macd\"].iloc[-1]:.2f}"})
            if sol_data["price"].iloc[-1] > sol_data["upper_band"].iloc[-1]: sol_rec["signals"].append({"indicator": "Bollinger", "signal": "SELL", "value": "Upper band crossed"})
            elif sol_data["price"].iloc[-1] < sol_data["lower_band"].iloc[-1]: sol_rec["signals"].append({"indicator": "Bollinger", "signal": "BUY", "value": "Lower band crossed"})
            else: sol_rec["signals"].append({"indicator": "Bollinger", "signal": "NEUTRAL", "value": "Within bands"})
            if sol_data["sma50"].iloc[-1] > sol_data["sma200"].iloc[-1]: sol_rec["signals"].append({"indicator": "Golden Cross", "signal": "BUY", "value": "SMA50 > SMA200"})
            else: sol_rec["signals"].append({"indicator": "Death Cross", "signal": "SELL", "value": "SMA50 < SMA200"})
        buy_signals = sum(1 for s in sol_rec["signals"] if s["signal"] == "BUY")
        sell_signals = sum(1 for s in sol_rec["signals"] if s["signal"] == "SELL")
        if buy_signals > sell_signals: sol_rec["overall"] = "BUY"
        elif sell_signals > buy_signals: sol_rec["overall"] = "SELL"
        else: sol_rec["overall"] = "HOLD"
        recommendations.append(sol_rec)
        
        logger.info("Buy/sell recommendations generated successfully")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating buy/sell recommendations: {e}", exc_info=True)
        return []

# --- Email Notification System (from email_notification_system.py) ---
# ... (integrate functions like render_email_template, send_email, generate_daily_update etc. here) ...
# Placeholder - assumes functions from email_notification_system.py are available
def send_email_alert(user_email, subject, content, charts=None):
    logger.info(f"Placeholder: Email would be sent to {user_email} with subject: {subject}")
    return True
# --- End Email System Placeholder ---

# --- Routes ---
@app.route("/")
@app.route("/home")
def home():
    try:
        trending_coins = get_trending_coins()
        btc_data = calculate_technical_indicators(get_crypto_data("bitcoin", days=30))
        eth_data = calculate_technical_indicators(get_crypto_data("ethereum", days=30))
        sol_data = calculate_technical_indicators(get_crypto_data("solana", days=30))
        
        btc_price_chart = generate_chart(btc_data, "bitcoin", "price")
        eth_price_chart = generate_chart(eth_data, "ethereum", "price")
        sol_price_chart = generate_chart(sol_data, "solana", "price")
        btc_rsi_chart = generate_chart(btc_data, "bitcoin", "rsi")
        eth_rsi_chart = generate_chart(eth_data, "ethereum", "rsi")
        sol_rsi_chart = generate_chart(sol_data, "solana", "rsi")
        
        economic_indicators = get_economic_indicators()
        
        # Get last fetch times for display
        last_fetch_times = {
            "bitcoin": get_last_fetch_time("market_chart", "bitcoin"),
            "ethereum": get_last_fetch_time("market_chart", "ethereum"),
            "solana": get_last_fetch_time("market_chart", "solana"),
            "trending": get_last_fetch_time("trending")
        }
        
        return render_template("home.html", 
                              trending_coins=trending_coins,
                              btc_price_chart=btc_price_chart,
                              eth_price_chart=eth_price_chart,
                              sol_price_chart=sol_price_chart,
                              btc_rsi_chart=btc_rsi_chart,
                              eth_rsi_chart=eth_rsi_chart,
                              sol_rsi_chart=sol_rsi_chart,
                              economic_indicators=economic_indicators,
                              last_fetch_times=last_fetch_times,
                              refresh_interval_hours=API_REFRESH_INTERVAL.total_seconds()/3600)
    except Exception as e:
        logger.error(f"Error in home route: {e}", exc_info=True)
        return render_template("500.html", error_message=str(e)), 500

@app.route("/strategy")
def strategy():
    try:
        asset_rotation = get_asset_rotation_strategy()
        recommendations = generate_buy_sell_recommendations()
        
        btc_data = calculate_technical_indicators(get_crypto_data("bitcoin", days=90))
        eth_data = calculate_technical_indicators(get_crypto_data("ethereum", days=90))
        sol_data = calculate_technical_indicators(get_crypto_data("solana", days=90))
        
        btc_macd_chart = generate_chart(btc_data, "bitcoin", "macd")
        eth_macd_chart = generate_chart(eth_data, "ethereum", "macd")
        sol_macd_chart = generate_chart(sol_data, "solana", "macd")
        
        return render_template("strategy.html",
                              asset_rotation=asset_rotation,
                              recommendations=recommendations,
                              btc_macd_chart=btc_macd_chart,
                              eth_macd_chart=eth_macd_chart,
                              sol_macd_chart=sol_macd_chart)
    except Exception as e:
        logger.error(f"Error in strategy route: {e}", exc_info=True)
        return render_template("500.html", error_message=str(e)), 500

@app.route("/register", methods=["GET", "POST"])
def register():
    # ... (keep existing implementation) ...
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        try:
            username = request.form.get("username")
            email = request.form.get("email")
            password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")
            if not all([username, email, password, confirm_password]):
                flash("All fields are required", "danger")
                return render_template("register.html")
            if password != confirm_password:
                flash("Passwords do not match", "danger")
                return render_template("register.html")
            if User.query.filter_by(username=username).first():
                flash("Username already exists", "danger")
                return render_template("register.html")
            if User.query.filter_by(email=email).first():
                flash("Email already registered", "danger")
                return render_template("register.html")
            hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            user_pref = UserPreference(user_id=user.id)
            db.session.add(user_pref)
            email_pref = EmailPreference(user_id=user.id) # Create email prefs too
            db.session.add(email_pref)
            db.session.commit()
            flash("Your account has been created! You can now log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in register route: {e}", exc_info=True)
            flash("An error occurred during registration. Please try again.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    # ... (keep existing implementation) ...
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        try:
            email = request.form.get("email")
            password = request.form.get("password")
            remember = "remember" in request.form
            user = User.query.filter_by(email=email).first()
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user, remember=remember)
                next_page = request.args.get("next")
                flash("Login successful!", "success")
                return redirect(next_page) if next_page else redirect(url_for("home"))
            else:
                flash("Login unsuccessful. Please check email and password.", "danger")
        except Exception as e:
            logger.error(f"Error in login route: {e}", exc_info=True)
            flash("An error occurred during login. Please try again.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    # ... (keep existing implementation) ...
    logout_user()
    return redirect(url_for("home"))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    # ... (keep existing implementation, maybe merge email prefs here later) ...
    try:
        user_pref = current_user.preferences or UserPreference(user_id=current_user.id)
        if request.method == "POST":
            user_pref.email_frequency = request.form.get("email_frequency", "daily")
            user_pref.preferred_coins = request.form.get("preferred_coins", "BTC,ETH,SOL")
            user_pref.preferred_indicators = request.form.get("preferred_indicators", "price,volume,rsi,macd")
            user_pref.theme = request.form.get("theme", "light")
            if not current_user.preferences:
                db.session.add(user_pref)
            db.session.commit()
            flash("Your preferences have been updated!", "success")
        return render_template("profile.html", user=current_user, preferences=user_pref)
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in profile route: {e}", exc_info=True)
        flash("An error occurred while updating your profile. Please try again.", "danger")
        return render_template("profile.html", user=current_user, preferences=user_pref)

@app.route("/profile/email-preferences", methods=["GET", "POST"])
@login_required
def email_preferences():
    # ... (integrate code from email_notification_system.py) ...
    try:
        prefs = current_user.email_preferences
        if not prefs:
            prefs = EmailPreference(user_id=current_user.id)
            db.session.add(prefs)
        if request.method == "POST":
            prefs.receive_daily = "receive_daily" in request.form
            prefs.receive_weekly = "receive_weekly" in request.form
            prefs.receive_monthly = "receive_monthly" in request.form
            prefs.include_market_summary = "include_market_summary" in request.form
            prefs.include_portfolio = "include_portfolio" in request.form
            prefs.include_recommendations = "include_recommendations" in request.form
            prefs.include_trending = "include_trending" in request.form
            prefs.price_alert_threshold = float(request.form.get("price_alert_threshold", 5.0))
            prefs.rsi_overbought = float(request.form.get("rsi_overbought", 70.0))
            prefs.rsi_oversold = float(request.form.get("rsi_oversold", 30.0))
            prefs.preferred_coins = request.form.get("preferred_coins", "BTC,ETH,SOL")
            prefs.html_format = "html_format" in request.form
            db.session.commit()
            flash("Email preferences updated successfully!", "success")
        return render_template("email_preferences.html", preferences=prefs)
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in email_preferences route: {e}", exc_info=True)
        flash("An error occurred while updating your email preferences. Please try again.", "danger")
        # Ensure prefs is defined for rendering
        prefs = current_user.email_preferences or EmailPreference(user_id=current_user.id)
        return render_template("email_preferences.html", preferences=prefs)

@app.route("/send-test-email/<email_type>")
@login_required
def send_test_email(email_type):
    # ... (integrate code from email_notification_system.py) ...
    # Note: This requires the email sending functions to be defined in this file
    flash("Test email functionality placeholder.", "info") # Placeholder
    return redirect(url_for("email_preferences"))

@app.route("/admin-dashboard")
@login_required
def admin_dashboard():
    # ... (keep existing implementation) ...
    if not current_user.is_admin:
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for("home"))
    try:
        users = User.query.all()
        return render_template("admin_dashboard.html", users=users)
    except Exception as e:
        logger.error(f"Error in admin_dashboard route: {e}", exc_info=True)
        return render_template("500.html", error_message=str(e)), 500

@app.route("/initialize-db")
def initialize_db_route():
    key = request.args.get("key")
    expected_key = get_env_var("DB_INIT_KEY", "default_secret_key_for_init") 
    if key != expected_key:
        logger.warning(f"Unauthorized attempt to initialize DB with key: {key}. Expected key starts with: {expected_key[:5]}...")
        return "Unauthorized", 403
    try:
        with app.app_context():
            logger.info("Attempting to drop and create all database tables.")
            db.drop_all() # Drops all tables defined in models
            db.create_all() # Creates all tables defined in models
            logger.info("Database tables dropped and created.")
            if not User.query.filter_by(username="admin").first():
                logger.info("Admin user not found, creating new admin user.")
                hashed_password = bcrypt.generate_password_hash("defaultadmin").decode("utf-8")
                admin_user = User(username="admin", email="admin@example.com", password=hashed_password, is_admin=True)
                db.session.add(admin_user)
                db.session.commit()
                admin_prefs = UserPreference(user_id=admin_user.id)
                db.session.add(admin_prefs)
                admin_email_prefs = EmailPreference(user_id=admin_user.id)
                db.session.add(admin_email_prefs)
                db.session.commit()
                logger.info("Admin user and preferences created.")
            else:
                logger.info("Admin user already exists.")
        logger.info("Database initialized successfully.")
        return "Database initialized!"
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error initializing database: {e}", exc_info=True)
        try:
            return render_template("500.html", error_message=str(e)), 500
        except:
            return f"Error initializing database: {e}. Additionally, 500.html template might be missing or broken.", 500

@app.route("/disclaimer")
def disclaimer_page():
    return render_template("disclaimer.html")

@app.route("/status")
def status_page():
    # ... (keep existing implementation) ...
    try:
        now = datetime.datetime.utcnow()
        cooling_period_remaining = None
        if last_rate_limit_hit is not None:
            time_since_limit = now - last_rate_limit_hit
            if time_since_limit < RATE_LIMIT_RESET:
                cooling_period_remaining = (RATE_LIMIT_RESET - time_since_limit).total_seconds()
        one_minute_ago = now - datetime.timedelta(minutes=1)
        current_api_calls = [ts for ts in api_call_timestamps if ts > one_minute_ago]
        cache_stats = {
            "total_items": len(api_cache),
            "cache_size_kb": sum(len(str(data)) for data, _ in api_cache.values()) / 1024 if api_cache else 0,
            "oldest_item_age_hours": max((now - timestamp).total_seconds() / 3600 for _, timestamp in api_cache.values()) if api_cache else 0,
            "newest_item_age_hours": min((now - timestamp).total_seconds() / 3600 for _, timestamp in api_cache.values()) if api_cache else 0
        }
        # Get DB stats
        db_stats = {
            "total_users": User.query.count(),
            "historical_prices": HistoricalPrice.query.count(),
            "api_logs": ApiFetchLog.query.count()
        }
        return render_template("status.html",
                              rate_limited=last_rate_limit_hit is not None and cooling_period_remaining is not None,
                              cooling_period_remaining=cooling_period_remaining,
                              api_calls_last_minute=len(current_api_calls),
                              max_calls_per_minute=MAX_CALLS_PER_MINUTE,
                              cache_stats=cache_stats,
                              db_stats=db_stats)
    except Exception as e:
        logger.error(f"Error in status page: {e}", exc_info=True)
        return render_template("500.html", error_message=str(e)), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    # Log the actual error
    logger.error(f"Internal Server Error: {e}", exc_info=True)
    # Pass a generic message to the template
    return render_template("500.html", error_message="An internal server error occurred."), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Ensure database is created on first run if needed (useful for local dev)
    with app.app_context():
        try:
            db.create_all()
            logger.info("Ensured database tables exist.")
        except Exception as e:
            logger.error(f"Error ensuring database tables exist: {e}")
    app.run(host="0.0.0.0", port=port, debug=False)

