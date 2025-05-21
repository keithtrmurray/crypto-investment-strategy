#!/usr/bin/env python3
"""
Web-based Crypto Investment Strategy service - Stable Core Version with Deferred Processing
"""

import os
import sys
import json
import datetime
import logging
import time
import random
import threading
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

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

# --- Global Variables ---
# These will be initialized after app startup
api_cache = {}
initialization_status = {
    "started": False,
    "completed": False,
    "progress": 0,
    "message": "Initialization not started",
    "last_updated": datetime.datetime.utcnow()
}
# --- End Global Variables ---

try:
    db = SQLAlchemy(app)
    bcrypt = Bcrypt(app)
    login_manager = LoginManager(app)
    login_manager.login_view = "login"
    logger.info("Flask extensions initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Flask extensions: {str(e)}", exc_info=True)
    sys.exit("Failed to initialize Flask extensions.")

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
    email_preferences = db.relationship("EmailPreference", backref="user_email_prefs", uselist=False)

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

# --- Deferred Initialization ---
def start_background_initialization():
    """Start background initialization process"""
    global initialization_status
    
    if initialization_status["started"]:
        logger.info("Background initialization already started")
        return
    
    initialization_status["started"] = True
    initialization_status["message"] = "Initialization started"
    initialization_status["last_updated"] = datetime.datetime.utcnow()
    
    # Start initialization in a background thread
    thread = threading.Thread(target=background_initialization_process)
    thread.daemon = True
    thread.start()
    logger.info("Background initialization thread started")

def background_initialization_process():
    """Background process to initialize data"""
    global initialization_status, api_cache
    
    try:
        logger.info("Starting background initialization process")
        
        # Step 1: Create data directories
        initialization_status["progress"] = 5
        initialization_status["message"] = "Creating data directories"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        os.makedirs("data", exist_ok=True)
        os.makedirs("static/img", exist_ok=True)
        time.sleep(1)  # Simulate work
        
        # Step 2: Load cache if exists
        initialization_status["progress"] = 10
        initialization_status["message"] = "Loading cache"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        load_cache_from_disk()
        time.sleep(1)  # Simulate work
        
        # Step 3: Initialize Bitcoin data
        initialization_status["progress"] = 20
        initialization_status["message"] = "Fetching Bitcoin data"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        fetch_crypto_data("bitcoin", days=30)
        time.sleep(2)  # Simulate work
        
        # Step 4: Initialize Ethereum data
        initialization_status["progress"] = 40
        initialization_status["message"] = "Fetching Ethereum data"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        fetch_crypto_data("ethereum", days=30)
        time.sleep(2)  # Simulate work
        
        # Step 5: Initialize Solana data
        initialization_status["progress"] = 60
        initialization_status["message"] = "Fetching Solana data"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        fetch_crypto_data("solana", days=30)
        time.sleep(2)  # Simulate work
        
        # Step 6: Initialize trending coins
        initialization_status["progress"] = 80
        initialization_status["message"] = "Fetching trending coins"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        fetch_trending_coins()
        time.sleep(2)  # Simulate work
        
        # Step 7: Save cache to disk
        initialization_status["progress"] = 90
        initialization_status["message"] = "Saving cache to disk"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        save_cache_to_disk()
        time.sleep(1)  # Simulate work
        
        # Initialization complete
        initialization_status["progress"] = 100
        initialization_status["completed"] = True
        initialization_status["message"] = "Initialization completed"
        initialization_status["last_updated"] = datetime.datetime.utcnow()
        logger.info("Background initialization process completed")
        
    except Exception as e:
        logger.error(f"Error in background initialization: {e}", exc_info=True)
        initialization_status["message"] = f"Initialization error: {str(e)}"
        initialization_status["last_updated"] = datetime.datetime.utcnow()

# --- Cache Management ---
def load_cache_from_disk():
    global api_cache
    try:
        cache_file = "data/api_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
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
        cache_file = "data/api_cache.json"
        serializable_cache = {}
        for key, (data, timestamp) in api_cache.items():
            key_str = json.dumps(key) if isinstance(key, tuple) else key
            serializable_cache[key_str] = (data, timestamp.isoformat())
        with open(cache_file, "w") as f:
            json.dump(serializable_cache, f)
        logger.info(f"Saved {len(api_cache)} items to cache file")
    except Exception as e:
        logger.error(f"Error saving cache to disk: {e}", exc_info=True)

# --- Data Fetching (Deferred) ---
def fetch_crypto_data(symbol, days=30):
    """Fetch crypto data in background thread"""
    thread = threading.Thread(target=_fetch_crypto_data, args=(symbol, days))
    thread.daemon = True
    thread.start()
    logger.info(f"Started background thread to fetch data for {symbol}")
    return None  # Return immediately, data will be fetched in background

def _fetch_crypto_data(symbol, days=30):
    """Actual implementation of data fetching"""
    global api_cache
    cache_key = (symbol, days)
    now = datetime.datetime.utcnow()
    
    try:
        # Check if we already have this data cached
        if cache_key in api_cache:
            cached_data, timestamp = api_cache[cache_key]
            age = now - timestamp
            if age < datetime.timedelta(hours=12):  # 12-hour cache
                logger.info(f"Using cached data for {symbol} (age: {age.total_seconds()/3600:.1f} hours)")
                return
        
        # Fetch from API with rate limiting
        logger.info(f"Fetching data for {symbol} from API")
        time.sleep(random.uniform(1, 3))  # Random delay to avoid rate limits
        
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            logger.warning(f"Rate limit hit for {symbol}, using dummy data")
            # Use dummy data for rate limits
            dummy_data = generate_dummy_data(symbol, days)
            api_cache[cache_key] = (dummy_data, now)
            save_cache_to_disk()
            return
        
        response.raise_for_status()
        data = response.json()
        
        if not data or "prices" not in data:
            logger.warning(f"Incomplete data for {symbol}, using dummy data")
            dummy_data = generate_dummy_data(symbol, days)
            api_cache[cache_key] = (dummy_data, now)
            save_cache_to_disk()
            return
        
        # Process and cache the data
        processed_data = process_crypto_data(data)
        api_cache[cache_key] = (processed_data, now)
        save_cache_to_disk()
        logger.info(f"Successfully fetched and cached data for {symbol}")
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
        # Use dummy data on error
        dummy_data = generate_dummy_data(symbol, days)
        api_cache[cache_key] = (dummy_data, now)
        save_cache_to_disk()

def process_crypto_data(data):
    """Process raw API data into a format suitable for caching"""
    try:
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        
        processed_data = {
            "prices": prices,
            "volumes": volumes
        }
        
        return processed_data
    except Exception as e:
        logger.error(f"Error processing crypto data: {e}", exc_info=True)
        return {}

def generate_dummy_data(symbol, days):
    """Generate dummy data for when API calls fail"""
    base_price = 0
    if symbol == "bitcoin":
        base_price = 50000
    elif symbol == "ethereum":
        base_price = 3000
    elif symbol == "solana":
        base_price = 100
    else:
        base_price = 10
    
    now = datetime.datetime.utcnow()
    prices = []
    volumes = []
    
    for i in range(days):
        timestamp = int((now - datetime.timedelta(days=days-i-1)).timestamp() * 1000)
        price = base_price * (1 + 0.1 * random.random() - 0.05)
        volume = base_price * 1000000 * (0.5 + random.random())
        prices.append([timestamp, price])
        volumes.append([timestamp, volume])
    
    return {
        "prices": prices,
        "volumes": volumes
    }

def fetch_trending_coins():
    """Fetch trending coins in background thread"""
    thread = threading.Thread(target=_fetch_trending_coins)
    thread.daemon = True
    thread.start()
    logger.info("Started background thread to fetch trending coins")
    return None  # Return immediately, data will be fetched in background

def _fetch_trending_coins():
    """Actual implementation of trending coins fetching"""
    global api_cache
    cache_key = "trending_coins"
    now = datetime.datetime.utcnow()
    
    try:
        # Check if we already have this data cached
        if cache_key in api_cache:
            cached_data, timestamp = api_cache[cache_key]
            age = now - timestamp
            if age < datetime.timedelta(hours=12):  # 12-hour cache
                logger.info(f"Using cached trending coins data (age: {age.total_seconds()/3600:.1f} hours)")
                return
        
        # Fetch from API with rate limiting
        logger.info("Fetching trending coins from API")
        time.sleep(random.uniform(1, 3))  # Random delay to avoid rate limits
        
        url = "https://api.coingecko.com/api/v3/search/trending"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 429:
            logger.warning("Rate limit hit for trending coins, using dummy data")
            # Use dummy data for rate limits
            dummy_data = generate_dummy_trending()
            api_cache[cache_key] = (dummy_data, now)
            save_cache_to_disk()
            return
        
        response.raise_for_status()
        data = response.json()
        
        if not data or "coins" not in data:
            logger.warning("Incomplete trending coins data, using dummy data")
            dummy_data = generate_dummy_trending()
            api_cache[cache_key] = (dummy_data, now)
            save_cache_to_disk()
            return
        
        # Process and cache the data
        trending = []
        for coin_data in data["coins"][:5]:
            item = coin_data.get("item", {})
            if not item: continue
            trending.append({
                "id": item.get("id", "N/A"), 
                "name": item.get("name", "N/A"),
                "symbol": item.get("symbol", "N/A"), 
                "market_cap_rank": item.get("market_cap_rank", "N/A"),
                "price_btc": item.get("price_btc", 0), 
                "score": item.get("score", 0),
                "thumb": item.get("thumb", "")
            })
        
        api_cache[cache_key] = (trending, now)
        save_cache_to_disk()
        logger.info(f"Successfully fetched and cached {len(trending)} trending coins")
        
    except Exception as e:
        logger.error(f"Error fetching trending coins: {e}", exc_info=True)
        # Use dummy data on error
        dummy_data = generate_dummy_trending()
        api_cache[cache_key] = (dummy_data, now)
        save_cache_to_disk()

def generate_dummy_trending():
    """Generate dummy trending coins data"""
    return [
        {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC", "market_cap_rank": 1, "price_btc": 1.0, "score": 100, "thumb": ""},
        {"id": "ethereum", "name": "Ethereum", "symbol": "ETH", "market_cap_rank": 2, "price_btc": 0.05, "score": 99, "thumb": ""},
        {"id": "solana", "name": "Solana", "symbol": "SOL", "market_cap_rank": 5, "price_btc": 0.00123, "score": 98, "thumb": ""},
        {"id": "cardano", "name": "Cardano", "symbol": "ADA", "market_cap_rank": 8, "price_btc": 0.00002, "score": 97, "thumb": ""},
        {"id": "polkadot", "name": "Polkadot", "symbol": "DOT", "market_cap_rank": 12, "price_btc": 0.0001, "score": 96, "thumb": ""}
    ]

def get_crypto_data(symbol, days=30):
    """Get crypto data from cache or trigger background fetch"""
    cache_key = (symbol, days)
    now = datetime.datetime.utcnow()
    
    # Check if we have data in cache
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        age = now - timestamp
        logger.info(f"Using cached data for {symbol} (age: {age.total_seconds()/3600:.1f} hours)")
        return cached_data
    
    # If not in cache, trigger background fetch and return dummy data
    fetch_crypto_data(symbol, days)
    return generate_dummy_data(symbol, days)

def get_trending_coins():
    """Get trending coins from cache or trigger background fetch"""
    cache_key = "trending_coins"
    now = datetime.datetime.utcnow()
    
    # Check if we have data in cache
    if cache_key in api_cache:
        cached_data, timestamp = api_cache[cache_key]
        age = now - timestamp
        logger.info(f"Using cached trending coins data (age: {age.total_seconds()/3600:.1f} hours)")
        return cached_data
    
    # If not in cache, trigger background fetch and return dummy data
    fetch_trending_coins()
    return generate_dummy_trending()

# --- Routes ---
@app.route("/")
@app.route("/home")
def home():
    try:
        # Start background initialization if not already started
        if not initialization_status["started"]:
            start_background_initialization()
        
        # Get data (will use cache or trigger background fetch)
        trending_coins = get_trending_coins()
        btc_data = get_crypto_data("bitcoin", days=30)
        eth_data = get_crypto_data("ethereum", days=30)
        sol_data = get_crypto_data("solana", days=30)
        
        # Prepare data for template
        initialization_progress = initialization_status["progress"]
        initialization_message = initialization_status["message"]
        initialization_complete = initialization_status["completed"]
        
        return render_template("home.html", 
                              trending_coins=trending_coins,
                              btc_data=btc_data,
                              eth_data=eth_data,
                              sol_data=sol_data,
                              initialization_progress=initialization_progress,
                              initialization_message=initialization_message,
                              initialization_complete=initialization_complete)
    except Exception as e:
        logger.error(f"Error in home route: {e}", exc_info=True)
        return render_template("500.html", error_message=str(e)), 500

@app.route("/strategy")
def strategy():
    try:
        # Start background initialization if not already started
        if not initialization_status["started"]:
            start_background_initialization()
        
        # Get data (will use cache or trigger background fetch)
        btc_data = get_crypto_data("bitcoin", days=90)
        eth_data = get_crypto_data("ethereum", days=90)
        sol_data = get_crypto_data("solana", days=90)
        
        # Prepare data for template
        initialization_progress = initialization_status["progress"]
        initialization_message = initialization_status["message"]
        initialization_complete = initialization_status["completed"]
        
        return render_template("strategy.html",
                              btc_data=btc_data,
                              eth_data=eth_data,
                              sol_data=sol_data,
                              initialization_progress=initialization_progress,
                              initialization_message=initialization_message,
                              initialization_complete=initialization_complete)
    except Exception as e:
        logger.error(f"Error in strategy route: {e}", exc_info=True)
        return render_template("500.html", error_message=str(e)), 500

@app.route("/register", methods=["GET", "POST"])
def register():
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
            email_pref = EmailPreference(user_id=user.id)
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
    logout_user()
    return redirect(url_for("home"))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
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

@app.route("/disclaimer")
def disclaimer_page():
    return render_template("disclaimer.html")

@app.route("/migration-guide")
def migration_guide():
    return render_template("migration_guide.html")

@app.route("/status")
def status_page():
    try:
        # Get initialization status
        init_status = {
            "started": initialization_status["started"],
            "completed": initialization_status["completed"],
            "progress": initialization_status["progress"],
            "message": initialization_status["message"],
            "last_updated": initialization_status["last_updated"]
        }
        
        # Get cache stats
        cache_stats = {
            "total_items": len(api_cache),
            "bitcoin_cached": (("bitcoin", 30) in api_cache),
            "ethereum_cached": (("ethereum", 30) in api_cache),
            "solana_cached": (("solana", 30) in api_cache),
            "trending_cached": ("trending_coins" in api_cache)
        }
        
        # Get DB stats
        db_stats = {
            "total_users": User.query.count(),
            "historical_prices": HistoricalPrice.query.count(),
            "api_logs": ApiFetchLog.query.count()
        }
        
        return render_template("status.html",
                              initialization_status=init_status,
                              cache_stats=cache_stats,
                              db_stats=db_stats)
    except Exception as e:
        logger.error(f"Error in status page: {e}", exc_info=True)
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
            db.drop_all()
            db.create_all()
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

@app.route("/api/initialization-status")
def api_initialization_status():
    """API endpoint to get initialization status"""
    return jsonify({
        "started": initialization_status["started"],
        "completed": initialization_status["completed"],
        "progress": initialization_status["progress"],
        "message": initialization_status["message"],
        "last_updated": initialization_status["last_updated"].isoformat()
    })

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal Server Error: {e}", exc_info=True)
    return render_template("500.html", error_message="An internal server error occurred."), 500

@app.before_first_request
def before_first_request():
    """Initialize app before first request"""
    try:
        # Ensure database tables exist
        with app.app_context():
            db.create_all()
            logger.info("Ensured database tables exist.")
        
        # Create data directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("static/img", exist_ok=True)
        
        logger.info("App initialized before first request")
    except Exception as e:
        logger.error(f"Error in before_first_request: {e}", exc_info=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
