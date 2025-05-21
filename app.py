#!/usr/bin/env python3
"""
Web-based Crypto Investment Strategy service - Minimal Version
Designed to start reliably on Render's free tier
"""

import os
import sys
import json
import datetime
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required

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
MAINTENANCE_MODE = True  # Set to True to show maintenance message
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

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    email_frequency = db.Column(db.String(20), default="daily")
    preferred_coins = db.Column(db.String(200), default="BTC,ETH,SOL")
    preferred_indicators = db.Column(db.String(200), default="price,volume,rsi,macd")
    theme = db.Column(db.String(20), default="light")
# --- End Database Models ---

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {str(e)}", exc_info=True)
        return None

# --- Static Data ---
def get_dummy_trending_coins():
    """Generate dummy trending coins data"""
    return [
        {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC", "market_cap_rank": 1, "price_btc": 1.0, "score": 100, "thumb": ""},
        {"id": "ethereum", "name": "Ethereum", "symbol": "ETH", "market_cap_rank": 2, "price_btc": 0.05, "score": 99, "thumb": ""},
        {"id": "solana", "name": "Solana", "symbol": "SOL", "market_cap_rank": 5, "price_btc": 0.00123, "score": 98, "thumb": ""},
        {"id": "cardano", "name": "Cardano", "symbol": "ADA", "market_cap_rank": 8, "price_btc": 0.00002, "score": 97, "thumb": ""},
        {"id": "polkadot", "name": "Polkadot", "symbol": "DOT", "market_cap_rank": 12, "price_btc": 0.0001, "score": 96, "thumb": ""}
    ]

def get_dummy_crypto_data(symbol):
    """Generate dummy crypto price data"""
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
    
    for i in range(30):
        timestamp = int((now - datetime.timedelta(days=30-i-1)).timestamp() * 1000)
        price = base_price * (1 + 0.1 * (i/30) - 0.05)
        volume = base_price * 1000000 * (0.5 + i/60)
        prices.append([timestamp, price])
        volumes.append([timestamp, volume])
    
    return {
        "prices": prices,
        "volumes": volumes
    }
# --- End Static Data ---

# --- Routes ---
@app.route("/")
@app.route("/home")
def home():
    try:
        if MAINTENANCE_MODE:
            return render_template("maintenance.html", 
                                  message="The application is currently initializing. Basic functionality is available, but full features are being set up. Please check back in a few hours.",
                                  progress=30)
        
        # Use static dummy data
        trending_coins = get_dummy_trending_coins()
        btc_data = get_dummy_crypto_data("bitcoin")
        eth_data = get_dummy_crypto_data("ethereum")
        sol_data = get_dummy_crypto_data("solana")
        
        return render_template("home.html", 
                              trending_coins=trending_coins,
                              btc_data=btc_data,
                              eth_data=eth_data,
                              sol_data=sol_data)
    except Exception as e:
        logger.error(f"Error in home route: {e}", exc_info=True)
        return render_template("500.html", error_message=str(e)), 500

@app.route("/strategy")
def strategy():
    try:
        if MAINTENANCE_MODE:
            return render_template("maintenance.html", 
                                  message="The strategy section is currently being set up. Please check back in a few hours.",
                                  progress=20)
        
        # Use static dummy data
        btc_data = get_dummy_crypto_data("bitcoin")
        eth_data = get_dummy_crypto_data("ethereum")
        sol_data = get_dummy_crypto_data("solana")
        
        return render_template("strategy.html",
                              btc_data=btc_data,
                              eth_data=eth_data,
                              sol_data=sol_data)
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
        # Static status information
        status_info = {
            "app_status": "Minimal mode active",
            "maintenance_mode": MAINTENANCE_MODE,
            "server_time": datetime.datetime.utcnow().isoformat(),
            "version": "1.0.0-minimal",
            "uptime": "Just started",
            "memory_usage": "Low",
            "database_connected": True
        }
        
        return render_template("status.html", status=status_info)
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
        
        logger.info("App initialized before first request")
    except Exception as e:
        logger.error(f"Error in before_first_request: {e}", exc_info=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
