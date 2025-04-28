#!/usr/bin/env python3
"""
Web-based Crypto Investment Strategy service - Heroku version
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

# Set up Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'crypto_investment_strategy_secret_key'

# Use PostgreSQL on Heroku, SQLite locally
if 'DATABASE_URL' in os.environ:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Create data directory if it doesn't exist
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
    return User.query.get(int(user_id))

# Crypto data functions
def get_crypto_data(symbol, days=30):
    """Get cryptocurrency price data"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Create DataFrame
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add volume
        volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('timestamp', inplace=True)
        df['volume'] = volume_df['volume']
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        # Return dummy data if API fails
        dates = pd.date_range(end=datetime.datetime.now(), periods=days)
        df = pd.DataFrame(index=dates)
        df['price'] = [50000 - i * 100 + i * i * 5 for i in range(days)]
        df['volume'] = [2000000000 - i * 10000000 + i * i * 500000 for i in range(days)]
        return df

def get_trending_coins():
    """Get trending cryptocurrencies"""
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url)
        data = response.json()
        
        trending = []
        for coin in data['coins'][:5]:  # Get top 5
            item = coin['item']
            trending.append({
                'id': item['id'],
                'name': item['name'],
                'symbol': item['symbol'],
                'market_cap_rank': item['market_cap_rank'],
                'price_btc': item['price_btc'],
                'score': item.get('score', 0),
                'thumb': item['thumb']
            })
        return trending
    except Exception as e:
        print(f"Error fetching trending coins: {e}")
        # Return dummy data if API fails
        return [
            {'id': 'solana', 'name': 'Solana', 'symbol': 'SOL', 'market_cap_rank': 5, 'price_btc': 0.00123, 'score': 100, 'thumb': ''},
            {'id': 'cardano', 'name': 'Cardano', 'symbol': 'ADA', 'market_cap_rank': 8, 'price_btc': 0.00002, 'score': 98, 'thumb': ''},
            {'id': 'polkadot', 'name': 'Polkadot', 'symbol': 'DOT', 'market_cap_rank': 12, 'price_btc': 0.00034, 'score': 95, 'thumb': ''},
            {'id': 'avalanche-2', 'name': 'Avalanche', 'symbol': 'AVAX', 'market_cap_rank': 11, 'price_btc': 0.00045, 'score': 92, 'thumb': ''},
            {'id': 'chainlink', 'name': 'Chainlink', 'symbol': 'LINK', 'market_cap_rank': 15, 'price_btc': 0.00028, 'score': 90, 'thumb': ''}
        ]

def get_economic_indicators():
    """Get economic indicators"""
    # In a real implementation, this would fetch from economic data APIs
    # For now, we'll use dummy data
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
    """Calculate technical indicators for a dataframe with price data"""
    # Calculate RSI
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['signal']
    
    # Calculate Bollinger Bands
    df['sma20'] = df['price'].rolling(window=20).mean()
    df['std20'] = df['price'].rolling(window=20).std()
    df['upper_band'] = df['sma20'] + (df['std20'] * 2)
    df['lower_band'] = df['sma20'] - (df['std20'] * 2)
    
    # Calculate Moving Averages
    df['sma50'] = df['price'].rolling(window=50).mean()
    df['sma200'] = df['price'].rolling(window=200).mean()
    
    return df

def generate_price_chart(df, symbol):
    """Generate price chart with indicators"""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['price'], label='Price')
    plt.plot(df.index, df['sma20'], label='SMA 20', linestyle='--')
    plt.plot(df.index, df['upper_band'], label='Upper Band', linestyle=':')
    plt.plot(df.index, df['lower_band'], label='Lower Band', linestyle=':')
    plt.title(f'{symbol.upper()} Price with Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def generate_indicator_chart(df, symbol, indicator='rsi'):
    """Generate chart for a specific indicator"""
    plt.figure(figsize=(10, 4))
    
    if indicator == 'rsi':
        plt.plot(df.index, df['rsi'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title(f'{symbol.upper()} RSI')
        plt.ylabel('RSI')
        plt.ylim(0, 100)
    
    elif indicator == 'macd':
        plt.plot(df.index, df['macd'], label='MACD')
        plt.plot(df.index, df['signal'], label='Signal')
        plt.bar(df.index, df['macd_histogram'], label='Histogram', alpha=0.3)
        plt.title(f'{symbol.upper()} MACD')
        plt.ylabel('MACD')
    
    elif indicator == 'volume':
        plt.bar(df.index, df['volume'], label='Volume', alpha=0.6)
        plt.title(f'{symbol.upper()} Volume')
        plt.ylabel('Volume')
    
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def get_asset_rotation_strategy():
    """Generate asset rotation strategy recommendations"""
    # In a real implementation, this would use more sophisticated analysis
    # For now, we'll use a simplified approach
    
    # Get data for main coins
    btc_data = get_crypto_data('bitcoin', days=30)
    eth_data = get_crypto_data('ethereum', days=30)
    sol_data = get_crypto_data('solana', days=30)
    
    # Calculate performance
    btc_perf = (btc_data['price'].iloc[-1] / btc_data['price'].iloc[0] - 1) * 100
    eth_perf = (eth_data['price'].iloc[-1] / eth_data['price'].iloc[0] - 1) * 100
    sol_perf = (sol_data['price'].iloc[-1] / sol_data['price'].iloc[0] - 1) * 100
    
    # Get trending coins
    trending = get_trending_coins()
    
    # Get economic indicators
    econ = get_economic_indicators()
    
    # Determine allocation based on performance and indicators
    allocations = []
    
    # Bitcoin allocation
    if btc_perf > 10 and econ['usd_trend'] == 'falling':
        btc_alloc = 40
        btc_signal = 'BUY'
    elif btc_perf < -10 and econ['usd_trend'] == 'rising':
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
    per_trending = remaining / len(trending)
    
    for coin in trending:
        allocations.append({
            'coin': f"{coin['name']} ({coin['symbol']})",
            'allocation': round(per_trending),
            'signal': 'BUY' if coin['score'] > 95 else 'RESEARCH',
            'performance': 'Trending'
        })
    
    return allocations

def generate_buy_sell_recommendations():
    """Generate buy/sell recommendations"""
    # Get data for main coins
    btc_data = get_crypto_data('bitcoin', days=30)
    eth_data = get_crypto_data('ethereum', days=30)
    sol_data = get_crypto_data('solana', days=30)
    
    # Calculate indicators
    btc_data = calculate_technical_indicators(btc_data)
    eth_data = calculate_technical_indicators(eth_data)
    sol_data = calculate_technical_indicators(sol_data)
    
    # Get economic indicators
    econ = get_economic_indicators()
    
    recommendations = []
    
    # Bitcoin recommendation
    btc_rsi = btc_data['rsi'].iloc[-1]
    btc_macd_hist = btc_data['macd_histogram'].iloc[-1]
    btc_price = btc_data['price'].iloc[-1]
    btc_sma50 = btc_data['sma50'].iloc[-1]
    
    if btc_rsi < 30 and btc_macd_hist > 0 and econ['usd_trend'] == 'falling':
        btc_rec = 'STRONG BUY'
        btc_reason = 'Oversold RSI, positive MACD momentum, and weakening USD'
    elif btc_rsi > 70 and btc_macd_hist < 0 and econ['usd_trend'] == 'rising':
        btc_rec = 'STRONG SELL'
        btc_reason = 'Overbought RSI, negative MACD momentum, and strengthening USD'
    elif btc_price > btc_sma50 and btc_macd_hist > 0:
        btc_rec = 'BUY'
        btc_reason = 'Price above 50-day MA with positive momentum'
    elif btc_price < btc_sma50 and btc_macd_hist < 0:
        btc_rec = 'SELL'
        btc_reason = 'Price below 50-day MA with negative momentum'
    else:
        btc_rec = 'HOLD'
        btc_reason = 'Mixed signals, maintain current position'
    
    recommendations.append({
        'coin': 'Bitcoin (BTC)',
        'price': f"${btc_price:,.2f}",
        'recommendation': btc_rec,
        'reason': btc_reason
    })
    
    # Ethereum recommendation
    eth_rsi = eth_data['rsi'].iloc[-1]
    eth_macd_hist = eth_data['macd_histogram'].iloc[-1]
    eth_price = eth_data['price'].iloc[-1]
    eth_sma50 = eth_data['sma50'].iloc[-1]
    
    if eth_rsi < 30 and eth_macd_hist > 0:
        eth_rec = 'STRONG BUY'
        eth_reason = 'Oversold RSI with positive MACD momentum'
    elif eth_rsi > 70 and eth_macd_hist < 0:
        eth_rec = 'STRONG SELL'
        eth_reason = 'Overbought RSI with negative MACD momentum'
    elif eth_price > eth_sma50 and eth_macd_hist > 0:
        eth_rec = 'BUY'
        eth_reason = 'Price above 50-day MA with positive momentum'
    elif eth_price < eth_sma50 and eth_macd_hist < 0:
        eth_rec = 'SELL'
        eth_reason = 'Price below 50-day MA with negative momentum'
    else:
        eth_rec = 'HOLD'
        eth_reason = 'Mixed signals, maintain current position'
    
    recommendations.append({
        'coin': 'Ethereum (ETH)',
        'price': f"${eth_price:,.2f}",
        'recommendation': eth_rec,
        'reason': eth_reason
    })
    
    # Solana recommendation
    sol_rsi = sol_data['rsi'].iloc[-1]
    sol_macd_hist = sol_data['macd_histogram'].iloc[-1]
    sol_price = sol_data['price'].iloc[-1]
    sol_sma50 = sol_data['sma50'].iloc[-1]
    
    if sol_rsi < 30 and sol_macd_hist > 0:
        sol_rec = 'STRONG BUY'
        sol_reason = 'Oversold RSI with positive MACD momentum'
    elif sol_rsi > 70 and sol_macd_hist < 0:
        sol_rec = 'STRONG SELL'
        sol_reason = 'Overbought RSI with negative MACD momentum'
    elif sol_price > sol_sma50 and sol_macd_hist > 0:
        sol_rec = 'BUY'
        sol_reason = 'Price above 50-day MA with positive momentum'
    elif sol_price < sol_sma50 and sol_macd_hist < 0:
        sol_rec = 'SELL'
        sol_reason = 'Price below 50-day MA with negative momentum'
    else:
        sol_rec = 'HOLD'
        sol_reason = 'Mixed signals, maintain current position'
    
    recommendations.append({
        'coin': 'Solana (SOL)',
        'price': f"${sol_price:,.2f}",
        'recommendation': sol_rec,
        'reason': sol_reason
    })
    
    # Add trending coins
    trending = get_trending_coins()
    for coin in trending:
        recommendations.append({
            'coin': f"{coin['name']} ({coin['symbol']})",
            'price': f"${coin['price_btc'] * btc_price:,.2f}",
            'recommendation': 'RESEARCH',
            'reason': 'Trending coin, research fundamentals before investing'
        })
    
    return recommendations

def send_email_report(email_address):
    """Send email report with crypto analysis"""
    try:
        # Generate recommendations
        recommendations = generate_buy_sell_recommendations()
        allocations = get_asset_rotation_strategy()
        econ = get_economic_indicators()
        
        # Get data for charts
        btc_data = get_crypto_data('bitcoin', days=30)
        btc_data = calculate_technical_indicators(btc_data)
        
        # Generate charts
        price_chart = generate_price_chart(btc_data, 'bitcoin')
        rsi_chart = generate_indicator_chart(btc_data, 'bitcoin', 'rsi')
        
        # Create email content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .header {{ background-color: #0066cc; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .buy {{ color: green; font-weight: bold; }}
                .sell {{ color: red; font-weight: bold; }}
                .hold {{ color: orange; font-weight: bold; }}
                .research {{ color: blue; font-weight: bold; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Crypto Investment Strategy Daily Report</h1>
                <p>{datetime.datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="section">
                <h2>Market Overview</h2>
                <p>USD Index: {econ['usd_index']} ({econ['usd_trend']})</p>
                <p>Inflation Rate: {econ['inflation_rate']}% ({econ['inflation_trend']})</p>
                <p>Interest Rate: {econ['interest_rate']}% ({econ['interest_trend']})</p>
                <p>Global Liquidity: {econ['global_liquidity']} ({econ['liquidity_trend']})</p>
                <p>Stock Market: {econ['stock_market']}</p>
                <p>Bond Market: {econ['bond_market']}</p>
                <p>Gold Price: ${econ['gold_price']} ({econ['gold_trend']})</p>
            </div>
            
            <div class="section">
                <h2>Buy/Sell Recommendations</h2>
                <table>
                    <tr>
                        <th>Coin</th>
                        <th>Price</th>
                        <th>Recommendation</th>
                        <th>Reason</th>
                    </tr>
        """
        
        for rec in recommendations:
            rec_class = rec['recommendation'].lower().replace(' ', '')
            html_content += f"""
                    <tr>
                        <td>{rec['coin']}</td>
                        <td>{rec['price']}</td>
                        <td class="{rec_class}">{rec['recommendation']}</td>
                        <td>{rec['reason']}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Asset Rotation Strategy</h2>
                <p>Recommended portfolio allocation based on current market conditions:</p>
                <table>
                    <tr>
                        <th>Coin</th>
                        <th>Allocation %</th>
                        <th>Signal</th>
                        <th>Performance</th>
                    </tr>
        """
        
        for alloc in allocations:
            signal_class = alloc['signal'].lower()
            html_content += f"""
                    <tr>
                        <td>{alloc['coin']}</td>
                        <td>{alloc['allocation']}%</td>
                        <td class="{signal_class}">{alloc['signal']}</td>
                        <td>{alloc['performance']}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Bitcoin Analysis</h2>
                <img src="cid:price_chart" style="width: 100%; max-width: 800px;">
                <img src="cid:rsi_chart" style="width: 100%; max-width: 800px;">
            </div>
            
            <div class="footer">
                <p>This report is generated by the Crypto Investment Strategy service.</p>
                <p>DISCLAIMER: This is not financial advice. Always do your own research before investing.</p>
            </div>
        </body>
        </html>
        """
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = 'crypto.investment.strategy@example.com'
        msg['To'] = email_address
        msg['Subject'] = f'Crypto Investment Strategy Daily Report - {datetime.datetime.now().strftime("%B %d, %Y")}'
        
        # Attach HTML content
        msg.attach(MIMEText(html_content, 'html'))
        
        # Attach images
        # For price chart
        price_img_data = base64.b64decode(price_chart.split(',')[1])
        price_img = MIMEImage(price_img_data)
        price_img.add_header('Content-ID', '<price_chart>')
        msg.attach(price_img)
        
        # For RSI chart
        rsi_img_data = base64.b64decode(rsi_chart.split(',')[1])
        rsi_img = MIMEImage(rsi_img_data)
        rsi_img.add_header('Content-ID', '<rsi_chart>')
        msg.attach(rsi_img)
        
        # In a real implementation, you would send the email here
        # For now, we'll just save it to a file
        os.makedirs('data', exist_ok=True)
        with open(f'data/email_report_{datetime.datetime.now().strftime("%Y%m%d")}.html', 'w') as f:
            f.write(html_content)
        
        print(f"Email report saved to data/email_report_{datetime.datetime.now().strftime('%Y%m%d')}.html")
        print(f"In a production environment, this would be sent to {email_address}")
        
        return True
    except Exception as e:
        print(f"Error sending email report: {e}")
        return False

# Email service thread
def email_service_thread():
    """Background thread for sending email reports"""
    while True:
        try:
            # Send email to keithtrmurray@aol.com
            send_email_report('keithtrmurray@aol.com')
            
            # Sleep for 24 hours
            time.sleep(86400)  # 24 hours in seconds
        except Exception as e:
            print(f"Error in email service thread: {e}")
            time.sleep(3600)  # Sleep for 1 hour on error

# Routes
@app.route('/')
def home():
    return render_template('home.html', now=datetime.datetime.now())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check username and password.', 'danger')
    
    return render_template('login.html', now=datetime.datetime.now())

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first()
        if user_exists:
            flash('Username or email already exists.', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            
            preferences = UserPreference(user=user)
            db.session.add(preferences)
            
            db.session.commit()
            
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html', now=datetime.datetime.now())

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get data for main coins
    btc_data = get_crypto_data('bitcoin', days=30)
    eth_data = get_crypto_data('ethereum', days=30)
    sol_data = get_crypto_data('solana', days=30)
    
    # Calculate indicators
    btc_data = calculate_technical_indicators(btc_data)
    eth_data = calculate_technical_indicators(eth_data)
    sol_data = calculate_technical_indicators(sol_data)
    
    # Generate charts
    btc_price_chart = generate_price_chart(btc_data, 'bitcoin')
    eth_price_chart = generate_price_chart(eth_data, 'ethereum')
    sol_price_chart = generate_price_chart(sol_data, 'solana')
    
    btc_rsi_chart = generate_indicator_chart(btc_data, 'bitcoin', 'rsi')
    eth_rsi_chart = generate_indicator_chart(eth_data, 'ethereum', 'rsi')
    sol_rsi_chart = generate_indicator_chart(sol_data, 'solana', 'rsi')
    
    btc_macd_chart = generate_indicator_chart(btc_data, 'bitcoin', 'macd')
    eth_macd_chart = generate_indicator_chart(eth_data, 'ethereum', 'macd')
    sol_macd_chart = generate_indicator_chart(sol_data, 'solana', 'macd')
    
    # Get recommendations and allocations
    recommendations = generate_buy_sell_recommendations()
    allocations = get_asset_rotation_strategy()
    
    # Get economic indicators
    econ = get_economic_indicators()
    
    return render_template(
        'dashboard.html',
        btc_price_chart=btc_price_chart,
        eth_price_chart=eth_price_chart,
        sol_price_chart=sol_price_chart,
        btc_rsi_chart=btc_rsi_chart,
        eth_rsi_chart=eth_rsi_chart,
        sol_rsi_chart=sol_rsi_chart,
        btc_macd_chart=btc_macd_chart,
        eth_macd_chart=eth_macd_chart,
        sol_macd_chart=sol_macd_chart,
        recommendations=recommendations,
        allocations=allocations,
        econ=econ,
        now=datetime.datetime.now()
    )

@app.route('/preferences', methods=['GET', 'POST'])
@login_required
def preferences():
    if request.method == 'POST':
        email_frequency = request.form.get('email_frequency')
        preferred_coins = request.form.get('preferred_coins')
        preferred_indicators = request.form.get('preferred_indicators')
        theme = request.form.get('theme')
        
        current_user.preferences.email_frequency = email_frequency
        current_user.preferences.preferred_coins = preferred_coins
        current_user.preferences.preferred_indicators = preferred_indicators
        current_user.preferences.theme = theme
        
        db.session.commit()
        
        flash('Preferences updated successfully!', 'success')
        return redirect(url_for('preferences'))
    
    return render_template('preferences.html', now=datetime.datetime.now())

@app.route('/education')
def education():
    return render_template('education.html', now=datetime.datetime.now())

@app.route('/api/crypto/<symbol>')
def api_crypto(symbol):
    days = request.args.get('days', 30, type=int)
    df = get_crypto_data(symbol, days=days)
    df = calculate_technical_indicators(df)
    
    # Convert to JSON
    data = {
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'prices': df['price'].tolist(),
        'volumes': df['volume'].tolist(),
        'rsi': df['rsi'].tolist(),
        'macd': df['macd'].tolist(),
        'signal': df['signal'].tolist(),
        'histogram': df['macd_histogram'].tolist(),
        'upper_band': df['upper_band'].tolist(),
        'lower_band': df['lower_band'].tolist(),
        'sma20': df['sma20'].tolist(),
        'sma50': df['sma50'].tolist()
    }
    
    return jsonify(data)

@app.route('/api/trending')
def api_trending():
    trending = get_trending_coins()
    return jsonify(trending)

@app.route('/api/economic')
def api_economic():
    econ = get_economic_indicators()
    return jsonify(econ)

@app.route('/api/recommendations')
def api_recommendations():
    recommendations = generate_buy_sell_recommendations()
    return jsonify(recommendations)

@app.route('/api/allocations')
def api_allocations():
    allocations = get_asset_rotation_strategy()
    return jsonify(allocations)

# Create database and admin user
def initialize_database():
    with app.app_context():
        db.create_all()
        
        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            # Create admin user
            hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin = User(
                username='admin',
                email='keithtrmurray@aol.com',
                password=hashed_password,
                is_admin=True,
                subscription_type='premium',
                subscription_start=datetime.datetime.utcnow(),
                subscription_end=datetime.datetime.utcnow() + datetime.timedelta(days=36500)
            )
            db.session.add(admin)
            
            # Create admin preferences
            preferences = UserPreference(user=admin)
            db.session.add(preferences)
            
            db.session.commit()
            print("Admin user created successfully!")

# Create HTML templates
def create_templates():
    os.makedirs('templates', exist_ok=True)
    
    # Base layout template
    with open('templates/layout.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Investment Strategy{% block title %}{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }
        .navbar {
            background-color: #0066cc;
        }
        .navbar-brand {
            font-weight: bold;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            font-weight: bold;
            background-color: #f1f8ff;
        }
        .btn-primary {
            background-color: #0066cc;
            border-color: #0066cc;
        }
        .btn-primary:hover {
            background-color: #0056b3;
            border-color: #0056b3;
        }
        .footer {
            margin-top: 50px;
            padding: 20px 0;
            background-color: #343a40;
            color: white;
        }
        .buy {
            color: green;
            font-weight: bold;
        }
        .sell {
            color: red;
            font-weight: bold;
        }
        .hold {
            color: orange;
            font-weight: bold;
        }
        .research {
            color: blue;
            font-weight: bold;
        }
        .chart-container {
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
        }
    </style>
    {% block styles %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('home') }}">
                <i class="fas fa-chart-line me-2"></i>Crypto Investment Strategy
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('home') }}">Home</a>
                    </li>
                    {% if current_user.is_authenticated %}
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('dashboard') }}">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('preferences') }}">Preferences</a>
                    </li>
                    {% endif %}
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('education') }}">Education</a>
                    </li>
                </ul>
                <ul class="navbar-nav">
                    {% if current_user.is_authenticated %}
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('logout') }}">Logout</a>
                    </li>
                    {% else %}
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('login') }}">Login</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('register') }}">Register</a>
                    </li>
                    {% endif %}
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>

    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5>Crypto Investment Strategy</h5>
                    <p>Your trusted source for cryptocurrency investment insights and recommendations.</p>
                </div>
                <div class="col-md-3">
                    <h5>Links</h5>
                    <ul class="list-unstyled">
                        <li><a href="{{ url_for('home') }}" class="text-white">Home</a></li>
                        <li><a href="{{ url_for('education') }}" class="text-white">Education</a></li>
                        {% if current_user.is_authenticated %}
                        <li><a href="{{ url_for('dashboard') }}" class="text-white">Dashboard</a></li>
                        {% else %}
                        <li><a href="{{ url_for('login') }}" class="text-white">Login</a></li>
                        {% endif %}
                    </ul>
                </div>
                <div class="col-md-3">
                    <h5>Contact</h5>
                    <p>Email: keithtrmurray@aol.com</p>
                </div>
            </div>
            <hr class="bg-white">
            <div class="text-center">
                <p>DISCLAIMER: This is not financial advice. Always do your own research before investing.</p>
                <p>&copy; {{ now.year }} Crypto Investment Strategy. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
        ''')
    
    # Home page template
    with open('templates/home.html', 'w') as f:
        f.write('''
{% extends "layout.html" %}

{% block title %} - Home{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-body">
                <h1 class="card-title">Welcome to Crypto Investment Strategy</h1>
                <p class="lead">Your trusted source for cryptocurrency investment insights and recommendations.</p>
                <hr>
                <p>Our service provides daily analysis and recommendations for Bitcoin, Ethereum, Solana, and trending altcoins, helping you make informed investment decisions.</p>
                <p>We analyze technical indicators, market trends, and economic factors to provide you with actionable insights.</p>
                <div class="mt-4">
                    {% if current_user.is_authenticated %}
                    <a href="{{ url_for('dashboard') }}" class="btn btn-primary btn-lg">Go to Dashboard</a>
                    {% else %}
                    <a href="{{ url_for('register') }}" class="btn btn-primary btn-lg me-2">Register Now</a>
                    <a href="{{ url_for('login') }}" class="btn btn-outline-primary btn-lg">Login</a>
                    {% endif %}
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h2>Our Features</h2>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-4">
                            <h4><i class="fas fa-chart-line text-primary me-2"></i>Real-time Market Data</h4>
                            <p>Access up-to-date price charts and market data for Bitcoin, Ethereum, Solana, and trending altcoins.</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-4">
                            <h4><i class="fas fa-signal text-primary me-2"></i>Technical Indicators</h4>
                            <p>Analyze RSI, MACD, Bollinger Bands, and other technical indicators to identify market trends.</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-4">
                            <h4><i class="fas fa-exchange-alt text-primary me-2"></i>Asset Rotation Strategy</h4>
                            <p>Get recommendations on optimal portfolio allocation based on current market conditions.</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-4">
                            <h4><i class="fas fa-envelope text-primary me-2"></i>Daily Email Reports</h4>
                            <p>Receive daily email reports with market analysis and investment recommendations.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h3>Market Overview</h3>
            </div>
            <div class="card-body">
                <div id="market-overview-loading">Loading market data...</div>
                <div id="market-overview" style="display: none;">
                    <div class="mb-3">
                        <h5>Bitcoin (BTC)</h5>
                        <p class="mb-1">Price: <span id="btc-price"></span></p>
                        <p class="mb-1">24h Change: <span id="btc-change"></span></p>
                    </div>
                    <div class="mb-3">
                        <h5>Ethereum (ETH)</h5>
                        <p class="mb-1">Price: <span id="eth-price"></span></p>
                        <p class="mb-1">24h Change: <span id="eth-change"></span></p>
                    </div>
                    <div class="mb-3">
                        <h5>Solana (SOL)</h5>
                        <p class="mb-1">Price: <span id="sol-price"></span></p>
                        <p class="mb-1">24h Change: <span id="sol-change"></span></p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Trending Coins</h3>
            </div>
            <div class="card-body">
                <div id="trending-loading">Loading trending coins...</div>
                <div id="trending-coins" style="display: none;"></div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Economic Indicators</h3>
            </div>
            <div class="card-body">
                <div id="econ-loading">Loading economic data...</div>
                <div id="econ-indicators" style="display: none;"></div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    // Fetch market data
    fetch('/api/crypto/bitcoin?days=1')
        .then(response => response.json())
        .then(data => {
            const lastIndex = data.prices.length - 1;
            const prevIndex = data.prices.length - 2;
            const btcPrice = data.prices[lastIndex];
            const btcPrevPrice = data.prices[prevIndex];
            const btcChange = ((btcPrice - btcPrevPrice) / btcPrevPrice) * 100;
            
            document.getElementById('btc-price').textContent = '$' + btcPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            document.getElementById('btc-change').textContent = btcChange.toFixed(2) + '%';
            document.getElementById('btc-change').className = btcChange >= 0 ? 'text-success' : 'text-danger';
            
            return fetch('/api/crypto/ethereum?days=1');
        })
        .then(response => response.json())
        .then(data => {
            const lastIndex = data.prices.length - 1;
            const prevIndex = data.prices.length - 2;
            const ethPrice = data.prices[lastIndex];
            const ethPrevPrice = data.prices[prevIndex];
            const ethChange = ((ethPrice - ethPrevPrice) / ethPrevPrice) * 100;
            
            document.getElementById('eth-price').textContent = '$' + ethPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            document.getElementById('eth-change').textContent = ethChange.toFixed(2) + '%';
            document.getElementById('eth-change').className = ethChange >= 0 ? 'text-success' : 'text-danger';
            
            return fetch('/api/crypto/solana?days=1');
        })
        .then(response => response.json())
        .then(data => {
            const lastIndex = data.prices.length - 1;
            const prevIndex = data.prices.length - 2;
            const solPrice = data.prices[lastIndex];
            const solPrevPrice = data.prices[prevIndex];
            const solChange = ((solPrice - solPrevPrice) / solPrevPrice) * 100;
            
            document.getElementById('sol-price').textContent = '$' + solPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            document.getElementById('sol-change').textContent = solChange.toFixed(2) + '%';
            document.getElementById('sol-change').className = solChange >= 0 ? 'text-success' : 'text-danger';
            
            document.getElementById('market-overview-loading').style.display = 'none';
            document.getElementById('market-overview').style.display = 'block';
        })
        .catch(error => {
            console.error('Error fetching market data:', error);
            document.getElementById('market-overview-loading').textContent = 'Failed to load market data.';
        });
    
    // Fetch trending coins
    fetch('/api/trending')
        .then(response => response.json())
        .then(data => {
            const trendingHtml = data.map(coin => `
                <div class="mb-3">
                    <h5>${coin.name} (${coin.symbol})</h5>
                    <p class="mb-1">Rank: #${coin.market_cap_rank || 'N/A'}</p>
                    <p class="mb-0">Score: ${coin.score}</p>
                </div>
            `).join('');
            
            document.getElementById('trending-coins').innerHTML = trendingHtml;
            document.getElementById('trending-loading').style.display = 'none';
            document.getElementById('trending-coins').style.display = 'block';
        })
        .catch(error => {
            console.error('Error fetching trending coins:', error);
            document.getElementById('trending-loading').textContent = 'Failed to load trending coins.';
        });
    
    // Fetch economic indicators
    fetch('/api/economic')
        .then(response => response.json())
        .then(data => {
            const econHtml = `
                <p><strong>USD Index:</strong> ${data.usd_index} <span class="text-${data.usd_trend === 'rising' ? 'danger' : 'success'}">(${data.usd_trend})</span></p>
                <p><strong>Inflation:</strong> ${data.inflation_rate}% <span class="text-${data.inflation_trend === 'rising' ? 'danger' : 'success'}">(${data.inflation_trend})</span></p>
                <p><strong>Interest Rate:</strong> ${data.interest_rate}% <span class="text-${data.interest_trend === 'rising' ? 'danger' : 'success'}">(${data.interest_trend})</span></p>
                <p><strong>Global Liquidity:</strong> ${data.global_liquidity} <span class="text-${data.liquidity_trend === 'tightening' ? 'danger' : 'success'}">(${data.liquidity_trend})</span></p>
                <p><strong>Gold:</strong> $${data.gold_price} <span class="text-${data.gold_trend === 'rising' ? 'success' : 'danger'}">(${data.gold_trend})</span></p>
            `;
            
            document.getElementById('econ-indicators').innerHTML = econHtml;
            document.getElementById('econ-loading').style.display = 'none';
            document.getElementById('econ-indicators').style.display = 'block';
        })
        .catch(error => {
            console.error('Error fetching economic indicators:', error);
            document.getElementById('econ-loading').textContent = 'Failed to load economic indicators.';
        });
</script>
{% endblock %}
        ''')
    
    # Login page template
    with open('templates/login.html', 'w') as f:
        f.write('''
{% extends "layout.html" %}

{% block title %} - Login{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-6 offset-md-3">
        <div class="card">
            <div class="card-header">
                <h2>Login</h2>
            </div>
            <div class="card-body">
                <form method="POST" action="">
                    <div class="mb-3">
                        <label for="username" class="form-label">Username</label>
                        <input type="text" class="form-control" id="username" name="username" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Login</button>
                </form>
                <div class="mt-3">
                    <small>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></small>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        ''')
    
    # Register page template
    with open('templates/register.html', 'w') as f:
        f.write('''
{% extends "layout.html" %}

{% block title %} - Register{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-6 offset-md-3">
        <div class="card">
            <div class="card-header">
                <h2>Register</h2>
            </div>
            <div class="card-body">
                <form method="POST" action="">
                    <div class="mb-3">
                        <label for="username" class="form-label">Username</label>
                        <input type="text" class="form-control" id="username" name="username" required>
                    </div>
                    <div class="mb-3">
                        <label for="email" class="form-label">Email</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Register</button>
                </form>
                <div class="mt-3">
                    <small>Already have an account? <a href="{{ url_for('login') }}">Login here</a></small>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        ''')
    
    # Dashboard page template
    with open('templates/dashboard.html', 'w') as f:
        f.write('''
{% extends "layout.html" %}

{% block title %} - Dashboard{% endblock %}

{% block content %}
<h1 class="mb-4">Crypto Investment Dashboard</h1>

<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>Market Overview</h3>
            </div>
            <div class="card-body">
                <ul class="nav nav-tabs" id="marketTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="btc-tab" data-bs-toggle="tab" data-bs-target="#btc" type="button" role="tab">Bitcoin</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="eth-tab" data-bs-toggle="tab" data-bs-target="#eth" type="button" role="tab">Ethereum</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="sol-tab" data-bs-toggle="tab" data-bs-target="#sol" type="button" role="tab">Solana</button>
                    </li>
                </ul>
                <div class="tab-content mt-3" id="marketTabsContent">
                    <div class="tab-pane fade show active" id="btc" role="tabpanel">
                        <h4>Bitcoin (BTC) Price Chart</h4>
                        <div class="chart-container">
                            <img src="{{ btc_price_chart }}" alt="Bitcoin Price Chart" class="img-fluid">
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <h5>RSI</h5>
                                <img src="{{ btc_rsi_chart }}" alt="Bitcoin RSI Chart" class="img-fluid">
                            </div>
                            <div class="col-md-6">
                                <h5>MACD</h5>
                                <img src="{{ btc_macd_chart }}" alt="Bitcoin MACD Chart" class="img-fluid">
                            </div>
                        </div>
                    </div>
                    <div class="tab-pane fade" id="eth" role="tabpanel">
                        <h4>Ethereum (ETH) Price Chart</h4>
                        <div class="chart-container">
                            <img src="{{ eth_price_chart }}" alt="Ethereum Price Chart" class="img-fluid">
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <h5>RSI</h5>
                                <img src="{{ eth_rsi_chart }}" alt="Ethereum RSI Chart" class="img-fluid">
                            </div>
                            <div class="col-md-6">
                                <h5>MACD</h5>
                                <img src="{{ eth_macd_chart }}" alt="Ethereum MACD Chart" class="img-fluid">
                            </div>
                        </div>
                    </div>
                    <div class="tab-pane fade" id="sol" role="tabpanel">
                        <h4>Solana (SOL) Price Chart</h4>
                        <div class="chart-container">
                            <img src="{{ sol_price_chart }}" alt="Solana Price Chart" class="img-fluid">
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <h5>RSI</h5>
                                <img src="{{ sol_rsi_chart }}" alt="Solana RSI Chart" class="img-fluid">
                            </div>
                            <div class="col-md-6">
                                <h5>MACD</h5>
                                <img src="{{ sol_macd_chart }}" alt="Solana MACD Chart" class="img-fluid">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Asset Rotation Strategy</h3>
            </div>
            <div class="card-body">
                <p>Recommended portfolio allocation based on current market conditions:</p>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Coin</th>
                            <th>Allocation %</th>
                            <th>Signal</th>
                            <th>Performance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for alloc in allocations %}
                        <tr>
                            <td>{{ alloc.coin }}</td>
                            <td>{{ alloc.allocation }}%</td>
                            <td class="{{ alloc.signal.lower() }}">{{ alloc.signal }}</td>
                            <td>{{ alloc.performance }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h3>Buy/Sell Recommendations</h3>
            </div>
            <div class="card-body">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Coin</th>
                            <th>Price</th>
                            <th>Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for rec in recommendations %}
                        <tr>
                            <td>{{ rec.coin }}</td>
                            <td>{{ rec.price }}</td>
                            <td class="{{ rec.recommendation.lower().replace(' ', '') }}">{{ rec.recommendation }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Economic Indicators</h3>
            </div>
            <div class="card-body">
                <table class="table">
                    <tbody>
                        <tr>
                            <th>USD Index</th>
                            <td>{{ econ.usd_index }} <span class="text-{{ 'danger' if econ.usd_trend == 'rising' else 'success' }}">({{ econ.usd_trend }})</span></td>
                        </tr>
                        <tr>
                            <th>Inflation Rate</th>
                            <td>{{ econ.inflation_rate }}% <span class="text-{{ 'danger' if econ.inflation_trend == 'rising' else 'success' }}">({{ econ.inflation_trend }})</span></td>
                        </tr>
                        <tr>
                            <th>Interest Rate</th>
                            <td>{{ econ.interest_rate }}% <span class="text-{{ 'danger' if econ.interest_trend == 'rising' else 'success' }}">({{ econ.interest_trend }})</span></td>
                        </tr>
                        <tr>
                            <th>Global Liquidity</th>
                            <td>{{ econ.global_liquidity }} <span class="text-{{ 'danger' if econ.liquidity_trend == 'tightening' else 'success' }}">({{ econ.liquidity_trend }})</span></td>
                        </tr>
                        <tr>
                            <th>Stock Market</th>
                            <td>{{ econ.stock_market }}</td>
                        </tr>
                        <tr>
                            <th>Bond Market</th>
                            <td>{{ econ.bond_market }}</td>
                        </tr>
                        <tr>
                            <th>Gold Price</th>
                            <td>${{ econ.gold_price }} <span class="text-{{ 'success' if econ.gold_trend == 'rising' else 'danger' }}">({{ econ.gold_trend }})</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Email Reports</h3>
            </div>
            <div class="card-body">
                <p>Daily reports are being sent to: <strong>{{ current_user.email }}</strong></p>
                <p>You can customize your report preferences in the <a href="{{ url_for('preferences') }}">Preferences</a> section.</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        ''')
    
    # Preferences page template
    with open('templates/preferences.html', 'w') as f:
        f.write('''
{% extends "layout.html" %}

{% block title %} - Preferences{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8 offset-md-2">
        <div class="card">
            <div class="card-header">
                <h2>User Preferences</h2>
            </div>
            <div class="card-body">
                <form method="POST" action="">
                    <div class="mb-3">
                        <label for="email_frequency" class="form-label">Email Report Frequency</label>
                        <select class="form-select" id="email_frequency" name="email_frequency">
                            <option value="daily" {% if current_user.preferences.email_frequency == 'daily' %}selected{% endif %}>Daily</option>
                            <option value="weekly" {% if current_user.preferences.email_frequency == 'weekly' %}selected{% endif %}>Weekly</option>
                            <option value="monthly" {% if current_user.preferences.email_frequency == 'monthly' %}selected{% endif %}>Monthly</option>
                        </select>
                    </div>
                    
                    <div class="mb-3">
                        <label for="preferred_coins" class="form-label">Preferred Cryptocurrencies</label>
                        <input type="text" class="form-control" id="preferred_coins" name="preferred_coins" value="{{ current_user.preferences.preferred_coins }}">
                        <div class="form-text">Enter comma-separated coin symbols (e.g., BTC,ETH,SOL)</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="preferred_indicators" class="form-label">Preferred Technical Indicators</label>
                        <input type="text" class="form-control" id="preferred_indicators" name="preferred_indicators" value="{{ current_user.preferences.preferred_indicators }}">
                        <div class="form-text">Enter comma-separated indicators (e.g., price,volume,rsi,macd)</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="theme" class="form-label">Theme</label>
                        <select class="form-select" id="theme" name="theme">
                            <option value="light" {% if current_user.preferences.theme == 'light' %}selected{% endif %}>Light</option>
                            <option value="dark" {% if current_user.preferences.theme == 'dark' %}selected{% endif %}>Dark</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Save Preferences</button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        ''')
    
    # Education page template
    with open('templates/education.html', 'w') as f:
        f.write('''
{% extends "layout.html" %}

{% block title %} - Education{% endblock %}

{% block content %}
<h1 class="mb-4">Cryptocurrency Education Center</h1>

<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h3>Technical Indicators Explained</h3>
            </div>
            <div class="card-body">
                <div class="accordion" id="indicatorsAccordion">
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingRSI">
                            <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseRSI" aria-expanded="true" aria-controls="collapseRSI">
                                Relative Strength Index (RSI)
                            </button>
                        </h2>
                        <div id="collapseRSI" class="accordion-collapse collapse show" aria-labelledby="headingRSI" data-bs-parent="#indicatorsAccordion">
                            <div class="accordion-body">
                                <p>The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions in a market.</p>
                                
                                <h5>How to Interpret RSI:</h5>
                                <ul>
                                    <li><strong>RSI > 70:</strong> Generally considered overbought, suggesting a potential reversal to the downside.</li>
                                    <li><strong>RSI < 30:</strong> Generally considered oversold, suggesting a potential reversal to the upside.</li>
                                    <li><strong>RSI Divergence:</strong> When price makes a new high/low but RSI doesn't, it can signal a potential reversal.</li>
                                </ul>
                                
                                <p>RSI is particularly useful in ranging markets but can be less reliable during strong trends. During strong uptrends, RSI can remain in overbought territory for extended periods.</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingMACD">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseMACD" aria-expanded="false" aria-controls="collapseMACD">
                                Moving Average Convergence Divergence (MACD)
                            </button>
                        </h2>
                        <div id="collapseMACD" class="accordion-collapse collapse" aria-labelledby="headingMACD" data-bs-parent="#indicatorsAccordion">
                            <div class="accordion-body">
                                <p>The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. It consists of the MACD line, signal line, and histogram.</p>
                                
                                <h5>Components of MACD:</h5>
                                <ul>
                                    <li><strong>MACD Line:</strong> The difference between the 12-period and 26-period Exponential Moving Averages (EMA).</li>
                                    <li><strong>Signal Line:</strong> The 9-period EMA of the MACD Line.</li>
                                    <li><strong>Histogram:</strong> The difference between the MACD Line and the Signal Line.</li>
                                </ul>
                                
                                <h5>How to Interpret MACD:</h5>
                                <ul>
                                    <li><strong>MACD Line crosses above Signal Line:</strong> Bullish signal, potential buy opportunity.</li>
                                    <li><strong>MACD Line crosses below Signal Line:</strong> Bearish signal, potential sell opportunity.</li>
                                    <li><strong>MACD Line crosses above zero:</strong> Indicates upward momentum.</li>
                                    <li><strong>MACD Line crosses below zero:</strong> Indicates downward momentum.</li>
                                    <li><strong>Divergence:</strong> When price makes a new high/low but MACD doesn't, it can signal a potential reversal.</li>
                                </ul>
                                
                                <p>MACD is effective for identifying changes in the strength, direction, momentum, and duration of a trend in a stock's price.</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingBB">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseBB" aria-expanded="false" aria-controls="collapseBB">
                                Bollinger Bands
                            </button>
                        </h2>
                        <div id="collapseBB" class="accordion-collapse collapse" aria-labelledby="headingBB" data-bs-parent="#indicatorsAccordion">
                            <div class="accordion-body">
                                <p>Bollinger Bands are a volatility indicator consisting of three lines: a simple moving average (middle band) and two standard deviation lines (upper and lower bands). They help identify whether prices are high or low on a relative basis.</p>
                                
                                <h5>Components of Bollinger Bands:</h5>
                                <ul>
                                    <li><strong>Middle Band:</strong> 20-period Simple Moving Average (SMA).</li>
                                    <li><strong>Upper Band:</strong> Middle Band + (2 × 20-period Standard Deviation).</li>
                                    <li><strong>Lower Band:</strong> Middle Band - (2 × 20-period Standard Deviation).</li>
                                </ul>
                                
                                <h5>How to Interpret Bollinger Bands:</h5>
                                <ul>
                                    <li><strong>Price touching Upper Band:</strong> Potentially overbought condition.</li>
                                    <li><strong>Price touching Lower Band:</strong> Potentially oversold condition.</li>
                                    <li><strong>Bands Squeeze (narrowing):</strong> Indicates low volatility, often preceding a significant price movement.</li>
                                    <li><strong>Bands Expansion (widening):</strong> Indicates high volatility.</li>
                                    <li><strong>Price moving outside the bands:</strong> Continuation of the current trend.</li>
                                </ul>
                                
                                <p>Bollinger Bands are useful for identifying potential reversal points and measuring market volatility.</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="headingMA">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseMA" aria-expanded="false" aria-controls="collapseMA">
                                Moving Averages
                            </button>
                        </h2>
                        <div id="collapseMA" class="accordion-collapse collapse" aria-labelledby="headingMA" data-bs-parent="#indicatorsAccordion">
                            <div class="accordion-body">
                                <p>Moving Averages smooth out price data to create a single flowing line, making it easier to identify the direction of the trend. They are lagging indicators that help confirm trends and support/resistance levels.</p>
                                
                                <h5>Types of Moving Averages:</h5>
                                <ul>
                                    <li><strong>Simple Moving Average (SMA):</strong> The average of a security's price over a specific period.</li>
                                    <li><strong>Exponential Moving Average (EMA):</strong> Gives more weight to recent prices, making it more responsive to new information.</li>
                                </ul>
                                
                                <h5>Common Moving Averages:</h5>
                                <ul>
                                    <li><strong>20-day MA:</strong> Short-term trend indicator.</li>
                                    <li><strong>50-day MA:</strong> Medium-term trend indicator.</li>
                                    <li><strong>200-day MA:</strong> Long-term trend indicator.</li>
                                </ul>
                                
                                <h5>How to Interpret Moving Averages:</h5>
                                <ul>
                                    <li><strong>Price above MA:</strong> Bullish signal, uptrend.</li>
                                    <li><strong>Price below MA:</strong> Bearish signal, downtrend.</li>
                                    <li><strong>Short-term MA crosses above Long-term MA:</strong> Golden Cross, bullish signal.</li>
                                    <li><strong>Short-term MA crosses below Long-term MA:</strong> Death Cross, bearish signal.</li>
                                    <li><strong>MAs as Support/Resistance:</strong> Price often bounces off MAs during trends.</li>
                                </ul>
                                
                                <p>Moving Averages are versatile indicators used in various strategies, from trend following to identifying potential entry and exit points.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Asset Rotation Strategy</h3>
            </div>
            <div class="card-body">
                <p>Asset rotation is an investment strategy that involves shifting investments between different asset classes or sectors based on their relative performance and market conditions. In the context of cryptocurrency investing, it means allocating your portfolio among different cryptocurrencies based on their current performance, trends, and market indicators.</p>
                
                <h5>Key Principles of Asset Rotation:</h5>
                <ol>
                    <li><strong>Performance-Based Allocation:</strong> Allocate more capital to assets that are performing well and reduce exposure to underperforming assets.</li>
                    <li><strong>Risk Management:</strong> Diversify across multiple cryptocurrencies to reduce overall portfolio risk.</li>
                    <li><strong>Market Cycle Awareness:</strong> Adjust allocations based on the current market cycle (bull market, bear market, accumulation, distribution).</li>
                    <li><strong>Macro-Economic Factors:</strong> Consider broader economic indicators like inflation, interest rates, and USD strength when making allocation decisions.</li>
                </ol>
                
                <h5>Our Asset Rotation Methodology:</h5>
                <p>Our Crypto Investment Strategy service uses a comprehensive approach to asset rotation that considers:</p>
                <ul>
                    <li><strong>Technical Indicators:</strong> RSI, MACD, Moving Averages, and other technical signals.</li>
                    <li><strong>Performance Metrics:</strong> Recent price performance, volatility, and momentum.</li>
                    <li><strong>Market Trends:</strong> Identifying trending cryptocurrencies with strong fundamentals.</li>
                    <li><strong>Economic Indicators:</strong> Analyzing how broader economic factors affect different cryptocurrencies.</li>
                </ul>
                
                <p>The asset rotation recommendations provided in your dashboard and email reports are designed to help you optimize your cryptocurrency portfolio allocation based on current market conditions.</p>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h3>Cryptocurrency Basics</h3>
            </div>
            <div class="card-body">
                <h5>What is Cryptocurrency?</h5>
                <p>Cryptocurrency is a digital or virtual currency that uses cryptography for security and operates on a technology called blockchain, which is a distributed ledger enforced by a network of computers.</p>
                
                <h5>Key Cryptocurrencies:</h5>
                <ul>
                    <li><strong>Bitcoin (BTC):</strong> The first and most valuable cryptocurrency, often referred to as "digital gold."</li>
                    <li><strong>Ethereum (ETH):</strong> A platform for decentralized applications and smart contracts.</li>
                    <li><strong>Solana (SOL):</strong> A high-performance blockchain supporting fast, secure, and scalable decentralized applications.</li>
                </ul>
                
                <h5>Investment Considerations:</h5>
                <ul>
                    <li><strong>Volatility:</strong> Cryptocurrencies can experience significant price swings.</li>
                    <li><strong>Research:</strong> Always research fundamentals before investing.</li>
                    <li><strong>Risk Management:</strong> Only invest what you can afford to lose.</li>
                    <li><strong>Long-term Perspective:</strong> Consider the technology's long-term potential.</li>
                </ul>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Economic Indicators Guide</h3>
            </div>
            <div class="card-body">
                <h5>How Economic Factors Affect Crypto:</h5>
                <ul>
                    <li><strong>USD Strength:</strong> A stronger USD often correlates with weaker crypto prices, and vice versa.</li>
                    <li><strong>Inflation:</strong> High inflation can drive interest in Bitcoin as an inflation hedge.</li>
                    <li><strong>Interest Rates:</strong> Higher rates typically reduce liquidity and can negatively impact risk assets like crypto.</li>
                    <li><strong>Global Liquidity:</strong> More liquidity in the financial system often benefits crypto markets.</li>
                </ul>
                
                <h5>Interpreting Our Economic Indicators:</h5>
                <ul>
                    <li><strong>USD Index Rising:</strong> Potential headwind for crypto prices.</li>
                    <li><strong>USD Index Falling:</strong> Potential tailwind for crypto prices.</li>
                    <li><strong>Inflation Rising:</strong> May benefit Bitcoin as a store of value.</li>
                    <li><strong>Interest Rates Rising:</strong> May reduce capital flow to crypto markets.</li>
                    <li><strong>Global Liquidity Expanding:</strong> Generally positive for crypto markets.</li>
                </ul>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h3>Glossary of Terms</h3>
            </div>
            <div class="card-body">
                <dl>
                    <dt>Altcoin</dt>
                    <dd>Any cryptocurrency other than Bitcoin.</dd>
                    
                    <dt>Bull Market</dt>
                    <dd>A market condition where prices are rising or expected to rise.</dd>
                    
                    <dt>Bear Market</dt>
                    <dd>A market condition where prices are falling or expected to fall.</dd>
                    
                    <dt>HODL</dt>
                    <dd>A term derived from a misspelling of "hold," referring to a buy-and-hold strategy.</dd>
                    
                    <dt>Market Cap</dt>
                    <dd>The total value of a cryptocurrency, calculated by multiplying the price by the circulating supply.</dd>
                    
                    <dt>DeFi</dt>
                    <dd>Decentralized Finance; financial services built on blockchain technology.</dd>
                    
                    <dt>NFT</dt>
                    <dd>Non-Fungible Token; a unique digital asset representing ownership of a specific item.</dd>
                    
                    <dt>Staking</dt>
                    <dd>The process of actively participating in transaction validation on a proof-of-stake blockchain.</dd>
                </dl>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        ''')

# Main function
def initialize_app():
    # Create templates
    create_templates()
    
    # Initialize database
    initialize_database()
    
    # Start email service thread
    email_thread = threading.Thread(target=email_service_thread, daemon=True)
    email_thread.start()

if __name__ == '__main__':
    # Initialize the app
    initialize_app()
    
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get('PORT', 5001))
    
    # Run the application
    print("Starting Crypto Investment Strategy web service...")
    app.run(host='0.0.0.0', port=port)
