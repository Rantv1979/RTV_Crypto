"""
Global Crypto/Forex/Commodities AI Trader (Autonomous Mode)
Features:
- Local Machine Learning (Random Forest) for Free AI predictions.
- Fully Autonomous Loop (Scans -> Predicts -> Executes).
- No Paid APIs required.
"""

import os
import time
import threading
import sys
import logging
import pytz
import traceback
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

# GUI & Data
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# --- FREE AI LIBRARIES (Scikit-Learn) ---
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================
st.set_page_config(page_title="Autonomous AI Trader", layout="wide", initial_sidebar_state="expanded")

# Assets (Yahoo Finance Tickers)
ASSETS = {
    "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
    "FOREX": ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"],
    "COMMODITIES": ["GC=F", "CL=F", "SI=F"] # Gold, Oil, Silver
}

ALL_TICKERS = ASSETS["CRYPTO"] + ASSETS["FOREX"] + ASSETS["COMMODITIES"]

# ================= UTILITY FUNCTIONS =================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

def compute_indicators(df):
    """Generate features for the AI to learn from"""
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = rsi(df['Close'])
    df['EMA_8'] = ema(df['Close'], 8)
    df['EMA_21'] = ema(df['Close'], 21)
    df['MACD'] = ema(df['Close'], 12) - ema(df['Close'], 26)
    
    # AI Features: Normalized difference between EMAs, RSI value, Volatility
    df['Feat_Trend'] = (df['EMA_8'] - df['EMA_21']) / df['Close']
    df['Feat_RSI'] = df['RSI'] / 100.0
    df['Feat_Vol'] = df['Returns'].rolling(5).std()
    
    # Target: 1 if price went UP in next candle, 0 if DOWN
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

# ================= CORE AI ENGINE (FREE / LOCAL) =================
class LocalAITrader:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False
        self.accuracy = 0.0
        self.last_training_time = None

    def train_model(self, tickers):
        """Fetches historical data and trains the Random Forest model"""
        logger.info("Training AI Model on historical data...")
        master_data = []
        
        for ticker in tickers:
            try:
                # Get last 60 days of hourly data for training
                data = yf.download(ticker, period="1mo", interval="1h", progress=False)
                if len(data) < 50: continue
                
                # Cleanup MultiIndex if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                data = compute_indicators(data)
                master_data.append(data)
            except Exception as e:
                logger.error(f"Training error {ticker}: {e}")
        
        if not master_data:
            return False, "No data collected"

        # Combine all asset data to make a general market model
        full_df = pd.concat(master_data)
        
        # Features (X) and Target (y)
        features = ['Feat_Trend', 'Feat_RSI', 'Feat_Vol']
        X = full_df[features]
        y = full_df['Target']
        
        # Split and Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, preds)
        self.is_trained = True
        self.last_training_time = datetime.now()
        
        return True, f"Trained on {len(full_df)} candles. Accuracy: {self.accuracy:.1%}"

    def predict(self, ticker_df):
        """Predicts probability of price going UP for the specific asset"""
        if not self.is_trained: return 0.5
        
        try:
            # Prepare latest candle
            rec = ticker_df.iloc[[-1]].copy()
            features = rec[['Feat_Trend', 'Feat_RSI', 'Feat_Vol']]
            
            # Probability of Class 1 (Buy)
            prob_buy = self.model.predict_proba(features)[0][1]
            return prob_buy
        except Exception:
            return 0.5

# ================= TRADING SYSTEM =================
class AutonomousSystem:
    def __init__(self):
        self.ai = LocalAITrader()
        self.active = False
        self.positions = {} # Symbol -> {Entry, Qty, PnL}
        self.balance = 50000.0 # Paper Money
        self.trade_log = []
        self.thread = None
        self.stop_event = threading.Event()

    def start_autonomous_loop(self):
        if self.active: return
        self.active = True
        self.stop_event.clear()
        
        # Train AI on startup if not trained
        if not self.ai.is_trained:
            success, msg = self.ai.train_model(ALL_TICKERS[:5]) # Train on a subset to be fast
            print(msg)
            
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()

    def stop_autonomous_loop(self):
        self.active = False
        self.stop_event.set()

    def _background_loop(self):
        """The brain that runs 24/7 in background"""
        logger.info("Autonomous Loop Started")
        
        while not self.stop_event.is_set():
            try:
                # 1. Manage Active Positions (Check Stops/Targets)
                self._manage_positions()
                
                # 2. Scan for New Trades
                # We limit scan frequency to avoid yahoo rate limits (every 60 seconds)
                self._scan_markets()
                
                # Sleep
                time.sleep(60) 
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                time.sleep(10)

    def _manage_positions(self):
        for symbol in list(self.positions.keys()):
            try:
                # Get Live Price
                df = yf.download(symbol, period="1d", interval="1m", progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                curr_price = df['Close'].iloc[-1]
                pos = self.positions[symbol]
                
                # Calculate PnL
                pnl_pct = (curr_price - pos['entry']) / pos['entry']
                
                # Dynamic Exit Rules
                take_profit = 0.015  # 1.5%
                stop_loss = -0.01    # 1.0%
                
                if pnl_pct >= take_profit:
                    self._close_position(symbol, curr_price, "AI Target Hit")
                elif pnl_pct <= stop_loss:
                    self._close_position(symbol, curr_price, "AI Stop Loss")
                    
            except Exception as e:
                pass

    def _scan_markets(self):
        if len(self.positions) >= 3: return # Max 3 concurrent trades
        
        # Pick random assets to check (to spread load)
        check_list = random.sample(ALL_TICKERS, 4)
        
        for symbol in check_list:
            if symbol in self.positions: continue
            
            # Fetch Data
            df = yf.download(symbol, period="5d", interval="15m", progress=False)
            if len(df) < 20: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Prepare AI Features
            df = compute_indicators(df)
            
            # AI PREDICTION
            ai_confidence = self.ai.predict(df)
            
            # Execution Logic
            # If AI is > 70% sure it's a buy
            if ai_confidence > 0.70:
                self._execute_trade(symbol, df['Close'].iloc[-1], "BUY", ai_confidence)
            
            time.sleep(2) # Polite delay

    def _execute_trade(self, symbol, price, side, confidence):
        qty = 1000 / price # $1000 per trade
        self.balance -= 1000
        
        self.positions[symbol] = {
            "entry": price, "qty": qty, "time": datetime.now(), "conf": confidence
        }
        
        log_entry = {
            "Time": datetime.now().strftime("%H:%M"),
            "Symbol": symbol,
            "Action": "BUY",
            "Price": price,
            "AI_Conf": f"{confidence:.0%}"
        }
        self.trade_log.insert(0, log_entry)
        logger.info(f"AI TRADED: {symbol} @ {price}")

    def _close_position(self, symbol, price, reason):
        pos = self.positions[symbol]
        revenue = pos['qty'] * price
        pnl = revenue - 1000
        
        self.balance += revenue
        del self.positions[symbol]
        
        log_entry = {
            "Time": datetime.now().strftime("%H:%M"),
            "Symbol": symbol,
            "Action": "SELL",
            "Price": price,
            "Reason": reason,
            "PnL": f"${pnl:.2f}"
        }
        self.trade_log.insert(0, log_entry)

# ================= UI & APP STATE =================
if 'system' not in st.session_state:
    st.session_state.system = AutonomousSystem()

sys_core = st.session_state.system

# Auto Refresh UI every 5 seconds
st_autorefresh(interval=5000, key="ui_refresh")

# --- UI LAYOUT ---
st.title("🤖 Autonomous AI Trader (Free Mode)")
st.markdown("Running **Local Random Forest AI** • No API Keys • Paper Trading")

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bot Status", "ACTIVE 🟢" if sys_core.active else "IDLE 🔴")
m2.metric("AI Accuracy (Backtest)", f"{sys_core.ai.accuracy:.1%}" if sys_core.ai.is_trained else "Pending")
m3.metric("Paper Balance", f"${sys_core.balance:,.2f}")
m4.metric("Active Trades", len(sys_core.positions))

# Controls
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("▶ START AUTONOMOUS MODE"):
        if not sys_core.active:
            with st.spinner("Training AI on history..."):
                sys_core.start_autonomous_loop()
            st.success("Bot Started! It is now scanning markets.")
            time.sleep(1)
            st.rerun()

with c2:
    if st.button("⏹ STOP BOT"):
        sys_core.stop_autonomous_loop()
        st.rerun()

with c3:
    if st.button("🧠 Retrain AI Model"):
        with st.spinner("Fetching data and retraining..."):
            s, m = sys_core.ai.train_model(ALL_TICKERS)
            st.info(m)

# Main Dashboard
tab1, tab2 = st.tabs(["📊 Live Positions & Logs", "🧠 AI Brain View"])

with tab1:
    # Active Positions
    st.subheader("Active Holdings")
    if sys_core.positions:
        pos_data = []
        for sym, data in sys_core.positions.items():
            # Get current price quickly for UI
            curr = data['entry'] # Placeholder for speed, real logic in background
            pnl_tracker = (curr - data['entry']) * data['qty']
            pos_data.append({
                "Asset": sym,
                "Entry Price": f"{data['entry']:.4f}",
                "AI Confidence": f"{data['conf']:.0%}",
                "Est PnL": "Calculating..."
            })
        st.table(pos_data)
    else:
        st.info("No active trades. AI is scanning...")

    # Logs
    st.subheader("Transaction Log")
    if sys_core.trade_log:
        st.dataframe(pd.DataFrame(sys_core.trade_log))

with tab2:
    st.write("### How the Free AI works")
    st.write("""
    1. **Data Collection:** The bot pulls the last 30 days of hourly data for Crypto/Forex.
    2. **Feature Engineering:** It calculates RSI, EMA trends, and Volatility locally.
    3. **Training:** It uses a `RandomForestClassifier` to find patterns (e.g., "When RSI < 30 and Trend is up, price usually rises").
    4. **Inference:** Every minute, it feeds live data to this model. If the model says "70% chance of Up", it buys.
    """)
    
    # Visualize Asset Data
    asset_view = st.selectbox("Inspect Asset Data", ALL_TICKERS)
    if st.button("Analyze Asset"):
        d = yf.download(asset_view, period="5d", interval="1h", progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d = compute_indicators(d)
        
        st.line_chart(d['Close'])
        
        # Show what the AI sees
        latest = d.iloc[[-1]]
        st.write("Current AI Features:")
        st.json({
            "RSI": round(latest['RSI'].values[0], 2),
            "Trend_Strength": round(latest['Feat_Trend'].values[0], 5),
            "Volatility": round(latest['Feat_Vol'].values[0], 5)
        })
        
        prob = sys_core.ai.predict(d)
        st.metric("AI Buy Probability", f"{prob:.1%}")
