import os
import time
import threading
import sys
import logging
import random
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# TERMINAL STYLING (The "Attractive UI")
# ==========================================
st.set_page_config(page_title="AI TERMINAL v2.0", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #00FF41; font-family: 'Courier New', Courier, monospace; }
    .stMetric { background-color: #161B22; border: 1px solid #30363D; padding: 10px; border-radius: 5px; }
    .stButton>button { width: 100%; background-color: #21262D; color: #58A6FF; border: 1px solid #30363D; }
    .stButton>button:hover { border-color: #00FF41; color: #00FF41; }
    .trade-log { font-size: 12px; color: #8B949E; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CORE TRADING ENGINE
# ==========================================

ASSETS = {
    "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
    "FOREX": ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"],
    "COMMODITIES": ["GC=F", "CL=F", "SI=F"]
}
ALL_TICKERS = [item for sublist in ASSETS.values() for item in sublist]

def compute_indicators(df):
    df = df.copy()
    # Technical Indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['Std_20'] = df['Close'].rolling(20).std()
    df['Upper'] = df['SMA_20'] + (df['Std_20'] * 2)
    df['Lower'] = df['SMA_20'] - (df['Std_20'] * 2)
    
    # AI Features
    df['Feat_BB_Pos'] = (df['Close'] - df['Lower']) / (df['Upper'] - df['Lower'])
    df['Feat_Momentum'] = df['Close'].pct_change(5)
    df['Feat_Vol'] = df['Returns'].rolling(10).std()
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

class AI_Brain:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=150, max_depth=7)
        self.is_trained = False
        self.accuracy = 0.0

    def train(self):
        data_list = []
        for ticker in ALL_TICKERS[:6]: # Sample for speed
            d = yf.download(ticker, period="1mo", interval="1h", progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                data_list.append(compute_indicators(d))
        
        if not data_list: return False
        
        full_df = pd.concat(data_list)
        features = ['Feat_BB_Pos', 'Feat_Momentum', 'Feat_Vol']
        X = full_df[features]
        y = full_df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
        self.is_trained = True
        return True

class TerminalSystem:
    def __init__(self):
        self.brain = AI_Brain()
        self.active = False
        self.balance = 100000.0
        self.positions = {}
        self.logs = []
        self.start_time = datetime.now()

    def add_log(self, msg, type="INFO"):
        t = datetime.now().strftime("%H:%M:%S")
        self.logs.insert(0, f"[{t}] {type}: {msg}")

    def scan_and_trade(self):
        if not self.active: return
        
        ticker = random.choice(ALL_TICKERS)
        if ticker in self.positions: return
        
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if len(df) < 30: return
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = compute_indicators(df)
        feat = df[['Feat_BB_Pos', 'Feat_Momentum', 'Feat_Vol']].iloc[[-1]]
        prob = self.brain.model.predict_proba(feat)[0][1]
        
        if prob > 0.65: # Entry threshold
            price = df['Close'].iloc[-1]
            self.positions[ticker] = {'entry': price, 'prob': prob, 'time': t.now()}
            self.add_log(f"EXECUTED LONG: {ticker} @ {price:.2f} (Conf: {prob:.1%})", "BUY")

# Initialize System in Session State
if 'bot' not in st.session_state:
    st.session_state.bot = TerminalSystem()
    st.session_state.bot.brain.train()

bot = st.session_state.bot
st_autorefresh(interval=3000, key="bot_loop")

# ==========================================
# UI RENDERING
# ==========================================

# Header
c1, c2, c3 = st.columns([2, 3, 2])
with c1:
    st.markdown(f"### ⚡ TERMINAL v2.0\n`SYSTEM STATUS: {'RUNNING' if bot.active else 'IDLE'}`")
with c2:
    if st.button("TOGGLE AUTONOMOUS MODE"):
        bot.active = not bot.active
        bot.add_log(f"System State Changed to: {bot.active}")
with c3:
    st.markdown(f"**Uptime:** {str(datetime.now() - bot.start_time).split('.')[0]}")

st.markdown("---")

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("EQUITY (USD)", f"${bot.balance:,.2f}", "+0.2%")
m2.metric("AI WIN RATE", f"{bot.brain.accuracy:.1%}")
m3.metric("OPEN POSITIONS", len(bot.positions))
m4.metric("MARKET LOAD", "OPTIMAL")

# Main Content
col_main, col_side = st.columns([2, 1])

with col_main:
    # Logic to simulate live trades for the UI demo
    if bot.active:
        bot.scan_and_trade()

    # Live Chart of a major pair
    st.markdown("#### 📈 MARKET OVERVIEW")
    chart_ticker = st.selectbox("Select Watchlist Asset", ALL_TICKERS)
    c_data = yf.download(chart_ticker, period="1d", interval="15m", progress=False)
    if isinstance(c_data.columns, pd.MultiIndex): c_data.columns = c_data.columns.get_level_values(0)
    
    fig = go.Figure(data=[go.Candlestick(x=c_data.index,
                open=c_data['Open'], high=c_data['High'],
                low=c_data['Low'], close=c_data['Close'],
                increasing_line_color='#00FF41', decreasing_line_color='#FF4B4B')])
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown("#### 📜 TERMINAL LOGS")
    log_box = "\n".join(bot.logs[:15])
    st.code(log_box if log_box else "Awaiting market signal...", language="bash")
    
    st.markdown("#### 💼 ACTIVE TRADES")
    if bot.positions:
        for t, d in bot.positions.items():
            st.markdown(f"`{t}`: Entry **{d['entry']:.2f}** | Conf: **{d['prob']:.0%}**")
    else:
        st.write("Scanning for opportunities...")

# Background processing (The Loop)
if bot.active:
    time.sleep(1) # Prevent CPU thrashing
