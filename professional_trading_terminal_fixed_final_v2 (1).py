import os
import time
import threading
import random
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. ENHANCED INDICATOR ENGINE
# ==========================================
def compute_indicators(df):
    df = df.copy()
    # Technical Indicators
    df['Returns'] = df['Close'].pct_change()
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR for Risk Management
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    # AI Features (Normalized)
    df['Trend_Signal'] = (df['EMA_8'] - df['EMA_21']) / df['Close']
    df['Momentum_Signal'] = df['RSI'] / 100.0
    df['Volatility_Signal'] = df['Returns'].rolling(10).std()
    
    # Target: 1 if next candle close is higher
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

# ==========================================
# 2. PROFESSIONAL TRADER CLASS
# ==========================================
class AutonomousSystem:
    def __init__(self):
        self.active = False
        self.balance = 100000.0
        self.positions = {}
        self.history = []
        self.logs = []
        self.model = RandomForestClassifier(n_estimators=100, max_depth=7)
        self.is_trained = False
        self.feature_cols = ['Trend_Signal', 'Momentum_Signal', 'Volatility_Signal']

    def train_ai(self, tickers):
        data_list = []
        for t in tickers[:5]:
            d = yf.download(t, period="1mo", interval="1h", progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                data_list.append(compute_indicators(d))
        
        if data_list:
            full_df = pd.concat(data_list)
            X = full_df[self.feature_cols]
            y = full_df['Target']
            self.model.fit(X, y)
            self.is_trained = True
            return True
        return False

# ==========================================
# 3. UI & VISUALIZATION
# ==========================================
st.set_page_config(page_title="AI TERMINAL v3.2", layout="wide")

if 'bot' not in st.session_state:
    st.session_state.bot = AutonomousSystem()

bot = st.session_state.bot
st_autorefresh(interval=5000, key="bot_update")

# Metric Calculations
wins = len([x for x in bot.history if x.get('PnL', 0) > 0])
win_rate = f"{(wins/len(bot.history)*100):.1f}%" if bot.history else "0%"

# Header Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("EQUITY", f"${bot.balance:,.2f}")
m2.metric("ACTIVE", len(bot.positions))
m3.metric("WIN RATE", win_rate)
m4.metric("STATUS", "ONLINE" if bot.active else "OFFLINE")

tab1, tab2, tab3 = st.tabs(["📊 Terminal", "🧠 AI Brain", "📜 History"])

with tab1:
    st.subheader("Live Market Scanner")
    st.code("\n".join(bot.logs[:5]) if bot.logs else "System standby...", language="bash")
    # Candlestick chart logic here...

with tab2:
    st.subheader("AI Feature Weighting")
    if bot.is_trained:
        # Generate the Importance Chart
        importances = bot.model.feature_importances_
        fig = go.Figure(go.Bar(
            x=['Trend Strength', 'Momentum (RSI)', 'Volatility'],
            y=importances,
            marker_color='#00FF41'
        ))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Train the model to see indicator weights.")

with tab3:
    st.subheader("Trade Ledger")
    if bot.history:
        st.table(pd.DataFrame(bot.history))
    else:
        st.write("No closed trades in this session.")

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    if st.button("START BOT"): 
        bot.active = True
        bot.train_ai(["BTC-USD", "ETH-USD", "EURUSD=X"])
    if st.button("STOP BOT"): bot.active = False
