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

# ==========================================
# 1. CORE ANALYTICS & RISK MODULE
# ==========================================
def compute_indicators(df):
    """Calculates technical features and volatility-based risk levels."""
    df = df.copy()
    
    # Technical Indicators
    df['Returns'] = df['Close'].pct_change()
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR for Volatility-Adjusted SL/TP
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    # AI Features
    df['Trend_Signal'] = (df['EMA_8'] - df['EMA_21']) / df['Close']
    df['Momentum_Signal'] = df['RSI'] / 100.0
    df['Volatility_Signal'] = df['Returns'].rolling(10).std()
    
    # Target: 1 if next price is higher
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

# ==========================================
# 2. AUTONOMOUS TRADING ENGINE
# ==========================================
class AutonomousSystem:
    def __init__(self):
        self.active = False
        self.balance = 100000.0
        self.positions = {}
        self.history = []
        self.logs = []
        self.model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        self.is_trained = False
        self.feature_cols = ['Trend_Signal', 'Momentum_Signal', 'Volatility_Signal']
        self.tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "EURUSD=X", "GBPUSD=X", "GC=F", "CL=F"]

    def add_log(self, msg):
        t = datetime.now().strftime('%H:%M:%S')
        self.logs.insert(0, f"[{t}] {msg}")

    def train_ai(self):
        self.add_log("Training AI on global market history...")
        data_list = []
        for t in self.tickers:
            d = yf.download(t, period="1mo", interval="1h", progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                data_list.append(compute_indicators(d))
        
        if data_list:
            full_df = pd.concat(data_list)
            self.model.fit(full_df[self.feature_cols], full_df['Target'])
            self.is_trained = True
            self.add_log("AI Brain optimized and ready.")
            return True
        return False

    def scan_markets(self, threshold, max_pos):
        if not self.active or len(self.positions) >= max_pos: return
        
        symbol = random.choice(self.tickers)
        if symbol in self.positions: return
        
        df = yf.download(symbol, period="5d", interval="15m", progress=False)
        if len(df) < 30: return
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = compute_indicators(df)
        latest = df.iloc[[-1]]
        prob = self.model.predict_proba(latest[self.feature_cols])[0][1]
        
        if prob >= threshold:
            price = latest['Close'].values[0]
            atr = latest['ATR'].values[0]
            
            # Risk Management: 2:1 Reward/Risk Ratio
            sl = price - (atr * 1.5)
            tp = price + (atr * 3.0)
            
            self.positions[symbol] = {
                'entry': price, 'sl': sl, 'tp': tp, 
                'qty': 1000 / price, 'prob': prob, 'time': datetime.now()
            }
            self.add_log(f"BUY {symbol} @ {price:.2f} (Conf: {prob:.1%})")

# ==========================================
# 3. TERMINAL UI (STREAMLIT)
# ==========================================
st.set_page_config(page_title="AI TRADING TERMINAL", layout="wide")

# Custom CSS for Terminal Look
st.markdown("""<style>
    .stMetric { background-color: #0E1117; border: 1px solid #30363D; padding: 15px; border-radius: 10px; }
    code { color: #00FF41 !important; }
</style>""", unsafe_allow_html=True)

if 'bot' not in st.session_state:
    st.session_state.bot = AutonomousSystem()

bot = st.session_state.bot
st_autorefresh(interval=5000, key="terminal_refresh")

# Header Metrics
wins = len([x for x in bot.history if x.get('PnL', 0) > 0])
wr = f"{(wins/len(bot.history)*100):.1f}%" if bot.history else "0%"

st.title("📟 AI AUTONOMOUS TERMINAL v3.3")
m1, m2, m3, m4 = st.columns(4)
m1.metric("EQUITY", f"${bot.balance:,.2f}")
m2.metric("ACTIVE TRADES", len(bot.positions))
m3.metric("WIN RATE", wr)
m4.metric("BOT STATUS", "ONLINE" if bot.active else "OFFLINE")

# Main Interface Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 DASHBOARD", "🧠 AI BRAIN", "📜 HISTORY", "⚙️ SETTINGS"])

with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("System Logs")
        st.code("\n".join(bot.logs[:12]) if bot.logs else "Kernel standby...", language="bash")
        
        # Simulated Auto-Scanner for UI
        if bot.active:
            bot.scan_markets(st.session_state.get('conf_t', 0.7), st.session_state.get('max_p', 3))

    with col_r:
        st.subheader("Open Positions")
        if bot.positions:
            for s, d in bot.positions.items():
                st.info(f"**{s}**\n\nEntry: {d['entry']:.2f} | TP: {d['tp']:.2f}")
        else:
            st.write("No active exposure.")

with tab2:
    st.subheader("Feature Importance Analysis")
    if bot.is_trained:
        # Visualizing weights
        fig = go.Figure(go.Bar(
            x=['Trend Strength', 'Momentum (RSI)', 'Volatility'],
            y=bot.model.feature_importances_,
            marker_color='#00FF41'
        ))
        fig.update_layout(template="plotly_dark", height=350, yaxis_title="Weight")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("AI model not trained. Start the bot to see weights.")

with tab3:
    st.subheader("Trade History Ledger")
    if bot.history:
        st.dataframe(pd.DataFrame(bot.history), use_container_width=True)
    else:
        st.info("No realized trades in current session.")

with tab4:
    st.subheader("Real-Time Logic Tuning")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.conf_t = st.slider("Min Confidence", 0.50, 0.95, 0.70, 0.05)
    with c2:
        st.session_state.max_p = st.slider("Max Open Trades", 1, 10, 3)
    with c3:
        if st.button("▶️ START BOT", use_container_width=True):
            bot.active = True
            bot.train_ai()
            st.rerun()
        if st.button("⏹️ STOP BOT", use_container_width=True):
            bot.active = False
            st.rerun()
