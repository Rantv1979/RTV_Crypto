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

# --- UI CONFIG ---
st.set_page_config(page_title="AI TERMINAL v3.0", layout="wide")

# --- RISK MANAGEMENT MODULE ---
def calculate_atr(df, period=14):
    """Calculates Average True Range to measure market volatility."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

# --- THE SYSTEM ---
class ProfessionalTrader:
    def __init__(self):
        self.active = False
        self.balance = 100000.0
        self.positions = {}  # Active trades
        self.history = []    # Closed trades archive
        self.logs = []
        self.model = RandomForestClassifier(n_estimators=100)
        self.trained = False

    def add_log(self, msg):
        self.logs.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def execute_trade(self, symbol, price, df, prob):
        # VOLATILITY BASED RISK (ATR)
        atr = calculate_atr(df).iloc[-1]
        
        # Professional Risk: TP = 3x ATR, SL = 1.5x ATR (2:1 Reward/Risk)
        stop_loss = price - (atr * 1.5)
        take_profit = price + (atr * 3)
        
        self.positions[symbol] = {
            'entry': price,
            'sl': stop_loss,
            'tp': take_profit,
            'qty': 1000 / price,
            'time': datetime.now(),
            'confidence': prob
        }
        self.add_log(f"🚀 LONG {symbol} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

    def close_trade(self, symbol, current_price, reason):
        pos = self.positions.pop(symbol)
        pnl = (current_price - pos['entry']) * pos['qty']
        self.balance += (1000 + pnl)
        
        record = {
            "Symbol": symbol,
            "Entry": f"{pos['entry']:.2f}",
            "Exit": f"{current_price:.2f}",
            "PnL": round(pnl, 2),
            "Reason": reason,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.history.insert(0, record)
        self.add_log(f"🔒 CLOSED {symbol} @ {current_price:.2f} | PnL: ${pnl:.2f} ({reason})")

# --- ENGINE LOGIC ---
if 'bot' not in st.session_state:
    st.session_state.bot = ProfessionalTrader()

bot = st.session_state.bot
st_autorefresh(interval=5000, key="auto_update")

# --- UI LAYOUT ---
st.title("📟 AI AUTONOMOUS TERMINAL v3.0")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("ACCOUNT EQUITY", f"${bot.balance:,.2f}")
m2.metric("ACTIVE TRADES", len(bot.positions))
m3.metric("BOT STATUS", "ONLINE" if bot.active else "OFFLINE")
m4.metric("WIN RATE", f"{len([x for x in bot.history if x['PnL'] > 0]) / len(bot.history) * 100:.1f}%" if bot.history else "0%")

# Control Center
tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "📜 TRADING HISTORY", "⚙️ SYSTEM SETTINGS"])

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Live Market Analysis")
        # Trading Logic Trigger (Simplified for Demo)
        if bot.active and len(bot.positions) < 3:
            # Here you would call your scanning logic
            pass
            
        st.info("AI is currently monitoring 11 global assets for volatility breakouts.")
        st.code("\n".join(bot.logs[:10]), language="bash")

    with col_right:
        st.subheader("Active Positions")
        if bot.positions:
            for sym, d in bot.positions.items():
                with st.container():
                    st.markdown(f"**{sym}**")
                    st.caption(f"Entry: {d['entry']:.4f} → TP: {d['tp']:.4f}")
                    st.progress(0.65) # Placeholder for price proximity
        else:
            st.write("No active exposure.")

with tab2:
    st.subheader("Closed Trade Archive")
    if bot.history:
        history_df = pd.DataFrame(bot.history)
        st.dataframe(history_df, use_container_width=True)
        
        # Mini Stats
        total_pnl = sum([x['PnL'] for x in bot.history])
        st.write(f"**Total Realized PnL:** :green[+${total_pnl:.2f}]" if total_pnl > 0 else f"**Total Realized PnL:** :red[${total_pnl:.2f}]")
    else:
        st.info("No trades archived yet.")

with tab3:
    if st.button("▶️ ACTIVATE TERMINAL"): bot.active = True
    if st.button("⏹️ DEACTIVATE TERMINAL"): bot.active = False
    st.slider("Risk Per Trade ($)", 500, 5000, 1000)
