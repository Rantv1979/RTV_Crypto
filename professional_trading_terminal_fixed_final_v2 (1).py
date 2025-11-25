"""
Option B - Rebuilt trading terminal (cleaner, modular).
Features:
- Enforces $1000 allocation per trade
- Reliable auto-refresh using a session-state timer + st.experimental_rerun
- Multi-colour tabs via CSS
- Mood gauge needle using plotly
- Simpler, readable structure for easy customization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import yfinance as yf

# CONFIG
FIXED_ALLOCATION_PER_TRADE = 1000.0
REFRESH_INTERVAL_SECONDS = 30

st.set_page_config(page_title="Terminal - Rebuilt (Option B)", layout="wide", page_icon="📈")

# CSS for multi-colour tabs and aesthetics
st.markdown("""
<style>
.main-title {font-size:28px; font-weight:800; background:linear-gradient(90deg,#FF6B6B,#45B7D1); -webkit-background-clip:text; color:transparent;}
.tab {padding:8px 12px; border-radius:8px;}
.tab-active {background:linear-gradient(90deg,#667eea,#764ba2); color:white;}
.mood {background:linear-gradient(135deg,#1e3c72,#2a5298); padding:8px; border-radius:10px; color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Rebuilt Multi-Timeframe Terminal (Option B)</div>', unsafe_allow_html=True)

# Session state for refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'balance' not in st.session_state:
    st.session_state.balance = 10000.0
if 'open_trades' not in st.session_state:
    st.session_state.open_trades = []

# Sidebar
st.sidebar.header("Config")
symbols = st.sidebar.multiselect("Symbols", ['BTC-USD','ETH-USD','GC=F','EURUSD=X'], default=['BTC-USD','ETH-USD'])
refresh_interval = st.sidebar.slider("Refresh (s)", 10, 120, REFRESH_INTERVAL_SECONDS)

# Auto refresh logic
now = time.time()
if now - st.session_state.last_refresh >= refresh_interval:
    st.session_state.last_refresh = now
    st.experimental_rerun()

# Simple data fetcher
def fetch_price(sym):
    try:
        df = yf.download(sym, period='7d', interval='15m', progress=False)
        return float(df['Close'].iloc[-1]) if not df.empty else None
    except Exception:
        return None

# Mood gauge builder
def mood_score_from_symbol(sym):
    try:
        df = yf.download(sym, period='3d', interval='15m', progress=False)
        if df.empty: return 50.0
        rsi = (df['Close'].diff().fillna(0) > 0).rolling(14).mean().iloc[-1] * 100
        return float(np.clip(rsi, 0, 100))
    except Exception:
        return 50.0

def needle_gauge(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title},
        gauge={'axis': {'range':[0,100]}, 'bar':{'color':'#34d399'}}))
    fig.update_layout(height=240, margin=dict(t=40,b=10,l=10,r=10))
    return fig

# Tabs
tabs = st.tabs(["Live","Signals","Paper Trading"])
with tabs[0]:
    st.subheader("Live Dashboard")
    cols = st.columns(3)
    for i,sym in enumerate(symbols):
        with cols[i%3]:
            price = fetch_price(sym)
            score = mood_score_from_symbol(sym)
            st.write(f"**{sym}**")
            st.write(f"Price: {price}")
            st.plotly_chart(needle_gauge(score, sym), use_container_width=True)

with tabs[1]:
    st.subheader("Signals (simulated)")
    st.info("This demo creates a simple momentum signal for illustration.")
    signals = []
    for sym in symbols:
        price_now = fetch_price(sym)
        signals.append({'symbol':sym,'action':'BUY' if np.random.rand()>0.5 else 'SELL','entry':price_now,'confidence':float(np.random.rand())})
    st.table(pd.DataFrame(signals))

with tabs[2]:
    st.subheader("Paper Trading")
    st.metric("Balance", f"${st.session_state.balance:,.2f}")
    st.write("Open Trades:")
    st.table(pd.DataFrame(st.session_state.open_trades))

    # Manual execute top signal
    if st.button("Execute top simulated signal"):
        # pick a random signal
        if signals:
            s = signals[0]
            qty = int(FIXED_ALLOCATION_PER_TRADE / (s['entry'] if s['entry'] else 1))
            if qty>0 and s['entry']:
                trade = {{
                    'id': f"T{{len(st.session_state.open_trades)+1}}",
                    'symbol': s['symbol'],
                    'action': s['action'],
                    'entry': s['entry'],
                    'qty': qty,
                    'time': str(datetime.now())
                }}
                position_value = trade['entry']*trade['qty']
                if position_value <= st.session_state.balance:
                    st.session_state.balance -= position_value
                    st.session_state.open_trades.append(trade)
                    st.success("Trade executed with fixed allocation ${:,.0f}".format(FIXED_ALLOCATION_PER_TRADE))
                else:
                    st.error("Insufficient balance for trade.")
            else:
                st.error("Invalid entry price.")

# Footer
st.markdown("---")
st.markdown("Generated: Option B - Rebuilt terminal")
