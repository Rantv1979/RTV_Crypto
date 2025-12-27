# app.py - RTV SMC Intraday Algorithmic Trading Terminal Pro
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time as tm
import threading
from queue import Queue
import json
import os
import schedule
import asyncio
from typing import Dict, List, Optional, Tuple, Set
import random
import traceback
from collections import defaultdict
import pytz
warnings.filterwarnings('ignore')

# Page configuration for professional trading terminal
st.set_page_config(
    page_title="RTV SMC Pro Trading Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Trading Terminal CSS with auto-refresh indicators
st.markdown("""
<style>
    /* Main Theme */
    :root {
        --primary-bg: #0a0e17;
        --secondary-bg: #141b2d;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-purple: #8b5cf6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #334155;
    }
    
    /* Main Container */
    .stApp {
        background: var(--primary-bg);
        color: var(--text-primary);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #60a5fa, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 1.5px;
        margin: 0;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .terminal-tag {
        background: linear-gradient(90deg, #10b981, #3b82f6);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 8px;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    /* Auto-refresh indicator */
    .refresh-indicator {
        position: absolute;
        top: 15px;
        right: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(59, 130, 246, 0.1);
        padding: 6px 12px;
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .refresh-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        margin-bottom: 25px;
    }
    
    @media (max-width: 1200px) {
        .metric-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    .metric-card {
        background: linear-gradient(145deg, var(--secondary-bg), #1a2238);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
        border-color: var(--accent-blue);
    }
    
    .metric-title {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-value {
        color: var(--text-primary);
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        margin-bottom: 5px;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .change-positive {
        color: var(--accent-green);
    }
    
    .change-negative {
        color: var(--accent-red);
    }
    
    /* Signal Cards */
    .signal-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    
    .signal-card {
        background: linear-gradient(145deg, var(--secondary-bg), #1a2238);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    
    .signal-card.buy::before {
        background: linear-gradient(90deg, var(--accent-green), #059669);
    }
    
    .signal-card.sell::before {
        background: linear-gradient(90deg, var(--accent-red), #dc2626);
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .signal-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .confidence-badge {
        background: linear-gradient(135deg, var(--accent-yellow), #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .signal-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin: 15px 0;
    }
    
    .signal-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .signal-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    
    .signal-value {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Courier New', monospace;
    }
    
    .execute-btn {
        width: 100%;
        padding: 12px;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    
    .execute-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--secondary-bg);
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        color: var(--text-secondary);
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: var(--accent-blue);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white !important;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Trade History */
    .trade-log {
        background: var(--secondary-bg);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid var(--border-color);
    }
    
    .trade-entry {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 4px solid var(--accent-blue);
        transition: all 0.3s ease;
    }
    
    .trade-entry:hover {
        background: rgba(59, 130, 246, 0.1);
        transform: translateX(5px);
    }
    
    .trade-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .trade-asset {
        font-weight: 700;
        color: var(--text-primary);
        font-size: 1.1rem;
    }
    
    .trade-pnl {
        font-weight: 700;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
    }
    
    .pnl-positive {
        color: var(--accent-green);
    }
    
    .pnl-negative {
        color: var(--accent-red);
    }
    
    .trade-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Auto-refresh Controls */
    .refresh-controls {
        background: var(--secondary-bg);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin-bottom: 20px;
    }
    
    .last-update {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 10px;
        text-align: center;
    }
    
    /* Loading Spinner */
    .spinner {
        border: 3px solid rgba(59, 130, 246, 0.1);
        border-top: 3px solid var(--accent-blue);
        border-radius: 50%;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
        display: inline-block;
        vertical-align: middle;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: var(--secondary-bg);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-blue);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-purple);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with comprehensive data structure
if 'paper_portfolio' not in st.session_state:
    st.session_state.paper_portfolio = {
        'balance': 50000.00,
        'positions': {},
        'trade_history': [],
        'total_pnl': 0.00,
        'daily_pnl': 0.00,
        'equity_curve': [50000.00],
        'winning_trades': 0,
        'losing_trades': 0,
        'max_drawdown': 0.00,
        'peak_equity': 50000.00
    }

if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

if 'active_signals' not in st.session_state:
    st.session_state.active_signals = {}

if 'traded_symbols' not in st.session_state:
    st.session_state.traded_symbols = set()

if 'market_data' not in st.session_state:
    st.session_state.market_data = {}

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 30  # seconds

# Enhanced Asset Configuration for Intraday Trading
ASSET_CONFIG = {
    "Cryptocurrencies": {
        "BTC/USD": {"symbol": "BTC-USD", "pip_size": 0.01, "lot_size": 0.001, "sector": "Crypto", "volatility": "High"},
        "ETH/USD": {"symbol": "ETH-USD", "pip_size": 0.01, "lot_size": 0.01, "sector": "Crypto", "volatility": "High"},
        "SOL/USD": {"symbol": "SOL-USD", "pip_size": 0.001, "lot_size": 0.1, "sector": "Crypto", "volatility": "Very High"},
        "XRP/USD": {"symbol": "XRP-USD", "pip_size": 0.0001, "lot_size": 1000, "sector": "Crypto", "volatility": "High"},
        "ADA/USD": {"symbol": "ADA-USD", "pip_size": 0.0001, "lot_size": 1000, "sector": "Crypto", "volatility": "High"},
    },
    "Forex": {
        "EUR/USD": {"symbol": "EURUSD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex", "volatility": "Low"},
        "GBP/USD": {"symbol": "GBPUSD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex", "volatility": "Medium"},
        "USD/JPY": {"symbol": "JPY=X", "pip_size": 0.01, "lot_size": 10000, "sector": "Forex", "volatility": "Medium"},
        "AUD/USD": {"symbol": "AUDUSD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex", "volatility": "Medium"},
        "USD/CAD": {"symbol": "CAD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex", "volatility": "Low"},
    },
    "Commodities": {
        "Gold": {"symbol": "GC=F", "pip_size": 0.10, "lot_size": 1, "sector": "Commodities", "volatility": "Medium"},
        "Silver": {"symbol": "SI=F", "pip_size": 0.01, "lot_size": 10, "sector": "Commodities", "volatility": "High"},
        "Crude Oil": {"symbol": "CL=F", "pip_size": 0.01, "lot_size": 10, "sector": "Commodities", "volatility": "High"},
        "Natural Gas": {"symbol": "NG=F", "pip_size": 0.001, "lot_size": 100, "sector": "Commodities", "volatility": "Very High"},
    },
    "Indices": {
        "S&P 500": {"symbol": "^GSPC", "pip_size": 0.25, "lot_size": 1, "sector": "Indices", "volatility": "Medium"},
        "NASDAQ": {"symbol": "^IXIC", "pip_size": 0.25, "lot_size": 1, "sector": "Indices", "volatility": "High"},
        "Dow Jones": {"symbol": "^DJI", "pip_size": 1.0, "lot_size": 1, "sector": "Indices", "volatility": "Medium"},
        "Russell 2000": {"symbol": "^RUT", "pip_size": 0.10, "lot_size": 1, "sector": "Indices", "volatility": "High"},
    },
    "Tech Stocks": {
        "Apple": {"symbol": "AAPL", "pip_size": 0.01, "lot_size": 10, "sector": "Tech", "volatility": "Medium"},
        "Microsoft": {"symbol": "MSFT", "pip_size": 0.01, "lot_size": 10, "sector": "Tech", "volatility": "Medium"},
        "Tesla": {"symbol": "TSLA", "pip_size": 0.01, "lot_size": 10, "sector": "Tech", "volatility": "Very High"},
        "NVIDIA": {"symbol": "NVDA", "pip_size": 0.01, "lot_size": 10, "sector": "Tech", "volatility": "Very High"},
        "Amazon": {"symbol": "AMZN", "pip_size": 0.01, "lot_size": 10, "sector": "Tech", "volatility": "Medium"},
    }
}

# Enhanced Trading Strategies
TRADING_STRATEGIES = {
    "SMC Pro": {
        "description": "Smart Money Concepts with multi-timeframe confirmation",
        "indicators": ["FVG", "Order Blocks", "Market Structure", "Liquidity", "RSI", "MACD"],
        "timeframes": ["5m", "15m", "1h"],
        "confidence_threshold": 0.75
    },
    "Momentum Breakout": {
        "description": "Breakout strategy with volume confirmation",
        "indicators": ["Bollinger Bands", "Volume", "ATR", "RSI"],
        "timeframes": ["5m", "15m"],
        "confidence_threshold": 0.70
    },
    "Mean Reversion": {
        "description": "Mean reversion with RSI extremes",
        "indicators": ["RSI", "Bollinger Bands", "Moving Averages"],
        "timeframes": ["5m", "15m"],
        "confidence_threshold": 0.80
    },
    "Trend Following": {
        "description": "Follow the trend with multiple MA confirmation",
        "indicators": ["EMA 9", "EMA 21", "EMA 50", "MACD"],
        "timeframes": ["15m", "1h"],
        "confidence_threshold": 0.65
    }
}

# Terminal Header with Auto-refresh Indicator
st.markdown(f"""
<div class="main-header">
    <h1 class="main-title">🚀 RTV SMC ALGORITHMIC TRADING TERMINAL</h1>
    <p style="text-align: center; color: #94a3b8; margin: 10px 0;">Professional Intraday Trading System</p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
        <span class="terminal-tag">💰 Real-time P&L</span>
        <span class="terminal-tag">📊 Multi-Asset</span>
        <span class="terminal-tag">⚡ Auto-Refresh</span>
        <span class="terminal-tag">🤖 AI Signals</span>
    </div>
    <div class="refresh-indicator">
        <div class="refresh-dot"></div>
        <span style="color: #94a3b8; font-size: 0.9rem;">Auto-refresh Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ TRADING CONFIGURATION")

# Operating Mode
mode = st.sidebar.selectbox(
    "Operating Mode",
    ["📡 Live Trading", "🔍 Signal Scanner", "📊 Portfolio", "⚙️ Settings"],
    index=0
)

# Auto-refresh Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 AUTO-REFRESH")

refresh_enabled = st.sidebar.toggle("Enable Auto-refresh", value=True, key="refresh_toggle")
refresh_interval = st.sidebar.selectbox(
    "Refresh Interval",
    ["10 seconds", "30 seconds", "1 minute", "5 minutes", "Manual"],
    index=1
)

if refresh_enabled:
    interval_seconds = {
        "10 seconds": 10,
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300
    }.get(refresh_interval, 30)
    st.session_state.refresh_interval = interval_seconds
    st.session_state.auto_refresh = True
else:
    st.session_state.auto_refresh = False

# Trading Strategy Selection
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 TRADING STRATEGY")

selected_strategy = st.sidebar.selectbox(
    "Primary Strategy",
    list(TRADING_STRATEGIES.keys()),
    index=0
)

strategy_info = TRADING_STRATEGIES[selected_strategy]
st.sidebar.info(f"**{selected_strategy}**: {strategy_info['description']}")

# Asset Selection
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 ASSET SELECTION")

asset_category = st.sidebar.selectbox(
    "Asset Category",
    list(ASSET_CONFIG.keys()),
    index=0
)

selected_asset = st.sidebar.selectbox(
    "Select Asset",
    list(ASSET_CONFIG[asset_category].keys())
)

asset_info = ASSET_CONFIG[asset_category][selected_asset]

# Timeframe
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "30m", "1h"],
    index=1
)

# Risk Management
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ RISK MANAGEMENT")

col1, col2 = st.sidebar.columns(2)
with col1:
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, step=0.1)
    stop_loss_atr = st.slider("Stop Loss (ATR)", 1.0, 3.0, 1.5, step=0.1)
with col2:
    take_profit_atr = st.slider("Take Profit (ATR)", 1.0, 5.0, 2.5, step=0.1)
    max_positions = st.slider("Max Positions", 1, 10, 5)

# Advanced Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ ADVANCED PARAMETERS")

with st.sidebar.expander("SMC Parameters"):
    fvg_period = st.slider("FVG Lookback", 3, 20, 5)
    swing_period = st.slider("Swing Period", 2, 10, 3)
    rsi_period = st.slider("RSI Period", 7, 21, 14)
    atr_period = st.slider("ATR Period", 7, 21, 14)

# Data Fetching with Caching
@st.cache_data(ttl=10)  # Very short TTL for intraday
def fetch_intraday_data(symbol, interval='5m', period='1d'):
    """Fetch intraday market data with error handling"""
    try:
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m',
            '30m': '30m', '1h': '60m'
        }
        
        yf_interval = interval_map.get(interval, '5m')
        
        period_map = {
            '1m': '1d', '5m': '1d', '15m': '5d',
            '30m': '5d', '1h': '5d'
        }
        
        yf_period = period_map.get(interval, '1d')
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yf_period, interval=yf_interval)
        
        if df.empty:
            # Fallback method
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            df = yf.download(symbol, start=start_date, end=end_date, 
                           interval=yf_interval, progress=False, auto_adjust=True)
        
        if not df.empty:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = [col.lower() for col in df.columns]
            df.dropna(inplace=True)
            
            # Add technical indicators
            df = add_enhanced_indicators(df)
            
            return df
            
    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
    
    return pd.DataFrame()

def add_enhanced_indicators(df):
    """Add comprehensive technical indicators"""
    if len(df) < 20:
        return df
    
    # Price calculations
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # ATR with proper handling
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()
    
    # RSI with smoothing
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(loss != 0, gain / loss, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Bollinger Bands with proper column naming
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    
    # Support/Resistance
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
    return df

# Enhanced SMC Algorithm with Multiple Strategies
class AdvancedSMCAlgorithm:
    def __init__(self, strategy="SMC Pro", fvg_lookback=5, swing_period=3):
        self.strategy = strategy
        self.config = TRADING_STRATEGIES.get(strategy, TRADING_STRATEGIES["SMC Pro"])
        self.fvg_lookback = fvg_lookback
        self.swing_period = swing_period
        
    def analyze_market_structure(self, df):
        """Advanced market structure analysis"""
        df = df.copy()
        
        # Identify swing points
        df['swing_high'] = False
        df['swing_low'] = False
        
        for i in range(self.swing_period, len(df)-self.swing_period):
            # Swing High
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'swing_high'] = True
            
            # Swing Low
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'swing_low'] = True
        
        # Determine market structure
        df['market_structure'] = 'neutral'
        swing_highs = df[df['swing_high']]
        swing_lows = df[df['swing_low']]
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for trend
            highs = swing_highs['high'].values[-2:]
            lows = swing_lows['low'].values[-2:]
            
            if len(highs) == 2 and len(lows) == 2:
                if highs[1] > highs[0] and lows[1] > lows[0]:
                    df['market_structure'] = 'uptrend'
                elif highs[1] < highs[0] and lows[1] < lows[0]:
                    df['market_structure'] = 'downtrend'
                elif (highs[1] > highs[0] and lows[1] < lows[0]) or (highs[1] < highs[0] and lows[1] > lows[0]):
                    df['market_structure'] = 'consolidation'
        
        return df
    
    def identify_fvgs(self, df):
        """Identify Fair Value Gaps"""
        df = df.copy()
        
        df['fvg_bullish'] = np.nan
        df['fvg_bearish'] = np.nan
        
        for i in range(1, len(df)-1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG (gap up)
            if current['high'] < next_candle['low']:
                gap_size = next_candle['low'] - current['high']
                if gap_size > (df['atr'].iloc[i] * 0.15):  # Minimum gap threshold
                    df.loc[df.index[i], 'fvg_bullish'] = current['high']
                    df.loc[df.index[i], 'fvg_width'] = gap_size
                    df.loc[df.index[i], 'fvg_strength'] = gap_size / df['atr'].iloc[i]
            
            # Bearish FVG (gap down)
            elif current['low'] > next_candle['high']:
                gap_size = current['low'] - next_candle['high']
                if gap_size > (df['atr'].iloc[i] * 0.15):
                    df.loc[df.index[i], 'fvg_bearish'] = current['low']
                    df.loc[df.index[i], 'fvg_width'] = gap_size
                    df.loc[df.index[i], 'fvg_strength'] = gap_size / df['atr'].iloc[i]
        
        return df
    
    def identify_orderblocks(self, df):
        """Identify Order Blocks"""
        df = df.copy()
        
        df['ob_bullish'] = np.nan
        df['ob_bearish'] = np.nan
        
        for i in range(2, len(df)-2):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            candle_body = abs(current['close'] - current['open'])
            candle_range = current['high'] - current['low']
            
            # Bullish Order Block
            if (candle_body > candle_range * 0.6 and  # Strong body
                current['close'] > current['open'] and  # Bullish candle
                next_candle['low'] >= current['low']):  # Next candle doesn't break low
                
                df.loc[df.index[i], 'ob_bullish'] = current['low']
                df.loc[df.index[i], 'ob_strength'] = candle_body / df['atr'].iloc[i]
            
            # Bearish Order Block
            elif (candle_body > candle_range * 0.6 and  # Strong body
                  current['close'] < current['open'] and  # Bearish candle
                  next_candle['high'] <= current['high']):  # Next candle doesn't break high
                
                df.loc[df.index[i], 'ob_bearish'] = current['high']
                df.loc[df.index[i], 'ob_strength'] = candle_body / df['atr'].iloc[i]
        
        return df
    
    def generate_smc_signals(self, df, asset_info):
        """Generate SMC-based trading signals"""
        signals = []
        
        if len(df) < 30:
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Position sizing
        atr_value = latest['atr'] if not pd.isna(latest['atr']) else df['atr'].mean()
        if atr_value == 0:
            atr_value = df['close'].std() * 0.01
        
        stop_distance = atr_value * stop_loss_atr
        risk_amount = st.session_state.paper_portfolio['balance'] * risk_per_trade / 100
        position_size = risk_amount / stop_distance
        position_size = min(position_size, asset_info['lot_size'] * 5)  # Conservative sizing
        
        # Get FVG data
        fvg_bullish = latest.get('fvg_bullish', np.nan)
        fvg_width = latest.get('fvg_width', 0)
        
        # Signal 1: FVG Fill with RSI Confluence
        if not pd.isna(fvg_bullish):
            fvg_top = fvg_bullish + fvg_width
            fvg_bottom = fvg_bullish
            current_price = latest['close']
            
            # Check if price is filling the FVG
            if fvg_bottom <= current_price <= fvg_top:
                rsi = latest['rsi']
                confidence = 0.65
                
                # Increase confidence with RSI oversold
                if rsi < 35:
                    confidence += 0.15
                
                # Increase confidence with volume
                if latest.get('volume_ratio', 1) > 1.5:
                    confidence += 0.10
                
                # Increase confidence with market structure
                if latest['market_structure'] == 'uptrend':
                    confidence += 0.10
                
                if confidence >= self.config['confidence_threshold']:
                    entry = current_price
                    stop_loss = fvg_bottom - (atr_value * 0.5)
                    take_profit = entry + (atr_value * take_profit_atr)
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'asset_name': selected_asset,
                        'type': 'BUY',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': min(confidence, 0.95),
                        'strategy': 'SMC FVG Fill',
                        'reason': f'Bullish FVG fill with RSI: {rsi:.1f}',
                        'conditions': ['FVG', f'RSI: {rsi:.1f}', 'Volume Confirmation'],
                        'timestamp': datetime.now()
                    })
        
        # Signal 2: Bearish FVG
        fvg_bearish = latest.get('fvg_bearish', np.nan)
        if not pd.isna(fvg_bearish):
            fvg_top = fvg_bearish
            fvg_bottom = fvg_bearish - latest.get('fvg_width', 0)
            current_price = latest['close']
            
            if fvg_bottom <= current_price <= fvg_top:
                rsi = latest['rsi']
                confidence = 0.65
                
                if rsi > 65:
                    confidence += 0.15
                
                if latest.get('volume_ratio', 1) > 1.5:
                    confidence += 0.10
                
                if latest['market_structure'] == 'downtrend':
                    confidence += 0.10
                
                if confidence >= self.config['confidence_threshold']:
                    entry = current_price
                    stop_loss = fvg_top + (atr_value * 0.5)
                    take_profit = entry - (atr_value * take_profit_atr)
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'asset_name': selected_asset,
                        'type': 'SELL',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': min(confidence, 0.95),
                        'strategy': 'SMC FVG Fill',
                        'reason': f'Bearish FVG fill with RSI: {rsi:.1f}',
                        'conditions': ['FVG', f'RSI: {rsi:.1f}', 'Volume Confirmation'],
                        'timestamp': datetime.now()
                    })
        
        # Signal 3: Order Block + RSI
        ob_bullish = latest.get('ob_bullish', np.nan)
        if not pd.isna(ob_bullish):
            current_price = latest['close']
            rsi = latest['rsi']
            
            if abs(current_price - ob_bullish) / current_price < 0.002:  # Within 0.2% of OB
                confidence = 0.70
                
                if rsi < 40:
                    confidence += 0.15
                
                if latest['market_structure'] == 'uptrend':
                    confidence += 0.10
                
                if confidence >= self.config['confidence_threshold']:
                    entry = current_price
                    stop_loss = ob_bullish - (atr_value * stop_loss_atr)
                    take_profit = entry + (atr_value * take_profit_atr)
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'asset_name': selected_asset,
                        'type': 'BUY',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'strategy': 'SMC Order Block',
                        'reason': f'Bullish Order Block with RSI: {rsi:.1f}',
                        'conditions': ['Order Block', f'RSI: {rsi:.1f}', 'Uptrend'],
                        'timestamp': datetime.now()
                    })
        
        # Signal 4: Break of Structure (BOS)
        swing_highs = df[df['swing_high']]
        swing_lows = df[df['swing_low']]
        
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            last_swing_high = swing_highs['high'].iloc[-1]
            last_swing_low = swing_lows['low'].iloc[-1]
            current_price = latest['close']
            
            # Bullish BOS
            if current_price > last_swing_high and latest['rsi'] > 50:
                confidence = 0.75
                distance = current_price - last_swing_high
                
                if distance > atr_value * 0.5:
                    confidence += 0.10
                
                if latest.get('volume_ratio', 1) > 1.2:
                    confidence += 0.10
                
                if confidence >= self.config['confidence_threshold']:
                    entry = current_price
                    stop_loss = last_swing_low
                    take_profit = entry + (distance * 2)
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'asset_name': selected_asset,
                        'type': 'BUY',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'strategy': 'SMC Break of Structure',
                        'reason': f'Bullish BOS above swing high',
                        'conditions': ['BOS', f'RSI: {latest["rsi"]:.1f}', 'Volume Confirmation'],
                        'timestamp': datetime.now()
                    })
            
            # Bearish BOS
            elif current_price < last_swing_low and latest['rsi'] < 50:
                confidence = 0.75
                distance = last_swing_low - current_price
                
                if distance > atr_value * 0.5:
                    confidence += 0.10
                
                if latest.get('volume_ratio', 1) > 1.2:
                    confidence += 0.10
                
                if confidence >= self.config['confidence_threshold']:
                    entry = current_price
                    stop_loss = last_swing_high
                    take_profit = entry - (distance * 2)
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'asset_name': selected_asset,
                        'type': 'SELL',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'strategy': 'SMC Break of Structure',
                        'reason': f'Bearish BOS below swing low',
                        'conditions': ['BOS', f'RSI: {latest["rsi"]:.1f}', 'Volume Confirmation'],
                        'timestamp': datetime.now()
                    })
        
        # Sort by confidence
        signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        return signals[:3]  # Return top 3 signals

# Enhanced Trading Engine
class TradingEngine:
    def __init__(self):
        self.portfolio = st.session_state.paper_portfolio
        self.traded_symbols = st.session_state.traded_symbols
        
    def can_trade_symbol(self, symbol):
        """Check if we can trade this symbol (not already in positions)"""
        for pos_id, position in self.portfolio['positions'].items():
            if position['asset'] == symbol and position['status'] == 'OPEN':
                return False
        return True
    
    def execute_trade(self, signal, current_price):
        """Execute a trade with position management"""
        symbol = signal['asset']
        
        # Check if already trading this symbol
        if not self.can_trade_symbol(symbol):
            return False, f"Already trading {symbol}"
        
        # Check max positions
        if len(self.portfolio['positions']) >= max_positions:
            return False, "Maximum positions reached"
        
        # Calculate position size with risk management
        atr_value = signal.get('atr', current_price * 0.01)
        stop_distance = abs(signal['entry'] - signal['stop_loss'])
        
        if stop_distance == 0:
            return False, "Invalid stop loss"
        
        risk_amount = self.portfolio['balance'] * risk_per_trade / 100
        position_size = risk_amount / stop_distance
        
        # Adjust for lot size
        position_size = min(position_size, asset_info['lot_size'] * 3)
        
        trade_value = signal['entry'] * position_size
        
        # Check margin
        if trade_value > self.portfolio['balance'] * 0.8:
            return False, "Insufficient margin"
        
        # Create trade
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        trade = {
            'id': trade_id,
            'timestamp': datetime.now(),
            'asset': symbol,
            'asset_name': signal['asset_name'],
            'type': signal['type'],
            'entry_price': signal['entry'],
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'current_price': current_price,
            'status': 'OPEN',
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'strategy': signal.get('strategy', 'Unknown'),
            'reason': signal.get('reason', ''),
            'confidence': signal.get('confidence', 0.5)
        }
        
        # Update portfolio
        self.portfolio['positions'][trade_id] = trade
        self.traded_symbols.add(symbol)
        
        # Log trade
        log_entry = {
            'timestamp': datetime.now(),
            'action': 'OPEN',
            'trade_id': trade_id,
            'asset': signal['asset_name'],
            'symbol': symbol,
            'type': signal['type'],
            'entry_price': signal['entry'],
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'strategy': signal.get('strategy', 'Unknown'),
            'reason': signal.get('reason', ''),
            'confidence': signal.get('confidence', 0.5)
        }
        
        st.session_state.trade_log.append(log_entry)
        
        return True, trade_id
    
    def update_positions(self, market_prices):
        """Update all positions with current prices"""
        total_pnl = 0
        positions_closed = []
        
        for trade_id, position in list(self.portfolio['positions'].items()):
            if position['status'] == 'OPEN':
                symbol = position['asset']
                current_price = market_prices.get(symbol)
                
                if current_price:
                    # Calculate P&L
                    if position['type'] == 'BUY':
                        pnl = (current_price - position['entry_price']) * position['size']
                    else:  # SELL
                        pnl = (position['entry_price'] - current_price) * position['size']
                    
                    pnl_percent = (pnl / (position['entry_price'] * position['size'])) * 100
                    
                    position['current_price'] = current_price
                    position['pnl'] = pnl
                    position['pnl_percent'] = pnl_percent
                    
                    total_pnl += pnl
                    
                    # Check exit conditions
                    exit_reason = None
                    
                    # Stop Loss
                    if (position['type'] == 'BUY' and current_price <= position['stop_loss']) or \
                       (position['type'] == 'SELL' and current_price >= position['stop_loss']):
                        exit_reason = "Stop Loss Hit"
                    
                    # Take Profit
                    elif (position['type'] == 'BUY' and current_price >= position['take_profit']) or \
                         (position['type'] == 'SELL' and current_price <= position['take_profit']):
                        exit_reason = "Take Profit Hit"
                    
                    if exit_reason:
                        self.close_position(trade_id, current_price, exit_reason)
                        positions_closed.append(trade_id)
        
        # Update portfolio P&L
        self.portfolio['total_pnl'] += total_pnl
        self.portfolio['daily_pnl'] += total_pnl
        
        # Update equity curve
        current_equity = self.portfolio['balance'] + total_pnl
        self.portfolio['equity_curve'].append(current_equity)
        
        # Update win/loss stats
        for trade_id in positions_closed:
            if trade_id in self.portfolio['positions']:
                position = self.portfolio['positions'][trade_id]
                if position['pnl'] > 0:
                    self.portfolio['winning_trades'] += 1
                else:
                    self.portfolio['losing_trades'] += 1
        
        # Calculate drawdown
        if self.portfolio['equity_curve']:
            current_peak = max(self.portfolio['equity_curve'])
            current_value = self.portfolio['equity_curve'][-1]
            drawdown = ((current_peak - current_value) / current_peak) * 100
            self.portfolio['max_drawdown'] = max(self.portfolio['max_drawdown'], drawdown)
        
        return total_pnl
    
    def close_position(self, trade_id, exit_price, reason="Manual Close"):
        """Close a position"""
        if trade_id in self.portfolio['positions']:
            position = self.portfolio['positions'][trade_id]
            
            # Calculate final P&L
            if position['type'] == 'BUY':
                final_pnl = (exit_price - position['entry_price']) * position['size']
            else:
                final_pnl = (position['entry_price'] - exit_price) * position['size']
            
            final_pnl_percent = (final_pnl / (position['entry_price'] * position['size'])) * 100
            
            # Update portfolio balance
            self.portfolio['balance'] += final_pnl
            
            # Log closure
            log_entry = {
                'timestamp': datetime.now(),
                'action': 'CLOSE',
                'trade_id': trade_id,
                'asset': position['asset_name'],
                'symbol': position['asset'],
                'type': position['type'],
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': final_pnl,
                'pnl_percent': final_pnl_percent,
                'reason': reason,
                'holding_time': (datetime.now() - position['timestamp']).total_seconds() / 60
            }
            
            st.session_state.trade_log.append(log_entry)
            
            # Remove from positions
            del self.portfolio['positions'][trade_id]
            if position['asset'] in self.traded_symbols:
                self.traded_symbols.remove(position['asset'])
            
            return True
        
        return False

# Auto-refresh logic
def auto_refresh_data():
    """Auto-refresh market data and signals"""
    if st.session_state.auto_refresh:
        current_time = datetime.now()
        last_update = st.session_state.last_update
        
        if (current_time - last_update).seconds >= st.session_state.refresh_interval:
            st.session_state.last_update = current_time
            st.rerun()

# Main Dashboard
def main():
    # Initialize trading engine
    trading_engine = TradingEngine()
    smc_algo = AdvancedSMCAlgorithm(strategy=selected_strategy, fvg_lookback=fvg_period, swing_period=swing_period)
    
    # Display Dashboard Metrics
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        balance = st.session_state.paper_portfolio['balance']
        total_pnl = st.session_state.paper_portfolio['total_pnl']
        change_color = "change-positive" if total_pnl >= 0 else "change-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ACCOUNT BALANCE</div>
            <div class="metric-value">${balance:,.2f}</div>
            <div class="metric-change {change_color}">${total_pnl:+,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        daily_pnl = st.session_state.paper_portfolio['daily_pnl']
        change_color = "change-positive" if daily_pnl >= 0 else "change-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DAILY P&L</div>
            <div class="metric-value">${daily_pnl:+,.2f}</div>
            <div class="metric-change {change_color}">{daily_pnl/balance*100:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        winning = st.session_state.paper_portfolio['winning_trades']
        losing = st.session_state.paper_portfolio['losing_trades']
        total = winning + losing
        win_rate = (winning / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">WIN RATE</div>
            <div class="metric-value">{win_rate:.1f}%</div>
            <div class="metric-change">{winning}W / {losing}L</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        open_positions = len(st.session_state.paper_portfolio['positions'])
        max_dd = st.session_state.paper_portfolio['max_drawdown']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">OPEN POSITIONS</div>
            <div class="metric-value">{open_positions}</div>
            <div class="metric-change">Max DD: {max_dd:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        equity_growth = ((st.session_state.paper_portfolio['equity_curve'][-1] / 50000) - 1) * 100
        change_color = "change-positive" if equity_growth >= 0 else "change-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">EQUITY GROWTH</div>
            <div class="metric-value {change_color}">{equity_growth:+.2f}%</div>
            <div class="metric-change">Since inception</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh controls
    last_update_str = st.session_state.last_update.strftime('%H:%M:%S')
    refresh_interval_str = st.session_state.refresh_interval
    
    st.markdown(f"""
    <div class="refresh-controls">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>🔄 Auto-refresh Status:</strong>
                <span style="color: #10b981; margin-left: 10px;">ACTIVE</span>
                <span style="color: #94a3b8; margin-left: 20px;">Interval: {refresh_interval_str}s</span>
            </div>
            <div>
                <span style="color: #94a3b8;">Last update: {last_update_str}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 LIVE TRADING", 
        "🔍 SIGNAL SCANNER", 
        "💰 PORTFOLIO", 
        "📊 TRADE HISTORY"
    ])
    
    with tab1:
        st.subheader("Live Trading Dashboard")
        
        # Fetch market data
        with st.spinner(f"📡 Fetching {selected_asset} data..."):
            df = fetch_intraday_data(asset_info['symbol'], timeframe)
        
        if not df.empty:
            # Display price info
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${current_price:.4f}", f"{change_pct:+.2f}%")
            with col2:
                atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                st.metric("ATR", f"${atr_value:.4f}")
            with col3:
                rsi_value = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
                st.metric("RSI", f"{rsi_value:.1f}")
            with col4:
                volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
                st.metric("Volume", f"{volume:,.0f}")
            
            # Generate signals
            df_analyzed = smc_algo.analyze_market_structure(df)
            df_analyzed = smc_algo.identify_fvgs(df_analyzed)
            df_analyzed = smc_algo.identify_orderblocks(df_analyzed)
            
            signals = smc_algo.generate_smc_signals(df_analyzed, asset_info)
            
            if signals:
                st.success(f"🎯 Found {len(signals)} trading signals")
                
                # Display signals
                st.markdown('<div class="signal-container">', unsafe_allow_html=True)
                
                for i, signal in enumerate(signals[:3]):  # Show top 3 signals
                    signal_type_class = "buy" if signal['type'] == 'BUY' else "sell"
                    confidence_color = "#10b981" if signal['confidence'] > 0.8 else "#f59e0b" if signal['confidence'] > 0.7 else "#ef4444"
                    
                    st.markdown(f"""
                    <div class="signal-card {signal_type_class}">
                        <div class="signal-header">
                            <div class="signal-title">{signal['asset_name']} - {signal['type']}</div>
                            <div class="confidence-badge" style="background: {confidence_color};">{signal['confidence']*100:.0f}%</div>
                        </div>
                        <div class="signal-grid">
                            <div class="signal-item">
                                <div class="signal-label">Entry Price</div>
                                <div class="signal-value">${signal['entry']:.4f}</div>
                            </div>
                            <div class="signal-item">
                                <div class="signal-label">Position Size</div>
                                <div class="signal-value">{signal['size']:.4f}</div>
                            </div>
                            <div class="signal-item">
                                <div class="signal-label">Stop Loss</div>
                                <div class="signal-value" style="color: #ef4444;">${signal['stop_loss']:.4f}</div>
                            </div>
                            <div class="signal-item">
                                <div class="signal-label">Take Profit</div>
                                <div class="signal-value" style="color: #10b981;">${signal['take_profit']:.4f}</div>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div style="color: #94a3b8; font-size: 0.9rem;"><strong>Strategy:</strong> {signal.get('strategy', 'SMC')}</div>
                            <div style="color: #94a3b8; font-size: 0.9rem;"><strong>Reason:</strong> {signal.get('reason', '')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Execute button
                    if st.button(f"🚀 Execute {signal['type']} #{i+1}", key=f"execute_{i}"):
                        success, trade_id = trading_engine.execute_trade(signal, current_price)
                        if success:
                            st.success(f"✅ Trade executed successfully! ID: {trade_id}")
                            tm.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Trade execution failed")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📊 No trading signals detected for current market conditions")
        
        # Quick Actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh All Data", type="primary"):
                st.session_state.last_update = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("📊 Scan All Assets"):
                with st.spinner("Scanning all assets for opportunities..."):
                    # Implement multi-asset scan
                    st.info("Multi-asset scan feature coming soon!")
        
        with col3:
            if st.button("💰 Update Positions"):
                # Update positions with current prices
                market_prices = {asset_info['symbol']: current_price}
                pnl = trading_engine.update_positions(market_prices)
                st.success(f"Positions updated! Current P&L: ${pnl:+.2f}")
                st.rerun()
    
    with tab2:
        st.subheader("Multi-Asset Signal Scanner")
        
        col1, col2 = st.columns(2)
        with col1:
            scan_categories = st.multiselect(
                "Categories to Scan",
                list(ASSET_CONFIG.keys()),
                default=["Cryptocurrencies", "Forex", "Tech Stocks"]
            )
        with col2:
            min_confidence = st.slider("Minimum Confidence", 70, 95, 75)
        
        if st.button("🚀 Start Scan", type="primary"):
            all_signals = []
            
            with st.spinner(f"🔍 Scanning {len(scan_categories)} categories..."):
                progress_bar = st.progress(0)
                total_assets = sum(len(ASSET_CONFIG[cat]) for cat in scan_categories)
                scanned = 0
                
                for cat_idx, category in enumerate(scan_categories):
                    st.write(f"**{category}**")
                    
                    for asset_name, info in ASSET_CONFIG[category].items():
                        try:
                            df = fetch_intraday_data(info['symbol'], timeframe)
                            if not df.empty and len(df) > 30:
                                df = smc_algo.analyze_market_structure(df)
                                df = smc_algo.identify_fvgs(df)
                                df = smc_algo.identify_orderblocks(df)
                                
                                signals = smc_algo.generate_smc_signals(df, info)
                                
                                for signal in signals:
                                    if signal['confidence'] * 100 >= min_confidence:
                                        signal['category'] = category
                                        signal['asset_display'] = asset_name
                                        all_signals.append(signal)
                        except:
                            continue
                        
                        scanned += 1
                        progress_bar.progress(scanned / total_assets)
                
                progress_bar.empty()
            
            if all_signals:
                st.success(f"🎯 Found {len(all_signals)} high-confidence signals")
                
                # Group by category
                for category in scan_categories:
                    category_signals = [s for s in all_signals if s['category'] == category]
                    
                    if category_signals:
                        st.markdown(f"### {category}")
                        
                        for signal in category_signals[:5]:  # Top 5 per category
                            with st.expander(f"{signal['asset_display']} - {signal['type']} ({signal['confidence']*100:.0f}%)"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Entry:** ${signal['entry']:.4f}")
                                    st.write(f"**Stop Loss:** ${signal['stop_loss']:.4f}")
                                with col2:
                                    st.write(f"**Take Profit:** ${signal['take_profit']:.4f}")
                                    st.write(f"**Size:** {signal['size']:.4f}")
                                
                                st.write(f"**Reason:** {signal.get('reason', '')}")
                                
                                if st.button(f"Execute {signal['type']}", key=f"scan_exec_{signal['asset']}_{signal['timestamp']}"):
                                    success, trade_id = trading_engine.execute_trade(signal, signal['entry'])
                                    if success:
                                        st.success(f"Trade executed: {trade_id}")
                                        st.rerun()
                                    else:
                                        st.error("Execution failed")
            else:
                st.warning("No signals found meeting the criteria")
    
    with tab3:
        st.subheader("Portfolio Overview")
        
        # Open Positions
        st.markdown("### 📊 Open Positions")
        positions = st.session_state.paper_portfolio['positions']
        
        if positions:
            for trade_id, position in positions.items():
                pnl = position['pnl']
                pnl_percent = position.get('pnl_percent', 0)
                pnl_color = "pnl-positive" if pnl >= 0 else "pnl-negative"
                
                st.markdown(f"""
                <div class="trade-entry">
                    <div class="trade-header">
                        <div class="trade-asset">{position['asset_name']} - {position['type']}</div>
                        <div class="trade-pnl {pnl_color}">${pnl:+,.2f} ({pnl_percent:+.2f}%)</div>
                    </div>
                    <div class="trade-details">
                        <div><strong>Entry:</strong> ${position['entry_price']:.4f}</div>
                        <div><strong>Current:</strong> ${position.get('current_price', position['entry_price']):.4f}</div>
                        <div><strong>Stop Loss:</strong> ${position['stop_loss']:.4f}</div>
                        <div><strong>Take Profit:</strong> ${position['take_profit']:.4f}</div>
                        <div><strong>Size:</strong> {position['size']:.4f}</div>
                        <div><strong>Strategy:</strong> {position.get('strategy', 'Unknown')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Close button
                if st.button(f"Close Position", key=f"close_{trade_id}"):
                    trading_engine.close_position(trade_id, position.get('current_price', position['entry_price']), "Manual Close")
                    st.success(f"Position {trade_id} closed")
                    tm.sleep(1)
                    st.rerun()
        else:
            st.info("No open positions")
        
        # Portfolio Stats
        st.markdown("---")
        st.subheader("📈 Portfolio Statistics")
        
        # Calculate performance metrics
        closed_trades = [log for log in st.session_state.trade_log if log['action'] == 'CLOSE']
        
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
            
            total_wins = len(winning_trades)
            total_losses = len(losing_trades)
            total_trades = total_wins + total_losses
            
            avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t.get('pnl', 0)) for t in losing_trades]) if losing_trades else 0
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            
            profit_factor = (sum(t.get('pnl', 0) for t in winning_trades) / 
                           abs(sum(t.get('pnl', 0) for t in losing_trades))) if losing_trades else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col2:
                st.metric("Total Trades", total_trades)
            
            with col3:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            
            with col4:
                st.metric("Avg Win/Loss", f"${avg_win:.2f}/${avg_loss:.2f}")
    
    with tab4:
        st.subheader("Trade History & Analytics")
        
        if st.session_state.trade_log:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                unique_assets = sorted(set([log.get('asset', '') for log in st.session_state.trade_log if log.get('asset')]))
                filter_asset = st.selectbox("Filter Asset", ["All"] + unique_assets)
            with col2:
                filter_action = st.selectbox("Filter Action", ["All", "OPEN", "CLOSE"])
            with col3:
                filter_type = st.selectbox("Filter Type", ["All", "BUY", "SELL"])
            
            # Apply filters
            filtered_logs = st.session_state.trade_log
            
            if filter_asset != "All":
                filtered_logs = [log for log in filtered_logs if log.get('asset') == filter_asset]
            
            if filter_action != "All":
                filtered_logs = [log for log in filtered_logs if log['action'] == filter_action]
            
            if filter_type != "All":
                filtered_logs = [log for log in filtered_logs if log.get('type') == filter_type]
            
            # Display logs
            st.markdown('<div class="trade-log">', unsafe_allow_html=True)
            
            for log in reversed(filtered_logs[-50:]):  # Last 50 trades
                timestamp = log['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                asset = log.get('asset', 'Unknown')
                action = log['action']
                trade_type = log.get('type', '')
                
                if action == 'OPEN':
                    bg_color = "rgba(59, 130, 246, 0.1)"
                    icon = "📈" if trade_type == 'BUY' else "📉"
                else:
                    bg_color = "rgba(16, 185, 129, 0.1)" if log.get('pnl', 0) >= 0 else "rgba(239, 68, 68, 0.1)"
                    icon = "✅" if log.get('pnl', 0) >= 0 else "❌"
                
                st.markdown(f"""
                <div style="background: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #3b82f6;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <strong>{icon} {action} - {asset}</strong>
                            <span style="color: #94a3b8; margin-left: 10px;">{timestamp}</span>
                        </div>
                        <div>
                            {f"<span style='color: #10b981; font-weight: bold;'>+${log.get('pnl', 0):.2f}</span>" if action == 'CLOSE' and log.get('pnl', 0) >= 0 else ""}
                            {f"<span style='color: #ef4444; font-weight: bold;'>-${abs(log.get('pnl', 0)):.2f}</span>" if action == 'CLOSE' and log.get('pnl', 0) < 0 else ""}
                        </div>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">
                        {f"<strong>Type:</strong> {trade_type} | " if trade_type else ""}
                        {f"<strong>Entry:</strong> ${log.get('entry_price', 0):.4f} | " if log.get('entry_price') else ""}
                        {f"<strong>Exit:</strong> ${log.get('exit_price', 0):.4f} | " if log.get('exit_price') else ""}
                        {f"<strong>Size:</strong> {log.get('size', 0):.4f} | " if log.get('size') else ""}
                        {f"<strong>Strategy:</strong> {log.get('strategy', 'Unknown')}" if log.get('strategy') else ""}
                    </div>
                    {f"<div style='color: #d1d5db; font-size: 0.85rem; margin-top: 5px;'><strong>Reason:</strong> {log.get('reason', '')}</div>" if log.get('reason') else ""}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Export button
            if st.button("📥 Export Trade History"):
                df_log = pd.DataFrame(st.session_state.trade_log)
                csv = df_log.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No trade history yet. Execute some trades to see them here!")
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        auto_refresh_data()
        
        # Show countdown
        current_time = datetime.now()
        time_since_update = (current_time - st.session_state.last_update).seconds
        time_until_refresh = max(0, st.session_state.refresh_interval - time_since_update)
        
        st.markdown(f"""
        <div class="last-update">
            ⏱️ Next auto-refresh in {time_until_refresh} seconds
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><strong>⚠️ RISK DISCLAIMER:</strong> This is a paper trading simulation for educational purposes only.</p>
        <p>Past performance does not guarantee future results. Trading involves substantial risk of loss.</p>
        <p style='margin-top: 10px; font-size: 0.9rem;'>RTV SMC Algorithmic Trading Terminal Pro v2.0 • © 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
