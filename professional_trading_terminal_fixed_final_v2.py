# app.py - RTV SMC Professional Algorithmic Trading Terminal
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="RTV SMC Algo Trading Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Theme
st.markdown("""
<style>
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #0f172a;
        --accent-color: #3b82f6;
        --profit-color: #10b981;
        --loss-color: #ef4444;
        --neutral-color: #6b7280;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ffffff, #dbeafe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-color);
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    
    .signal-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1);
    }
    
    .signal-buy {
        border-left: 4px solid var(--profit-color);
    }
    
    .signal-sell {
        border-left: 4px solid var(--loss-color);
    }
    
    .tab-container {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 10px;
        padding: 0.5rem;
        margin-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #334155;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: white !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--secondary-color), #0f172a);
    }
    
    .dataframe {
        background-color: #1e293b !important;
        color: white !important;
    }
    
    .trade-log-entry {
        background: #1e293b;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--accent-color);
    }
    
    .status-active {
        color: var(--profit-color);
        font-weight: bold;
    }
    
    .status-closed {
        color: var(--neutral-color);
        font-weight: bold;
    }
    
    .profit-positive {
        color: var(--profit-color);
        font-weight: bold;
    }
    
    .profit-negative {
        color: var(--loss-color);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Terminal Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">📊 RTV SMC ALGO TRADING TERMINAL</h1>
    <p class="subtitle">Institutional Grade Algorithmic Trading System</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'paper_portfolio' not in st.session_state:
    st.session_state.paper_portfolio = {
        'balance': 100000.00,
        'positions': {},
        'trade_history': [],
        'pnl': 0.00,
        'equity_curve': [100000.00]
    }

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}

if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# Enhanced Asset Configuration
ASSET_CONFIG = {
    "Cryptocurrencies": {
        "BTC/USD": {"symbol": "BTC-USD", "pip_size": 0.01, "lot_size": 0.001, "sector": "Crypto"},
        "ETH/USD": {"symbol": "ETH-USD", "pip_size": 0.01, "lot_size": 0.01, "sector": "Crypto"},
        "SOL/USD": {"symbol": "SOL-USD", "pip_size": 0.001, "lot_size": 0.1, "sector": "Crypto"},
        "XRP/USD": {"symbol": "XRP-USD", "pip_size": 0.0001, "lot_size": 100, "sector": "Crypto"},
        "ADA/USD": {"symbol": "ADA-USD", "pip_size": 0.0001, "lot_size": 100, "sector": "Crypto"},
    },
    "Forex": {
        "EUR/USD": {"symbol": "EURUSD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex"},
        "GBP/USD": {"symbol": "GBPUSD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex"},
        "USD/JPY": {"symbol": "JPY=X", "pip_size": 0.01, "lot_size": 10000, "sector": "Forex"},
        "AUD/USD": {"symbol": "AUDUSD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex"},
        "USD/CAD": {"symbol": "CAD=X", "pip_size": 0.0001, "lot_size": 10000, "sector": "Forex"},
    },
    "Commodities": {
        "Gold": {"symbol": "GC=F", "pip_size": 0.10, "lot_size": 1, "sector": "Commodities"},
        "Silver": {"symbol": "SI=F", "pip_size": 0.01, "lot_size": 10, "sector": "Commodities"},
        "Crude Oil": {"symbol": "CL=F", "pip_size": 0.01, "lot_size": 10, "sector": "Commodities"},
        "Copper": {"symbol": "HG=F", "pip_size": 0.0005, "lot_size": 100, "sector": "Commodities"},
        "Natural Gas": {"symbol": "NG=F", "pip_size": 0.001, "lot_size": 100, "sector": "Commodities"},
    },
    "Indices": {
        "S&P 500": {"symbol": "^GSPC", "pip_size": 0.25, "lot_size": 1, "sector": "Indices"},
        "NASDAQ": {"symbol": "^IXIC", "pip_size": 0.25, "lot_size": 1, "sector": "Indices"},
        "Dow Jones": {"symbol": "^DJI", "pip_size": 1.0, "lot_size": 1, "sector": "Indices"},
        "FTSE 100": {"symbol": "^FTSE", "pip_size": 1.0, "lot_size": 1, "sector": "Indices"},
        "DAX": {"symbol": "^GDAXI", "pip_size": 1.0, "lot_size": 1, "sector": "Indices"},
    },
    "Stocks": {
        "Apple": {"symbol": "AAPL", "pip_size": 0.01, "lot_size": 10, "sector": "Tech"},
        "Microsoft": {"symbol": "MSFT", "pip_size": 0.01, "lot_size": 10, "sector": "Tech"},
        "Tesla": {"symbol": "TSLA", "pip_size": 0.01, "lot_size": 10, "sector": "Auto"},
        "Amazon": {"symbol": "AMZN", "pip_size": 0.01, "lot_size": 10, "sector": "Retail"},
        "Google": {"symbol": "GOOGL", "pip_size": 0.01, "lot_size": 10, "sector": "Tech"},
    }
}

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ SYSTEM CONFIGURATION")

# Mode Selection
mode = st.sidebar.selectbox(
    "Operating Mode",
    ["Live Analysis", "Multi-Asset Scan", "Paper Trading", "Backtesting"],
    index=0
)

# Multi-Asset Scan Configuration
if mode == "Multi-Asset Scan":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 SCAN PARAMETERS")
    
    scan_categories = st.sidebar.multiselect(
        "Categories to Scan",
        list(ASSET_CONFIG.keys()),
        default=["Cryptocurrencies", "Forex", "Commodities"]
    )
    
    min_confidence = st.sidebar.slider("Minimum Confidence (%)", 60, 95, 75)
    max_signals = st.sidebar.slider("Max Signals per Category", 1, 10, 5)
else:
    # Single Asset Selection
    asset_category = st.sidebar.selectbox(
        "Asset Category",
        list(ASSET_CONFIG.keys()),
        index=2
    )
    
    selected_asset = st.sidebar.selectbox(
        "Select Asset",
        list(ASSET_CONFIG[asset_category].keys())
    )
    
    asset_info = ASSET_CONFIG[asset_category][selected_asset]

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    index=1
)

# SMC Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 SMC PARAMETERS")
col1, col2 = st.sidebar.columns(2)
with col1:
    fvg_period = st.slider("FVG Lookback", 3, 20, 5)
    swing_period = st.slider("Swing Period", 2, 10, 3)
with col2:
    rsi_period = st.slider("RSI Period", 7, 21, 14)
    atr_period = st.slider("ATR Period", 7, 21, 14)

# Risk Management
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 RISK MANAGEMENT")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0)
stop_loss_atr = st.sidebar.slider("Stop Loss (ATR)", 1.0, 3.0, 2.0)
take_profit_atr = st.sidebar.slider("Take Profit (ATR)", 1.0, 4.0, 3.0)

# Trading Hours Filter (for Forex/Indices)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏰ TRADING HOURS")
trading_hours = st.sidebar.checkbox("Filter by Trading Hours", value=False)
if trading_hours:
    market_open = st.sidebar.time_input("Market Open", value=datetime.strptime("09:30", "%H:%M").time())
    market_close = st.sidebar.time_input("Market Close", value=datetime.strptime("16:00", "%H:%M").time())

# Data Fetching Function
@st.cache_data(ttl=60)
def fetch_market_data(symbol, interval='5m', period='7d'):
    try:
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '60m', '4h': '60m', '1d': '1d'
        }
        
        yf_interval = interval_map.get(interval, '5m')
        
        period_map = {
            '1m': '1d', '5m': '7d', '15m': '7d', '30m': '30d',
            '1h': '30d', '4h': '60d', '1d': '180d'
        }
        
        yf_period = period_map.get(interval, '7d')
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yf_period, interval=yf_interval)
        
        if df.empty:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df = yf.download(symbol, start=start_date, end=end_date, interval=yf_interval, progress=False)
        
        if len(df) > 0 and 'Open' in df.columns:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.dropna(inplace=True)
            df = add_technical_indicators(df)
            return df
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    
    return pd.DataFrame()

def add_technical_indicators(df):
    if len(df) < 20:
        return df
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        )
    )
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['MA9'] = df['Close'].rolling(window=9).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df

# Enhanced SMC Algorithm
class SMCAlgorithm:
    def __init__(self, fvg_lookback=5, swing_period=3):
        self.fvg_lookback = fvg_lookback
        self.swing_period = swing_period
        
    def analyze_market_structure(self, df):
        df = df.copy()
        
        # Identify swing points
        df['Swing_High'] = False
        df['Swing_Low'] = False
        
        for i in range(self.swing_period, len(df)-self.swing_period):
            if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'Swing_High'] = True
            
            if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'Swing_Low'] = True
        
        df['Market_Structure'] = 'Neutral'
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_2_highs = swing_highs['High'].iloc[-2:]
            last_2_lows = swing_lows['Low'].iloc[-2:]
            
            if len(last_2_highs) == 2 and len(last_2_lows) == 2:
                if (last_2_highs.iloc[1] > last_2_highs.iloc[0] and
                    last_2_lows.iloc[1] > last_2_lows.iloc[0]):
                    df['Market_Structure'] = 'Uptrend'
                elif (last_2_highs.iloc[1] < last_2_highs.iloc[0] and
                      last_2_lows.iloc[1] < last_2_lows.iloc[0]):
                    df['Market_Structure'] = 'Downtrend'
                elif (last_2_highs.iloc[1] > last_2_highs.iloc[0] and
                      last_2_lows.iloc[1] < last_2_lows.iloc[0]):
                    df['Market_Structure'] = 'Consolidation'
        
        return df
    
    def identify_fvgs(self, df):
        df = df.copy()
        df['FVG_Bullish'] = np.nan
        df['FVG_Bearish'] = np.nan
        df['FVG_Width'] = np.nan
        df['FVG_Strength'] = np.nan
        
        for i in range(self.fvg_lookback, len(df)-1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG
            if current['High'] < next_candle['Low']:
                gap = next_candle['Low'] - current['High']
                if gap > current['ATR'] * 0.1:
                    df.loc[df.index[i], 'FVG_Bullish'] = current['High']
                    df.loc[df.index[i], 'FVG_Width'] = gap
                    df.loc[df.index[i], 'FVG_Strength'] = gap / current['ATR']
            
            # Bearish FVG
            elif current['Low'] > next_candle['High']:
                gap = current['Low'] - next_candle['High']
                if gap > current['ATR'] * 0.1:
                    df.loc[df.index[i], 'FVG_Bearish'] = current['Low']
                    df.loc[df.index[i], 'FVG_Width'] = gap
                    df.loc[df.index[i], 'FVG_Strength'] = gap / current['ATR']
        
        return df
    
    def identify_orderblocks(self, df):
        df = df.copy()
        df['OB_Bullish'] = np.nan
        df['OB_Bearish'] = np.nan
        df['OB_Strength'] = np.nan
        
        for i in range(2, len(df)-2):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            # Bullish Order Block
            body_size = abs(current['Close'] - current['Open'])
            candle_range = current['High'] - current['Low']
            
            if (current['Close'] > current['Open'] and
                body_size > candle_range * 0.6 and
                next_candle['Low'] >= current['Low']):
                
                df.loc[df.index[i], 'OB_Bullish'] = current['Low']
                df.loc[df.index[i], 'OB_Strength'] = body_size / current['ATR']
            
            # Bearish Order Block
            elif (current['Close'] < current['Open'] and
                  body_size > candle_range * 0.6 and
                  next_candle['High'] <= current['High']):
                
                df.loc[df.index[i], 'OB_Bearish'] = current['High']
                df.loc[df.index[i], 'OB_Strength'] = body_size / current['ATR']
        
        return df
    
    def generate_signals(self, df, asset_info):
        signals = []
        
        if len(df) < 20:
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate position size
        atr_value = latest['ATR'] if not pd.isna(latest['ATR']) else df['ATR'].mean()
        if atr_value == 0:
            atr_value = df['Close'].std() * 0.01
        
        stop_distance = atr_value * stop_loss_atr
        risk_amount = st.session_state.paper_portfolio['balance'] * risk_per_trade / 100
        position_size = risk_amount / stop_distance
        position_size = min(position_size, asset_info['lot_size'] * 10)
        
        # Signal 1: FVG + RSI confluence
        if not pd.isna(latest.get('FVG_Bullish', np.nan)):
            fvg_top = latest['FVG_Bullish'] + latest.get('FVG_Width', 0)
            if latest['Low'] <= fvg_top <= latest['High']:
                rsi_score = max(0, (40 - latest['RSI']) / 40) if latest['RSI'] < 40 else 0
                confidence = 0.65 + (rsi_score * 0.2)
                
                if confidence * 100 >= 60:
                    entry = fvg_top
                    stop_loss = latest['FVG_Bullish'] - atr_value * 0.5
                    take_profit = entry + atr_value * take_profit_atr
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'name': list(ASSET_CONFIG.keys())[0],
                        'type': 'BUY',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'reason': 'Bullish FVG fill',
                        'conditions': ['FVG', f'RSI: {latest["RSI"]:.1f}', 'Price at FVG'],
                        'sector': asset_info.get('sector', 'General'),
                        'timestamp': datetime.now()
                    })
        
        if not pd.isna(latest.get('FVG_Bearish', np.nan)):
            fvg_bottom = latest['FVG_Bearish'] - latest.get('FVG_Width', 0)
            if latest['Low'] <= fvg_bottom <= latest['High']:
                rsi_score = max(0, (latest['RSI'] - 60) / 40) if latest['RSI'] > 60 else 0
                confidence = 0.65 + (rsi_score * 0.2)
                
                if confidence * 100 >= 60:
                    entry = fvg_bottom
                    stop_loss = latest['FVG_Bearish'] + atr_value * 0.5
                    take_profit = entry - atr_value * take_profit_atr
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'name': list(ASSET_CONFIG.keys())[0],
                        'type': 'SELL',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'reason': 'Bearish FVG fill',
                        'conditions': ['FVG', f'RSI: {latest["RSI"]:.1f}', 'Price at FVG'],
                        'sector': asset_info.get('sector', 'General'),
                        'timestamp': datetime.now()
                    })
        
        # Signal 2: Order Block + Market Structure
        if not pd.isna(latest.get('OB_Bullish', np.nan)):
            ob_price = latest['OB_Bullish']
            if latest['Low'] <= ob_price <= latest['High']:
                structure_score = 0.2 if latest['Market_Structure'] == 'Uptrend' else 0
                confidence = 0.7 + structure_score
                
                if confidence * 100 >= 60:
                    entry = ob_price
                    stop_loss = ob_price - atr_value * stop_loss_atr
                    take_profit = entry + atr_value * take_profit_atr
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'name': list(ASSET_CONFIG.keys())[0],
                        'type': 'BUY',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'reason': 'Bullish Order Block',
                        'conditions': ['Order Block', latest['Market_Structure'], 'Price at OB'],
                        'sector': asset_info.get('sector', 'General'),
                        'timestamp': datetime.now()
                    })
        
        if not pd.isna(latest.get('OB_Bearish', np.nan)):
            ob_price = latest['OB_Bearish']
            if latest['Low'] <= ob_price <= latest['High']:
                structure_score = 0.2 if latest['Market_Structure'] == 'Downtrend' else 0
                confidence = 0.7 + structure_score
                
                if confidence * 100 >= 60:
                    entry = ob_price
                    stop_loss = ob_price + atr_value * stop_loss_atr
                    take_profit = entry - atr_value * take_profit_atr
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'name': list(ASSET_CONFIG.keys())[0],
                        'type': 'SELL',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'reason': 'Bearish Order Block',
                        'conditions': ['Order Block', latest['Market_Structure'], 'Price at OB'],
                        'sector': asset_info.get('sector', 'General'),
                        'timestamp': datetime.now()
                    })
        
        # Signal 3: Break of Structure (BOS)
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            last_swing_high = swing_highs['High'].iloc[-1]
            last_swing_low = swing_lows['Low'].iloc[-1]
            
            # BOS to upside
            if latest['Close'] > last_swing_high and latest['RSI'] > 50:
                rsi_score = (latest['RSI'] - 50) / 30
                confidence = 0.75 + min(rsi_score * 0.2, 0.15)
                
                if confidence * 100 >= 60:
                    entry = latest['Close']
                    stop_loss = last_swing_low
                    take_profit = entry + (entry - stop_loss) * 2
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'name': list(ASSET_CONFIG.keys())[0],
                        'type': 'BUY',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'reason': 'Break of Structure (BOS)',
                        'conditions': ['BOS', f'RSI: {latest["RSI"]:.1f}', 'Above swing high'],
                        'sector': asset_info.get('sector', 'General'),
                        'timestamp': datetime.now()
                    })
            
            # BOS to downside
            elif latest['Close'] < last_swing_low and latest['RSI'] < 50:
                rsi_score = (50 - latest['RSI']) / 30
                confidence = 0.75 + min(rsi_score * 0.2, 0.15)
                
                if confidence * 100 >= 60:
                    entry = latest['Close']
                    stop_loss = last_swing_high
                    take_profit = entry - (stop_loss - entry) * 2
                    
                    signals.append({
                        'asset': asset_info['symbol'],
                        'name': list(ASSET_CONFIG.keys())[0],
                        'type': 'SELL',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_size,
                        'confidence': confidence,
                        'reason': 'Break of Structure (BOS)',
                        'conditions': ['BOS', f'RSI: {latest["RSI"]:.1f}', 'Below swing low'],
                        'sector': asset_info.get('sector', 'General'),
                        'timestamp': datetime.now()
                    })
        
        # Sort by confidence
        signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        return signals[:5]

# Enhanced Paper Trading Engine
class PaperTrading:
    def __init__(self):
        self.portfolio = st.session_state.paper_portfolio
    
    def execute_trade(self, signal, current_price, asset_name):
        trade_value = signal['entry'] * signal['size']
        
        if trade_value > self.portfolio['balance'] * 0.95:
            return False, "Insufficient balance"
        
        trade_id = f"{asset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade = {
            'id': trade_id,
            'timestamp': datetime.now(),
            'asset': signal['asset'],
            'asset_name': asset_name,
            'type': signal['type'],
            'entry_price': signal['entry'],
            'size': signal['size'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'status': 'OPEN',
            'pnl': 0.00,
            'max_pnl': 0.00,
            'min_pnl': 0.00,
            'reason': signal['reason'],
            'conditions': signal['conditions'],
            'confidence': signal['confidence']
        }
        
        self.portfolio['positions'][trade_id] = trade
        self.portfolio['balance'] -= trade_value * 0.001
        
        # Log trade
        log_entry = {
            'timestamp': datetime.now(),
            'action': 'OPEN',
            'trade_id': trade_id,
            'asset': asset_name,
            'type': signal['type'],
            'entry_price': signal['entry'],
            'size': signal['size'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'reason': signal['reason'],
            'confidence': signal['confidence']
        }
        st.session_state.trade_log.append(log_entry)
        
        return True, trade_id
    
    def close_position(self, trade_id, exit_price, reason="Exit Condition"):
        if trade_id in self.portfolio['positions']:
            position = self.portfolio['positions'][trade_id]
            
            if position['type'] == 'BUY':
                final_pnl = (exit_price - position['entry_price']) * position['size']
            else:
                final_pnl = (position['entry_price'] - exit_price) * position['size']
            
            trade_value = position['entry_price'] * position['size']
            self.portfolio['balance'] += trade_value + final_pnl
            
            # Log closure
            log_entry = {
                'timestamp': datetime.now(),
                'action': 'CLOSE',
                'trade_id': trade_id,
                'asset': position['asset_name'],
                'type': position['type'],
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': final_pnl,
                'reason': reason,
                'roi': (final_pnl / (position['entry_price'] * position['size'])) * 100
            }
            st.session_state.trade_log.append(log_entry)
            
            del self.portfolio['positions'][trade_id]
            return True
        
        return False

# Backtesting Engine
class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
    def run_backtest(self, df, asset_info, smc_algo):
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'trades': []
        }
        
        capital = self.initial_capital
        peak_capital = capital
        returns = []
        
        for i in range(50, len(df)-10):
            current_data = df.iloc[:i+1].copy()
            current_data = smc_algo.analyze_market_structure(current_data)
            current_data = smc_algo.identify_fvgs(current_data)
            current_data = smc_algo.identify_orderblocks(current_data)
            
            signals = smc_algo.generate_signals(current_data, asset_info)
            
            if signals and i < len(df) - 10:
                signal = signals[0]
                entry_price = signal['entry']
                exit_index = min(i + 15, len(df) - 1)
                
                if signal['type'] == 'BUY':
                    exit_price = df['Close'].iloc[exit_index]
                    pnl = (exit_price - entry_price) * signal['size']
                else:
                    exit_price = df['Close'].iloc[exit_index]
                    pnl = (entry_price - exit_price) * signal['size']
                
                capital += pnl
                results['total_pnl'] += pnl
                results['total_trades'] += 1
                
                if pnl > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1
                
                peak_capital = max(peak_capital, capital)
                drawdown = (peak_capital - capital) / peak_capital
                results['max_drawdown'] = max(results['max_drawdown'], drawdown)
                
                returns.append(pnl / capital if capital > 0 else 0)
                
                results['trades'].append({
                    'entry_time': df.index[i],
                    'exit_time': df.index[exit_index],
                    'type': signal['type'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'size': signal['size'],
                    'roi': (pnl / (entry_price * signal['size'])) * 100
                })
        
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades'] * 100
            total_wins = sum(t['pnl'] for t in results['trades'] if t['pnl'] > 0)
            total_losses = abs(sum(t['pnl'] for t in results['trades'] if t['pnl'] < 0))
            
            if total_losses > 0:
                results['profit_factor'] = total_wins / total_losses
            
            if len(returns) > 1:
                returns_array = np.array(returns)
                if returns_array.std() > 0:
                    results['sharpe_ratio'] = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
        
        results['max_drawdown'] *= 100
        results['final_capital'] = capital
        
        return results

# Multi-Asset Scanner
def scan_all_assets(categories, min_confidence, max_signals):
    all_signals = []
    smc_algo = SMCAlgorithm(fvg_lookback=fvg_period, swing_period=swing_period)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_assets = sum(len(ASSET_CONFIG[cat]) for cat in categories)
    current_asset = 0
    
    for category in categories:
        for asset_name, asset_info in ASSET_CONFIG[category].items():
            current_asset += 1
            progress = current_asset / total_assets
            progress_bar.progress(progress)
            status_text.text(f"Scanning {asset_name}...")
            
            try:
                df = fetch_market_data(asset_info['symbol'], timeframe)
                if not df.empty:
                    df = smc_algo.analyze_market_structure(df)
                    df = smc_algo.identify_fvgs(df)
                    df = smc_algo.identify_orderblocks(df)
                    
                    signals = smc_algo.generate_signals(df, asset_info)
                    
                    for signal in signals:
                        if signal['confidence'] * 100 >= min_confidence:
                            signal['category'] = category
                            signal['asset_name'] = asset_name
                            all_signals.append(signal)
                            
                            if len([s for s in all_signals if s['category'] == category]) >= max_signals:
                                break
            except:
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by confidence and category
    all_signals = sorted(all_signals, key=lambda x: (x['category'], -x['confidence']))
    
    return all_signals

# Main Application
def main():
    smc_algo = SMCAlgorithm(fvg_lookback=fvg_period, swing_period=swing_period)
    backtester = Backtester(initial_capital=100000)
    paper_trader = PaperTrading()
    
    # Dashboard Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-title">ACCOUNT BALANCE</div><div class="metric-value">${:,.2f}</div></div>'.format(
            st.session_state.paper_portfolio['balance']
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-title">TOTAL P&L</div><div class="metric-value profit-positive">${:,.2f}</div></div>'.format(
            st.session_state.paper_portfolio['pnl']
        ), unsafe_allow_html=True)
    
    with col3:
        open_positions = len(st.session_state.paper_portfolio['positions'])
        st.markdown('<div class="metric-card"><div class="metric-title">OPEN POSITIONS</div><div class="metric-value">{}</div></div>'.format(
            open_positions
        ), unsafe_allow_html=True)
    
    with col4:
        total_trades = len(st.session_state.paper_portfolio['trade_history'])
        st.markdown('<div class="metric-card"><div class="metric-title">TOTAL TRADES</div><div class="metric-value">{}</div></div>'.format(
            total_trades
        ), unsafe_allow_html=True)
    
    with col5:
        if len(st.session_state.paper_portfolio['equity_curve']) > 1:
            equity_growth = ((st.session_state.paper_portfolio['equity_curve'][-1] / 100000) - 1) * 100
            color_class = "profit-positive" if equity_growth >= 0 else "profit-negative"
            st.markdown(f'<div class="metric-card"><div class="metric-title">EQUITY GROWTH</div><div class="metric-value {color_class}">{equity_growth:+.2f}%</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="metric-title">EQUITY GROWTH</div><div class="metric-value">0.00%</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 LIVE ANALYSIS", 
        "🔍 MULTI-ASSET SCAN", 
        "🤖 AUTO TRADING", 
        "📊 BACKTESTING", 
        "📋 TRADE HISTORY"
    ])
    
    if mode == "Multi-Asset Scan":
        with tab2:
            st.subheader("Multi-Asset Scanner")
            
            if st.button("🚀 Start Scan", type="primary"):
                with st.spinner("Scanning all assets..."):
                    all_signals = scan_all_assets(scan_categories, min_confidence, max_signals)
                
                if all_signals:
                    st.success(f"🎯 Found {len(all_signals)} trading signals across {len(scan_categories)} categories")
                    
                    # Group by category
                    for category in scan_categories:
                        category_signals = [s for s in all_signals if s['category'] == category]
                        
                        if category_signals:
                            st.markdown(f"### {category}")
                            
                            for signal in category_signals[:max_signals]:
                                signal_class = "signal-buy" if signal['type'] == 'BUY' else "signal-sell"
                                confidence_color = "#10b981" if signal['confidence'] > 0.8 else "#f59e0b" if signal['confidence'] > 0.7 else "#ef4444"
                                
                                st.markdown(f"""
                                <div class="signal-card {signal_class}">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <h4 style="margin: 0; color: white;">{signal['asset_name']} - {signal['type']}</h4>
                                        <span style="background: {confidence_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;">
                                            Confidence: {signal['confidence']*100:.0f}%
                                        </span>
                                    </div>
                                    <div style="margin-top: 10px;">
                                        <p style="margin: 2px 0; color: #94a3b8;"><strong>Entry:</strong> ${signal['entry']:.4f}</p>
                                        <p style="margin: 2px 0; color: #94a3b8;"><strong>Stop Loss:</strong> ${signal['stop_loss']:.4f}</p>
                                        <p style="margin: 2px 0; color: #94a3b8;"><strong>Take Profit:</strong> ${signal['take_profit']:.4f}</p>
                                        <p style="margin: 2px 0; color: #94a3b8;"><strong>Reason:</strong> {signal['reason']}</p>
                                        <p style="margin: 2px 0; color: #94a3b8;"><strong>Conditions:</strong> {', '.join(signal['conditions'])}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Execute button
                                if st.button(f"Execute {signal['asset_name']} {signal['type']}", key=f"execute_{signal['asset_name']}_{signal['timestamp']}"):
                                    success, trade_id = paper_trader.execute_trade(signal, signal['entry'], signal['asset_name'])
                                    if success:
                                        st.success(f"✅ Trade executed: {trade_id}")
                                    else:
                                        st.error("❌ Trade execution failed")
                else:
                    st.warning("⚠️ No signals found meeting the criteria")
        
        # Set default tab to Scanner
        tab_to_show = tab2
    else:
        # Single Asset Analysis
        with tab1:
            st.subheader(f"Live Analysis - {selected_asset}")
            
            df = fetch_market_data(asset_info['symbol'], timeframe)
            
            if not df.empty:
                current_price = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.4f}", f"{change_pct:+.2f}%")
                with col2:
                    atr_value = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
                    st.metric("ATR", f"${atr_value:.4f}")
                with col3:
                    rsi_value = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                    st.metric("RSI", f"{rsi_value:.1f}")
                with col4:
                    volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                    st.metric("Volume", f"{volume:,.0f}")
                
                # Price Chart
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=(f"{selected_asset} Price Chart", "RSI", "Volume")
                )
                
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index[-100:],
                        open=df['Open'].tail(100),
                        high=df['High'].tail(100),
                        low=df['Low'].tail(100),
                        close=df['Close'].tail(100),
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Moving Averages
                if 'MA9' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index[-100:], y=df['MA9'].tail(100), 
                                 name="MA9", line=dict(color='orange', width=1)),
                        row=1, col=1
                    )
                
                if 'MA21' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index[-100:], y=df['MA21'].tail(100), 
                                 name="MA21", line=dict(color='blue', width=1)),
                        row=1, col=1
                    )
                
                # RSI
                if 'RSI' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index[-100:], y=df['RSI'].tail(100), 
                                 name="RSI", line=dict(color='purple', width=1)),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Volume
                if 'Volume' in df.columns:
                    colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                            else 'red' for i in range(len(df)-100, len(df))]
                    fig.add_trace(
                        go.Bar(x=df.index[-100:], y=df['Volume'].tail(100), 
                              name="Volume", marker_color=colors),
                        row=3, col=1
                    )
                
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Market Structure Analysis
                df_analyzed = smc_algo.analyze_market_structure(df)
                df_analyzed = smc_algo.identify_fvgs(df_analyzed)
                df_analyzed = smc_algo.identify_orderblocks(df_analyzed)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Market Structure")
                    structure = df_analyzed['Market_Structure'].iloc[-1]
                    if structure == 'Uptrend':
                        st.success(f"📈 {structure}")
                    elif structure == 'Downtrend':
                        st.error(f"📉 {structure}")
                    else:
                        st.info(f"⚪ {structure}")
                
                with col2:
                    st.markdown("### Active FVGs")
                    fvg_bullish = df_analyzed[~pd.isna(df_analyzed['FVG_Bullish'])]
                    fvg_bearish = df_analyzed[~pd.isna(df_analyzed['FVG_Bearish'])]
                    
                    if not fvg_bullish.empty:
                        for idx in fvg_bullish.index[-3:]:
                            price = fvg_bullish.loc[idx, 'FVG_Bullish']
                            width = fvg_bullish.loc[idx, 'FVG_Width']
                            st.write(f"**Bullish:** ${price:.4f} (Width: ${width:.4f})")
                    
                    if not fvg_bearish.empty:
                        for idx in fvg_bearish.index[-3:]:
                            price = fvg_bearish.loc[idx, 'FVG_Bearish']
                            width = fvg_bearish.loc[idx, 'FVG_Width']
                            st.write(f"**Bearish:** ${price:.4f} (Width: ${width:.4f})")
                
                with col3:
                    st.markdown("### Order Blocks")
                    ob_bullish = df_analyzed[~pd.isna(df_analyzed['OB_Bullish'])]
                    ob_bearish = df_analyzed[~pd.isna(df_analyzed['OB_Bearish'])]
                    
                    if not ob_bullish.empty:
                        for idx in ob_bullish.index[-3:]:
                            price = ob_bullish.loc[idx, 'OB_Bullish']
                            st.write(f"**Bullish OB:** ${price:.4f}")
                    
                    if not ob_bearish.empty:
                        for idx in ob_bearish.index[-3:]:
                            price = ob_bearish.loc[idx, 'OB_Bearish']
                            st.write(f"**Bearish OB:** ${price:.4f}")
        
        with tab3:
            st.subheader("Automated Trading Signals")
            
            if 'df_analyzed' not in locals():
                df = fetch_market_data(asset_info['symbol'], timeframe)
                df_analyzed = smc_algo.analyze_market_structure(df)
                df_analyzed = smc_algo.identify_fvgs(df_analyzed)
                df_analyzed = smc_algo.identify_orderblocks(df_analyzed)
            
            signals = smc_algo.generate_signals(df_analyzed, asset_info)
            
            if signals:
                st.success(f"🎯 {len(signals)} trading signals generated")
                
                for i, signal in enumerate(signals):
                    signal_class = "signal-buy" if signal['type'] == 'BUY' else "signal-sell"
                    confidence_color = "#10b981" if signal['confidence'] > 0.8 else "#f59e0b" if signal['confidence'] > 0.7 else "#ef4444"
                    
                    st.markdown(f"""
                    <div class="signal-card {signal_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: white;">Signal #{i+1}: {signal['type']}</h3>
                            <span style="background: {confidence_color}; color: white; padding: 4px 12px; border-radius: 15px; font-weight: bold;">
                                Confidence: {signal['confidence']*100:.0f}%
                            </span>
                        </div>
                        <div style="margin-top: 15px;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                <div>
                                    <p style="margin: 5px 0; color: #94a3b8;"><strong>Entry Price:</strong></p>
                                    <p style="margin: 5px 0; font-size: 1.2rem; color: white;">${signal['entry']:.4f}</p>
                                </div>
                                <div>
                                    <p style="margin: 5px 0; color: #94a3b8;"><strong>Position Size:</strong></p>
                                    <p style="margin: 5px 0; font-size: 1.2rem; color: white;">{signal['size']:.4f}</p>
                                </div>
                                <div>
                                    <p style="margin: 5px 0; color: #94a3b8;"><strong>Stop Loss:</strong></p>
                                    <p style="margin: 5px 0; font-size: 1.1rem; color: #ef4444;">${signal['stop_loss']:.4f}</p>
                                </div>
                                <div>
                                    <p style="margin: 5px 0; color: #94a3b8;"><strong>Take Profit:</strong></p>
                                    <p style="margin: 5px 0; font-size: 1.1rem; color: #10b981;">${signal['take_profit']:.4f}</p>
                                </div>
                            </div>
                            <div style="margin-top: 15px;">
                                <p style="margin: 5px 0; color: #94a3b8;"><strong>Reason:</strong> {signal['reason']}</p>
                                <p style="margin: 5px 0; color: #94a3b8;"><strong>Conditions Met:</strong> {', '.join(signal['conditions'])}</p>
                                <p style="margin: 5px 0; color: #94a3b8;"><strong>Sector:</strong> {signal['sector']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Execute button for each signal
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(f"📤 Execute {signal['type']} #{i+1}", key=f"execute_{i}", type="primary"):
                            success, trade_id = paper_trader.execute_trade(signal, signal['entry'], selected_asset)
                            if success:
                                st.success(f"✅ Trade executed successfully! ID: {trade_id}")
                                st.rerun()
                            else:
                                st.error("❌ Trade execution failed")
            else:
                st.info("📊 No trading signals generated for current market conditions")
                
                # Manual trade entry
                with st.expander("Manual Trade Entry"):
                    col1, col2 = st.columns(2)
                    with col1:
                        manual_type = st.selectbox("Type", ["BUY", "SELL"])
                        manual_entry = st.number_input("Entry Price", value=float(current_price))
                        manual_size = st.number_input("Position Size", value=1.0)
                    with col2:
                        manual_sl = st.number_input("Stop Loss", value=float(current_price * 0.98))
                        manual_tp = st.number_input("Take Profit", value=float(current_price * 1.02))
                    
                    if st.button("Execute Manual Trade"):
                        manual_signal = {
                            'asset': asset_info['symbol'],
                            'type': manual_type,
                            'entry': manual_entry,
                            'stop_loss': manual_sl,
                            'take_profit': manual_tp,
                            'size': manual_size,
                            'confidence': 0.5,
                            'reason': 'Manual Trade',
                            'conditions': ['Manual Entry'],
                            'sector': asset_info.get('sector', 'General'),
                            'timestamp': datetime.now()
                        }
                        
                        success, trade_id = paper_trader.execute_trade(manual_signal, manual_entry, selected_asset)
                        if success:
                            st.success(f"✅ Manual trade executed: {trade_id}")
                        else:
                            st.error("❌ Manual trade failed")
    
    with tab4:
        st.subheader("Strategy Backtesting")
        
        col1, col2 = st.columns(2)
        with col1:
            backtest_days = st.slider("Backtest Period (days)", 30, 365, 90)
        with col2:
            initial_capital = st.number_input("Initial Capital", value=100000.0)
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner(f"Running backtest on {selected_asset}..."):
                # Fetch historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=backtest_days)
                
                try:
                    df_backtest = yf.download(asset_info['symbol'], start=start_date, end=end_date, interval='1d')
                    if not df_backtest.empty:
                        df_backtest = add_technical_indicators(df_backtest)
                        
                        backtester = Backtester(initial_capital=initial_capital)
                        results = backtester.run_backtest(df_backtest, asset_info, smc_algo)
                        st.session_state.backtest_results = results
                        
                        st.success("✅ Backtest completed!")
                    else:
                        st.error("❌ Could not fetch historical data")
                except Exception as e:
                    st.error(f"❌ Backtest error: {e}")
        
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", results['total_trades'])
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            with col3:
                st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
            
            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric("Total P&L", f"${results['total_pnl']:,.2f}")
            with col6:
                st.metric("Final Capital", f"${results['final_capital']:,.2f}")
            with col7:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            
            # Equity Curve
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum() + initial_capital
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trades_df['exit_time'],
                    y=trades_df['cumulative_pnl'],
                    mode='lines',
                    name='Equity Curve',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                fig.update_layout(
                    title="Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade Distribution
                fig2 = go.Figure()
                
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                
                fig2.add_trace(go.Bar(
                    x=['Winning Trades', 'Losing Trades'],
                    y=[len(winning_trades), len(losing_trades)],
                    marker_color=['#10b981', '#ef4444'],
                    name='Trade Count'
                ))
                
                fig2.update_layout(
                    title="Trade Distribution",
                    xaxis_title="Trade Type",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Detailed Trades
                with st.expander("View Detailed Trades"):
                    st.dataframe(trades_df.sort_values('exit_time', ascending=False))
    
    with tab5:
        st.subheader("Trade History Log")
        
        if st.session_state.trade_log:
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_asset = st.selectbox("Filter by Asset", 
                                          ["All"] + list(set([log['asset'] for log in st.session_state.trade_log])))
            with col2:
                filter_type = st.selectbox("Filter by Type", 
                                         ["All", "BUY", "SELL"])
            with col3:
                filter_action = st.selectbox("Filter by Action", 
                                           ["All", "OPEN", "CLOSE"])
            
            # Apply filters
            filtered_logs = st.session_state.trade_log
            
            if filter_asset != "All":
                filtered_logs = [log for log in filtered_logs if log['asset'] == filter_asset]
            
            if filter_type != "All":
                filtered_logs = [log for log in filtered_logs if log.get('type') == filter_type]
            
            if filter_action != "All":
                filtered_logs = [log for log in filtered_logs if log['action'] == filter_action]
            
            # Display logs
            for log in reversed(filtered_logs[-50:]):  # Show last 50 logs
                timestamp_str = log['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                
                if log['action'] == 'OPEN':
                    bg_color = "#1e3a8a"
                    icon = "📈" if log.get('type') == 'BUY' else "📉"
                else:
                    bg_color = "#0f172a"
                    icon = "✅" if log.get('pnl', 0) >= 0 else "❌"
                
                st.markdown(f"""
                <div style="background: {bg_color}; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #3b82f6;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{icon} {log['action']} - {log['asset']}</strong>
                            <span style="color: #94a3b8; margin-left: 10px;">{timestamp_str}</span>
                        </div>
                        <div>
                            {f"<span style='color: #10b981; font-weight: bold;'>+${log['pnl']:,.2f}</span>" if log['action'] == 'CLOSE' and log.get('pnl') else ""}
                            {f"<span style='color: #f59e0b; font-size: 0.9rem;'>(Confidence: {log.get('confidence', 0)*100:.0f}%)</span>" if log.get('confidence') else ""}
                        </div>
                    </div>
                    <div style="margin-top: 8px; color: #94a3b8; font-size: 0.9rem;">
                        {f"<strong>Type:</strong> {log.get('type', 'N/A')} | " if log.get('type') else ""}
                        {f"<strong>Entry:</strong> ${log.get('entry_price', 0):.4f} | " if log.get('entry_price') else ""}
                        {f"<strong>Exit:</strong> ${log.get('exit_price', 0):.4f} | " if log.get('exit_price') else ""}
                        {f"<strong>Size:</strong> {log.get('size', 0):.4f} | " if log.get('size') else ""}
                        {f"<strong>Stop Loss:</strong> ${log.get('stop_loss', 0):.4f} | " if log.get('stop_loss') else ""}
                        {f"<strong>Take Profit:</strong> ${log.get('take_profit', 0):.4f}" if log.get('take_profit') else ""}
                    </div>
                    {f"<div style='margin-top: 5px; color: #d1d5db; font-size: 0.85rem;'><strong>Reason:</strong> {log.get('reason', 'N/A')}</div>" if log.get('reason') else ""}
                    {f"<div style='margin-top: 5px; color: #d1d5db; font-size: 0.85rem;'><strong>ROI:</strong> {log.get('roi', 0):.2f}%</div>" if log.get('roi') else ""}
                </div>
                """, unsafe_allow_html=True)
            
            # Summary Statistics
            st.markdown("---")
            st.subheader("Trade Summary Statistics")
            
            closed_trades = [log for log in st.session_state.trade_log if log['action'] == 'CLOSE']
            
            if closed_trades:
                total_pnl = sum(log.get('pnl', 0) for log in closed_trades)
                winning_trades = [log for log in closed_trades if log.get('pnl', 0) > 0]
                losing_trades = [log for log in closed_trades if log.get('pnl', 0) < 0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Closed Trades", len(closed_trades))
                with col2:
                    st.metric("Winning Trades", len(winning_trades))
                with col3:
                    st.metric("Losing Trades", len(losing_trades))
                with col4:
                    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                col5, col6 = st.columns(2)
                with col5:
                    avg_win = np.mean([log.get('pnl', 0) for log in winning_trades]) if winning_trades else 0
                    st.metric("Average Win", f"${avg_win:,.2f}")
                with col6:
                    avg_loss = np.mean([log.get('pnl', 0) for log in losing_trades]) if losing_trades else 0
                    st.metric("Average Loss", f"${avg_loss:,.2f}")
                
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
                st.info("No closed trades yet")
        else:
            st.info("No trade history yet. Execute some trades to see them here!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><strong>⚠️ RISK DISCLAIMER:</strong> This is a paper trading simulation for educational purposes only.</p>
        <p>Past performance does not guarantee future results. Trading involves substantial risk of loss.</p>
        <p style='margin-top: 10px; font-size: 0.9rem;'>RTV SMC Algo Trading Terminal v2.0 • © 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
