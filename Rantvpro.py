# institutional_trading_bot.py - Professional Autonomous Algorithmic Trading System
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time as tm
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass, asdict
from enum import Enum

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Institutional Trading Terminal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Risk Management Constants
MAX_PORTFOLIO_RISK = 0.02
MAX_DRAWDOWN_THRESHOLD = 0.15
MAX_DAILY_LOSS = 0.05
MIN_SHARPE_RATIO = 1.5
COMMISSION_RATE = 0.001
MIN_PROFIT_TARGET = 0.015
MAX_HOLDING_PERIOD = 240

class SystemState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CIRCUIT_BREAKER = "circuit_breaker"

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    .stApp {
        background: #0a0e17;
        color: #f8fafc;
    }
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #334155;
        position: relative;
    }
    .system-status {
        position: absolute;
        top: 20px;
        right: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(0, 0, 0, 0.3);
        padding: 8px 16px;
        border-radius: 20px;
    }
    .metric-card {
        background: linear-gradient(145deg, #141b2d, #1a2238);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .status-active {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #10b981;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .signal-card {
        background: #141b2d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Position:
    id: str
    asset: str
    entry_time: datetime
    entry_price: float
    quantity: float
    side: str
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    strategy: str = ""

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.system_state = SystemState.ACTIVE
        st.session_state.portfolio = {
            'cash': 100000.00,
            'equity': 100000.00,
            'positions': {},
            'trade_history': [],
            'equity_curve': [100000.00],
            'timestamps': [datetime.now()]
        }
        st.session_state.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_commission': 0.0
        }
        st.session_state.alerts = []
        st.session_state.circuit_breaker = False

init_session_state()

# ============================================================================
# COMPREHENSIVE ASSET CONFIGURATION
# ============================================================================

ASSETS = {
    "Cryptocurrencies": {
        "BTC/USD": {"symbol": "BTC-USD", "min_size": 0.001, "max_size": 1.0, "typical_spread": 0.001},
        "ETH/USD": {"symbol": "ETH-USD", "min_size": 0.01, "max_size": 10.0, "typical_spread": 0.001},
        "SOL/USD": {"symbol": "SOL-USD", "min_size": 0.1, "max_size": 100.0, "typical_spread": 0.002},
        "XRP/USD": {"symbol": "XRP-USD", "min_size": 10, "max_size": 10000, "typical_spread": 0.001},
        "ADA/USD": {"symbol": "ADA-USD", "min_size": 10, "max_size": 10000, "typical_spread": 0.001},
        "DOGE/USD": {"symbol": "DOGE-USD", "min_size": 100, "max_size": 100000, "typical_spread": 0.002},
        "AVAX/USD": {"symbol": "AVAX-USD", "min_size": 1, "max_size": 1000, "typical_spread": 0.002},
        "DOT/USD": {"symbol": "DOT-USD", "min_size": 1, "max_size": 1000, "typical_spread": 0.002},
        "MATIC/USD": {"symbol": "MATIC-USD", "min_size": 10, "max_size": 10000, "typical_spread": 0.002},
        "LINK/USD": {"symbol": "LINK-USD", "min_size": 1, "max_size": 1000, "typical_spread": 0.002}
    },
    "Major Forex Pairs": {
        "EUR/USD": {"symbol": "EURUSD=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.00002},
        "GBP/USD": {"symbol": "GBPUSD=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.00003},
        "USD/JPY": {"symbol": "JPY=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.003},
        "USD/CHF": {"symbol": "CHF=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.00003},
        "AUD/USD": {"symbol": "AUDUSD=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.00003},
        "USD/CAD": {"symbol": "CAD=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.00003},
        "NZD/USD": {"symbol": "NZDUSD=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.00004}
    },
    "Cross Currency Pairs": {
        "EUR/GBP": {"symbol": "EURGBP=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.00003},
        "EUR/JPY": {"symbol": "EURJPY=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.003},
        "GBP/JPY": {"symbol": "GBPJPY=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.004},
        "AUD/JPY": {"symbol": "AUDJPY=X", "min_size": 1000, "max_size": 100000, "typical_spread": 0.003}
    },
    "Precious Metals": {
        "Gold (XAU/USD)": {"symbol": "GC=F", "min_size": 0.1, "max_size": 10, "typical_spread": 0.10},
        "Silver (XAG/USD)": {"symbol": "SI=F", "min_size": 1, "max_size": 100, "typical_spread": 0.01},
        "Platinum": {"symbol": "PL=F", "min_size": 0.1, "max_size": 10, "typical_spread": 0.50},
        "Palladium": {"symbol": "PA=F", "min_size": 0.1, "max_size": 10, "typical_spread": 1.00},
        "Copper": {"symbol": "HG=F", "min_size": 1, "max_size": 100, "typical_spread": 0.01}
    },
    "Energy Commodities": {
        "Crude Oil (WTI)": {"symbol": "CL=F", "min_size": 1, "max_size": 100, "typical_spread": 0.01},
        "Brent Crude": {"symbol": "BZ=F", "min_size": 1, "max_size": 100, "typical_spread": 0.01},
        "Natural Gas": {"symbol": "NG=F", "min_size": 10, "max_size": 1000, "typical_spread": 0.001},
        "Heating Oil": {"symbol": "HO=F", "min_size": 1, "max_size": 100, "typical_spread": 0.01},
        "Gasoline": {"symbol": "RB=F", "min_size": 1, "max_size": 100, "typical_spread": 0.01}
    },
    "Agricultural Commodities": {
        "Corn": {"symbol": "ZC=F", "min_size": 10, "max_size": 1000, "typical_spread": 0.25},
        "Wheat": {"symbol": "ZW=F", "min_size": 10, "max_size": 1000, "typical_spread": 0.25},
        "Soybeans": {"symbol": "ZS=F", "min_size": 10, "max_size": 1000, "typical_spread": 0.25},
        "Coffee": {"symbol": "KC=F", "min_size": 1, "max_size": 100, "typical_spread": 0.05},
        "Sugar": {"symbol": "SB=F", "min_size": 10, "max_size": 1000, "typical_spread": 0.01},
        "Cotton": {"symbol": "CT=F", "min_size": 1, "max_size": 100, "typical_spread": 0.01},
        "Cocoa": {"symbol": "CC=F", "min_size": 1, "max_size": 100, "typical_spread": 1.0}
    },
    "Major Indices": {
        "S&P 500": {"symbol": "^GSPC", "min_size": 1, "max_size": 100, "typical_spread": 0.25},
        "NASDAQ": {"symbol": "^IXIC", "min_size": 1, "max_size": 100, "typical_spread": 0.50},
        "Dow Jones": {"symbol": "^DJI", "min_size": 1, "max_size": 100, "typical_spread": 1.0},
        "Russell 2000": {"symbol": "^RUT", "min_size": 1, "max_size": 100, "typical_spread": 0.10},
        "DAX (Germany)": {"symbol": "^GDAXI", "min_size": 1, "max_size": 100, "typical_spread": 1.0},
        "FTSE 100 (UK)": {"symbol": "^FTSE", "min_size": 1, "max_size": 100, "typical_spread": 0.50},
        "CAC 40 (France)": {"symbol": "^FCHI", "min_size": 1, "max_size": 100, "typical_spread": 0.50},
        "Nikkei 225": {"symbol": "^N225", "min_size": 1, "max_size": 100, "typical_spread": 10.0},
        "Hang Seng": {"symbol": "^HSI", "min_size": 1, "max_size": 100, "typical_spread": 5.0}
    },
    "Tech Stocks": {
        "Apple": {"symbol": "AAPL", "min_size": 1, "max_size": 1000, "typical_spread": 0.01},
        "Microsoft": {"symbol": "MSFT", "min_size": 1, "max_size": 1000, "typical_spread": 0.01},
        "Amazon": {"symbol": "AMZN", "min_size": 1, "max_size": 500, "typical_spread": 0.01},
        "Tesla": {"symbol": "TSLA", "min_size": 1, "max_size": 500, "typical_spread": 0.01},
        "NVIDIA": {"symbol": "NVDA", "min_size": 1, "max_size": 500, "typical_spread": 0.02},
        "Meta": {"symbol": "META", "min_size": 1, "max_size": 500, "typical_spread": 0.02},
        "Alphabet": {"symbol": "GOOGL", "min_size": 1, "max_size": 500, "typical_spread": 0.02},
        "Netflix": {"symbol": "NFLX", "min_size": 1, "max_size": 500, "typical_spread": 0.05}
    },
    "Bank & Finance Stocks": {
        "JPMorgan": {"symbol": "JPM", "min_size": 1, "max_size": 1000, "typical_spread": 0.01},
        "Bank of America": {"symbol": "BAC", "min_size": 10, "max_size": 5000, "typical_spread": 0.01},
        "Goldman Sachs": {"symbol": "GS", "min_size": 1, "max_size": 500, "typical_spread": 0.02},
        "Morgan Stanley": {"symbol": "MS", "min_size": 1, "max_size": 1000, "typical_spread": 0.01},
        "Citigroup": {"symbol": "C", "min_size": 1, "max_size": 2000, "typical_spread": 0.01}
    }
}

# ============================================================================
# DATA PROVIDER
# ============================================================================

@st.cache_data(ttl=60)
def fetch_market_data(symbol: str, interval: str = '5m') -> Optional[pd.DataFrame]:
    try:
        interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '60m'}
        yf_interval = interval_map.get(interval, '5m')
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='1d', interval=yf_interval)
        
        if df.empty:
            return None
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = [col.lower() for col in df.columns]
        df = df.dropna()
        
        if len(df) < 20:
            return None
        
        df = add_indicators(df)
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    
    try:
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Bollinger Bands
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
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
        
    except Exception as e:
        pass
    
    return df

# ============================================================================
# STRATEGY ENGINE
# ============================================================================

def generate_signals(df: pd.DataFrame, asset_info: Dict) -> List[Dict]:
    signals = []
    
    if len(df) < 50:
        return signals
    
    latest = df.iloc[-1]
    current_price = latest['close']
    atr = latest['atr'] if not pd.isna(latest['atr']) else current_price * 0.02
    
    # Strategy 1: RSI Oversold + BB
    if (latest['rsi'] < 30 and 
        latest['close'] < latest['bb_lower'] and
        latest['volume_ratio'] > 1.2):
        
        confidence = 0.75
        if latest['rsi'] < 25:
            confidence += 0.05
        if latest['volume_ratio'] > 1.5:
            confidence += 0.05
            
        signals.append({
            'asset': asset_info['symbol'],
            'type': 'BUY',
            'entry': current_price,
            'stop_loss': current_price - (atr * 1.5),
            'take_profit': latest['bb_middle'],
            'confidence': min(confidence, 0.95),
            'strategy': 'RSI Mean Reversion',
            'reason': f'Oversold RSI: {latest["rsi"]:.1f}, Below BB'
        })
    
    # Strategy 2: Momentum Breakout
    if (latest['close'] > latest['bb_upper'] and
        latest['rsi'] > 55 and latest['rsi'] < 75 and
        latest['volume_ratio'] > 1.8 and
        latest['macd_hist'] > 0):
        
        confidence = 0.72
        if latest['volume_ratio'] > 2.5:
            confidence += 0.08
            
        signals.append({
            'asset': asset_info['symbol'],
            'type': 'BUY',
            'entry': current_price,
            'stop_loss': latest['bb_middle'],
            'take_profit': current_price + (current_price - latest['bb_middle']) * 2,
            'confidence': min(confidence, 0.95),
            'strategy': 'Momentum Breakout',
            'reason': f'BB Breakout, RSI: {latest["rsi"]:.1f}, Strong Volume'
        })
    
    # Strategy 3: EMA Crossover
    if (latest['ema_9'] > latest['ema_21'] and
        df.iloc[-2]['ema_9'] <= df.iloc[-2]['ema_21'] and
        latest['rsi'] > 50 and
        latest['volume_ratio'] > 1.3):
        
        signals.append({
            'asset': asset_info['symbol'],
            'type': 'BUY',
            'entry': current_price,
            'stop_loss': current_price - (atr * 2.0),
            'take_profit': current_price + (atr * 3.0),
            'confidence': 0.78,
            'strategy': 'EMA Crossover',
            'reason': 'Bullish EMA 9/21 crossover'
        })
    
    # SELL Signals
    if (latest['rsi'] > 70 and
        latest['close'] > latest['bb_upper'] and
        latest['volume_ratio'] > 1.2):
        
        signals.append({
            'asset': asset_info['symbol'],
            'type': 'SELL',
            'entry': current_price,
            'stop_loss': current_price + (atr * 1.5),
            'take_profit': latest['bb_middle'],
            'confidence': 0.73,
            'strategy': 'Overbought Reversal',
            'reason': f'Overbought RSI: {latest["rsi"]:.1f}, Above BB'
        })
    
    # Filter by confidence
    signals = [s for s in signals if s['confidence'] >= 0.70]
    signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
    
    return signals[:3]

# ============================================================================
# RISK MANAGER
# ============================================================================

def calculate_position_size(signal: Dict, asset_info: Dict) -> float:
    equity = st.session_state.portfolio['equity']
    max_risk = equity * MAX_PORTFOLIO_RISK
    
    stop_distance = abs(signal['entry'] - signal['stop_loss'])
    if stop_distance == 0:
        return 0.0
    
    position_size = max_risk / stop_distance
    position_size = min(position_size, asset_info['max_size'])
    position_size = max(position_size, asset_info['min_size'])
    
    return round(position_size, 8)

def check_trade_allowed(signal: Dict) -> Tuple[bool, str]:
    if st.session_state.system_state != SystemState.ACTIVE:
        return False, f"System {st.session_state.system_state.value}"
    
    if st.session_state.circuit_breaker:
        return False, "Circuit breaker active"
    
    if len(st.session_state.portfolio['positions']) >= 5:
        return False, "Max positions reached"
    
    # Check risk/reward
    risk = abs(signal['entry'] - signal['stop_loss'])
    reward = abs(signal['take_profit'] - signal['entry'])
    if risk > 0 and reward / risk < 1.5:
        return False, "R:R ratio too low"
    
    return True, "Approved"

# ============================================================================
# EXECUTION ENGINE
# ============================================================================

def execute_trade(signal: Dict, asset_info: Dict) -> Tuple[bool, str]:
    position_size = calculate_position_size(signal, asset_info)
    
    if position_size == 0:
        return False, "Invalid position size"
    
    allowed, reason = check_trade_allowed(signal)
    if not allowed:
        return False, reason
    
    # Calculate costs
    current_price = signal['entry']
    commission = position_size * current_price * COMMISSION_RATE
    slippage = current_price * asset_info.get('typical_spread', 0.001)
    
    if signal['type'] == 'BUY':
        execution_price = current_price + slippage
    else:
        execution_price = current_price - slippage
    
    total_cost = (position_size * execution_price) + commission
    
    if total_cost > st.session_state.portfolio['cash']:
        return False, "Insufficient cash"
    
    # Create position
    pos_id = f"{asset_info['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"
    
    position = Position(
        id=pos_id,
        asset=asset_info['symbol'],
        entry_time=datetime.now(),
        entry_price=execution_price,
        quantity=position_size,
        side=signal['type'],
        stop_loss=signal['stop_loss'],
        take_profit=signal['take_profit'],
        current_price=execution_price,
        strategy=signal.get('strategy', 'Unknown')
    )
    
    # Update portfolio
    st.session_state.portfolio['cash'] -= total_cost
    st.session_state.portfolio['positions'][pos_id] = asdict(position)
    st.session_state.stats['total_commission'] += commission
    
    return True, pos_id

def update_positions(current_price: float, symbol: str):
    positions_to_close = []
    
    for pos_id, pos in list(st.session_state.portfolio['positions'].items()):
        if pos['asset'] == symbol:
            # Calculate P&L
            if pos['side'] == 'BUY':
                pnl = (current_price - pos['entry_price']) * pos['quantity']
            else:
                pnl = (pos['entry_price'] - current_price) * pos['quantity']
            
            pos['current_price'] = current_price
            pos['unrealized_pnl'] = pnl
            
            # Check exit conditions
            if pos['side'] == 'BUY':
                if current_price <= pos['stop_loss']:
                    positions_to_close.append((pos_id, current_price, "Stop Loss"))
                elif current_price >= pos['take_profit']:
                    positions_to_close.append((pos_id, current_price, "Take Profit"))
            else:
                if current_price >= pos['stop_loss']:
                    positions_to_close.append((pos_id, current_price, "Stop Loss"))
                elif current_price <= pos['take_profit']:
                    positions_to_close.append((pos_id, current_price, "Take Profit"))
    
    for pos_id, price, reason in positions_to_close:
        close_position(pos_id, price, reason)

def close_position(pos_id: str, close_price: float, reason: str):
    if pos_id not in st.session_state.portfolio['positions']:
        return
    
    pos = st.session_state.portfolio['positions'][pos_id]
    
    # Calculate P&L
    if pos['side'] == 'BUY':
        pnl = (close_price - pos['entry_price']) * pos['quantity']
    else:
        pnl = (pos['entry_price'] - close_price) * pos['quantity']
    
    commission = pos['quantity'] * close_price * COMMISSION_RATE
    pnl -= commission
    
    # Update portfolio
    st.session_state.portfolio['cash'] += (pos['quantity'] * close_price)
    st.session_state.portfolio['equity'] = st.session_state.portfolio['cash']
    
    # Update stats
    stats = st.session_state.stats
    stats['total_trades'] += 1
    stats['total_commission'] += commission
    
    if pnl > 0:
        stats['winning_trades'] += 1
        stats['avg_win'] = ((stats['avg_win'] * (stats['winning_trades'] - 1)) + pnl) / stats['winning_trades']
    else:
        stats['losing_trades'] += 1
        stats['avg_loss'] = ((stats['avg_loss'] * (stats['losing_trades'] - 1)) + abs(pnl)) / stats['losing_trades']
    
    total = stats['winning_trades'] + stats['losing_trades']
    stats['win_rate'] = (stats['winning_trades'] / total * 100) if total > 0 else 0
    
    if stats['avg_loss'] > 0:
        stats['profit_factor'] = stats['avg_win'] / stats['avg_loss']
    
    # Record trade
    trade_record = {
        **pos,
        'close_time': datetime.now(),
        'close_price': close_price,
        'pnl': pnl,
        'reason': reason
    }
    st.session_state.portfolio['trade_history'].append(trade_record)
    
    # Update equity curve
    st.session_state.portfolio['equity_curve'].append(st.session_state.portfolio['equity'])
    st.session_state.portfolio['timestamps'].append(datetime.now())
    
    # Remove position
    del st.session_state.portfolio['positions'][pos_id]

# ============================================================================
# USER INTERFACE
# ============================================================================

def render_header():
    state = st.session_state.system_state
    state_color = "status-active" if state == SystemState.ACTIVE else "status-paused"
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="text-align: center; font-size: 2.5rem; font-weight: 800; 
                   background: linear-gradient(45deg, #60a5fa, #8b5cf6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏦 INSTITUTIONAL ALGORITHMIC TRADING SYSTEM
        </h1>
        <p style="text-align: center; color: #94a3b8; margin: 10px 0;">
            Professional-Grade Autonomous Trading Platform • 90+ Assets
        </p>
        <div class="system-status">
            <div class="{state_color}"></div>
            <span style="color: #f8fafc; font-weight: 600;">{state.value.upper()}</span>
        </div>
