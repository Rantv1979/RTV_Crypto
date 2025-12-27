# institutional_trading_bot.py - Professional Autonomous Algorithmic Trading System
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time as tm
import json
import os
from typing import Dict, List, Optional, Tuple, Set
import random
import traceback
from collections import defaultdict, deque
import pytz
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import logging

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title="Institutional Trading Terminal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Risk Management Constants
MAX_PORTFOLIO_RISK = 0.02  # 2% max portfolio risk
MAX_CORRELATION_EXPOSURE = 0.3  # 30% max correlated positions
MAX_DRAWDOWN_THRESHOLD = 0.15  # 15% max drawdown before circuit breaker
MAX_DAILY_LOSS = 0.05  # 5% max daily loss
MIN_SHARPE_RATIO = 1.5
MAX_LEVERAGE = 1.0  # No leverage for safety
POSITION_SIZING_METHOD = "kelly_criterion"  # or "fixed_fractional"

# Data Quality Thresholds
MIN_DATA_POINTS = 100
MAX_MISSING_DATA_PCT = 0.05  # 5% max missing data
MAX_PRICE_DEVIATION = 0.10  # 10% max price deviation from moving average
MIN_VOLUME_THRESHOLD = 1000

# Trading Parameters
SLIPPAGE_MODEL = "realistic"  # realistic, conservative, optimistic
COMMISSION_RATE = 0.001  # 0.1% per trade
MIN_PROFIT_TARGET = 0.015  # 1.5% minimum profit target
MAX_HOLDING_PERIOD = 240  # minutes

# System States
class SystemState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CIRCUIT_BREAKER = "circuit_breaker"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================

st.markdown("""
<style>
    :root {
        --primary-bg: #0a0e17;
        --secondary-bg: #141b2d;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #334155;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
    }
    
    .stApp {
        background: var(--primary-bg);
        color: var(--text-primary);
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
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
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-active { background: var(--success); }
    .status-paused { background: var(--warning); }
    .status-error { background: var(--danger); }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .metric-card {
        background: linear-gradient(145deg, var(--secondary-bg), #1a2238);
        padding: 1.5rem;
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
        background: linear-gradient(180deg, var(--accent-blue), #8b5cf6);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
    }
    
    .risk-indicator {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-low { background: rgba(16, 185, 129, 0.2); color: var(--success); }
    .risk-medium { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
    .risk-high { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
    .risk-critical { background: rgba(239, 68, 68, 0.4); color: #fff; }
    
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.1);
        border-color: var(--info);
        color: var(--info);
    }
    
    .alert-success {
        background: rgba(16, 185, 129, 0.1);
        border-color: var(--success);
        color: var(--success);
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border-color: var(--warning);
        color: var(--warning);
    }
    
    .alert-danger {
        background: rgba(239, 68, 68, 0.1);
        border-color: var(--danger);
        color: var(--danger);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Order:
    id: str
    timestamp: datetime
    asset: str
    order_type: OrderType
    side: str  # BUY or SELL
    quantity: float
    price: float
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
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
    realized_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    strategy: str = ""
    risk_score: float = 0.0
    
@dataclass
class RiskMetrics:
    portfolio_var: float  # Value at Risk
    portfolio_cvar: float  # Conditional VaR
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    overall_risk_score: float

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.system_state = SystemState.ACTIVE
        st.session_state.portfolio = {
            'cash': 100000.00,
            'equity': 100000.00,
            'positions': {},
            'orders': [],
            'trade_history': [],
            'equity_curve': [100000.00],
            'daily_pnl': [],
            'timestamps': [datetime.now()]
        }
        st.session_state.risk_metrics = RiskMetrics(
            portfolio_var=0.0,
            portfolio_cvar=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            liquidity_risk=0.0,
            overall_risk_score=0.0
        )
        st.session_state.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_holding_time': 0.0,
            'total_commission': 0.0,
            'total_slippage': 0.0
        }
        st.session_state.alerts = []
        st.session_state.market_data_cache = {}
        st.session_state.last_update = datetime.now()
        st.session_state.circuit_breaker_triggered = False
        st.session_state.daily_loss_limit_hit = False

initialize_session_state()

# ============================================================================
# ENHANCED ASSET CONFIGURATION
# ============================================================================

ASSET_CONFIG = {
    "Cryptocurrencies": {
        "BTC/USD": {
            "symbol": "BTC-USD",
            "pip_size": 0.01,
            "lot_size": 0.001,
            "min_trade_size": 0.001,
            "max_trade_size": 1.0,
            "typical_spread": 0.001,
            "avg_daily_volume": 50000000000,
            "volatility_regime": "high",
            "correlation_group": "crypto"
        },
        "ETH/USD": {
            "symbol": "ETH-USD",
            "pip_size": 0.01,
            "lot_size": 0.01,
            "min_trade_size": 0.01,
            "max_trade_size": 10.0,
            "typical_spread": 0.001,
            "avg_daily_volume": 20000000000,
            "volatility_regime": "high",
            "correlation_group": "crypto"
        }
    },
    "Forex": {
        "EUR/USD": {
            "symbol": "EURUSD=X",
            "pip_size": 0.0001,
            "lot_size": 10000,
            "min_trade_size": 1000,
            "max_trade_size": 100000,
            "typical_spread": 0.00002,
            "avg_daily_volume": 1000000000000,
            "volatility_regime": "low",
            "correlation_group": "forex_major"
        },
        "GBP/USD": {
            "symbol": "GBPUSD=X",
            "pip_size": 0.0001,
            "lot_size": 10000,
            "min_trade_size": 1000,
            "max_trade_size": 100000,
            "typical_spread": 0.00003,
            "avg_daily_volume": 500000000000,
            "volatility_regime": "medium",
            "correlation_group": "forex_major"
        }
    },
    "Indices": {
        "S&P 500": {
            "symbol": "^GSPC",
            "pip_size": 0.25,
            "lot_size": 1,
            "min_trade_size": 1,
            "max_trade_size": 100,
            "typical_spread": 0.25,
            "avg_daily_volume": 100000000000,
            "volatility_regime": "medium",
            "correlation_group": "us_equity"
        },
        "NASDAQ": {
            "symbol": "^IXIC",
            "pip_size": 0.25,
            "lot_size": 1,
            "min_trade_size": 1,
            "max_trade_size": 100,
            "typical_spread": 0.50,
            "avg_daily_volume": 80000000000,
            "volatility_regime": "high",
            "correlation_group": "us_equity"
        }
    }
}

# ============================================================================
# DATA QUALITY & VALIDATION
# ============================================================================

class DataValidator:
    """Comprehensive data quality validation"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, asset: str) -> Tuple[bool, List[str]]:
        """Validate market data quality"""
        errors = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
            
        # Check minimum data points
        if len(df) < MIN_DATA_POINTS:
            errors.append(f"Insufficient data: {len(df)} < {MIN_DATA_POINTS}")
            
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > MAX_MISSING_DATA_PCT:
            errors.append(f"Too many missing values: {missing_pct:.1%}")
            
        # Check for price anomalies
        if 'close' in df.columns:
            price_changes = df['close'].pct_change()
            extreme_moves = abs(price_changes) > MAX_PRICE_DEVIATION
            if extreme_moves.any():
                errors.append(f"Extreme price movements detected: {extreme_moves.sum()} instances")
                
        # Check volume
        if 'volume' in df.columns:
            zero_volume = (df['volume'] < MIN_VOLUME_THRESHOLD).sum()
            if zero_volume > len(df) * 0.1:
                errors.append(f"Low volume periods: {zero_volume} candles")
                
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            errors.append("Duplicate timestamps detected")
            
        # Check chronological order
        if not df.index.is_monotonic_increasing:
            errors.append("Data not in chronological order")
            
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        df = df.copy()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by index
        df = df.sort_index()
        
        # Forward fill missing values (conservative approach)
        df = df.fillna(method='ffill', limit=3)
        
        # Drop remaining NaN
        df = df.dropna()
        
        # Remove extreme outliers (beyond 5 sigma)
        if 'close' in df.columns:
            z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            df = df[z_scores < 5]
        
        return df
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame) -> List[Dict]:
        """Detect data anomalies"""
        anomalies = []
        
        if 'close' in df.columns:
            # Price gaps
            price_changes = df['close'].pct_change()
            large_gaps = price_changes[abs(price_changes) > 0.05]
            
            for idx, change in large_gaps.items():
                anomalies.append({
                    'timestamp': idx,
                    'type': 'price_gap',
                    'severity': 'high' if abs(change) > 0.10 else 'medium',
                    'value': change,
                    'message': f"Large price gap: {change:.2%}"
                })
        
        return anomalies

# ============================================================================
# ENHANCED DATA FETCHING WITH REDUNDANCY
# ============================================================================

class DataProvider:
    """Multi-source data provider with failover"""
    
    def __init__(self):
        self.primary_source = "yfinance"
        self.cache = {}
        self.cache_ttl = 60  # seconds
        self.validator = DataValidator()
        
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch data with caching and validation"""
        cache_key = f"{symbol}_{interval}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
        
        # Fetch fresh data
        try:
            df = self._fetch_yfinance(symbol, interval, period)
            
            if df is not None and not df.empty:
                # Validate data
                is_valid, errors = self.validator.validate_dataframe(df, symbol)
                
                if not is_valid:
                    logger.warning(f"Data validation failed for {symbol}: {errors}")
                    # Try to clean data
                    df = self.validator.clean_data(df)
                    is_valid, errors = self.validator.validate_dataframe(df, symbol)
                    
                    if not is_valid:
                        logger.error(f"Data cleaning failed for {symbol}")
                        return None
                
                # Add indicators
                df = self._add_technical_indicators(df)
                
                # Cache data
                self.cache[cache_key] = (df, datetime.now())
                return df
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
            
        return None
    
    def _fetch_yfinance(self, symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch from yfinance"""
        try:
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '60m'}
            yf_interval = interval_map.get(interval, '5m')
            
            period_map = {'1m': '1d', '5m': '1d', '15m': '5d', '30m': '5d', '1h': '5d'}
            yf_period = period_map.get(interval, '1d')
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=yf_period, interval=yf_interval)
            
            if df.empty:
                # Fallback
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                df = yf.download(symbol, start=start_date, end=end_date, 
                               interval=yf_interval, progress=False, auto_adjust=True)
            
            if not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.columns = [col.lower() for col in df.columns]
                df = df.dropna()
                return df
                
        except Exception as e:
            logger.error(f"yfinance fetch error: {str(e)}")
            
        return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        if len(df) < 50:
            return df
            
        df = df.copy()
        
        try:
            # Price metrics
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # ATR (14-period)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            df['atr_percent'] = df['atr'] / df['close']
            
            # RSI (14-period)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_delta'] = df['volume'].diff()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
            
            # Support/Resistance
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            df['range'] = df['resistance'] - df['support']
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['roc'] = df['close'].pct_change(periods=10)
            
            # Trend strength (ADX approximation)
            high_diff = df['high'].diff()
            low_diff = df['low'].diff().abs()
            df['plus_dm'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            df['minus_dm'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
        except Exception as e:
            logger.error(f"Error adding indicators: {str(e)}")
            
        return df

# ============================================================================
# ADVANCED RISK MANAGEMENT
# ============================================================================

class RiskManager:
    """Institutional-grade risk management"""
    
    def __init__(self):
        self.portfolio = st.session_state.portfolio
        self.risk_metrics = st.session_state.risk_metrics
        
    def calculate_position_size(self, 
                                signal: Dict, 
                                current_price: float,
                                asset_config: Dict) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        
        # Get risk parameters
        account_equity = self.portfolio['equity']
        max_risk_per_trade = account_equity * MAX_PORTFOLIO_RISK
        
        # Calculate stop distance
        stop_distance = abs(signal['entry'] - signal['stop_loss'])
        if stop_distance == 0:
            return 0.0
        
        # Kelly Criterion adjustment
        win_rate = st.session_state.performance_stats['win_rate'] / 100 if st.session_state.performance_stats['win_rate'] > 0 else 0.5
        profit_factor = st.session_state.performance_stats['profit_factor'] if st.session_state.performance_stats['profit_factor'] > 0 else 1.5
        
        # Kelly formula: f = (bp - q) / b where:
        # f = fraction of capital to wager
        # b = odds received on wager (profit factor)
        # p = probability of winning (win rate)
        # q = probability of losing (1 - win rate)
        
        kelly_fraction = ((profit_factor * win_rate) - (1 - win_rate)) / profit_factor
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust for confidence
        confidence_adj = signal.get('confidence', 0.7)
        adjusted_fraction = kelly_fraction * confidence_adj * 0.5  # Half-Kelly for safety
        
        # Calculate position size
        max_position_value = account_equity * adjusted_fraction
        position_size = max_position_value / current_price
        
        # Apply risk limit per trade
        risk_based_size = max_risk_per_trade / stop_distance
        position_size = min(position_size, risk_based_size)
        
        # Apply asset limits
        position_size = min(position_size, asset_config['max_trade_size'])
        position_size = max(position_size, asset_config['min_trade_size'])
        
        return round(position_size, 8)
    
    def check_trade_allowed(self, signal: Dict, position_size: float) -> Tuple[bool, str]:
        """Comprehensive trade validation"""
        
        # Check system state
        if st.session_state.system_state != SystemState.ACTIVE:
            return False, f"System state: {st.session_state.system_state.value}"
        
        # Check circuit breaker
        if st.session_state.circuit_breaker_triggered:
            return False, "Circuit breaker active - max drawdown exceeded"
        
        # Check daily loss limit
        if st.session_state.daily_loss_limit_hit:
            return False, "Daily loss limit reached"
        
        # Check max positions
        max_positions = 5
        if len(self.portfolio['positions']) >= max_positions:
            return False, f"Maximum positions ({max_positions}) reached"
        
        # Check correlation risk
        correlation_group = signal.get('correlation_group', 'unknown')
        correlated_positions = sum(1 for p in self.portfolio['positions'].values() 
                                  if p.get('correlation_group') == correlation_group)
        
        max_correlated = 2
        if correlated_positions >= max_correlated:
            return False, f"Too many correlated positions in {correlation_group}"
        
        # Check concentration risk
        position_value = position_size * signal['entry']
        max_position_value = self.portfolio['equity'] * 0.20  # 20% max per position
        
        if position_value > max_position_value:
            return False, f"Position size exceeds 20% of equity"
        
        # Check minimum profit target
        potential_profit = abs(signal['take_profit'] - signal['entry']) / signal['entry']
        if potential_profit < MIN_PROFIT_TARGET:
            return False, f"Profit target too small: {potential_profit:.2%}"
        
        # Check risk/reward ratio
        risk = abs(signal['entry'] - signal['stop_loss'])
        reward = abs(signal['take_profit'] - signal['entry'])
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < 1.5:
            return False, f"Risk/reward ratio too low: {rr_ratio:.2f}"
        
        return True, "Trade approved"
    
    def calculate_portfolio_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        equity_curve = self.portfolio['equity_curve']
        positions = self.portfolio['positions']
        
        if len(equity_curve) < 2:
            return self.risk_metrics
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        if len(returns) < 2:
            return self.risk_metrics
        
        # Sharpe Ratio (annualized)
        risk_free_rate = 0.02 / 252  # 2% annual, daily rate
        excess_returns = returns - risk_free_rate
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino = np.sqrt(252) * (excess_returns.mean() / downside_std) if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        current_drawdown = abs(drawdown.iloc[-1])
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Correlation risk (average correlation of open positions)
        correlation_risk = 0.0
        if len(positions) > 1:
            # Simplified correlation estimate based on correlation groups
            groups = [p.get('correlation_group', 'unknown') for p in positions.values()]
            unique_groups = len(set(groups))
            correlation_risk = 1 - (unique_groups / len(positions))
        
        # Concentration risk (largest position as % of portfolio)
        concentration_risk = 0.0
        if positions:
            position_values = [abs(p.quantity * p.current_price) for p in positions.values()]
            if sum(position_values) > 0:
                concentration_risk = max(position_values) / sum(position_values)
        
        # Liquidity risk (simplified)
        liquidity_risk = len(positions) / 10  # Assumes max 10 positions
        
        # Overall risk score (weighted average)
        overall_risk = (
            0.3 * min(max_drawdown / MAX_DRAWDOWN_THRESHOLD, 1.0) +
            0.2 * correlation_risk +
            0.2 * concentration_risk +
            0.15 * liquidity_risk +
            0.15 * (1 - min(sharpe / MIN_SHARPE_RATIO, 1.0))
        )
        
        return RiskMetrics(
            portfolio_var=var_95,
            portfolio_cvar=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            overall_risk_score=overall_risk
        )
    
    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should trigger"""
        metrics = self.calculate_portfolio_metrics()
        
        # Drawdown circuit breaker
        if metrics.current_drawdown > MAX_DRAWDOWN_THRESHOLD:
            st.session_state.circuit_breaker_triggered = True
            self._add_alert('CRITICAL', 'Circuit Breaker', 
                          f'Max drawdown exceeded: {metrics.current_drawdown:.2%}')
            return True
        
        # Daily loss limit
        daily_pnl = self.portfolio['daily_pnl']
        if daily_pnl and len(daily_pnl) > 0:
            today_pnl = sum(daily_pnl[-1:]) if isinstance(daily_pnl, list) else daily_pnl
            if today_pnl < -self.portfolio['equity'] * MAX_DAILY_LOSS:
                st.session_state.daily_loss_limit_hit = True
                self._add_alert('CRITICAL', 'Daily Loss Limit', 
                              f'Daily loss limit exceeded: ${today_pnl:.2f}')
                return True
        
        return False
    
    def _add_alert(self, severity: str, title: str, message: str):
        """Add system alert"""
        st.session_state.alerts.append({
            'timestamp': datetime.now(),
            'severity': severity,
            'title': title,
            'message': message
        })

# ============================================================================
# ADVANCED TRADING STRATEGIES
# ============================================================================

class InstitutionalStrategy:
    """Professional trading strategy with multiple algorithms"""
    
    def __init__(self):
        self.min_confidence = 0.75
        self.lookback_period = 50
        
    def analyze_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced market structure analysis"""
        df = df.copy()
        
        if len(df) < 20:
            return df
        
        # Identify swing points
        swing_period = 5
        df['swing_high'] = False
        df['swing_low'] = False
        
        for i in range(swing_period, len(df) - swing_period):
            # Swing High
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, swing_period+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, swing_period+1)):
                df.iloc[i, df.columns.get_loc('swing_high')] = True
            
            # Swing Low
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, swing_period+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, swing_period+1)):
                df.iloc[i, df.columns.get_loc('swing_low')] = True
        
        # Market structure bias
        swing_highs = df[df['swing_high']]['high'].tail(3)
        swing_lows = df[df['swing_low']]['low'].tail(3)
        
        df['trend'] = 'neutral'
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if swing_highs.iloc[-1] > swing_highs.iloc[-2] and swing_lows.iloc[-1] > swing_lows.iloc[-2]:
                df['trend'] = 'uptrend'
            elif swing_highs.iloc[-1] < swing_highs.iloc[-2] and swing_lows.iloc[-1] < swing_lows.iloc[-2]:
                df['trend'] = 'downtrend'
        
        return df
    
    def identify_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify institutional order blocks"""
        df = df.copy()
        df['ob_bullish'] = np.nan
        df['ob_bearish'] = np.nan
        df['ob_strength'] = 0.0
        
        for i in range(3, len(df) - 1):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            body = abs(current['close'] - current['open'])
            candle_range = current['high'] - current['low']
            
            if candle_range == 0:
                continue
            
            body_ratio = body / candle_range
            
            # Bullish Order Block
            if (body_ratio > 0.6 and 
                current['close'] > current['open'] and
                next_candle['low'] >= current['low']):
                
                volume_strength = current['volume'] / df['volume_sma'].iloc[i] if df['volume_sma'].iloc[i] > 0 else 1
                df.iloc[i, df.columns.get_loc('ob_bullish')] = current['low']
                df.iloc[i, df.columns.get_loc('ob_strength')] = min(body_ratio * volume_strength, 5.0)
            
            # Bearish Order Block
            elif (body_ratio > 0.6 and 
                  current['close'] < current['open'] and
                  next_candle['high'] <= current['high']):
                
                volume_strength = current['volume'] / df['volume_sma'].iloc[i] if df['volume_sma'].iloc[i] > 0 else 1
                df.iloc[i, df.columns.get_loc('ob_bearish')] = current['high']
                df.iloc[i, df.columns.get_loc('ob_strength')] = min(body_ratio * volume_strength, 5.0)
        
        return df
    
    def identify_liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify liquidity grab zones"""
        df = df.copy()
        df['liquidity_high'] = df['high'].rolling(window=20).max()
        df['liquidity_low'] = df['low'].rolling(window=20).min()
        df['liquidity_grabbed'] = False
        
        for i in range(20, len(df)):
            current = df.iloc[i]
            
            # Check if price swept liquidity
            if current['high'] >= df['liquidity_high'].iloc[i-1] or \
               current['low'] <= df['liquidity_low'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('liquidity_grabbed')] = True
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, asset_info: Dict) -> List[Dict]:
        """Generate high-probability trading signals"""
        signals = []
        
        if len(df) < 50:
            return signals
        
        # Analyze market structure
        df = self.analyze_market_structure(df)
        df = self.identify_order_blocks(df)
        df = self.identify_liquidity_zones(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = latest['close']
        atr = latest['atr'] if not pd.isna(latest['atr']) else current_price * 0.02
        
        # Strategy 1: Order Block + Trend Confirmation
        if not pd.isna(latest.get('ob_bullish', np.nan)):
            ob_level = latest['ob_bullish']
            distance_to_ob = abs(current_price - ob_level) / current_price
            
            if (distance_to_ob < 0.01 and  # Within 1% of OB
                latest['trend'] == 'uptrend' and
                latest['rsi'] < 50 and
                latest['volume_ratio'] > 1.2):
                
                confidence = 0.75
                confidence += 0.05 if latest['rsi'] < 40 else 0
                confidence += 0.05 if latest['volume_ratio'] > 1.5 else 0
                confidence += 0.05 if latest['ob_strength'] > 2.0 else 0
                
                signals.append({
                    'asset': asset_info['symbol'],
                    'asset_name': 'Asset',
                    'type': 'BUY',
                    'entry': current_price,
                    'stop_loss': ob_level - (atr * 1.5),
                    'take_profit': current_price + (atr * 3.0),
                    'confidence': min(confidence, 0.95),
                    'strategy': 'Order Block Bullish',
                    'reason': f'Bullish OB at {ob_level:.4f}, RSI: {latest["rsi"]:.1f}',
                    'correlation_group': asset_info.get('correlation_group', 'unknown'),
                    'timestamp': datetime.now()
                })
        
        # Strategy 2: Liquidity Grab Reversal
        if latest.get('liquidity_grabbed', False):
            if (latest['close'] > latest['open'] and  # Bullish reversal
                prev['close'] < prev['open'] and  # After bearish move
                latest['volume_ratio'] > 1.5 and
                latest['rsi'] > 30):
                
                confidence = 0.70
                confidence += 0.05 if latest['volume_ratio'] > 2.0 else 0
                confidence += 0.05 if latest['rsi'] > 40 else 0
                confidence += 0.05 if latest['trend'] == 'uptrend' else 0
                
                signals.append({
                    'asset': asset_info['symbol'],
                    'asset_name': 'Asset',
                    'type': 'BUY',
                    'entry': current_price,
                    'stop_loss': latest['low'] - (atr * 1.0),
                    'take_profit': current_price + (atr * 2.5),
                    'confidence': min(confidence, 0.95),
                    'strategy': 'Liquidity Grab Reversal',
                    'reason': f'Liquidity sweep reversal, Volume: {latest["volume_ratio"]:.1f}x',
                    'correlation_group': asset_info.get('correlation_group', 'unknown'),
                    'timestamp': datetime.now()
                })
        
        # Strategy 3: Momentum Breakout
        if (latest['close'] > latest['bb_upper'] and
            latest['rsi'] > 55 and latest['rsi'] < 75 and
            latest['volume_ratio'] > 1.8 and
            latest['trend'] == 'uptrend'):
            
            confidence = 0.72
            confidence += 0.05 if latest['macd_hist'] > 0 else 0
            confidence += 0.05 if latest['volume_ratio'] > 2.5 else 0
            
            signals.append({
                'asset': asset_info['symbol'],
                'asset_name': 'Asset',
                'type': 'BUY',
                'entry': current_price,
                'stop_loss': latest['bb_middle'],
                'take_profit': current_price + (current_price - latest['bb_middle']) * 2,
                'confidence': min(confidence, 0.95),
                'strategy': 'Momentum Breakout',
                'reason': f'BB breakout, RSI: {latest["rsi"]:.1f}, Vol: {latest["volume_ratio"]:.1f}x',
                'correlation_group': asset_info.get('correlation_group', 'unknown'),
                'timestamp': datetime.now()
            })
        
        # Strategy 4: Mean Reversion
        if (latest['rsi'] < 30 and
            latest['close'] < latest['bb_lower'] and
            latest['trend'] != 'downtrend' and
            latest['volume_ratio'] > 1.3):
            
            confidence = 0.68
            confidence += 0.05 if latest['rsi'] < 25 else 0
            confidence += 0.05 if latest['bb_position'] < 0.1 else 0
            
            signals.append({
                'asset': asset_info['symbol'],
                'asset_name': 'Asset',
                'type': 'BUY',
                'entry': current_price,
                'stop_loss': current_price - (atr * 2.0),
                'take_profit': latest['bb_middle'],
                'confidence': min(confidence, 0.95),
                'strategy': 'Mean Reversion',
                'reason': f'Oversold: RSI {latest["rsi"]:.1f}, Below BB',
                'correlation_group': asset_info.get('correlation_group', 'unknown'),
                'timestamp': datetime.now()
            })
        
        # SELL Signals (mirror logic)
        if not pd.isna(latest.get('ob_bearish', np.nan)):
            ob_level = latest['ob_bearish']
            distance_to_ob = abs(current_price - ob_level) / current_price
            
            if (distance_to_ob < 0.01 and
                latest['trend'] == 'downtrend' and
                latest['rsi'] > 50 and
                latest['volume_ratio'] > 1.2):
                
                confidence = 0.75
                confidence += 0.05 if latest['rsi'] > 60 else 0
                confidence += 0.05 if latest['volume_ratio'] > 1.5 else 0
                
                signals.append({
                    'asset': asset_info['symbol'],
                    'asset_name': 'Asset',
                    'type': 'SELL',
                    'entry': current_price,
                    'stop_loss': ob_level + (atr * 1.5),
                    'take_profit': current_price - (atr * 3.0),
                    'confidence': min(confidence, 0.95),
                    'strategy': 'Order Block Bearish',
                    'reason': f'Bearish OB at {ob_level:.4f}, RSI: {latest["rsi"]:.1f}',
                    'correlation_group': asset_info.get('correlation_group', 'unknown'),
                    'timestamp': datetime.now()
                })
        
        # Filter by minimum confidence
        signals = [s for s in signals if s['confidence'] >= self.min_confidence]
        
        # Sort by confidence
        signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        
        return signals[:3]  # Top 3 signals

# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """Professional order execution with realistic modeling"""
    
    def __init__(self):
        self.portfolio = st.session_state.portfolio
        self.risk_manager = RiskManager()
        
    def execute_order(self, signal: Dict, asset_info: Dict) -> Tuple[bool, str, Optional[Position]]:
        """Execute trade with comprehensive checks"""
        
        # Calculate position size
        current_price = signal['entry']
        position_size = self.risk_manager.calculate_position_size(signal, current_price, asset_info)
        
        if position_size == 0:
            return False, "Position size calculated as 0", None
        
        # Check if trade allowed
        allowed, reason = self.risk_manager.check_trade_allowed(signal, position_size)
        if not allowed:
            logger.warning(f"Trade rejected: {reason}")
            return False, reason, None
        
        # Calculate costs
        commission = position_size * current_price * COMMISSION_RATE
        slippage = self._calculate_slippage(current_price, position_size, asset_info)
        
        # Adjust entry price for slippage
        if signal['type'] == 'BUY':
            execution_price = current_price + slippage
        else:
            execution_price = current_price - slippage
        
        # Calculate total cost
        total_cost = (position_size * execution_price) + commission
        
        # Check if enough cash
        if total_cost > self.portfolio['cash']:
            return False, "Insufficient cash", None
        
        # Create position
        position_id = self._generate_position_id(asset_info['symbol'])
        
        position = Position(
            id=position_id,
            asset=asset_info['symbol'],
            entry_time=datetime.now(),
            entry_price=execution_price,
            quantity=position_size,
            side=signal['type'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            current_price=execution_price,
            strategy=signal.get('strategy', 'Unknown'),
            risk_score=0.0
        )
        
        # Update portfolio
        self.portfolio['cash'] -= total_cost
        self.portfolio['positions'][position_id] = asdict(position)
        
        # Record order
        order = Order(
            id=f"ORD_{position_id}",
            timestamp=datetime.now(),
            asset=asset_info['symbol'],
            order_type=OrderType.MARKET,
            side=signal['type'],
            quantity=position_size,
            price=current_price,
            status=OrderStatus.FILLED,
            filled_quantity=position_size,
            filled_price=execution_price,
            commission=commission,
            slippage=slippage
        )
        
        self.portfolio['orders'].append(asdict(order))
        
        # Update stats
        st.session_state.performance_stats['total_commission'] += commission
        st.session_state.performance_stats['total_slippage'] += slippage
        
        logger.info(f"Position opened: {position_id} - {signal['type']} {position_size} @ {execution_price:.4f}")
        
        return True, position_id, position
    
    def update_positions(self, market_prices: Dict[str, float]):
        """Update all open positions"""
        
        positions_to_close = []
        
        for pos_id, pos_dict in list(self.portfolio['positions'].items()):
            symbol = pos_dict['asset']
            current_price = market_prices.get(symbol)
            
            if current_price is None:
                continue
            
            # Calculate P&L
            if pos_dict['side'] == 'BUY':
                unrealized_pnl = (current_price - pos_dict['entry_price']) * pos_dict['quantity']
            else:  # SELL
                unrealized_pnl = (pos_dict['entry_price'] - current_price) * pos_dict['quantity']
            
            pnl_percent = (unrealized_pnl / (pos_dict['entry_price'] * pos_dict['quantity'])) * 100
            
            # Update position
            pos_dict['current_price'] = current_price
            pos_dict['unrealized_pnl'] = unrealized_pnl
            pos_dict['max_pnl'] = max(pos_dict.get('max_pnl', 0), unrealized_pnl)
            pos_dict['min_pnl'] = min(pos_dict.get('min_pnl', 0), unrealized_pnl)
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            # Stop Loss
            if pos_dict['side'] == 'BUY' and current_price <= pos_dict['stop_loss']:
                should_close = True
                close_reason = "Stop Loss Hit"
            elif pos_dict['side'] == 'SELL' and current_price >= pos_dict['stop_loss']:
                should_close = True
                close_reason = "Stop Loss Hit"
            
            # Take Profit
            elif pos_dict['side'] == 'BUY' and current_price >= pos_dict['take_profit']:
                should_close = True
                close_reason = "Take Profit Hit"
            elif pos_dict['side'] == 'SELL' and current_price <= pos_dict['take_profit']:
                should_close = True
                close_reason = "Take Profit Hit"
            
            # Max holding period
            holding_time = (datetime.now() - pos_dict['entry_time']).seconds / 60
            if holding_time > MAX_HOLDING_PERIOD:
                should_close = True
                close_reason = "Max Holding Period"
            
            # Trailing stop (optional)
            if pos_dict.get('max_pnl', 0) > 0:
                drawdown_from_peak = (pos_dict['max_pnl'] - unrealized_pnl) / pos_dict['max_pnl']
                if drawdown_from_peak > 0.3:  # 30% retracement from peak
                    should_close = True
                    close_reason = "Trailing Stop"
            
            if should_close:
                positions_to_close.append((pos_id, current_price, close_reason))
        
        # Close positions
        for pos_id, close_price, reason in positions_to_close:
            self.close_position(pos_id, close_price, reason)
    
    def close_position(self, position_id: str, close_price: float, reason: str = "Manual"):
        """Close a position"""
        
        if position_id not in self.portfolio['positions']:
            return False
        
        pos_dict = self.portfolio['positions'][position_id]
        
        # Calculate final P&L
        if pos_dict['side'] == 'BUY':
            realized_pnl = (close_price - pos_dict['entry_price']) * pos_dict['quantity']
        else:
            realized_pnl = (pos_dict['entry_price'] - close_price) * pos_dict['quantity']
        
        # Calculate costs
        commission = pos_dict['quantity'] * close_price * COMMISSION_RATE
        realized_pnl -= commission
        
        # Update portfolio
        self.portfolio['cash'] += (pos_dict['quantity'] * close_price)
        self.portfolio['equity'] = self.portfolio['cash'] + sum(
            p.get('unrealized_pnl', 0) for p in self.portfolio['positions'].values()
        )
        
        # Record trade
        holding_time = (datetime.now() - pos_dict['entry_time']).total_seconds() / 60
        
        trade_record = {
            'close_time': datetime.now(),
            'holding_time_minutes': holding_time,
            'realized_pnl': realized_pnl,
            'pnl_percent': (realized_pnl / (pos_dict['entry_price'] * pos_dict['quantity'])) * 100,
            'close_price': close_price,
            'close_reason': reason,
            **pos_dict
        }
        
        self.portfolio['trade_history'].append(trade_record)
        
        # Update statistics
        stats = st.session_state.performance_stats
        stats['total_trades'] += 1
        
        if realized_pnl > 0:
            stats['winning_trades'] += 1
            stats['avg_win'] = ((stats['avg_win'] * (stats['winning_trades'] - 1)) + realized_pnl) / stats['winning_trades']
        else:
            stats['losing_trades'] += 1
            stats['avg_loss'] = ((stats['avg_loss'] * (stats['losing_trades'] - 1)) + abs(realized_pnl)) / stats['losing_trades']
        
        total_trades = stats['winning_trades'] + stats['losing_trades']
        stats['win_rate'] = (stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        if stats['avg_loss'] > 0:
            stats['profit_factor'] = stats['avg_win'] / stats['avg_loss']
        
        # Update average holding time
        stats['avg_holding_time'] = ((stats['avg_holding_time'] * (total_trades - 1)) + holding_time) / total_trades
        
        # Remove position
        del self.portfolio['positions'][position_id]
        
        # Update equity curve
        self.portfolio['equity_curve'].append(self.portfolio['equity'])
        self.portfolio['timestamps'].append(datetime.now())
        
        logger.info(f"Position closed: {position_id} - P&L: ${realized_pnl:.2f} - Reason: {reason}")
        
        return True
    
    def _calculate_slippage(self, price: float, quantity: float, asset_info: Dict) -> float:
        """Calculate realistic slippage"""
        
        if SLIPPAGE_MODEL == "realistic":
            # Slippage based on typical spread and quantity
            spread = asset_info.get('typical_spread', price * 0.001)
            volume_impact = quantity / asset_info.get('avg_daily_volume', 1000000)
            slippage = spread * (1 + volume_impact * 100)
            
        elif SLIPPAGE_MODEL == "conservative":
            slippage = price * 0.002  # 0.2%
            
        else:  # optimistic
            slippage = price * 0.0005  # 0.05%
        
        return slippage
    
    def _generate_position_id(self, symbol: str) -> str:
        """Generate unique position ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = random.randint(1000, 9999)
        return f"{symbol}_{timestamp}_{random_suffix}"

# ============================================================================
# USER INTERFACE
# ============================================================================

def render_header():
    """Render professional header"""
    
    system_state = st.session_state.system_state
    state_colors = {
        SystemState.ACTIVE: "status-active",
        SystemState.PAUSED: "status-paused",
        SystemState.ERROR: "status-error"
    }
    
    state_color = state_colors.get(system_state, "status-paused")
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="text-align: center; font-size: 2.5rem; font-weight: 800; 
                   background: linear-gradient(45deg, #60a5fa, #8b5cf6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏦 INSTITUTIONAL ALGORITHMIC TRADING SYSTEM
        </h1>
        <p style="text-align: center; color: #94a3b8; margin: 10px 0;">
            Professional-Grade Autonomous Trading Platform
        </p>
        <div class="system-status">
            <div class="status-indicator {state_color}"></div>
            <span style="color: #f8fafc; font-weight: 600;">{system_state.value.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_dashboard():
    """Render key metrics"""
    
    portfolio = st.session_state.portfolio
    stats = st.session_state.performance_stats
    risk = st.session_state.risk_metrics
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        equity = portfolio['equity']
        pnl = equity - 100000
        pnl_pct = (pnl / 100000) * 100
        color = "🟢" if pnl >= 0 else "🔴"
        st.metric("Portfolio Equity", f"${equity:,.2f}", f"{color} {pnl_pct:+.2f}%")
    
    with col2:
        cash = portfolio['cash']
        cash_pct = (cash / equity * 100) if equity > 0 else 0
        st.metric("Available Cash", f"${cash:,.2f}", f"{cash_pct:.1f}% liquid")
    
    with col3:
        win_rate = stats['win_rate']
        total = stats['winning_trades'] + stats['losing_trades']
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{stats['winning_trades']}W / {stats['losing_trades']}L")
    
    with col4:
        open_positions = len(portfolio['positions'])
        max_positions = 5
        st.metric("Open Positions", f"{open_positions}/{max_positions}", 
                 f"{(open_positions/max_positions*100):.0f}% capacity")
    
    with col5:
        sharpe = risk.sharpe_ratio
        risk_level = "Low" if sharpe > 2 else "Medium" if sharpe > 1 else "High"
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", risk_level)

def render_risk_dashboard():
    """Render risk metrics"""
    
    st.subheader("📊 Risk Management Dashboard")
    
    risk = st.session_state.risk_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_score = risk.overall_risk_score
        if risk_score < 0.3:
            risk_class = "risk-low"
            risk_label = "LOW"
        elif risk_score < 0.6:
            risk_class = "risk-medium"
            risk_label = "MEDIUM"
        elif risk_score < 0.8:
            risk_class = "risk-high"
            risk_label = "HIGH"
        else:
            risk_class = "risk-critical"
            risk_label = "CRITICAL"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">OVERALL RISK</div>
            <div class="risk-indicator {risk_class}">{risk_label}</div>
            <div style="margin-top: 8px; font-size: 1.2rem; font-weight: 600;">{risk_score:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        drawdown = risk.current_drawdown
        max_dd = risk.max_drawdown
        dd_pct = (drawdown / MAX_DRAWDOWN_THRESHOLD * 100) if MAX_DRAWDOWN_THRESHOLD > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">DRAWDOWN</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{drawdown:.2%}</div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #94a3b8;">
                Max: {max_dd:.2%} | Limit: {MAX_DRAWDOWN_THRESHOLD:.0%}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        var = abs(risk.portfolio_var)
        cvar = abs(risk.portfolio_cvar)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">VALUE AT RISK (95%)</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">{var:.2%}</div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #94a3b8;">
                CVaR: {cvar:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        correlation = risk.correlation_risk
        concentration = risk.concentration_risk
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">PORTFOLIO RISK</div>
            <div style="font-size: 1.2rem; font-weight: 600;">Correlation: {correlation:.2%}</div>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #94a3b8;">
                Concentration: {concentration:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_alerts():
    """Render system alerts"""
    
    alerts = st.session_state.alerts[-5:]  # Last 5 alerts
    
    if alerts:
        st.subheader("⚠️ System Alerts")
        
        for alert in reversed(alerts):
            severity = alert['severity'].lower()
            alert_class = f"alert-{severity}" if severity in ['info', 'success', 'warning'] else "alert-danger"
            
            timestamp = alert['timestamp'].strftime("%H:%M:%S")
            
            st.markdown(f"""
            <div class="alert-box {alert_class}">
                <strong>{alert['title']}</strong> - {timestamp}<br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)

def render_trading_interface(data_provider, strategy, execution_engine):
    """Render main trading interface"""
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Live Trading",
        "💼 Portfolio",
        "📊 Performance",
        "⚙️ System Settings"
    ])
    
    with tab1:
        render_live_trading(data_provider, strategy, execution_engine)
    
    with tab2:
        render_portfolio()
    
    with tab3:
        render_performance()
    
    with tab4:
        render_settings()

def render_live_trading(data_provider, strategy, execution_engine):
    """Render live trading interface"""
    
    st.subheader("📈 Live Market Analysis")
    
    # Asset selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category = st.selectbox("Asset Category", list(ASSET_CONFIG.keys()))
        asset_name = st.selectbox("Select Asset", list(ASSET_CONFIG[category].keys()))
        asset_info = ASSET_CONFIG[category][asset_name]
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=1)
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
    
    # Fetch and analyze data
    with st.spinner(f"Fetching {asset_name} data..."):
        df = data_provider.fetch_data(asset_info['symbol'], timeframe)
    
    if df is not None and not df.empty:
        # Display current price
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "🟢" if price_change >= 0 else "🔴"
            st.metric("Current Price", f"${current_price:.4f}", f"{color} {price_change:+.2f}%")
        
        with col2:
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
            st.metric("ATR", f"${atr:.4f}", f"{(atr/current_price*100):.2f}%")
        
        with col3:
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
            st.metric("RSI", f"{rsi:.1f}", rsi_color)
        
        with col4:
            volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
            st.metric("Volume", f"{volume:,.0f}")
        
        # Generate signals
        st.markdown("---")
        st.subheader("🎯 Trading Signals")
        
        signals = strategy.generate_signals(df, asset_info)
        
        if signals:
            st.success(f"✅ Found {len(signals)} high-confidence signals")
            
            for i, signal in enumerate(signals):
                with st.expander(f"{'🟢 BUY' if signal['type'] == 'BUY' else '🔴 SELL'} Signal #{i+1} - Confidence: {signal['confidence']*100:.0f}%"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Strategy:** {signal['strategy']}")
                        st.write(f"**Entry Price:** ${signal['entry']:.4f}")
                        st.write(f"**Stop Loss:** ${signal['stop_loss']:.4f}")
                        st.write(f"**Take Profit:** ${signal['take_profit']:.4f}")
                    
                    with col2:
                        risk = abs(signal['entry'] - signal['stop_loss'])
                        reward = abs(signal['take_profit'] - signal['entry'])
                        rr_ratio = reward / risk if risk > 0 else 0
                        
                        st.write(f"**Risk:** ${risk:.4f}")
                        st.write(f"**Reward:** ${reward:.4f}")
                        st.write(f"**R:R Ratio:** {rr_ratio:.2f}:1")
                        st.write(f"**Reason:** {signal['reason']}")
                    
                    if st.button(f"🚀 Execute {signal['type']}", key=f"exec_{i}"):
                        success, msg, position = execution_engine.execute_order(signal, asset_info)
                        
                        if success:
                            st.success(f"✅ Order executed successfully! Position ID: {msg}")
                            tm.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Order rejected: {msg}")
        else:
            st.info("ℹ️ No high-confidence signals at the moment")
        
        # Price chart
        st.markdown("---")
        st.subheader("📊 Price Chart")
        
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price', 'RSI', 'Volume'),
            vertical_spacing=0.05
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        if 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                          line=dict(color='rgba(59, 130, 246, 0.5)', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                          line=dict(color='rgba(59, 130, 246, 0.5)', dash='dash'),
                          fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)'),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                          line=dict(color='#8b5cf6')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume
        colors = ['red' if row['close'] < row['open'] else 'green' for idx, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume',
                  marker_color=colors),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ Failed to fetch market data")

def render_portfolio():
    """Render portfolio view"""
    
    st.subheader("💼 Active Positions")
    
    portfolio = st.session_state.portfolio
    positions = portfolio['positions']
    
    if positions:
        for pos_id, pos in positions.items():
            pnl = pos.get('unrealized_pnl', 0)
            pnl_color = "🟢" if pnl >= 0 else "🔴"
            
            with st.expander(f"{pnl_color} {pos['asset']} - {pos['side']} - P&L: ${pnl:+,.2f}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Entry:** ${pos['entry_price']:.4f}")
                    st.write(f"**Current:** ${pos.get('current_price', pos['entry_price']):.4f}")
                    st.write(f"**Quantity:** {pos['quantity']:.4f}")
                
                with col2:
                    st.write(f"**Stop Loss:** ${pos['stop_loss']:.4f}")
                    st.write(f"**Take Profit:** ${pos['take_profit']:.4f}")
                    st.write(f"**Strategy:** {pos.get('strategy', 'Unknown')}")
                
                with col3:
                    entry_time = pos['entry_time']
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    holding_time = (datetime.now() - entry_time).seconds / 60
                    
                    st.write(f"**Unrealized P&L:** ${pnl:+,.2f}")
                    st.write(f"**Max P&L:** ${pos.get('max_pnl', 0):+,.2f}")
                    st.write(f"**Holding Time:** {holding_time:.1f} min")
                
                if st.button(f"❌ Close Position", key=f"close_{pos_id}"):
                    engine = ExecutionEngine()
                    current_price = pos.get('current_price', pos['entry_price'])
                    engine.close_position(pos_id, current_price, "Manual Close")
                    st.success(f"Position {pos_id} closed")
                    tm.sleep(1)
                    st.rerun()
    else:
        st.info("No open positions")
    
    # Trade History
    st.markdown("---")
    st.subheader("📜 Trade History")
    
    trade_history = portfolio['trade_history']
    
    if trade_history:
        df_history = pd.DataFrame(trade_history[-20:])  # Last 20 trades
        
        # Format for display
        display_cols = ['close_time', 'asset', 'side', 'entry_price', 'close_price', 
                       'quantity', 'realized_pnl', 'pnl_percent', 'close_reason']
        
        if all(col in df_history.columns for col in display_cols):
            df_display = df_history[display_cols].copy()
            df_display['close_time'] = pd.to_datetime(df_display['close_time']).dt.strftime('%Y-%m-%d %H:%M')
            df_display = df_display.sort_values('close_time', ascending=False)
            
            st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No trade history yet")

def render_performance():
    """Render performance analytics"""
    
    st.subheader("📊 Performance Analytics")
    
    portfolio = st.session_state.portfolio
    stats = st.session_state.performance_stats
    
    # Equity curve
    if len(portfolio['equity_curve']) > 1:
        df_equity = pd.DataFrame({
            'timestamp': portfolio['timestamps'],
            'equity': portfolio['equity_curve']
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_equity['timestamp'],
            y=df_equity['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#10b981', width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        fig.add_hline(y=100000, line_dash="dash", line_color="gray",
                     annotation_text="Initial Capital")
        
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", stats['total_trades'])
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    
    with col2:
        st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        st.metric("Avg Win", f"${stats['avg_win']:.2f}")
    
    with col3:
        st.metric("Avg Loss", f"${stats['avg_loss']:.2f}")
        st.metric("Avg Hold Time", f"{stats['avg_holding_time']:.1f} min")
    
    with col4:
        total_costs = stats['total_commission'] + stats['total_slippage']
        st.metric("Total Commission", f"${stats['total_commission']:.2f}")
        st.metric("Total Slippage", f"${stats['total_slippage']:.2f}")

def render_settings():
    """Render system settings"""
    
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎛️ System Controls")
        
        current_state = st.session_state.system_state
        
        if st.button("▶️ Activate System", disabled=(current_state == SystemState.ACTIVE)):
            st.session_state.system_state = SystemState.ACTIVE
            st.session_state.circuit_breaker_triggered = False
            st.session_state.daily_loss_limit_hit = False
            st.success("System activated")
            st.rerun()
        
        if st.button("⏸️ Pause System", disabled=(current_state == SystemState.PAUSED)):
            st.session_state.system_state = SystemState.PAUSED
            st.warning("System paused")
            st.rerun()
        
        if st.button("🔄 Reset Circuit Breaker", 
                    disabled=not st.session_state.circuit_breaker_triggered):
            st.session_state.circuit_breaker_triggered = False
            st.session_state.system_state = SystemState.ACTIVE
            st.success("Circuit breaker reset")
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Risk Parameters")
        st.info(f"""
        - **Max Portfolio Risk:** {MAX_PORTFOLIO_RISK:.1%}
        - **Max Drawdown:** {MAX_DRAWDOWN_THRESHOLD:.1%}
        - **Max Daily Loss:** {MAX_DAILY_LOSS:.1%}
        - **Commission Rate:** {COMMISSION_RATE:.2%}
        - **Min Profit Target:** {MIN_PROFIT_TARGET:.2%}
        """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize components
    data_provider = DataProvider()
    strategy = InstitutionalStrategy()
    execution_engine = ExecutionEngine()
    risk_manager = RiskManager()
    
    # Render UI
    render_header()
    
    # Check circuit breakers
    risk_manager.check_circuit_breaker()
    
    # Display alerts
    render_alerts()
    
    # Main dashboard
    render_metrics_dashboard()
    
    st.markdown("---")
    
    # Risk dashboard
    render_risk_dashboard()
    
    st.markdown("---")
    
    # Calculate and update risk metrics
    st.session_state.risk_metrics = risk_manager.calculate_portfolio_metrics()
    
    # Main trading interface
    render_trading_interface(data_provider, strategy, execution_engine)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><strong>⚠️ RISK DISCLAIMER:</strong> This is a sophisticated trading system for educational and research purposes.</p>
        <p>Algorithmic trading involves substantial risk of loss. Past performance does not guarantee future results.</p>
        <p style='margin-top: 10px; font-size: 0.9rem;'>
            Institutional Algorithmic Trading System v1.0 • Professional Grade • © 2024
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
