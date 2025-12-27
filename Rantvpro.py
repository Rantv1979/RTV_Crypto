# app.py - RTV SMC Intraday Algorithmic Trading Terminal Pro - ENHANCED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time as tm
import random
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings('ignore')

# Try to import optional dependencies with fallbacks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    st.error("⚠️ yfinance not installed. Please run: pip install yfinance")
    YFINANCE_AVAILABLE = False
    # Create mock yfinance for development
    class MockYFinance:
        class Ticker:
            def history(self, **kwargs):
                return pd.DataFrame()
    yf = MockYFinance()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("Plotly not available. Charts will be limited.")
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    st.warning("SciPy not available. Some statistics features will be limited.")
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    st.warning("statsmodels not available. ADF test will use fallback.")
    STATSMODELS_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="RTV SMC Pro Trading Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Trading Terminal CSS
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
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        margin-bottom: 25px;
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
    
    /* Advanced Cards */
    .advanced-metric-card {
        background: linear-gradient(145deg, var(--secondary-bg), #1a2238);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
    }
    
    .advanced-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    }
    
    /* Regime indicators */
    .regime-trending {
        background: rgba(16, 185, 129, 0.2);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .regime-ranging {
        background: rgba(245, 158, 11, 0.2);
        color: var(--accent-yellow);
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .regime-volatile {
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
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
    st.session_state.refresh_interval = 30

if 'enhanced_metrics' not in st.session_state:
    st.session_state.enhanced_metrics = {
        'regime_history': {},
        'correlation_matrix': None,
        'kelly_metrics': {},
        'backtest_results': {}
    }

# ============================================================================
# ASSET CONFIGURATION
# ============================================================================
ASSET_CONFIG = {
    "Cryptocurrencies": {
        "BTC/USD": {"symbol": "BTC-USD", "pip_size": 0.01, "lot_size": 0.001, 
                   "sector": "Crypto", "volatility": "High",
                   "spread": 0.0002, "commission": 0.001, "slippage": 0.0005},
        "ETH/USD": {"symbol": "ETH-USD", "pip_size": 0.01, "lot_size": 0.01, 
                   "sector": "Crypto", "volatility": "High",
                   "spread": 0.0003, "commission": 0.001, "slippage": 0.0005},
        "SOL/USD": {"symbol": "SOL-USD", "pip_size": 0.001, "lot_size": 0.1, 
                   "sector": "Crypto", "volatility": "Very High",
                   "spread": 0.0005, "commission": 0.0015, "slippage": 0.001},
    },
    "Commodities": {
        "Gold": {"symbol": "GC=F", "pip_size": 0.10, "lot_size": 1, 
                "sector": "Commodities", "volatility": "Medium",
                "spread": 0.0002, "commission": 0.0005, "slippage": 0.0001},
        "Silver": {"symbol": "SI=F", "pip_size": 0.01, "lot_size": 10, 
                  "sector": "Commodities", "volatility": "High",
                  "spread": 0.0003, "commission": 0.0005, "slippage": 0.0002},
        "Crude Oil": {"symbol": "CL=F", "pip_size": 0.01, "lot_size": 10, 
                     "sector": "Commodities", "volatility": "High",
                     "spread": 0.0003, "commission": 0.0005, "slippage": 0.0002},
    },
    "Forex Pairs": {
        "EUR/USD": {"symbol": "EURUSD=X", "pip_size": 0.0001, "lot_size": 10000, 
                   "sector": "Forex", "volatility": "Low",
                   "spread": 0.00001, "commission": 0.0001, "slippage": 0.00005},
        "GBP/USD": {"symbol": "GBPUSD=X", "pip_size": 0.0001, "lot_size": 10000, 
                   "sector": "Forex", "volatility": "Medium",
                   "spread": 0.00002, "commission": 0.0001, "slippage": 0.00005},
        "USD/JPY": {"symbol": "JPY=X", "pip_size": 0.01, "lot_size": 10000, 
                   "sector": "Forex", "volatility": "Medium",
                   "spread": 0.0003, "commission": 0.0001, "slippage": 0.0001},
    }
}

# ============================================================================
# ENHANCED TRADING STRATEGIES
# ============================================================================
TRADING_STRATEGIES = {
    "SMC Pro": {
        "description": "Smart Money Concepts with multi-timeframe confirmation",
        "indicators": ["FVG", "Order Blocks", "Market Structure", "Liquidity"],
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
    }
}

# ============================================================================
# CORE COMPONENTS
# ============================================================================

class RealisticCostCalculator:
    """Calculate realistic trading costs"""
    
    @staticmethod
    def calculate_entry_cost(price: float, size: float, asset_config: dict) -> float:
        """Calculate total entry cost"""
        spread_cost = price * asset_config['spread'] * size
        commission_cost = price * size * asset_config['commission']
        slippage_cost = price * asset_config['slippage'] * size
        return spread_cost + commission_cost + slippage_cost
    
    @staticmethod
    def get_effective_entry_price(price: float, asset_config: dict, is_buy: bool) -> float:
        """Get effective entry price after costs"""
        if is_buy:
            return price * (1 + asset_config['spread'] + asset_config['slippage'])
        else:
            return price * (1 - asset_config['spread'] - asset_config['slippage'])

class MarketRegimeDetector:
    """Detect market regime with fallback for missing dependencies"""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback = lookback_periods
    
    def detect_regime(self, prices: pd.Series) -> Dict[str, Any]:
        """Detect current market regime"""
        if len(prices) < self.lookback:
            return {"regime": "Unknown", "confidence": 0.0, "metrics": {}}
        
        recent_prices = prices.iloc[-self.lookback:]
        returns = recent_prices.pct_change().dropna()
        
        # Basic metrics that don't require special libraries
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        # Simple trend detection
        trend_strength = self._simple_trend_strength(recent_prices)
        
        # Regime classification
        regime = "Ranging"
        confidence = 0.5
        
        if trend_strength > 0.6:
            regime = "Trending"
            confidence = min(0.9, trend_strength)
        elif volatility > 0.3:
            regime = "Volatile"
            confidence = min(0.8, volatility)
        
        return {
            "regime": regime,
            "confidence": confidence,
            "metrics": {
                "volatility": volatility,
                "trend_strength": trend_strength,
            }
        }
    
    def _simple_trend_strength(self, prices: pd.Series) -> float:
        """Simple trend strength calculation without scipy"""
        if len(prices) < 2:
            return 0.0
        
        # Calculate linear trend using basic statistics
        x = np.arange(len(prices))
        y = prices.values
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize slope to 0-1 range
        price_range = prices.max() - prices.min()
        if price_range > 0:
            normalized_slope = abs(slope * n / price_range)
            return min(1.0, normalized_slope)
        
        return 0.0

class CorrelationFilter:
    """Filter positions based on correlation"""
    
    def __init__(self, max_correlation: float = 0.7):
        self.max_correlation = max_correlation
        self.correlation_matrix = None
    
    def update_correlations(self, price_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix"""
        close_prices = {}
        
        for asset, df in price_data.items():
            if not df.empty and 'close' in df.columns:
                close_prices[asset] = df['close']
        
        if len(close_prices) >= 2:
            try:
                combined_df = pd.DataFrame(close_prices)
                self.correlation_matrix = combined_df.corr()
            except:
                self.correlation_matrix = None
    
    def can_add_position(self, new_asset: str, current_positions: List[str]) -> bool:
        """Check correlation constraints"""
        if self.correlation_matrix is None or new_asset not in self.correlation_matrix.columns:
            return True
        
        for position in current_positions:
            if position in self.correlation_matrix.columns:
                try:
                    corr = abs(self.correlation_matrix.loc[new_asset, position])
                    if corr > self.max_correlation:
                        return False
                except:
                    continue
        
        return True

class KellyCriterionCalculator:
    """Dynamic position sizing"""
    
    @staticmethod
    def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction"""
        if avg_loss == 0:
            return 0.1
        
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        
        kelly_f = (p * b - q) / b if b > 0 else 0
        return max(0.01, min(0.25, kelly_f * 0.5))

class EnhancedTradingEngine:
    """Main trading engine with enhanced features"""
    
    def __init__(self):
        self.portfolio = st.session_state.paper_portfolio
        self.cost_calculator = RealisticCostCalculator()
        self.regime_detector = MarketRegimeDetector()
        self.correlation_filter = CorrelationFilter()
        self.kelly_calculator = KellyCriterionCalculator()
    
    def calculate_position_size(self, asset: str, signal: Dict, current_price: float) -> float:
        """Calculate dynamic position size"""
        # Get historical performance
        strategy_key = f"{asset}_{signal.get('strategy', 'default')}"
        win_rate = st.session_state.enhanced_metrics['kelly_metrics'].get(
            f"{strategy_key}_win_rate", 0.5
        )
        avg_win = st.session_state.enhanced_metrics['kelly_metrics'].get(
            f"{strategy_key}_avg_win", 0.02
        )
        avg_loss = st.session_state.enhanced_metrics['kelly_metrics'].get(
            f"{strategy_key}_avg_loss", 0.01
        )
        
        # Kelly fraction
        kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
            win_rate, avg_win, avg_loss
        )
        
        # Stop distance
        stop_distance = abs(signal['entry'] - signal['stop_loss'])
        if stop_distance == 0:
            stop_distance = current_price * 0.01
        
        # Risk amount
        risk_amount = self.portfolio['balance'] * risk_per_trade / 100
        position_size = risk_amount / stop_distance
        
        # Adjust with Kelly
        kelly_size = self.portfolio['balance'] * kelly_fraction / current_price
        position_size = min(position_size, kelly_size)
        
        # Asset constraints
        asset_config = self._get_asset_config(asset)
        if asset_config:
            max_size = asset_config['lot_size'] * 5
            position_size = min(position_size, max_size)
        
        return position_size
    
    def execute_trade(self, signal: Dict, current_price: float) -> Tuple[bool, str]:
        """Execute trade with enhanced features"""
        asset_config = self._get_asset_config(signal['asset'])
        if not asset_config:
            return False, "Asset config not found"
        
        # Check correlation
        current_positions = [
            pos['asset'] for pos in self.portfolio['positions'].values() 
            if pos['status'] == 'OPEN'
        ]
        
        if not self.correlation_filter.can_add_position(signal['asset'], current_positions):
            return False, "Correlation constraint"
        
        # Calculate position size
        position_size = self.calculate_position_size(signal['asset'], signal, current_price)
        
        # Calculate costs
        is_buy = signal['type'] == 'BUY'
        effective_price = self.cost_calculator.get_effective_entry_price(
            signal['entry'], asset_config, is_buy
        )
        
        entry_cost = self.cost_calculator.calculate_entry_cost(
            effective_price, position_size, asset_config
        )
        
        # Create trade
        trade_id = f"{signal['asset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        trade = {
            'id': trade_id,
            'timestamp': datetime.now(),
            'asset': signal['asset'],
            'asset_name': signal['asset_name'],
            'type': signal['type'],
            'entry_price': effective_price,
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'current_price': current_price,
            'status': 'OPEN',
            'pnl': -entry_cost,
            'strategy': signal.get('strategy', 'Unknown'),
            'confidence': signal.get('confidence', 0.5),
            'costs': {'entry_cost': entry_cost}
        }
        
        # Update portfolio
        self.portfolio['positions'][trade_id] = trade
        self.portfolio['balance'] -= entry_cost
        
        # Log trade
        log_entry = {
            'timestamp': datetime.now(),
            'action': 'OPEN',
            'trade_id': trade_id,
            'asset': signal['asset_name'],
            'symbol': signal['asset'],
            'type': signal['type'],
            'entry_price': effective_price,
            'size': position_size,
            'costs': entry_cost
        }
        
        st.session_state.trade_log.append(log_entry)
        return True, trade_id
    
    def _get_asset_config(self, symbol: str) -> Optional[Dict]:
        """Get asset configuration"""
        for category in ASSET_CONFIG.values():
            for name, config in category.items():
                if config['symbol'] == symbol:
                    return config
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_regime_analysis(regime_info: Dict):
    """Display market regime analysis"""
    regime = regime_info.get('regime', 'Unknown')
    confidence = regime_info.get('confidence', 0.0)
    
    regime_colors = {
        'Trending': '#10b981',
        'Ranging': '#f59e0b',
        'Volatile': '#ef4444'
    }
    
    color = regime_colors.get(regime, '#94a3b8')
    
    st.markdown(f"""
    <div class="advanced-metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: {color};">
                    📊 Market Regime: <strong>{regime}</strong>
                </h4>
                <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                    Confidence: {confidence*100:.1f}%
                </p>
            </div>
            <div style="text-align: right;">
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    Volatility: {regime_info.get('metrics', {}).get('volatility', 0)*100:.1f}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_correlation_analysis():
    """Display correlation analysis"""
    current_positions = [
        pos['asset'] for pos in st.session_state.paper_portfolio['positions'].values() 
        if pos['status'] == 'OPEN'
    ]
    
    if not current_positions:
        return
    
    st.markdown(f"""
    <div class="advanced-metric-card">
        <h4 style="margin: 0 0 10px 0; color: #94a3b8;">
            🔗 Portfolio Correlation
        </h4>
        <p style="margin: 0; color: #d1d5db;">
            Active Positions: {len(current_positions)}<br>
            Max Correlation: 70%
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 RTV SMC ALGORITHMIC TRADING TERMINAL PRO</h1>
        <p style="text-align: center; color: #94a3b8; margin: 10px 0;">
            Professional Intraday Trading System with Enhanced Features
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for yfinance
    if not YFINANCE_AVAILABLE:
        st.error("""
        ## ⚠️ Required Library Missing
        
        Please install yfinance to use this application:
        ```
        pip install yfinance
        ```
        
        Or install all requirements:
        ```
        pip install streamlit pandas numpy yfinance plotly scipy statsmodels
        ```
        """)
        return
    
    # Sidebar
    st.sidebar.markdown("### ⚙️ TRADING CONFIGURATION")
    
    # Global variables (would be from sidebar in full implementation)
    global risk_per_trade
    risk_per_trade = 1.0
    
    # Operating Mode
    mode = st.sidebar.selectbox(
        "Operating Mode",
        ["📡 Live Trading", "🔍 Signal Scanner", "📊 Portfolio"],
        index=0
    )
    
    # Asset Selection
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
    
    # Strategy Selection
    selected_strategy = st.sidebar.selectbox(
        "Strategy",
        list(TRADING_STRATEGIES.keys()),
        index=0
    )
    
    # Risk Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ RISK MANAGEMENT")
    
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)
    
    # Enhanced Features
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 ENHANCED FEATURES")
    
    use_kelly = st.sidebar.checkbox("Kelly Criterion Sizing", value=True)
    use_regime = st.sidebar.checkbox("Market Regime Detection", value=True)
    use_correlation = st.sidebar.checkbox("Correlation Filter", value=True)
    
    # Main Dashboard
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        balance = st.session_state.paper_portfolio['balance']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">BALANCE</div>
            <div class="metric-value">${balance:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        daily_pnl = st.session_state.paper_portfolio['daily_pnl']
        color = "change-positive" if daily_pnl >= 0 else "change-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DAILY P&L</div>
            <div class="metric-value">${daily_pnl:+,.2f}</div>
            <div class="metric-change {color}">{daily_pnl/balance*100:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        open_positions = len(st.session_state.paper_portfolio['positions'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">OPEN POSITIONS</div>
            <div class="metric-value">{open_positions}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        win_rate = 0
        if st.session_state.paper_portfolio['winning_trades'] + st.session_state.paper_portfolio['losing_trades'] > 0:
            win_rate = (st.session_state.paper_portfolio['winning_trades'] / 
                       (st.session_state.paper_portfolio['winning_trades'] + 
                        st.session_state.paper_portfolio['losing_trades']) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">WIN RATE</div>
            <div class="metric-value">{win_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        max_dd = st.session_state.paper_portfolio['max_drawdown']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAX DRAWDOWN</div>
            <div class="metric-value">{max_dd:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Analytics
    st.markdown("### 📊 Enhanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if use_regime and asset_info:
            # Simulate regime detection
            regime_info = {
                'regime': 'Trending',
                'confidence': 0.75,
                'metrics': {'volatility': 0.25}
            }
            display_regime_analysis(regime_info)
    
    with col2:
        if use_correlation:
            display_correlation_analysis()
    
    # Trading Interface
    st.markdown("### 🎯 Trading Signals")
    
    # Initialize trading engine
    trading_engine = EnhancedTradingEngine()
    
    # Simulate signal generation
    with st.expander("Generate Trading Signal", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            signal_type = st.selectbox("Signal Type", ["BUY", "SELL"])
            entry_price = st.number_input("Entry Price", value=100.0)
            stop_loss = st.number_input("Stop Loss", value=98.0)
            take_profit = st.number_input("Take Profit", value=105.0)
        
        with col2:
            confidence = st.slider("Confidence", 0.0, 1.0, 0.75, 0.05)
            strategy = st.selectbox("Strategy", list(TRADING_STRATEGIES.keys()))
        
        if st.button("🚀 Generate & Execute Signal", type="primary"):
            # Create signal
            signal = {
                'asset': asset_info['symbol'],
                'asset_name': selected_asset,
                'type': signal_type,
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': strategy,
                'confidence': confidence
            }
            
            # Execute trade
            success, trade_id = trading_engine.execute_trade(signal, entry_price)
            
            if success:
                st.success(f"✅ Trade executed! ID: {trade_id}")
                st.rerun()
            else:
                st.error("❌ Trade execution failed")
    
    # Portfolio Overview
    st.markdown("### 💼 Portfolio Overview")
    
    positions = st.session_state.paper_portfolio['positions']
    
    if positions:
        for trade_id, position in positions.items():
            pnl_color = "color: #10b981;" if position['pnl'] >= 0 else "color: #ef4444;"
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin: 5px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>{position['asset_name']} - {position['type']}</strong>
                        <div style="color: #94a3b8; font-size: 0.9rem;">
                            Entry: ${position['entry_price']:.4f} | 
                            Size: {position['size']:.4f}
                        </div>
                    </div>
                    <div style="{pnl_color} font-weight: bold;">
                        ${position['pnl']:+,.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No open positions")
    
    # Trade History
    st.markdown("### 📋 Trade History")
    
    if st.session_state.trade_log:
        # Show last 10 trades
        for log in reversed(st.session_state.trade_log[-10:]):
            timestamp = log['timestamp'].strftime("%H:%M:%S")
            action = log['action']
            asset = log.get('asset', 'Unknown')
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); padding: 10px; border-radius: 6px; margin: 5px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                    <span>{timestamp} - {action} {asset}</span>
                    <span>${log.get('entry_price', 0):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trades yet")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><strong>⚠️ RISK DISCLAIMER:</strong> This is a paper trading simulation for educational purposes only.</p>
        <p style='margin-top: 10px; font-size: 0.9rem;'>RTV SMC Algorithmic Trading Terminal Pro v2.0 • Enhanced Edition</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
