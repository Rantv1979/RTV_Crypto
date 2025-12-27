# app.py - RTV SMC Intraday Algorithmic Trading Terminal Pro - ENHANCED VERSION
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
from typing import Dict, List, Optional, Tuple, Set, Any
import random
import traceback
from collections import defaultdict
import pytz
import requests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ENHANCED ASSET CONFIGURATION WITH REALISTIC COSTS
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
        "XRP/USD": {"symbol": "XRP-USD", "pip_size": 0.0001, "lot_size": 1000, 
                   "sector": "Crypto", "volatility": "High",
                   "spread": 0.0001, "commission": 0.001, "slippage": 0.0005},
        "ADA/USD": {"symbol": "ADA-USD", "pip_size": 0.0001, "lot_size": 1000, 
                   "sector": "Crypto", "volatility": "High",
                   "spread": 0.0001, "commission": 0.001, "slippage": 0.0005},
        "DOGE/USD": {"symbol": "DOGE-USD", "pip_size": 0.00001, "lot_size": 10000, 
                    "sector": "Crypto", "volatility": "Very High",
                    "spread": 0.0002, "commission": 0.002, "slippage": 0.001},
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
        "Natural Gas": {"symbol": "NG=F", "pip_size": 0.001, "lot_size": 100, 
                       "sector": "Commodities", "volatility": "Very High",
                       "spread": 0.0005, "commission": 0.001, "slippage": 0.0003},
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
        "AUD/USD": {"symbol": "AUDUSD=X", "pip_size": 0.0001, "lot_size": 10000, 
                   "sector": "Forex", "volatility": "Medium",
                   "spread": 0.00003, "commission": 0.0001, "slippage": 0.00005},
        "NZD/USD": {"symbol": "NZDUSD=X", "pip_size": 0.0001, "lot_size": 10000, 
                   "sector": "Forex", "volatility": "Medium",
                   "spread": 0.00004, "commission": 0.0001, "slippage": 0.00005},
        "USD/CHF": {"symbol": "CHF=X", "pip_size": 0.0001, "lot_size": 10000, 
                   "sector": "Forex", "volatility": "Low",
                   "spread": 0.00002, "commission": 0.0001, "slippage": 0.00005},
    },
    "Indices": {
        "S&P 500": {"symbol": "^GSPC", "pip_size": 0.25, "lot_size": 1, 
                   "sector": "Indices", "volatility": "Medium",
                   "spread": 0.0001, "commission": 0.0005, "slippage": 0.0002},
        "NASDAQ": {"symbol": "^IXIC", "pip_size": 0.25, "lot_size": 1, 
                  "sector": "Indices", "volatility": "High",
                  "spread": 0.0002, "commission": 0.0005, "slippage": 0.0003},
        "Dow Jones": {"symbol": "^DJI", "pip_size": 1.0, "lot_size": 1, 
                     "sector": "Indices", "volatility": "Medium",
                     "spread": 0.0001, "commission": 0.0005, "slippage": 0.0002},
        "Russell 2000": {"symbol": "^RUT", "pip_size": 0.10, "lot_size": 1, 
                        "sector": "Indices", "volatility": "High",
                        "spread": 0.0003, "commission": 0.0005, "slippage": 0.0004},
    },
    "Tech Stocks": {
        "Apple": {"symbol": "AAPL", "pip_size": 0.01, "lot_size": 10, 
                 "sector": "Tech", "volatility": "Medium",
                 "spread": 0.0001, "commission": 0.0005, "slippage": 0.0002},
        "Microsoft": {"symbol": "MSFT", "pip_size": 0.01, "lot_size": 10, 
                     "sector": "Tech", "volatility": "Medium",
                     "spread": 0.0001, "commission": 0.0005, "slippage": 0.0002},
        "Tesla": {"symbol": "TSLA", "pip_size": 0.01, "lot_size": 10, 
                 "sector": "Tech", "volatility": "Very High",
                 "spread": 0.0003, "commission": 0.001, "slippage": 0.0005},
        "NVIDIA": {"symbol": "NVDA", "pip_size": 0.01, "lot_size": 10, 
                  "sector": "Tech", "volatility": "Very High",
                  "spread": 0.0002, "commission": 0.001, "slippage": 0.0004},
        "Amazon": {"symbol": "AMZN", "pip_size": 0.01, "lot_size": 10, 
                  "sector": "Tech", "volatility": "Medium",
                  "spread": 0.0001, "commission": 0.0005, "slippage": 0.0002},
    }
}

# ============================================================================
# 2. MODULAR ARCHITECTURE - PROFESSIONAL COMPONENTS
# ============================================================================

class RealisticCostCalculator:
    """Calculate realistic trading costs including commission, slippage, and spread"""
    
    @staticmethod
    def calculate_entry_cost(price: float, size: float, asset_config: dict) -> float:
        """Calculate total entry cost"""
        spread_cost = price * asset_config['spread'] * size
        commission_cost = price * size * asset_config['commission']
        slippage_cost = price * asset_config['slippage'] * size
        return spread_cost + commission_cost + slippage_cost
    
    @staticmethod
    def calculate_exit_cost(price: float, size: float, asset_config: dict) -> float:
        """Calculate total exit cost"""
        spread_cost = price * asset_config['spread'] * size
        commission_cost = price * size * asset_config['commission']
        return spread_cost + commission_cost
    
    @staticmethod
    def get_effective_entry_price(price: float, asset_config: dict, is_buy: bool) -> float:
        """Get effective entry price after costs"""
        if is_buy:
            return price * (1 + asset_config['spread'] + asset_config['slippage'])
        else:
            return price * (1 - asset_config['spread'] - asset_config['slippage'])

class MarketRegimeDetector:
    """Detect market regime (Trending, Ranging, Volatile)"""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback = lookback_periods
    
    def detect_regime(self, prices: pd.Series) -> Dict[str, Any]:
        """Detect current market regime"""
        if len(prices) < self.lookback:
            return {"regime": "Unknown", "confidence": 0.0, "metrics": {}}
        
        recent_prices = prices.iloc[-self.lookback:]
        returns = recent_prices.pct_change().dropna()
        
        # Calculate metrics
        volatility = returns.std() * np.sqrt(252)
        trend_strength = self._calculate_trend_strength(recent_prices)
        adf_stat, adf_pvalue = self._calculate_adf(recent_prices)
        hurst_exponent = self._calculate_hurst_exponent(recent_prices)
        
        # Regime classification
        regime = "Ranging"
        confidence = 0.5
        
        if trend_strength > 0.6 and adf_pvalue > 0.05:
            regime = "Trending"
            confidence = min(0.9, trend_strength)
        elif volatility > 0.3:  # 30% annualized volatility threshold
            regime = "Volatile"
            confidence = min(0.8, volatility)
        elif hurst_exponent > 0.65:
            regime = "Trending"
            confidence = hurst_exponent
        elif hurst_exponent < 0.35:
            regime = "Mean Reverting"
            confidence = 1 - hurst_exponent
        
        return {
            "regime": regime,
            "confidence": confidence,
            "metrics": {
                "volatility": volatility,
                "trend_strength": trend_strength,
                "hurst_exponent": hurst_exponent,
                "adf_pvalue": adf_pvalue
            }
        }
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate R-squared of linear trend"""
        x = np.arange(len(prices))
        y = prices.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return abs(r_value)
    
    def _calculate_adf(self, prices: pd.Series):
        """Calculate Augmented Dickey-Fuller test for stationarity"""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(prices, maxlag=1)
        return result[0], result[1]
    
    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent for trend detection"""
        lags = range(2, min(20, len(prices)//2))
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

class CorrelationFilter:
    """Filter positions based on correlation thresholds"""
    
    def __init__(self, max_correlation: float = 0.7, lookback_days: int = 90):
        self.max_correlation = max_correlation
        self.lookback = lookback_days
        self.correlation_matrix = None
    
    def update_correlations(self, price_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix for all assets"""
        close_prices = {}
        
        for asset, df in price_data.items():
            if not df.empty and 'close' in df.columns:
                close_prices[asset] = df['close']
        
        if len(close_prices) >= 2:
            combined_df = pd.DataFrame(close_prices)
            self.correlation_matrix = combined_df.corr()
        else:
            self.correlation_matrix = None
    
    def can_add_position(self, new_asset: str, current_positions: List[str]) -> bool:
        """Check if new position can be added based on correlation"""
        if self.correlation_matrix is None or new_asset not in self.correlation_matrix.columns:
            return True
        
        for position in current_positions:
            if position in self.correlation_matrix.columns:
                corr = abs(self.correlation_matrix.loc[new_asset, position])
                if corr > self.max_correlation:
                    return False
        
        return True
    
    def get_portfolio_correlation(self, positions: List[str]) -> float:
        """Calculate average correlation of portfolio"""
        if self.correlation_matrix is None or len(positions) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if positions[i] in self.correlation_matrix.columns and positions[j] in self.correlation_matrix.columns:
                    corr = abs(self.correlation_matrix.loc[positions[i], positions[j]])
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0

class KellyCriterionCalculator:
    """Dynamic position sizing using Kelly Criterion"""
    
    @staticmethod
    def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        if avg_loss == 0:
            return 0.1  # Conservative default
        
        b = avg_win / abs(avg_loss)  # win/loss ratio
        p = win_rate
        q = 1 - p
        
        kelly_f = (p * b - q) / b if b > 0 else 0
        
        # Apply conservative fraction (half-Kelly is common)
        return max(0.01, min(0.25, kelly_f * 0.5))
    
    @staticmethod
    def calculate_position_size(balance: float, kelly_fraction: float, 
                              risk_per_trade: float, stop_distance: float, 
                              price: float) -> float:
        """Calculate position size using Kelly Criterion"""
        # Base position size from risk per trade
        risk_amount = balance * risk_per_trade / 100
        base_size = risk_amount / stop_distance
        
        # Adjust with Kelly fraction
        kelly_size = balance * kelly_fraction / price
        
        # Use the smaller of the two for conservative sizing
        return min(base_size, kelly_size)

class MultiTimeframeAnalyzer:
    """Analyze multiple timeframes for confluence"""
    
    def __init__(self, timeframes: List[str] = ['5m', '15m', '1h', '4h']):
        self.timeframes = timeframes
        self.confluence_threshold = 0.7  # 70% agreement
    
    def analyze_confluence(self, asset: str, signals: Dict[str, Dict]) -> Dict:
        """Analyze signal confluence across timeframes"""
        timeframe_signals = {}
        
        for tf in self.timeframes:
            if tf in signals:
                tf_signal = signals[tf]
                timeframe_signals[tf] = {
                    'direction': tf_signal.get('type', 'NEUTRAL'),
                    'strength': tf_signal.get('confidence', 0.5),
                    'key_levels': tf_signal.get('key_levels', [])
                }
        
        # Calculate confluence score
        if not timeframe_signals:
            return {'confluence_score': 0, 'primary_direction': 'NEUTRAL', 'agreement': 0}
        
        # Count directional agreement
        buy_signals = sum(1 for s in timeframe_signals.values() if s['direction'] == 'BUY')
        sell_signals = sum(1 for s in timeframe_signals.values() if s['direction'] == 'SELL')
        total_signals = len(timeframe_signals)
        
        agreement = max(buy_signals, sell_signals) / total_signals if total_signals > 0 else 0
        primary_direction = 'BUY' if buy_signals > sell_signals else 'SELL' if sell_signals > buy_signals else 'NEUTRAL'
        
        # Weighted confluence score
        weights = {'5m': 0.2, '15m': 0.3, '1h': 0.3, '4h': 0.2}
        weighted_strength = sum(weights.get(tf, 0.25) * timeframe_signals[tf]['strength'] 
                              for tf in timeframe_signals)
        
        confluence_score = weighted_strength * agreement
        
        return {
            'confluence_score': confluence_score,
            'primary_direction': primary_direction,
            'agreement': agreement,
            'timeframe_signals': timeframe_signals,
            'has_confluence': confluence_score >= self.confluence_threshold
        }

class WalkForwardBacktester:
    """Walk-forward backtesting engine"""
    
    def __init__(self, train_ratio: float = 0.7, window_size: int = 100):
        self.train_ratio = train_ratio
        self.window_size = window_size
        self.results = []
    
    def run_walk_forward(self, data: pd.DataFrame, strategy_func, 
                        strategy_params: Dict) -> Dict:
        """Run walk-forward backtest"""
        if len(data) < self.window_size * 2:
            return {"error": "Insufficient data"}
        
        test_results = []
        total_windows = len(data) - self.window_size
        
        for i in range(0, total_windows, self.window_size // 2):
            train_data = data.iloc[i:i+self.window_size]
            test_data = data.iloc[i+self.window_size:i+self.window_size*2]
            
            if len(train_data) < self.window_size or len(test_data) < self.window_size:
                continue
            
            # Train on training window
            trained_params = self._train_strategy(train_data, strategy_func, strategy_params)
            
            # Test on out-of-sample window
            test_performance = self._test_strategy(test_data, strategy_func, trained_params)
            test_results.append(test_performance)
        
        # Aggregate results
        if test_results:
            return self._aggregate_results(test_results)
        return {"error": "No valid test windows"}
    
    def _train_strategy(self, data: pd.DataFrame, strategy_func, 
                       initial_params: Dict) -> Dict:
        """Train strategy parameters (simplified - implement optimization logic)"""
        # This is a placeholder - implement proper parameter optimization
        return initial_params
    
    def _test_strategy(self, data: pd.DataFrame, strategy_func, 
                      params: Dict) -> Dict:
        """Test strategy on out-of-sample data"""
        # This is a placeholder - implement proper backtesting
        returns = data['close'].pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return {
            "sharpe_ratio": sharpe,
            "total_return": (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
            "max_drawdown": self._calculate_max_drawdown(data['close']),
            "win_rate": 0.5  # Placeholder
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = prices / prices.cummax()
        return (1 - cumulative.min()) * 100
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results"""
        return {
            "avg_sharpe": np.mean([r.get("sharpe_ratio", 0) for r in results]),
            "avg_return": np.mean([r.get("total_return", 0) for r in results]),
            "max_drawdown": np.max([r.get("max_drawdown", 0) for r in results]),
            "consistency": np.std([r.get("total_return", 0) for r in results]),
            "num_windows": len(results)
        }

# ============================================================================
# 3. ENHANCED TRADING ENGINE WITH ALL FEATURES
# ============================================================================

class EnhancedTradingEngine:
    """Enhanced trading engine with all professional features"""
    
    def __init__(self):
        self.portfolio = st.session_state.paper_portfolio
        self.traded_symbols = st.session_state.traded_symbols
        self.cost_calculator = RealisticCostCalculator()
        self.regime_detector = MarketRegimeDetector()
        self.correlation_filter = CorrelationFilter(max_correlation=0.7)
        self.kelly_calculator = KellyCriterionCalculator()
        self.multi_tf_analyzer = MultiTimeframeAnalyzer()
        self.backtester = WalkForwardBacktester()
        
        # Performance metrics
        self.win_rate_history = []
        self.kelly_fractions = {}
        self.regime_history = {}
    
    def calculate_dynamic_position_size(self, asset: str, signal: Dict, 
                                      current_price: float) -> float:
        """Calculate position size with Kelly Criterion and regime adaptation"""
        
        # Get historical performance for this asset/strategy
        strategy_key = f"{asset}_{signal.get('strategy', 'default')}"
        win_rate = self.kelly_fractions.get(f"{strategy_key}_win_rate", 0.5)
        avg_win = self.kelly_fractions.get(f"{strategy_key}_avg_win", 0.02)
        avg_loss = self.kelly_fractions.get(f"{strategy_key}_avg_loss", 0.01)
        
        # Calculate Kelly fraction
        kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
            win_rate, avg_win, avg_loss
        )
        
        # Adjust for market regime
        regime = self.regime_history.get(asset, {}).get('regime', 'Unknown')
        if regime == 'Volatile':
            kelly_fraction *= 0.5  # Reduce size in volatile markets
        elif regime == 'Trending':
            kelly_fraction *= 1.2  # Increase in strong trends
        elif regime == 'Ranging':
            kelly_fraction *= 0.8  # Reduce in ranging markets
        
        # Calculate stop distance
        stop_distance = abs(signal['entry'] - signal['stop_loss'])
        if stop_distance == 0:
            stop_distance = current_price * 0.01  # Default 1% stop
        
        # Calculate final position size
        position_size = self.kelly_calculator.calculate_position_size(
            balance=self.portfolio['balance'],
            kelly_fraction=kelly_fraction,
            risk_per_trade=risk_per_trade,
            stop_distance=stop_distance,
            price=current_price
        )
        
        # Apply asset-specific lot size constraints
        asset_config = self._get_asset_config(asset)
        if asset_config:
            max_size = asset_config['lot_size'] * 5
            position_size = min(position_size, max_size)
        
        return position_size
    
    def check_correlation_constraints(self, new_asset: str) -> bool:
        """Check if new position violates correlation constraints"""
        current_positions = [
            pos['asset'] for pos in self.portfolio['positions'].values() 
            if pos['status'] == 'OPEN'
        ]
        
        return self.correlation_filter.can_add_position(new_asset, current_positions)
    
    def analyze_market_regime(self, asset: str, price_data: pd.DataFrame) -> Dict:
        """Analyze market regime for adaptive strategies"""
        if 'close' in price_data.columns:
            regime_info = self.regime_detector.detect_regime(price_data['close'])
            self.regime_history[asset] = regime_info
            return regime_info
        return {"regime": "Unknown", "confidence": 0.0}
    
    def execute_trade_with_costs(self, signal: Dict, current_price: float) -> Tuple[bool, str]:
        """Execute trade with realistic costs"""
        
        asset_config = self._get_asset_config(signal['asset'])
        if not asset_config:
            return False, "Asset configuration not found"
        
        # Check correlation constraints
        if not self.check_correlation_constraints(signal['asset']):
            return False, f"Correlation constraint violated for {signal['asset']}"
        
        # Check max positions
        if len(self.portfolio['positions']) >= max_positions:
            return False, "Maximum positions reached"
        
        # Calculate position size with Kelly Criterion
        position_size = self.calculate_dynamic_position_size(
            signal['asset'], signal, current_price
        )
        
        # Calculate realistic costs
        is_buy = signal['type'] == 'BUY'
        effective_entry_price = self.cost_calculator.get_effective_entry_price(
            signal['entry'], asset_config, is_buy
        )
        
        entry_cost = self.cost_calculator.calculate_entry_cost(
            effective_entry_price, position_size, asset_config
        )
        
        trade_value = effective_entry_price * position_size + entry_cost
        
        # Check margin
        if trade_value > self.portfolio['balance'] * 0.7:  # 70% max utilization
            return False, "Insufficient margin"
        
        # Create trade
        trade_id = f"{signal['asset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        trade = {
            'id': trade_id,
            'timestamp': datetime.now(),
            'asset': signal['asset'],
            'asset_name': signal['asset_name'],
            'type': signal['type'],
            'entry_price': effective_entry_price,
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'current_price': current_price,
            'status': 'OPEN',
            'pnl': -entry_cost,  # Initial P&L includes entry costs
            'pnl_percent': 0.0,
            'strategy': signal.get('strategy', 'Unknown'),
            'reason': signal.get('reason', ''),
            'confidence': signal.get('confidence', 0.5),
            'costs': {
                'spread': asset_config['spread'],
                'commission': asset_config['commission'],
                'slippage': asset_config['slippage'],
                'entry_cost': entry_cost
            },
            'market_regime': self.regime_history.get(signal['asset'], {}).get('regime', 'Unknown')
        }
        
        # Update portfolio
        self.portfolio['positions'][trade_id] = trade
        self.portfolio['balance'] -= entry_cost
        self.traded_symbols.add(signal['asset'])
        
        # Log trade
        log_entry = {
            'timestamp': datetime.now(),
            'action': 'OPEN',
            'trade_id': trade_id,
            'asset': signal['asset_name'],
            'symbol': signal['asset'],
            'type': signal['type'],
            'entry_price': effective_entry_price,
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'strategy': signal.get('strategy', 'Unknown'),
            'reason': signal.get('reason', ''),
            'confidence': signal.get('confidence', 0.5),
            'costs': entry_cost,
            'market_regime': trade['market_regime']
        }
        
        st.session_state.trade_log.append(log_entry)
        
        return True, trade_id
    
    def update_positions_with_costs(self, market_prices: Dict[str, float]) -> float:
        """Update positions with current prices and calculate exit costs"""
        total_pnl = 0
        positions_closed = []
        
        for trade_id, position in list(self.portfolio['positions'].items()):
            if position['status'] == 'OPEN':
                symbol = position['asset']
                current_price = market_prices.get(symbol)
                
                if current_price:
                    # Get asset config for exit costs
                    asset_config = self._get_asset_config(symbol)
                    
                    # Calculate raw P&L
                    if position['type'] == 'BUY':
                        raw_pnl = (current_price - position['entry_price']) * position['size']
                    else:  # SELL
                        raw_pnl = (position['entry_price'] - current_price) * position['size']
                    
                    # Calculate exit costs (will be realized when closing)
                    if asset_config:
                        exit_cost = self.cost_calculator.calculate_exit_cost(
                            current_price, position['size'], asset_config
                        )
                        net_pnl = raw_pnl - exit_cost - position['costs']['entry_cost']
                    else:
                        net_pnl = raw_pnl - position['costs']['entry_cost']
                    
                    pnl_percent = (net_pnl / (position['entry_price'] * position['size'])) * 100
                    
                    position['current_price'] = current_price
                    position['pnl'] = net_pnl
                    position['pnl_percent'] = pnl_percent
                    position['exit_cost_estimate'] = exit_cost if asset_config else 0
                    
                    total_pnl += net_pnl
                    
                    # Check exit conditions
                    exit_reason = None
                    
                    # Stop Loss (including costs)
                    stop_level = position['stop_loss']
                    if position['type'] == 'BUY':
                        if current_price <= stop_level:
                            exit_reason = "Stop Loss Hit"
                    else:  # SELL
                        if current_price >= stop_level:
                            exit_reason = "Stop Loss Hit"
                    
                    # Take Profit
                    tp_level = position['take_profit']
                    if position['type'] == 'BUY':
                        if current_price >= tp_level:
                            exit_reason = "Take Profit Hit"
                    else:  # SELL
                        if current_price <= tp_level:
                            exit_reason = "Take Profit Hit"
                    
                    # Regime-based exit (if regime changes significantly)
                    current_regime = self.regime_history.get(symbol, {}).get('regime', 'Unknown')
                    if (position['market_regime'] == 'Trending' and 
                        current_regime in ['Ranging', 'Volatile']):
                        exit_reason = "Regime Change Exit"
                    
                    if exit_reason:
                        self.close_position_with_costs(trade_id, current_price, exit_reason, asset_config)
                        positions_closed.append(trade_id)
        
        # Update performance metrics
        self._update_kelly_metrics(positions_closed)
        
        # Update portfolio metrics
        self.portfolio['total_pnl'] += total_pnl
        self.portfolio['daily_pnl'] += total_pnl
        
        # Update equity curve
        current_equity = self.portfolio['balance'] + total_pnl
        self.portfolio['equity_curve'].append(current_equity)
        
        # Update drawdown
        if self.portfolio['equity_curve']:
            current_peak = max(self.portfolio['equity_curve'])
            current_value = self.portfolio['equity_curve'][-1]
            drawdown = ((current_peak - current_value) / current_peak) * 100
            self.portfolio['max_drawdown'] = max(self.portfolio['max_drawdown'], drawdown)
        
        return total_pnl
    
    def close_position_with_costs(self, trade_id: str, exit_price: float, 
                                reason: str, asset_config: Dict = None):
        """Close position with realistic exit costs"""
        if trade_id in self.portfolio['positions']:
            position = self.portfolio['positions'][trade_id]
            
            # Calculate exit costs
            if asset_config:
                exit_cost = self.cost_calculator.calculate_exit_cost(
                    exit_price, position['size'], asset_config
                )
            else:
                exit_cost = 0
            
            # Calculate final P&L
            if position['type'] == 'BUY':
                raw_pnl = (exit_price - position['entry_price']) * position['size']
            else:
                raw_pnl = (position['entry_price'] - exit_price) * position['size']
            
            net_pnl = raw_pnl - exit_cost - position['costs']['entry_cost']
            final_pnl_percent = (net_pnl / (position['entry_price'] * position['size'])) * 100
            
            # Update portfolio balance
            self.portfolio['balance'] += net_pnl
            
            # Update win/loss stats
            if net_pnl > 0:
                self.portfolio['winning_trades'] += 1
            else:
                self.portfolio['losing_trades'] += 1
            
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
                'pnl': net_pnl,
                'pnl_percent': final_pnl_percent,
                'reason': reason,
                'holding_time': (datetime.now() - position['timestamp']).total_seconds() / 60,
                'exit_cost': exit_cost,
                'total_costs': exit_cost + position['costs']['entry_cost'],
                'market_regime_entry': position['market_regime'],
                'market_regime_exit': self.regime_history.get(position['asset'], {}).get('regime', 'Unknown')
            }
            
            st.session_state.trade_log.append(log_entry)
            
            # Remove from positions
            del self.portfolio['positions'][trade_id]
            if position['asset'] in self.traded_symbols:
                self.traded_symbols.remove(position['asset'])
            
            return True
        
        return False
    
    def _update_kelly_metrics(self, closed_trades: List[str]):
        """Update Kelly Criterion metrics based on closed trades"""
        closed_positions = []
        for trade_id in closed_trades:
            # Find corresponding log entry
            for log in reversed(st.session_state.trade_log):
                if log.get('trade_id') == trade_id and log['action'] == 'CLOSE':
                    closed_positions.append(log)
                    break
        
        if not closed_positions:
            return
        
        # Group by asset and strategy
        trades_by_strategy = defaultdict(list)
        for trade in closed_positions:
            strategy_key = f"{trade['symbol']}_{trade.get('strategy', 'default')}"
            trades_by_strategy[strategy_key].append(trade)
        
        # Update metrics for each strategy
        for strategy_key, trades in trades_by_strategy.items():
            if len(trades) >= 5:  # Require minimum trades for meaningful stats
                winning_trades = [t for t in trades if t['pnl'] > 0]
                losing_trades = [t for t in trades if t['pnl'] < 0]
                
                win_rate = len(winning_trades) / len(trades) if trades else 0.5
                avg_win = np.mean([t['pnl_percent']/100 for t in winning_trades]) if winning_trades else 0.02
                avg_loss = abs(np.mean([t['pnl_percent']/100 for t in losing_trades])) if losing_trades else 0.01
                
                self.kelly_fractions[f"{strategy_key}_win_rate"] = win_rate
                self.kelly_fractions[f"{strategy_key}_avg_win"] = avg_win
                self.kelly_fractions[f"{strategy_key}_avg_loss"] = avg_loss
    
    def _get_asset_config(self, symbol: str) -> Optional[Dict]:
        """Get asset configuration by symbol"""
        for category, assets in ASSET_CONFIG.items():
            for name, config in assets.items():
                if config['symbol'] == symbol:
                    return config
        return None
    
    def run_walk_forward_test(self, asset: str, data: pd.DataFrame, 
                            strategy_params: Dict) -> Dict:
        """Run walk-forward backtest for strategy optimization"""
        return self.backtester.run_walk_forward(data, None, strategy_params)  # Placeholder

# ============================================================================
# 4. ENHANCED UI COMPONENTS
# ============================================================================

def display_regime_analysis(regime_info: Dict):
    """Display market regime analysis in UI"""
    regime = regime_info.get('regime', 'Unknown')
    confidence = regime_info.get('confidence', 0.0)
    metrics = regime_info.get('metrics', {})
    
    regime_colors = {
        'Trending': '#10b981',
        'Ranging': '#f59e0b',
        'Volatile': '#ef4444',
        'Mean Reverting': '#8b5cf6',
        'Unknown': '#94a3b8'
    }
    
    color = regime_colors.get(regime, '#94a3b8')
    
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #141b2d, #1a2238); 
                padding: 1rem; border-radius: 10px; border-left: 4px solid {color};
                margin: 10px 0;">
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
                    Volatility: {metrics.get('volatility', 0)*100:.1f}% |
                    Trend Strength: {metrics.get('trend_strength', 0):.2f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_correlation_analysis(correlation_filter: CorrelationFilter, 
                               current_positions: List[str]):
    """Display portfolio correlation analysis"""
    if not current_positions:
        return
    
    avg_correlation = correlation_filter.get_portfolio_correlation(current_positions)
    correlation_color = '#10b981' if avg_correlation < 0.5 else '#f59e0b' if avg_correlation < 0.7 else '#ef4444'
    
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #141b2d, #1a2238); 
                padding: 1rem; border-radius: 10px; border: 1px solid #334155;
                margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: #94a3b8;">
                    🔗 Portfolio Correlation Analysis
                </h4>
                <p style="margin: 5px 0 0 0; color: {correlation_color}; font-size: 1.1rem;">
                    Average Correlation: <strong>{avg_correlation:.2f}</strong>
                </p>
            </div>
            <div style="text-align: right;">
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    Max Allowed: 0.70 | Positions: {len(current_positions)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_kelly_sizing_info(kelly_fraction: float, position_size: float, 
                            win_rate: float, avg_win: float, avg_loss: float):
    """Display Kelly Criterion position sizing information"""
    
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #141b2d, #1a2238); 
                padding: 1rem; border-radius: 10px; border: 1px solid #334155;
                margin: 10px 0;">
        <h4 style="margin: 0 0 15px 0; color: #94a3b8;">
            🎯 Dynamic Position Sizing (Kelly Criterion)
        </h4>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Kelly Fraction</div>
                <div style="color: #3b82f6; font-size: 1.2rem; font-weight: bold;">{kelly_fraction:.3f}</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Position Size</div>
                <div style="color: #10b981; font-size: 1.2rem; font-weight: bold;">{position_size:.4f}</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Win Rate</div>
                <div style="color: #f59e0b; font-size: 1.2rem; font-weight: bold;">{win_rate*100:.1f}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Win/Loss Ratio</div>
                <div style="color: #8b5cf6; font-size: 1.2rem; font-weight: bold;">{avg_win/avg_loss:.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_multi_timeframe_confluence(confluence_info: Dict):
    """Display multi-timeframe confluence analysis"""
    
    if not confluence_info.get('timeframe_signals'):
        return
    
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #141b2d, #1a2238); 
                padding: 1rem; border-radius: 10px; border: 1px solid #334155;
                margin: 10px 0;">
        <h4 style="margin: 0 0 15px 0; color: #94a3b8;">
            ⚡ Multi-Timeframe Confluence
        </h4>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
            <div>
                <span style="color: #94a3b8;">Overall Score:</span>
                <span style="color: #3b82f6; font-size: 1.3rem; font-weight: bold; margin-left: 10px;">
                    {confluence_info['confluence_score']:.2f}
                </span>
            </div>
            <div>
                <span style="color: #94a3b8;">Agreement:</span>
                <span style="color: #10b981; font-size: 1.1rem; font-weight: bold; margin-left: 10px;">
                    {confluence_info['agreement']*100:.0f}%
                </span>
            </div>
            <div>
                <span style="color: #94a3b8;">Primary Direction:</span>
                <span style="color: {'#10b981' if confluence_info['primary_direction'] == 'BUY' else '#ef4444' if confluence_info['primary_direction'] == 'SELL' else '#f59e0b'}; 
                      font-size: 1.1rem; font-weight: bold; margin-left: 10px;">
                    {confluence_info['primary_direction']}
                </span>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
    """, unsafe_allow_html=True)
    
    # Display individual timeframe signals
    for tf, signal in confluence_info['timeframe_signals'].items():
        direction = signal['direction']
        strength = signal['strength']
        color = '#10b981' if direction == 'BUY' else '#ef4444' if direction == 'SELL' else '#f59e0b'
        
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.9rem;">{tf}</div>
                <div style="color: {color}; font-size: 1.1rem; font-weight: bold;">{direction}</div>
                <div style="color: #f59e0b; font-size: 0.9rem;">Strength: {strength:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def display_cost_breakdown(entry_cost: float, asset_config: Dict, 
                         position_size: float, price: float):
    """Display realistic cost breakdown"""
    
    spread_cost = price * asset_config['spread'] * position_size
    commission_cost = price * position_size * asset_config['commission']
    slippage_cost = price * asset_config['slippage'] * position_size
    
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #141b2d, #1a2238); 
                padding: 1rem; border-radius: 10px; border: 1px solid #334155;
                margin: 10px 0;">
        <h4 style="margin: 0 0 15px 0; color: #94a3b8;">
            💰 Realistic Trading Costs
        </h4>
        
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Spread</div>
                <div style="color: #ef4444; font-size: 1rem; font-weight: bold;">${spread_cost:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">{asset_config['spread']*10000:.1f} bps</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Commission</div>
                <div style="color: #f59e0b; font-size: 1rem; font-weight: bold;">${commission_cost:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">{asset_config['commission']*100:.2f}%</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Slippage</div>
                <div style="color: #8b5cf6; font-size: 1rem; font-weight: bold;">${slippage_cost:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">{asset_config['slippage']*100:.2f}%</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Total Entry</div>
                <div style="color: #3b82f6; font-size: 1rem; font-weight: bold;">${entry_cost:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Costs</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.85rem;">Effective Price</div>
                <div style="color: #10b981; font-size: 1rem; font-weight: bold;">
                    ${price * (1 + asset_config['spread'] + asset_config['slippage']):.4f}
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Entry</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 5. INTEGRATED MAIN FUNCTION WITH ENHANCED FEATURES
# ============================================================================

def main_enhanced():
    """Enhanced main function with all professional features"""
    
    # Initialize enhanced trading engine
    trading_engine = EnhancedTradingEngine()
    smc_algo = AdvancedSMCAlgorithm(strategy=selected_strategy, 
                                   fvg_lookback=fvg_period, 
                                   swing_period=swing_period)
    
    # Display Professional Dashboard
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    # ... [Existing metric grid code] ...
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Analytics Section
    st.markdown("### 📊 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market Regime Detection
        if selected_asset and asset_info:
            df = fetch_intraday_data(asset_info['symbol'], timeframe)
            if not df.empty and 'close' in df.columns:
                regime_info = trading_engine.analyze_market_regime(
                    asset_info['symbol'], df
                )
                display_regime_analysis(regime_info)
    
    with col2:
        # Portfolio Correlation
        current_positions = [
            pos['asset'] for pos in st.session_state.paper_portfolio['positions'].values() 
            if pos['status'] == 'OPEN'
        ]
        if current_positions:
            display_correlation_analysis(
                trading_engine.correlation_filter, 
                current_positions
            )
    
    # Multi-Timeframe Analysis
    st.markdown("### ⚡ Multi-Timeframe Analysis")
    
    timeframes_to_analyze = ['5m', '15m', '1h', '4h']
    multi_tf_signals = {}
    
    for tf in timeframes_to_analyze:
        df_tf = fetch_intraday_data(asset_info['symbol'], tf)
        if not df_tf.empty:
            df_analyzed = smc_algo.analyze_market_structure(df_tf)
            df_analyzed = smc_algo.identify_fvgs(df_analyzed)
            df_analyzed = smc_algo.identify_orderblocks(df_analyzed)
            
            signals = smc_algo.generate_smc_signals(df_analyzed, asset_info)
            if signals:
                multi_tf_signals[tf] = signals[0] if signals else {}
    
    # Analyze confluence
    if multi_tf_signals:
        confluence_info = trading_engine.multi_tf_analyzer.analyze_confluence(
            asset_info['symbol'], multi_tf_signals
        )
        display_multi_timeframe_confluence(confluence_info)
    
    # Enhanced Trading Signals with Realistic Costs
    st.markdown("### 🎯 Enhanced Trading Signals")
    
    if selected_asset and asset_info:
        df = fetch_intraday_data(asset_info['symbol'], timeframe)
        
        if not df.empty:
            current_price = df['close'].iloc[-1]
            
            # Generate signals
            df_analyzed = smc_algo.analyze_market_structure(df)
            df_analyzed = smc_algo.identify_fvgs(df_analyzed)
            df_analyzed = smc_algo.identify_orderblocks(df_analyzed)
            
            signals = smc_algo.generate_smc_signals(df_analyzed, asset_info)
            
            if signals:
                for i, signal in enumerate(signals[:3]):
                    # Calculate Kelly-based position size
                    position_size = trading_engine.calculate_dynamic_position_size(
                        signal['asset'], signal, current_price
                    )
                    signal['size'] = position_size
                    
                    # Calculate realistic costs
                    entry_cost = RealisticCostCalculator.calculate_entry_cost(
                        current_price, position_size, asset_info
                    )
                    
                    # Display signal with enhanced information
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Display signal card
                        signal_type_class = "buy" if signal['type'] == 'BUY' else "sell"
                        st.markdown(f"""
                        <div class="signal-card {signal_type_class}">
                            <div class="signal-header">
                                <div class="signal-title">{signal['asset_name']} - {signal['type']}</div>
                                <div class="confidence-badge">{signal['confidence']*100:.0f}%</div>
                            </div>
                            <div class="signal-grid">
                                <div class="signal-item">
                                    <div class="signal-label">Entry Price</div>
                                    <div class="signal-value">${signal['entry']:.4f}</div>
                                </div>
                                <div class="signal-item">
                                    <div class="signal-label">Position Size</div>
                                    <div class="signal-value">{position_size:.4f}</div>
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
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Display cost breakdown
                        display_cost_breakdown(entry_cost, asset_info, position_size, current_price)
                        
                        # Execute button with enhanced validation
                        if st.button(f"🚀 Execute {signal['type']}", key=f"enhanced_execute_{i}"):
                            # Check correlation constraints
                            if trading_engine.check_correlation_constraints(signal['asset']):
                                success, trade_id = trading_engine.execute_trade_with_costs(
                                    signal, current_price
                                )
                                if success:
                                    st.success(f"✅ Trade executed! ID: {trade_id}")
                                    st.rerun()
                                else:
                                    st.error("❌ Trade execution failed")
                            else:
                                st.warning("⚠️ Trade rejected: Correlation constraint violation")
    
    # Walk-Forward Backtesting Section
    st.markdown("---")
    st.markdown("### 📈 Walk-Forward Backtesting")
    
    if st.button("Run Walk-Forward Analysis", type="secondary"):
        with st.spinner("Running walk-forward backtest..."):
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            historical_data = yf.download(
                asset_info['symbol'], 
                start=start_date, 
                end=end_date,
                interval='1d'
            )
            
            if not historical_data.empty:
                results = trading_engine.run_walk_forward_test(
                    asset_info['symbol'], 
                    historical_data,
                    {"strategy": selected_strategy}
                )
                
                if "error" not in results:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Sharpe", f"{results.get('avg_sharpe', 0):.2f}")
                    
                    with col2:
                        st.metric("Avg Return", f"{results.get('avg_return', 0):.1f}%")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.1f}%")
                    
                    with col4:
                        st.metric("Consistency", f"{results.get('consistency', 0):.1f}")
                else:
                    st.error("Backtesting failed")
    
    # Enhanced Portfolio Management
    st.markdown("---")
    st.markdown("### 💼 Enhanced Portfolio Management")
    
    # Display portfolio analytics
    positions = st.session_state.paper_portfolio['positions']
    if positions:
        # Calculate portfolio metrics
        total_exposure = sum(
            pos['entry_price'] * pos['size'] 
            for pos in positions.values() 
            if pos['status'] == 'OPEN'
        )
        
        portfolio_beta = trading_engine.correlation_filter.get_portfolio_correlation(
            [pos['asset'] for pos in positions.values() if pos['status'] == 'OPEN']
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Exposure", f"${total_exposure:,.2f}")
        
        with col2:
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        
        with col3:
            # Calculate cost efficiency
            total_costs = sum(
                pos['costs']['entry_cost'] 
                for pos in positions.values() 
                if 'costs' in pos
            )
            st.metric("Total Costs", f"${total_costs:,.2f}")
        
        with col4:
            # Calculate regime distribution
            regimes = [
                pos.get('market_regime', 'Unknown') 
                for pos in positions.values() 
                if pos['status'] == 'OPEN'
            ]
            if regimes:
                regime_counts = pd.Series(regimes).value_counts()
                main_regime = regime_counts.index[0] if len(regime_counts) > 0 else "Unknown"
                st.metric("Main Regime", main_regime)

# ============================================================================
# 6. SIDEBAR ENHANCEMENTS
# ============================================================================

# Add to sidebar configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 ADVANCED SETTINGS")

# Dynamic Risk Management
with st.sidebar.expander("Dynamic Risk Parameters"):
    use_kelly = st.checkbox("Use Kelly Criterion", value=True)
    kelly_fraction = st.slider("Kelly Fraction (%)", 10, 100, 25, 
                              help="Percentage of Kelly to use (25% = quarter-Kelly)")
    max_portfolio_correlation = st.slider("Max Portfolio Correlation", 0.5, 0.9, 0.7, 0.05)

# Market Regime Settings
with st.sidebar.expander("Regime Detection"):
    regime_lookback = st.slider("Regime Lookback (periods)", 20, 200, 50)
    use_regime_adaptive = st.checkbox("Adaptive Strategies", value=True)

# Cost Settings
with st.sidebar.expander("Trading Costs"):
    commission_multiplier = st.slider("Commission Multiplier", 0.5, 2.0, 1.0, 0.1)
    slippage_multiplier = st.slider("Slippage Multiplier", 0.5, 3.0, 1.0, 0.1)

# Backtesting Settings
with st.sidebar.expander("Backtesting"):
    walk_forward_windows = st.slider("Walk-Forward Windows", 3, 20, 10)
    min_backtest_trades = st.slider("Min Trades for Backtest", 10, 100, 30)

# ============================================================================
# 7. ENHANCED CSS FOR NEW COMPONENTS
# ============================================================================

st.markdown("""
<style>
    /* Enhanced metric cards */
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
    
    /* Regime indicator */
    .regime-indicator {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .regime-trending {
        background: rgba(16, 185, 129, 0.2);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .regime-ranging {
        background: rgba(245, 158, 11, 0.2);
        color: var(--accent-yellow);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .regime-volatile {
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Cost breakdown */
    .cost-breakdown {
        background: rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .cost-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .cost-item:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    /* Confluence meter */
    .confluence-meter {
        height: 8px;
        background: var(--border-color);
        border-radius: 4px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .confluence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Enhanced tabs for advanced features */
    .advanced-tab {
        background: linear-gradient(145deg, var(--secondary-bg), #1a2238);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize session state for enhanced features
    if 'enhanced_metrics' not in st.session_state:
        st.session_state.enhanced_metrics = {
            'regime_history': {},
            'correlation_matrix': None,
            'kelly_metrics': {},
            'backtest_results': {}
        }
    
    # Run enhanced main function
    main_enhanced()
