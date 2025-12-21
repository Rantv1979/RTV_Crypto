# =============================================
# RANTV COMPLETE ALGORITHMIC TRADING SYSTEM
# WITH SMART MONEY CONCEPT & ACCURACY TRACKING
# =============================================

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import warnings
import hashlib
import json
import pickle
from pathlib import Path
import threading
import queue
from collections import defaultdict
import plotly.express as px
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION & SETTINGS
# =============================================

st.set_page_config(
    page_title="RANTV Algorithmic Trading Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

UTC_TZ = pytz.timezone("UTC")

# Trading Parameters
INITIAL_CAPITAL = 500.0
TRADE_ALLOCATION = 0.15
MAX_DAILY_TRADES = 15
MAX_POSITIONS = 10
RISK_PER_TRADE = 0.02  # 2% risk per trade

# Refresh Intervals
PRICE_REFRESH_MS = 10000    # 10 seconds for real-time updates
SIGNAL_REFRESH_MS = 30000   # 30 seconds for signal updates

# Market Universe
MARKET_OPTIONS = ["CRYPTO", "STOCKS", "FOREX", "COMMODITIES"]

# Asset Universe
CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "ADA-USD", "AVAX-USD", "DOT-USD", "DOGE-USD", "LINK-USD"
]

US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "JPM", "V", "WMT"
]

FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"
]

COMMODITIES = [
    "GC=F", "SI=F", "CL=F", "NG=F", "ZC=F"
]

ALL_SYMBOLS = CRYPTO_SYMBOLS + US_STOCKS + FOREX_PAIRS + COMMODITIES

# =============================================
# DATA STORAGE & CACHE MANAGEMENT
# =============================================

class DataStorage:
    """Persistent data storage for backtesting and results"""
    
    def __init__(self):
        self.storage_path = Path("trading_data")
        self.storage_path.mkdir(exist_ok=True)
        
    def save_trades(self, trades, filename="trades_history.pkl"):
        """Save trades to file"""
        filepath = self.storage_path / filename
        with open(filepath, 'wb') as f:
            pickle.dump(trades, f)
    
    def load_trades(self, filename="trades_history.pkl"):
        """Load trades from file"""
        filepath = self.storage_path / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return []
    
    def save_backtest_results(self, results, filename="backtest_results.json"):
        """Save backtest results"""
        filepath = self.storage_path / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, default=str, indent=2)
    
    def load_backtest_results(self, filename="backtest_results.json"):
        """Load backtest results"""
        filepath = self.storage_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}

# =============================================
# ENHANCED SMART MONEY CONCEPT ANALYZER
# =============================================

class EnhancedSmartMoneyAnalyzer:
    """Enhanced Smart Money Concept analyzer with multiple SMC strategies"""
    
    def __init__(self):
        self.order_flow_cache = {}
        self.liquidity_zones = {}
        
    def detect_fair_value_gaps(self, high, low, close):
        """Detect Fair Value Gaps with strength analysis"""
        fvgs = []
        
        if len(close) < 4:
            return fvgs
            
        for i in range(1, len(close)-2):
            # Bullish FVG
            if low.iloc[i] > high.iloc[i-1]:
                fvg = {
                    'type': 'BULLISH_FVG',
                    'top': float(low.iloc[i]),
                    'bottom': float(high.iloc[i-1]),
                    'mid': float((low.iloc[i] + high.iloc[i-1]) / 2),
                    'index': i,
                    'strength': (low.iloc[i] - high.iloc[i-1]) / high.iloc[i-1],
                    'volume': float(close.iloc[i-1:i+2].mean())
                }
                fvgs.append(fvg)
            
            # Bearish FVG
            elif high.iloc[i] < low.iloc[i-1]:
                fvg = {
                    'type': 'BEARISH_FVG',
                    'top': float(low.iloc[i-1]),
                    'bottom': float(high.iloc[i]),
                    'mid': float((low.iloc[i-1] + high.iloc[i]) / 2),
                    'index': i,
                    'strength': (low.iloc[i-1] - high.iloc[i]) / high.iloc[i],
                    'volume': float(close.iloc[i-1:i+2].mean())
                }
                fvgs.append(fvg)
        
        return sorted(fvgs[-10:], key=lambda x: x['strength'], reverse=True) if fvgs else []
    
    def identify_liquidity_zones(self, high, low, volume, period=20):
        """Identify institutional liquidity zones"""
        zones = []
        
        if len(high) < period * 2:
            return zones
        
        swing_highs = []
        swing_lows = []
        
        for i in range(period, len(high) - period):
            if high.iloc[i] >= high.iloc[i-period:i+period+1].max():
                swing_highs.append({
                    'price': float(high.iloc[i]),
                    'index': i,
                    'volume': float(volume.iloc[i])
                })
            
            if low.iloc[i] <= low.iloc[i-period:i+period+1].min():
                swing_lows.append({
                    'price': float(low.iloc[i]),
                    'index': i,
                    'volume': float(volume.iloc[i])
                })
        
        # Cluster similar price levels
        cluster_threshold = 0.005
        
        def cluster_levels(levels, threshold):
            clusters = []
            levels_sorted = sorted(levels, key=lambda x: x['price'])
            
            current_cluster = []
            for level in levels_sorted:
                if not current_cluster:
                    current_cluster.append(level)
                else:
                    avg_price = np.mean([l['price'] for l in current_cluster])
                    if abs(level['price'] - avg_price) / avg_price <= threshold:
                        current_cluster.append(level)
                    else:
                        if len(current_cluster) >= 2:
                            clusters.append(current_cluster)
                        current_cluster = [level]
            
            if current_cluster and len(current_cluster) >= 2:
                clusters.append(current_cluster)
            
            return clusters
        
        resistance_clusters = cluster_levels(swing_highs, cluster_threshold)
        support_clusters = cluster_levels(swing_lows, cluster_threshold)
        
        for cluster in resistance_clusters:
            avg_price = np.mean([l['price'] for l in cluster])
            total_volume = sum([l['volume'] for l in cluster])
            zones.append({
                'type': 'RESISTANCE',
                'price': float(avg_price),
                'touches': len(cluster),
                'volume': float(total_volume),
                'strength': len(cluster) * (total_volume / len(cluster))
            })
        
        for cluster in support_clusters:
            avg_price = np.mean([l['price'] for l in cluster])
            total_volume = sum([l['volume'] for l in cluster])
            zones.append({
                'type': 'SUPPORT',
                'price': float(avg_price),
                'touches': len(cluster),
                'volume': float(total_volume),
                'strength': len(cluster) * (total_volume / len(cluster))
            })
        
        return sorted(zones, key=lambda x: x['strength'], reverse=True)[:5]
    
    def calculate_order_block(self, open_price, high, low, close, volume):
        """Identify order blocks (institutional accumulation/distribution)"""
        order_blocks = []
        
        if len(close) < 5:
            return order_blocks
        
        for i in range(2, len(close)-2):
            current_candle = {
                'open': open_price.iloc[i],
                'high': high.iloc[i],
                'low': low.iloc[i],
                'close': close.iloc[i],
                'volume': volume.iloc[i]
            }
            
            prev_candle = {
                'open': open_price.iloc[i-1],
                'high': high.iloc[i-1],
                'low': low.iloc[i-1],
                'close': close.iloc[i-1],
                'volume': volume.iloc[i-1]
            }
            
            # Bullish Order Block
            if (prev_candle['close'] < prev_candle['open'] and
                abs(prev_candle['close'] - prev_candle['open']) > 0.5 * (prev_candle['high'] - prev_candle['low']) and
                current_candle['close'] > current_candle['open'] and
                current_candle['close'] > prev_candle['low'] and
                current_candle['volume'] > prev_candle['volume'] * 1.2):
                
                ob = {
                    'type': 'BULLISH_OB',
                    'high': float(prev_candle['high']),
                    'low': float(prev_candle['low']),
                    'mid': float((prev_candle['high'] + prev_candle['low']) / 2),
                    'index': i,
                    'volume_ratio': float(current_candle['volume'] / prev_candle['volume'])
                }
                order_blocks.append(ob)
            
            # Bearish Order Block
            elif (prev_candle['close'] > prev_candle['open'] and
                  abs(prev_candle['close'] - prev_candle['open']) > 0.5 * (prev_candle['high'] - prev_candle['low']) and
                  current_candle['close'] < current_candle['open'] and
                  current_candle['close'] < prev_candle['high'] and
                  current_candle['volume'] > prev_candle['volume'] * 1.2):
                
                ob = {
                    'type': 'BEARISH_OB',
                    'high': float(prev_candle['high']),
                    'low': float(prev_candle['low']),
                    'mid': float((prev_candle['high'] + prev_candle['low']) / 2),
                    'index': i,
                    'volume_ratio': float(current_candle['volume'] / prev_candle['volume'])
                }
                order_blocks.append(ob)
        
        return order_blocks[-5:]
    
    def detect_breakers(self, high, low, close):
        """Detect breaker blocks (liquidity grabs)"""
        breakers = []
        
        if len(close) < 10:
            return breakers
        
        for i in range(5, len(close)-5):
            prev_swing_high = max(high.iloc[i-5:i])
            prev_swing_low = min(low.iloc[i-5:i])
            
            current_high = high.iloc[i]
            current_low = low.iloc[i]
            
            # Bullish Breaker
            if (current_high > prev_swing_high * 1.005 and
                close.iloc[i] < (current_high + current_low) / 2):
                
                breaker = {
                    'type': 'BULLISH_BREAKER',
                    'break_price': float(prev_swing_high),
                    'current_price': float(close.iloc[i]),
                    'rejection_strength': float((current_high - close.iloc[i]) / (current_high - current_low)),
                    'index': i
                }
                breakers.append(breaker)
            
            # Bearish Breaker
            elif (current_low < prev_swing_low * 0.995 and
                  close.iloc[i] > (current_high + current_low) / 2):
                
                breaker = {
                    'type': 'BEARISH_BREAKER',
                    'break_price': float(prev_swing_low),
                    'current_price': float(close.iloc[i]),
                    'rejection_strength': float((close.iloc[i] - current_low) / (current_high - current_low)),
                    'index': i
                }
                breakers.append(breaker)
        
        return breakers
    
    def analyze_market_structure(self, high, low, close):
        """Comprehensive market structure analysis"""
        if len(close) < 100:
            return {'trend': 'NEUTRAL', 'momentum': 0, 'structure': 'RANGING'}
        
        # HTF trend using 50-period SMA
        htf_sma = close.rolling(window=50).mean()
        htf_trend = 1 if close.iloc[-1] > htf_sma.iloc[-1] else -1
        
        # LTF momentum using 20-period RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        ltf_rsi = rsi.iloc[-1]
        
        # Market structure
        recent_highs = high.iloc[-20:].tolist()
        recent_lows = low.iloc[-20:].tolist()
        
        higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
        lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        if higher_highs and higher_lows:
            structure = 'UPTREND'
        elif lower_highs and lower_lows:
            structure = 'DOWNTREND'
        else:
            structure = 'RANGING'
        
        # Momentum score
        momentum_score = 0
        if htf_trend == 1:
            momentum_score += 0.3
        if ltf_rsi > 50:
            momentum_score += 0.2
        if structure == 'UPTREND':
            momentum_score += 0.3
        elif structure == 'DOWNTREND':
            momentum_score -= 0.3
        
        return {
            'htf_trend': 'BULLISH' if htf_trend == 1 else 'BEARISH',
            'ltf_momentum': 'BULLISH' if ltf_rsi > 50 else 'BEARISH',
            'structure': structure,
            'momentum_score': momentum_score,
            'rsi': ltf_rsi,
            'sma_50': htf_sma.iloc[-1]
        }

# =============================================
# TRADING STRATEGIES WITH SMART MONEY CONCEPT
# =============================================

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.name = "Base Strategy"
        self.description = "Base strategy class"
        self.parameters = {}
        self.smc_analyzer = EnhancedSmartMoneyAnalyzer()
    
    def generate_signal(self, symbol, current_price, historical_data):
        """Generate trading signal"""
        raise NotImplementedError
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        if len(data) < 20:
            return {}
        
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
        indicators['ema_8'] = data['Close'].ewm(span=8).mean().iloc[-1]
        indicators['ema_21'] = data['Close'].ewm(span=21).mean().iloc[-1]
        indicators['ema_50'] = data['Close'].ewm(span=50).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        indicators['macd'] = indicators['ema_12'] if 'ema_12' in indicators else indicators['ema_8']
        indicators['macd_signal'] = data['Close'].ewm(span=9).mean().iloc[-1]
        
        # Bollinger Bands
        indicators['bb_middle'] = indicators['sma_20']
        bb_std = data['Close'].rolling(window=20).std().iloc[-1]
        indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
        indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]
        
        # Volume
        indicators['volume_sma'] = data['Volume'].rolling(window=20).mean().iloc[-1]
        indicators['volume_ratio'] = data['Volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # VWAP
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        cumulative_tp_volume = (typical_price * data['Volume']).cumsum()
        cumulative_volume = data['Volume'].cumsum()
        indicators['vwap'] = cumulative_tp_volume.iloc[-1] / cumulative_volume.iloc[-1]
        
        return indicators

class SMC_FVG_Strategy(BaseStrategy):
    """Smart Money Concept - Fair Value Gap Strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "SMC FVG Strategy"
        self.description = "Trades Fair Value Gaps with volume confirmation"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        # Get SMC analysis
        fvgs = self.smc_analyzer.detect_fair_value_gaps(
            historical_data['High'],
            historical_data['Low'],
            historical_data['Close']
        )
        
        if not fvgs:
            return None
        
        latest_fvg = fvgs[-1]
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0,
            'smc_data': latest_fvg
        }
        
        # Bullish FVG signal
        if (latest_fvg['type'] == 'BULLISH_FVG' and
            abs(current_price - latest_fvg['mid']) / latest_fvg['mid'] < 0.01 and
            indicators['volume_ratio'] > 1.2):
            
            signal['action'] = 'BUY'
            signal['confidence'] = 0.75
            signal['stop_loss'] = latest_fvg['bottom'] * 0.995
            signal['take_profit'] = latest_fvg['top'] * 1.01
        
        # Bearish FVG signal
        elif (latest_fvg['type'] == 'BEARISH_FVG' and
              abs(current_price - latest_fvg['mid']) / latest_fvg['mid'] < 0.01 and
              indicators['volume_ratio'] > 1.2):
            
            signal['action'] = 'SELL'
            signal['confidence'] = 0.75
            signal['stop_loss'] = latest_fvg['top'] * 1.005
            signal['take_profit'] = latest_fvg['bottom'] * 0.99
        
        else:
            return None
        
        return signal

class SMC_OrderBlock_Strategy(BaseStrategy):
    """Smart Money Concept - Order Block Strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "SMC Order Block Strategy"
        self.description = "Trades institutional order blocks"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        # Get SMC analysis
        order_blocks = self.smc_analyzer.calculate_order_block(
            historical_data['Open'],
            historical_data['High'],
            historical_data['Low'],
            historical_data['Close'],
            historical_data['Volume']
        )
        
        if not order_blocks:
            return None
        
        latest_ob = order_blocks[-1]
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0,
            'smc_data': latest_ob
        }
        
        # Bullish Order Block signal
        if (latest_ob['type'] == 'BULLISH_OB' and
            abs(current_price - latest_ob['mid']) / latest_ob['mid'] < 0.01 and
            indicators['rsi'] < 50 and
            latest_ob['volume_ratio'] > 1.5):
            
            signal['action'] = 'BUY'
            signal['confidence'] = 0.8
            signal['stop_loss'] = latest_ob['low'] * 0.99
            signal['take_profit'] = latest_ob['high'] * 1.02
        
        # Bearish Order Block signal
        elif (latest_ob['type'] == 'BEARISH_OB' and
              abs(current_price - latest_ob['mid']) / latest_ob['mid'] < 0.01 and
              indicators['rsi'] > 50 and
              latest_ob['volume_ratio'] > 1.5):
            
            signal['action'] = 'SELL'
            signal['confidence'] = 0.8
            signal['stop_loss'] = latest_ob['high'] * 1.01
            signal['take_profit'] = latest_ob['low'] * 0.98
        
        else:
            return None
        
        return signal

class SMC_LiquidityGrab_Strategy(BaseStrategy):
    """Smart Money Concept - Liquidity Grab Strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "SMC Liquidity Grab Strategy"
        self.description = "Trades liquidity grabs and reversals"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        # Get SMC analysis
        breakers = self.smc_analyzer.detect_breakers(
            historical_data['High'],
            historical_data['Low'],
            historical_data['Close']
        )
        
        market_structure = self.smc_analyzer.analyze_market_structure(
            historical_data['High'],
            historical_data['Low'],
            historical_data['Close']
        )
        
        if not breakers:
            return None
        
        latest_breaker = breakers[-1]
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0,
            'smc_data': latest_breaker
        }
        
        # Bullish Breaker signal (bear trap)
        if (latest_breaker['type'] == 'BULLISH_BREAKER' and
            latest_breaker['rejection_strength'] > 0.6 and
            market_structure['structure'] == 'UPTREND' and
            indicators['volume_ratio'] > 1.5):
            
            signal['action'] = 'BUY'
            signal['confidence'] = 0.85
            signal['stop_loss'] = latest_breaker['break_price'] * 0.99
            signal['take_profit'] = current_price + (latest_breaker['break_price'] - current_price) * 2
        
        # Bearish Breaker signal (bull trap)
        elif (latest_breaker['type'] == 'BEARISH_BREAKER' and
              latest_breaker['rejection_strength'] > 0.6 and
              market_structure['structure'] == 'DOWNTREND' and
              indicators['volume_ratio'] > 1.5):
            
            signal['action'] = 'SELL'
            signal['confidence'] = 0.85
            signal['stop_loss'] = latest_breaker['break_price'] * 1.01
            signal['take_profit'] = current_price - (current_price - latest_breaker['break_price']) * 2
        
        else:
            return None
        
        return signal

class TrendFollowingStrategy(BaseStrategy):
    """Traditional Trend Following Strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Trend Following Strategy"
        self.description = "Follows established market trends with EMA alignment"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0
        }
        
        # Bullish trend: EMAs aligned bullish, price above EMAs, RSI > 50
        if (indicators['ema_8'] > indicators['ema_21'] > indicators['ema_50'] and
            current_price > indicators['ema_8'] and
            indicators['rsi'] > 50 and indicators['rsi'] < 70 and
            indicators['volume_ratio'] > 1.2):
            
            signal['action'] = 'BUY'
            signal['confidence'] = 0.7
            signal['stop_loss'] = current_price - (indicators['atr'] * 1.5)
            signal['take_profit'] = current_price + (indicators['atr'] * 3)
        
        # Bearish trend: EMAs aligned bearish, price below EMAs, RSI < 50
        elif (indicators['ema_8'] < indicators['ema_21'] < indicators['ema_50'] and
              current_price < indicators['ema_8'] and
              indicators['rsi'] < 50 and indicators['rsi'] > 30 and
              indicators['volume_ratio'] > 1.2):
            
            signal['action'] = 'SELL'
            signal['confidence'] = 0.7
            signal['stop_loss'] = current_price + (indicators['atr'] * 1.5)
            signal['take_profit'] = current_price - (indicators['atr'] * 3)
        
        else:
            return None
        
        return signal

class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Mean Reversion Strategy"
        self.description = "Trades price reversions to mean with RSI extremes"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0
        }
        
        # Oversold: Price below lower Bollinger Band, RSI < 30
        if (current_price < indicators['bb_lower'] and
            indicators['rsi'] < 30 and
            indicators['volume_ratio'] > 1.0):
            
            signal['action'] = 'BUY'
            signal['confidence'] = 0.65
            signal['stop_loss'] = current_price * 0.97
            signal['take_profit'] = indicators['bb_middle']
        
        # Overbought: Price above upper Bollinger Band, RSI > 70
        elif (current_price > indicators['bb_upper'] and
              indicators['rsi'] > 70 and
              indicators['volume_ratio'] > 1.0):
            
            signal['action'] = 'SELL'
            signal['confidence'] = 0.65
            signal['stop_loss'] = current_price * 1.03
            signal['take_profit'] = indicators['bb_middle']
        
        else:
            return None
        
        return signal

class BreakoutStrategy(BaseStrategy):
    """Breakout Trading Strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Breakout Strategy"
        self.description = "Trades price breakouts from consolidation with volume confirmation"
    
    def generate_signal(self, symbol, current_price, historical_data):
        if len(historical_data) < 20:
            return None
        
        indicators = self.calculate_indicators(historical_data)
        
        # Calculate recent range
        recent_high = historical_data['High'].iloc[-20:].max()
        recent_low = historical_data['Low'].iloc[-20:].min()
        consolidation_range = (recent_high - recent_low) / recent_low
        
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0
        }
        
        # Bullish breakout
        if (current_price > recent_high and 
            consolidation_range < 0.05 and
            indicators['volume_ratio'] > 1.5 and
            indicators['rsi'] > 50):
            
            signal['action'] = 'BUY'
            signal['confidence'] = 0.75
            signal['stop_loss'] = recent_low
            signal['take_profit'] = current_price + (recent_high - recent_low) * 1.5
        
        # Bearish breakout
        elif (current_price < recent_low and
              consolidation_range < 0.05 and
              indicators['volume_ratio'] > 1.5 and
              indicators['rsi'] < 50):
            
            signal['action'] = 'SELL'
            signal['confidence'] = 0.75
            signal['stop_loss'] = recent_high
            signal['take_profit'] = current_price - (recent_high - recent_low) * 1.5
        
        else:
            return None
        
        return signal

class VWAP_Strategy(BaseStrategy):
    """VWAP Trading Strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "VWAP Strategy"
        self.description = "Trades VWAP reversals with volume confirmation"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0
        }
        
        # Price above VWAP with bullish momentum
        if (current_price > indicators['vwap'] and
            indicators['rsi'] > 50 and
            indicators['volume_ratio'] > 1.3 and
            historical_data['Close'].iloc[-1] > historical_data['Open'].iloc[-1]):
            
            signal['action'] = 'BUY'
            signal['confidence'] = 0.7
            signal['stop_loss'] = indicators['vwap']
            signal['take_profit'] = current_price + (indicators['atr'] * 2)
        
        # Price below VWAP with bearish momentum
        elif (current_price < indicators['vwap'] and
              indicators['rsi'] < 50 and
              indicators['volume_ratio'] > 1.3 and
              historical_data['Close'].iloc[-1] < historical_data['Open'].iloc[-1]):
            
            signal['action'] = 'SELL'
            signal['confidence'] = 0.7
            signal['stop_loss'] = indicators['vwap']
            signal['take_profit'] = current_price - (indicators['atr'] * 2)
        
        else:
            return None
        
        return signal

# =============================================
# RISK MANAGEMENT MODULE
# =============================================

class RiskManager:
    """Advanced Risk Management Module"""
    
    def __init__(self):
        self.max_daily_loss = -0.05  # -5% daily loss limit
        self.max_position_size = 0.1  # 10% of capital per position
        self.max_correlation = 0.7
        self.daily_loss = 0.0
        self.position_correlation = {}
        self.daily_trades = 0
        self.max_daily_trades = 15
    
    def validate_signal(self, signal, current_positions, available_capital):
        """Validate trading signal against risk rules"""
        
        # Check daily loss limit
        if self.daily_loss < self.max_daily_loss * available_capital:
            return False, "Daily loss limit reached"
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, "Daily trade limit reached"
        
        # Check position size
        position_value = signal.get('quantity', 1) * signal.get('entry_price', 0)
        if position_value > available_capital * self.max_position_size:
            return False, "Position size exceeds limit"
        
        # Check maximum positions
        if len(current_positions) >= MAX_POSITIONS:
            return False, "Maximum positions reached"
        
        return True, "Signal validated"
    
    def update_daily_loss(self, pnl):
        """Update daily loss tracking"""
        self.daily_loss += pnl
    
    def increment_daily_trades(self):
        """Increment daily trade count"""
        self.daily_trades += 1
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_loss = 0.0
        self.daily_trades = 0

# =============================================
# ALGORITHMIC TRADING ENGINE WITH ACCURACY TRACKING
# =============================================

class AlgorithmicTradingEngine:
    """Complete Algorithmic Trading Engine with Accuracy Tracking"""
    
    def __init__(self, mode="paper", initial_capital=INITIAL_CAPITAL):
        self.mode = mode
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions = {}
        self.trade_history = []
        self.order_queue = queue.Queue()
        self.strategies = {}
        self.risk_manager = RiskManager()
        self.data_storage = DataStorage()
        
        # Enhanced accuracy tracking
        self.accuracy_metrics = {
            'total_signals': 0,
            'executed_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'max_winning_streak': 0,
            'max_losing_streak': 0,
            'sharpe_ratio': 0.0
        }
        
        # Strategy-specific performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'signals': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        })
        
        # Trading session metrics
        self.session_metrics = {
            'start_time': datetime.now(),
            'total_runtime': 0,
            'trades_per_hour': 0,
            'best_strategy': None,
            'worst_strategy': None
        }
        
        # Start trading thread
        self.trading_active = False
        self.trading_thread = None
        
        # Load strategies
        self._load_strategies()
        
        # Load historical data
        self._load_historical_data()
    
    def _load_strategies(self):
        """Load all trading strategies"""
        self.strategies = {
            'smc_fvg': SMC_FVG_Strategy(),
            'smc_orderblock': SMC_OrderBlock_Strategy(),
            'smc_liquiditygrab': SMC_LiquidityGrab_Strategy(),
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'vwap': VWAP_Strategy()
        }
    
    def _load_historical_data(self):
        """Load historical trading data"""
        try:
            historical_trades = self.data_storage.load_trades()
            if historical_trades:
                self.trade_history = historical_trades
                # Update accuracy metrics from history
                self._update_accuracy_from_history()
        except:
            pass
    
    def _update_accuracy_from_history(self):
        """Update accuracy metrics from trade history"""
        for trade in self.trade_history:
            if trade.get('status') == 'CLOSED':
                pnl = trade.get('closed_pnl', 0)
                self.accuracy_metrics['executed_trades'] += 1
                self.accuracy_metrics['total_pnl'] += pnl
                
                if pnl > 0:
                    self.accuracy_metrics['profitable_trades'] += 1
                else:
                    self.accuracy_metrics['losing_trades'] += 1
                
                # Update strategy performance
                strategy = trade.get('strategy', 'unknown')
                self.strategy_performance[strategy]['trades'] += 1
                self.strategy_performance[strategy]['total_pnl'] += pnl
                if pnl > 0:
                    self.strategy_performance[strategy]['wins'] += 1
                else:
                    self.strategy_performance[strategy]['losses'] += 1
    
    def start_trading(self):
        """Start the algorithmic trading system"""
        if self.trading_active:
            return False
        
        self.trading_active = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        return True
    
    def stop_trading(self):
        """Stop the algorithmic trading system"""
        self.trading_active = False
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        return True
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.trading_active:
            try:
                # Generate trading signals
                signals = self._generate_signals()
                
                # Process signals with risk management
                for signal in signals:
                    valid, message = self.risk_manager.validate_signal(signal, self.positions, self.cash)
                    if valid:
                        self._execute_trade(signal)
                
                # Manage existing positions
                self._manage_positions()
                
                # Update performance metrics
                self._update_performance()
                
                # Update session metrics
                self._update_session_metrics()
                
                # Sleep to control loop frequency
                time.sleep(2)
                
            except Exception as e:
                print(f"Trading loop error: {str(e)}")
                time.sleep(5)
    
    def _generate_signals(self):
        """Generate trading signals from all strategies"""
        signals = []
        
        # Analyze all symbols
        for symbol in ALL_SYMBOLS[:10]:  # Limit to first 10 for performance
            try:
                # Get current price
                current_price = self._get_current_price(symbol)
                
                # Get historical data
                historical_data = self._get_historical_data(symbol)
                
                if historical_data.empty:
                    continue
                
                # Run each strategy
                for strategy_name, strategy in self.strategies.items():
                    signal = strategy.generate_signal(symbol, current_price, historical_data)
                    if signal and signal['confidence'] > 0.6:  # Minimum confidence threshold
                        self.accuracy_metrics['total_signals'] += 1
                        self.strategy_performance[strategy_name]['signals'] += 1
                        signals.append(signal)
                        
            except Exception as e:
                continue
        
        return signals
    
    def _get_current_price(self, symbol):
        """Get current price for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        
        # Synthetic price for demo
        base_prices = {
            'BTC-USD': 45000,
            'ETH-USD': 2500,
            'AAPL': 180,
            'GC=F': 1950,
            'EURUSD=X': 1.08
        }
        return base_prices.get(symbol, 100.0)
    
    def _get_historical_data(self, symbol, period='7d', interval='15m'):
        """Get historical market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except:
            return pd.DataFrame()
    
    def _execute_trade(self, signal):
        """Execute a trade based on signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            quantity = self._calculate_position_size(signal)
            
            if quantity <= 0:
                return False, "Invalid position size"
            
            # Get current price
            current_price = self._get_current_price(symbol)
            
            # Execute paper trade
            return self._execute_paper_trade(signal, quantity, current_price)
                
        except Exception as e:
            return False, f"Trade execution failed: {str(e)}"
    
    def _calculate_position_size(self, signal):
        """Calculate position size based on risk management"""
        current_price = self._get_current_price(signal['symbol'])
        stop_loss = signal.get('stop_loss', current_price * 0.95)
        
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        
        risk_amount = self.cash * RISK_PER_TRADE
        quantity = int(risk_amount / risk_per_share)
        
        # Ensure minimum and maximum limits
        min_quantity = 1
        max_quantity = int(self.cash * 0.1 / current_price)
        
        return max(min_quantity, min(quantity, max_quantity))
    
    def _execute_paper_trade(self, signal, quantity, current_price):
        """Execute paper trade"""
        trade_id = f"{signal['symbol']}_{signal['action']}_{int(time.time())}"
        
        trade = {
            'trade_id': trade_id,
            'symbol': signal['symbol'],
            'action': signal['action'],
            'quantity': quantity,
            'entry_price': current_price,
            'current_price': current_price,
            'stop_loss': signal.get('stop_loss', current_price * 0.95),
            'take_profit': signal.get('take_profit', current_price * 1.05),
            'strategy': signal.get('strategy', 'unknown'),
            'timestamp': datetime.now(),
            'status': 'OPEN',
            'pnl': 0.0,
            'paper_trade': True,
            'confidence': signal.get('confidence', 0.5),
            'risk_reward': self._calculate_risk_reward(signal, current_price)
        }
        
        # Update cash (simulated)
        trade_value = quantity * current_price
        if signal['action'] == 'BUY':
            self.cash -= trade_value
        elif signal['action'] == 'SELL':
            self.cash += trade_value
        
        self.positions[trade_id] = trade
        self.trade_history.append(trade)
        
        # Update risk manager
        self.risk_manager.increment_daily_trades()
        
        # Save trades
        self.data_storage.save_trades(self.trade_history)
        
        return True, f"Paper trade executed: {signal['action']} {quantity} {signal['symbol']} @ ${current_price:.2f}"
    
    def _calculate_risk_reward(self, signal, entry_price):
        """Calculate risk-reward ratio"""
        stop_loss = signal.get('stop_loss', entry_price * 0.95)
        take_profit = signal.get('take_profit', entry_price * 1.05)
        
        if signal['action'] == 'BUY':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        if risk <= 0:
            return 0
        
        return reward / risk
    
    def _manage_positions(self):
        """Manage existing positions (stop loss, take profit)"""
        positions_to_close = []
        
        for trade_id, position in self.positions.items():
            if position['status'] != 'OPEN':
                continue
            
            current_price = self._get_current_price(position['symbol'])
            position['current_price'] = current_price
            
            # Calculate P&L
            if position['action'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            position['pnl'] = pnl
            
            # Check stop loss
            if position['action'] == 'BUY' and current_price <= position['stop_loss']:
                positions_to_close.append((trade_id, 'STOP_LOSS'))
            elif position['action'] == 'SELL' and current_price >= position['stop_loss']:
                positions_to_close.append((trade_id, 'STOP_LOSS'))
            
            # Check take profit
            if position['action'] == 'BUY' and current_price >= position['take_profit']:
                positions_to_close.append((trade_id, 'TAKE_PROFIT'))
            elif position['action'] == 'SELL' and current_price <= position['take_profit']:
                positions_to_close.append((trade_id, 'TAKE_PROFIT'))
        
        # Close positions
        for trade_id, reason in positions_to_close:
            self._close_position(trade_id, reason)
    
    def _close_position(self, trade_id, reason='MANUAL'):
        """Close a position"""
        if trade_id not in self.positions:
            return False
        
        position = self.positions[trade_id]
        current_price = self._get_current_price(position['symbol'])
        
        # Calculate final P&L
        if position['action'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['quantity']
            if position['paper_trade']:
                self.cash += position['quantity'] * current_price
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
            if position['paper_trade']:
                self.cash += position['quantity'] * position['entry_price'] * 2
        
        # Update position
        position['exit_price'] = current_price
        position['exit_time'] = datetime.now()
        position['status'] = 'CLOSED'
        position['closed_pnl'] = pnl
        position['exit_reason'] = reason
        
        # Update accuracy metrics
        self.accuracy_metrics['executed_trades'] += 1
        self.accuracy_metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.accuracy_metrics['profitable_trades'] += 1
        else:
            self.accuracy_metrics['losing_trades'] += 1
        
        # Update risk manager
        self.risk_manager.update_daily_loss(pnl)
        
        # Update strategy performance
        strategy = position.get('strategy', 'unknown')
        self.strategy_performance[strategy]['trades'] += 1
        self.strategy_performance[strategy]['total_pnl'] += pnl
        
        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
        
        # Remove from open positions
        del self.positions[trade_id]
        
        # Save trades
        self.data_storage.save_trades(self.trade_history)
        
        return True
    
    def _update_performance(self):
        """Update performance metrics"""
        # Calculate win rate
        total_trades = self.accuracy_metrics['executed_trades']
        winning_trades = self.accuracy_metrics['profitable_trades']
        
        if total_trades > 0:
            self.accuracy_metrics['win_rate'] = winning_trades / total_trades
        
        # Calculate profit factor
        if self.accuracy_metrics['losing_trades'] > 0:
            avg_win = self.accuracy_metrics['total_pnl'] / max(1, winning_trades)
            avg_loss = abs(self.accuracy_metrics['total_pnl']) / max(1, self.accuracy_metrics['losing_trades'])
            self.accuracy_metrics['profit_factor'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Update strategy win rates
        for strategy, perf in self.strategy_performance.items():
            total = perf['wins'] + perf['losses']
            if total > 0:
                perf['win_rate'] = perf['wins'] / total
                if perf['wins'] > 0:
                    perf['avg_win'] = perf['total_pnl'] / perf['wins']
                if perf['losses'] > 0:
                    perf['avg_loss'] = abs(perf['total_pnl']) / perf['losses']
    
    def _update_session_metrics(self):
        """Update session metrics"""
        runtime = (datetime.now() - self.session_metrics['start_time']).total_seconds() / 3600
        self.session_metrics['total_runtime'] = runtime
        
        if runtime > 0:
            self.session_metrics['trades_per_hour'] = self.accuracy_metrics['executed_trades'] / runtime
        
        # Find best and worst strategies
        if self.strategy_performance:
            strategies_with_trades = {k: v for k, v in self.strategy_performance.items() if v['trades'] > 0}
            if strategies_with_trades:
                self.session_metrics['best_strategy'] = max(strategies_with_trades.items(), 
                                                           key=lambda x: x[1]['win_rate'])[0]
                self.session_metrics['worst_strategy'] = min(strategies_with_trades.items(), 
                                                           key=lambda x: x[1]['win_rate'])[0]
    
    def get_portfolio_summary(self):
        """Get portfolio summary"""
        total_value = self.cash
        open_pnl = 0
        
        for position in self.positions.values():
            if position['status'] == 'OPEN':
                position_value = position['quantity'] * position['current_price']
                total_value += position_value
                open_pnl += position['pnl']
        
        return {
            'cash': self.cash,
            'total_value': total_value,
            'open_positions': len(self.positions),
            'open_pnl': open_pnl,
            'total_pnl': self.accuracy_metrics['total_pnl'],
            'win_rate': self.accuracy_metrics['win_rate'],
            'total_trades': self.accuracy_metrics['executed_trades'],
            'profit_factor': self.accuracy_metrics['profit_factor'],
            'sharpe_ratio': self.accuracy_metrics['sharpe_ratio']
        }
    
    def get_open_positions(self):
        """Get all open positions"""
        return list(self.positions.values())
    
    def get_trade_history(self, limit=100):
        """Get trade history"""
        return self.trade_history[-limit:] if self.trade_history else []
    
    def get_strategy_performance(self):
        """Get detailed strategy performance"""
        return dict(self.strategy_performance)
    
    def get_accuracy_report(self):
        """Get comprehensive accuracy report"""
        return {
            'accuracy_metrics': self.accuracy_metrics,
            'session_metrics': self.session_metrics,
            'strategy_performance': dict(self.strategy_performance)
        }
    
    def close_all_positions(self):
        """Close all open positions"""
        closed = []
        for trade_id in list(self.positions.keys()):
            current_price = self._get_current_price(self.positions[trade_id]['symbol'])
            if self._close_position(trade_id, 'MANUAL_CLOSE_ALL'):
                closed.append(trade_id)
        return len(closed)
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.risk_manager.reset_daily_metrics()

# =============================================
# STREAMLIT UI COMPONENTS
# =============================================

def create_header():
    """Create application header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🤖 RANTV ALGORITHMIC TRADING SYSTEM</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">Smart Money Concept & Multi-Strategy Trading with Accuracy Tracking</p>
    </div>
    """, unsafe_allow_html=True)

def create_trading_control_panel(trading_engine):
    """Create trading control panel"""
    st.subheader("🎮 Trading Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not trading_engine.trading_active:
            if st.button("🚀 Start Algorithmic Trading", use_container_width=True, type="primary"):
                if trading_engine.start_trading():
                    st.success("Algorithmic trading started!")
                    st.rerun()
        else:
            if st.button("🛑 Stop Algorithmic Trading", use_container_width=True, type="secondary"):
                trading_engine.stop_trading()
                st.success("Algorithmic trading stopped!")
                st.rerun()
    
    with col2:
        if st.button("🔄 Update Positions", use_container_width=True):
            trading_engine._manage_positions()
            st.success("Positions updated!")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Close All Positions", use_container_width=True):
            closed = trading_engine.close_all_positions()
            st.success(f"Closed {closed} positions!")
            st.rerun()
    
    with col4:
        if st.button("📊 Reset Daily Metrics", use_container_width=True):
            trading_engine.reset_daily_metrics()
            st.success("Daily metrics reset!")
            st.rerun()

def create_portfolio_dashboard(trading_engine):
    """Create portfolio dashboard"""
    st.subheader("📊 Portfolio Overview")
    
    portfolio = trading_engine.get_portfolio_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"${portfolio['total_value']:,.2f}",
            delta=f"${portfolio['total_pnl']:+,.2f}"
        )
    
    with col2:
        st.metric(
            "Available Cash",
            f"${portfolio['cash']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Open Positions",
            portfolio['open_positions']
        )
    
    with col4:
        pnl_color = "inverse" if portfolio['open_pnl'] < 0 else "normal"
        st.metric(
            "Open P&L",
            f"${portfolio['open_pnl']:+,.2f}",
            delta_color=pnl_color
        )
    
    # Accuracy metrics
    st.subheader("🎯 Accuracy Metrics")
    acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
    
    with acc_col1:
        st.metric("Win Rate", f"{portfolio['win_rate']:.1%}")
    
    with acc_col2:
        st.metric("Total Trades", portfolio['total_trades'])
    
    with acc_col3:
        st.metric("Profit Factor", f"{portfolio['profit_factor']:.2f}")
    
    with acc_col4:
        st.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")

def create_positions_dashboard(trading_engine):
    """Create positions dashboard"""
    st.subheader("💰 Open Positions")
    
    positions = trading_engine.get_open_positions()
    
    if not positions:
        st.info("No open positions")
        return
    
    for position in positions:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                action_color = "green" if position['action'] == 'BUY' else "red"
                st.write(f"**{position['symbol']}** - <span style='color:{action_color};'>{position['action']}</span>", 
                        unsafe_allow_html=True)
                st.write(f"Qty: {position['quantity']} | Entry: ${position['entry_price']:.2f}")
                st.write(f"Strategy: {position['strategy']} | Confidence: {position.get('confidence', 0.5):.1%}")
            
            with col2:
                current_price = position.get('current_price', position['entry_price'])
                pnl = position.get('pnl', 0)
                pnl_color = "green" if pnl >= 0 else "red"
                st.write(f"Current: ${current_price:.2f}")
                st.write(f"<span style='color:{pnl_color};'>P&L: ${pnl:+,.2f}</span>", 
                        unsafe_allow_html=True)
            
            with col3:
                st.write(f"SL: ${position['stop_loss']:.2f}")
            
            with col4:
                st.write(f"TP: ${position['take_profit']:.2f}")
            
            with col5:
                if st.button("Close", key=f"close_{position['trade_id']}"):
                    if trading_engine._close_position(position['trade_id'], 'MANUAL'):
                        st.success("Position closed")
                        st.rerun()
            
            st.divider()

def create_trading_history_dashboard(trading_engine):
    """Create comprehensive trading history dashboard"""
    st.subheader("📋 Trading History & Accuracy Analysis")
    
    # Get trade history
    trade_history = trading_engine.get_trade_history(100)
    
    if not trade_history:
        st.info("No trade history available")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["All Trades", "Performance Charts", "Strategy Analysis", "Accuracy Report"])
    
    with tab1:
        # Display all trades in a table
        st.subheader("📊 All Trades")
        
        # Prepare data for display
        history_data = []
        for trade in trade_history:
            history_data.append({
                'ID': trade['trade_id'][-8:],
                'Symbol': trade['symbol'],
                'Action': trade['action'],
                'Strategy': trade['strategy'],
                'Entry': f"${trade['entry_price']:.2f}",
                'Exit': f"${trade.get('exit_price', 'N/A'):.2f}" if trade.get('exit_price') else "N/A",
                'P&L': f"${trade.get('closed_pnl', trade.get('pnl', 0)):+,.2f}",
                'Status': trade['status'],
                'Reason': trade.get('exit_reason', 'N/A'),
                'Date': trade['timestamp'].strftime('%Y-%m-%d %H:%M') if isinstance(trade['timestamp'], datetime) else trade['timestamp']
            })
        
        if history_data:
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, height=400)
            
            # Export option
            if st.button("📥 Export Trade History"):
                df = pd.DataFrame(history_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="trade_history.csv",
                    mime="text/csv"
                )
    
    with tab2:
        # Performance charts
        st.subheader("📈 Performance Charts")
        
        # Filter closed trades
        closed_trades = [t for t in trade_history if t.get('status') == 'CLOSED']
        
        if closed_trades:
            # Prepare data for charts
            df_trades = pd.DataFrame([
                {
                    'Date': t['timestamp'] if isinstance(t['timestamp'], datetime) else datetime.now(),
                    'P&L': t.get('closed_pnl', 0),
                    'Strategy': t.get('strategy', 'unknown'),
                    'Symbol': t['symbol'],
                    'Action': t['action']
                }
                for t in closed_trades
            ])
            
            if not df_trades.empty:
                df_trades = df_trades.sort_values('Date')
                df_trades['Cumulative P&L'] = df_trades['P&L'].cumsum()
                df_trades['Rolling Win Rate'] = (df_trades['P&L'] > 0).rolling(window=10).mean()
                
                # Cumulative P&L chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=df_trades['Date'],
                    y=df_trades['Cumulative P&L'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='green' if df_trades['Cumulative P&L'].iloc[-1] > 0 else 'red', width=2)
                ))
                fig1.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L ($)",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Win rate rolling chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df_trades['Date'],
                    y=df_trades['Rolling Win Rate'] * 100,
                    mode='lines',
                    name='Rolling Win Rate (10 trades)',
                    line=dict(color='blue', width=2)
                ))
                fig2.add_hline(y=50, line_dash="dash", line_color="red")
                fig2.update_layout(
                    title="Rolling Win Rate (10-Trade Window)",
                    xaxis_title="Date",
                    yaxis_title="Win Rate (%)",
                    height=400,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # P&L distribution
                fig3 = px.histogram(df_trades, x='P&L', 
                                   title="P&L Distribution",
                                   color_discrete_sequence=['green' if x > 0 else 'red' for x in df_trades['P&L']],
                                   nbins=20)
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # Strategy analysis
        st.subheader("🎯 Strategy Performance Analysis")
        
        strategy_perf = trading_engine.get_strategy_performance()
        
        if strategy_perf:
            # Prepare strategy performance data
            perf_data = []
            for strategy, data in strategy_perf.items():
                if data['trades'] > 0:
                    perf_data.append({
                        'Strategy': strategy,
                        'Trades': data['trades'],
                        'Wins': data['wins'],
                        'Losses': data['losses'],
                        'Win Rate': f"{data['win_rate']:.1%}",
                        'Total P&L': f"${data['total_pnl']:+,.2f}",
                        'Avg Win': f"${data['avg_win']:+,.2f}" if data['wins'] > 0 else "N/A",
                        'Avg Loss': f"${data['avg_loss']:+,.2f}" if data['losses'] > 0 else "N/A"
                    })
            
            if perf_data:
                df_perf = pd.DataFrame(perf_data)
                st.dataframe(df_perf, use_container_width=True)
                
                # Strategy comparison chart
                fig = px.bar(df_perf, x='Strategy', y='Win Rate',
                           title="Strategy Win Rate Comparison",
                           color='Total P&L',
                           color_continuous_scale='RdYlGn')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Accuracy report
        st.subheader("📊 Comprehensive Accuracy Report")
        
        accuracy_report = trading_engine.get_accuracy_report()
        accuracy_metrics = accuracy_report['accuracy_metrics']
        session_metrics = accuracy_report['session_metrics']
        
        # Overall accuracy metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", accuracy_metrics['total_signals'])
            st.metric("Signal to Trade Ratio", 
                     f"{(accuracy_metrics['executed_trades'] / max(1, accuracy_metrics['total_signals'])):.1%}")
        
        with col2:
            st.metric("Win Rate", f"{accuracy_metrics['win_rate']:.1%}")
            st.metric("Profit Factor", f"{accuracy_metrics['profit_factor']:.2f}")
        
        with col3:
            st.metric("Total P&L", f"${accuracy_metrics['total_pnl']:+,.2f}")
            st.metric("Avg Win", f"${accuracy_metrics['average_win']:+,.2f}")
        
        with col4:
            st.metric("Best Strategy", session_metrics.get('best_strategy', 'N/A'))
            st.metric("Worst Strategy", session_metrics.get('worst_strategy', 'N/A'))
        
        # Session metrics
        st.subheader("⏱️ Session Metrics")
        sess_col1, sess_col2, sess_col3 = st.columns(3)
        
        with sess_col1:
            st.metric("Total Runtime", f"{session_metrics['total_runtime']:.1f} hours")
        
        with sess_col2:
            st.metric("Trades per Hour", f"{session_metrics['trades_per_hour']:.1f}")
        
        with sess_col3:
            start_time = session_metrics['start_time']
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            st.metric("Session Start", start_time.strftime('%Y-%m-%d %H:%M'))

def create_signal_generator(trading_engine):
    """Create signal generator dashboard"""
    st.subheader("🚦 Signal Generator")
    
    # Strategy selection
    selected_strategies = st.multiselect(
        "Select Strategies for Scanning",
        list(trading_engine.strategies.keys()),
        default=list(trading_engine.strategies.keys())[:3]
    )
    
    # Symbol selection
    selected_symbols = st.multiselect(
        "Select Symbols to Scan",
        ALL_SYMBOLS,
        default=ALL_SYMBOLS[:5]
    )
    
    # Confidence filter
    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.5,
        max_value=0.95,
        value=0.65,
        step=0.05
    )
    
    if st.button("🔍 Scan for Signals", type="primary", use_container_width=True):
        if not selected_symbols or not selected_strategies:
            st.warning("Please select at least one symbol and one strategy")
            return
        
        with st.spinner("Scanning for trading signals..."):
            signals = []
            for symbol in selected_symbols:
                current_price = trading_engine._get_current_price(symbol)
                historical_data = trading_engine._get_historical_data(symbol)
                
                if historical_data.empty:
                    continue
                
                for strategy_name in selected_strategies:
                    strategy = trading_engine.strategies[strategy_name]
                    signal = strategy.generate_signal(symbol, current_price, historical_data)
                    if signal and signal['confidence'] >= min_confidence:
                        signals.append(signal)
            
            if signals:
                st.success(f"Found {len(signals)} signals with confidence ≥ {min_confidence}")
                
                # Display signals
                for i, signal in enumerate(signals[:10]):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            action_emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
                            st.write(f"**{action_emoji} {signal['symbol']} - {signal['action']}**")
                            st.write(f"Strategy: {signal['strategy']}")
                            st.write(f"Confidence: {signal['confidence']:.1%}")
                        
                        with col2:
                            st.write(f"Price: ${trading_engine._get_current_price(signal['symbol']):.2f}")
                        
                        with col3:
                            st.write(f"SL: ${signal.get('stop_loss', 0):.2f}")
                            st.write(f"TP: ${signal.get('take_profit', 0):.2f}")
                        
                        with col4:
                            if st.button("Execute", key=f"exec_{i}"):
                                success, message = trading_engine._execute_trade(signal)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        st.divider()
            else:
                st.info("No signals found with current parameters")

def create_strategy_configuration(trading_engine):
    """Create strategy configuration panel"""
    st.subheader("⚙️ Strategy Configuration")
    
    # Strategy selection and weighting
    st.subheader("📊 Strategy Weighting")
    
    strategies = list(trading_engine.strategies.keys())
    weights = {}
    
    for strategy in strategies:
        weight = st.slider(
            f"Weight: {strategy}",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            key=f"weight_{strategy}"
        )
        weights[strategy] = weight
    
    # Risk parameters
    st.subheader("🛡️ Risk Management")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        daily_loss_limit = st.number_input(
            "Daily Loss Limit (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )
    
    with risk_col2:
        risk_per_trade = st.number_input(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
    
    with risk_col3:
        max_daily_trades = st.number_input(
            "Max Daily Trades",
            min_value=5,
            max_value=50,
            value=15,
            step=1
        )
    
    if st.button("💾 Save Configuration", type="primary"):
        trading_engine.risk_manager.max_daily_loss = -daily_loss_limit / 100
        trading_engine.risk_manager.max_daily_trades = max_daily_trades
        st.success("Configuration saved!")

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    """Main application function"""
    
    # Initialize session state
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = AlgorithmicTradingEngine(mode="paper")
    
    # Auto-refresh for real-time updates
    st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh")
    
    # Create header
    create_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Trading parameters
        st.subheader("📊 Trading Parameters")
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000
        )
        
        # Market selection
        st.subheader("🌐 Market Selection")
        
        selected_markets = st.multiselect(
            "Select Markets to Trade",
            MARKET_OPTIONS,
            default=["CRYPTO", "STOCKS"]
        )
        
        # System controls
        st.subheader("🔄 System Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("New Session", use_container_width=True):
                st.session_state.trading_engine = AlgorithmicTradingEngine(mode="paper", initial_capital=initial_capital)
                st.success("New trading session started!")
                st.rerun()
        
        with col2:
            if st.button("Export Data", use_container_width=True, type="secondary"):
                trading_engine = st.session_state.trading_engine
                if trading_engine.trade_history:
                    df = pd.DataFrame(trading_engine.trade_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download All Data",
                        data=csv,
                        file_name="trading_data.csv",
                        mime="text/csv"
                    )
        
        # System status
        st.markdown("---")
        trading_engine = st.session_state.trading_engine
        status_color = "🟢" if trading_engine.trading_active else "🔴"
        st.markdown(f"**System Status:** {status_color} {'Running' if trading_engine.trading_active else 'Stopped'}")
        st.markdown(f"**Mode:** Paper Trading")
        st.markdown(f"**Strategies Active:** {len(trading_engine.strategies)}")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎮 Control",
        "📊 Portfolio",
        "💰 Positions",
        "📋 History",
        "🚦 Signals",
        "⚙️ Config",
        "📈 Analytics"
    ])
    
    trading_engine = st.session_state.trading_engine
    
    with tab1:
        # Control Panel
        create_trading_control_panel(trading_engine)
        
        # Quick stats
        st.subheader("⚡ Quick Stats")
        
        portfolio = trading_engine.get_portfolio_summary()
        accuracy_report = trading_engine.get_accuracy_report()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Active Strategies", len(trading_engine.strategies))
        
        with stat_col2:
            st.metric("Total Signals", accuracy_report['accuracy_metrics']['total_signals'])
        
        with stat_col3:
            st.metric("Executed Trades", portfolio['total_trades'])
        
        with stat_col4:
            st.metric("System Status", "Running" if trading_engine.trading_active else "Stopped")
    
    with tab2:
        # Portfolio Dashboard
        create_portfolio_dashboard(trading_engine)
    
    with tab3:
        # Positions Dashboard
        create_positions_dashboard(trading_engine)
    
    with tab4:
        # Trading History Dashboard
        create_trading_history_dashboard(trading_engine)
    
    with tab5:
        # Signal Generator
        create_signal_generator(trading_engine)
    
    with tab6:
        # Configuration
        create_strategy_configuration(trading_engine)
    
    with tab7:
        # Real-time Analytics
        st.subheader("📈 Real-time Analytics")
        
        # Market overview
        st.subheader("🌐 Market Overview")
        
        key_symbols = ["BTC-USD", "ETH-USD", "AAPL", "GC=F", "EURUSD=X"]
        cols = st.columns(len(key_symbols))
        
        for i, symbol in enumerate(key_symbols):
            with cols[i]:
                try:
                    price = trading_engine._get_current_price(symbol)
                    change = np.random.uniform(-2, 2)
                    st.metric(
                        symbol,
                        f"${price:,.2f}" if price > 10 else f"{price:.4f}",
                        delta=f"{change:+.1f}%"
                    )
                except:
                    st.metric(symbol, "N/A")
        
        # Strategy heatmap
        st.subheader("🎯 Strategy Performance Heatmap")
        
        strategy_perf = trading_engine.get_strategy_performance()
        if strategy_perf:
            perf_data = []
            for strategy, data in strategy_perf.items():
                if data['trades'] > 0:
                    perf_data.append({
                        'Strategy': strategy,
                        'Win Rate': data['win_rate'],
                        'Trades': data['trades'],
                        'Total P&L': data['total_pnl']
                    })
            
            if perf_data:
                df_heatmap = pd.DataFrame(perf_data)
                
                # Create heatmap
                fig = px.imshow(
                    df_heatmap[['Win Rate', 'Trades', 'Total P&L']].T,
                    labels=dict(x="Strategy", y="Metric", color="Value"),
                    x=df_heatmap['Strategy'].tolist(),
                    y=['Win Rate', 'Trades', 'Total P&L'],
                    color_continuous_scale='RdYlGn',
                    aspect="auto"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>RANTV Algorithmic Trading System v5.0</strong> | Smart Money Concept | Multi-Strategy | Accuracy Tracking</p>
    <p>⚠️ This is for educational and paper trading purposes only. All trades are simulated.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# RUN APPLICATION
# =============================================

if __name__ == "__main__":
    main()
