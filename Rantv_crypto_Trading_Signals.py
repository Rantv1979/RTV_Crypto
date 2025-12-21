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

# Trading Parameters - REALISTIC $1000 CAPITAL
INITIAL_CAPITAL = 1000.0
TRADE_ALLOCATION = 0.15
MAX_DAILY_TRADES = 15
MAX_POSITIONS = 10
RISK_PER_TRADE = 0.02  # 2% risk per trade ($20 on $1000)

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
        indicators['macd'] = indicators['ema_8']
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
        
        # Check for duplicate trades (prevent duplicate positions on same symbol)
        for pos in current_positions.values():
            if pos['symbol'] == signal.get('symbol'):
                return False, "Duplicate position on same symbol"
        
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
# ALGORITHMIC TRADING ENGINE WITH SESSION-BASED HISTORY
# =============================================

class AlgorithmicTradingEngine:
    """Complete Algorithmic Trading Engine with Session-Based History"""
    
    def __init__(self, mode="paper", initial_capital=INITIAL_CAPITAL):
        self.mode = mode
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions = {}
        self.trade_history = []  # Session-based trade history (reset on new session)
        self.order_queue = queue.Queue()
        self.strategies = {}
        self.risk_manager = RiskManager()
        self.data_storage = DataStorage()
        
        # Session-based accuracy tracking
        self.accuracy_metrics = {
            'total_signals': 0,
            'executed_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_wins': 0.0,
            'total_losses': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'max_winning_streak': 0,
            'max_losing_streak': 0,
            'sharpe_ratio': 0.0
        }
        
        # Session-based strategy performance tracking
        self.strategy_performance = defaultdict(lambda: {
            'signals': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_wins': 0.0,
            'total_losses': 0.0,
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
        
        # Track recent trades to prevent duplicates
        self.recent_trades = []
        self.max_recent_trades = 10
        
        # Start trading thread
        self.trading_active = False
        self.trading_thread = None
        
        # Load strategies
        self._load_strategies()
        
        # NOTE: Do NOT load historical data from file
        # We want fresh session-based data only
    
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
        
        # Synthetic price for demo - REALISTIC VALUES
        base_prices = {
            'BTC-USD': 45000,
            'ETH-USD': 2500,
            'SOL-USD': 100,
            'XRP-USD': 0.5,
            'BNB-USD': 300,
            'ADA-USD': 0.5,
            'AVAX-USD': 40,
            'DOT-USD': 7,
            'DOGE-USD': 0.15,
            'LINK-USD': 15,
            'AAPL': 180,
            'MSFT': 320,
            'GOOGL': 140,
            'AMZN': 150,
            'TSLA': 180,
            'NVDA': 500,
            'META': 350,
            'JPM': 180,
            'V': 250,
            'WMT': 160,
            'GC=F': 1950,
            'SI=F': 24,
            'CL=F': 75,
            'NG=F': 3,
            'ZC=F': 450,
            'EURUSD=X': 1.08,
            'GBPUSD=X': 1.26,
            'USDJPY=X': 148,
            'USDCHF=X': 0.88,
            'AUDUSD=X': 0.66
        }
        return base_prices.get(symbol, 100.0)
    
    def _get_historical_data(self, symbol, period='7d', interval='15m'):
        """Get historical market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except:
            # Return synthetic data for demo
            dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
            prices = np.random.normal(loc=100, scale=5, size=100).cumsum() + 100
            return pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, size=100)
            }, index=dates)
    
    def _execute_trade(self, signal):
        """Execute a trade based on signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            
            # Check for duplicate trades (recent trades)
            recent_trades_on_symbol = [t for t in self.recent_trades if t['symbol'] == symbol]
            if len(recent_trades_on_symbol) > 0:
                return False, "Duplicate trade on same symbol recently"
            
            # Get current price
            current_price = self._get_current_price(symbol)
            
            # Calculate position size - REALISTIC FOR $1000 CAPITAL
            quantity, position_value = self._calculate_realistic_position_size(signal, current_price)
            
            if quantity <= 0:
                return False, "Invalid position size"
            
            # Execute paper trade
            success, message = self._execute_paper_trade(signal, quantity, current_price, position_value)
            
            if success:
                # Add to recent trades
                self.recent_trades.append({
                    'symbol': symbol,
                    'time': datetime.now(),
                    'action': action
                })
                # Keep only recent trades
                if len(self.recent_trades) > self.max_recent_trades:
                    self.recent_trades.pop(0)
            
            return success, message
                
        except Exception as e:
            return False, f"Trade execution failed: {str(e)}"
    
    def _calculate_realistic_position_size(self, signal, current_price):
        """Calculate REALISTIC position size for $1000 capital"""
        # For $1000 capital with 2% risk per trade = $20 risk
        max_risk_amount = 20.0
        
        # Get stop loss from signal or use default 5% stop
        stop_loss = signal.get('stop_loss', current_price * 0.95)
        
        # Calculate risk per unit
        if signal['action'] == 'BUY':
            risk_per_unit = current_price - stop_loss
        else:
            risk_per_unit = stop_loss - current_price
        
        # Ensure minimum risk per unit (avoid division by zero)
        if risk_per_unit <= 0:
            risk_per_unit = current_price * 0.05  # Default 5% risk
        
        # Calculate quantity based on risk
        quantity = max_risk_amount / risk_per_unit
        
        # For expensive assets like BTC, we need fractional shares
        # Round to appropriate decimal places
        if 'USD' in signal['symbol'] or '-' in signal['symbol']:  # Crypto
            quantity = round(quantity, 4)
        elif any(x in signal['symbol'] for x in ['=X', '=F']):  # Forex/Commodities
            quantity = round(quantity, 2)
        else:  # Stocks
            quantity = round(quantity, 2)
        
        # Minimum quantity check
        min_quantity = 0.001 if 'USD' in signal['symbol'] or '-' in signal['symbol'] else 0.01
        if quantity < min_quantity:
            quantity = min_quantity
        
        # Calculate position value
        position_value = quantity * current_price
        
        # Ensure we don't exceed available cash
        if position_value > self.cash * 0.9:  # Max 90% of cash
            # Recalculate based on available cash
            quantity = (self.cash * 0.9) / current_price
            if 'USD' in signal['symbol'] or '-' in signal['symbol']:
                quantity = round(quantity, 4)
            else:
                quantity = round(quantity, 2)
            position_value = quantity * current_price
        
        return quantity, position_value
    
    def _execute_paper_trade(self, signal, quantity, current_price, position_value):
        """Execute paper trade with REALISTIC values"""
        trade_id = f"{signal['symbol']}_{signal['action']}_{int(time.time())}"
        
        # Ensure we have enough cash
        if signal['action'] == 'BUY' and position_value > self.cash:
            return False, f"Insufficient cash: ${self.cash:.2f} available, need ${position_value:.2f}"
        
        # Create realistic position value based on $1000 capital
        realistic_position_value = min(position_value, self.cash * 0.9)  # Max 90% of cash
        
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
            'pnl_percentage': 0.0,
            'position_value': realistic_position_value,
            'paper_trade': True,
            'confidence': signal.get('confidence', 0.5),
            'risk_reward': self._calculate_risk_reward(signal, current_price)
        }
        
        # Update cash (simulated) - REALISTIC
        if signal['action'] == 'BUY':
            self.cash -= realistic_position_value
        elif signal['action'] == 'SELL':
            self.cash += realistic_position_value
        
        self.positions[trade_id] = trade
        self.trade_history.append(trade)  # Add to session history
        
        # Update risk manager
        self.risk_manager.increment_daily_trades()
        
        return True, f"Trade executed: {signal['action']} {quantity:.4f} {signal['symbol']} @ ${current_price:.2f} (Value: ${realistic_position_value:.2f})"
    
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
            
            # Calculate P&L - REALISTIC CALCULATION
            if position['action'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            position['pnl'] = pnl
            
            # Calculate percentage P&L
            if position['position_value'] > 0:
                position['pnl_percentage'] = (pnl / position['position_value']) * 100
            
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
        """Close a position - REALISTIC P&L CALCULATION"""
        if trade_id not in self.positions:
            return False
        
        position = self.positions[trade_id]
        current_price = self._get_current_price(position['symbol'])
        
        # Calculate final P&L - REALISTIC CALCULATION
        if position['action'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['quantity']
            # Return cash from position sale
            if position['paper_trade']:
                self.cash += position['quantity'] * current_price
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
            # For short selling, we need to return the borrowed shares
            if position['paper_trade']:
                # In paper trading, we just add the profit/loss to cash
                self.cash += position['position_value'] + pnl
        
        # Calculate percentage P&L
        pnl_percentage = (pnl / position['position_value']) * 100 if position['position_value'] > 0 else 0
        
        # Update position
        position['exit_price'] = current_price
        position['exit_time'] = datetime.now()
        position['status'] = 'CLOSED'
        position['closed_pnl'] = pnl
        position['closed_pnl_percentage'] = pnl_percentage
        position['exit_reason'] = reason
        
        # Update accuracy metrics - SESSION BASED ONLY
        self.accuracy_metrics['executed_trades'] += 1
        self.accuracy_metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.accuracy_metrics['profitable_trades'] += 1
            self.accuracy_metrics['total_wins'] += pnl
        else:
            self.accuracy_metrics['losing_trades'] += 1
            self.accuracy_metrics['total_losses'] += abs(pnl)
        
        # Update risk manager
        self.risk_manager.update_daily_loss(pnl)
        
        # Update strategy performance - SESSION BASED ONLY
        strategy = position.get('strategy', 'unknown')
        self.strategy_performance[strategy]['trades'] += 1
        self.strategy_performance[strategy]['total_pnl'] += pnl
        
        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
            self.strategy_performance[strategy]['total_wins'] += pnl
        else:
            self.strategy_performance[strategy]['losses'] += 1
            self.strategy_performance[strategy]['total_losses'] += abs(pnl)
        
        # Remove from open positions
        del self.positions[trade_id]
        
        return True
    
    def _update_performance(self):
        """Update performance metrics - SESSION BASED ONLY"""
        # Calculate win rate
        total_trades = self.accuracy_metrics['executed_trades']
        winning_trades = self.accuracy_metrics['profitable_trades']
        
        if total_trades > 0:
            self.accuracy_metrics['win_rate'] = winning_trades / total_trades
        
        # Calculate profit factor
        if self.accuracy_metrics['total_losses'] > 0:
            self.accuracy_metrics['profit_factor'] = self.accuracy_metrics['total_wins'] / self.accuracy_metrics['total_losses']
        elif self.accuracy_metrics['total_wins'] > 0:
            self.accuracy_metrics['profit_factor'] = float('inf')
        
        # Calculate average win/loss
        if winning_trades > 0:
            self.accuracy_metrics['average_win'] = self.accuracy_metrics['total_wins'] / winning_trades
        
        losing_trades = self.accuracy_metrics['losing_trades']
        if losing_trades > 0:
            self.accuracy_metrics['average_loss'] = self.accuracy_metrics['total_losses'] / losing_trades
        
        # Update strategy win rates
        for strategy, perf in self.strategy_performance.items():
            total = perf['wins'] + perf['losses']
            if total > 0:
                perf['win_rate'] = perf['wins'] / total
                if perf['wins'] > 0:
                    perf['avg_win'] = perf['total_wins'] / perf['wins']
                if perf['losses'] > 0:
                    perf['avg_loss'] = perf['total_losses'] / perf['losses']
    
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
        """Get portfolio summary - SESSION BASED"""
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
            'profitable_trades': self.accuracy_metrics['profitable_trades'],
            'losing_trades': self.accuracy_metrics['losing_trades'],
            'total_wins': self.accuracy_metrics['total_wins'],
            'total_losses': self.accuracy_metrics['total_losses'],
            'profit_factor': self.accuracy_metrics['profit_factor'],
            'sharpe_ratio': self.accuracy_metrics['sharpe_ratio']
        }
    
    def get_open_positions(self):
        """Get all open positions"""
        return list(self.positions.values())
    
    def get_trade_history(self, limit=100):
        """Get trade history - SESSION BASED ONLY"""
        return self.trade_history[-limit:] if self.trade_history else []
    
    def get_strategy_performance(self):
        """Get detailed strategy performance - SESSION BASED"""
        return dict(self.strategy_performance)
    
    def get_accuracy_report(self):
        """Get comprehensive accuracy report - SESSION BASED"""
        return {
            'accuracy_metrics': self.accuracy_metrics,
            'session_metrics': self.session_metrics,
            'strategy_performance': dict(self.strategy_performance)
        }
    
    def close_all_positions(self):
        """Close all open positions"""
        closed = []
        for trade_id in list(self.positions.keys()):
            if self._close_position(trade_id, 'MANUAL_CLOSE_ALL'):
                closed.append(trade_id)
        return len(closed)
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.risk_manager.reset_daily_metrics()
        self.recent_trades = []

# =============================================
# STREAMLIT UI COMPONENTS
# =============================================

def create_header():
    """Create application header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🤖 RANTV ALGORITHMIC TRADING SYSTEM</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">Smart Money Concept & Multi-Strategy Trading</p>
        <p style="color: rgba(255,255,255,0.7); margin: 5px 0 0 0; font-size: 0.9em;">💰 <strong>Session-Based Trading | $1,000 Capital | Realistic P&L</strong></p>
    </div>
    """, unsafe_allow_html=True)

def create_trading_control_panel(trading_engine):
    """Create trading control panel"""
    st.subheader("🎮 Trading Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not trading_engine.trading_active:
            if st.button("🚀 Start Trading", use_container_width=True, type="primary"):
                if trading_engine.start_trading():
                    st.success("Algorithmic trading started!")
                    st.rerun()
        else:
            if st.button("🛑 Stop Trading", use_container_width=True, type="secondary"):
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
        if st.button("📊 Reset Session", use_container_width=True):
            # Reset session metrics but keep engine running
            trading_engine.reset_daily_metrics()
            st.success("Session metrics reset!")
            st.rerun()

def create_portfolio_dashboard(trading_engine):
    """Create portfolio dashboard"""
    st.subheader("📊 Portfolio Overview")
    
    portfolio = trading_engine.get_portfolio_summary()
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pnl_color = "normal" if portfolio['total_pnl'] >= 0 else "inverse"
        st.metric(
            "Total Value",
            f"${portfolio['total_value']:,.2f}",
            delta=f"${portfolio['total_pnl']:+,.2f}",
            delta_color=pnl_color
        )
    
    with col2:
        st.metric("Available Cash", f"${portfolio['cash']:,.2f}")
    
    with col3:
        st.metric("Open Positions", portfolio['open_positions'])
    
    with col4:
        open_pnl_color = "normal" if portfolio['open_pnl'] >= 0 else "inverse"
        st.metric(
            "Open P&L",
            f"${portfolio['open_pnl']:+,.2f}",
            delta_color=open_pnl_color
        )
    
    # P&L Dashboard
    st.subheader("💰 Profit & Loss Dashboard (Session)")
    
    pl_col1, pl_col2, pl_col3, pl_col4, pl_col5, pl_col6 = st.columns(6)
    
    with pl_col1:
        st.metric("Total Trades", portfolio['total_trades'])
    
    with pl_col2:
        st.metric("Winning Trades", portfolio['profitable_trades'])
    
    with pl_col3:
        st.metric("Losing Trades", portfolio['losing_trades'])
    
    with pl_col4:
        win_rate_color = "normal" if portfolio['win_rate'] >= 0.5 else "inverse"
        st.metric(
            "Win Rate",
            f"{portfolio['win_rate']:.1%}",
            delta_color=win_rate_color
        )
    
    with pl_col5:
        st.metric("Total Wins", f"${portfolio['total_wins']:+,.2f}")
    
    with pl_col6:
        st.metric("Total Losses", f"${portfolio['total_losses']:+,.2f}")

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
                st.write(f"Qty: {position['quantity']:.4f} | Entry: ${position['entry_price']:.2f}")
                st.write(f"Strategy: {position['strategy']} | Value: ${position.get('position_value', 0):.2f}")
            
            with col2:
                current_price = position.get('current_price', position['entry_price'])
                pnl = position.get('pnl', 0)
                pnl_percentage = position.get('pnl_percentage', 0)
                pnl_color = "green" if pnl >= 0 else "red"
                st.write(f"Current: ${current_price:.2f}")
                st.write(f"<span style='color:{pnl_color};'>P&L: ${pnl:+,.2f} ({pnl_percentage:+.1f}%)</span>", 
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
    """Create comprehensive trading history dashboard - SESSION BASED"""
    st.subheader("📋 Trading History (Current Session Only)")
    
    # Get session trade history
    trade_history = trading_engine.get_trade_history(100)
    
    if not trade_history:
        st.info("No trades in current session. Start trading to generate trades!")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["All Trades", "Performance Charts", "Strategy Analysis"])
    
    with tab1:
        # Display all trades in a table
        st.subheader("📊 All Trades (Current Session)")
        
        # Prepare data for display
        history_data = []
        for idx, trade in enumerate(trade_history, 1):
            # Calculate P&L
            pnl = 0
            pnl_percentage = 0
            
            if trade['status'] == 'CLOSED':
                pnl = trade.get('closed_pnl', 0)
                if trade.get('position_value', 0) > 0:
                    pnl_percentage = (pnl / trade.get('position_value', 0)) * 100
            else:
                pnl = trade.get('pnl', 0)
                if trade.get('position_value', 0) > 0:
                    pnl_percentage = (pnl / trade.get('position_value', 0)) * 100
            
            # Format data
            history_data.append({
                'ID': idx,
                'Symbol': trade['symbol'],
                'Action': trade['action'],
                'Strategy': trade['strategy'],
                'Qty': f"{trade['quantity']:.4f}",
                'Entry': f"${trade['entry_price']:.2f}",
                'Exit': f"${trade.get('exit_price', 'N/A'):.2f}" if trade.get('exit_price') else "N/A",
                'P&L': f"${pnl:+,.2f}",
                'P&L %': f"{pnl_percentage:+.1f}%" if pnl_percentage else "N/A",
                'Value': f"${trade.get('position_value', 0):.2f}",
                'Status': trade['status'][0]  # 'O' for Open, 'C' for Closed
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            
            # Style the dataframe
            def color_pnl(val):
                color = 'green' if '+' in val else 'red' if '-' in val else 'black'
                return f'color: {color}'
            
            styled_df = df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Export option
            if st.button("📥 Export Session History"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with tab2:
        # Performance charts for current session
        st.subheader("📈 Performance Charts (Current Session)")
        
        # Filter closed trades
        closed_trades = [t for t in trade_history if t.get('status') == 'CLOSED']
        
        if closed_trades:
            # Prepare data for charts
            df_trades = pd.DataFrame([
                {
                    'Date': t['timestamp'] if isinstance(t['timestamp'], datetime) else datetime.now(),
                    'P&L': t.get('closed_pnl', 0),
                    'P&L %': t.get('closed_pnl_percentage', 0),
                    'Strategy': t.get('strategy', 'unknown'),
                    'Symbol': t['symbol'],
                    'Action': t['action'],
                    'Value': t.get('position_value', 0)
                }
                for t in closed_trades
            ])
            
            if not df_trades.empty:
                df_trades = df_trades.sort_values('Date')
                df_trades['Cumulative P&L'] = df_trades['P&L'].cumsum()
                df_trades['Rolling Win Rate'] = (df_trades['P&L'] > 0).rolling(window=5, min_periods=1).mean()
                
                # Cumulative P&L chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=df_trades['Date'],
                    y=df_trades['Cumulative P&L'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='green' if df_trades['Cumulative P&L'].iloc[-1] > 0 else 'red', width=2),
                    hovertemplate='Date: %{x}<br>Cumulative P&L: $%{y:.2f}<extra></extra>'
                ))
                fig1.update_layout(
                    title="Cumulative P&L (Current Session)",
                    xaxis_title="Time",
                    yaxis_title="Cumulative P&L ($)",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Daily P&L bar chart
                if len(df_trades) > 1:
                    df_trades['Hour'] = pd.to_datetime(df_trades['Date']).dt.floor('H')
                    hourly_pnl = df_trades.groupby('Hour')['P&L'].sum().reset_index()
                    
                    fig2 = go.Figure()
                    colors = ['green' if x >= 0 else 'red' for x in hourly_pnl['P&L']]
                    fig2.add_trace(go.Bar(
                        x=hourly_pnl['Hour'],
                        y=hourly_pnl['P&L'],
                        name='Hourly P&L',
                        marker_color=colors
                    ))
                    fig2.update_layout(
                        title="Hourly P&L Distribution",
                        xaxis_title="Hour",
                        yaxis_title="P&L ($)",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Win rate chart
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=df_trades['Date'],
                    y=df_trades['Rolling Win Rate'] * 100,
                    mode='lines',
                    name='Rolling Win Rate (5 trades)',
                    line=dict(color='blue', width=2)
                ))
                fig3.add_hline(y=50, line_dash="dash", line_color="red")
                fig3.update_layout(
                    title="Rolling Win Rate",
                    xaxis_title="Time",
                    yaxis_title="Win Rate (%)",
                    height=400,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No closed trades in current session yet.")
    
    with tab3:
        # Strategy analysis for current session
        st.subheader("🎯 Strategy Performance (Current Session)")
        
        strategy_perf = trading_engine.get_strategy_performance()
        
        if strategy_perf:
            # Filter strategies with trades
            strategies_with_trades = {k: v for k, v in strategy_perf.items() if v['trades'] > 0}
            
            if strategies_with_trades:
                # Prepare strategy performance data
                perf_data = []
                for strategy, data in strategies_with_trades.items():
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
                
                df_perf = pd.DataFrame(perf_data)
                
                # Display as table
                st.dataframe(df_perf, use_container_width=True)
                
                # Strategy comparison chart
                if len(df_perf) > 1:
                    fig = px.bar(df_perf, x='Strategy', y='Win Rate',
                               title="Strategy Win Rate Comparison",
                               color='Total P&L',
                               color_continuous_scale='RdYlGn')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy has executed trades in this session yet.")
        else:
            st.info("No strategy performance data available.")

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
                            current_price = trading_engine._get_current_price(signal['symbol'])
                            st.write(f"Price: ${current_price:.2f}")
                        
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

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    """Main application function"""
    
    # Initialize session state
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = AlgorithmicTradingEngine(mode="paper", initial_capital=INITIAL_CAPITAL)
    
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
            value=1000,
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("New Session", use_container_width=True, type="primary"):
                # Create new engine with fresh session
                st.session_state.trading_engine = AlgorithmicTradingEngine(mode="paper", initial_capital=initial_capital)
                st.success("New trading session started!")
                st.rerun()
        
        with col2:
            if st.button("Refresh", use_container_width=True, type="secondary"):
                st.rerun()
        
        with col3:
            trading_engine = st.session_state.trading_engine
            if trading_engine.trade_history:
                df = pd.DataFrame(trading_engine.trade_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Export Session",
                    data=csv,
                    file_name=f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # System status
        st.markdown("---")
        trading_engine = st.session_state.trading_engine
        status_color = "🟢" if trading_engine.trading_active else "🔴"
        st.markdown(f"**System Status:** {status_color} {'Running' if trading_engine.trading_active else 'Stopped'}")
        st.markdown(f"**Mode:** Paper Trading")
        st.markdown(f"**Capital:** ${trading_engine.initial_capital:,.2f}")
        st.markdown(f"**Session Start:** {trading_engine.session_metrics['start_time'].strftime('%H:%M:%S')}")
        st.markdown(f"**Trades This Session:** {trading_engine.accuracy_metrics['executed_trades']}")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎮 Control",
        "📊 Portfolio",
        "💰 Positions",
        "📋 History",
        "🚦 Signals"
    ])
    
    trading_engine = st.session_state.trading_engine
    
    with tab1:
        # Control Panel
        create_trading_control_panel(trading_engine)
        
        # Quick stats
        st.subheader("⚡ Quick Stats (Current Session)")
        
        portfolio = trading_engine.get_portfolio_summary()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Active Strategies", len(trading_engine.strategies))
        
        with stat_col2:
            st.metric("Total Signals", portfolio['total_trades'])
        
        with stat_col3:
            st.metric("Open Positions", portfolio['open_positions'])
        
        with stat_col4:
            st.metric("Session P&L", f"${portfolio['total_pnl']:+,.2f}")
    
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>RANTV Algorithmic Trading System v6.0</strong> | Session-Based Trading | Smart Money Concept</p>
    <p>⚠️ This is for educational and paper trading purposes only. All trades are simulated.</p>
    <p>💰 <strong>Session-Based History Only | $1000 Capital | Realistic P&L</strong></p>
    <p><em>Note: Each browser session starts fresh. History is not saved between sessions.</em></p>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# RUN APPLICATION
# =============================================

if __name__ == "__main__":
    main()
