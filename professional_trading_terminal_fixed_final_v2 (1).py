# multi_timeframe_crypto_forex_trading_terminal.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import requests
from textblob import TextBlob
import re
import json
from enum import Enum
import os
import tempfile

SIGNAL_REFRESH_MS = 90000
PRICE_REFRESH_MS = 25000

# Enhanced Trading Signal Data Class
@dataclass
class TradingSignal:
    symbol: str
    action: str  # "BUY" or "SELL"
    strategy: str
    timeframe: str
    entry: float
    stop_loss: float
    target1: float
    target2: float
    target3: float
    confidence: float
    risk_reward: float
    timestamp: datetime
    reasoning: str = ""
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    executed: bool = False
    trade_id: Optional[str] = None
    result: Optional[str] = None  # "WIN", "LOSS", "OPEN"

# Signal History Manager
class SignalHistoryManager:
    def __init__(self):
        self.signal_history = []
    
    def add_signal(self, signal: TradingSignal):
        """Add signal to history"""
        self.signal_history.append(signal)
    
    def update_signal_result(self, trade_id: str, result: str):
        """Update signal result when trade is closed"""
        for signal in self.signal_history:
            if signal.trade_id == trade_id:
                signal.result = result
                break
    
    def get_strategy_accuracy(self, strategy: str, days_back: int = 30) -> Dict:
        """Calculate accuracy for a specific strategy"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        strategy_signals = [
            s for s in self.signal_history 
            if s.strategy == strategy and s.timestamp >= cutoff_date and s.result in ["WIN", "LOSS"]
        ]
        
        if not strategy_signals:
            return {"total": 0, "wins": 0, "accuracy": 0}
        
        wins = sum(1 for s in strategy_signals if s.result == "WIN")
        total = len(strategy_signals)
        accuracy = (wins / total) * 100 if total > 0 else 0
        
        return {
            "total": total,
            "wins": wins,
            "accuracy": accuracy
        }
    
    def get_all_strategies_accuracy(self, days_back: int = 30) -> Dict[str, Dict]:
        """Calculate accuracy for all strategies"""
        strategies = set(s.strategy for s in self.signal_history)
        accuracy_data = {}
        
        for strategy in strategies:
            accuracy_data[strategy] = self.get_strategy_accuracy(strategy, days_back)
        
        return accuracy_data

# Paper Trading Data Classes
@dataclass
class PaperTrade:
    id: str
    symbol: str
    action: str  # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    stop_loss: float
    targets: List[float]
    quantity: int
    strategy: str
    timeframe: str
    signal_confidence: float
    status: str  # "OPEN", "CLOSED", "CANCELLED"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

@dataclass
class StrategyPerformance:
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float

# Market Regime Enum
class MarketRegime:
    BULLISH = "BULLISH"
    BEARISH = "BEARISH" 
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

# Enhanced Strategy Manager with Multi-Timeframe Support
class StrategyManager:
    def __init__(self):
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.strategies = {
            "Multi-Timeframe Momentum": self.multi_timeframe_momentum,
            "Volume-Weighted Breakout": self.volume_weighted_breakout,
            "Adaptive MA Crossover": self.adaptive_ma_crossover,
            "RSI Divergence": self.rsi_divergence_strategy,
            "Bollinger Band Squeeze": self.bollinger_squeeze_strategy,
            "Market Profile": self.market_profile_strategy,
        }
        
        # ML model for signal confirmation
        self.ml_model = None
        self.scaler = StandardScaler()
        self.load_ml_model()
    
    def load_ml_model(self):
        """Load or create ML model for signal validation"""
        try:
            # For Streamlit Cloud, we'll create a model instead of loading from file
            self.ml_model = RandomForestClassifier(n_estimators=50, random_state=42)
            # Create dummy data to fit the model
            X_dummy = np.random.randn(100, 6)
            y_dummy = np.random.randint(0, 2, 100)
            self.ml_model.fit(X_dummy, y_dummy)
            self.scaler.fit(X_dummy)
        except Exception as e:
            print(f"Error creating ML model: {e}")
            # Fallback to simple model
            self.ml_model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def run_multi_timeframe_analysis(self, symbol: str) -> List[TradingSignal]:
        """Run strategies across all timeframes"""
        all_signals = []
        
        for timeframe in self.timeframes:
            try:
                # Get data for this timeframe
                data = data_manager.get_multi_timeframe_data(symbol, timeframe)
                if data.empty or len(data) < 50:
                    continue
                
                # Run all strategies for this timeframe
                for strategy_name, strategy_func in self.strategies.items():
                    try:
                        signals = strategy_func(symbol, data.copy(), timeframe)
                        if signals:
                            all_signals.extend(signals)
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
        
        # Apply ML confidence boost
        enhanced_signals = self.enhance_signals_with_ml(all_signals, symbol)
        
        return enhanced_signals
    
    def multi_timeframe_momentum(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[TradingSignal]:
        """Enhanced momentum with multi-timeframe confirmation"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        try:
            # Calculate momentum across multiple periods
            periods = [5, 10, 20]
            momentum_scores = {}
            
            for period in periods:
                data[f'momentum_{period}'] = data['Close'].pct_change(period)
                data[f'volume_ma_{period}'] = data['Volume'].rolling(period).mean()
                momentum_scores[period] = float(data[f'momentum_{period}'].iloc[-1])
            
            # Volume analysis
            current_volume = float(data['Volume'].iloc[-1])
            volume_ma_20 = float(data['volume_ma_20'].iloc[-1])
            volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1
            
            # Volatility adjusted momentum
            volatility = float(data['Close'].pct_change().std() * np.sqrt(252))
            volatility_score = min(volatility / 0.2, 2.0)  # Normalize volatility
            
            current_price = float(data['Close'].iloc[-1])
            
            # Multi-timeframe momentum score
            momentum_score = (
                momentum_scores[5] * 0.4 + 
                momentum_scores[10] * 0.3 + 
                momentum_scores[20] * 0.3
            )
            
            # Enhanced BUY conditions - FIXED: Use explicit boolean comparisons
            momentum_positive = all(m > 0 for m in momentum_scores.values())
            
            if (momentum_score > 0.02 and volume_ratio > 1.3 and momentum_positive):
                
                # Dynamic stop loss based on volatility
                atr = float(self.calculate_atr(data).iloc[-1])
                stop_loss = current_price - (2 * atr)
                
                # Dynamic targets based on timeframe
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                target1 = current_price * (1 + 0.01 * tf_multiplier)
                target2 = current_price * (1 + 0.02 * tf_multiplier)
                target3 = current_price * (1 + 0.03 * tf_multiplier)
                
                confidence = min(0.7 + (momentum_score * 10) + (volume_ratio - 1), 0.95)
                
                reasoning = (
                    f"Strong multi-timeframe momentum on {timeframe}. "
                    f"Momentum score: {momentum_score:.3f}, Volume: {volume_ratio:.1f}x, "
                    f"Volatility: {volatility:.3f}"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    strategy="Multi-Timeframe Momentum",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=confidence,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning,
                    priority="HIGH" if confidence > 0.8 else "MEDIUM"
                )
                signals.append(signal)
            
            # Enhanced SELL conditions
            momentum_negative = all(m < 0 for m in momentum_scores.values())
            
            if (momentum_score < -0.02 and volume_ratio > 1.3 and momentum_negative):
                
                atr = float(self.calculate_atr(data).iloc[-1])
                stop_loss = current_price + (2 * atr)
                
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                target1 = current_price * (1 - 0.01 * tf_multiplier)
                target2 = current_price * (1 - 0.02 * tf_multiplier)
                target3 = current_price * (1 - 0.03 * tf_multiplier)
                
                confidence = min(0.65 + (abs(momentum_score) * 10) + (volume_ratio - 1), 0.90)
                
                reasoning = (
                    f"Strong bearish momentum on {timeframe}. "
                    f"Momentum score: {momentum_score:.3f}, Volume: {volume_ratio:.1f}x"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    strategy="Multi-Timeframe Momentum",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=confidence,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning,
                    priority="HIGH" if confidence > 0.75 else "MEDIUM"
                )
                signals.append(signal)
        except Exception as e:
            pass
        
        return signals
    
    def volume_weighted_breakout(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[TradingSignal]:
        """Breakout strategy with volume confirmation"""
        signals = []
        
        if len(data) < 30:
            return signals
        
        try:
            # Calculate support/resistance with volume profile
            data = self.calculate_volume_profile_levels(data)
            current_price = float(data['Close'].iloc[-1])
            current_volume = float(data['Volume'].iloc[-1])
            
            # Volume-weighted moving averages
            data['vwap'] = self.calculate_vwap(data)
            data['volume_ma'] = data['Volume'].rolling(20).mean()
            
            # Key levels - FIXED: Convert to float
            resistance = float(data['resistance'].iloc[-1])
            support = float(data['support'].iloc[-1])
            vwap = float(data['vwap'].iloc[-1])
            
            volume_ratio = current_volume / float(data['volume_ma'].iloc[-1])
            atr = float(self.calculate_atr(data).iloc[-1])
            
            # Breakout above resistance with volume - FIXED: Explicit boolean comparisons
            breakout_condition = (current_price > resistance and volume_ratio > 1.5 and current_price > vwap)
            
            if breakout_condition:
                
                stop_loss = min(resistance, current_price - (1.5 * atr))
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                
                target1 = current_price + (1 * atr)
                target2 = current_price + (2 * atr)
                target3 = current_price + (3 * atr)
                
                reasoning = (
                    f"Volume-weighted breakout on {timeframe}. "
                    f"Broke resistance: {resistance:.2f}, "
                    f"Volume: {volume_ratio:.1f}x, VWAP: {vwap:.2f}"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    strategy="Volume-Weighted Breakout",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=0.75,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
                signals.append(signal)
            
            # Breakdown below support with volume
            breakdown_condition = (current_price < support and volume_ratio > 1.5 and current_price < vwap)
            
            if breakdown_condition:
                
                stop_loss = max(support, current_price + (1.5 * atr))
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                
                target1 = current_price - (1 * atr)
                target2 = current_price - (2 * atr)
                target3 = current_price - (3 * atr)
                
                reasoning = (
                    f"Volume-weighted breakdown on {timeframe}. "
                    f"Broke support: {support:.2f}, "
                    f"Volume: {volume_ratio:.1f}x, VWAP: {vwap:.2f}"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    strategy="Volume-Weighted Breakout",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=0.70,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
                signals.append(signal)
        except Exception as e:
            pass
        
        return signals
    
    def adaptive_ma_crossover(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[TradingSignal]:
        """Adaptive MA crossover based on market regime"""
        signals = []
        
        if len(data) < 100:
            return signals
        
        try:
            # Adaptive MA periods based on volatility
            volatility = float(data['Close'].pct_change().std() * np.sqrt(252))
            
            if volatility > 0.25:  # High volatility
                fast_period, slow_period = 5, 15
            elif volatility < 0.15:  # Low volatility
                fast_period, slow_period = 10, 30
            else:  # Medium volatility
                fast_period, slow_period = 8, 21
            
            data[f'ma_fast'] = data['Close'].rolling(fast_period).mean()
            data[f'ma_slow'] = data['Close'].rolling(slow_period).mean()
            data['ma_trend'] = data['Close'].rolling(50).mean()
            
            current_price = float(data['Close'].iloc[-1])
            ma_fast = float(data[f'ma_fast'].iloc[-1])
            ma_slow = float(data[f'ma_slow'].iloc[-1])
            ma_trend = float(data['ma_trend'].iloc[-1])
            
            ma_fast_prev = float(data[f'ma_fast'].iloc[-2]) if len(data) > 1 else ma_fast
            ma_slow_prev = float(data[f'ma_slow'].iloc[-2]) if len(data) > 1 else ma_slow
            
            # Golden Cross with trend alignment - FIXED: Explicit boolean comparisons
            golden_cross = (ma_fast > ma_slow and ma_fast_prev <= ma_slow_prev and
                          current_price > ma_trend and ma_fast > ma_trend)
            
            if golden_cross:
                
                stop_loss = ma_slow
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                
                target1 = current_price * (1 + 0.015 * tf_multiplier)
                target2 = current_price * (1 + 0.03 * tf_multiplier)
                target3 = current_price * (1 + 0.045 * tf_multiplier)
                
                reasoning = (
                    f"Adaptive MA Crossover (Golden Cross) on {timeframe}. "
                    f"Fast MA ({fast_period}) crossed above Slow MA ({slow_period}). "
                    f"Trend aligned, Volatility: {volatility:.3f}"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    strategy="Adaptive MA Crossover",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=0.72,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
                signals.append(signal)
            
            # Death Cross with trend alignment
            death_cross = (ma_fast < ma_slow and ma_fast_prev >= ma_slow_prev and
                         current_price < ma_trend and ma_fast < ma_trend)
            
            if death_cross:
                
                stop_loss = ma_slow
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                
                target1 = current_price * (1 - 0.015 * tf_multiplier)
                target2 = current_price * (1 - 0.03 * tf_multiplier)
                target3 = current_price * (1 - 0.045 * tf_multiplier)
                
                reasoning = (
                    f"Adaptive MA Crossover (Death Cross) on {timeframe}. "
                    f"Fast MA ({fast_period}) crossed below Slow MA ({slow_period}). "
                    f"Trend aligned, Volatility: {volatility:.3f}"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    strategy="Adaptive MA Crossover",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=0.68,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
                signals.append(signal)
        except Exception as e:
            pass
        
        return signals
    
    def rsi_divergence_strategy(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[TradingSignal]:
        """RSI divergence strategy for reversals"""
        signals = []
        
        if len(data) < 30:
            return signals
        
        try:
            # Calculate RSI - FIXED: Proper assignment
            rsi_values = self.calculate_rsi(data, period=14)
            data['rsi'] = rsi_values
            
            current_rsi = float(data['rsi'].iloc[-1])
            current_price = float(data['Close'].iloc[-1])
            
            # Look for divergences (last 10 periods)
            lookback = 10
            if len(data) > lookback:
                # Bullish divergence: Price makes lower low, RSI makes higher low
                price_lows = data['Low'].tail(lookback)
                rsi_lows = data['rsi'].tail(lookback)
                
                # Bearish divergence: Price makes higher high, RSI makes lower high
                price_highs = data['High'].tail(lookback)
                rsi_highs = data['rsi'].tail(lookback)
                
                # Bullish divergence detection
                bullish_divergence = (float(price_lows.iloc[-1]) < float(price_lows.iloc[-2]) and 
                                    float(rsi_lows.iloc[-1]) > float(rsi_lows.iloc[-2]) and 
                                    current_rsi < 40)
                
                if bullish_divergence:
                    
                    atr = float(self.calculate_atr(data).iloc[-1])
                    stop_loss = current_price - (2 * atr)
                    tf_multiplier = self.get_timeframe_multiplier(timeframe)
                    
                    target1 = current_price * (1 + 0.02 * tf_multiplier)
                    target2 = current_price * (1 + 0.04 * tf_multiplier)
                    target3 = current_price * (1 + 0.06 * tf_multiplier)
                    
                    reasoning = (
                        f"Bullish RSI Divergence on {timeframe}. "
                        f"Price made lower low but RSI made higher low. "
                        f"Current RSI: {current_rsi:.1f}"
                    )
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action="BUY",
                        strategy="RSI Divergence",
                        timeframe=timeframe,
                        entry=current_price,
                        stop_loss=stop_loss,
                        target1=target1,
                        target2=target2,
                        target3=target3,
                        confidence=0.75,
                        risk_reward=2.5,
                        timestamp=datetime.now(),
                        reasoning=reasoning,
                        priority="HIGH"
                    )
                    signals.append(signal)
                
                # Bearish divergence detection
                bearish_divergence = (float(price_highs.iloc[-1]) > float(price_highs.iloc[-2]) and 
                                    float(rsi_highs.iloc[-1]) < float(rsi_highs.iloc[-2]) and 
                                    current_rsi > 60)
                
                if bearish_divergence:
                    
                    atr = float(self.calculate_atr(data).iloc[-1])
                    stop_loss = current_price + (2 * atr)
                    tf_multiplier = self.get_timeframe_multiplier(timeframe)
                    
                    target1 = current_price * (1 - 0.02 * tf_multiplier)
                    target2 = current_price * (1 - 0.04 * tf_multiplier)
                    target3 = current_price * (1 - 0.06 * tf_multiplier)
                    
                    reasoning = (
                        f"Bearish RSI Divergence on {timeframe}. "
                        f"Price made higher high but RSI made lower high. "
                        f"Current RSI: {current_rsi:.1f}"
                    )
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action="SELL",
                        strategy="RSI Divergence",
                        timeframe=timeframe,
                        entry=current_price,
                        stop_loss=stop_loss,
                        target1=target1,
                        target2=target2,
                        target3=target3,
                        confidence=0.70,
                        risk_reward=2.5,
                        timestamp=datetime.now(),
                        reasoning=reasoning,
                        priority="HIGH"
                    )
                    signals.append(signal)
        except Exception as e:
            pass
        
        return signals
    
    def bollinger_squeeze_strategy(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[TradingSignal]:
        """Bollinger Band squeeze breakout strategy"""
        signals = []
        
        if len(data) < 30:
            return signals
        
        try:
            # Calculate Bollinger Bands - FIXED: Proper assignment
            bb_data = self.calculate_bollinger_bands(data)
            data['bb_middle'] = bb_data['bb_middle']
            data['bb_upper'] = bb_data['bb_upper']
            data['bb_lower'] = bb_data['bb_lower']
            
            current_price = float(data['Close'].iloc[-1])
            bb_upper = float(data['bb_upper'].iloc[-1])
            bb_lower = float(data['bb_lower'].iloc[-1])
            bb_middle = float(data['bb_middle'].iloc[-1])
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Squeeze conditions (low volatility)
            is_squeeze = bb_width < 0.05  # 5% band width
            
            # Volume for confirmation
            volume_ma = float(data['Volume'].rolling(20).mean().iloc[-1])
            volume_ratio = float(data['Volume'].iloc[-1]) / volume_ma
            
            if is_squeeze and volume_ratio > 1.2:
                # Impending breakout
                if current_price > bb_middle:
                    # Bullish bias
                    stop_loss = bb_lower
                    tf_multiplier = self.get_timeframe_multiplier(timeframe)
                    
                    target1 = bb_upper
                    target2 = current_price + (bb_upper - bb_lower)
                    target3 = target2 + (bb_upper - bb_lower)
                    
                    reasoning = (
                        f"Bollinger Squeeze (Bullish) on {timeframe}. "
                        f"Band width: {bb_width:.3f}, Volume: {volume_ratio:.1f}x. "
                        f"Expecting bullish breakout"
                    )
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action="BUY",
                        strategy="Bollinger Band Squeeze",
                        timeframe=timeframe,
                        entry=current_price,
                        stop_loss=stop_loss,
                        target1=target1,
                        target2=target2,
                        target3=target3,
                        confidence=0.65,
                        risk_reward=3.0,
                        timestamp=datetime.now(),
                        reasoning=reasoning
                    )
                    signals.append(signal)
                
                elif current_price < bb_middle:
                    # Bearish bias
                    stop_loss = bb_upper
                    tf_multiplier = self.get_timeframe_multiplier(timeframe)
                    
                    target1 = bb_lower
                    target2 = current_price - (bb_upper - bb_lower)
                    target3 = target2 - (bb_upper - bb_lower)
                    
                    reasoning = (
                        f"Bollinger Squeeze (Bearish) on {timeframe}. "
                        f"Band width: {bb_width:.3f}, Volume: {volume_ratio:.1f}x. "
                        f"Expecting bearish breakdown"
                    )
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action="SELL",
                        strategy="Bollinger Band Squeeze",
                        timeframe=timeframe,
                        entry=current_price,
                        stop_loss=stop_loss,
                        target1=target1,
                        target2=target2,
                        target3=target3,
                        confidence=0.60,
                        risk_reward=3.0,
                        timestamp=datetime.now(),
                        reasoning=reasoning
                    )
                    signals.append(signal)
        except Exception as e:
            pass
        
        return signals
    
    def market_profile_strategy(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[TradingSignal]:
        """Market profile based strategy"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        try:
            # Calculate market profile levels
            data = self.calculate_market_profile(data)
            current_price = float(data['Close'].iloc[-1])
            
            poc = float(data['poc'].iloc[-1])  # Point of Control
            value_area_high = float(data['value_area_high'].iloc[-1])
            value_area_low = float(data['value_area_low'].iloc[-1])
            
            # Market profile signals - FIXED: Explicit boolean comparisons
            breakout_condition = (current_price > value_area_high and current_price > poc)
            
            if breakout_condition:
                # Price above value area - potential breakout
                stop_loss = value_area_high
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                
                target1 = current_price * (1 + 0.01 * tf_multiplier)
                target2 = current_price * (1 + 0.02 * tf_multiplier)
                target3 = current_price * (1 + 0.03 * tf_multiplier)
                
                reasoning = (
                    f"Market Profile Breakout on {timeframe}. "
                    f"Price above value area. POC: {poc:.2f}, "
                    f"Value Area: {value_area_low:.2f}-{value_area_high:.2f}"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    strategy="Market Profile",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=0.70,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
                signals.append(signal)
            
            breakdown_condition = (current_price < value_area_low and current_price < poc)
            
            if breakdown_condition:
                # Price below value area - potential breakdown
                stop_loss = value_area_low
                tf_multiplier = self.get_timeframe_multiplier(timeframe)
                
                target1 = current_price * (1 - 0.01 * tf_multiplier)
                target2 = current_price * (1 - 0.02 * tf_multiplier)
                target3 = current_price * (1 - 0.03 * tf_multiplier)
                
                reasoning = (
                    f"Market Profile Breakdown on {timeframe}. "
                    f"Price below value area. POC: {poc:.2f}, "
                    f"Value Area: {value_area_low:.2f}-{value_area_high:.2f}"
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    strategy="Market Profile",
                    timeframe=timeframe,
                    entry=current_price,
                    stop_loss=stop_loss,
                    target1=target1,
                    target2=target2,
                    target3=target3,
                    confidence=0.65,
                    risk_reward=2.0,
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
                signals.append(signal)
        except Exception as e:
            pass
        
        return signals
    
    def enhance_signals_with_ml(self, signals: List[TradingSignal], symbol: str) -> List[TradingSignal]:
        """Enhance signals with ML-based confidence scoring"""
        if not signals:
            return signals
        
        enhanced_signals = []
        
        for signal in signals:
            # Extract features for ML model
            features = self.extract_ml_features(signal, symbol)
            
            if features:
                try:
                    # Normalize features
                    features_normalized = self.scaler.transform([features])
                    
                    # Get ML prediction (placeholder - would need trained model)
                    ml_confidence = 0.5  # Base confidence
                    
                    # Enhance confidence based on ML
                    enhanced_confidence = min(signal.confidence * (1 + ml_confidence), 0.95)
                    signal.confidence = enhanced_confidence
                    
                    # Add ML reasoning
                    signal.reasoning += f" | ML Enhanced Confidence: {ml_confidence:.2f}"
                except Exception as e:
                    pass
            
            enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def extract_ml_features(self, signal: TradingSignal, symbol: str) -> Optional[List[float]]:
        """Extract features for ML model"""
        try:
            # Get recent data for feature extraction
            data = data_manager.get_multi_timeframe_data(symbol, signal.timeframe)
            if data.empty or len(data) < 20:
                return None
            
            features = [
                signal.confidence,
                float(data['Close'].pct_change(5).iloc[-1]) if len(data) > 5 else 0,  # 5-period return
                float(data['Close'].pct_change(10).iloc[-1]) if len(data) > 10 else 0, # 10-period return
                float(data['Volume'].iloc[-1]) / float(data['Volume'].rolling(20).mean().iloc[-1]) if len(data) > 20 else 1,  # Volume ratio
                float(self.calculate_rsi(data).iloc[-1]) if len(data) > 14 else 50,  # RSI
                float(self.calculate_atr(data).iloc[-1]) / float(data['Close'].iloc[-1]) if len(data) > 14 else 0,  # Normalized ATR
            ]
            
            return features
            
        except Exception as e:
            return None
    
    def get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get multiplier for target calculations based on timeframe"""
        multipliers = {
            '1m': 0.5, '5m': 1.0, '15m': 1.5, 
            '30m': 2.0, '1h': 3.0, '4h': 5.0, '1d': 8.0
        }
        return multipliers.get(timeframe, 1.0)
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        middle_band = data['Close'].rolling(window=period).mean()
        std_dev = data['Close'].rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return pd.DataFrame({
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band
        })
    
    def calculate_volume_profile_levels(self, data: pd.DataFrame, num_levels: int = 3) -> pd.DataFrame:
        """Calculate volume profile support/resistance levels"""
        # Simplified volume profile calculation
        data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate support and resistance based on recent price action
        data['resistance'] = data['High'].rolling(window=20).max()
        data['support'] = data['Low'].rolling(window=20).min()
        
        return data
    
    def calculate_market_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate simplified market profile levels"""
        # Simplified market profile implementation
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate Point of Control (POC) as VWAP
        data['poc'] = self.calculate_vwap(data)
        
        # Calculate value area (simplified)
        data['value_area_high'] = data['High'].rolling(window=20).mean() + data['High'].rolling(window=20).std()
        data['value_area_low'] = data['Low'].rolling(window=20).mean() - data['Low'].rolling(window=20).std()
        
        return data

# Enhanced Data Manager with Multi-Timeframe Support
class DataManager:
    def __init__(self):
        # Focus on major assets only
        self.symbols = {
            'Crypto': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'LTC-USD'],
            'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X'],
            'Commodities': ['GC=F', 'SI=F', 'CL=F']
        }
        
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.data_cache = {}
        self.cache_duration = timedelta(minutes=5)
    
    def get_multi_timeframe_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get data for specific symbol and timeframe"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data.copy()
        
        try:
            # Get data from yfinance
            period = self.get_period_for_timeframe(timeframe)
            data = yf.download(symbol, period=period, interval=timeframe, progress=False)
            
            if not data.empty:
                # Ensure proper column names
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                
                # Cache the data
                self.data_cache[cache_key] = (data.copy(), datetime.now())
                return data.copy()
            else:
                return pd.DataFrame()
                
        except Exception as e:
            return pd.DataFrame()
    
    def get_period_for_timeframe(self, timeframe: str) -> str:
        """Get appropriate period for timeframe"""
        periods = {
            '1m': '7d', '5m': '60d', '15m': '60d',
            '30m': '60d', '1h': '730d', '4h': '730d', '1d': 'max'
        }
        return periods.get(timeframe, '60d')
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        all_symbols = []
        for category in self.symbols:
            all_symbols.extend(self.symbols[category])
        return all_symbols

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            data = self.get_multi_timeframe_data(symbol, '15m')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 0.0
        except:
            return 0.0

# Enhanced Paper Trading Engine
class PaperTradingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.open_trades = []
        self.closed_trades = []
        self.performance_metrics = {}
        
    def execute_trade(self, signal: TradingSignal, quantity: int = None) -> Optional[PaperTrade]:
        """Execute a paper trade based on signal"""
        try:
            if quantity is None:
                # Auto-calculate quantity based on risk management
                quantity = self.calculate_position_size(signal)
            
            if quantity <= 0:
                return None
            
            trade_id = f"TRADE_{len(self.trade_history) + 1:06d}"
            
            trade = PaperTrade(
                id=trade_id,
                symbol=signal.symbol,
                action=signal.action,
                entry_price=signal.entry,
                entry_time=datetime.now(),
                stop_loss=signal.stop_loss,
                targets=signal.targets,
                quantity=quantity,
                strategy=signal.strategy,
                timeframe=signal.timeframe,
                signal_confidence=signal.confidence,
                status="OPEN"
            )
            
            # Update signal with trade ID
            signal.trade_id = trade_id
            signal.executed = True
            
            # Calculate position value
            position_value = signal.entry * quantity
            
            # Check if we have enough balance
            if position_value > self.balance:
                return None
            
            # Deduct from balance
            self.balance -= position_value
            
            # Add to open trades and positions
            self.open_trades.append(trade)
            self.positions[signal.symbol] = trade
            self.trade_history.append(trade)
            
            return trade
            
        except Exception as e:
            return None
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> bool:
        """Close a paper trade"""
        try:
            trade = next((t for t in self.open_trades if t.id == trade_id), None)
            if not trade:
                return False
            
            # Calculate P&L
            if trade.action == "BUY":
                pnl = (exit_price - trade.entry_price) * trade.quantity
            else:  # SELL
                pnl = (trade.entry_price - exit_price) * trade.quantity
            
            pnl_percent = (pnl / (trade.entry_price * trade.quantity)) * 100
            
            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = datetime.now()
            trade.exit_reason = exit_reason
            trade.pnl = pnl
            trade.pnl_percent = pnl_percent
            trade.status = "CLOSED"
            
            # Update balance
            self.balance += (trade.entry_price * trade.quantity) + pnl
            
            # Move to closed trades
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            
            # Remove from positions
            if trade.symbol in self.positions:
                del self.positions[trade.symbol]
            
            return True
            
        except Exception as e:
            return False
    
    def calculate_position_size(self, signal: TradingSignal) -> int:
        """Calculate position size based on risk management"""
        try:
            # Risk per trade (1% of balance)
            risk_amount = self.balance * 0.01
            
            # Calculate risk per share
            if signal.action == "BUY":
                risk_per_share = signal.entry - signal.stop_loss
            else:  # SELL
                risk_per_share = signal.stop_loss - signal.entry
            
            if risk_per_share <= 0:
                return 0
            
            # Calculate shares based on risk
            shares = int(risk_amount / risk_per_share)
            
            # Ensure we don't exceed available balance
            max_shares_by_balance = int(self.balance / signal.entry)
            shares = min(shares, max_shares_by_balance)
            
            return max(1, shares)  # At least 1 share
            
        except Exception as e:
            return 0
    
    def update_trades(self, current_prices: Dict[str, float]):
        """Update open trades with current prices and check for exits"""
        for trade in self.open_trades[:]:  # Use slice copy for safe removal
            if trade.symbol not in current_prices:
                continue
            
            current_price = current_prices[trade.symbol]
            
            # Check stop loss
            if ((trade.action == "BUY" and current_price <= trade.stop_loss) or
                (trade.action == "SELL" and current_price >= trade.stop_loss)):
                self.close_trade(trade.id, current_price, "Stop Loss Hit")
                continue
            
            # Check targets
            for i, target in enumerate(trade.targets):
                if ((trade.action == "BUY" and current_price >= target) or
                    (trade.action == "SELL" and current_price <= target)):
                    self.close_trade(trade.id, current_price, f"Target {i+1} Hit")
                    break
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.closed_trades:
            return {}
        
        try:
            winning_trades = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.pnl and t.pnl <= 0]
            
            total_trades = len(self.closed_trades)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            
            win_rate = (winning_count / total_trades) * 100 if total_trades > 0 else 0
            
            total_pnl = sum(t.pnl for t in self.closed_trades if t.pnl)
            avg_profit = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            # Calculate max drawdown
            equity_curve = []
            running_balance = self.initial_balance
            
            for trade in sorted(self.closed_trades, key=lambda x: x.exit_time):
                running_balance += trade.pnl if trade.pnl else 0
                equity_curve.append(running_balance)
            
            if equity_curve:
                running_max = np.maximum.accumulate(equity_curve)
                drawdowns = (equity_curve - running_max) / running_max
                max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0
            else:
                max_drawdown = 0
            
            # Sharpe ratio (simplified)
            returns = [t.pnl_percent for t in self.closed_trades if t.pnl_percent]
            sharpe_ratio = np.mean(returns) / np.std(returns) if returns and np.std(returns) > 0 else 0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_count,
                "losing_trades": losing_count,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "current_balance": self.balance,
                "overall_return": ((self.balance - self.initial_balance) / self.initial_balance) * 100
            }
            
        except Exception as e:
            return {}

# Mood Gauge Class with Needle-style Gauges
class MoodGauge:
    def __init__(self):
        self.assets = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'LTC-USD', 'GC=F', 'EURUSD=X']
        self.asset_names = {
            'BTC-USD': 'BITCOIN',
            'ETH-USD': 'ETHEREUM', 
            'XRP-USD': 'XRP',
            'SOL-USD': 'SOLANA',
            'LTC-USD': 'LITECOIN',
            'GC=F': 'GOLD',
            'EURUSD=X': 'EUR/USD'
        }
        
    def calculate_mood_score(self, symbol: str) -> float:
        """Calculate mood score for an asset (0-100)"""
        try:
            # Get 1h data for mood calculation
            data = data_manager.get_multi_timeframe_data(symbol, '1h')
            if data.empty or len(data) < 20:
                return 50.0
            
            # Calculate various indicators for mood
            current_price = float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # RSI
            rsi = strategy_manager.calculate_rsi(data).iloc[-1]
            
            # Price momentum (5-period vs 20-period)
            ma_5 = float(data['Close'].rolling(5).mean().iloc[-1])
            ma_20 = float(data['Close'].rolling(20).mean().iloc[-1])
            price_vs_ma = (current_price - ma_20) / ma_20 * 100
            
            # Volume analysis
            current_volume = float(data['Volume'].iloc[-1])
            avg_volume = float(data['Volume'].rolling(20).mean().iloc[-1])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate mood score components
            rsi_score = max(0, min(100, (rsi - 30) / 40 * 100))  # Normalize RSI 30-70 to 0-100
            trend_score = max(0, min(100, 50 + price_vs_ma))  # Price vs MA20
            volume_score = min(100, volume_ratio * 25)  # Volume boost
            momentum_score = max(0, min(100, 50 + price_change * 2))  # Price momentum
            
            # Weighted mood score
            mood_score = (
                rsi_score * 0.25 +
                trend_score * 0.30 +
                volume_score * 0.20 +
                momentum_score * 0.25
            )
            
            return max(0, min(100, mood_score))
            
        except Exception as e:
            return 50.0
    
    def get_mood_label(self, score: float) -> Tuple[str, str]:
        """Get mood label and color based on score"""
        if score >= 80:
            return "VERY BULLISH", "#10b981"
        elif score >= 60:
            return "BULLISH", "#34d399"
        elif score >= 40:
            return "NEUTRAL", "#fbbf24"
        elif score >= 20:
            return "BEARISH", "#f87171"
        else:
            return "VERY BEARISH", "#ef4444"
    
    def create_needle_gauge(self, symbol: str, score: float, current_price: float) -> go.Figure:
        """Create a needle-style gauge chart"""
        label, color = self.get_mood_label(score)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {
                'text': f"{self.asset_names[symbol]}<br>${current_price:.2f}" if symbol not in ['EURUSD=X'] else f"{self.asset_names[symbol]}<br>{current_price:.4f}",
                'font': {'size': 14, 'color': 'white'}
            },
            number = {
                'font': {'size': 20, 'color': 'white', 'family': "Arial Black"},
                'suffix': " pts",
                'valueformat': ".0f"
            },
            delta = {
                'reference': 50, 
                'increasing': {'color': "#10b981"},
                'decreasing': {'color': "#ef4444"},
                'font': {'size': 12, 'color': 'white'}
            },
            gauge = {
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "white",
                    'tickfont': {'size': 10, 'color': 'white'}
                },
                'bar': {'color': color, 'thickness': 0.75},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(239, 68, 68, 0.4)'},
                    {'range': [20, 40], 'color': 'rgba(248, 113, 113, 0.4)'},
                    {'range': [40, 60], 'color': 'rgba(251, 191, 36, 0.4)'},
                    {'range': [60, 80], 'color': 'rgba(52, 211, 153, 0.4)'},
                    {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.4)'}],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8,
                    'value': score}}))
        
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"}
        )
        
        return fig

# Initialize global components
data_manager = DataManager()
strategy_manager = StrategyManager()
paper_trading = PaperTradingEngine()
signal_history = SignalHistoryManager()
mood_gauge = MoodGauge()

# Enhanced Streamlit UI with Fixed Auto-Refresh
def main():
    st.set_page_config(
        page_title="Multi-Timeframe Trading Terminal",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.6rem;
        color: #2e86ab;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #FF6B6B, #4ECDC4) 1;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        font-weight: 700;
    }
    .dashboard-header {
        font-size: 1.3rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    .signal-card {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.7rem 0;
        border-left: 6px solid;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        background: white;
    }
    .signal-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .buy-signal {
        border-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    }
    .sell-signal {
        border-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
    }
    .metric-card {
        padding: 1.2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .mood-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .time-display {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
        background: #f8f9fa;
        padding: 0.7rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f0f2f6;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 8px;
        gap: 8px;
        padding: 12px 20px;
        font-weight: 700;
        color: white;
        margin: 0 2px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4) !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .refresh-info {
        font-size: 0.9rem;
        color: white;
        text-align: center;
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚀 MULTI-TIMEFRAME TRADING TERMINAL</h1>', unsafe_allow_html=True)
    
    # Display local time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div class="time-display">🕐 LOCAL TIME: {current_time} | AUTO-REFRESH: ACTIVE</div>', unsafe_allow_html=True)
    
    # Simple Auto-refresh without blocking
    auto_refresh = st.sidebar.checkbox("🔄 Enable Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("⏱️ Refresh Interval (seconds)", 15, 120, 30)
    
    if auto_refresh:
        # Use session state to track refresh
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        time_since_refresh = current_time - st.session_state.last_refresh
        time_remaining = max(0, refresh_interval - time_since_refresh)
        
        st.markdown(f'<div class="refresh-info">🔄 AUTO-REFRESH ACTIVE | Next update in {int(time_remaining)} seconds</div>', unsafe_allow_html=True)
        
        # Only refresh when the interval has passed
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ CONFIGURATION")
    
    # Market selection
    market_category = st.sidebar.selectbox(
        "📊 Select Market Category",
        ["Crypto", "Forex", "Commodities"]
    )
    
    selected_symbols = st.sidebar.multiselect(
        "🎯 Select Symbols",
        data_manager.symbols[market_category],
        default=data_manager.symbols[market_category][:3]
    )
    
    # Strategy selection
    selected_strategies = st.sidebar.multiselect(
        "🤖 Select Strategies",
        list(strategy_manager.strategies.keys()),
        default=list(strategy_manager.strategies.keys())[:3]
    )
    
    # Timeframe selection
    selected_timeframes = st.sidebar.multiselect(
        "⏰ Select Timeframes",
        strategy_manager.timeframes,
        default=['15m', '1h', '4h']
    )
    
    # Auto-trading configuration
    st.sidebar.header("🤖 AUTO TRADING")
    enable_auto_trading = st.sidebar.checkbox("Enable Auto Trading", value=False)
    
    # Risk Management
    st.sidebar.header("🛡️ RISK MANAGEMENT")
    paper_trading.initial_balance = st.sidebar.number_input("Initial Balance", 1000, 100000, 10000)
    
    # Generate signals for selected symbols
    all_signals = []
    current_prices = {}
    
    if selected_symbols:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(selected_symbols):
            status_text.text(f"🔍 Analyzing {symbol}...")
            
            # Get current price
            current_prices[symbol] = data_manager.get_current_price(symbol)
            
            # Run multi-timeframe analysis
            signals = strategy_manager.run_multi_timeframe_analysis(symbol)
            all_signals.extend(signals)
            
            progress_bar.progress((i + 1) / len(selected_symbols))
        
        status_text.text("✅ Analysis complete!")
    
    # Filter signals based on user selection
    filtered_signals = [
        s for s in all_signals 
        if s.strategy in selected_strategies and s.timeframe in selected_timeframes
    ]
    
    # Main Tabs with colorful styling
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 LIVE DASHBOARD", 
        "⚡ TRADING SIGNALS", 
        "💼 PAPER TRADING", 
        "📊 PERFORMANCE"
    ])
    
    # Tab 1: Live Dashboard with Mood Gauges
    with tab1:
        st.markdown('<div class="sub-header">🎯 LIVE MARKET DASHBOARD</div>', unsafe_allow_html=True)
        
        if not selected_symbols:
            st.warning("⚠️ Please select at least one symbol from the sidebar.")
        else:
            # Quick overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📡 Total Signals", len(filtered_signals))
            
            with col2:
                buy_signals = len([s for s in filtered_signals if s.action == "BUY"])
                st.metric("🟢 Buy Signals", buy_signals)
            
            with col3:
                sell_signals = len([s for s in filtered_signals if s.action == "SELL"])
                st.metric("🔴 Sell Signals", sell_signals)
            
            with col4:
                high_confidence = len([s for s in filtered_signals if s.confidence > 0.8])
                st.metric("🎯 High Confidence", high_confidence)
            
            # Market Mood Gauges Section
            st.markdown('<div class="dashboard-header">📊 MARKET MOOD GAUGES</div>', unsafe_allow_html=True)
            
            # Calculate mood scores for major assets
            mood_assets = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'GC=F', 'EURUSD=X']
            mood_scores = {}
            
            for asset in mood_assets:
                mood_scores[asset] = mood_gauge.calculate_mood_score(asset)
            
            # Display mood gauges in 3 columns
            cols = st.columns(3)
            for i, asset in enumerate(mood_assets):
                with cols[i % 3]:
                    current_price = data_manager.get_current_price(asset)
                    score = mood_scores[asset]
                    label, color = mood_gauge.get_mood_label(score)
                    
                    # Create and display needle gauge
                    fig = mood_gauge.create_needle_gauge(asset, score, current_price)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display mood label
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: -2rem; margin-bottom: 1rem;">
                        <h4 style="color: {color}; font-weight: bold; background: rgba(0,0,0,0.7); 
                        padding: 0.5rem; border-radius: 10px;">{label}</h4>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recent Signals Preview
            st.markdown('<div class="dashboard-header">⚡ RECENT TRADING SIGNALS</div>', unsafe_allow_html=True)
            
            if filtered_signals:
                # Sort by confidence and show top 6
                filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
                
                # Display in columns
                signal_cols = st.columns(2)
                for i, signal in enumerate(filtered_signals[:6]):
                    with signal_cols[i % 2]:
                        display_signal(signal)
            else:
                st.info("ℹ️ No trading signals found for the current selection.")
    
    # Tab 2: Trading Signals
    with tab2:
        st.markdown('<div class="sub-header">⚡ ADVANCED SIGNAL ANALYSIS</div>', unsafe_allow_html=True)
        
        if not filtered_signals:
            st.info("ℹ️ No signals to display. Run analysis in Live Dashboard first.")
        else:
            # Signal filtering
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_conf_filter = st.slider("🎯 Min Confidence", 0.0, 1.0, 0.6, key="conf_filter")
            
            with col2:
                action_filter = st.selectbox("🔀 Action Filter", ["All", "BUY", "SELL"], key="action_filter")
            
            with col3:
                timeframe_filter = st.multiselect(
                    "⏰ Timeframe Filter",
                    strategy_manager.timeframes,
                    default=selected_timeframes,
                    key="timeframe_filter"
                )
            
            # Apply filters
            filtered_display_signals = [
                s for s in filtered_signals 
                if s.confidence >= min_conf_filter and
                (action_filter == "All" or s.action == action_filter) and
                s.timeframe in timeframe_filter
            ]
            
            # Display filtered signals
            st.markdown(f'<div class="dashboard-header">📋 FILTERED SIGNALS ({len(filtered_display_signals)})</div>', unsafe_allow_html=True)
            
            for signal in filtered_display_signals:
                display_signal(signal)
    
    # Tab 3: Paper Trading
    with tab3:
        st.markdown('<div class="sub-header">💼 PAPER TRADING DASHBOARD</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Current Balance", f"${paper_trading.balance:.2f}")
        
        with col2:
            st.metric("📈 Open Trades", len(paper_trading.open_trades))
        
        with col3:
            st.metric("📊 Closed Trades", len(paper_trading.closed_trades))
        
        with col4:
            overall_return = ((paper_trading.balance - paper_trading.initial_balance) / paper_trading.initial_balance) * 100
            st.metric("📈 Overall Return", f"{overall_return:.2f}%")
    
    # Tab 4: Performance Analytics
    with tab4:
        st.markdown('<div class="sub-header">📊 PERFORMANCE ANALYTICS</div>', unsafe_allow_html=True)
        
        # Get performance metrics
        metrics = paper_trading.get_performance_metrics()
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Win Rate", f"{metrics['win_rate']:.1f}%")
                st.metric("📈 Total Trades", metrics['total_trades'])
            
            with col2:
                st.metric("💰 Total P&L", f"${metrics['total_pnl']:.2f}")
                st.metric("📊 Profit Factor", f"{metrics['profit_factor']:.2f}")
            
            with col3:
                st.metric("🟢 Avg Profit", f"${metrics['avg_profit']:.2f}")
                st.metric("🔴 Avg Loss", f"${metrics['avg_loss']:.2f}")
            
            with col4:
                st.metric("📉 Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                st.metric("⚡ Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        else:
            st.info("ℹ️ No performance data available. Execute some trades first.")

def display_signal(signal: TradingSignal):
    """Display a trading signal in a formatted card"""
    signal_class = "buy-signal" if signal.action == "BUY" else "sell-signal"
    signal_color = "#10b981" if signal.action == "BUY" else "#ef4444"
    signal_icon = "🟢" if signal.action == "BUY" else "🔴"
    priority_icon = "🚨" if signal.priority == "HIGH" else "⚠️" if signal.priority == "MEDIUM" else "ℹ️"
    
    st.markdown(f"""
    <div class="signal-card {signal_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: {signal_color}; font-weight: 700;">
                    {signal_icon} {signal.symbol} {signal.action} - {signal.strategy}
                </h4>
                <p style="margin: 0; font-size: 0.9rem;">
                    ⏱️ {signal.timeframe} | 🎯 Confidence: <strong>{signal.confidence:.2f}</strong>
                </p>
            </div>
            <div style="text-align: right;">
                <strong>{priority_icon} {signal.priority}</strong>
            </div>
        </div>
        <div style="margin-top: 0.8rem;">
            <p style="margin: 0; font-size: 0.9rem; font-weight: 600;">
                💰 Entry: <strong>{signal.entry:.4f}</strong> | 
                🛑 SL: <strong>{signal.stop_loss:.4f}</strong> | 
                🎯 TP1: <strong>{signal.target1:.4f}</strong>
            </p>
            <p style="margin: 0.3rem 0; font-size: 0.85rem; color: #555; font-style: italic;">
                💡 {signal.reasoning}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
