"""
Streamlit Trading Bot for Commodities & Cryptocurrencies
Supports: USOIL, GOLD, BTC, SOLANA, XRP, ETH
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_ta as ta  # Alternative to TA-Lib
from datetime import datetime, timedelta
import sqlite3
import json
import time
import threading
import queue
import warnings
import requests
from typing import Dict, List, Tuple, Optional
import logging
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multi-Asset Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .positive {
        color: #00C853;
    }
    .negative {
        color: #FF5252;
    }
    .signal-buy {
        background-color: #C8E6C9;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #FFCDD2;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DataHandler:
    """Handles data collection for multiple asset types"""
    
    # Asset mapping
    ASSET_SYMBOLS = {
        'USOIL': 'CL=F',  # Crude Oil Futures
        'GOLD': 'GC=F',   # Gold Futures
        'BTC': 'BTC-USD',
        'SOLANA': 'SOL-USD',
        'XRP': 'XRP-USD',
        'ETH': 'ETH-USD'
    }
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        '15m': '15m',
        '1h': '60m',
        '4h': '60m',
        '1d': '1d'
    }
    
    def __init__(self, timeframe: str = '15m', lookback_days: int = 30):
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.data_cache = {}
        
    def fetch_data(self, asset_name: str) -> pd.DataFrame:
        """Fetch historical data for an asset"""
        try:
            symbol = self.ASSET_SYMBOLS.get(asset_name, asset_name)
            yf_timeframe = self.TIMEFRAME_MAP.get(self.timeframe, '15m')
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_timeframe
            )
            
            if data.empty:
                logger.warning(f"No data found for {asset_name} ({symbol})")
                return pd.DataFrame()
            
            # Rename columns to match our expected format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Calculate all indicators
            data = self.calculate_indicators(data, asset_name)
            self.data_cache[asset_name] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {asset_name}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        """Calculate all technical indicators using pandas_ta"""
        if df.empty:
            return df
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Moving Averages
        df['EMA8'] = ta.ema(close, length=8)
        df['EMA21'] = ta.ema(close, length=21)
        df['EMA50'] = ta.ema(close, length=50)
        df['SMA20'] = ta.sma(close, length=20)
        
        # VWAP (daily)
        typical_price = (high + low + close) / 3
        df['VWAP'] = ta.vwap(high, low, close, volume)
        
        # Bollinger Bands
        bb = ta.bbands(close, length=20, std=2)
        df['BB_upper'] = bb['BBU_20_2.0']
        df['BB_middle'] = bb['BBM_20_2.0']
        df['BB_lower'] = bb['BBL_20_2.0']
        
        # RSI
        df['RSI'] = ta.rsi(close, length=14)
        
        # MACD
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # ADX
        df['ADX'] = ta.adx(high, low, close, length=14)['ADX_14']
        
        # ATR for stop loss calculation
        df['ATR'] = ta.atr(high, low, close, length=14)
        
        # Volume indicators
        df['Volume_SMA20'] = ta.sma(volume, length=20)
        df['Volume_Ratio'] = volume / df['Volume_SMA20']
        
        # Support and Resistance
        df['Support'] = low.rolling(window=20).min()
        df['Resistance'] = high.rolling(window=20).max()
        
        # Price position indicators
        df['Price_VWAP_Ratio'] = close / df['VWAP']
        df['Price_BB_Position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Trend indicators
        df['EMA_Trend'] = np.where(
            (df['EMA8'] > df['EMA21']) & (df['EMA21'] > df['EMA50']), 1,
            np.where(
                (df['EMA8'] < df['EMA21']) & (df['EMA21'] < df['EMA50']), -1, 0
            )
        )
        
        return df
    
    def get_latest_data(self, asset_name: str) -> Dict:
        """Get latest data point with all indicators"""
        if asset_name not in self.data_cache:
            self.fetch_data(asset_name)
        
        if asset_name in self.data_cache and not self.data_cache[asset_name].empty:
            latest = self.data_cache[asset_name].iloc[-1].to_dict()
            latest['asset'] = asset_name
            latest['symbol'] = self.ASSET_SYMBOLS.get(asset_name, asset_name)
            latest['timestamp'] = datetime.now()
            return latest
        
        return {}


class SignalGenerator:
    """Generates trading signals based on strategy rules"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        
    def _initialize_strategies(self) -> Dict:
        """Initialize all trading strategies with weights"""
        return {
            'trend_following': [
                {
                    'name': 'EMA_VWAP_Confluence',
                    'type': 'BUY',
                    'weight': 3,
                    'conditions': [
                        lambda d: d['close'] > d.get('EMA8', 0),
                        lambda d: d.get('EMA8', 0) > d.get('EMA21', 0),
                        lambda d: d.get('EMA21', 0) > d.get('EMA50', 0),
                        lambda d: d['close'] > d.get('VWAP', 0),
                        lambda d: d.get('ADX', 0) > 25,
                        lambda d: d.get('Volume_Ratio', 0) > 1.2
                    ]
                },
                {
                    'name': 'MACD_Momentum',
                    'type': 'BUY',
                    'weight': 2,
                    'conditions': [
                        lambda d: d.get('MACD', 0) > d.get('MACD_signal', 0),
                        lambda d: d.get('EMA8', 0) > d.get('EMA21', 0),
                        lambda d: d['close'] > d.get('VWAP', 0),
                        lambda d: d.get('Volume_Ratio', 0) > 1.0
                    ]
                }
            ],
            'mean_reversion': [
                {
                    'name': 'RSI_Oversold',
                    'type': 'BUY',
                    'weight': 2,
                    'conditions': [
                        lambda d: 25 < d.get('RSI', 50) < 35,
                        lambda d: d['close'] > d.get('Support', 0),
                        lambda d: d.get('Volume_Ratio', 0) > 0.8
                    ]
                },
                {
                    'name': 'Bollinger_Reversion',
                    'type': 'BUY',
                    'weight': 3,
                    'conditions': [
                        lambda d: d['close'] <= d.get('BB_lower', 0) * 1.01,
                        lambda d: d.get('RSI', 50) < 40,
                        lambda d: d.get('Volume_Ratio', 0) > 1.0
                    ]
                }
            ],
            'breakout': [
                {
                    'name': 'Volume_Breakout_BUY',
                    'type': 'BUY',
                    'weight': 4,
                    'conditions': [
                        lambda d: d.get('Volume_Ratio', 0) > 1.8,
                        lambda d: d['close'] > d.get('Resistance', 0),
                        lambda d: d.get('RSI', 50) < 70,
                        lambda d: d.get('ADX', 0) > 20
                    ]
                },
                {
                    'name': 'Support_Resistance_Breakout_SELL',
                    'type': 'SELL',
                    'weight': 4,
                    'conditions': [
                        lambda d: d.get('Volume_Ratio', 0) > 1.8,
                        lambda d: d['close'] < d.get('Support', 0),
                        lambda d: d.get('RSI', 50) > 30,
                        lambda d: d.get('ADX', 0) > 20
                    ]
                }
            ],
            'bearish': [
                {
                    'name': 'EMA_VWAP_Downtrend',
                    'type': 'SELL',
                    'weight': 3,
                    'conditions': [
                        lambda d: d['close'] < d.get('EMA8', 0),
                        lambda d: d.get('EMA8', 0) < d.get('EMA21', 0),
                        lambda d: d.get('EMA21', 0) < d.get('EMA50', 0),
                        lambda d: d['close'] < d.get('VWAP', 0),
                        lambda d: d.get('ADX', 0) > 25,
                        lambda d: d.get('Volume_Ratio', 0) > 1.2
                    ]
                },
                {
                    'name': 'RSI_Overbought',
                    'type': 'SELL',
                    'weight': 2,
                    'conditions': [
                        lambda d: 65 < d.get('RSI', 50) < 75,
                        lambda d: d['close'] < d.get('Resistance', 0),
                        lambda d: d.get('Volume_Ratio', 0) > 0.8
                    ]
                },
                {
                    'name': 'Bollinger_Rejection',
                    'type': 'SELL',
                    'weight': 3,
                    'conditions': [
                        lambda d: d['close'] >= d.get('BB_upper', 0) * 0.99,
                        lambda d: d.get('RSI', 50) > 60,
                        lambda d: d.get('Volume_Ratio', 0) > 1.0
                    ]
                }
            ]
        }
    
    def generate_signal(self, data: Dict) -> Dict:
        """Generate trading signal based on all strategies"""
        if not data or 'close' not in data:
            return {'signal_type': 'HOLD', 'signal_score': 0, 'triggered_strategies': []}
        
        buy_score = 0
        sell_score = 0
        triggered_strategies = []
        
        # Check all strategies
        for category, strategies in self.strategies.items():
            for strategy in strategies:
                try:
                    # Check if all conditions are met
                    conditions_met = all(condition(data) for condition in strategy['conditions'])
                    
                    if conditions_met:
                        weight = strategy['weight']
                        strategy_name = f"{strategy['name']}"
                        
                        if strategy['type'] == 'BUY':
                            buy_score += weight
                            triggered_strategies.append(f"{strategy_name}(+{weight})")
                        elif strategy['type'] == 'SELL':
                            sell_score += weight
                            triggered_strategies.append(f"{strategy_name}(+{weight})")
                except Exception as e:
                    continue
        
        # Determine final signal
        signal_type = 'HOLD'
        signal_score = 0
        
        # Strong buy if score >= 6 and significantly higher than sell
        if buy_score >= 6 and buy_score > sell_score + 2:
            signal_type = 'STRONG_BUY'
            signal_score = buy_score
        elif buy_score >= 4 and buy_score > sell_score:
            signal_type = 'BUY'
            signal_score = buy_score
        elif sell_score >= 6 and sell_score > buy_score + 2:
            signal_type = 'STRONG_SELL'
            signal_score = sell_score
        elif sell_score >= 4 and sell_score > buy_score:
            signal_type = 'SELL'
            signal_score = sell_score
        elif buy_score >= 3 and sell_score >= 3:
            signal_type = 'CONFLICT'
            signal_score = max(buy_score, sell_score)
        
        return {
            'signal_type': signal_type,
            'signal_score': signal_score,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'triggered_strategies': triggered_strategies[:5],  # Limit to top 5
            'timestamp': datetime.now(),
            'asset': data.get('asset', ''),
            'price': data.get('close', 0)
        }


class RiskManager:
    """Manages risk, position sizing, and stop/target calculations"""
    
    def __init__(self, capital: float = 10000, max_risk_per_trade: float = 1.0):
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, asset_type: str = 'crypto') -> float:
        """Calculate position size based on risk and asset type"""
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        # Adjust max risk based on asset type
        max_risk_pct = self.max_risk_per_trade
        if asset_type == 'commodity':
            max_risk_pct = min(max_risk_pct, 0.5)  # Lower risk for commodities
        
        max_risk_amount = self.capital * (max_risk_pct / 100)
        position_size = max_risk_amount / risk_per_share
        
        # Apply position limits based on asset
        if asset_type == 'crypto':
            min_position_value = 50  # $50 minimum for crypto
        else:
            min_position_value = 100  # $100 minimum for commodities
        
        min_shares = min_position_value / entry_price if entry_price > 0 else 0
        
        return max(min_shares, position_size)
    
    def calculate_stop_target(self, entry_price: float, signal_type: str, 
                             atr: float, support: float, resistance: float,
                             asset_volatility: float = 1.0) -> Tuple[float, float, float]:
        """Calculate stop loss and take profit with volatility adjustment"""
        
        # Adjust ATR based on asset volatility
        adjusted_atr = atr * asset_volatility
        atr_stop = 1.5 * adjusted_atr
        
        if signal_type in ['BUY', 'STRONG_BUY']:
            # Stop loss: support or ATR-based
            stop_loss = min(support, entry_price - atr_stop) if support > 0 else entry_price - atr_stop
            
            # Take profit: 2.5:1 reward ratio minimum
            min_reward = 2.5 * abs(entry_price - stop_loss)
            profit_target = max(resistance, entry_price + min_reward) if resistance > 0 else entry_price + min_reward
            
        else:  # SELL or STRONG_SELL
            stop_loss = max(resistance, entry_price + atr_stop) if resistance > 0 else entry_price + atr_stop
            
            min_reward = 2.5 * abs(entry_price - stop_loss)
            profit_target = min(support, entry_price - min_reward) if support > 0 else entry_price - min_reward
        
        # Calculate actual risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(profit_target - entry_price)
        
        if risk > 0:
            actual_rr = reward / risk
        else:
            actual_rr = 0
        
        # Ensure minimum R:R
        if actual_rr < 2.5:
            if signal_type in ['BUY', 'STRONG_BUY']:
                profit_target = entry_price + (2.5 * risk)
            else:
                profit_target = entry_price - (2.5 * risk)
            actual_rr = 2.5
        
        return stop_loss, profit_target, actual_rr


class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_handler = DataHandler(
            timeframe=config.get('timeframe', '15m'),
            lookback_days=config.get('lookback_days', 30)
        )
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(
            capital=config.get('capital', 10000),
            max_risk_per_trade=config.get('max_risk_per_trade', 1.0)
        )
        
        self.assets = config.get('assets', ['BTC', 'ETH', 'USOIL', 'GOLD', 'XRP', 'SOLANA'])
        self.running = False
        self.signals = {}
        self.trades = []
        self.performance = {}
        
        # Load existing trades if any
        self.load_trades()
    
    def load_trades(self):
        """Load trades from file"""
        try:
            if os.path.exists('trades.json'):
                with open('trades.json', 'r') as f:
                    self.trades = json.load(f)
        except:
            self.trades = []
    
    def save_trades(self):
        """Save trades to file"""
        try:
            with open('trades.json', 'w') as f:
                json.dump(self.trades, f, default=str)
        except:
            pass
    
    def analyze_asset(self, asset_name: str) -> Optional[Dict]:
        """Run analysis for a single asset"""
        try:
            # Fetch latest data
            data = self.data_handler.get_latest_data(asset_name)
            
            if not data or 'close' not in data:
                return None
            
            # Generate signal
            signal = self.signal_generator.generate_signal(data)
            
            # Determine asset type for risk management
            asset_type = 'crypto' if asset_name in ['BTC', 'ETH', 'XRP', 'SOLANA'] else 'commodity'
            
            # Calculate volatility multiplier
            volatility_multiplier = 1.5 if asset_type == 'crypto' else 1.0
            
            # Calculate entry, stop, and target if there's a signal
            if signal['signal_type'] != 'HOLD':
                entry_price = data['close']
                stop_loss, take_profit, rr_ratio = self.risk_manager.calculate_stop_target(
                    entry_price=entry_price,
                    signal_type=signal['signal_type'],
                    atr=data.get('ATR', entry_price * 0.02),
                    support=data.get('Support', entry_price * 0.95),
                    resistance=data.get('Resistance', entry_price * 1.05),
                    asset_volatility=volatility_multiplier
                )
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    asset_type=asset_type
                )
                
                signal.update({
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size,
                    'risk_reward': rr_ratio,
                    'asset_type': asset_type,
                    'indicators': {
                        'RSI': data.get('RSI', 0),
                        'MACD': data.get('MACD', 0),
                        'ADX': data.get('ADX', 0),
                        'Volume_Ratio': data.get('Volume_Ratio', 0),
                        'BB_Position': data.get('Price_BB_Position', 0.5)
                    }
                })
            
            self.signals[asset_name] = signal
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {asset_name}: {str(e)}")
            return None
    
    def execute_trade(self, asset: str, signal: Dict, action: str = 'paper'):
        """Execute a trade (paper or live)"""
        if signal['signal_type'] in ['HOLD', 'CONFLICT']:
            return None
        
        trade = {
            'id': f"{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'asset': asset,
            'timestamp': datetime.now().isoformat(),
            'signal_type': signal['signal_type'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'position_size': signal['position_size'],
            'risk_reward': signal['risk_reward'],
            'signal_score': signal['signal_score'],
            'status': 'OPEN',
            'action': action,
            'pnl': 0,
            'pnl_percent': 0
        }
        
        self.trades.append(trade)
        self.save_trades()
        
        logger.info(f"Executed {action} trade for {asset}: {signal['signal_type']} @ ${signal['entry_price']:.2f}")
        
        return trade
    
    def update_trades(self, current_prices: Dict):
        """Update open trades with current prices"""
        for trade in self.trades:
            if trade['status'] == 'OPEN' and trade['asset'] in current_prices:
                current_price = current_prices[trade['asset']]
                entry_price = trade['entry_price']
                position_size = trade['position_size']
                
                # Calculate P&L
                if trade['signal_type'] in ['BUY', 'STRONG_BUY']:
                    pnl = (current_price - entry_price) * position_size
                else:  # SELL or STRONG_SELL
                    pnl = (entry_price - current_price) * position_size
                
                pnl_percent = (pnl / (entry_price * position_size)) * 100
                
                trade['current_price'] = current_price
                trade['pnl'] = pnl
                trade['pnl_percent'] = pnl_percent
                
                # Check stop loss and take profit
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']
                
                if (trade['signal_type'] in ['BUY', 'STRONG_BUY'] and current_price <= stop_loss) or \
                   (trade['signal_type'] in ['SELL', 'STRONG_SELL'] and current_price >= stop_loss):
                    trade['status'] = 'CLOSED'
                    trade['exit_reason'] = 'STOP_LOSS'
                    trade['exit_price'] = stop_loss
                elif (trade['signal_type'] in ['BUY', 'STRONG_BUY'] and current_price >= take_profit) or \
                     (trade['signal_type'] in ['SELL', 'STRONG_SELL'] and current_price <= take_profit):
                    trade['status'] = 'CLOSED'
                    trade['exit_reason'] = 'TAKE_PROFIT'
                    trade['exit_price'] = take_profit
        
        self.save_trades()
    
    def run_analysis(self):
        """Run analysis for all assets"""
        self.signals = {}
        current_prices = {}
        
        for asset in self.assets:
            signal = self.analyze_asset(asset)
            if signal and 'price' in signal:
                current_prices[asset] = signal['price']
        
        # Update existing trades
        self.update_trades(current_prices)
        
        return self.signals
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        open_trades = [t for t in self.trades if t['status'] == 'OPEN']
        
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t.get('pnl', 0) for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        open_pnl = sum(t.get('pnl', 0) for t in open_trades)
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': total_pnl,
            'open_pnl': open_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'open_trades': len(open_trades)
        }


class StreamlitApp:
    """Streamlit application for the trading bot"""
    
    def __init__(self):
        self.bot = None
        self.analysis_thread = None
        self.last_update = None
        
    def initialize_bot(self, config: Dict):
        """Initialize the trading bot"""
        self.bot = TradingBot(config)
        st.session_state.bot_initialized = True
        st.session_state.config = config
        
    def run_analysis_in_thread(self):
        """Run analysis in a separate thread"""
        if self.bot:
            self.bot.run_analysis()
            self.last_update = datetime.now()
    
    def display_header(self):
        """Display application header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<h1 class="main-header">📈 Multi-Asset Trading Bot</h1>', unsafe_allow_html=True)
            st.markdown("**Real-time trading signals for Commodities & Cryptocurrencies**")
            
            if self.last_update:
                st.caption(f"Last updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def display_sidebar(self):
        """Display sidebar with configuration"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/bitcoin--v1.png", width=100)
            st.title("Configuration")
            
            # Trading Parameters
            st.subheader("Trading Parameters")
            capital = st.number_input("Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
            max_risk = st.slider("Max Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
            
            # Asset Selection
            st.subheader("Assets to Trade")
            assets = []
            
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("BTC", True): assets.append("BTC")
                if st.checkbox("ETH", True): assets.append("ETH")
                if st.checkbox("USOIL", True): assets.append("USOIL")
            with col2:
                if st.checkbox("GOLD", True): assets.append("GOLD")
                if st.checkbox("XRP", True): assets.append("XRP")
                if st.checkbox("SOLANA", True): assets.append("SOLANA")
            
            # Timeframe
            st.subheader("Analysis Timeframe")
            timeframe = st.selectbox(
                "Select Timeframe",
                ["15m", "1h", "4h", "1d"],
                index=0
            )
            
            # Trading Mode
            st.subheader("Trading Mode")
            trading_mode = st.radio(
                "Select Mode",
                ["Paper Trading", "Live Trading"],
                index=0
            )
            
            # Action Buttons
            st.subheader("Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Initialize Bot", use_container_width=True):
                    config = {
                        'capital': capital,
                        'max_risk_per_trade': max_risk,
                        'assets': assets,
                        'timeframe': timeframe,
                        'lookback_days': 30
                    }
                    self.initialize_bot(config)
                    st.success("Trading bot initialized!")
            
            with col2:
                if st.button("🔄 Run Analysis", use_container_width=True):
                    if self.bot:
                        self.run_analysis_in_thread()
                        st.rerun()
                    else:
                        st.warning("Please initialize the bot first")
            
            # Auto-refresh
            st.subheader("Auto-Refresh")
            auto_refresh = st.checkbox("Enable Auto-Refresh", False)
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60, 10)
                time.sleep(refresh_interval)
                st.rerun()
            
            # Performance Summary
            if self.bot and hasattr(self.bot, 'trades'):
                st.divider()
                st.subheader("Performance Summary")
                metrics = self.bot.get_performance_metrics()
                
                st.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                st.metric("Open Trades", metrics['open_trades'])
    
    def display_signals_dashboard(self):
        """Display trading signals dashboard"""
        st.header("📊 Trading Signals Dashboard")
        
        if not self.bot or not self.bot.signals:
            st.info("No signals available. Run analysis to generate signals.")
            return
        
        # Create metrics row
        cols = st.columns(len(self.bot.signals))
        
        for idx, (asset, signal) in enumerate(self.bot.signals.items()):
            with cols[idx]:
                # Determine color and icon based on signal
                if signal['signal_type'] in ['STRONG_BUY', 'BUY']:
                    color = "#00C853"
                    icon = "🟢"
                elif signal['signal_type'] in ['STRONG_SELL', 'SELL']:
                    color = "#FF5252"
                    icon = "🔴"
                else:
                    color = "#FFC107"
                    icon = "🟡"
                
                # Create metric card
                st.markdown(f"""
                <div style='border: 2px solid {color}; padding: 15px; border-radius: 10px; text-align: center;'>
                    <h3>{icon} {asset}</h3>
                    <h4 style='color: {color};'>{signal['signal_type']}</h4>
                    <p>Score: <b>{signal['signal_score']}</b></p>
                    <p>Price: <b>${signal.get('price', 0):.2f}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show action buttons
                if signal['signal_type'] != 'HOLD':
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📈 Buy", key=f"buy_{asset}", use_container_width=True):
                            trade = self.bot.execute_trade(asset, signal, 'paper')
                            if trade:
                                st.success(f"Paper trade executed for {asset}")
                    with col2:
                        if st.button(f"📉 Sell", key=f"sell_{asset}", use_container_width=True):
                            # For sell signals, we would short
                            st.info("Short selling not implemented in paper trading")
        
        # Detailed signals table
        st.subheader("Detailed Signal Analysis")
        
        signals_data = []
        for asset, signal in self.bot.signals.items():
            signals_data.append({
                'Asset': asset,
                'Signal': signal['signal_type'],
                'Score': signal['signal_score'],
                'Price': f"${signal.get('price', 0):.2f}",
                'Buy Score': signal['buy_score'],
                'Sell Score': signal['sell_score'],
                'Strategies': ', '.join(signal['triggered_strategies'][:3]),
                'RSI': signal.get('indicators', {}).get('RSI', 0),
                'Volume Ratio': signal.get('indicators', {}).get('Volume_Ratio', 0)
            })
        
        if signals_data:
            df_signals = pd.DataFrame(signals_data)
            st.dataframe(df_signals, use_container_width=True)
    
    def display_charts(self):
        """Display price charts for selected assets"""
        st.header("📈 Price Charts")
        
        if not self.bot:
            return
        
        # Let user select asset to chart
        selected_asset = st.selectbox("Select Asset to Chart", self.bot.assets)
        
        if selected_asset and selected_asset in self.bot.data_handler.data_cache:
            df = self.bot.data_handler.data_cache[selected_asset]
            
            if not df.empty:
                # Create candlestick chart
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=(f'{selected_asset} Price', 'RSI', 'Volume')
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
                
                # Add EMAs
                for ema_period in [8, 21, 50]:
                    if f'EMA{ema_period}' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[f'EMA{ema_period}'],
                                name=f'EMA{ema_period}',
                                line=dict(width=1)
                            ),
                            row=1, col=1
                        )
                
                # Add Bollinger Bands
                if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['BB_upper'],
                            name='BB Upper',
                            line=dict(width=1, color='gray'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['BB_lower'],
                            name='BB Lower',
                            line=dict(width=1, color='gray'),
                            fill='tonexty',
                            fillcolor='rgba(128, 128, 128, 0.1)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                
                # RSI
                if 'RSI' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['RSI'],
                            name='RSI',
                            line=dict(color='purple', width=1)
                        ),
                        row=2, col=1
                    )
                    # Add RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Volume
                colors = ['red' if row['close'] < row['open'] else 'green' 
                         for _, row in df.iterrows()]
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors
                    ),
                    row=3, col=1
                )
                
                # Update layout
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                fig.update_xaxes(title_text="Date", row=3, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)
                fig.update_yaxes(title_text="Volume", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
    
    def display_trades(self):
        """Display active and historical trades"""
        st.header("💼 Trade Management")
        
        if not self.bot or not self.bot.trades:
            st.info("No trades yet. Signals will appear here when executed.")
            return
        
        # Tabs for open and closed trades
        tab1, tab2 = st.tabs(["📊 Open Trades", "📋 Trade History"])
        
        with tab1:
            open_trades = [t for t in self.bot.trades if t['status'] == 'OPEN']
            
            if open_trades:
                # Create DataFrame for display
                trades_data = []
                for trade in open_trades:
                    trades_data.append({
                        'ID': trade['id'][-8:],
                        'Asset': trade['asset'],
                        'Type': trade['signal_type'],
                        'Entry Price': f"${trade['entry_price']:.2f}",
                        'Current Price': f"${trade.get('current_price', trade['entry_price']):.2f}",
                        'Stop Loss': f"${trade['stop_loss']:.2f}",
                        'Take Profit': f"${trade['take_profit']:.2f}",
                        'Position Size': f"{trade['position_size']:.4f}",
                        'P&L': f"${trade.get('pnl', 0):.2f}",
                        'P&L %': f"{trade.get('pnl_percent', 0):.2f}%",
                        'R:R': f"{trade['risk_reward']:.2f}:1"
                    })
                
                df_trades = pd.DataFrame(trades_data)
                st.dataframe(df_trades, use_container_width=True)
                
                # Close trade button
                st.subheader("Close Trade")
                trade_ids = [t['id'] for t in open_trades]
                selected_trade = st.selectbox("Select Trade to Close", trade_ids)
                
                if st.button("Close Selected Trade", type="primary"):
                    for trade in self.bot.trades:
                        if trade['id'] == selected_trade:
                            trade['status'] = 'CLOSED'
                            trade['exit_reason'] = 'MANUAL'
                            trade['exit_price'] = trade.get('current_price', trade['entry_price'])
                            self.bot.save_trades()
                            st.success(f"Trade {selected_trade[-8:]} closed manually")
                            st.rerun()
            else:
                st.info("No open trades")
        
        with tab2:
            closed_trades = [t for t in self.bot.trades if t['status'] == 'CLOSED']
            
            if closed_trades:
                # Create DataFrame for display
                trades_data = []
                for trade in closed_trades[-20:]:  # Show last 20 trades
                    pnl = trade.get('pnl', 0)
                    pnl_color = "positive" if pnl > 0 else "negative"
                    
                    trades_data.append({
                        'ID': trade['id'][-8:],
                        'Asset': trade['asset'],
                        'Type': trade['signal_type'],
                        'Entry': f"${trade['entry_price']:.2f}",
                        'Exit': f"${trade.get('exit_price', 0):.2f}",
                        'P&L': f"${pnl:.2f}",
                        'P&L %': f"{trade.get('pnl_percent', 0):.2f}%",
                        'Result': 'WIN' if pnl > 0 else 'LOSS',
                        'Reason': trade.get('exit_reason', 'N/A'),
                        'Duration': 'N/A'
                    })
                
                df_closed = pd.DataFrame(trades_data)
                st.dataframe(df_closed, use_container_width=True)
                
                # Performance summary
                st.subheader("Performance Summary")
                metrics = self.bot.get_performance_metrics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", metrics['total_trades'])
                with col2:
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                with col3:
                    st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
                with col4:
                    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            else:
                st.info("No closed trades yet")
    
    def display_asset_details(self):
        """Display detailed analysis for each asset"""
        st.header("🔍 Asset Details")
        
        if not self.bot:
            return
        
        # Create tabs for each asset
        tabs = st.tabs(self.bot.assets)
        
        for idx, asset in enumerate(self.bot.assets):
            with tabs[idx]:
                if asset in self.bot.signals:
                    signal = self.bot.signals[asset]
                    
                    # Display signal information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${signal.get('price', 0):.2f}")
                    with col2:
                        st.metric("Signal", signal['signal_type'])
                    with col3:
                        st.metric("Signal Score", signal['signal_score'])
                    
                    # Display indicators
                    st.subheader("Technical Indicators")
                    
                    if 'indicators' in signal:
                        indicators = signal['indicators']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            rsi = indicators.get('RSI', 50)
                            rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "orange"
                            st.metric("RSI", f"{rsi:.1f}", delta_color="off")
                            st.progress(min(max(rsi / 100, 0), 1))
                        
                        with col2:
                            vol_ratio = indicators.get('Volume_Ratio', 1)
                            st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
                        
                        with col3:
                            adx = indicators.get('ADX', 0)
                            st.metric("ADX", f"{adx:.1f}")
                        
                        with col4:
                            bb_pos = indicators.get('BB_Position', 0.5)
                            st.metric("BB Position", f"{bb_pos:.2f}")
                    
                    # Display triggered strategies
                    if signal['triggered_strategies']:
                        st.subheader("Triggered Strategies")
                        for strategy in signal['triggered_strategies']:
                            st.write(f"• {strategy}")
                    
                    # Display trade parameters if signal exists
                    if signal['signal_type'] != 'HOLD':
                        st.subheader("Trade Parameters")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Entry Price", f"${signal.get('entry_price', 0):.2f}")
                        with col2:
                            st.metric("Stop Loss", f"${signal.get('stop_loss', 0):.2f}")
                        with col3:
                            st.metric("Take Profit", f"${signal.get('take_profit', 0):.2f}")
                        
                        st.metric("Risk/Reward Ratio", f"{signal.get('risk_reward', 0):.2f}:1")
    
    def run(self):
        """Main Streamlit application runner"""
        self.display_header()
        self.display_sidebar()
        
        if hasattr(st.session_state, 'bot_initialized') and st.session_state.bot_initialized:
            # Display main content in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Dashboard",
                "📈 Charts",
                "💼 Trades",
                "🔍 Analysis"
            ])
            
            with tab1:
                self.display_signals_dashboard()
            
            with tab2:
                self.display_charts()
            
            with tab3:
                self.display_trades()
            
            with tab4:
                self.display_asset_details()
        else:
            # Show welcome screen
            st.info("👈 Please configure the bot in the sidebar and click 'Initialize Bot' to get started.")
            
            # Display features
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("📈 Multi-Asset Support")
                st.write("Trade 6 major assets:")
                st.write("• USOIL (Crude Oil)")
                st.write("• GOLD")
                st.write("• BTC (Bitcoin)")
                st.write("• ETH (Ethereum)")
                st.write("• XRP (Ripple)")
                st.write("• SOLANA")
            
            with col2:
                st.subheader("⚙️ Advanced Strategies")
                st.write("Multiple trading strategies:")
                st.write("• Trend Following")
                st.write("• Mean Reversion")
                st.write("• Breakout Trading")
                st.write("• Volume Analysis")
                st.write("• Risk Management")
            
            with col3:
                st.subheader("🛡️ Risk Management")
                st.write("Professional risk controls:")
                st.write("• ATR-based Stop Loss")
                st.write("• Position Sizing")
                st.write("• 2.5:1 Minimum R:R")
                st.write("• Max Risk per Trade")
                st.write("• Paper Trading Mode")


def main():
    """Main function to run the Streamlit app"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
