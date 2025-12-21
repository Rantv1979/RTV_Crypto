# =============================================
# RANTV COMPLETE ALGORITHMIC TRADING SYSTEM
# WITH REAL TRADING & PAPER TRADING MODES
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
import requests
from scipy import stats
import warnings
import hashlib
import json
import pickle
from pathlib import Path
import threading
import queue
import schedule
import ccxt
import alpaca_trade_api as tradeapi
from binance.client import Client
import asyncio
import websockets
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
INITIAL_CAPITAL = 100000.0
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

# Exchange Mapping
EXCHANGE_MAPPING = {
    "BTC-USD": "binance",
    "ETH-USD": "binance",
    "SOL-USD": "binance",
    "XRP-USD": "binance",
    "BNB-USD": "binance",
    "AAPL": "alpaca",
    "MSFT": "alpaca",
    "GOOGL": "alpaca",
    "AMZN": "alpaca",
    "TSLA": "alpaca",
    "EURUSD=X": "oanda",
    "GBPUSD=X": "oanda",
    "USDJPY=X": "oanda",
    "GC=F": "alpaca",
    "SI=F": "alpaca"
}

# =============================================
# EXCHANGE INTEGRATION MODULES
# =============================================

class ExchangeConnector:
    """Unified exchange connector for multiple brokers"""
    
    def __init__(self, mode="paper"):
        self.mode = mode  # 'paper' or 'live'
        self.connections = {}
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize all exchange connections"""
        # Load API keys from environment or config
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self):
        """Load API keys from configuration"""
        # In production, load from secure storage
        return {
            'alpaca': {
                'api_key': st.secrets.get('ALPACA_API_KEY', ''),
                'api_secret': st.secrets.get('ALPACA_SECRET_KEY', ''),
                'paper': self.mode == 'paper'
            },
            'binance': {
                'api_key': st.secrets.get('BINANCE_API_KEY', ''),
                'api_secret': st.secrets.get('BINANCE_SECRET_KEY', '')
            },
            'oanda': {
                'api_key': st.secrets.get('OANDA_API_KEY', ''),
                'account_id': st.secrets.get('OANDA_ACCOUNT_ID', '')
            }
        }
    
    def connect_alpaca(self):
        """Connect to Alpaca trading API"""
        try:
            if self.mode == 'paper':
                base_url = 'https://paper-api.alpaca.markets'
            else:
                base_url = 'https://api.alpaca.markets'
            
            api = tradeapi.REST(
                self.api_keys['alpaca']['api_key'],
                self.api_keys['alpaca']['api_secret'],
                base_url,
                api_version='v2'
            )
            
            # Test connection
            account = api.get_account()
            st.success(f"✅ Alpaca Connected (Paper: {self.mode == 'paper'})")
            return api
        except Exception as e:
            st.error(f"❌ Alpaca Connection Failed: {str(e)}")
            return None
    
    def connect_binance(self):
        """Connect to Binance API"""
        try:
            client = Client(
                self.api_keys['binance']['api_key'],
                self.api_keys['binance']['api_secret']
            )
            
            # Test connection
            account_info = client.get_account()
            st.success("✅ Binance Connected")
            return client
        except Exception as e:
            st.error(f"❌ Binance Connection Failed: {str(e)}")
            return None
    
    def get_exchange(self, symbol):
        """Get appropriate exchange for symbol"""
        exchange_name = EXCHANGE_MAPPING.get(symbol, 'alpaca')
        
        if exchange_name not in self.connections:
            if exchange_name == 'alpaca':
                self.connections[exchange_name] = self.connect_alpaca()
            elif exchange_name == 'binance':
                self.connections[exchange_name] = self.connect_binance()
        
        return self.connections.get(exchange_name)

# =============================================
# REAL-TIME MARKET DATA STREAM
# =============================================

class MarketDataStream:
    """Real-time market data streaming"""
    
    def __init__(self):
        self.price_stream = {}
        self.order_book = {}
        self.volume_profile = {}
        self.last_update = {}
        self.subscriptions = set()
        
    def subscribe(self, symbols):
        """Subscribe to symbols for real-time updates"""
        for symbol in symbols:
            self.subscriptions.add(symbol)
            self.price_stream[symbol] = {
                'bid': 0,
                'ask': 0,
                'last': 0,
                'volume': 0,
                'timestamp': None
            }
    
    def get_real_time_price(self, symbol):
        """Get real-time price for symbol"""
        if symbol in self.price_stream and self.price_stream[symbol]['last'] > 0:
            return self.price_stream[symbol]['last']
        
        # Fallback to Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                self.price_stream[symbol] = {
                    'bid': price * 0.999,
                    'ask': price * 1.001,
                    'last': price,
                    'volume': float(data['Volume'].iloc[-1]),
                    'timestamp': datetime.now()
                }
                return price
        except:
            pass
        
        return 100.0  # Default fallback
    
    def simulate_market_data(self):
        """Simulate real-time market data updates"""
        for symbol in self.subscriptions:
            if symbol not in self.price_stream:
                continue
            
            current = self.price_stream[symbol]['last']
            if current == 0:
                current = self.get_real_time_price(symbol)
            
            # Simulate price movement
            change = np.random.normal(0, 0.001) * current
            new_price = current + change
            
            self.price_stream[symbol] = {
                'bid': new_price * 0.9995,
                'ask': new_price * 1.0005,
                'last': new_price,
                'volume': np.random.randint(100, 10000),
                'timestamp': datetime.now()
            }

# =============================================
# ALGORITHMIC TRADING ENGINE
# =============================================

class AlgorithmicTradingEngine:
    """Core algorithmic trading engine"""
    
    def __init__(self, mode="paper", initial_capital=INITIAL_CAPITAL):
        self.mode = mode
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions = {}
        self.trade_history = []
        self.order_queue = queue.Queue()
        self.exchange = ExchangeConnector(mode)
        self.market_data = MarketDataStream()
        self.strategies = {}
        self.risk_manager = RiskManager()
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Start trading thread
        self.trading_active = False
        self.trading_thread = None
        
        # Load strategies
        self._load_strategies()
    
    def _load_strategies(self):
        """Load trading strategies"""
        self.strategies = {
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'arbitrage': ArbitrageStrategy(),
            'market_making': MarketMakingStrategy()
        }
    
    def start_trading(self):
        """Start the algorithmic trading system"""
        if self.trading_active:
            return False
        
        self.trading_active = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        st.success("🚀 Algorithmic Trading Started!")
        return True
    
    def stop_trading(self):
        """Stop the algorithmic trading system"""
        self.trading_active = False
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        
        st.info("🛑 Algorithmic Trading Stopped")
        return True
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.trading_active:
            try:
                # 1. Update market data
                self.market_data.simulate_market_data()
                
                # 2. Generate trading signals
                signals = self._generate_signals()
                
                # 3. Process signals with risk management
                for signal in signals:
                    if self.risk_manager.validate_signal(signal, self.positions, self.cash):
                        self._execute_trade(signal)
                
                # 4. Manage existing positions
                self._manage_positions()
                
                # 5. Update performance metrics
                self._update_performance()
                
                # Sleep to control loop frequency
                time.sleep(1)  # 1 second between iterations
                
            except Exception as e:
                st.error(f"Trading loop error: {str(e)}")
                time.sleep(5)
    
    def _generate_signals(self):
        """Generate trading signals from all strategies"""
        signals = []
        
        # Get real-time prices for subscribed symbols
        symbols = list(self.market_data.subscriptions)
        if not symbols:
            symbols = ALL_SYMBOLS[:10]  # Default to first 10 symbols
        
        for symbol in symbols:
            current_price = self.market_data.get_real_time_price(symbol)
            
            # Get historical data for analysis
            historical_data = self._get_historical_data(symbol)
            
            # Run each strategy
            for strategy_name, strategy in self.strategies.items():
                signal = strategy.generate_signal(symbol, current_price, historical_data)
                if signal and signal['confidence'] > 0.6:  # Minimum confidence threshold
                    signals.append(signal)
        
        return signals
    
    def _get_historical_data(self, symbol, period='1d', interval='15m'):
        """Get historical market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except:
            # Return synthetic data for demo
            return self._generate_synthetic_data(symbol, period)
    
    def _generate_synthetic_data(self, symbol, period='1d'):
        """Generate synthetic market data"""
        if period == '1d':
            periods = 96  # 15-min candles in a day
        elif period == '7d':
            periods = 672
        else:
            periods = 96
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='15min')
        
        # Base price based on symbol
        if "BTC" in symbol:
            base_price = 45000
            volatility = 0.02
        elif "ETH" in symbol:
            base_price = 2500
            volatility = 0.025
        elif "AAPL" in symbol:
            base_price = 180
            volatility = 0.015
        else:
            base_price = 100
            volatility = 0.01
        
        # Generate price series
        np.random.seed(hash(symbol) % 10000)
        returns = np.random.normal(0, volatility/16, periods)
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC data
        opens = prices * (1 + np.random.normal(0, 0.001, periods))
        highs = opens * (1 + abs(np.random.normal(0, 0.005, periods)))
        lows = opens * (1 - abs(np.random.normal(0, 0.005, periods)))
        closes = prices
        
        # Add volume
        volume = np.random.randint(1000, 100000, periods) * (1 + abs(returns) * 10)
        
        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volume
        }, index=dates)
    
    def _execute_trade(self, signal):
        """Execute a trade based on signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            quantity = self._calculate_position_size(signal)
            
            if quantity <= 0:
                return False, "Invalid position size"
            
            # Get current price
            current_price = self.market_data.get_real_time_price(symbol)
            
            if self.mode == 'paper':
                # Paper trading execution
                return self._execute_paper_trade(signal, quantity, current_price)
            else:
                # Real trading execution
                return self._execute_real_trade(signal, quantity, current_price)
                
        except Exception as e:
            return False, f"Trade execution failed: {str(e)}"
    
    def _calculate_position_size(self, signal):
        """Calculate position size based on risk management"""
        current_price = self.market_data.get_real_time_price(signal['symbol'])
        stop_loss = signal.get('stop_loss', current_price * 0.95)
        
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        
        risk_amount = self.cash * RISK_PER_TRADE
        quantity = int(risk_amount / risk_per_share)
        
        # Ensure minimum and maximum limits
        min_quantity = 1
        max_quantity = int(self.cash * 0.1 / current_price)  # Max 10% of capital
        
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
            'paper_trade': True
        }
        
        # Update cash (simulated)
        trade_value = quantity * current_price
        if signal['action'] == 'BUY':
            self.cash -= trade_value
        elif signal['action'] == 'SELL':
            self.cash += trade_value  # Simplified for paper trading
        
        self.positions[trade_id] = trade
        self.trade_history.append(trade)
        
        return True, f"Paper trade executed: {signal['action']} {quantity} {signal['symbol']} @ ${current_price:.2f}"
    
    def _execute_real_trade(self, signal, quantity, current_price):
        """Execute real trade on exchange"""
        symbol = signal['symbol']
        action = signal['action'].lower()
        exchange_name = EXCHANGE_MAPPING.get(symbol, 'alpaca')
        
        try:
            if exchange_name == 'alpaca':
                # Alpaca trading
                api = self.exchange.connect_alpaca()
                if not api:
                    return False, "Alpaca connection failed"
                
                # Convert symbol format
                trade_symbol = symbol.replace('-USD', '') if '-USD' in symbol else symbol
                
                order = api.submit_order(
                    symbol=trade_symbol,
                    qty=quantity,
                    side=action,
                    type='market',
                    time_in_force='gtc'
                )
                
                trade_id = order.id
                
            elif exchange_name == 'binance':
                # Binance trading
                client = self.exchange.connect_binance()
                if not client:
                    return False, "Binance connection failed"
                
                # Convert symbol format
                trade_symbol = symbol.replace('-USD', 'USDT')
                
                if action == 'buy':
                    order = client.create_order(
                        symbol=trade_symbol,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                else:
                    order = client.create_order(
                        symbol=trade_symbol,
                        side='SELL',
                        type='MARKET',
                        quantity=quantity
                    )
                
                trade_id = order['orderId']
            
            else:
                return False, f"Unsupported exchange for {symbol}"
            
            # Record trade
            trade = {
                'trade_id': trade_id,
                'symbol': symbol,
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
                'paper_trade': False,
                'exchange': exchange_name,
                'order_id': trade_id
            }
            
            self.positions[trade_id] = trade
            self.trade_history.append(trade)
            
            return True, f"Real trade executed on {exchange_name}: {signal['action']} {quantity} {symbol} @ ${current_price:.2f}"
            
        except Exception as e:
            return False, f"Trade execution error: {str(e)}"
    
    def _manage_positions(self):
        """Manage existing positions (stop loss, take profit)"""
        positions_to_close = []
        
        for trade_id, position in self.positions.items():
            if position['status'] != 'OPEN':
                continue
            
            current_price = self.market_data.get_real_time_price(position['symbol'])
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
        current_price = self.market_data.get_real_time_price(position['symbol'])
        
        # Calculate final P&L
        if position['action'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['quantity']
            # Return cash (simplified for paper trading)
            if position['paper_trade']:
                self.cash += position['quantity'] * current_price
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
            if position['paper_trade']:
                self.cash += position['quantity'] * position['entry_price'] * 2  # Simplified
        
        # Update position
        position['exit_price'] = current_price
        position['exit_time'] = datetime.now()
        position['status'] = 'CLOSED'
        position['closed_pnl'] = pnl
        position['exit_reason'] = reason
        
        # Update performance
        self.performance['total_trades'] += 1
        self.performance['total_pnl'] += pnl
        self.performance['daily_pnl'] += pnl
        
        if pnl > 0:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        
        # Remove from open positions
        del self.positions[trade_id]
        
        return True
    
    def _update_performance(self):
        """Update performance metrics"""
        if self.performance['total_trades'] > 0:
            win_rate = self.performance['winning_trades'] / self.performance['total_trades']
            
            # Calculate Sharpe ratio (simplified)
            if len(self.trade_history) >= 10:
                recent_pnls = [t.get('closed_pnl', 0) for t in self.trade_history[-10:] if 'closed_pnl' in t]
                if len(recent_pnls) >= 5:
                    avg_return = np.mean(recent_pnls)
                    std_return = np.std(recent_pnls)
                    if std_return > 0:
                        self.performance['sharpe_ratio'] = avg_return / std_return * np.sqrt(252)  # Annualized
    
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
            'total_pnl': self.performance['total_pnl'],
            'daily_pnl': self.performance['daily_pnl'],
            'win_rate': self.performance['winning_trades'] / max(1, self.performance['total_trades']),
            'total_trades': self.performance['total_trades'],
            'sharpe_ratio': self.performance['sharpe_ratio']
        }
    
    def get_open_positions(self):
        """Get all open positions"""
        return list(self.positions.values())
    
    def get_trade_history(self, limit=50):
        """Get trade history"""
        return self.trade_history[-limit:] if self.trade_history else []

# =============================================
# TRADING STRATEGIES
# =============================================

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.name = "Base Strategy"
        self.description = "Base strategy class"
        self.parameters = {}
    
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
        indicators['ema_12'] = data['Close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = data['Close'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
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
        
        return indicators

class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Trend Following"
        self.description = "Follows established market trends"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0
        }
        
        # Bullish trend: Price above both SMAs and SMAs aligned
        if (current_price > indicators['sma_20'] > indicators['sma_50'] and
            indicators['rsi'] > 50 and indicators['rsi'] < 70):
            signal['action'] = 'BUY'
            signal['confidence'] = 0.7
            signal['stop_loss'] = current_price * 0.95
            signal['take_profit'] = current_price * 1.10
        
        # Bearish trend: Price below both SMAs and SMAs aligned
        elif (current_price < indicators['sma_20'] < indicators['sma_50'] and
              indicators['rsi'] < 50 and indicators['rsi'] > 30):
            signal['action'] = 'SELL'
            signal['confidence'] = 0.7
            signal['stop_loss'] = current_price * 1.05
            signal['take_profit'] = current_price * 0.90
        
        else:
            return None
        
        return signal

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Mean Reversion"
        self.description = "Trades price reversions to mean"
    
    def generate_signal(self, symbol, current_price, historical_data):
        indicators = self.calculate_indicators(historical_data)
        
        if len(indicators) == 0:
            return None
        
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0
        }
        
        # Oversold: Price below lower Bollinger Band and RSI < 30
        if current_price < indicators['bb_lower'] and indicators['rsi'] < 30:
            signal['action'] = 'BUY'
            signal['confidence'] = 0.65
            signal['stop_loss'] = current_price * 0.97
            signal['take_profit'] = indicators['bb_middle']
        
        # Overbought: Price above upper Bollinger Band and RSI > 70
        elif current_price > indicators['bb_upper'] and indicators['rsi'] > 70:
            signal['action'] = 'SELL'
            signal['confidence'] = 0.65
            signal['stop_loss'] = current_price * 1.03
            signal['take_profit'] = indicators['bb_middle']
        
        else:
            return None
        
        return signal

class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Breakout"
        self.description = "Trades price breakouts from consolidation"
    
    def generate_signal(self, symbol, current_price, historical_data):
        if len(historical_data) < 20:
            return None
        
        # Calculate recent volatility
        recent_atr = historical_data['Close'].pct_change().std() * np.sqrt(252)
        avg_atr = historical_data['Close'].pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252)
        
        # Calculate consolidation range
        recent_high = historical_data['High'].iloc[-20:].max()
        recent_low = historical_data['Low'].iloc[-20:].min()
        consolidation_range = (recent_high - recent_low) / recent_low
        
        signal = {
            'symbol': symbol,
            'strategy': self.name,
            'confidence': 0.0
        }
        
        # Bullish breakout: Price breaks above consolidation with volume
        if (current_price > recent_high and 
            consolidation_range < 0.05 and  # Tight consolidation
            historical_data['Volume'].iloc[-1] > historical_data['Volume'].rolling(window=20).mean().iloc[-1] * 1.5):
            signal['action'] = 'BUY'
            signal['confidence'] = 0.75
            signal['stop_loss'] = recent_low
            signal['take_profit'] = current_price + (recent_high - recent_low) * 1.5
        
        # Bearish breakout: Price breaks below consolidation with volume
        elif (current_price < recent_low and
              consolidation_range < 0.05 and
              historical_data['Volume'].iloc[-1] > historical_data['Volume'].rolling(window=20).mean().iloc[-1] * 1.5):
            signal['action'] = 'SELL'
            signal['confidence'] = 0.75
            signal['stop_loss'] = recent_high
            signal['take_profit'] = current_price - (recent_high - recent_low) * 1.5
        
        else:
            return None
        
        return signal

class ArbitrageStrategy(BaseStrategy):
    """Statistical arbitrage strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Arbitrage"
        self.description = "Statistical arbitrage between correlated assets"
    
    def generate_signal(self, symbol, current_price, historical_data):
        # This is a simplified version
        # In production, you'd compare prices across multiple exchanges
        return None

class MarketMakingStrategy(BaseStrategy):
    """Market making strategy"""
    
    def __init__(self):
        super().__init__()
        self.name = "Market Making"
        self.description = "Provides liquidity and captures bid-ask spread"
    
    def generate_signal(self, symbol, current_price, historical_data):
        # Market making requires real-time order book data
        return None

# =============================================
# RISK MANAGEMENT MODULE
# =============================================

class RiskManager:
    """Risk management module"""
    
    def __init__(self):
        self.max_daily_loss = -0.05  # -5% daily loss limit
        self.max_position_size = 0.1  # 10% of capital per position
        self.max_correlation = 0.7  # Maximum correlation between positions
        self.max_leverage = 3.0  # Maximum leverage
        self.daily_loss = 0.0
        self.position_correlation = {}
    
    def validate_signal(self, signal, current_positions, available_capital):
        """Validate trading signal against risk rules"""
        
        # 1. Check daily loss limit
        if self.daily_loss < self.max_daily_loss * available_capital:
            return False, "Daily loss limit reached"
        
        # 2. Check position size
        position_value = signal.get('quantity', 1) * signal.get('entry_price', 0)
        if position_value > available_capital * self.max_position_size:
            return False, "Position size exceeds limit"
        
        # 3. Check maximum positions
        if len(current_positions) >= MAX_POSITIONS:
            return False, "Maximum positions reached"
        
        # 4. Check correlation (simplified)
        symbol = signal['symbol']
        if symbol in self.position_correlation:
            if self.position_correlation[symbol] > self.max_correlation:
                return False, "High correlation with existing positions"
        
        # 5. Check volatility
        if signal.get('volatility', 0) > 0.05:  # 5% daily volatility threshold
            return False, "High volatility warning"
        
        return True, "Signal validated"
    
    def update_daily_loss(self, pnl):
        """Update daily loss tracking"""
        self.daily_loss += pnl
    
    def reset_daily_loss(self):
        """Reset daily loss tracking"""
        self.daily_loss = 0.0

# =============================================
# STREAMLIT UI COMPONENTS
# =============================================

def create_header():
    """Create application header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🤖 RANTV ALGORITHMIC TRADING SYSTEM</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">Real-time Algorithmic Trading with Multiple Strategies</p>
    </div>
    """, unsafe_allow_html=True)

def create_trading_control_panel(trading_engine):
    """Create trading control panel"""
    st.subheader("🎮 Trading Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not trading_engine.trading_active:
            if st.button("🚀 Start Algorithmic Trading", use_container_width=True, type="primary"):
                trading_engine.market_data.subscribe(ALL_SYMBOLS[:20])
                if trading_engine.start_trading():
                    st.rerun()
        else:
            if st.button("🛑 Stop Algorithmic Trading", use_container_width=True, type="secondary"):
                trading_engine.stop_trading()
                st.rerun()
    
    with col2:
        if st.button("🔄 Update All Positions", use_container_width=True):
            trading_engine._manage_positions()
            st.success("Positions updated!")
            st.rerun()
    
    with col3:
        if st.button("📊 Force Signal Scan", use_container_width=True):
            signals = trading_engine._generate_signals()
            st.info(f"Generated {len(signals)} signals")
            st.rerun()
    
    # Trading mode selector
    st.subheader("⚙️ Trading Mode")
    mode_col1, mode_col2 = st.columns(2)
    
    with mode_col1:
        trading_mode = st.selectbox(
            "Select Trading Mode",
            ["Paper Trading", "Live Trading"],
            index=0
        )
    
    with mode_col2:
        if st.button("Apply Mode Change", use_container_width=True):
            if trading_engine.trading_active:
                st.warning("Stop trading before changing mode")
            else:
                new_mode = "paper" if trading_mode == "Paper Trading" else "live"
                trading_engine.mode = new_mode
                st.success(f"Trading mode changed to {trading_mode}")

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
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Win Rate", f"{portfolio['win_rate']:.1%}")
    
    with perf_col2:
        st.metric("Total Trades", portfolio['total_trades'])
    
    with perf_col3:
        st.metric("Daily P&L", f"${portfolio['daily_pnl']:+,.2f}")
    
    with perf_col4:
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
                st.write(f"Strategy: {position['strategy']}")
            
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

def create_strategy_configuration(trading_engine):
    """Create strategy configuration panel"""
    st.subheader("🎯 Strategy Configuration")
    
    # Strategy selection
    selected_strategies = st.multiselect(
        "Select Active Strategies",
        list(trading_engine.strategies.keys()),
        default=['trend_following', 'mean_reversion', 'breakout']
    )
    
    # Strategy parameters
    st.subheader("⚙️ Strategy Parameters")
    
    for strategy_name in selected_strategies:
        strategy = trading_engine.strategies[strategy_name]
        
        with st.expander(f"{strategy.name} Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                min_confidence = st.slider(
                    f"Minimum Confidence ({strategy.name})",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.65,
                    step=0.05,
                    key=f"conf_{strategy_name}"
                )
            
            with col2:
                max_position_size = st.slider(
                    f"Max Position % ({strategy.name})",
                    min_value=1,
                    max_value=20,
                    value=10,
                    step=1,
                    key=f"pos_{strategy_name}"
                )
    
    # Risk parameters
    st.subheader("🛡️ Risk Parameters")
    
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
        max_positions = st.number_input(
            "Max Positions",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )
    
    if st.button("💾 Save Configuration", type="primary"):
        trading_engine.risk_manager.max_daily_loss = -daily_loss_limit / 100
        st.success("Configuration saved!")

def create_signal_monitor(trading_engine):
    """Create real-time signal monitor"""
    st.subheader("🚦 Real-time Signal Monitor")
    
    # Generate sample signals
    signals = []
    for symbol in ALL_SYMBOLS[:10]:
        current_price = trading_engine.market_data.get_real_time_price(symbol)
        historical_data = trading_engine._get_historical_data(symbol)
        
        for strategy in trading_engine.strategies.values():
            signal = strategy.generate_signal(symbol, current_price, historical_data)
            if signal and signal['confidence'] > 0.6:
                signals.append(signal)
    
    if not signals:
        st.info("No active signals at the moment")
        return
    
    # Display signals
    for signal in signals[:5]:  # Show top 5
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                action_emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
                st.write(f"**{action_emoji} {signal['symbol']} - {signal['action']}**")
                st.write(f"Strategy: {signal['strategy']}")
                st.write(f"Confidence: {signal['confidence']:.1%}")
            
            with col2:
                st.write(f"Price: ${signal.get('entry_price', 0):.2f}")
            
            with col3:
                st.write(f"SL: ${signal.get('stop_loss', 0):.2f}")
                st.write(f"TP: ${signal.get('take_profit', 0):.2f}")
            
            with col4:
                if st.button("Trade", key=f"trade_{signal['symbol']}_{signal['action']}"):
                    success, message = trading_engine._execute_trade(signal)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            st.divider()

def create_performance_analytics(trading_engine):
    """Create performance analytics dashboard"""
    st.subheader("📈 Performance Analytics")
    
    # Get trade history
    trade_history = trading_engine.get_trade_history(100)
    
    if not trade_history:
        st.info("No trade history available")
        return
    
    # Convert to DataFrame
    trades_df = pd.DataFrame([
        {
            'Date': trade['timestamp'],
            'Symbol': trade['symbol'],
            'Action': trade['action'],
            'Entry': trade['entry_price'],
            'Exit': trade.get('exit_price', None),
            'P&L': trade.get('closed_pnl', trade.get('pnl', 0)),
            'Strategy': trade['strategy'],
            'Status': trade['status']
        }
        for trade in trade_history
    ])
    
    if trades_df.empty:
        st.info("No trade data to display")
        return
    
    # Performance metrics
    st.subheader("📊 Trade Statistics")
    
    closed_trades = trades_df[trades_df['Status'] == 'CLOSED']
    if not closed_trades.empty:
        winning_trades = closed_trades[closed_trades['P&L'] > 0]
        losing_trades = closed_trades[closed_trades['P&L'] <= 0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(closed_trades))
        
        with col2:
            win_rate = len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1%}")
        
        with col3:
            avg_win = winning_trades['P&L'].mean() if len(winning_trades) > 0 else 0
            st.metric("Avg Win", f"${avg_win:+,.2f}")
        
        with col4:
            avg_loss = losing_trades['P&L'].mean() if len(losing_trades) > 0 else 0
            st.metric("Avg Loss", f"${avg_loss:+,.2f}")
        
        # Equity curve
        st.subheader("📈 Equity Curve")
        
        closed_trades = closed_trades.sort_values('Date')
        closed_trades['Cumulative P&L'] = closed_trades['P&L'].cumsum() + trading_engine.initial_capital
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closed_trades['Date'],
            y=closed_trades['Cumulative P&L'],
            mode='lines',
            name='Equity Curve',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy performance
        st.subheader("🎯 Strategy Performance")
        
        strategy_perf = closed_trades.groupby('Strategy').agg({
            'P&L': ['count', 'sum', 'mean'],
            'Symbol': 'nunique'
        }).round(2)
        
        strategy_perf.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Unique Symbols']
        st.dataframe(strategy_perf, use_container_width=True)

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    """Main application function"""
    
    # Initialize session state
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = AlgorithmicTradingEngine(mode="paper")
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    # Auto-refresh for real-time updates
    if st.session_state.auto_refresh:
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
        
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        
        # Market selection
        st.subheader("🌐 Market Selection")
        
        selected_markets = st.multiselect(
            "Select Markets to Trade",
            MARKET_OPTIONS,
            default=["CRYPTO", "STOCKS"]
        )
        
        # Symbol selection
        st.subheader("🎯 Symbol Selection")
        
        selected_symbols = st.multiselect(
            "Select Symbols to Monitor",
            ALL_SYMBOLS,
            default=ALL_SYMBOLS[:10]
        )
        
        if st.button("📡 Subscribe to Symbols", use_container_width=True):
            trading_engine = st.session_state.trading_engine
            trading_engine.market_data.subscribe(selected_symbols)
            st.success(f"Subscribed to {len(selected_symbols)} symbols")
        
        # System controls
        st.subheader("🔄 System Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reset Session", use_container_width=True):
                st.session_state.trading_engine = AlgorithmicTradingEngine(mode="paper")
                st.success("New trading session started!")
                st.rerun()
        
        with col2:
            if st.button("Clear History", use_container_width=True, type="secondary"):
                trading_engine = st.session_state.trading_engine
                trading_engine.trade_history = []
                trading_engine.positions = {}
                st.success("Trade history cleared!")
                st.rerun()
        
        # System status
        st.markdown("---")
        trading_engine = st.session_state.trading_engine
        status_color = "🟢" if trading_engine.trading_active else "🔴"
        st.markdown(f"**System Status:** {status_color} {'Running' if trading_engine.trading_active else 'Stopped'}")
        st.markdown(f"**Mode:** {'Paper Trading' if trading_engine.mode == 'paper' else 'Live Trading'}")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎮 Control Panel",
        "📊 Portfolio",
        "💰 Positions",
        "🚦 Signals",
        "📈 Analytics",
        "⚙️ Configuration"
    ])
    
    trading_engine = st.session_state.trading_engine
    
    with tab1:
        # Control Panel
        create_trading_control_panel(trading_engine)
        
        # Quick stats
        st.subheader("⚡ Quick Stats")
        
        portfolio = trading_engine.get_portfolio_summary()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Active Strategies", len(trading_engine.strategies))
        
        with stat_col2:
            st.metric("Subscribed Symbols", len(trading_engine.market_data.subscriptions))
        
        with stat_col3:
            st.metric("Queue Size", trading_engine.order_queue.qsize())
        
        with stat_col4:
            uptime = "N/A"  # You can track uptime if needed
            st.metric("System Uptime", uptime)
    
    with tab2:
        # Portfolio Dashboard
        create_portfolio_dashboard(trading_engine)
        
        # Market overview
        st.subheader("🌐 Market Overview")
        
        # Display key market prices
        key_symbols = ["BTC-USD", "ETH-USD", "AAPL", "GC=F", "EURUSD=X"]
        cols = st.columns(len(key_symbols))
        
        for i, symbol in enumerate(key_symbols):
            with cols[i]:
                try:
                    price = trading_engine.market_data.get_real_time_price(symbol)
                    change = np.random.uniform(-2, 2)  # Simulated change
                    st.metric(
                        symbol,
                        f"${price:,.2f}" if price > 10 else f"{price:.4f}",
                        delta=f"{change:+.1f}%"
                    )
                except:
                    st.metric(symbol, "N/A")
    
    with tab3:
        # Positions Dashboard
        create_positions_dashboard(trading_engine)
        
        # Position analytics
        st.subheader("📊 Position Analytics")
        
        positions = trading_engine.get_open_positions()
        if positions:
            total_exposure = sum(p['quantity'] * p.get('current_price', p['entry_price']) 
                               for p in positions)
            avg_position_size = total_exposure / len(positions) if positions else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Exposure", f"${total_exposure:,.2f}")
            
            with col2:
                st.metric("Average Position", f"${avg_position_size:,.2f}")
            
            with col3:
                buy_positions = sum(1 for p in positions if p['action'] == 'BUY')
                sell_positions = len(positions) - buy_positions
                st.metric("Bias", f"{buy_positions}B/{sell_positions}S")
    
    with tab4:
        # Signal Monitor
        create_signal_monitor(trading_engine)
        
        # Manual trading
        st.subheader("👨‍💼 Manual Trading")
        
        with st.form("manual_trade_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                manual_symbol = st.selectbox("Symbol", ALL_SYMBOLS)
                manual_action = st.selectbox("Action", ["BUY", "SELL"])
            
            with col2:
                manual_quantity = st.number_input("Quantity", min_value=1, value=1)
                manual_price = st.number_input("Price", min_value=0.01, value=100.0, step=0.01)
            
            with col3:
                manual_sl = st.number_input("Stop Loss", min_value=0.01, value=95.0, step=0.01)
                manual_tp = st.number_input("Take Profit", min_value=0.01, value=105.0, step=0.01)
            
            submitted = st.form_submit_button("Execute Manual Trade")
            
            if submitted:
                signal = {
                    'symbol': manual_symbol,
                    'action': manual_action,
                    'strategy': 'MANUAL',
                    'confidence': 1.0,
                    'stop_loss': manual_sl,
                    'take_profit': manual_tp
                }
                
                success, message = trading_engine._execute_trade(signal)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    with tab5:
        # Performance Analytics
        create_performance_analytics(trading_engine)
    
    with tab6:
        # Configuration
        create_strategy_configuration(trading_engine)
        
        # Exchange configuration
        st.subheader("🏦 Exchange Configuration")
        
        exchange_col1, exchange_col2 = st.columns(2)
        
        with exchange_col1:
            alpaca_key = st.text_input("Alpaca API Key", type="password")
            alpaca_secret = st.text_input("Alpaca Secret Key", type="password")
        
        with exchange_col2:
            binance_key = st.text_input("Binance API Key", type="password")
            binance_secret = st.text_input("Binance Secret Key", type="password")
        
        if st.button("Test Connections", type="secondary"):
            st.info("Testing exchange connections...")
            # Here you would test the connections with provided keys
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>RANTV Algorithmic Trading System v4.0</strong> | Multi-Strategy Algo Trading | Real-time Execution</p>
    <p>⚠️ Algorithmic trading involves significant risk. Use at your own risk. Past performance does not guarantee future results.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# RUN APPLICATION
# =============================================

if __name__ == "__main__":
    main()
