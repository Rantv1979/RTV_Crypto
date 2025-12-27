# app.py - Complete SMC Algorithmic Trading System
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
import random
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SMC Algo Trading System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .signal-bullish {
        background: linear-gradient(135deg, #D1FAE5, #10B981);
        color: #065F46;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #10B981;
    }
    .signal-bearish {
        background: linear-gradient(135deg, #FEE2E2, #EF4444);
        color: #991B1B;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #EF4444;
    }
    .paper-trade {
        background: linear-gradient(135deg, #E0F2FE, #0EA5E9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #0EA5E9;
    }
    .tab-content {
        padding: 1rem;
        background: #f8fafc;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🤖 SMC Algorithmic Trading System</h1>', unsafe_allow_html=True)

# Initialize session state for paper trading
if 'paper_portfolio' not in st.session_state:
    st.session_state.paper_portfolio = {
        'balance': 10000.00,
        'positions': {},
        'trade_history': [],
        'pnl': 0.00
    }

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}

# Asset configuration with proper symbols
ASSET_CONFIG = {
    "Cryptocurrencies": {
        "BTC/USD": {"symbol": "BTC-USD", "pip_size": 0.01, "lot_size": 0.001},
        "ETH/USD": {"symbol": "ETH-USD", "pip_size": 0.01, "lot_size": 0.01},
        "SOL/USD": {"symbol": "SOL-USD", "pip_size": 0.001, "lot_size": 0.1},
    },
    "Forex": {
        "EUR/USD": {"symbol": "EURUSD=X", "pip_size": 0.0001, "lot_size": 10000},
        "GBP/USD": {"symbol": "GBPUSD=X", "pip_size": 0.0001, "lot_size": 10000},
        "USD/JPY": {"symbol": "JPY=X", "pip_size": 0.01, "lot_size": 10000},
        "AUD/USD": {"symbol": "AUDUSD=X", "pip_size": 0.0001, "lot_size": 10000},
    },
    "Commodities": {
        "Gold": {"symbol": "GC=F", "pip_size": 0.10, "lot_size": 1},
        "Silver": {"symbol": "SI=F", "pip_size": 0.01, "lot_size": 10},
        "Crude Oil": {"symbol": "CL=F", "pip_size": 0.01, "lot_size": 10},
        "Copper": {"symbol": "HG=F", "pip_size": 0.0005, "lot_size": 100},
    },
    "Indices": {
        "S&P 500": {"symbol": "^GSPC", "pip_size": 0.25, "lot_size": 1},
        "NASDAQ": {"symbol": "^IXIC", "pip_size": 0.25, "lot_size": 1},
        "Dow Jones": {"symbol": "^DJI", "pip_size": 1.0, "lot_size": 1},
    }
}

# Sidebar Configuration
st.sidebar.header("⚙️ Trading Configuration")

# Mode Selection
mode = st.sidebar.selectbox(
    "Operating Mode",
    ["Backtesting", "Paper Trading", "Live Analysis"],
    index=1
)

# Asset Selection
asset_category = st.sidebar.selectbox(
    "Asset Category",
    list(ASSET_CONFIG.keys()),
    index=2  # Default to Commodities
)

selected_asset = st.sidebar.selectbox(
    "Select Asset",
    list(ASSET_CONFIG[asset_category].keys())
)

asset_info = ASSET_CONFIG[asset_category][selected_asset]

# Timeframe selection including 5m and 15m
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["5m", "15m", "30m", "1h", "4h", "1d"],
    index=0
)

# SMC Parameters
st.sidebar.header("📊 SMC Parameters")
fvg_period = st.sidebar.slider("FVG Lookback", 3, 20, 5)
swing_period = st.sidebar.slider("Swing Period", 2, 10, 3)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
atr_period = st.sidebar.slider("ATR Period", 7, 21, 14)

# Risk Management
st.sidebar.header("💰 Risk Management")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0)
stop_loss_atr = st.sidebar.slider("Stop Loss (ATR)", 1.0, 3.0, 2.0)
take_profit_atr = st.sidebar.slider("Take Profit (ATR)", 1.0, 4.0, 3.0)

# Data Fetching with accurate pricing
@st.cache_data(ttl=300)
def fetch_market_data(symbol, interval='5m', period='7d'):
    """Fetch real market data with accurate pricing"""
    try:
        # Map interval for yfinance
        interval_map = {
            '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '60m', '4h': '60m', '1d': '1d'
        }
        
        yf_interval = interval_map.get(interval, '5m')
        
        # Determine period based on interval
        period_map = {
            '5m': '7d', '15m': '7d', '30m': '30d',
            '1h': '30d', '4h': '60d', '1d': '180d'
        }
        
        yf_period = period_map.get(interval, '7d')
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yf_period, interval=yf_interval)
        
        if df.empty:
            # Try alternative method
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df = yf.download(symbol, start=start_date, end=end_date, interval=yf_interval, progress=False)
        
        # Ensure we have proper OHLC data
        if len(df) > 0 and 'Open' in df.columns:
            # Clean and format data
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.dropna(inplace=True)
            
            # Add technical indicators
            df = add_technical_indicators(df)
            
            return df
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    
    return pd.DataFrame()

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    if len(df) < 20:
        return df
    
    # Calculate ATR (Average True Range)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        )
    )
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    return df

# SMC Analysis Engine
class SMCAlgorithm:
    def __init__(self, fvg_lookback=5, swing_period=3):
        self.fvg_lookback = fvg_lookback
        self.swing_period = swing_period
        
    def analyze_market_structure(self, df):
        """Advanced market structure analysis"""
        df = df.copy()
        
        # Identify swing points
        df['Swing_High'] = False
        df['Swing_Low'] = False
        
        for i in range(self.swing_period, len(df)-self.swing_period):
            # Swing High detection
            if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'Swing_High'] = True
            
            # Swing Low detection
            if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'Swing_Low'] = True
        
        # Determine market structure
        df['Market_Structure'] = 'Neutral'
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for Higher Highs and Higher Lows
            if (swing_highs['High'].iloc[-1] > swing_highs['High'].iloc[-2] and
                swing_lows['Low'].iloc[-1] > swing_lows['Low'].iloc[-2]):
                df['Market_Structure'] = 'Uptrend'
            
            # Check for Lower Highs and Lower Lows
            elif (swing_highs['High'].iloc[-1] < swing_highs['High'].iloc[-2] and
                  swing_lows['Low'].iloc[-1] < swing_lows['Low'].iloc[-2]):
                df['Market_Structure'] = 'Downtrend'
        
        return df
    
    def identify_fvgs(self, df):
        """Identify Fair Value Gaps"""
        df = df.copy()
        
        df['FVG_Bullish'] = np.nan
        df['FVG_Bearish'] = np.nan
        df['FVG_Width'] = np.nan
        
        for i in range(self.fvg_lookback, len(df)-1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG
            if current['High'] < next_candle['Low']:
                gap = next_candle['Low'] - current['High']
                if gap > current['ATR'] * 0.1:  # Minimum gap size
                    df.loc[df.index[i], 'FVG_Bullish'] = current['High']
                    df.loc[df.index[i], 'FVG_Width'] = gap
            
            # Bearish FVG
            elif current['Low'] > next_candle['High']:
                gap = current['Low'] - next_candle['High']
                if gap > current['ATR'] * 0.1:  # Minimum gap size
                    df.loc[df.index[i], 'FVG_Bearish'] = current['Low']
                    df.loc[df.index[i], 'FVG_Width'] = gap
        
        return df
    
    def identify_orderblocks(self, df):
        """Identify Order Blocks"""
        df = df.copy()
        
        df['OB_Bullish'] = np.nan
        df['OB_Bearish'] = np.nan
        
        for i in range(2, len(df)-2):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            # Bullish Order Block
            if (current['Close'] > current['Open'] and  # Bullish candle
                (current['Close'] - current['Open']) > (current['High'] - current['Low']) * 0.6 and  # Strong body
                next_candle['Low'] >= current['Low']):  # Next candle doesn't break low
                
                df.loc[df.index[i], 'OB_Bullish'] = current['Low']
            
            # Bearish Order Block
            elif (current['Close'] < current['Open'] and  # Bearish candle
                  (current['Open'] - current['Close']) > (current['High'] - current['Low']) * 0.6 and  # Strong body
                  next_candle['High'] <= current['High']):  # Next candle doesn't break high
                
                df.loc[df.index[i], 'OB_Bearish'] = current['High']
        
        return df
    
    def generate_signals(self, df, asset_info):
        """Generate trading signals with SMC confluence"""
        signals = []
        
        if len(df) < 20:
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate position size based on risk
        atr_value = latest['ATR']
        if pd.isna(atr_value) or atr_value == 0:
            atr_value = df['ATR'].mean()
        
        stop_distance = atr_value * stop_loss_atr
        position_size = (st.session_state.paper_portfolio['balance'] * risk_per_trade / 100) / stop_distance
        position_size = min(position_size, asset_info['lot_size'] * 10)  # Cap position size
        
        # Signal 1: FVG + RSI confluence
        if not pd.isna(latest.get('FVG_Bullish', np.nan)):
            fvg_top = latest['FVG_Bullish'] + latest.get('FVG_Width', 0)
            if latest['Low'] <= fvg_top <= latest['High'] and latest['RSI'] < 40:
                entry = fvg_top
                stop_loss = latest['FVG_Bullish'] - atr_value * 0.5
                take_profit = entry + atr_value * take_profit_atr
                
                signals.append({
                    'type': 'BUY',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'confidence': 0.75,
                    'reason': 'Bullish FVG fill with oversold RSI',
                    'conditions': ['FVG', 'RSI < 40', 'Price at FVG']
                })
        
        if not pd.isna(latest.get('FVG_Bearish', np.nan)):
            fvg_bottom = latest['FVG_Bearish'] - latest.get('FVG_Width', 0)
            if latest['Low'] <= fvg_bottom <= latest['High'] and latest['RSI'] > 60:
                entry = fvg_bottom
                stop_loss = latest['FVG_Bearish'] + atr_value * 0.5
                take_profit = entry - atr_value * take_profit_atr
                
                signals.append({
                    'type': 'SELL',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'confidence': 0.75,
                    'reason': 'Bearish FVG fill with overbought RSI',
                    'conditions': ['FVG', 'RSI > 60', 'Price at FVG']
                })
        
        # Signal 2: Order Block + Market Structure
        if not pd.isna(latest.get('OB_Bullish', np.nan)):
            ob_price = latest['OB_Bullish']
            if latest['Low'] <= ob_price <= latest['High'] and latest['Market_Structure'] == 'Uptrend':
                entry = ob_price
                stop_loss = ob_price - atr_value * stop_loss_atr
                take_profit = entry + atr_value * take_profit_atr
                
                signals.append({
                    'type': 'BUY',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'confidence': 0.85,
                    'reason': 'Bullish Order Block in uptrend',
                    'conditions': ['Order Block', 'Uptrend', 'Price at OB']
                })
        
        if not pd.isna(latest.get('OB_Bearish', np.nan)):
            ob_price = latest['OB_Bearish']
            if latest['Low'] <= ob_price <= latest['High'] and latest['Market_Structure'] == 'Downtrend':
                entry = ob_price
                stop_loss = ob_price + atr_value * stop_loss_atr
                take_profit = entry - atr_value * take_profit_atr
                
                signals.append({
                    'type': 'SELL',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'confidence': 0.85,
                    'reason': 'Bearish Order Block in downtrend',
                    'conditions': ['Order Block', 'Downtrend', 'Price at OB']
                })
        
        # Signal 3: Market Structure Break
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        if len(swing_highs) > 0 and len(swing_lows) > 0:
            last_swing_high = swing_highs['High'].iloc[-1]
            last_swing_low = swing_lows['Low'].iloc[-1]
            
            # Break of Structure (BOS)
            if latest['Close'] > last_swing_high and latest['RSI'] > 50:
                entry = latest['Close']
                stop_loss = last_swing_low
                take_profit = entry + (entry - stop_loss) * 2
                
                signals.append({
                    'type': 'BUY',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'confidence': 0.80,
                    'reason': 'Break of Structure (BOS) to upside',
                    'conditions': ['BOS', 'RSI > 50', 'Above swing high']
                })
            
            elif latest['Close'] < last_swing_low and latest['RSI'] < 50:
                entry = latest['Close']
                stop_loss = last_swing_high
                take_profit = entry - (stop_loss - entry) * 2
                
                signals.append({
                    'type': 'SELL',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'confidence': 0.80,
                    'reason': 'Break of Structure (BOS) to downside',
                    'conditions': ['BOS', 'RSI < 50', 'Below swing low']
                })
        
        # Filter and rank signals
        if signals:
            signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        
        return signals[:3]  # Return top 3 signals

# Backtesting Engine
class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        
    def run_backtest(self, df, asset_info, smc_algo):
        """Run backtest on historical data"""
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'trades': []
        }
        
        capital = self.initial_capital
        peak_capital = capital
        max_drawdown = 0
        
        for i in range(50, len(df)-5):  # Leave room for exit
            current_data = df.iloc[:i+1].copy()
            current_data = smc_algo.analyze_market_structure(current_data)
            current_data = smc_algo.identify_fvgs(current_data)
            current_data = smc_algo.identify_orderblocks(current_data)
            
            signals = smc_algo.generate_signals(current_data, asset_info)
            
            if signals and i < len(df) - 5:
                signal = signals[0]  # Take best signal
                entry_price = signal['entry']
                exit_index = min(i + 20, len(df) - 1)  # Max 20 periods hold
                
                # Simulate trade outcome
                if signal['type'] == 'BUY':
                    exit_price = df['Close'].iloc[exit_index]
                    pnl = (exit_price - entry_price) * signal['size']
                else:  'SELL'
                    exit_price = df['Close'].iloc[exit_index]
                    pnl = (entry_price - exit_price) * signal['size']
                
                # Update capital
                capital += pnl
                results['total_pnl'] += pnl
                results['total_trades'] += 1
                
                if pnl > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1
                
                # Track drawdown
                peak_capital = max(peak_capital, capital)
                drawdown = (peak_capital - capital) / peak_capital
                max_drawdown = max(max_drawdown, drawdown)
                
                # Record trade
                results['trades'].append({
                    'entry_time': df.index[i],
                    'exit_time': df.index[exit_index],
                    'type': signal['type'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'size': signal['size']
                })
        
        # Calculate metrics
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades'] * 100
            total_wins = sum(t['pnl'] for t in results['trades'] if t['pnl'] > 0)
            total_losses = abs(sum(t['pnl'] for t in results['trades'] if t['pnl'] < 0))
            
            if total_losses > 0:
                results['profit_factor'] = total_wins / total_losses
        
        results['max_drawdown'] = max_drawdown * 100
        results['final_capital'] = capital
        
        return results

# Paper Trading Engine
class PaperTrading:
    def __init__(self):
        self.portfolio = st.session_state.paper_portfolio
    
    def execute_trade(self, signal, current_price, asset_symbol):
        """Execute paper trade"""
        trade_value = signal['entry'] * signal['size']
        
        if trade_value > self.portfolio['balance'] * 0.9:  # 90% max exposure
            return False, "Insufficient balance"
        
        # Record trade
        trade_id = f"{asset_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade = {
            'id': trade_id,
            'timestamp': datetime.now(),
            'asset': asset_symbol,
            'type': signal['type'],
            'entry_price': signal['entry'],
            'size': signal['size'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'status': 'OPEN',
            'pnl': 0.00
        }
        
        self.portfolio['positions'][trade_id] = trade
        self.portfolio['balance'] -= trade_value * 0.001  # Simulate commission
        
        # Add to history
        self.portfolio['trade_history'].append({
            **trade,
            'action': 'OPEN'
        })
        
        return True, f"Trade {trade_id} opened successfully"
    
    def update_positions(self, current_prices):
        """Update open positions with current prices"""
        total_pnl = 0
        
        for trade_id, position in list(self.portfolio['positions'].items()):
            if position['status'] == 'OPEN':
                current_price = current_prices.get(position['asset'], 0)
                
                if current_price > 0:
                    # Calculate P&L
                    if position['type'] == 'BUY':
                        pnl = (current_price - position['entry_price']) * position['size']
                    else:  # SELL
                        pnl = (position['entry_price'] - current_price) * position['size']
                    
                    position['pnl'] = pnl
                    total_pnl += pnl
                    
                    # Check exit conditions
                    if (position['type'] == 'BUY' and 
                        (current_price <= position['stop_loss'] or current_price >= position['take_profit'])):
                        self.close_position(trade_id, current_price)
                    
                    elif (position['type'] == 'SELL' and 
                          (current_price >= position['stop_loss'] or current_price <= position['take_profit'])):
                        self.close_position(trade_id, current_price)
        
        self.portfolio['pnl'] = total_pnl
        return total_pnl
    
    def close_position(self, trade_id, exit_price):
        """Close a position"""
        if trade_id in self.portfolio['positions']:
            position = self.portfolio['positions'][trade_id]
            
            # Calculate final P&L
            if position['type'] == 'BUY':
                final_pnl = (exit_price - position['entry_price']) * position['size']
            else:  # SELL
                final_pnl = (position['entry_price'] - exit_price) * position['size']
            
            # Update balance
            trade_value = position['entry_price'] * position['size']
            self.portfolio['balance'] += trade_value + final_pnl
            
            # Record closure
            self.portfolio['trade_history'].append({
                'id': trade_id,
                'timestamp': datetime.now(),
                'asset': position['asset'],
                'type': position['type'],
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': final_pnl,
                'action': 'CLOSE'
            })
            
            # Remove from open positions
            del self.portfolio['positions'][trade_id]
            
            return True
        
        return False

# Main Application
def main():
    # Initialize engines
    smc_algo = SMCAlgorithm(fvg_lookback=fvg_period, swing_period=swing_period)
    backtester = Backtester(initial_capital=10000)
    paper_trader = PaperTrading()
    
    # Fetch market data
    with st.spinner(f"Fetching {selected_asset} data..."):
        df = fetch_market_data(asset_info['symbol'], timeframe)
    
    if df.empty:
        st.error("Could not fetch market data. Please check your internet connection.")
        return
    
    # Display dashboard metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = df['Close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100
        st.metric("24h Change", f"{change_pct:+.2f}%")
    
    with col3:
        atr_value = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
        st.metric("ATR", f"${atr_value:.2f}")
    
    with col4:
        rsi_value = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        st.metric("RSI", f"{rsi_value:.1f}")
    
    with col5:
        st.metric("Paper Balance", f"${st.session_state.paper_portfolio['balance']:.2f}")
    
    st.markdown("---")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Live Analysis", 
        "🤖 Auto Trading", 
        "📊 Backtesting", 
        "💰 Paper Trading"
    ])
    
    with tab1:
        st.subheader("Live Market Analysis")
        
        # Analyze current market
        df_analyzed = smc_algo.analyze_market_structure(df)
        df_analyzed = smc_algo.identify_fvgs(df_analyzed)
        df_analyzed = smc_algo.identify_orderblocks(df_analyzed)
        
        # Display market structure
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Market Structure")
            structure = df_analyzed['Market_Structure'].iloc[-1]
            
            if structure == 'Uptrend':
                st.success(f"📈 {structure}")
            elif structure == 'Downtrend':
                st.error(f"📉 {structure}")
            else:
                st.info(f"⚪ {structure}")
            
            # Display swing points
            swing_highs = df_analyzed[df_analyzed['Swing_High']]
            swing_lows = df_analyzed[df_analyzed['Swing_Low']]
            
            if not swing_highs.empty:
                last_high = swing_highs['High'].iloc[-1]
                st.write(f"**Last Swing High:** ${last_high:.2f}")
            
            if not swing_lows.empty:
                last_low = swing_lows['Low'].iloc[-1]
                st.write(f"**Last Swing Low:** ${last_low:.2f}")
        
        with col2:
            st.markdown("### Active FVGs")
            fvg_bullish = df_analyzed[~pd.isna(df_analyzed['FVG_Bullish'])]
            fvg_bearish = df_analyzed[~pd.isna(df_analyzed['FVG_Bearish'])]
            
            if not fvg_bullish.empty:
                st.write("**Bullish FVGs:**")
                for idx in fvg_bullish.index[-3:]:
                    price = fvg_bullish.loc[idx, 'FVG_Bullish']
                    st.write(f"- ${price:.2f}")
            
            if not fvg_bearish.empty:
                st.write("**Bearish FVGs:**")
                for idx in fvg_bearish.index[-3:]:
                    price = fvg_bearish.loc[idx, 'FVG_Bearish']
                    st.write(f"- ${price:.2f}")
        
        # Display price chart
        st.subheader("Price Chart")
        chart_data = df_analyzed[['Close', 'MA20', 'MA50']].tail(100)
        st.line_chart(chart_data)
        
        # Display RSI chart
        st.subheader("RSI Indicator")
        if 'RSI' in df_analyzed.columns:
            rsi_data = pd.DataFrame({
                'RSI': df_analyzed['RSI'].tail(100),
                'Oversold': 30,
                'Overbought': 70
            })
            st.line_chart(rsi_data)
    
    with tab2:
        st.subheader("Automated Trading Signals")
        
        # Generate current signals
        signals = smc_algo.generate_signals(df_analyzed, asset_info)
        
        if signals:
            st.success(f"🎯 {len(signals)} trading signals generated")
            
            for i, signal in enumerate(signals):
                if signal['type'] == 'BUY':
                    st.markdown(f"""
                    <div class="signal-bullish">
                        <h3>📈 Signal #{i+1}: BUY (Confidence: {signal['confidence']*100:.0f}%)</h3>
                        <p><strong>Reason:</strong> {signal['reason']}</p>
                        <p><strong>Entry:</strong> ${signal['entry']:.2f}</p>
                        <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</p>
                        <p><strong>Take Profit:</strong> ${signal['take_profit']:.2f}</p>
                        <p><strong>Position Size:</strong> {signal['size']:.4f} units</p>
                        <p><strong>Conditions Met:</strong> {', '.join(signal['conditions'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-bearish">
                        <h3>📉 Signal #{i+1}: SELL (Confidence: {signal['confidence']*100:.0f}%)</h3>
                        <p><strong>Reason:</strong> {signal['reason']}</p>
                        <p><strong>Entry:</strong> ${signal['entry']:.2f}</p>
                        <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</p>
                        <p><strong>Take Profit:</strong> ${signal['take_profit']:.2f}</p>
                        <p><strong>Position Size:</strong> {signal['size']:.4f} units</p>
                        <p><strong>Conditions Met:</strong> {', '.join(signal['conditions'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Auto-execute button
            if mode == "Paper Trading":
                col1, col2, col3 = st.columns(3)
                with col2:
                    if st.button("🤖 Execute Best Signal", type="primary"):
                        best_signal = signals[0]
                        success, message = paper_trader.execute_trade(
                            best_signal, 
                            current_price, 
                            asset_info['symbol']
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        else:
            st.info("No trading signals generated for current market conditions.")
    
    with tab3:
        st.subheader("Strategy Backtesting")
        
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                results = backtester.run_backtest(df, asset_info, smc_algo)
                st.session_state.backtest_results = results
            
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Display backtest results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", results['total_trades'])
            
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            
            with col3:
                st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            
            with col4:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
            
            # P&L chart
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                
                st.subheader("Equity Curve")
                st.line_chart(trades_df['cumulative_pnl'])
                
                # Trade statistics
                st.subheader("Trade Statistics")
                st.dataframe(trades_df.tail(10))
    
    with tab4:
        st.subheader("Paper Trading Dashboard")
        
        # Portfolio overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Account Balance", 
                f"${paper_trader.portfolio['balance']:.2f}",
                f"{paper_trader.portfolio['pnl']:+.2f}"
            )
        
        with col2:
            open_positions = len(paper_trader.portfolio['positions'])
            st.metric("Open Positions", open_positions)
        
        with col3:
            total_trades = len(paper_trader.portfolio['trade_history'])
            st.metric("Total Trades", total_trades)
        
        # Open positions
        st.subheader("📊 Open Positions")
        if paper_trader.portfolio['positions']:
            for trade_id, position in paper_trader.portfolio['positions'].items():
                pnl_color = "green" if position['pnl'] >= 0 else "red"
                
                st.markdown(f"""
                <div class="paper-trade">
                    <h4>{position['asset']} - {position['type']} (ID: {trade_id})</h4>
                    <p><strong>Entry:</strong> ${position['entry_price']:.2f} | 
                    <strong>Current P&L:</strong> <span style="color:{pnl_color}">${position['pnl']:.2f}</span></p>
                    <p><strong>Stop Loss:</strong> ${position['stop_loss']:.2f} | 
                    <strong>Take Profit:</strong> ${position['take_profit']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No open positions")
        
        # Trade history
        st.subheader("📋 Trade History")
        if paper_trader.portfolio['trade_history']:
            history_df = pd.DataFrame(paper_trader.portfolio['trade_history'])
            st.dataframe(history_df.tail(10))
        else:
            st.info("No trade history yet")
        
        # Manual controls
        st.subheader("🔄 Manual Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reset Paper Account"):
                st.session_state.paper_portfolio = {
                    'balance': 10000.00,
                    'positions': {},
                    'trade_history': [],
                    'pnl': 0.00
                }
                st.success("Paper account reset to $10,000")
        
        with col2:
            if st.button("Update Positions"):
                current_prices = {asset_info['symbol']: current_price}
                paper_trader.update_positions(current_prices)
                st.success("Positions updated with current prices")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p><strong>⚠️ DISCLAIMER:</strong> This is a paper trading simulation for educational purposes only.</p>
    <p>Past performance does not guarantee future results. Trade at your own risk.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
