import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ccxt
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SMC Multi-Asset Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .asset-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .asset-card:hover {
        transform: translateY(-5px);
    }
    .crypto-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .forex-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .commodity-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .signal-bullish {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        border-left: 4px solid #10B981;
    }
    .signal-bearish {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        border-left: 4px solid #EF4444;
    }
    .concept-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .trend-up { background-color: #D1FAE5; color: #065F46; }
    .trend-down { background-color: #FEE2E2; color: #991B1B; }
    .trend-neutral { background-color: #E5E7EB; color: #374151; }
</style>
""", unsafe_allow_html=True)

# Asset Configuration
ASSET_CONFIG = {
    # Cryptocurrencies
    "crypto": {
        "BTC/USDT": {"symbol": "BTC-USD", "exchange": "binance", "type": "crypto"},
        "ETH/USDT": {"symbol": "ETH-USD", "exchange": "binance", "type": "crypto"},
        "SOL/USD": {"symbol": "SOL-USD", "exchange": "binance", "type": "crypto"},
        "XRP/USD": {"symbol": "XRP-USD", "exchange": "binance", "type": "crypto"},
        "ADA/USD": {"symbol": "ADA-USD", "exchange": "binance", "type": "crypto"},
    },
    
    # Forex Pairs
    "forex": {
        "EUR/USD": {"symbol": "EURUSD=X", "exchange": "oanda", "type": "forex"},
        "GBP/USD": {"symbol": "GBPUSD=X", "exchange": "oanda", "type": "forex"},
        "USD/JPY": {"symbol": "JPY=X", "exchange": "oanda", "type": "forex"},
        "USD/CHF": {"symbol": "CHF=X", "exchange": "oanda", "type": "forex"},
        "AUD/USD": {"symbol": "AUDUSD=X", "exchange": "oanda", "type": "forex"},
        "NZD/USD": {"symbol": "NZDUSD=X", "exchange": "oanda", "type": "forex"},
        "USD/CAD": {"symbol": "CAD=X", "exchange": "oanda", "type": "forex"},
    },
    
    # Commodities
    "commodities": {
        "Gold": {"symbol": "GC=F", "exchange": "comex", "type": "commodity"},
        "Silver": {"symbol": "SI=F", "exchange": "comex", "type": "commodity"},
        "Crude Oil": {"symbol": "CL=F", "exchange": "nymex", "type": "commodity"},
        "Natural Gas": {"symbol": "NG=F", "exchange": "nymex", "type": "commodity"},
        "Copper": {"symbol": "HG=F", "exchange": "comex", "type": "commodity"},
        "Brent Oil": {"symbol": "BZ=F", "exchange": "ice", "type": "commodity"},
    }
}

# Initialize CCXT exchanges
exchanges = {
    'binance': ccxt.binance(),
    'kraken': ccxt.kraken(),
}

# Data fetching functions
def fetch_crypto_data(symbol, timeframe='1h', limit=500):
    """Fetch cryptocurrency data from CCXT"""
    try:
        exchange = exchanges['binance']
        symbol_ccxt = symbol.replace('-USD', '/USDT')
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except:
        # Fallback to yfinance
        return fetch_yfinance_data(symbol, '1h', limit)

def fetch_yfinance_data(symbol, interval='1h', period='30d'):
    """Fetch data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            # Try with different symbol format for forex
            if '=X' in symbol:
                df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def fetch_asset_data(asset_symbol, asset_type, timeframe='1h'):
    """Fetch data based on asset type"""
    if asset_type == 'crypto':
        return fetch_crypto_data(asset_symbol, timeframe)
    else:
        # For forex and commodities, use yfinance
        period_map = {
            '1m': '7d', '5m': '30d', '15m': '30d',
            '30m': '60d', '1h': '60d', '4h': '180d',
            '1d': '1y', '1wk': '2y'
        }
        period = period_map.get(timeframe, '30d')
        return fetch_yfinance_data(asset_symbol, timeframe, period)

# SMC Calculation Functions
class SmartMoneyAnalyzer:
    def __init__(self):
        self.fvg_lookback = 5
        self.swing_period = 3
        
    def calculate_market_structure(self, df):
        """Calculate market structure including BOS and CHOCH"""
        df = df.copy()
        
        # Identify swing highs and lows
        df['Swing_High'] = False
        df['Swing_Low'] = False
        
        for i in range(self.swing_period, len(df)-self.swing_period):
            # Swing High
            if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'Swing_High'] = True
            
            # Swing Low
            if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, self.swing_period+1)) and \
               all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, self.swing_period+1)):
                df.loc[df.index[i], 'Swing_Low'] = True
        
        # Calculate Higher Highs (HH), Lower Lows (LL)
        df['HH'] = np.nan
        df['HL'] = np.nan
        df['LH'] = np.nan
        df['LL'] = np.nan
        
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        # Identify market structure
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Uptrend: Higher Highs and Higher Lows
            df['Trend'] = 'Neutral'
            last_idx = df.index[-1]
            
            # Check last two swing highs and lows
            last_swing_highs = swing_highs.index[-2:]
            last_swing_lows = swing_lows.index[-2:]
            
            if len(last_swing_highs) == 2 and len(last_swing_lows) == 2:
                if (df.loc[last_swing_highs[-1], 'High'] > df.loc[last_swing_highs[-2], 'High'] and
                    df.loc[last_swing_lows[-1], 'Low'] > df.loc[last_swing_lows[-2], 'Low']):
                    df.loc[last_idx, 'Trend'] = 'Uptrend'
                elif (df.loc[last_swing_highs[-1], 'High'] < df.loc[last_swing_highs[-2], 'High'] and
                      df.loc[last_swing_lows[-1], 'Low'] < df.loc[last_swing_lows[-2], 'Low']):
                    df.loc[last_idx, 'Trend'] = 'Downtrend'
        
        # Detect Break of Structure (BOS)
        df['BOS'] = False
        df['CHOCH'] = False
        
        return df
    
    def calculate_fair_value_gaps(self, df):
        """Calculate Fair Value Gaps"""
        df = df.copy()
        
        df['FVG_Bullish'] = np.nan
        df['FVG_Bearish'] = np.nan
        df['FVG_Width'] = np.nan
        
        for i in range(self.fvg_lookback, len(df)-1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG (Price gapped up)
            if current['High'] < next_candle['Low']:
                df.loc[df.index[i], 'FVG_Bullish'] = current['High']
                df.loc[df.index[i], 'FVG_Width'] = next_candle['Low'] - current['High']
            
            # Bearish FVG (Price gapped down)
            elif current['Low'] > next_candle['High']:
                df.loc[df.index[i], 'FVG_Bearish'] = current['Low']
                df.loc[df.index[i], 'FVG_Width'] = current['Low'] - next_candle['High']
        
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
            candle_size = abs(current['Close'] - current['Open'])
            body_ratio = candle_size / (current['High'] - current['Low']) if (current['High'] - current['Low']) > 0 else 0
            
            if (current['Close'] > current['Open'] and  # Bullish candle
                candle_size > 0 and
                body_ratio > 0.6 and  # Strong body
                next_candle['Low'] > current['Low']):  # Next candle doesn't break low
                df.loc[df.index[i], 'OB_Bullish'] = current['Low']
            
            # Bearish Order Block
            elif (current['Close'] < current['Open'] and  # Bearish candle
                  candle_size > 0 and
                  body_ratio > 0.6 and  # Strong body
                  next_candle['High'] < current['High']):  # Next candle doesn't break high
                df.loc[df.index[i], 'OB_Bearish'] = current['High']
        
        return df
    
    def identify_liquidity_levels(self, df, period=20):
        """Identify liquidity levels"""
        df = df.copy()
        
        # Recent highs and lows
        df['Recent_High'] = df['High'].rolling(window=period).max()
        df['Recent_Low'] = df['Low'].rolling(window=period).min()
        
        # Equal highs/lows (liquidity pools)
        df['Equal_High'] = df['High'] == df['High'].rolling(window=period).max()
        df['Equal_Low'] = df['Low'] == df['Low'].rolling(window=period).min()
        
        return df
    
    def identify_supply_demand_zones(self, df, threshold=1.5):
        """Identify Supply and Demand Zones"""
        zones = []
        
        # Calculate percentage changes
        df['pct_change'] = df['Close'].pct_change() * 100
        
        # Find significant moves
        significant_moves = df[abs(df['pct_change']) > threshold]
        
        for idx in significant_moves.index:
            if df.loc[idx, 'pct_change'] > threshold:
                # Demand Zone (after strong bullish move)
                zone = {
                    'type': 'Demand',
                    'start': idx,
                    'end': df.index[-1],
                    'high': df.loc[idx, 'High'],
                    'low': df.loc[idx, 'Low'],
                    'strength': df.loc[idx, 'pct_change']
                }
                zones.append(zone)
            elif df.loc[idx, 'pct_change'] < -threshold:
                # Supply Zone (after strong bearish move)
                zone = {
                    'type': 'Supply',
                    'start': idx,
                    'end': df.index[-1],
                    'high': df.loc[idx, 'High'],
                    'low': df.loc[idx, 'Low'],
                    'strength': abs(df.loc[idx, 'pct_change'])
                }
                zones.append(zone)
        
        return zones
    
    def calculate_mitigation_blocks(self, df):
        """Calculate Mitigation Blocks (failed order blocks)"""
        df = df.copy()
        
        df['Mitigation_Bullish'] = np.nan
        df['Mitigation_Bearish'] = np.nan
        
        for i in range(5, len(df)):
            # Check for broken bullish OB that gets reclaimed
            for j in range(max(0, i-10), i):
                if not pd.isna(df.loc[df.index[j], 'OB_Bullish']):
                    if df.loc[df.index[i], 'Low'] < df.loc[df.index[j], 'OB_Bullish'] and \
                       df.loc[df.index[i], 'Close'] > df.loc[df.index[j], 'OB_Bullish']:
                        df.loc[df.index[i], 'Mitigation_Bullish'] = df.loc[df.index[j], 'OB_Bullish']
            
            # Check for broken bearish OB that gets reclaimed
            for j in range(max(0, i-10), i):
                if not pd.isna(df.loc[df.index[j], 'OB_Bearish']):
                    if df.loc[df.index[i], 'High'] > df.loc[df.index[j], 'OB_Bearish'] and \
                       df.loc[df.index[i], 'Close'] < df.loc[df.index[j], 'OB_Bearish']:
                        df.loc[df.index[i], 'Mitigation_Bearish'] = df.loc[df.index[j], 'OB_Bearish']
        
        return df
    
    def generate_signals(self, df, zones):
        """Generate trading signals based on SMC confluence"""
        signals = []
        
        if len(df) < 10:
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check FVG signals
        if not pd.isna(latest.get('FVG_Bullish', np.nan)):
            if latest['Low'] <= latest['FVG_Bullish'] <= latest['High']:
                signals.append({
                    'type': 'BUY',
                    'concept': 'FVG Fill',
                    'price': latest['FVG_Bullish'],
                    'confidence': 'Medium',
                    'description': 'Price filling bullish Fair Value Gap'
                })
        
        if not pd.isna(latest.get('FVG_Bearish', np.nan)):
            if latest['Low'] <= latest['FVG_Bearish'] <= latest['High']:
                signals.append({
                    'type': 'SELL',
                    'concept': 'FVG Fill',
                    'price': latest['FVG_Bearish'],
                    'confidence': 'Medium',
                    'description': 'Price filling bearish Fair Value Gap'
                })
        
        # Check Order Block signals
        if not pd.isna(latest.get('OB_Bullish', np.nan)):
            if latest['Low'] <= latest['OB_Bullish'] <= latest['High']:
                signals.append({
                    'type': 'BUY',
                    'concept': 'Order Block',
                    'price': latest['OB_Bullish'],
                    'confidence': 'High',
                    'description': 'Price at bullish Order Block'
                })
        
        if not pd.isna(latest.get('OB_Bearish', np.nan)):
            if latest['Low'] <= latest['OB_Bearish'] <= latest['High']:
                signals.append({
                    'type': 'SELL',
                    'concept': 'Order Block',
                    'price': latest['OB_Bearish'],
                    'confidence': 'High',
                    'description': 'Price at bearish Order Block'
                })
        
        # Check Supply/Demand Zones
        current_price = latest['Close']
        for zone in zones[-5:]:  # Check last 5 zones
            if zone['low'] <= current_price <= zone['high']:
                if zone['type'] == 'Demand':
                    signals.append({
                        'type': 'BUY',
                        'concept': 'Demand Zone',
                        'price': current_price,
                        'confidence': 'High',
                        'description': f'Price in Demand Zone (Strength: {zone["strength"]:.1f}%)'
                    })
                else:
                    signals.append({
                        'type': 'SELL',
                        'concept': 'Supply Zone',
                        'price': current_price,
                        'confidence': 'High',
                        'description': f'Price in Supply Zone (Strength: {zone["strength"]:.1f}%)'
                    })
        
        # Liquidity grab signals
        if latest['High'] > latest['Recent_High'] * 1.002:  # 0.2% above recent high
            signals.append({
                'type': 'SELL',
                'concept': 'Liquidity Grab',
                'price': latest['Recent_High'],
                'confidence': 'Medium',
                'description': 'Likely liquidity grab above recent high'
            })
        
        if latest['Low'] < latest['Recent_Low'] * 0.998:  # 0.2% below recent low
            signals.append({
                'type': 'BUY',
                'concept': 'Liquidity Grab',
                'price': latest['Recent_Low'],
                'confidence': 'Medium',
                'description': 'Likely liquidity grab below recent low'
            })
        
        return signals

# Main Application
def main():
    st.title("📊 SMC Multi-Asset Trading Dashboard")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Asset Selection
    asset_category = st.sidebar.selectbox(
        "Asset Category",
        ["Cryptocurrencies", "Forex", "Commodities"]
    )
    
    # Get assets for selected category
    if asset_category == "Cryptocurrencies":
        assets = ASSET_CONFIG['crypto']
        category_key = 'crypto'
    elif asset_category == "Forex":
        assets = ASSET_CONFIG['forex']
        category_key = 'forex'
    else:
        assets = ASSET_CONFIG['commodities']
        category_key = 'commodities'
    
    # Asset selection grid
    st.sidebar.subheader("Select Asset")
    selected_asset = st.sidebar.selectbox(
        "Choose Asset",
        list(assets.keys())
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"],
        index=4
    )
    
    # SMC Parameters
    st.sidebar.subheader("SMC Parameters")
    fvg_period = st.sidebar.slider("FVG Lookback", 3, 20, 5)
    swing_period = st.sidebar.slider("Swing Period", 2, 10, 3)
    zone_threshold = st.sidebar.slider("Zone Threshold %", 0.5, 5.0, 1.5)
    
    # Display Options
    st.sidebar.subheader("Display Options")
    show_fvg = st.sidebar.checkbox("Show FVGs", True)
    show_ob = st.sidebar.checkbox("Show Order Blocks", True)
    show_zones = st.sidebar.checkbox("Show Supply/Demand Zones", True)
    show_liquidity = st.sidebar.checkbox("Show Liquidity Levels", True)
    
    # Get asset configuration
    asset_config = assets[selected_asset]
    
    # Fetch data
    with st.spinner(f"Fetching {selected_asset} data..."):
        df = fetch_asset_data(
            asset_config['symbol'],
            asset_config['type'],
            timeframe
        )
    
    if df.empty:
        st.error(f"Could not fetch data for {selected_asset}")
        return
    
    # Initialize analyzer
    analyzer = SmartMoneyAnalyzer()
    analyzer.fvg_lookback = fvg_period
    analyzer.swing_period = swing_period
    
    # Calculate SMC indicators
    with st.spinner("Calculating SMC indicators..."):
        # Calculate all indicators
        df = analyzer.calculate_market_structure(df)
        df = analyzer.calculate_fair_value_gaps(df)
        df = analyzer.identify_orderblocks(df)
        df = analyzer.identify_liquidity_levels(df)
        zones = analyzer.identify_supply_demand_zones(df, zone_threshold)
        df = analyzer.calculate_mitigation_blocks(df)
        
        # Generate signals
        signals = analyzer.generate_signals(df, zones)
    
    # Display dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        st.metric(
            label="Current Price",
            value=f"{current_price:,.4f}",
            delta=f"{change_pct:.2f}%"
        )
    
    with col2:
        high_24h = df['High'].iloc[-24:].max() if len(df) >= 24 else df['High'].max()
        low_24h = df['Low'].iloc[-24:].min() if len(df) >= 24 else df['Low'].min()
        
        st.metric(
            label="24h Range",
            value=f"{low_24h:,.4f} - {high_24h:,.4f}",
            delta=f"{(high_24h - low_24h)/current_price*100:.1f}%"
        )
    
    with col3:
        volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
        avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
        
        st.metric(
            label="Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_ratio:.1f}x avg"
        )
    
    with col4:
        trend = df['Trend'].iloc[-1] if 'Trend' in df.columns else 'Neutral'
        trend_color = {
            'Uptrend': 'green',
            'Downtrend': 'red',
            'Neutral': 'gray'
        }.get(trend, 'gray')
        
        st.metric(
            label="Market Structure",
            value=trend,
            delta=""
        )
    
    # Chart Section
    st.subheader(f"{selected_asset} - SMC Analysis")
    
    # Create chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{selected_asset} Price Chart', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add FVGs
    if show_fvg:
        fvg_bullish = df[~pd.isna(df['FVG_Bullish'])]
        fvg_bearish = df[~pd.isna(df['FVG_Bearish'])]
        
        for idx in fvg_bullish.index:
            fig.add_shape(
                type="rect",
                x0=idx,
                x1=df.index[-1],
                y0=df.loc[idx, 'FVG_Bullish'],
                y1=df.loc[idx, 'FVG_Bullish'] + df.loc[idx, 'FVG_Width'],
                fillcolor="rgba(0,255,0,0.2)",
                line=dict(width=0),
                row=1, col=1
            )
        
        for idx in fvg_bearish.index:
            fig.add_shape(
                type="rect",
                x0=idx,
                x1=df.index[-1],
                y0=df.loc[idx, 'FVG_Bearish'] - df.loc[idx, 'FVG_Width'],
                y1=df.loc[idx, 'FVG_Bearish'],
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(width=0),
                row=1, col=1
            )
    
    # Add Order Blocks
    if show_ob:
        ob_bullish = df[~pd.isna(df['OB_Bullish'])]
        ob_bearish = df[~pd.isna(df['OB_Bearish'])]
        
        fig.add_trace(
            go.Scatter(
                x=ob_bullish.index,
                y=ob_bullish['OB_Bullish'],
                mode='markers',
                name='Bullish OB',
                marker=dict(color='darkgreen', size=12, symbol='square')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=ob_bearish.index,
                y=ob_bearish['OB_Bearish'],
                mode='markers',
                name='Bearish OB',
                marker=dict(color='darkred', size=12, symbol='square')
            ),
            row=1, col=1
        )
    
    # Add Supply/Demand Zones
    if show_zones:
        for zone in zones[-3:]:  # Show last 3 zones
            color = 'rgba(0,255,0,0.1)' if zone['type'] == 'Demand' else 'rgba(255,0,0,0.1)'
            line_color = 'green' if zone['type'] == 'Demand' else 'red'
            
            fig.add_shape(
                type="rect",
                x0=zone['start'],
                x1=zone['end'],
                y0=zone['low'],
                y1=zone['high'],
                fillcolor=color,
                line=dict(color=line_color, width=1),
                row=1, col=1
            )
    
    # Add Liquidity Levels
    if show_liquidity:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Recent_High'],
                mode='lines',
                name='Recent High',
                line=dict(color='orange', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Recent_Low'],
                mode='lines',
                name='Recent Low',
                line=dict(color='purple', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Add volume
    if 'Volume' in df.columns:
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signals Section
    st.subheader("🚦 Trading Signals")
    
    if signals:
        cols = st.columns(min(3, len(signals)))
        for idx, signal in enumerate(signals):
            with cols[idx % 3]:
                if signal['type'] == 'BUY':
                    st.markdown(f"""
                    <div class="signal-bullish">
                        <h4>📈 {signal['type']} Signal</h4>
                        <p><strong>Concept:</strong> {signal['concept']}</p>
                        <p><strong>Price:</strong> {signal['price']:.4f}</p>
                        <p><strong>Confidence:</strong> {signal['confidence']}</p>
                        <p>{signal['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-bearish">
                        <h4>📉 {signal['type']} Signal</h4>
                        <p><strong>Concept:</strong> {signal['concept']}</p>
                        <p><strong>Price:</strong> {signal['price']:.4f}</p>
                        <p><strong>Confidence:</strong> {signal['confidence']}</p>
                        <p>{signal['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No active trading signals detected.")
    
    # Market Structure Analysis
    st.subheader("🏛️ Market Structure Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Trend Analysis")
        trend = df['Trend'].iloc[-1] if 'Trend' in df.columns else 'Neutral'
        if trend == 'Uptrend':
            st.markdown('<span class="concept-badge trend-up">🔼 Uptrend</span>', unsafe_allow_html=True)
        elif trend == 'Downtrend':
            st.markdown('<span class="concept-badge trend-down">🔽 Downtrend</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="concept-badge trend-neutral">⚪ Range</span>', unsafe_allow_html=True)
        
        # Structure levels
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        if len(swing_highs) > 0:
            last_swing_high = swing_highs['High'].iloc[-1]
            st.metric("Last Swing High", f"{last_swing_high:.4f}")
        
        if len(swing_lows) > 0:
            last_swing_low = swing_lows['Low'].iloc[-1]
            st.metric("Last Swing Low", f"{last_swing_low:.4f}")
    
    with col2:
        st.markdown("### SMC Concepts Detected")
        
        concepts = []
        if not df['FVG_Bullish'].isna().all() or not df['FVG_Bearish'].isna().all():
            concepts.append("Fair Value Gaps")
        if not df['OB_Bullish'].isna().all() or not df['OB_Bearish'].isna().all():
            concepts.append("Order Blocks")
        if zones:
            concepts.append("Supply/Demand Zones")
        if not df['Mitigation_Bullish'].isna().all() or not df['Mitigation_Bearish'].isna().all():
            concepts.append("Mitigation Blocks")
        
        for concept in concepts:
            st.markdown(f'- {concept}')
    
    with col3:
        st.markdown("### Risk Levels")
        
        # Calculate volatility
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        volatility_pct = (atr / current_price) * 100
        
        if volatility_pct < 1:
            risk_level = "Low"
            color = "green"
        elif volatility_pct < 3:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "High"
            color = "red"
        
        st.metric("Volatility (ATR %)", f"{volatility_pct:.2f}%", risk_level)
        
        # Liquidity distance
        dist_to_high = ((df['Recent_High'].iloc[-1] - current_price) / current_price) * 100
        dist_to_low = ((current_price - df['Recent_Low'].iloc[-1]) / current_price) * 100
        
        st.metric("Dist to Liquidity High", f"{dist_to_high:.2f}%")
        st.metric("Dist to Liquidity Low", f"{dist_to_low:.2f}%")
    
    # Asset Comparison
    st.subheader("📊 Asset Comparison")
    
    # Get comparison data for category
    comparison_assets = list(assets.keys())[:5]  # First 5 assets
    comparison_data = []
    
    for asset_name in comparison_assets:
        asset_cfg = assets[asset_name]
        try:
            comp_df = fetch_asset_data(
                asset_cfg['symbol'],
                asset_cfg['type'],
                '1d'
            )
            if not comp_df.empty:
                change = ((comp_df['Close'].iloc[-1] - comp_df['Close'].iloc[-2]) / 
                         comp_df['Close'].iloc[-2]) * 100
                comparison_data.append({
                    'Asset': asset_name,
                    'Price': comp_df['Close'].iloc[-1],
                    'Change %': change,
                    'Type': asset_cfg['type']
                })
        except:
            continue
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True)
    
    # Strategy Guidelines
    with st.expander("📖 SMC Trading Strategy Guidelines"):
        st.markdown("""
        ## Smart Money Concepts Trading Rules
        
        ### Entry Conditions (Confluence Required):
        
        **Bullish Setup:**
        1. ✅ Price in or approaching Demand Zone
        2. ✅ Bullish FVG being filled
        3. ✅ Bullish Order Block present
        4. ✅ Price above recent liquidity low (no break)
        5. ✅ Market structure: Higher Highs & Higher Lows
        
        **Bearish Setup:**
        1. ✅ Price in or approaching Supply Zone
        2. ✅ Bearish FVG being filled
        3. ✅ Bearish Order Block present
        4. ✅ Price below recent liquidity high (no break)
        5. ✅ Market structure: Lower Highs & Lower Lows
        
        ### Risk Management:
        - Stop Loss: Below Demand Zone (bullish) / Above Supply Zone (bearish)
        - Take Profit 1: 1:1 Risk-Reward
        - Take Profit 2: Next liquidity level
        - Position Size: 1-2% of capital per trade
        
        ### Key Concepts:
        - **Fair Value Gaps (FVG):** Price voids that often get filled
        - **Order Blocks:** Areas where institutional orders were placed
        - **Liquidity Grabs:** Price moves to take out stops before reversing
        - **Mitigation Blocks:** Failed order blocks that get reclaimed
        - **BOS/CHOCH:** Break of Structure / Change of Character
        """)

if __name__ == "__main__":
    main()
