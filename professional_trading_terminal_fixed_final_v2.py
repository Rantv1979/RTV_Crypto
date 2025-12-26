import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
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

# Asset Configuration - Using yfinance symbols
ASSET_CONFIG = {
    # Cryptocurrencies
    "crypto": {
        "BTC/USD": {"symbol": "BTC-USD", "type": "crypto", "name": "Bitcoin"},
        "ETH/USD": {"symbol": "ETH-USD", "type": "crypto", "name": "Ethereum"},
        "SOL/USD": {"symbol": "SOL-USD", "type": "crypto", "name": "Solana"},
        "XRP/USD": {"symbol": "XRP-USD", "type": "crypto", "name": "Ripple"},
        "ADA/USD": {"symbol": "ADA-USD", "type": "crypto", "name": "Cardano"},
        "DOGE/USD": {"symbol": "DOGE-USD", "type": "crypto", "name": "Dogecoin"},
        "DOT/USD": {"symbol": "DOT1-USD", "type": "crypto", "name": "Polkadot"},
        "AVAX/USD": {"symbol": "AVAX-USD", "type": "crypto", "name": "Avalanche"},
    },
    
    # Forex Pairs
    "forex": {
        "EUR/USD": {"symbol": "EURUSD=X", "type": "forex", "name": "Euro/US Dollar"},
        "GBP/USD": {"symbol": "GBPUSD=X", "type": "forex", "name": "British Pound/US Dollar"},
        "USD/JPY": {"symbol": "JPY=X", "type": "forex", "name": "US Dollar/Japanese Yen"},
        "USD/CHF": {"symbol": "CHF=X", "type": "forex", "name": "US Dollar/Swiss Franc"},
        "AUD/USD": {"symbol": "AUDUSD=X", "type": "forex", "name": "Australian Dollar/US Dollar"},
        "USD/CAD": {"symbol": "CAD=X", "type": "forex", "name": "US Dollar/Canadian Dollar"},
        "NZD/USD": {"symbol": "NZDUSD=X", "type": "forex", "name": "New Zealand Dollar/US Dollar"},
        "EUR/GBP": {"symbol": "EURGBP=X", "type": "forex", "name": "Euro/British Pound"},
    },
    
    # Commodities
    "commodities": {
        "Gold": {"symbol": "GC=F", "type": "commodity", "name": "Gold Futures"},
        "Silver": {"symbol": "SI=F", "type": "commodity", "name": "Silver Futures"},
        "Crude Oil": {"symbol": "CL=F", "type": "commodity", "name": "Crude Oil WTI"},
        "Brent Oil": {"symbol": "BZ=F", "type": "commodity", "name": "Brent Crude Oil"},
        "Natural Gas": {"symbol": "NG=F", "type": "commodity", "name": "Natural Gas"},
        "Copper": {"symbol": "HG=F", "type": "commodity", "name": "Copper Futures"},
        "Platinum": {"symbol": "PL=F", "type": "commodity", "name": "Platinum Futures"},
        "Palladium": {"symbol": "PA=F", "type": "commodity", "name": "Palladium Futures"},
    },
    
    # Indices
    "indices": {
        "S&P 500": {"symbol": "^GSPC", "type": "index", "name": "S&P 500 Index"},
        "Dow Jones": {"symbol": "^DJI", "type": "index", "name": "Dow Jones Industrial"},
        "NASDAQ": {"symbol": "^IXIC", "type": "index", "name": "NASDAQ Composite"},
        "FTSE 100": {"symbol": "^FTSE", "type": "index", "name": "FTSE 100"},
        "DAX": {"symbol": "^GDAXI", "type": "index", "name": "German DAX"},
        "NIKKEI 225": {"symbol": "^N225", "type": "index", "name": "Nikkei 225"},
    }
}

# Data fetching function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(symbol, interval='1h', period='30d'):
    """Fetch data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Map period based on interval
        period_map = {
            '1m': '7d', '2m': '7d', '5m': '30d', '15m': '30d',
            '30m': '60d', '60m': '60d', '90m': '60d',
            '1h': '60d', '1d': '1y', '5d': '1y', '1wk': '2y', '1mo': '5y'
        }
        
        selected_period = period_map.get(interval, '30d')
        
        df = ticker.history(period=selected_period, interval=interval)
        
        if df.empty:
            # Alternative approach
            df = yf.download(symbol, period=selected_period, interval=interval, progress=False)
        
        # Ensure proper column names
        if not df.empty:
            if 'Open' not in df.columns and 'Close' in df.columns:
                # Rename columns if needed
                if len(df.columns) >= 4:
                    df.columns = ['Open', 'High', 'Low', 'Close'] + list(df.columns[4:])
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# SMC Calculation Functions
class SmartMoneyAnalyzer:
    def __init__(self):
        self.fvg_lookback = 5
        self.swing_period = 3
        
    def calculate_market_structure(self, df):
        """Calculate market structure including BOS and CHOCH"""
        if len(df) < 10:
            df['Swing_High'] = False
            df['Swing_Low'] = False
            df['Trend'] = 'Neutral'
            return df
            
        df = df.copy()
        
        # Calculate ATR for dynamic thresholds
        def calculate_atr(high, low, close, period=14):
            tr = pd.DataFrame(index=high.index)
            tr['HL'] = high - low
            tr['HC'] = abs(high - close.shift())
            tr['LC'] = abs(low - close.shift())
            tr['TR'] = tr[['HL', 'HC', 'LC']].max(axis=1)
            return tr['TR'].rolling(window=period).mean()
        
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        if len(atr) > 0:
            atr_value = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        else:
            atr_value = 0
        
        # Identify swing highs and lows with dynamic threshold
        threshold = max(atr_value * 0.5, df['Close'].iloc[-1] * 0.001)
        
        df['Swing_High'] = False
        df['Swing_Low'] = False
        
        for i in range(self.swing_period, len(df)-self.swing_period):
            # Check for swing high
            is_high = True
            for j in range(1, self.swing_period+1):
                if df['High'].iloc[i] <= df['High'].iloc[i-j] or df['High'].iloc[i] <= df['High'].iloc[i+j]:
                    is_high = False
                    break
            
            if is_high and df['High'].iloc[i] - df['Low'].iloc[i] > threshold:
                df.loc[df.index[i], 'Swing_High'] = True
            
            # Check for swing low
            is_low = True
            for j in range(1, self.swing_period+1):
                if df['Low'].iloc[i] >= df['Low'].iloc[i-j] or df['Low'].iloc[i] >= df['Low'].iloc[i+j]:
                    is_low = False
                    break
            
            if is_low and df['High'].iloc[i] - df['Low'].iloc[i] > threshold:
                df.loc[df.index[i], 'Swing_Low'] = True
        
        # Determine trend
        df['Trend'] = 'Neutral'
        
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Get recent swing points
            last_two_highs = swing_highs.index[-2:]
            last_two_lows = swing_lows.index[-2:]
            
            if len(last_two_highs) == 2 and len(last_two_lows) == 2:
                # Check for Higher Highs and Higher Lows (Uptrend)
                if (df.loc[last_two_highs[-1], 'High'] > df.loc[last_two_highs[0], 'High'] and
                    df.loc[last_two_lows[-1], 'Low'] > df.loc[last_two_lows[0], 'Low']):
                    df['Trend'] = 'Uptrend'
                
                # Check for Lower Highs and Lower Lows (Downtrend)
                elif (df.loc[last_two_highs[-1], 'High'] < df.loc[last_two_highs[0], 'High'] and
                      df.loc[last_two_lows[-1], 'Low'] < df.loc[last_two_lows[0], 'Low']):
                    df['Trend'] = 'Downtrend'
        
        return df
    
    def calculate_fair_value_gaps(self, df):
        """Calculate Fair Value Gaps"""
        df = df.copy()
        
        df['FVG_Bullish'] = np.nan
        df['FVG_Bearish'] = np.nan
        df['FVG_Width'] = np.nan
        df['FVG_Mid'] = np.nan
        
        for i in range(self.fvg_lookback, len(df)-1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG (Price gapped up)
            if current['High'] < next_candle['Low']:
                df.loc[df.index[i], 'FVG_Bullish'] = current['High']
                df.loc[df.index[i], 'FVG_Width'] = next_candle['Low'] - current['High']
                df.loc[df.index[i], 'FVG_Mid'] = current['High'] + (next_candle['Low'] - current['High']) / 2
            
            # Bearish FVG (Price gapped down)
            elif current['Low'] > next_candle['High']:
                df.loc[df.index[i], 'FVG_Bearish'] = current['Low']
                df.loc[df.index[i], 'FVG_Width'] = current['Low'] - next_candle['High']
                df.loc[df.index[i], 'FVG_Mid'] = next_candle['High'] + (current['Low'] - next_candle['High']) / 2
        
        return df
    
    def identify_orderblocks(self, df):
        """Identify Order Blocks"""
        df = df.copy()
        
        df['OB_Bullish'] = np.nan
        df['OB_Bearish'] = np.nan
        df['OB_Strength'] = np.nan
        
        for i in range(2, len(df)-2):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            # Calculate candle properties
            body_size = abs(current['Close'] - current['Open'])
            total_range = current['High'] - current['Low']
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Bullish Order Block
                if (current['Close'] > current['Open'] and  # Bullish candle
                    body_ratio > 0.6 and  # Strong body
                    body_size > total_range * 0.3 and  # Significant candle
                    next_candle['Low'] >= current['Low']):  # Next candle doesn't break low
                    
                    df.loc[df.index[i], 'OB_Bullish'] = current['Low']
                    df.loc[df.index[i], 'OB_Strength'] = body_ratio
                
                # Bearish Order Block
                elif (current['Close'] < current['Open'] and  # Bearish candle
                      body_ratio > 0.6 and  # Strong body
                      body_size > total_range * 0.3 and  # Significant candle
                      next_candle['High'] <= current['High']):  # Next candle doesn't break high
                    
                    df.loc[df.index[i], 'OB_Bearish'] = current['High']
                    df.loc[df.index[i], 'OB_Strength'] = body_ratio
        
        return df
    
    def identify_liquidity_levels(self, df, period=20):
        """Identify liquidity levels"""
        df = df.copy()
        
        # Recent highs and lows
        df['Recent_High'] = df['High'].rolling(window=period, min_periods=5).max()
        df['Recent_Low'] = df['Low'].rolling(window=period, min_periods=5).min()
        
        # Equal highs/lows (liquidity pools)
        df['Equal_High'] = False
        df['Equal_Low'] = False
        
        for i in range(period, len(df)):
            window_high = df['High'].iloc[i-period:i]
            window_low = df['Low'].iloc[i-period:i]
            
            if df['High'].iloc[i] >= window_high.max() * 0.999:
                df.loc[df.index[i], 'Equal_High'] = True
            
            if df['Low'].iloc[i] <= window_low.min() * 1.001:
                df.loc[df.index[i], 'Equal_Low'] = True
        
        return df
    
    def identify_supply_demand_zones(self, df, threshold_pct=1.5, zone_lookback=50):
        """Identify Supply and Demand Zones"""
        zones = []
        
        if len(df) < 20:
            return zones
        
        # Calculate percentage changes
        df['pct_change'] = df['Close'].pct_change() * 100
        
        # Find significant moves
        for i in range(zone_lookback, len(df)):
            # Look for impulse moves followed by consolidation
            if i < 5:
                continue
                
            # Check for significant bullish move (Demand Zone)
            bullish_move = False
            if (df['Close'].iloc[i] > df['Open'].iloc[i] and  # Bullish candle
                (df['Close'].iloc[i] - df['Low'].iloc[i]) > (df['High'].iloc[i] - df['Close'].iloc[i]) and  # Strong close near high
                df['pct_change'].iloc[i] > threshold_pct):
                
                # Look for consolidation after the move
                consolidation = True
                for j in range(1, min(6, len(df)-i)):
                    if abs(df['pct_change'].iloc[i+j]) > threshold_pct/2:
                        consolidation = False
                        break
                
                if consolidation:
                    zone = {
                        'type': 'Demand',
                        'start': df.index[i],
                        'high': df['High'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'strength': df['pct_change'].iloc[i],
                        'volume': df['Volume'].iloc[i] if 'Volume' in df.columns else 0
                    }
                    zones.append(zone)
            
            # Check for significant bearish move (Supply Zone)
            bearish_move = False
            if (df['Close'].iloc[i] < df['Open'].iloc[i] and  # Bearish candle
                (df['High'].iloc[i] - df['Close'].iloc[i]) > (df['Close'].iloc[i] - df['Low'].iloc[i]) and  # Strong close near low
                df['pct_change'].iloc[i] < -threshold_pct):
                
                # Look for consolidation after the move
                consolidation = True
                for j in range(1, min(6, len(df)-i)):
                    if abs(df['pct_change'].iloc[i+j]) > threshold_pct/2:
                        consolidation = False
                        break
                
                if consolidation:
                    zone = {
                        'type': 'Supply',
                        'start': df.index[i],
                        'high': df['High'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'strength': abs(df['pct_change'].iloc[i]),
                        'volume': df['Volume'].iloc[i] if 'Volume' in df.columns else 0
                    }
                    zones.append(zone)
        
        # Keep only recent zones
        if zones:
            zones = zones[-10:]  # Keep last 10 zones
        
        return zones
    
    def calculate_volume_profile(self, df, bins=20):
        """Calculate Volume Profile for the chart"""
        if 'Volume' not in df.columns:
            return None
        
        price_range = df['High'].max() - df['Low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        for i in range(len(df)):
            price_level = round(df['Low'].iloc[i] / bin_size) * bin_size
            volume_profile[price_level] = volume_profile.get(price_level, 0) + df['Volume'].iloc[i]
        
        return volume_profile
    
    def generate_signals(self, df, zones):
        """Generate trading signals based on SMC confluence"""
        signals = []
        
        if len(df) < 10:
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = latest['Close']
        trend = latest.get('Trend', 'Neutral')
        
        # 1. FVG Signals
        if not pd.isna(latest.get('FVG_Bullish', np.nan)):
            fvg_top = latest['FVG_Bullish'] + latest.get('FVG_Width', 0)
            if latest['Low'] <= fvg_top <= latest['High']:
                signals.append({
                    'type': 'BUY',
                    'concept': 'FVG Fill',
                    'entry': latest['FVG_Bullish'] + latest.get('FVG_Width', 0) * 0.5,
                    'stop_loss': latest['FVG_Bullish'] * 0.995,
                    'take_profit': latest['FVG_Bullish'] * 1.02,
                    'confidence': 'Medium',
                    'description': 'Price filling bullish Fair Value Gap'
                })
        
        if not pd.isna(latest.get('FVG_Bearish', np.nan)):
            fvg_bottom = latest['FVG_Bearish'] - latest.get('FVG_Width', 0)
            if latest['Low'] <= fvg_bottom <= latest['High']:
                signals.append({
                    'type': 'SELL',
                    'concept': 'FVG Fill',
                    'entry': latest['FVG_Bearish'] - latest.get('FVG_Width', 0) * 0.5,
                    'stop_loss': latest['FVG_Bearish'] * 1.005,
                    'take_profit': latest['FVG_Bearish'] * 0.98,
                    'confidence': 'Medium',
                    'description': 'Price filling bearish Fair Value Gap'
                })
        
        # 2. Order Block Signals
        if not pd.isna(latest.get('OB_Bullish', np.nan)):
            ob_price = latest['OB_Bullish']
            if latest['Low'] <= ob_price <= latest['High'] and trend in ['Uptrend', 'Neutral']:
                signals.append({
                    'type': 'BUY',
                    'concept': 'Order Block',
                    'entry': ob_price,
                    'stop_loss': ob_price * 0.99,
                    'take_profit': ob_price * 1.03,
                    'confidence': 'High',
                    'description': 'Price at bullish Order Block in uptrend'
                })
        
        if not pd.isna(latest.get('OB_Bearish', np.nan)):
            ob_price = latest['OB_Bearish']
            if latest['Low'] <= ob_price <= latest['High'] and trend in ['Downtrend', 'Neutral']:
                signals.append({
                    'type': 'SELL',
                    'concept': 'Order Block',
                    'entry': ob_price,
                    'stop_loss': ob_price * 1.01,
                    'take_profit': ob_price * 0.97,
                    'confidence': 'High',
                    'description': 'Price at bearish Order Block in downtrend'
                })
        
        # 3. Supply/Demand Zone Signals
        for zone in zones[-3:]:  # Check last 3 zones
            if zone['low'] <= current_price <= zone['high']:
                if zone['type'] == 'Demand' and trend in ['Uptrend', 'Neutral']:
                    signals.append({
                        'type': 'BUY',
                        'concept': 'Demand Zone',
                        'entry': current_price,
                        'stop_loss': zone['low'] * 0.995,
                        'take_profit': zone['high'] * 1.02,
                        'confidence': 'High',
                        'description': f'Price in Demand Zone (Strength: {zone["strength"]:.1f}%)'
                    })
                elif zone['type'] == 'Supply' and trend in ['Downtrend', 'Neutral']:
                    signals.append({
                        'type': 'SELL',
                        'concept': 'Supply Zone',
                        'entry': current_price,
                        'stop_loss': zone['high'] * 1.005,
                        'take_profit': zone['low'] * 0.98,
                        'confidence': 'High',
                        'description': f'Price in Supply Zone (Strength: {zone["strength"]:.1f}%)'
                    })
        
        # 4. Liquidity Grab Signals
        recent_high = latest.get('Recent_High', df['High'].iloc[-20:].max())
        recent_low = latest.get('Recent_Low', df['Low'].iloc[-20:].min())
        
        if latest['High'] > recent_high * 1.002:  # 0.2% above recent high
            signals.append({
                'type': 'SELL',
                'concept': 'Liquidity Grab',
                'entry': recent_high * 1.001,
                'stop_loss': latest['High'] * 1.005,
                'take_profit': recent_high * 0.99,
                'confidence': 'Medium',
                'description': 'Likely liquidity grab above recent high'
            })
        
        if latest['Low'] < recent_low * 0.998:  # 0.2% below recent low
            signals.append({
                'type': 'BUY',
                'concept': 'Liquidity Grab',
                'entry': recent_low * 0.999,
                'stop_loss': latest['Low'] * 0.995,
                'take_profit': recent_low * 1.01,
                'confidence': 'Medium',
                'description': 'Likely liquidity grab below recent low'
            })
        
        return signals
    
    def calculate_confluence_score(self, df, signals):
        """Calculate confluence score for each signal"""
        for signal in signals:
            score = 0
            
            # Base score for concept
            concept_scores = {
                'Order Block': 3,
                'Supply Zone': 3,
                'Demand Zone': 3,
                'FVG Fill': 2,
                'Liquidity Grab': 2
            }
            
            score += concept_scores.get(signal['concept'], 1)
            
            # Trend alignment bonus
            trend = df['Trend'].iloc[-1]
            if (signal['type'] == 'BUY' and trend == 'Uptrend') or \
               (signal['type'] == 'SELL' and trend == 'Downtrend'):
                score += 2
            
            # Volume confirmation bonus
            if 'Volume' in df.columns:
                avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
                current_volume = df['Volume'].iloc[-1]
                if current_volume > avg_volume * 1.2:
                    score += 1
            
            signal['confluence_score'] = min(10, score)
            signal['rating'] = 'High' if score >= 7 else 'Medium' if score >= 5 else 'Low'
        
        return signals

# Main Application
def main():
    st.title("📊 SMC Multi-Asset Trading Dashboard")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Asset Category Selection
    asset_category = st.sidebar.selectbox(
        "Asset Category",
        ["Cryptocurrencies", "Forex", "Commodities", "Indices"]
    )
    
    # Map category to key
    category_map = {
        "Cryptocurrencies": "crypto",
        "Forex": "forex",
        "Commodities": "commodities",
        "Indices": "indices"
    }
    
    category_key = category_map[asset_category]
    assets = ASSET_CONFIG[category_key]
    
    # Asset selection
    asset_names = list(assets.keys())
    selected_asset = st.sidebar.selectbox(
        "Select Asset",
        asset_names,
        index=0
    )
    
    asset_info = assets[selected_asset]
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"],
        index=4
    )
    
    # SMC Parameters
    st.sidebar.subheader("SMC Parameters")
    fvg_period = st.sidebar.slider("FVG Lookback", 3, 20, 5)
    swing_period = st.sidebar.slider("Swing Period", 2, 10, 3)
    zone_threshold = st.sidebar.slider("Zone Threshold %", 0.5, 5.0, 1.5)
    liquidity_period = st.sidebar.slider("Liquidity Period", 10, 100, 20)
    
    # Display Options
    st.sidebar.subheader("Display Options")
    show_fvg = st.sidebar.checkbox("Show FVGs", True)
    show_ob = st.sidebar.checkbox("Show Order Blocks", True)
    show_zones = st.sidebar.checkbox("Show Supply/Demand Zones", True)
    show_liquidity = st.sidebar.checkbox("Show Liquidity Levels", True)
    show_volume = st.sidebar.checkbox("Show Volume Profile", True)
    
    # Fetch data
    with st.spinner(f"Fetching {selected_asset} data..."):
        df = fetch_data(asset_info['symbol'], timeframe)
    
    if df.empty or len(df) < 20:
        st.error(f"Insufficient data for {selected_asset}. Please try a different timeframe or asset.")
        return
    
    # Initialize analyzer
    analyzer = SmartMoneyAnalyzer()
    analyzer.fvg_lookback = fvg_period
    analyzer.swing_period = swing_period
    
    # Calculate SMC indicators
    with st.spinner("Calculating SMC indicators..."):
        df = analyzer.calculate_market_structure(df)
        df = analyzer.calculate_fair_value_gaps(df)
        df = analyzer.identify_orderblocks(df)
        df = analyzer.identify_liquidity_levels(df, liquidity_period)
        zones = analyzer.identify_supply_demand_zones(df, zone_threshold)
        
        # Generate signals
        signals = analyzer.generate_signals(df, zones)
        signals = analyzer.calculate_confluence_score(df, signals)
        
        # Calculate volume profile
        volume_profile = analyzer.calculate_volume_profile(df)
    
    # Display dashboard metrics
    st.subheader(f"{asset_info['name']} ({selected_asset})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100
        
        st.metric(
            label="Current Price",
            value=f"${current_price:,.4f}",
            delta=f"{change_pct:+.2f}%"
        )
    
    with col2:
        # Calculate daily range
        if timeframe == '1d' and len(df) >= 1:
            daily_high = df['High'].iloc[-1]
            daily_low = df['Low'].iloc[-1]
        else:
            # For intraday, use last 24 candles or appropriate period
            lookback = min(24, len(df))
            daily_high = df['High'].iloc[-lookback:].max()
            daily_low = df['Low'].iloc[-lookback:].min()
        
        range_pct = ((daily_high - daily_low) / daily_low) * 100
        st.metric(
            label="Daily Range",
            value=f"${daily_low:,.2f} - ${daily_high:,.2f}",
            delta=f"{range_pct:.1f}%"
        )
    
    with col3:
        trend = df['Trend'].iloc[-1]
        trend_icon = "🔼" if trend == 'Uptrend' else "🔽" if trend == 'Downtrend' else "⚪"
        st.metric(
            label="Market Trend",
            value=f"{trend_icon} {trend}"
        )
    
    with col4:
        # Count active SMC signals
        active_signals = len(signals)
        st.metric(
            label="Active Signals",
            value=active_signals
        )
    
    # Chart Section
    st.subheader("📈 Price Chart with SMC Analysis")
    
    # Create chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{selected_asset} - {timeframe}', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add FVGs
    if show_fvg:
        fvg_bullish = df[~pd.isna(df['FVG_Bullish'])]
        fvg_bearish = df[~pd.isna(df['FVG_Bearish'])]
        
        for idx in fvg_bullish.index[-10:]:  # Show last 10 FVGs
            fig.add_shape(
                type="rect",
                x0=idx,
                x1=df.index[-1],
                y0=df.loc[idx, 'FVG_Bullish'],
                y1=df.loc[idx, 'FVG_Bullish'] + df.loc[idx, 'FVG_Width'],
                fillcolor="rgba(0,255,0,0.15)",
                line=dict(width=1, color="green"),
                row=1, col=1,
                name="Bullish FVG"
            )
        
        for idx in fvg_bearish.index[-10:]:  # Show last 10 FVGs
            fig.add_shape(
                type="rect",
                x0=idx,
                x1=df.index[-1],
                y0=df.loc[idx, 'FVG_Bearish'] - df.loc[idx, 'FVG_Width'],
                y1=df.loc[idx, 'FVG_Bearish'],
                fillcolor="rgba(255,0,0,0.15)",
                line=dict(width=1, color="red"),
                row=1, col=1,
                name="Bearish FVG"
            )
    
    # Add Order Blocks
    if show_ob:
        ob_bullish = df[~pd.isna(df['OB_Bullish'])]
        ob_bearish = df[~pd.isna(df['OB_Bearish'])]
        
        if not ob_bullish.empty:
            fig.add_trace(
                go.Scatter(
                    x=ob_bullish.index[-5:],  # Show last 5
                    y=ob_bullish['OB_Bullish'].iloc[-5:],
                    mode='markers',
                    name='Bullish OB',
                    marker=dict(
                        color='darkgreen',
                        size=12,
                        symbol='square',
                        line=dict(width=2, color='white')
                    )
                ),
                row=1, col=1
            )
        
        if not ob_bearish.empty:
            fig.add_trace(
                go.Scatter(
                    x=ob_bearish.index[-5:],  # Show last 5
                    y=ob_bearish['OB_Bearish'].iloc[-5:],
                    mode='markers',
                    name='Bearish OB',
                    marker=dict(
                        color='darkred',
                        size=12,
                        symbol='square',
                        line=dict(width=2, color='white')
                    )
                ),
                row=1, col=1
            )
    
    # Add Supply/Demand Zones
    if show_zones and zones:
        for zone in zones[-3:]:  # Show last 3 zones
            color = 'rgba(0,255,0,0.1)' if zone['type'] == 'Demand' else 'rgba(255,0,0,0.1)'
            line_color = 'green' if zone['type'] == 'Demand' else 'red'
            
            fig.add_shape(
                type="rect",
                x0=zone['start'],
                x1=df.index[-1],
                y0=zone['low'],
                y1=zone['high'],
                fillcolor=color,
                line=dict(color=line_color, width=1, dash='dash'),
                row=1, col=1,
                name=f"{zone['type']} Zone"
            )
    
    # Add Liquidity Levels
    if show_liquidity:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Recent_High'],
                mode='lines',
                name='Recent High',
                line=dict(color='orange', width=1, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Recent_Low'],
                mode='lines',
                name='Recent Low',
                line=dict(color='purple', width=1, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add volume
    if 'Volume' in df.columns:
        colors = ['#ef5350' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#26a69a' 
                 for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading Signals Section
    st.subheader("🚦 Trading Signals")
    
    if signals:
        # Sort signals by confluence score
        signals = sorted(signals, key=lambda x: x['confluence_score'], reverse=True)
        
        cols = st.columns(min(3, len(signals)))
        for idx, signal in enumerate(signals[:6]):  # Show top 6 signals
            with cols[idx % 3]:
                if signal['type'] == 'BUY':
                    st.markdown(f"""
                    <div class="signal-bullish">
                        <h4>📈 {signal['type']} - {signal['concept']}</h4>
                        <p><strong>Rating:</strong> {signal['rating']} ({signal['confluence_score']}/10)</p>
                        <p><strong>Entry:</strong> ${signal['entry']:.4f}</p>
                        <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.4f}</p>
                        <p><strong>Take Profit:</strong> ${signal['take_profit']:.4f}</p>
                        <p><small>{signal['description']}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-bearish">
                        <h4>📉 {signal['type']} - {signal['concept']}</h4>
                        <p><strong>Rating:</strong> {signal['rating']} ({signal['confluence_score']}/10)</p>
                        <p><strong>Entry:</strong> ${signal['entry']:.4f}</p>
                        <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.4f}</p>
                        <p><strong>Take Profit:</strong> ${signal['take_profit']:.4f}</p>
                        <p><small>{signal['description']}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No active trading signals detected for current market conditions.")
    
    # Market Analysis Section
    st.subheader("📊 Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Market Structure")
        
        # Display swing points
        swing_highs = df[df['Swing_High']]
        swing_lows = df[df['Swing_Low']]
        
        if len(swing_highs) > 0:
            last_swing_high = swing_highs.index[-1]
            st.metric("Last Swing High", 
                     f"${df.loc[last_swing_high, 'High']:.4f}",
                     delta=f"{last_swing_high.strftime('%Y-%m-%d')}")
        
        if len(swing_lows) > 0:
            last_swing_low = swing_lows.index[-1]
            st.metric("Last Swing Low", 
                     f"${df.loc[last_swing_low, 'Low']:.4f}",
                     delta=f"{last_swing_low.strftime('%Y-%m-%d')}")
        
        # Display FVG count
        fvg_count = len(df[~pd.isna(df['FVG_Bullish'])]) + len(df[~pd.isna(df['FVG_Bearish'])])
        st.metric("Active FVGs", fvg_count)
        
        # Display OB count
        ob_count = len(df[~pd.isna(df['OB_Bullish'])]) + len(df[~pd.isna(df['OB_Bearish'])])
        st.metric("Order Blocks", ob_count)
    
    with col2:
        st.markdown("### Risk Metrics")
        
        # Calculate volatility (ATR)
        def calculate_simple_atr(df, period=14):
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(period).mean()
        
        atr = calculate_simple_atr(df)
        if len(atr) > 0 and not pd.isna(atr.iloc[-1]):
            atr_value = atr.iloc[-1]
            atr_pct = (atr_value / current_price) * 100
            
            if atr_pct < 1:
                risk_level = "Low"
                color = "green"
            elif atr_pct < 3:
                risk_level = "Medium"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"
            
            st.metric("Volatility (ATR %)", f"{atr_pct:.2f}%", risk_level)
        
        # Calculate distance to liquidity
        recent_high = df['Recent_High'].iloc[-1] if 'Recent_High' in df.columns else df['High'].iloc[-20:].max()
        recent_low = df['Recent_Low'].iloc[-1] if 'Recent_Low' in df.columns else df['Low'].iloc[-20:].min()
        
        dist_to_high = ((recent_high - current_price) / current_price) * 100
        dist_to_low = ((current_price - recent_low) / current_price) * 100
        
        st.metric("To Recent High", f"{dist_to_high:.2f}%")
        st.metric("To Recent Low", f"{dist_to_low:.2f}%")
    
    # Data Table
    st.subheader("📋 Processed Data")
    
    display_cols = ['Open', 'High', 'Low', 'Close']
    if 'Volume' in df.columns:
        display_cols.append('Volume')
    if 'Trend' in df.columns:
        display_cols.append('Trend')
    
    st.dataframe(df[display_cols].tail(20), use_container_width=True)
    
    # Strategy Guidelines
    with st.expander("📚 SMC Trading Strategy Guidelines"):
        st.markdown("""
        ## Smart Money Concepts (SMC) Trading Rules
        
        ### Core Concepts:
        
        1. **Market Structure (BOS/CHOCH)**
           - Break of Structure (BOS): Price breaks key support/resistance
           - Change of Character (CHOCH): Trend reversal confirmation
        
        2. **Fair Value Gaps (FVG)**
           - Price voids where institutional orders accumulate
           - Bullish FVG: Gap between current high and next candle's low
           - Bearish FVG: Gap between current low and next candle's high
        
        3. **Order Blocks**
           - Areas where smart money placed significant orders
           - Bullish OB: Strong bullish candle + consolidation
           - Bearish OB: Strong bearish candle + consolidation
        
        4. **Supply & Demand Zones**
           - Areas of institutional order concentration
           - Demand Zone: After strong bullish move + consolidation
           - Supply Zone: After strong bearish move + consolidation
        
        5. **Liquidity Grabs**
           - Price moves to take out retail stops before reversing
           - Above recent highs (for shorts) or below recent lows (for longs)
        
        ### Entry Rules (Confluence Required):
        
        **High Probability Bullish Setup:**
        1. ✅ Price in Uptrend structure
        2. ✅ At Demand Zone or bullish FVG
        3. ✅ Bullish Order Block present
        4. ✅ Price above recent liquidity low
        5. ✅ Volume confirmation on entry
        
        **High Probability Bearish Setup:**
        1. ✅ Price in Downtrend structure
        2. ✅ At Supply Zone or bearish FVG
        3. ✅ Bearish Order Block present
        4. ✅ Price below recent liquidity high
        5. ✅ Volume confirmation on entry
        
        ### Risk Management:
        - **Position Size:** 1-2% of capital per trade
        - **Stop Loss:** Below Demand Zone (bullish) / Above Supply Zone (bearish)
        - **Take Profit 1:** 1:1 Risk-Reward ratio
        - **Take Profit 2:** Next liquidity level (1:2 RR)
        - **Max Daily Loss:** 5% of account
        
        ### Multi-Asset Considerations:
        
        **Cryptocurrencies:**
        - Higher volatility, adjust stop losses accordingly
        - 24/7 markets, watch for weekend gaps
        - Use larger timeframes (4h, 1d) for better structure
        
        **Forex:**
        - Session-based moves (Asian, London, NY)
        - Lower volatility, tighter stops
        - Watch for news events (NFP, CPI, Rate Decisions)
        
        **Commodities:**
        - Watch for inventory reports (EIA for oil)
        - Seasonal patterns (gold in September, oil in summer)
        - Dollar correlation (inverse for gold, oil)
        
        **Indices:**
        - Earnings season volatility
        - Macro-economic data dependent
        - Watch bond yields and VIX correlation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p><strong>Smart Money Concepts Trading Dashboard</strong> | Multi-Asset SMC Strategy</p>
        <p>Note: This is for educational purposes only. Past performance is not indicative of future results.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

