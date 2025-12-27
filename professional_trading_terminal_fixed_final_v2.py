# app.py - Smart Money Concepts Trading Dashboard
# Minimal dependencies for Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SMC Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-bullish {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        border-left: 5px solid #10B981;
        margin-bottom: 1rem;
    }
    .signal-bearish {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        border-left: 5px solid #EF4444;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Smart Money Concepts Trading Dashboard")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Asset selection
assets = {
    "Cryptocurrencies": {
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD", 
        "SOL/USD": "SOL-USD",
        "XRP/USD": "XRP-USD"
    },
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X"
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F",
        "Copper": "HG=F"
    }
}

asset_category = st.sidebar.selectbox("Asset Category", list(assets.keys()))
selected_pair = st.sidebar.selectbox("Select Asset", list(assets[asset_category].keys()))
symbol = assets[asset_category][selected_pair]

# Timeframe selection
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d", "1wk"], index=2)

# SMC Parameters
st.sidebar.header("SMC Parameters")
fvg_lookback = st.sidebar.slider("FVG Lookback", 3, 10, 5)
swing_period = st.sidebar.slider("Swing Period", 2, 5, 3)

# Display Options
st.sidebar.header("Display Options")
show_structure = st.sidebar.checkbox("Show Market Structure", True)
show_signals = st.sidebar.checkbox("Show Trading Signals", True)

# SMC Analysis Functions
def calculate_smc_indicators(df):
    """Calculate Smart Money Concepts indicators"""
    df = df.copy()
    
    # 1. Calculate Swing Highs and Lows
    df['Swing_High'] = False
    df['Swing_Low'] = False
    
    for i in range(swing_period, len(df) - swing_period):
        # Check for swing high
        if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, swing_period+1)) and \
           all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, swing_period+1)):
            df.loc[df.index[i], 'Swing_High'] = True
        
        # Check for swing low
        if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, swing_period+1)) and \
           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, swing_period+1)):
            df.loc[df.index[i], 'Swing_Low'] = True
    
    # 2. Determine Market Structure
    df['Market_Structure'] = 'Neutral'
    swing_highs = df[df['Swing_High']]
    swing_lows = df[df['Swing_Low']]
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs.index[-1] > swing_lows.index[-1]:
            if swing_highs['High'].iloc[-1] > swing_highs['High'].iloc[-2]:
                df.loc[swing_highs.index[-1]:, 'Market_Structure'] = 'Uptrend'
        else:
            if swing_lows['Low'].iloc[-1] < swing_lows['Low'].iloc[-2]:
                df.loc[swing_lows.index[-1]:, 'Market_Structure'] = 'Downtrend'
    
    # 3. Calculate Fair Value Gaps (FVG)
    df['FVG_Bullish'] = np.nan
    df['FVG_Bearish'] = np.nan
    
    for i in range(fvg_lookback, len(df)-1):
        if df['High'].iloc[i] < df['Low'].iloc[i+1]:
            df.loc[df.index[i], 'FVG_Bullish'] = df['High'].iloc[i]
        elif df['Low'].iloc[i] > df['High'].iloc[i+1]:
            df.loc[df.index[i], 'FVG_Bearish'] = df['Low'].iloc[i]
    
    # 4. Calculate Support and Resistance Levels
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    
    return df

def generate_trading_signals(df):
    """Generate trading signals based on SMC concepts"""
    signals = []
    
    if len(df) < 10:
        return signals
    
    latest = df.iloc[-1]
    
    # Check for bullish signals
    bullish_fvg = not pd.isna(latest['FVG_Bullish'])
    above_support = latest['Close'] > latest['Support']
    uptrend = latest['Market_Structure'] == 'Uptrend'
    
    if bullish_fvg and above_support:
        signals.append({
            'type': 'BUY',
            'strength': 'Strong' if uptrend else 'Medium',
            'reason': 'Bullish FVG at support level',
            'entry': latest['Close'],
            'stop_loss': latest['Support'] * 0.99,
            'take_profit': latest['Close'] * 1.02
        })
    
    # Check for bearish signals
    bearish_fvg = not pd.isna(latest['FVG_Bearish'])
    below_resistance = latest['Close'] < latest['Resistance']
    downtrend = latest['Market_Structure'] == 'Downtrend'
    
    if bearish_fvg and below_resistance:
        signals.append({
            'type': 'SELL',
            'strength': 'Strong' if downtrend else 'Medium',
            'reason': 'Bearish FVG at resistance level',
            'entry': latest['Close'],
            'stop_loss': latest['Resistance'] * 1.01,
            'take_profit': latest['Close'] * 0.98
        })
    
    return signals

# Generate synthetic data (for demonstration)
def generate_sample_data():
    """Generate sample price data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='H')
    n = len(dates)
    
    # Generate trending price with noise
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    prices = trend + noise
    
    df = pd.DataFrame({
        'Open': prices - np.random.uniform(0.5, 1.5, n),
        'High': prices + np.random.uniform(0.5, 2, n),
        'Low': prices - np.random.uniform(0.5, 2, n),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    return df

# Main app logic
def main():
    # Display asset info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Asset</h3>
            <h2>{}</h2>
        </div>
        """.format(selected_pair), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Category</h3>
            <h2>{}</h2>
        </div>
        """.format(asset_category), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Timeframe</h3>
            <h2>{}</h2>
        </div>
        """.format(timeframe), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Symbol</h3>
            <h2>{}</h2>
        </div>
        """.format(symbol), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generate and display sample data
    st.subheader("📈 Price Data Analysis")
    
    df = generate_sample_data()
    
    if not df.empty:
        # Calculate SMC indicators
        df_smc = calculate_smc_indicators(df)
        
        # Display latest price
        latest_price = df_smc['Close'].iloc[-1]
        prev_price = df_smc['Close'].iloc[-2]
        price_change = ((latest_price - prev_price) / prev_price) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${latest_price:.2f}")
        with col2:
            st.metric("24h Change", f"{price_change:.2f}%")
        with col3:
            structure = df_smc['Market_Structure'].iloc[-1]
            st.metric("Market Structure", structure)
        
        # Display price chart
        st.subheader("Price Chart")
        st.line_chart(df_smc['Close'].tail(100))
        
        # Display trading signals
        if show_signals:
            st.subheader("🚦 Trading Signals")
            signals = generate_trading_signals(df_smc)
            
            if signals:
                for signal in signals:
                    if signal['type'] == 'BUY':
                        st.markdown(f"""
                        <div class="signal-bullish">
                            <h3>📈 BUY SIGNAL ({signal['strength']})</h3>
                            <p><strong>Reason:</strong> {signal['reason']}</p>
                            <p><strong>Entry:</strong> ${signal['entry']:.2f}</p>
                            <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</p>
                            <p><strong>Take Profit:</strong> ${signal['take_profit']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="signal-bearish">
                            <h3>📉 SELL SIGNAL ({signal['strength']})</h3>
                            <p><strong>Reason:</strong> {signal['reason']}</p>
                            <p><strong>Entry:</strong> ${signal['entry']:.2f}</p>
                            <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</p>
                            <p><strong>Take Profit:</strong> ${signal['take_profit']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No trading signals detected for current market conditions.")
        
        # Display SMC Analysis
        if show_structure:
            st.subheader("🏛️ Market Structure Analysis")
            
            # Show swing points
            swing_highs = df_smc[df_smc['Swing_High']]
            swing_lows = df_smc[df_smc['Swing_Low']]
            
            col1, col2 = st.columns(2)
            with col1:
                if not swing_highs.empty:
                    st.write("**Recent Swing Highs:**")
                    for idx in swing_highs.index[-3:]:
                        st.write(f"- {idx.strftime('%Y-%m-%d %H:%M')}: ${swing_highs.loc[idx, 'High']:.2f}")
            
            with col2:
                if not swing_lows.empty:
                    st.write("**Recent Swing Lows:**")
                    for idx in swing_lows.index[-3:]:
                        st.write(f"- {idx.strftime('%Y-%m-%d %H:%M')}: ${swing_lows.loc[idx, 'Low']:.2f}")
            
            # Show FVGs
            st.subheader("⚡ Fair Value Gaps (FVG)")
            fvg_bullish = df_smc[~pd.isna(df_smc['FVG_Bullish'])]
            fvg_bearish = df_smc[~pd.isna(df_smc['FVG_Bearish'])]
            
            col1, col2 = st.columns(2)
            with col1:
                if not fvg_bullish.empty:
                    st.write("**Bullish FVGs:**")
                    for idx in fvg_bullish.index[-3:]:
                        st.write(f"- {idx.strftime('%Y-%m-%d %H:%M')}: ${fvg_bullish.loc[idx, 'FVG_Bullish']:.2f}")
            
            with col2:
                if not fvg_bearish.empty:
                    st.write("**Bearish FVGs:**")
                    for idx in fvg_bearish.index[-3:]:
                        st.write(f"- {idx.strftime('%Y-%m-%d %H:%M')}: ${fvg_bearish.loc[idx, 'FVG_Bearish']:.2f}")
        
        # Show raw data
        with st.expander("📊 View Raw Data"):
            st.dataframe(df_smc.tail(20))
    
    # SMC Education Section
    st.markdown("---")
    st.subheader("📚 Smart Money Concepts (SMC) Education")
    
    concepts = {
        "Market Structure": "Analysis of price highs and lows to determine trend direction and potential reversal points.",
        "Fair Value Gaps (FVG)": "Price voids that occur when there's a gap between candles, often targeted by institutional traders.",
        "Order Blocks": "Areas where institutional orders are concentrated, creating supply and demand imbalances.",
        "Liquidity Grabs": "Price moves designed to trigger stop losses before reversing in the intended direction.",
        "Break of Structure (BOS)": "When price breaks through key support/resistance levels, indicating trend continuation.",
        "Change of Character (CHOCH)": "When market structure shifts from uptrend to downtrend or vice versa."
    }
    
    for concept, description in concepts.items():
        with st.expander(f"**{concept}**"):
            st.write(description)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p><strong>Smart Money Concepts Trading Dashboard</strong> | Educational Purpose Only</p>
    <p>This dashboard demonstrates SMC concepts using synthetic data.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
