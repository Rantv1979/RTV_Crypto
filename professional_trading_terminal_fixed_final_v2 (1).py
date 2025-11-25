# final_trading_terminal_single_file.py
# Single-file Streamlit trading terminal
# Features:
# - Market coverage: Cryptos, Forex, Commodities (user requested symbols)
# - Signals auto-refresh every 120 seconds (2 minutes)
# - Price sections refresh faster (every 30 seconds)
# - Fixed allocation $1000 per trade
# - Prevent duplicate trade executions using unique signal IDs stored in session_state
# - Readable UI fonts, multi-colour tabs, mood gauge (needle-style via plotly)
# - Trade History tab, improved paper trading with PnL and support/resistance
# NOTE: This is a self-contained demo that uses yfinance for prices and simple strategies.

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import uuid

# ----- CONFIG -----
FIXED_ALLOCATION = 1000.0  # $1000 per trade
SIGNAL_REFRESH_SECONDS = 120  # 2 minutes for signals
PRICE_REFRESH_SECONDS = 30  # 30 seconds for price refresh (as requested)

# Market coverage
MARKETS = {
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F"]  # Gold, Silver, Crude Oil, Natural Gas
}

TIMEFRAMES = ["15m", "1h", "4h"]

# ----- UI Styling (readable fonts) -----
st.set_page_config(page_title="Unified Trading Terminal", layout="wide", page_icon="📈")
st.markdown(
    """
    <style>
    /* Readable system fonts for clarity */
    html, body, [class*="css"]  { font-family: Inter, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
    .stButton>button { font-weight: 700; }
    .main-title { font-size: 24px; font-weight: 800; }
    .sub-title { font-size: 16px; color: #666; }
    .small { font-size: 13px; color: #777; }
    /* Tabs color */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; padding: 6px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { background: linear-gradient(90deg,#667eea,#764ba2); border-radius: 8px; color: white; padding: 8px 12px; font-weight:700; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg,#FF6B6B,#4ECDC4) !important; color: white !important; }
    /* Auto-refresh counter */
    .refresh-counter {
        background: #1e3a8a;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 8px;
    }
    .mood-gauge-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">🚀 Unified Trading Terminal — Single File</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Market coverage: Crypto · Forex · Commodities — Prices auto-refresh: 30 seconds</div>', unsafe_allow_html=True)
st.write("")

# ----- Session state initialization -----
if "last_signal_refresh" not in st.session_state:
    st.session_state.last_signal_refresh = 0.0
if "last_price_refresh" not in st.session_state:
    st.session_state.last_price_refresh = 0.0
if "signals" not in st.session_state:
    st.session_state.signals = []  # list of dict signals
if "executed_signal_ids" not in st.session_state:
    st.session_state.executed_signal_ids = set()  # track executed signals to avoid duplicates
if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []  # store paper trades
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []  # store closed trades
if "balance" not in st.session_state:
    st.session_state.balance = 20000.0  # demo starting balance
if "last_prices" not in st.session_state:
    st.session_state.last_prices = {}
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Live Dashboard"

# ----- Enhanced Auto-Refresh Strategy -----
def setup_auto_refresh():
    """Enhanced auto-refresh strategy with manual override"""
    st.session_state.refresh_count += 1
    
    # Display refresh counter
    st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 14px;'>Refresh Count: <span class='refresh-counter'>{st.session_state.refresh_count}</span></div>", unsafe_allow_html=True)
    
    # Manual refresh controls
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("📊 Update Prices", use_container_width=True):
            st.rerun()
    
    # Auto-refresh logic
    now = time.time()
    signal_refresh_needed = (now - st.session_state.last_signal_refresh) >= SIGNAL_REFRESH_SECONDS
    price_refresh_needed = (now - st.session_state.last_price_refresh) >= PRICE_REFRESH_SECONDS
    
    return signal_refresh_needed, price_refresh_needed, now

# ----- Helpers -----
def fetch_latest_close(symbol: str, period: str = "7d", interval: str = "15m"):
    """Fetch latest close price using yfinance (cached in session during run)"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None, pd.DataFrame()
        price = float(df['Close'].iloc[-1])
        return price, df
    except Exception as e:
        return None, pd.DataFrame()

def generate_signal_id(symbol: str, strategy: str, timeframe: str, entry: float):
    """Generate a stable-ish unique ID for a signal so duplicates across refreshes can be recognized"""
    base = f"{symbol}|{strategy}|{timeframe}|{round(entry, 6)}"
    # Use uuid5 with namespace for stable deterministic id per same entry
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

def create_mood_gauge(score: float, title: str, price_display: str = ""):
    """Create a needle gauge with plotly matching the provided design"""
    # Define colors based on score ranges
    if score <= 25:
        gauge_color = '#ef4444'  # Red for Extreme Fear
        mood_text = "EXTREME FEAR"
    elif score <= 45:
        gauge_color = '#f97316'  # Orange for Fear
        mood_text = "FEAR"
    elif score <= 55:
        gauge_color = '#fbbf24'  # Yellow for Neutral
        mood_text = "NEUTRAL"
    elif score <= 75:
        gauge_color = '#84cc16'  # Light green for Greed
        mood_text = "GREED"
    else:
        gauge_color = '#10b981'  # Green for Extreme Greed
        mood_text = "EXTREME GREED"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "  ", 'font': {'size': 24}},
        title={'text': f"<br><span style='font-size:14px;color:gray'>{title}</span>", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#fef2f2'},
                {'range': [25, 45], 'color': '#fffbeb'},
                {'range': [45, 55], 'color': '#f0fdf4'},
                {'range': [55, 75], 'color': '#ecfdf5'},
                {'range': [75, 100], 'color': '#dcfce7'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score}
        }
    ))
    
    # Add mood text annotation
    fig.add_annotation(
        x=0.5, y=0.3,
        text=f"<b>{mood_text}</b>",
        showarrow=False,
        font=dict(size=16, color=gauge_color),
        xref="paper", yref="paper"
    )
    
    # Add price display if provided
    if price_display:
        fig.add_annotation(
            x=0.5, y=0.15,
            text=f"<b>{price_display}</b>",
            showarrow=False,
            font=dict(size=14, color="black"),
            xref="paper", yref="paper"
        )
    
    fig.update_layout(
        height=280, 
        margin=dict(t=60, b=40, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def calculate_support_resistance(df, window=20):
    """Calculate simple support and resistance levels"""
    if len(df) < window:
        return None, None
    
    high = df['High'].rolling(window=window).max()
    low = df['Low'].rolling(window=window).min()
    
    resistance = float(high.iloc[-1]) if not pd.isna(high.iloc[-1]) else None
    support = float(low.iloc[-1]) if not pd.isna(low.iloc[-1]) else None
    
    return support, resistance

def calculate_pnl(trade, current_price):
    """Calculate PnL for a trade"""
    if trade['action'] == 'BUY':
        pnl = (current_price - trade['entry']) * trade['qty']
    else:  # SELL
        pnl = (trade['entry'] - current_price) * trade['qty']
    
    pnl_percent = (pnl / (trade['entry'] * trade['qty'])) * 100
    return pnl, pnl_percent

def simple_momentum_strategy(symbol: str, df15m: pd.DataFrame):
    """Very simple momentum strategy for demonstration: if last close > SMA(20) -> BUY signal"""
    signals = []
    if df15m is None or df15m.empty or len(df15m) < 25:
        return signals
    close = df15m['Close']
    sma20 = close.rolling(20).mean()
    last = float(close.iloc[-1])
    sma_last = float(sma20.iloc[-1])
    timeframe = "15m"
    if last > sma_last and (last - sma_last)/sma_last > 0.002:  # small threshold
        entry = last
        stop_loss = last - (last * 0.01)  # 1% SL
        target1 = last * 1.01
        signal_id = generate_signal_id(symbol, "momentum_15m", timeframe, entry)
        signals.append(dict(
            id=signal_id, symbol=symbol, action="BUY", strategy="momentum_15m", timeframe=timeframe,
            entry=entry, stop_loss=stop_loss, target1=target1, confidence=0.7, timestamp=time.time()
        ))
    elif last < sma_last and (sma_last - last)/sma_last > 0.002:
        entry = last
        stop_loss = last + (last * 0.01)
        target1 = last * 0.99
        signal_id = generate_signal_id(symbol, "momentum_15m", timeframe, entry)
        signals.append(dict(
            id=signal_id, symbol=symbol, action="SELL", strategy="momentum_15m", timeframe=timeframe,
            entry=entry, stop_loss=stop_loss, target1=target1, confidence=0.68, timestamp=time.time()
        ))
    return signals

def sma_crossover_strategy(symbol: str, df15m: pd.DataFrame):
    """Simple SMA crossover: 5-period over 20-period"""
    signals = []
    if df15m is None or df15m.empty or len(df15m) < 30:
        return signals
    close = df15m['Close']
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    if float(sma5.iloc[-1]) > float(sma20.iloc[-1]) and float(sma5.iloc[-2]) <= float(sma20.iloc[-2]):
        entry = float(close.iloc[-1])
        signal_id = generate_signal_id(symbol, "sma_5_20", "15m", entry)
        signals.append(dict(
            id=signal_id, symbol=symbol, action="BUY", strategy="sma_5_20", timeframe="15m",
            entry=entry, stop_loss=entry - entry*0.01, target1=entry*1.01, confidence=0.65, timestamp=time.time()
        ))
    if float(sma5.iloc[-1]) < float(sma20.iloc[-1]) and float(sma5.iloc[-2]) >= float(sma20.iloc[-2]):
        entry = float(close.iloc[-1])
        signal_id = generate_signal_id(symbol, "sma_5_20", "15m", entry)
        signals.append(dict(
            id=signal_id, symbol=symbol, action="SELL", strategy="sma_5_20", timeframe="15m",
            entry=entry, stop_loss=entry + entry*0.01, target1=entry*0.99, confidence=0.63, timestamp=time.time()
        ))
    return signals

def generate_signals_for_symbol(symbol: str):
    """Generate signals for one symbol using available simple strategies (15m)"""
    _, df15m = fetch_latest_close(symbol, period="7d", interval="15m")
    signals = []
    signals += simple_momentum_strategy(symbol, df15m)
    signals += sma_crossover_strategy(symbol, df15m)
    return signals

def execute_paper_trade(signal: dict):
    """Execute a paper trade enforcing FIXED allocation per trade and preventing duplicates"""
    sig_id = signal["id"]
    if sig_id in st.session_state.executed_signal_ids:
        return None  # already executed previously
    entry = float(signal["entry"])
    if entry <= 0:
        return None
    qty = int(FIXED_ALLOCATION / entry)
    if qty <= 0:
        return None
    
    # Calculate support and resistance
    _, df = fetch_latest_close(signal["symbol"], period="7d", interval="15m")
    support, resistance = calculate_support_resistance(df)
    
    # Create trade record
    trade = dict(
        id=f"T{len(st.session_state.paper_trades)+1:05d}",
        signal_id=sig_id,
        symbol=signal["symbol"],
        action=signal["action"],
        entry=entry,
        qty=qty,
        entry_time=str(datetime.now()),
        stop_loss=signal.get("stop_loss"),
        target1=signal.get("target1"),
        strategy=signal.get("strategy"),
        support=support,
        resistance=resistance,
        status="OPEN"
    )
    # Deduct allocated capital from balance (reserve)
    st.session_state.balance -= FIXED_ALLOCATION
    st.session_state.paper_trades.append(trade)
    st.session_state.executed_signal_ids.add(sig_id)
    return trade

def close_trade(trade_id, close_price):
    """Close a paper trade and move to history"""
    for i, trade in enumerate(st.session_state.paper_trades):
        if trade['id'] == trade_id:
            # Calculate final PnL
            pnl, pnl_percent = calculate_pnl(trade, close_price)
            
            # Create history record
            history_trade = trade.copy()
            history_trade['exit_price'] = close_price
            history_trade['exit_time'] = str(datetime.now())
            history_trade['pnl'] = pnl
            history_trade['pnl_percent'] = pnl_percent
            history_trade['status'] = "CLOSED"
            
            # Add to history and remove from open trades
            st.session_state.trade_history.append(history_trade)
            st.session_state.paper_trades.pop(i)
            
            # Return capital plus PnL
            st.session_state.balance += FIXED_ALLOCATION + pnl
            return history_trade
    return None

# ----- Setup Auto-Refresh -----
signal_refresh_needed, price_refresh_needed, current_time = setup_auto_refresh()

# ----- Sidebar -----
st.sidebar.header("Configuration")
market_choice = st.sidebar.selectbox("Market", list(MARKETS.keys()), index=0)
selected_symbols = st.sidebar.multiselect("Symbols", MARKETS[market_choice], default=MARKETS[market_choice])
st.sidebar.markdown("---")
st.sidebar.header("Auto-refresh")
auto_refresh_signals = st.sidebar.checkbox("Auto-refresh signals (2 min)", value=True)
auto_refresh_prices = st.sidebar.checkbox("Auto-refresh prices (30 sec)", value=True)
st.sidebar.write("Signals refresh every 120s, prices refresh every 30s when enabled.")

st.sidebar.markdown("---")
st.sidebar.header("Paper trading")
st.sidebar.write("Fixed allocation per trade: $%d" % FIXED_ALLOCATION)
st.sidebar.metric("Balance (demo)", f"${st.session_state.balance:,.2f}")

# ----- Generate / Refresh Signals -----
# Price refresh path (do not rerun the whole app to update prices frequently)
if price_refresh_needed and auto_refresh_prices:
    # Update last prices cache for dashboard view
    for sym in selected_symbols:
        price, _ = fetch_latest_close(sym, period="7d", interval="15m")
        if price is not None:
            st.session_state.last_prices[sym] = price
    st.session_state.last_price_refresh = current_time

# Signals refresh path (every 2 minutes or when manually requested)
manual_refresh = st.sidebar.button("Refresh Signals Now")
if auto_refresh_signals and signal_refresh_needed or manual_refresh:
    # Generate new signals for current selection
    new_signals = []
    for sym in selected_symbols:
        try:
            new_signals.extend(generate_signals_for_symbol(sym))
        except Exception:
            pass
    # Merge with previous signals but avoid duplicates (same signal id)
    existing_ids = {s["id"] for s in st.session_state.signals}
    merged = st.session_state.signals.copy()
    added = 0
    for s in new_signals:
        if s["id"] not in existing_ids:
            merged.append(s)
            added += 1
    # Update session signals and refresh timestamp
    st.session_state.signals = merged
    st.session_state.last_signal_refresh = time.time()
    if auto_refresh_signals and signal_refresh_needed:
        # Force a quick re-run so UI updates immediately
        st.rerun()

# ----- Tabs -----
tab1, tab2, tab3, tab4 = st.tabs(["Live Dashboard", "Signals", "Paper Trading", "Trade History"])

# ----- TAB 1: Live Dashboard -----
with tab1:
    st.session_state.current_tab = "Live Dashboard"
    st.subheader("Live Dashboard — Prices & Market Mood")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    total_signals = len([s for s in st.session_state.signals if s["symbol"] in selected_symbols])
    col1.metric("Total Signals", total_signals)
    col2.metric("Buy Signals", len([s for s in st.session_state.signals if s.get("action")=="BUY" and s["symbol"] in selected_symbols]))
    col3.metric("Sell Signals", len([s for s in st.session_state.signals if s.get("action")=="SELL" and s["symbol"] in selected_symbols]))
    col4.metric("Open Trades", len(st.session_state.paper_trades))
    
    st.markdown("----")
    
    # Market Mood Index (MMI) gauge
    st.subheader("Market Mood Index (MMI)")
    mmi_col1, mmi_col2 = st.columns([1, 2])
    
    with mmi_col1:
        # Overall market mood (average of all selected symbols)
        overall_mood = 50.0
        mood_scores = []
        for sym in selected_symbols:
            df = yf.download(sym, period="7d", interval="15m", progress=False)
            if not df.empty and len(df['Close']) > 10:
                returns = df['Close'].pct_change().fillna(0)
                mood_score = float(np.clip(50 + returns.tail(15).mean() * 1000, 0, 100))
                mood_scores.append(mood_score)
        
        if mood_scores:
            overall_mood = sum(mood_scores) / len(mood_scores)
        
        st.metric("Overall Market Mood", f"{overall_mood:.1f}")
        st.write("**Sentiment Levels:**")
        st.write("• 0-25: Extreme Fear")
        st.write("• 26-45: Fear") 
        st.write("• 46-55: Neutral")
        st.write("• 56-75: Greed")
        st.write("• 76-100: Extreme Greed")
    
    with mmi_col2:
        fig = create_mood_gauge(overall_mood, "Market Mood Index", "Updated recently")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("----")
    
    # Individual symbol gauges
    st.subheader("Individual Symbol Analysis")
    pick = selected_symbols[:6]  # Show up to 6 symbols
    cols = st.columns(3)
    for i, sym in enumerate(pick):
        with cols[i % 3]:
            price = st.session_state.last_prices.get(sym) or fetch_latest_close(sym)[0] or 0.0
            # Mood score based on recent performance
            df = yf.download(sym, period="7d", interval="15m", progress=False)
            mood_score = 50.0
            if not df.empty and len(df['Close']) > 10:
                returns = df['Close'].pct_change().fillna(0)
                mood_score = float(np.clip(50 + returns.tail(15).mean() * 1000, 0, 100))
            
            # Clean symbol name for display
            display_name = sym.replace("-USD", "").replace("=X", "")
            st.markdown(f'<div class="mood-gauge-container">', unsafe_allow_html=True)
            fig = create_mood_gauge(mood_score, display_name, f"${price:.4f}" if price > 1 else f"${price:.6f}")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ----- TAB 2: Signals -----
with tab2:
    st.session_state.current_tab = "Signals"
    st.subheader("Trading Signals (auto-updates every 2 minutes)")
    
    if not st.session_state.signals:
        st.info("No signals generated yet. Click 'Refresh Signals Now' or enable auto-refresh.")
    else:
        # Show table of signals for selected symbols (without signal ID)
        table = pd.DataFrame(st.session_state.signals)
        if not table.empty:
            table_display = table[table['symbol'].isin(selected_symbols)].copy()
            table_display['timestamp'] = table_display['timestamp'].apply(lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"))
            
            # Remove signal ID from display
            display_columns = ['symbol', 'action', 'strategy', 'timeframe', 'entry', 'stop_loss', 'target1', 'confidence', 'timestamp']
            st.dataframe(
                table_display[display_columns].rename(columns={
                    'timeframe': 'TF', 
                    'target1': 'Take Profit',
                    'stop_loss': 'Stop Loss'
                }), 
                use_container_width=True
            )
            
            st.markdown("----")
            
            # Execution controls
            auto_execute = st.checkbox("Auto-execute high confidence signals (>=0.7)", value=False)
            exec_cols = st.columns([3, 1, 1])
            
            with exec_cols[1]:
                if st.button("Execute top signal"):
                    # pick highest confidence available that isn't executed
                    available = [s for s in st.session_state.signals if s['symbol'] in selected_symbols and s['id'] not in st.session_state.executed_signal_ids]
                    if available:
                        available = sorted(available, key=lambda x: x['confidence'], reverse=True)
                        trade = execute_paper_trade(available[0])
                        if trade:
                            st.success(f"Executed trade {trade['id']} for {trade['symbol']}")
                            st.rerun()
                        else:
                            st.error("Could not execute trade (maybe allocation too small).")
                    else:
                        st.warning("No available signals to execute.")
            
            # Auto-execute logic
            if auto_execute:
                executed_count = 0
                for s in sorted(st.session_state.signals, key=lambda x: x['confidence'], reverse=True):
                    if s['symbol'] not in selected_symbols:
                        continue
                    if s['confidence'] >= 0.7 and s['id'] not in st.session_state.executed_signal_ids:
                        t = execute_paper_trade(s)
                        if t:
                            executed_count += 1
                if executed_count > 0:
                    st.success(f"Auto-executed {executed_count} trades (fixed ${FIXED_ALLOCATION} allocation each).")
                    st.rerun()

# ----- TAB 3: Paper Trading -----
with tab3:
    st.session_state.current_tab = "Paper Trading"
    st.subheader("Paper Trading Dashboard")
    st.markdown(f"**Balance:** ${st.session_state.balance:,.2f}  •  **Allocated per trade:** ${FIXED_ALLOCATION:,.2f}")
    
    if st.session_state.paper_trades:
        # Create enhanced trades dataframe with PnL
        enhanced_trades = []
        for trade in st.session_state.paper_trades:
            current_price = st.session_state.last_prices.get(trade['symbol'], trade['entry'])
            pnl, pnl_percent = calculate_pnl(trade, current_price)
            
            enhanced_trade = trade.copy()
            enhanced_trade['current_price'] = current_price
            enhanced_trade['pnl'] = pnl
            enhanced_trade['pnl_percent'] = pnl_percent
            enhanced_trades.append(enhanced_trade)
        
        df_trades = pd.DataFrame(enhanced_trades)
        
        # Display columns: Remove signal_id, add PnL, Support/Resistance
        display_columns = ['id', 'symbol', 'action', 'entry', 'current_price', 'qty', 'pnl', 'pnl_percent', 
                          'stop_loss', 'target1', 'support', 'resistance', 'strategy', 'entry_time']
        
        st.dataframe(df_trades[display_columns], use_container_width=True)
        
        # Close trade functionality
        st.subheader("Close Trades")
        trade_options = {f"{trade['id']} - {trade['symbol']} - {trade['action']}": trade['id'] 
                        for trade in st.session_state.paper_trades}
        
        if trade_options:
            selected_trade = st.selectbox("Select trade to close:", list(trade_options.keys()))
            close_price = st.number_input("Close price:", value=st.session_state.last_prices.get(
                st.session_state.paper_trades[0]['symbol'], 
                st.session_state.paper_trades[0]['entry']
            ), step=0.0001, format="%.6f")
            
            if st.button("Close Trade"):
                closed_trade = close_trade(trade_options[selected_trade], close_price)
                if closed_trade:
                    st.success(f"Closed trade {closed_trade['id']}. PnL: ${closed_trade['pnl']:.2f} ({closed_trade['pnl_percent']:.2f}%)")
                    st.rerun()
                else:
                    st.error("Failed to close trade.")
    else:
        st.info("No paper trades executed yet. Execute signals from the Signals tab.")

# ----- TAB 4: Trade History -----
with tab4:
    st.session_state.current_tab = "Trade History"
    st.subheader("Trade History")
    
    if st.session_state.trade_history:
        df_history = pd.DataFrame(st.session_state.trade_history)
        
        # Calculate totals
        total_pnl = df_history['pnl'].sum()
        winning_trades = len(df_history[df_history['pnl'] > 0])
        losing_trades = len(df_history[df_history['pnl'] < 0])
        win_rate = (winning_trades / len(df_history)) * 100 if len(df_history) > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(df_history))
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Winning Trades", winning_trades)
        col4.metric("Total PnL", f"${total_pnl:.2f}")
        
        # Display history table
        history_columns = ['id', 'symbol', 'action', 'entry', 'exit_price', 'qty', 'pnl', 'pnl_percent', 
                          'strategy', 'entry_time', 'exit_time']
        st.dataframe(df_history[history_columns], use_container_width=True)
        
        # Option to clear history
        if st.button("Clear Trade History"):
            st.session_state.trade_history = []
            st.rerun()
    else:
        st.info("No trade history yet. Close some trades to see history here.")

# ----- Footer / Debug -----
st.markdown("---")
st.markdown(f"<div class='small'>Last signal refresh: {datetime.fromtimestamp(st.session_state.last_signal_refresh) if st.session_state.last_signal_refresh>0 else 'Never'}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='small'>Last price refresh: {datetime.fromtimestamp(st.session_state.last_price_refresh) if st.session_state.last_price_refresh>0 else 'Never'}</div>", unsafe_allow_html=True)

# ----- Auto-refresh JavaScript -----
if auto_refresh_prices or auto_refresh_signals:
    # Determine which refresh to use based on current tab
    if st.session_state.current_tab == "Live Dashboard" and auto_refresh_prices:
        refresh_seconds = PRICE_REFRESH_SECONDS
        time_since_refresh = time.time() - st.session_state.last_price_refresh
    elif st.session_state.current_tab in ["Signals", "Live Dashboard"] and auto_refresh_signals:
        refresh_seconds = SIGNAL_REFRESH_SECONDS
        time_since_refresh = time.time() - st.session_state.last_signal_refresh
    else:
        refresh_seconds = None
    
    if refresh_seconds:
        time_remaining = max(0, refresh_seconds - time_since_refresh)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF6B6B, #4ECDC4); color: white; padding: 10px; border-radius: 10px; text-align: center; margin: 10px 0;">
            🔄 AUTO-REFRESH ACTIVE | Next update in {int(time_remaining)} seconds
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-refresh using JavaScript when time is up
        if time_remaining <= 1:
            st.markdown("""
            <script>
            setTimeout(function() {
                window.location.reload();
            }, 1000);
            </script>
            """, unsafe_allow_html=True)
