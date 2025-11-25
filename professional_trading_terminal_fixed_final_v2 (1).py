
# final_trading_terminal_single_file.py
# Single-file Streamlit trading terminal
# Features:
# - Market coverage: Cryptos, Forex, Commodities (user requested symbols)
# - Signals auto-refresh every 120 seconds (2 minutes)
# - Price sections refresh faster (every 15 seconds)
# - Fixed allocation $1000 per trade
# - Prevent duplicate trade executions using unique signal IDs stored in session_state
# - Readable UI fonts, multi-colour tabs, mood gauge (needle-style via plotly)
# - Simple example strategies (momentum & sma crossover) for demonstration
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
PRICE_REFRESH_SECONDS = 15  # fast refresh for live prices

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
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">🚀 Unified Trading Terminal — Single File</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Market coverage: Crypto · Forex · Commodities — Signals auto-refresh: 2 minutes</div>', unsafe_allow_html=True)
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
if "balance" not in st.session_state:
    st.session_state.balance = 20000.0  # demo starting balance
if "last_prices" not in st.session_state:
    st.session_state.last_prices = {}

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
    """Create a needle gauge with plotly"""
    # range 0-100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"{title}<br>{price_display}", 'font': {'size': 12}},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color': '#10b981'},
            'steps':[{'range':[0,40],'color':'#ef4444'},{'range':[40,60],'color':'#fbbf24'},{'range':[60,100],'color':'#10b981'}]
        }
    ))
    fig.update_layout(height=220, margin=dict(t=30,b=10,l=10,r=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

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
        strategy=signal.get("strategy")
    )
    # Deduct allocated capital from balance (reserve)
    # To keep it simple, we reserve FIXED_ALLOCATION (not qty*entry) to consistently reflect $1000 allocation
    st.session_state.balance -= FIXED_ALLOCATION
    st.session_state.paper_trades.append(trade)
    st.session_state.executed_signal_ids.add(sig_id)
    return trade

# ----- Sidebar -----
st.sidebar.header("Configuration")
market_choice = st.sidebar.selectbox("Market", list(MARKETS.keys()), index=0)
selected_symbols = st.sidebar.multiselect("Symbols", MARKETS[market_choice], default=MARKETS[market_choice])
st.sidebar.markdown("---")
st.sidebar.header("Auto-refresh")
auto_refresh_signals = st.sidebar.checkbox("Auto-refresh signals (2 min)", value=True)
st.sidebar.write("Signals refresh every 120s when enabled. Prices refresh faster for dashboard.")

st.sidebar.markdown("---")
st.sidebar.header("Paper trading")
st.sidebar.write("Fixed allocation per trade: $%d" % FIXED_ALLOCATION)
st.sidebar.metric("Balance (demo)", f"${st.session_state.balance:,.2f}")

# ----- Generate / Refresh Signals -----
now = time.time()
signal_refresh_needed = (now - st.session_state.last_signal_refresh) >= SIGNAL_REFRESH_SECONDS
price_refresh_needed = (now - st.session_state.last_price_refresh) >= PRICE_REFRESH_SECONDS

# Price refresh path (do not rerun the whole app to update prices frequently)
if price_refresh_needed:
    # Update last prices cache for dashboard view
    for sym in selected_symbols:
        price, _ = fetch_latest_close(sym, period="7d", interval="15m")
        if price is not None:
            st.session_state.last_prices[sym] = price
    st.session_state.last_price_refresh = now

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
        st.experimental_rerun()

# ----- Tabs -----
tab1, tab2, tab3 = st.tabs(["Live Dashboard", "Signals", "Paper Trading"])

# ----- TAB 1: Live Dashboard -----
with tab1:
    st.subheader("Live Dashboard — Prices & Mood")
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    total_signals = len([s for s in st.session_state.signals if s["symbol"] in selected_symbols])
    col1.metric("Total Signals", total_signals)
    col2.metric("Buy Signals", len([s for s in st.session_state.signals if s.get("action")=="BUY" and s["symbol"] in selected_symbols]))
    col3.metric("Sell Signals", len([s for s in st.session_state.signals if s.get("action")=="SELL" and s["symbol"] in selected_symbols]))
    col4.metric("Executed Trades", len(st.session_state.paper_trades))
    st.markdown("----")
    # Mood gauges for selected symbols (show up to 6 in grid)
    pick = selected_symbols[:6]
    cols = st.columns(3)
    for i, sym in enumerate(pick):
        with cols[i % 3]:
            price = st.session_state.last_prices.get(sym) or fetch_latest_close(sym)[0] or 0.0
            # simple mood: normalized recent return over 15 periods
            df = yf.download(sym, period="7d", interval="15m", progress=False)
            mood_score = 50.0
            if not df.empty and len(df['Close'])>10:
                returns = df['Close'].pct_change().fillna(0)
                mood_score = float(np.clip(50 + returns.tail(15).mean()*1000, 0, 100))
            st.write(f"**{sym}** — {price}")
            fig = create_mood_gauge(mood_score, sym, price_display=f"{price:.4f}")
            st.plotly_chart(fig, use_container_width=True)

# ----- TAB 2: Signals -----
with tab2:
    st.subheader("Trading Signals (auto-updates every 2 minutes)")
    if not st.session_state.signals:
        st.info("No signals generated yet. Click 'Refresh Signals Now' or enable auto-refresh.")
    else:
        # Show table of signals for selected symbols
        table = pd.DataFrame(st.session_state.signals)
        if not table.empty:
            table_display = table[table['symbol'].isin(selected_symbols)].copy()
            table_display['timestamp'] = table_display['timestamp'].apply(lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"))
            st.dataframe(table_display[['id','symbol','action','strategy','timeframe','entry','stop_loss','target1','confidence','timestamp']].rename(columns={
                'id':'signal_id','timeframe':'tf','target1':'tp1'}), use_container_width=True)
            st.markdown("----")
            # Execution controls: execute high-confidence signals automatically on user request
            auto_execute = st.checkbox("Auto-execute high confidence signals (>=0.7)", value=False)
            exec_cols = st.columns([3,1,1])
            with exec_cols[1]:
                if st.button("Execute top signal"):
                    # pick highest confidence available that isn't executed
                    available = [s for s in st.session_state.signals if s['symbol'] in selected_symbols and s['id'] not in st.session_state.executed_signal_ids]
                    if available:
                        available = sorted(available, key=lambda x: x['confidence'], reverse=True)
                        trade = execute_paper_trade(available[0])
                        if trade:
                            st.success(f"Executed trade {trade['id']} for {trade['symbol']}")
                            st.experimental_rerun()
                        else:
                            st.error("Could not execute trade (maybe allocation too small).")
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
                if executed_count>0:
                    st.success(f"Auto-executed {executed_count} trades (fixed ${FIXED_ALLOCATION} allocation each).")
                    st.experimental_rerun()

# ----- TAB 3: Paper Trading -----
with tab3:
    st.subheader("Paper Trading Dashboard")
    st.markdown(f"**Balance:** ${st.session_state.balance:,.2f}  •  **Allocated per open trade:** ${FIXED_ALLOCATION:,.2f}")
    if st.session_state.paper_trades:
        df_trades = pd.DataFrame(st.session_state.paper_trades)
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("No paper trades executed yet. Execute signals from the Signals tab.")

# ----- Footer / Debug -----
st.markdown("---")
st.markdown(f"<div class='small'>Last signal refresh: {datetime.fromtimestamp(st.session_state.last_signal_refresh) if st.session_state.last_signal_refresh>0 else 'Never'}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='small'>Last price refresh: {datetime.fromtimestamp(st.session_state.last_price_refresh) if st.session_state.last_price_refresh>0 else 'Never'}</div>", unsafe_allow_html=True)
