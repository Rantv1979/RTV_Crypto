# trading_terminal_web_gold_crypto_forex_v4.py
"""
Market Terminal v4 — Adds Historical Backtester
- Live terminal + automatic logging (trade_log.csv)
- Auto-resolution of open trades (Option A)
- NEW: Historical backtester that simulates the same entry/target/SL logic
Notes:
- Backtester uses yfinance historical bars (1m, 5m, 15m...) depending on selection.
- For 1m intraday backtests, yfinance may provide limited lookback (usually ~7 days). Use 5m/15m or set days accordingly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
from datetime import datetime, timedelta
import math

# -----------------------
# Config / Constants
# -----------------------
TRADE_LOG_FILE = "trade_log.csv"
AUDIO_ALERT_URL = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
STYLES = """
<style>
.card { background: linear-gradient(180deg,#111 0%,#1a1a1a 100%); border-radius:10px; padding:12px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.04);}
.header { display:flex; align-items:center; justify-content:space-between; }
.small-muted { color:#9aa0a6; font-size:12px; }
.metric { font-weight:700; font-size:18px; }
.pulse { animation: pulse 1.6s infinite; box-shadow:0 0 12px rgba(255,200,60,0.25); }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,200,60,0.35);} 70%{box-shadow:0 0 0 12px rgba(255,200,60,0);} 100%{box-shadow:0 0 0 0 rgba(255,200,60,0);} }
</style>
"""

MARKETS = {
    "GOLD (FUT)": "GC=F",
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "XRP": "XRP-USD",
    "SOLANA": "SOL-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X"
}

# -----------------------
# Utility: Trade log persistence
# -----------------------
def ensure_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        df = pd.DataFrame(columns=[
            "id", "timestamp", "market", "signal", "entry", "t1", "t2", "t3", "sl",
            "status", "result_time", "result_type", "result_price", "notes"
        ])
        df.to_csv(TRADE_LOG_FILE, index=False)

def load_trade_log():
    ensure_trade_log()
    return pd.read_csv(TRADE_LOG_FILE)

def append_trade_row(row: dict):
    df = load_trade_log()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADE_LOG_FILE, index=False)

def update_trade_row(trade_id: int, updates: dict):
    df = load_trade_log()
    idx = df.index[df["id"] == trade_id]
    if not idx.empty:
        for k, v in updates.items():
            df.loc[idx, k] = v
        df.to_csv(TRADE_LOG_FILE, index=False)

# -----------------------
# Audio / notifications
# -----------------------
def play_sound_and_toast(message: str):
    st.markdown(f"""
        <audio autoplay>
            <source src="{AUDIO_ALERT_URL}" type="audio/ogg">
            Your browser does not support the audio element.
        </audio>
    """, unsafe_allow_html=True)
    st.toast(message, icon="🔔")

# -----------------------
# Indicator helpers
# -----------------------
def calculate_rsi(prices, period=14):
    series = pd.Series(prices).dropna()
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    return float(last) if not np.isnan(last) else 50.0

def sma(prices, period):
    series = pd.Series(prices).dropna()
    if len(series) == 0:
        return 0.0
    if len(series) < period:
        return float(series.mean())
    return float(series.iloc[-period:].mean())

def get_signal_from_series(series_prices):
    clean = [p for p in series_prices if p and p > 0]
    if len(clean) < 6:
        return {"signal": "HOLD", "rsi": 50.0, "sma5": 0.0, "sma15": 0.0}
    rsi_val = calculate_rsi(clean, 14)
    s5 = sma(clean, 5)
    s15 = sma(clean, 15)
    if rsi_val < 30 and s5 > s15:
        return {"signal": "BUY", "rsi": rsi_val, "sma5": s5, "sma15": s15}
    if rsi_val > 70 and s5 < s15:
        return {"signal": "SELL", "rsi": rsi_val, "sma5": s5, "sma15": s15}
    return {"signal": "HOLD", "rsi": rsi_val, "sma5": s5, "sma15": s15}

def calculate_targets(price, signal):
    if price is None or price <= 0:
        return None
    if signal == "BUY":
        return {"entry": price, "t1": price*1.01, "t2": price*1.02, "t3": price*1.03, "sl": price*0.99}
    if signal == "SELL":
        return {"entry": price, "t1": price*0.99, "t2": price*0.98, "t3": price*0.97, "sl": price*1.01}
    return None

def format_price(name, price):
    if price is None or price == 0 or math.isnan(price):
        return "N/A"
    usd_like = ["BITCOIN","ETHEREUM","GOLD (FUT)","XRP","SOLANA"]
    if name in usd_like:
        if price < 10:
            return f"${price:,.6f}"
        return f"${price:,.2f}"
    return f"{price:,.4f}"

# -----------------------
# OHLC fetch
# -----------------------
def fetch_ohlc(ticker, period="7d", interval="1m"):
    """
    Fetch OHLC from yfinance. period and interval should be chosen based on backtest requirements.
    Note: yfinance supports minute data for limited lookback (usually ~7 days for 1m).
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Open","High","Low","Close"])
        return df
    except Exception as e:
        return None

# -----------------------
# App initial state
# -----------------------
st.set_page_config(page_title="Market Terminal v4 — Backtester", layout="wide", page_icon="💱")
st.markdown(STYLES, unsafe_allow_html=True)
st.title("💱 Market Terminal — Gold, Crypto & Forex (with Backtester)")

if "v4_initialized" not in st.session_state:
    st.session_state.v4_initialized = True
    st.session_state.price_history = {k:[] for k in MARKETS.keys()}
    st.session_state.targets = {}
    st.session_state.last_signals = {}
    st.session_state.signal_alerts = {}
    st.session_state.update_count = 0
    st.session_state.last_update_time = datetime.now()
    ensure_trade_log()

# -----------------------
# Sidebar: controls + backtest inputs
# -----------------------
with st.sidebar:
    st.header("Controls")
    st.checkbox("Auto Refresh (Live)", value=True, key="auto_refresh_v4")
    st.session_state.auto_refresh = st.session_state.auto_refresh_v4
    st.number_input("Refresh interval (seconds)", min_value=10, max_value=300, value=25, key="refresh_v4")
    st.session_state.refresh_interval = st.session_state.refresh_v4
    
    st.checkbox("Auto Trade Execution", value=True, key="auto_trade")
    
    st.markdown("---")
    st.subheader("Backtester")
    backtest_market = st.selectbox("Market to backtest", options=list(MARKETS.keys()), index=0)
    backtest_interval = st.selectbox("Bar interval", options=["1m","5m","15m","30m","60m","1d"], index=1)
    backtest_days = st.slider("Backtest lookback (days)", min_value=1, max_value=90, value=30, step=1)
    run_backtest_btn = st.button("Run Historical Backtest")
    st.markdown("""
    Backtest simulates: same live entry rules (RSI14 & SMA5/SMA15) and the exact targets/SL:
    - T1 = +1%, T2 = +2%, T3 = +3% (BUY) or symmetric for SELL
    - SL = 1% adverse
    """)
    st.markdown("---")
    st.subheader("Trade Log")
    if st.button("Show recent trade log"):
        st.write(load_trade_log().sort_values("timestamp", ascending=False).head(40))
    if st.button("Clear trade log (backup)"):
        if os.path.exists(TRADE_LOG_FILE):
            backup = TRADE_LOG_FILE.replace(".csv", f"_{int(time.time())}.bak.csv")
            os.rename(TRADE_LOG_FILE, backup)
        ensure_trade_log()
        st.success("Trade log cleared (backup created if existed).")

# -----------------------
# Live update routine (same as before but slimmed)
# -----------------------
def fetch_snapshot(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m")
        if df is None or df.empty:
            return None
        o = float(df["Open"].iloc[0])
        c = float(df["Close"].iloc[-1])
        h = float(df["High"].max())
        l = float(df["Low"].min())
        vol = int(df["Volume"].sum()) if "Volume" in df.columns else 0
        pct = (c - o) / o * 100 if o else 0
        return {"open": o, "current": c, "high": h, "low": l, "vol": vol, "pct": pct}
    except:
        return None

def run_live_update():
    prices = {}
    signals = {}
    resolved_trades = []
    for name, ticker in MARKETS.items():
        snap = fetch_snapshot(ticker)
        if snap is None:
            continue
        prices[name] = snap["current"]
        # store history (last 200)
        hist = st.session_state.price_history.get(name, [])
        hist.append(snap["current"])
        st.session_state.price_history[name] = hist[-200:]
        sig = get_signal_from_series(st.session_state.price_history[name])
        signals[name] = {**sig, "ohlc": snap}
        # targets if active
        if sig["signal"] in ["BUY","SELL"]:
            st.session_state.targets[name] = calculate_targets(snap["current"], sig["signal"])
        else:
            st.session_state.targets.pop(name, None)
        # log new signals
        last = st.session_state.last_signals.get(name, "HOLD")
        if sig["signal"] in ["BUY","SELL"] and sig["signal"] != last:
            t = st.session_state.targets.get(name)
            tid = int(time.time()*1000)
            row = {
                "id": tid,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "market": name,
                "signal": sig["signal"],
                "entry": round(t["entry"],6) if t else None,
                "t1": round(t["t1"],6) if t else None,
                "t2": round(t["t2"],6) if t else None,
                "t3": round(t["t3"],6) if t else None,
                "sl": round(t["sl"],6) if t else None,
                "status": "OPEN",
                "result_time": "",
                "result_type": "",
                "result_price": "",
                "notes": "AUTO-LIVE"
            }
            append_trade_row(row)
            st.session_state.signal_alerts[name] = True
            play_sound_and_toast(f"{sig['signal']} — {name}")
        st.session_state.last_signals[name] = sig["signal"]
    # resolve open trades (snapshot)
    df = load_trade_log()
    open_trades = df[df["status"] == "OPEN"]
    for _, trade in open_trades.iterrows():
        tid = int(trade["id"])
        m = trade["market"]
        cur = prices.get(m)
        if cur is None:
            continue
        entry = float(trade["entry"]) if not pd.isna(trade["entry"]) else None
        sl = float(trade["sl"]) if not pd.isna(trade["sl"]) else None
        t1 = float(trade["t1"]) if not pd.isna(trade["t1"]) else None
        t2 = float(trade["t2"]) if not pd.isna(trade["t2"]) else None
        t3 = float(trade["t3"]) if not pd.isna(trade["t3"]) else None
        resolved = False
        result_type = ""
        result_price = None
        if trade["signal"] == "BUY":
            # check targets top-down (T3 last gives highest return but first hit matters)
            if t1 and cur >= t1:
                resolved=True; result_type="T1"; result_price=cur
            if not resolved and t2 and cur >= t2:
                resolved=True; result_type="T2"; result_price=cur
            if not resolved and t3 and cur >= t3:
                resolved=True; result_type="T3"; result_price=cur
            if not resolved and sl and cur <= sl:
                resolved=True; result_type="SL"; result_price=cur
        elif trade["signal"] == "SELL":
            if t1 and cur <= t1:
                resolved=True; result_type="T1"; result_price=cur
            if not resolved and t2 and cur <= t2:
                resolved=True; result_type="T2"; result_price=cur
            if not resolved and t3 and cur <= t3:
                resolved=True; result_type="T3"; result_price=cur
            if not resolved and sl and cur >= sl:
                resolved=True; result_type="SL"; result_price=cur
        if resolved:
            update_trade_row(tid, {
                "status": "CLOSED",
                "result_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "result_type": result_type,
                "result_price": round(result_price,6),
                "notes": "AUTO-RESOLVED-LIVE"
            })
            play_sound_and_toast(f"{m} trade {'WON' if result_type.startswith('T') else 'LOSS'} ({result_type})")
    st.session_state.update_count += 1
    st.session_state.last_update_time = datetime.now()
    return prices, signals

# -----------------------
# Backtester implementation
# -----------------------
def run_backtest(market_name, interval, days):
    ticker = MARKETS[market_name]
    # Choose period string for yfinance based on days
    period = f"{days}d" if interval != "1d" else f"{days}d"
    # Fetch data
    data = fetch_ohlc(ticker, period=period, interval=interval)
    if data is None or data.empty:
        st.error("No historical data returned. Try a different interval or fewer days.")
        return None

    # Prepare time-ordered list (ascending by index)
    data = data.copy()
    data = data.sort_index()
    closes = data["Close"].tolist()

    # We'll simulate scanning bar-by-bar.
    backtest_trades = []  # list of dicts with results
    open_trade = None

    price_series = []  # for indicators as we sweep bars

    for idx, row in data.iterrows():
        close_price = float(row["Close"])
        price_series.append(close_price)
        # compute signal
        s = get_signal_from_series(price_series)
        signal = s["signal"]
        # If there's no open trade and signal becomes BUY/SELL, open a trade at current close
        if open_trade is None and signal in ["BUY","SELL"]:
            targets = calculate_targets(close_price, signal)
            open_trade = {
                "market": market_name,
                "open_time": idx,
                "signal": signal,
                "entry": close_price,
                "t1": targets["t1"],
                "t2": targets["t2"],
                "t3": targets["t3"],
                "sl": targets["sl"],
                "status": "OPEN",
                "result_time": None,
                "result_type": None,
                "result_price": None
            }
            # After opening, continue to next bar to monitor outcome
            continue

        # If trade is open, check if this bar's high/low hit targets or SL.
        if open_trade is not None:
            # Use High/Low on the bar to detect intrabar hits (if available)
            high = float(row["High"])
            low = float(row["Low"])
            resolved = False
            result_type = None
            result_price = None

            if open_trade["signal"] == "BUY":
                # check targets in order (T1 -> T2 -> T3). If multiple are in same bar, pick the earliest
                if high >= open_trade["t1"]:
                    resolved = True; result_type = "T1"; result_price = open_trade["t1"]
                    # But if T2/T3 also hit in same bar, consider the highest target reached (we will record first target for simplicity)
                if not resolved and high >= open_trade["t2"]:
                    resolved = True; result_type = "T2"; result_price = open_trade["t2"]
                if not resolved and high >= open_trade["t3"]:
                    resolved = True; result_type = "T3"; result_price = open_trade["t3"]
                if not resolved and low <= open_trade["sl"]:
                    resolved = True; result_type = "SL"; result_price = open_trade["sl"]
            else:  # SELL
                if low <= open_trade["t1"]:
                    resolved = True; result_type = "T1"; result_price = open_trade["t1"]
                if not resolved and low <= open_trade["t2"]:
                    resolved = True; result_type = "T2"; result_price = open_trade["t2"]
                if not resolved and low <= open_trade["t3"]:
                    resolved = True; result_type = "T3"; result_price = open_trade["t3"]
                if not resolved and high >= open_trade["sl"]:
                    resolved = True; result_type = "SL"; result_price = open_trade["sl"]

            if resolved:
                open_trade["status"] = "CLOSED"
                open_trade["result_time"] = idx
                open_trade["result_type"] = result_type
                open_trade["result_price"] = result_price
                backtest_trades.append(open_trade)
                open_trade = None
                # After resolving, the loop continues (next bars can open new trades)
            else:
                # trade remains OPEN — continue scanning bars
                pass

    # If trade still open at the end, mark as CLOSED with 'NO-HIT' and use final close as result_price
    if open_trade is not None:
        open_trade["status"] = "CLOSED"
        open_trade["result_time"] = data.index[-1]
        open_trade["result_type"] = "NO-HIT"
        open_trade["result_price"] = float(data["Close"].iloc[-1])
        backtest_trades.append(open_trade)
        open_trade = None

    # Convert to DataFrame and compute metrics
    if not backtest_trades:
        st.info("Backtest found no trades (no signals during period).")
        return pd.DataFrame()

    bt = pd.DataFrame(backtest_trades)
    bt["entry"] = pd.to_numeric(bt["entry"], errors="coerce")
    bt["result_price"] = pd.to_numeric(bt["result_price"], errors="coerce")
    # pct return: for BUY = (result-entry)/entry ; SELL = (entry-result)/entry
    def pct_ret(row):
        if row["signal"] == "BUY" and row["entry"] and row["result_price"]:
            return (row["result_price"] - row["entry"]) / row["entry"] * 100
        if row["signal"] == "SELL" and row["entry"] and row["result_price"]:
            return (row["entry"] - row["result_price"]) / row["entry"] * 100
        return 0.0
    bt["pct_return"] = bt.apply(pct_ret, axis=1)
    bt["outcome"] = bt["result_type"].apply(lambda x: "WON" if str(x).startswith("T") else ("LOSS" if x=="SL" else "NO-HIT"))
    wins = len(bt[bt["outcome"]=="WON"])
    losses = len(bt[bt["outcome"]=="LOSS"])
    nohit = len(bt[bt["outcome"]=="NO-HIT"])
    total = len(bt)
    win_rate = wins / total * 100
    avg_return = bt["pct_return"].mean()

    summary = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "nohit": nohit,
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "trades_df": bt
    }
    return summary

# -----------------------
# Rendering helper: dashboard & backtest UI
# -----------------------
def render_live_dashboard(prices, signals):
    # Top metrics and cards
    df_log = load_trade_log()
    closed = df_log[df_log["status"]=="CLOSED"]
    total_closed = len(closed)
    total_open = len(df_log[df_log["status"]=="OPEN"])
    wins = len(closed[closed["result_type"].str.startswith("T")])
    losses = len(closed[closed["result_type"]=="SL"])
    win_rate = (wins / total_closed * 100) if total_closed>0 else 0.0
    avg_return = None
    if total_closed>0:
        closed["entry"] = pd.to_numeric(closed["entry"], errors="coerce")
        closed["result_price"] = pd.to_numeric(closed["result_price"], errors="coerce")
        closed["pct_return"] = np.where(closed["signal"]=="BUY",
                                        (closed["result_price"] - closed["entry"]) / closed["entry"] * 100,
                                        (closed["entry"] - closed["result_price"]) / closed["entry"] * 100)
        avg_return = closed["pct_return"].mean()

    col1,col2,col3,col4 = st.columns([2,1,1,1])
    with col1:
        st.markdown("<div class='card header'><div><h2 style='margin:0'>📊 Live Market Dashboard</h2><div class='small-muted'>RSI+SMA Strategy (Auto-logger)</div></div>"
                    f"<div style='text-align:right'><div class='metric'>{st.session_state.update_count} updates</div><div class='small-muted'>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div></div></div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><div class='small-muted'>Open Trades</div><div class='metric'>{total_open}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='card'><div class='small-muted'>Closed Trades</div><div class='metric'>{total_closed}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='card'><div class='small-muted'>Win Rate</div><div class='metric'>{win_rate:.1f}%</div></div>", unsafe_allow_html=True)

    markets = list(MARKETS.keys())
    left, right = st.columns(2)
    for i, name in enumerate(markets):
        sig = signals.get(name)
        price = prices.get(name)
        target_info = st.session_state.targets.get(name)
        card_html = "<div class='card"
        if st.session_state.signal_alerts.get(name, False) and sig and sig["signal"] in ["BUY","SELL"]:
            card_html += " pulse"
        card_html += "'>"
        emoji = "🟢" if sig and sig["signal"]=="BUY" else "🔴" if sig and sig["signal"]=="SELL" else "⚪"
        card_html += f"<div style='display:flex; justify-content:space-between;align-items:center;'>"
        card_html += f"<div><strong style='font-size:16px'>{name} {emoji}</strong><div class='small-muted'>Ticker: {MARKETS[name]}</div></div>"
        card_html += f"<div style='text-align:right'><div class='metric'>{format_price(name, price)}</div>"
        if sig:
            card_html += f"<div class='small-muted'>RSI {sig['rsi']:.1f} | SMA5 {sig['sma5']:.4f} | SMA15 {sig['sma15']:.4f}</div>"
        card_html += "</div></div><hr style='opacity:0.06'/>"
        if target_info:
            card_html += f"<div><strong>Entry:</strong> {format_price(name, target_info['entry'])} &nbsp; <strong>SL:</strong> {format_price(name, target_info['sl'])}</div>"
            card_html += "<div style='margin-top:6px'>"
            card_html += f"<span class='small-muted'>T1:</span> <strong>{format_price(name,target_info['t1'])}</strong> &nbsp; "
            card_html += f"<span class='small-muted'>T2:</span> <strong>{format_price(name,target_info['t2'])}</strong> &nbsp; "
            card_html += f"<span class='small-muted'>T3:</span> <strong>{format_price(name,target_info['t3'])}</strong>"
            card_html += "</div>"
        else:
            card_html += "<div class='small-muted'>No active targets</div>"
        card_html += "</div>"
        if i%2==0:
            left.markdown(card_html, unsafe_allow_html=True)
        else:
            right.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")
    perf_left, perf_right = st.columns([2,1])
    with perf_left:
        st.subheader("📈 Performance Summary")
        st.write(f"Closed: {total_closed} | Wins: {wins} | Losses: {losses} | Avg return: {avg_return:.2f}%" if avg_return is not None else f"Closed: {total_closed}")
        if total_closed>0:
            st.dataframe(closed.sort_values("result_time", ascending=False).head(20)[["timestamp","market","signal","entry","result_price","result_type","result_time"]], use_container_width=True)
    with perf_right:
        st.subheader("Recent Open Trades")
        df_log = load_trade_log()
        open_df = df_log[df_log["status"]=="OPEN"]
        if not open_df.empty:
            st.dataframe(open_df[["timestamp","market","signal","entry","t1","t2","t3","sl"]], use_container_width=True)
        else:
            st.info("No open trades.")

def render_signals_tab(prices, signals):
    st.header("📡 Live Signals")
    st.write("Real-time trading signals based on RSI and SMA indicators")
    
    # Create signals table
    signals_data = []
    for name in MARKETS.keys():
        sig = signals.get(name)
        price = prices.get(name)
        if sig:
            signals_data.append({
                "Market": name,
                "Current Price": format_price(name, price),
                "Signal": sig["signal"],
                "RSI": f"{sig['rsi']:.1f}",
                "SMA5": f"{sig['sma5']:.4f}",
                "SMA15": f"{sig['sma15']:.4f}",
                "Signal Strength": "Strong" if (sig["signal"] == "BUY" and sig["rsi"] < 25) or (sig["signal"] == "SELL" and sig["rsi"] > 75) else "Medium"
            })
    
    if signals_data:
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, use_container_width=True)
        
        # Show signal statistics
        buy_signals = len([s for s in signals_data if s["Signal"] == "BUY"])
        sell_signals = len([s for s in signals_data if s["Signal"] == "SELL"])
        hold_signals = len([s for s in signals_data if s["Signal"] == "HOLD"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BUY Signals", buy_signals)
        with col2:
            st.metric("SELL Signals", sell_signals)
        with col3:
            st.metric("HOLD Signals", hold_signals)
    else:
        st.info("No signal data available. Enable auto-refresh to get live signals.")

def render_paper_trading_tab():
    st.header("📝 Paper Trading")
    st.write("Manual trade execution for testing strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Open New Trade")
        with st.form("paper_trade_form"):
            market = st.selectbox("Market", options=list(MARKETS.keys()))
            signal_type = st.selectbox("Signal Type", options=["BUY", "SELL"])
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, format="%.6f")
            quantity = st.number_input("Quantity", min_value=0.0, step=0.01, value=1.0)
            sl_price = st.number_input("Stop Loss", min_value=0.0, step=0.0001, format="%.6f")
            tp1_price = st.number_input("Take Profit 1", min_value=0.0, step=0.0001, format="%.6f")
            tp2_price = st.number_input("Take Profit 2", min_value=0.0, step=0.0001, format="%.6f")
            tp3_price = st.number_input("Take Profit 3", min_value=0.0, step=0.0001, format="%.6f")
            notes = st.text_input("Notes")
            
            if st.form_submit_button("Execute Paper Trade"):
                tid = int(time.time()*1000)
                row = {
                    "id": tid,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "market": market,
                    "signal": signal_type,
                    "entry": round(entry_price, 6),
                    "t1": round(tp1_price, 6),
                    "t2": round(tp2_price, 6),
                    "t3": round(tp3_price, 6),
                    "sl": round(sl_price, 6),
                    "status": "OPEN",
                    "result_time": "",
                    "result_type": "",
                    "result_price": "",
                    "notes": f"PAPER-{notes}"
                }
                append_trade_row(row)
                st.success(f"Paper trade executed! Trade ID: {tid}")
    
    with col2:
        st.subheader("Active Paper Trades")
        df_log = load_trade_log()
        paper_trades = df_log[df_log["notes"].str.startswith("PAPER-", na=False)]
        open_paper = paper_trades[paper_trades["status"] == "OPEN"]
        
        if not open_paper.empty:
            st.dataframe(open_paper[["timestamp", "market", "signal", "entry", "t1", "t2", "t3", "sl", "notes"]], 
                        use_container_width=True)
            
            # Close trade manually
            trade_to_close = st.selectbox("Select trade to close", 
                                        options=open_paper["id"].tolist(),
                                        format_func=lambda x: f"ID: {x} - {open_paper[open_paper['id']==x]['market'].iloc[0]} {open_paper[open_paper['id']==x]['signal'].iloc[0]}")
            close_price = st.number_input("Close Price", min_value=0.0, step=0.0001, format="%.6f")
            close_reason = st.selectbox("Close Reason", options=["MANUAL", "T1", "T2", "T3", "SL"])
            
            if st.button("Close Trade"):
                update_trade_row(trade_to_close, {
                    "status": "CLOSED",
                    "result_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "result_type": close_reason,
                    "result_price": round(close_price, 6),
                    "notes": f"{open_paper[open_paper['id']==trade_to_close]['notes'].iloc[0]} - MANUAL_CLOSE"
                })
                st.success("Trade closed successfully!")
                st.rerun()
        else:
            st.info("No active paper trades")

def render_history_tab():
    st.header("📊 Trading History")
    st.write("Complete trade history and performance analytics")
    
    df_log = load_trade_log()
    
    if df_log.empty:
        st.info("No trade history available")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        market_filter = st.multiselect("Filter by Market", options=df_log["market"].unique(), default=df_log["market"].unique())
    with col2:
        signal_filter = st.multiselect("Filter by Signal", options=df_log["signal"].unique(), default=df_log["signal"].unique())
    with col3:
        status_filter = st.multiselect("Filter by Status", options=df_log["status"].unique(), default=df_log["status"].unique())
    
    # Apply filters
    filtered_df = df_log[
        (df_log["market"].isin(market_filter)) &
        (df_log["signal"].isin(signal_filter)) &
        (df_log["status"].isin(status_filter))
    ]
    
    # Display filtered data
    st.dataframe(filtered_df.sort_values("timestamp", ascending=False), use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Analytics")
    
    if len(filtered_df) > 0:
        closed_trades = filtered_df[filtered_df["status"] == "CLOSED"]
        
        if len(closed_trades) > 0:
            # Calculate returns
            closed_trades["entry"] = pd.to_numeric(closed_trades["entry"], errors="coerce")
            closed_trades["result_price"] = pd.to_numeric(closed_trades["result_price"], errors="coerce")
            closed_trades["pct_return"] = np.where(closed_trades["signal"] == "BUY",
                                                (closed_trades["result_price"] - closed_trades["entry"]) / closed_trades["entry"] * 100,
                                                (closed_trades["entry"] - closed_trades["result_price"]) / closed_trades["entry"] * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_trades = len(closed_trades)
                st.metric("Total Trades", total_trades)
            with col2:
                winning_trades = len(closed_trades[closed_trades["result_type"].str.startswith("T")])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                avg_return = closed_trades["pct_return"].mean()
                st.metric("Avg Return", f"{avg_return:.2f}%")
            with col4:
                total_return = closed_trades["pct_return"].sum()
                st.metric("Total Return", f"{total_return:.2f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Returns Distribution")
                st.bar_chart(closed_trades["pct_return"])
            with col2:
                st.subheader("Outcomes by Market")
                outcome_by_market = closed_trades.groupby(["market", "result_type"]).size().unstack(fill_value=0)
                st.bar_chart(outcome_by_market)
        else:
            st.info("No closed trades in the selected filter")
    else:
        st.info("No trades match the selected filters")

# -----------------------
# Main App with Tabs
# -----------------------

# Run live update
if st.session_state.auto_refresh:
    prices, signals = run_live_update()
else:
    prices, signals = {}, {}

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📡 Signals", "📝 Paper Trading", "📊 History"])

with tab1:
    render_live_dashboard(prices, signals)

with tab2:
    render_signals_tab(prices, signals)

with tab3:
    render_paper_trading_tab()

with tab4:
    render_history_tab()

# Clear session alert flags after render (they can be set again when new signal occurs)
st.session_state.signal_alerts = {}

# -----------------------
# Backtester UI: run when user clicks
# -----------------------
if run_backtest_btn:
    with st.spinner("Fetching historical data and running backtest..."):
        summary = run_backtest(backtest_market, backtest_interval, backtest_days)
    if summary is None:
        st.error("Backtest failed. Try different interval or fewer days.")
    elif isinstance(summary, pd.DataFrame) and summary.empty:
        st.info("No trades generated for the selected parameters / period.")
    else:
        st.success(f"Backtest completed: {summary['total_trades']} trades — Win rate {summary['win_rate']:.1f}% — Avg return {summary['avg_return_pct']:.3f}%")
        st.metric("Trades", summary["total_trades"])
        st.metric("Win rate", f"{summary['win_rate']:.1f}%")
        st.metric("Avg return (%)", f"{summary['avg_return_pct']:.3f}%")
        bt = summary["trades_df"].copy()
        # show trade table
        bt_display = bt[["open_time","signal","entry","result_type","result_price","pct_return","outcome"]].copy()
        bt_display = bt_display.rename(columns={"open_time":"Open Time","signal":"Signal","entry":"Entry","result_type":"Result","result_price":"Result Price","pct_return":"% Return","outcome":"Outcome"})
        st.dataframe(bt_display.sort_values("Open Time", ascending=False).head(100), use_container_width=True)
        # quick distribution
        hist_cols = st.columns(2)
        with hist_cols[0]:
            st.subheader("Return distribution")
            st.bar_chart(bt["pct_return"].fillna(0).values)
        with hist_cols[1]:
            st.subheader("Outcomes")
            st.write(bt["outcome"].value_counts())

# -----------------------
# Auto refresh: use Streamlit's native auto-refresh
# -----------------------
if st.session_state.auto_refresh:
    # Use Streamlit's native refresh capability
    refresh_interval = st.session_state.refresh_interval
    time.sleep(refresh_interval)
    st.rerun()
