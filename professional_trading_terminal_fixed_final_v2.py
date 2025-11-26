# trading_terminal_web_gold_crypto_forex_v3.py
"""
Market Terminal v3
- Markets: Gold (GC=F), BTC-USD, ETH-USD, XRP-USD, SOL-USD, EURUSD=X, GBPUSD=X, USDJPY=X
- Strategy: RSI(14) + SMA5/SMA15 confirmation
- Signals: BUY when RSI<30 and SMA5>SMA15; SELL when RSI>70 and SMA5<SMA15
- Targets: T1=+1% T2=+2% T3=+3% (BUY) / reverse for SELL
- SL: 1% adverse move
- Option A: Automatic trade result tracking (logs to CSV)
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
from datetime import datetime

# -----------------------
# CONFIG
# -----------------------
TRADE_LOG_FILE = "trade_log.csv"
STYLES = """
<style>
/* Dark card styling */
.card {
  background: linear-gradient(180deg, #121212 0%, #1b1b1b 100%);
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 12px;
  border: 1px solid rgba(255,255,255,0.04);
}
.header {
  display:flex; align-items:center; justify-content:space-between;
}
.pulse {
  animation: pulse 1.6s infinite;
  box-shadow: 0 0 12px rgba(255,200,60,0.25);
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(255,200,60,0.35); }
  70% { box-shadow: 0 0 0 12px rgba(255,200,60,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,200,60,0); }
}
.small-muted { color: #9aa0a6; font-size:12px; }
.metric { font-weight:700; font-size:18px; }
</style>
"""
AUDIO_ALERT_URL = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"

# -----------------------
# Helper functions
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

def play_sound_and_toast(message: str):
    # HTML5 audio autoplay + toast
    st.markdown(f"""
        <audio autoplay>
            <source src="{AUDIO_ALERT_URL}" type="audio/ogg">
            Your browser does not support the audio element.
        </audio>
    """, unsafe_allow_html=True)
    st.toast(message, icon="🔔")

def format_price_for_display(name, price):
    if price is None or price == 0:
        return "N/A"
    usd_like = ["BITCOIN","ETHEREUM","GOLD (FUT)","XRP","SOLANA"]
    if name in usd_like:
        if price < 10:
            return f"${price:,.6f}"
        return f"${price:,.2f}"
    return f"{price:,.4f}"

# -----------------------
# Indicators & signal logic
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
    if len(series) < period:
        return float(series.mean()) if len(series) > 0 else 0.0
    return float(series.iloc[-period:].mean())

def get_signal_from_prices(prices):
    clean = [p for p in prices if p and p > 0]
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

def calculate_targets_and_sl(price, signal):
    if price <= 0 or price is None:
        return None
    if signal == "BUY":
        return {
            "entry": price,
            "t1": price * 1.01,
            "t2": price * 1.02,
            "t3": price * 1.03,
            "sl": price * 0.99
        }
    elif signal == "SELL":
        return {
            "entry": price,
            "t1": price * 0.99,
            "t2": price * 0.98,
            "t3": price * 0.97,
            "sl": price * 1.01
        }
    return None

# -----------------------
# Markets config
# -----------------------
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
# App state initialize
# -----------------------
st.set_page_config(page_title="Market Terminal — Gold/Crypto/Forex (v3)", layout="wide", page_icon="💱")
st.markdown(STYLES, unsafe_allow_html=True)
st.title("💱 Market Terminal — Gold, Crypto & Forex (Auto Tracker)")

if "initialized_v3" not in st.session_state:
    st.session_state.initialized_v3 = True
    st.session_state.price_history = {k: [] for k in MARKETS.keys()}
    st.session_state.ohlc = {}
    st.session_state.last_signals = {}
    st.session_state.targets = {}
    st.session_state.signal_alerts = {}
    st.session_state.refresh_interval = 25
    st.session_state.auto_refresh = True
    st.session_state.update_count = 0
    ensure_trade_log()

# -----------------------
# Sidebar controls & strategy explanation
# -----------------------
with st.sidebar:
    st.header("Controls")
    st.checkbox("Auto Refresh", value=st.session_state.auto_refresh, key="auto_refresh_v3")
    st.session_state.auto_refresh = st.session_state.auto_refresh_v3
    st.slider("Refresh Interval (seconds)", 10, 120, st.session_state.refresh_interval, key="refresh_interval_v3")
    st.session_state.refresh_interval = st.session_state.refresh_interval_v3
    st.markdown("---")
    st.subheader("Strategy (RSI + SMA)")
    st.markdown("""
    - **RSI(14)**: oversold <30, overbought >70  
    - **SMA5 / SMA15**: short vs mid trend  
    - **BUY** when RSI < 30 **and** SMA5 > SMA15  
    - **SELL** when RSI > 70 **and** SMA5 < SMA15  
    - **Targets** (T1/T2/T3) are 1% / 2% / 3% from entry; **SL** = 1% adverse
    """)
    st.markdown("---")
    st.subheader("Trade Log")
    if st.button("Open trade log (CSV)"):
        df = load_trade_log()
        st.write(df.tail(20))
    if st.button("Clear trade log (CAUTION)"):
        # backup then clear
        if os.path.exists(TRADE_LOG_FILE):
            backup = TRADE_LOG_FILE.replace(".csv", f"_{int(time.time())}.bak.csv")
            os.rename(TRADE_LOG_FILE, backup)
        ensure_trade_log()
        st.success("Trade log cleared (backup made if existed).")
    st.markdown("---")
    st.caption("Note: Auto-tracking marks trades based on live price reaching targets/SL.")

# -----------------------
# Fetch latest OHLC snapshot (1m)
# -----------------------
def fetch_ohlc(sym):
    try:
        df = yf.Ticker(sym).history(period="1d", interval="1m")
        if df is None or df.empty:
            return None
        open_price = float(df["Open"].iloc[0])
        current = float(df["Close"].iloc[-1])
        high = float(df["High"].max())
        low = float(df["Low"].min())
        vol = int(df["Volume"].sum()) if "Volume" in df else 0
        change = current - open_price
        pct = (change / open_price) * 100 if open_price != 0 else 0
        return {"open": open_price, "current": current, "high": high, "low": low, "vol": vol, "change": change, "pct": pct}
    except Exception as e:
        return None

# -----------------------
# Update routine: fetch, compute signals, log new signals, track open trades
# -----------------------
def run_update():
    prices = {}
    signals = {}
    resolved_trades = []
    for name, ticker in MARKETS.items():
        ohlc = fetch_ohlc(ticker)
        if ohlc is None:
            continue
        prices[name] = ohlc["current"]
        # update history
        hist = st.session_state.price_history.get(name, [])
        hist.append(ohlc["current"])
        st.session_state.price_history[name] = hist[-200:]  # keep last 200
        # indicators & signal
        sig_data = get_signal_from_prices(st.session_state.price_history[name])
        signals[name] = {"signal": sig_data["signal"], "rsi": sig_data["rsi"], "sma5": sig_data["sma5"], "sma15": sig_data["sma15"], "ohlc": ohlc}
        # always compute + store targets for BUY/SELL
        if signals[name]["signal"] in ["BUY", "SELL"]:
            t = calculate_targets_and_sl(ohlc["current"], signals[name]["signal"])
            st.session_state.targets[name] = t
        else:
            st.session_state.targets.pop(name, None)
        # handle signal change => log new trade (if BUY/SELL and not same as last)
        last_signal = st.session_state.last_signals.get(name, "HOLD")
        if signals[name]["signal"] in ["BUY", "SELL"] and signals[name]["signal"] != last_signal:
            # create new trade log entry
            tid = int(time.time()*1000)  # unique id
            t = st.session_state.targets.get(name)
            row = {
                "id": tid,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "market": name,
                "signal": signals[name]["signal"],
                "entry": round(t["entry"], 6) if t else None,
                "t1": round(t["t1"], 6) if t else None,
                "t2": round(t["t2"], 6) if t else None,
                "t3": round(t["t3"], 6) if t else None,
                "sl": round(t["sl"], 6) if t else None,
                "status": "OPEN",
                "result_time": "",
                "result_type": "",
                "result_price": "",
                "notes": "AUTO-LOG"
            }
            append_trade_row(row)
            st.session_state.signal_alerts[name] = True
            # play audio & visual toast
            play_sound_and_toast(f"{signals[name]['signal']} — {name}")
        st.session_state.last_signals[name] = signals[name]["signal"]

    # Now check open trades and resolve if price reached any target/SL
    df = load_trade_log()
    open_trades = df[df["status"] == "OPEN"]
    for _, trade in open_trades.iterrows():
        tid = int(trade["id"])
        market = trade["market"]
        entry = float(trade["entry"]) if not pd.isna(trade["entry"]) else None
        sl = float(trade["sl"]) if not pd.isna(trade["sl"]) else None
        t1 = float(trade["t1"]) if not pd.isna(trade["t1"]) else None
        t2 = float(trade["t2"]) if not pd.isna(trade["t2"]) else None
        t3 = float(trade["t3"]) if not pd.isna(trade["t3"]) else None
        # current price
        cur_price = prices.get(market)
        if cur_price is None:
            continue
        signal = trade["signal"]
        resolved = False
        result_type = ""
        # BUY: check targets upward first (T1..T3) then SL below
        if signal == "BUY":
            if t1 and cur_price >= t1:
                resolved = True; result_type = "T1"
                result_price = cur_price
            if not resolved and t2 and cur_price >= t2:
                resolved = True; result_type = "T2"; result_price = cur_price
            if not resolved and t3 and cur_price >= t3:
                resolved = True; result_type = "T3"; result_price = cur_price
            if not resolved and sl and cur_price <= sl:
                resolved = True; result_type = "SL"; result_price = cur_price
        # SELL: check targets downward first (T1..T3) then SL above
        elif signal == "SELL":
            if t1 and cur_price <= t1:
                resolved = True; result_type = "T1"; result_price = cur_price
            if not resolved and t2 and cur_price <= t2:
                resolved = True; result_type = "T2"; result_price = cur_price
            if not resolved and t3 and cur_price <= t3:
                resolved = True; result_type = "T3"; result_price = cur_price
            if not resolved and sl and cur_price >= sl:
                resolved = True; result_type = "SL"; result_price = cur_price
        if resolved:
            # Mark trade as WON if target hit (T1/T2/T3), LOSS if SL
            status = "CLOSED"
            outcome = "WON" if result_type.startswith("T") else "LOSS"
            update_trade_row(tid, {
                "status": status,
                "result_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "result_type": result_type,
                "result_price": round(result_price, 6),
                "notes": f"AUTO_RESOLVED"
            })
            resolved_trades.append((tid, market, outcome, result_type))
            # play small sound/notify on resolution
            play_sound_and_toast(f"{market} trade {outcome} ({result_type})")
    st.session_state.update_count += 1
    return prices, signals, resolved_trades

# -----------------------
# Render dashboard
# -----------------------
def render_dashboard(prices, signals):
    # Top metrics
    df_log = load_trade_log()
    total_closed = len(df_log[df_log["status"] == "CLOSED"])
    total_open = len(df_log[df_log["status"] == "OPEN"])
    wins = len(df_log[(df_log["status"] == "CLOSED") & (df_log["result_type"].str.startswith("T"))])
    losses = len(df_log[(df_log["status"] == "CLOSED") & (df_log["result_type"] == "SL")])
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    avg_return = None
    if total_closed > 0:
        # crude return estimate using entry->result_price
        closed = df_log[df_log["status"] == "CLOSED"].copy()
        closed["entry"] = pd.to_numeric(closed["entry"], errors="coerce")
        closed["result_price"] = pd.to_numeric(closed["result_price"], errors="coerce")
        closed["pct_return"] = np.where(closed["signal"] == "BUY",
                                        (closed["result_price"] - closed["entry"]) / closed["entry"] * 100,
                                        (closed["entry"] - closed["result_price"]) / closed["entry"] * 100)
        avg_return = closed["pct_return"].mean()
    # layout header
    col1, col2, col3, col4 = st.columns([2,1,1,1])
    with col1:
        st.markdown("<div class='card header'><div><h2 style='margin:0'>📊 Live Market Dashboard</h2><div class='small-muted'>RSI + SMA strategy</div></div>"
                    f"<div style='text-align:right'><div class='metric'>{st.session_state.update_count} updates</div><div class='small-muted'>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div></div></div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><div class='small-muted'>Open Trades</div><div class='metric'>{total_open}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='card'><div class='small-muted'>Closed Trades</div><div class='metric'>{total_closed}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='card'><div class='small-muted'>Win Rate</div><div class='metric'>{win_rate:.1f}%</div></div>", unsafe_allow_html=True)

    # Two-column grid for market cards
    markets = list(MARKETS.keys())
    left_col, right_col = st.columns(2)
    for i, name in enumerate(markets):
        s = signals.get(name)
        price = prices.get(name)
        target_info = st.session_state.targets.get(name)
        card_html = "<div class='card"
        # pulse if alert
        if st.session_state.signal_alerts.get(name, False) and s and s["signal"] in ["BUY","SELL"]:
            card_html += " pulse"
        card_html += f"'>"
        # header row
        sig = s["signal"] if s else "N/A"
        emoji = "🟢" if sig == "BUY" else "🔴" if sig == "SELL" else "⚪"
        color = "#00FF00" if sig == "BUY" else "#FF4444" if sig == "SELL" else "#888888"
        card_html += f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        card_html += f"<div><strong style='font-size:16px'>{name} {emoji}</strong><div class='small-muted'>Ticker: {MARKETS[name]}</div></div>"
        card_html += f"<div style='text-align:right'><div style='font-weight:700; font-size:18px'>{format_price_for_display(name, price)}</div>"
        if s:
            card_html += f"<div class='small-muted'>RSI {s['rsi']:.1f} | SMA5 {s['sma5']:.4f} | SMA15 {s['sma15']:.4f}</div>"
        card_html += "</div></div><hr style='opacity:0.06'/>"
        # targets + entry + sl
        if target_info:
            entry = target_info["entry"]
            t1 = target_info["t1"]
            t2 = target_info["t2"]
            t3 = target_info["t3"]
            sl = target_info["sl"]
            card_html += f"<div><strong>Entry:</strong> {format_price_for_display(name, entry)} &nbsp;&nbsp;"
            card_html += f"<strong>SL:</strong> {format_price_for_display(name, sl)}</div>"
            card_html += "<div style='margin-top:6px'>"
            card_html += f"<span class='small-muted'>T1:</span> <strong>{format_price_for_display(name,t1)}</strong> &nbsp; "
            card_html += f"<span class='small-muted'>T2:</span> <strong>{format_price_for_display(name,t2)}</strong> &nbsp; "
            card_html += f"<span class='small-muted'>T3:</span> <strong>{format_price_for_display(name,t3)}</strong>"
            card_html += "</div>"
        else:
            card_html += "<div class='small-muted'>No active targets</div>"
        card_html += "</div>"  # end card
        if i % 2 == 0:
            left_col.markdown(card_html, unsafe_allow_html=True)
        else:
            right_col.markdown(card_html, unsafe_allow_html=True)
    # Performance panel (charts + logs)
    st.markdown("---")
    perf_col1, perf_col2 = st.columns([2,1])
    with perf_col1:
        st.subheader("📈 Performance Summary")
        st.write(f"Total closed trades: {total_closed} | Wins: {wins} | Losses: {losses} | Avg return: {avg_return:.2f}% " if avg_return is not None else f"Total closed trades: {total_closed}")
        # show last 20 closed trades
        closed_df = df_log[df_log["status"] == "CLOSED"].sort_values("result_time", ascending=False)
        if not closed_df.empty:
            st.dataframe(closed_df.head(20)[["timestamp","market","signal","entry","result_price","result_type","result_time"]], use_container_width=True)
        else:
            st.info("No closed trades yet.")
    with perf_col2:
        st.subheader("Recent Open Trades")
        open_df = df_log[df_log["status"] == "OPEN"]
        if not open_df.empty:
            st.dataframe(open_df[["timestamp","market","signal","entry","t1","t2","t3","sl"]], use_container_width=True)
        else:
            st.info("No open trades.")
    st.markdown("---")
    st.caption("Trades are auto-logged and auto-resolved when live price reaches target or SL. Use the sidebar to view or clear the CSV log.")

# -----------------------
# Run and auto-refresh
# -----------------------
prices, signals, resolved = run_update()
render_dashboard(prices, signals)

# clear visual alert markers after display cycle
st.session_state.signal_alerts = {}

# Auto refresh logic: rerun periodically (Streamlit environment)
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.experimental_rerun()
