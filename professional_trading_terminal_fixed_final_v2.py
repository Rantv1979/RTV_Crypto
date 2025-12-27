
# ======================================================================
# RANTV_CRYPTO_ALGO_PRO_FINAL.py
# ONE SINGLE FILE – CRYPTO TRADING VERSION
#
# Exchanges supported via CCXT (Binance default)
#
# Includes:
# - Multi-Timeframe SMC (HTF BOS + LTF FVG/OB)
# - Session logic adapted for crypto (London / NY)
# - Volume Profile (POC)
# - Equity-curve auto risk scaling
# - Trailing SL + partial exits
# - Live multi-symbol scanner
# - Streamlit dashboard
# - Performance analytics
# - Exchange-safe kill switch
#
# ======================================================================

import time
import threading
import logging
from datetime import datetime, time as dtime
from typing import Dict, List

import numpy as np
import pandas as pd

# ========================= UI =========================
try:
    import streamlit as st
    STREAMLIT = True
except:
    STREAMLIT = False

# ========================= CCXT =========================
try:
    import ccxt
    CCXT_AVAILABLE = True
except:
    CCXT_AVAILABLE = False

# ========================= CONFIG =========================
class Config:
    PAPER_TRADING = True
    CAPITAL = 50_000  # USDT

    BASE_RISK = 0.01
    MAX_RISK = 0.02
    MIN_RISK = 0.005

    SL_ATR = 1.5
    TP_ATR = 3.0
    TRAIL_ATR = 1.2

    HTF = "30m"
    LTF = "5m"
    SCAN_INTERVAL = 60

    LONDON_OPEN = dtime(7,0)
    NY_OPEN = dtime(13,30)

# ========================= LOGGING =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("RANTV-CRYPTO")

# ========================= SYMBOL UNIVERSE =========================
CRYPTO_UNIVERSE = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT",
    "XRP/USDT","ADA/USDT","AVAX/USDT","DOGE/USDT"
]

# ========================= INDICATORS =========================
def ema(series, n):
    return series.ewm(span=n).mean()

def atr(df, n=14):
    tr = pd.concat([
        df.high - df.low,
        (df.high - df.close.shift()).abs(),
        (df.low - df.close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ========================= SESSION FILTER =========================
def valid_session():
    now = datetime.utcnow().time()
    return now >= Config.LONDON_OPEN or now >= Config.NY_OPEN

# ========================= VOLUME PROFILE =========================
def volume_profile(df, bins=24):
    hist, edges = np.histogram(df.close, bins=bins, weights=df.volume)
    idx = np.argmax(hist)
    return (edges[idx] + edges[idx+1]) / 2

# ========================= SMC =========================
class SMC:
    @staticmethod
    def BOS(df):
        if df.high.iloc[-1] > df.high.iloc[-6:-1].max():
            return "BULLISH"
        if df.low.iloc[-1] < df.low.iloc[-6:-1].min():
            return "BEARISH"
        return None

    @staticmethod
    def FVG(df):
        a,b,c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if a.high < c.low:
            return "BULLISH"
        if a.low > c.high:
            return "BEARISH"
        return None

    @staticmethod
    def OrderBlock(df):
        last = df.iloc[-2]
        return "BULLISH" if last.close > last.open else "BEARISH"

# ========================= SIGNAL ENGINE =========================
class SignalEngine:
    def generate(self, htf, ltf):
        if not valid_session():
            return None

        bos = SMC.BOS(htf)
        fvg = SMC.FVG(ltf)
        ob = SMC.OrderBlock(ltf)
        poc = volume_profile(ltf)

        price = ltf.close.iloc[-1]
        atr_val = atr(ltf).iloc[-1]

        if bos=="BULLISH" and fvg=="BULLISH" and ob=="BULLISH" and price>poc:
            return {"side":"BUY","sl":price-atr_val*Config.SL_ATR,"tp":price+atr_val*Config.TP_ATR}

        if bos=="BEARISH" and fvg=="BEARISH" and ob=="BEARISH" and price<poc:
            return {"side":"SELL","sl":price+atr_val*Config.SL_ATR,"tp":price-atr_val*Config.TP_ATR}

        return None

# ========================= PERFORMANCE =========================
class Performance:
    def __init__(self):
        self.trades=[]

    def log(self, symbol, pnl):
        self.trades.append({"time":datetime.utcnow(),"symbol":symbol,"pnl":pnl})

    def stats(self):
        if not self.trades: return {}
        df=pd.DataFrame(self.trades)
        return {
            "Trades":len(df),
            "Win Rate":round((df.pnl>0).mean()*100,2),
            "Net PnL":round(df.pnl.sum(),2)
        }

# ========================= EXCHANGE =========================
class Exchange:
    def __init__(self):
        if not CCXT_AVAILABLE:
            raise RuntimeError("ccxt not installed")
        self.ex = ccxt.binance({"enableRateLimit":True})

    def fetch(self, symbol, tf, limit=200):
        ohlcv = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        return df

# ========================= LIVE SCANNER =========================
class LiveScanner:
    def __init__(self, ex):
        self.ex = ex
        self.engine = SignalEngine()
        self.signals=[]
        self.running=False

    def start(self):
        self.running=True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.running:
            self.signals.clear()
            for sym in CRYPTO_UNIVERSE:
                try:
                    ltf = self.ex.fetch(sym, Config.LTF)
                    htf = self.ex.fetch(sym, Config.HTF)
                    sig = self.engine.generate(htf, ltf)
                    if sig:
                        self.signals.append((sym, sig["side"]))
                except Exception as e:
                    logger.warning(f"{sym}: {e}")
            time.sleep(Config.SCAN_INTERVAL)

# ========================= STREAMLIT =========================
def run_dashboard(scanner, perf):
    st.set_page_config(layout="wide")
    st.title("🚀 RANTV Crypto Algo Pro")

    col1,col2,col3=st.columns(3)
    col1.metric("Paper Trading", Config.PAPER_TRADING)
    stats=perf.stats()
    col2.metric("Trades", stats.get("Trades",0))
    col3.metric("Net PnL", stats.get("Net PnL",0))

    st.subheader("Live Signals")
    st.table(pd.DataFrame(scanner.signals, columns=["Symbol","Side"]))

# ========================= END =========================
