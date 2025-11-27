# Crypto Intraday Trading Signals & Market Analysis - Crypto Version
"""
Modified from the original intraday terminal to focus on crypto & Gold (Yahoo tickers):
Symbols: GC=F (Gold futures), BTC-USD, ETH-USD, SOL-USD, XRP-USD, LTC-USD
Changes made:
 - Removed Indian-market (NIFTY/.NS) specific lists and currency (₹)
 - Kept trading strategies and logic intact
 - Auto-refresh enabled for 60 seconds (st_autorefresh)
 - Uses USD ($) formatting
 - Assumes crypto markets are 24/7, so market open/close restrictions removed
 - Keeps historical/backtest & demo-data fallbacks using yfinance

Usage: run with `streamlit run crypto_intraday_trader.py`
"""

import time
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Configuration ---
st.set_page_config(page_title="Crypto Intraday Terminal Pro", layout="wide", initial_sidebar_state="expanded")
UTC = pytz.UTC

CAPITAL = 200_000.0  # example capital in USD
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 50
MAX_SYMBOL_TRADES = 20
MAX_AUTO_TRADES = 20

# Auto-refresh interval (60 seconds)
AUTO_REFRESH_MS = 60 * 1000
# trigger autorefresh
st_autorefresh(interval=AUTO_REFRESH_MS, key="crypto_autorefresh")

# Universe of symbols requested by user
CRYPTO_SYMBOLS = ["GC=F", "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD"]

MARKET_OPTIONS = ["CRYPTO"]

# Trading strategies (kept from original design)
TRADING_STRATEGIES = {
    "EMA_VWAP_Confluence": {"name": "EMA + VWAP Confluence", "weight": 3, "type": "BUY"},
    "RSI_MeanReversion": {"name": "RSI Mean Reversion", "weight": 2, "type": "BUY"},
    "Bollinger_Reversion": {"name": "Bollinger Band Reversion", "weight": 2, "type": "BUY"},
    "MACD_Momentum": {"name": "MACD Momentum", "weight": 2, "type": "BUY"},
    "Support_Resistance_Breakout": {"name": "Support/Resistance Breakout", "weight": 3, "type": "BUY"},
    "EMA_VWAP_Downtrend": {"name": "EMA + VWAP Downtrend", "weight": 3, "type": "SELL"},
    "RSI_Overbought": {"name": "RSI Overbought Reversal", "weight": 2, "type": "SELL"},
    "Bollinger_Rejection": {"name": "Bollinger Band Rejection", "weight": 2, "type": "SELL"},
    "MACD_Bearish": {"name": "MACD Bearish Crossover", "weight": 2, "type": "SELL"},
    "Trend_Reversal": {"name": "Trend Reversal", "weight": 2, "type": "SELL"}
}

# --- Utilities & Indicators ---
def now_utc():
    return datetime.now(UTC)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))


def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)


def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_market_profile_vectorized(high, low, close, volume, bins=20):
    low_val = float(min(high.min(), low.min(), close.min()))
    high_val = float(max(high.max(), low.max(), close.max()))
    if np.isclose(low_val, high_val):
        high_val = low_val * 1.01 if low_val != 0 else 1.0
    edges = np.linspace(low_val, high_val, bins + 1)
    hist, _ = np.histogram(close, bins=edges, weights=volume)
    centers = (edges[:-1] + edges[1:]) / 2
    if hist.sum() == 0:
        poc = float(close.iloc[-1])
        va_high = poc * 1.01
        va_low = poc * 0.99
    else:
        idx = int(np.argmax(hist))
        poc = float(centers[idx])
        sorted_idx = np.argsort(hist)[::-1]
        cumulative = 0.0
        total = float(hist.sum())
        selected = []
        for i in sorted_idx:
            selected.append(centers[i])
            cumulative += hist[i]
            if cumulative / total >= 0.70:
                break
        va_high = float(max(selected))
        va_low = float(min(selected))
    profile = [{"price": float(c), "volume": int(v)} for c, v in zip(centers, hist)]
    return {"poc": poc, "value_area_high": va_high, "value_area_low": va_low, "profile": profile}


def calculate_support_resistance_advanced(high, low, close, period=20):
    resistance = []
    support = []
    ln = len(high)
    if ln < period * 2 + 1:
        return {"support": float(close.iloc[-1] * 0.98), "resistance": float(close.iloc[-1] * 1.02),
                "support_levels": [], "resistance_levels": []}
    for i in range(period, ln - period):
        if high.iloc[i] >= high.iloc[i - period:i + period + 1].max():
            resistance.append(float(high.iloc[i]))
        if low.iloc[i] <= low.iloc[i - period:i + period + 1].min():
            support.append(float(low.iloc[i]))
    recent_res = sorted(resistance)[-3:] if resistance else [float(close.iloc[-1] * 1.02)]
    recent_sup = sorted(support)[:3] if support else [float(close.iloc[-1] * 0.98)]
    return {"support": float(np.mean(recent_sup)), "resistance": float(np.mean(recent_res)),
            "support_levels": recent_sup, "resistance_levels": recent_res}


def adx(high, low, close, period=14):
    h = high.copy().reset_index(drop=True)
    l = low.copy().reset_index(drop=True)
    c = close.copy().reset_index(drop=True)
    df = pd.DataFrame({"high": h, "low": l, "close": c})
    df["tr"] = np.maximum(df["high"] - df["low"],
                          np.maximum((df["high"] - df["close"].shift()).abs(),
                                     (df["low"] - df["close"].shift()).abs()))
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]
    df["dm_pos"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
    df["dm_neg"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)
    df["tr_sum"] = df["tr"].rolling(window=period).sum()
    df["dm_pos_sum"] = df["dm_pos"].rolling(window=period).sum()
    df["dm_neg_sum"] = df["dm_neg"].rolling(window=period).sum()
    df["di_pos"] = 100 * (df["dm_pos_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["di_neg"] = 100 * (df["dm_neg_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["dx"] = (abs(df["di_pos"] - df["di_neg"]) / (df["di_pos"] + df["di_neg"]).replace(0, np.nan)) * 100
    df["adx"] = df["dx"].rolling(window=period).mean().fillna(0)
    return df["adx"].values

# --- Data Manager adapted for crypto ---
class CryptoDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.backtest_engine = BacktestEngine()
        self.market_profile_cache = {}
        self.last_rsi_scan = None

    def _validate_live_price(self, symbol):
        now_ts = time.time()
        key = f"price_{symbol}"
        if key in self.price_cache:
            cached = self.price_cache[key]
            if now_ts - cached["ts"] < 2:
                return cached["price"]
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 6), "ts": now_ts}
                return round(price, 6)
            df = ticker.history(period="2d", interval="5m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 6), "ts": now_ts}
                return round(price, 6)
        except Exception:
            pass
        # fallback
        base = 1000.0
        self.price_cache[key] = {"price": float(base), "ts": now_ts}
        return float(base)

    @st.cache_data(ttl=30)
    def _fetch_yf(_self, symbol, period, interval):
        try:
            return yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        except Exception:
            return pd.DataFrame()

    def get_symbol_data(self, symbol, interval="15m"):
        if interval == "15m":
            period = "7d"
        elif interval == "1m":
            period = "1d"
        elif interval == "5m":
            period = "2d"
        else:
            period = "14d"

        df = self._fetch_yf(symbol, period, interval)
        if df is None or df.empty or len(df) < 20:
            return self.create_validated_demo_data(symbol)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for e in expected:
            if e not in df.columns:
                if e.upper() in df.columns:
                    df[e] = df[e.upper()]
                else:
                    return self.create_validated_demo_data(symbol)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 20:
            return self.create_validated_demo_data(symbol)

        try:
            live_price = self._validate_live_price(symbol)
            current_close = df["Close"].iloc[-1]
            price_diff_pct = abs(live_price - current_close) / max(current_close, 1e-12)
            if price_diff_pct > 0.005:
                df.iloc[-1, df.columns.get_loc("Close")] = live_price
                df.iloc[-1, df.columns.get_loc("High")] = max(df.iloc[-1]["High"], live_price)
                df.iloc[-1, df.columns.get_loc("Low")] = min(df.iloc[-1]["Low"], live_price)
        except Exception:
            pass

        # Indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(method="ffill").fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"]) 
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"]) 
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"]) 
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()

        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]

        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]

        try:
            df_adx = adx(df["High"], df["Low"], df["Close"], period=14)
            df["ADX"] = pd.Series(df_adx, index=df.index).fillna(method="ffill").fillna(20)
        except Exception:
            df["ADX"] = 20

        # HTF trend using 1h
        try:
            htf = self._fetch_yf(symbol, period="7d", interval="1h")
            if htf is not None and len(htf) > 50:
                if isinstance(htf.columns, pd.MultiIndex):
                    htf.columns = ["_".join(map(str, col)).strip() for col in htf.columns.values]
                htf = htf.rename(columns={c: c.capitalize() for c in htf.columns})
                htf_close = htf["Close"]
                htf_ema50 = ema(htf_close, 50).iloc[-1]
                htf_ema200 = ema(htf_close, 200).iloc[-1] if len(htf_close) > 200 else ema(htf_close, 100).iloc[-1]
                df["HTF_Trend"] = 1 if htf_ema50 > htf_ema200 else -1
            else:
                df["HTF_Trend"] = 1
        except Exception:
            df["HTF_Trend"] = 1

        return df

    def create_validated_demo_data(self, symbol):
        live = self._validate_live_price(symbol)
        periods = 300
        end = now_utc()
        dates = pd.date_range(end=end, periods=periods, freq="15min")
        base = float(live)
        rng = np.random.default_rng(int(abs(hash(symbol)) % (2 ** 32 - 1)))
        returns = rng.normal(0, 0.0015, periods)
        prices = base * np.cumprod(1 + returns)
        openp = prices * (1 + rng.normal(0, 0.0012, periods))
        highp = prices * (1 + abs(rng.normal(0, 0.0045, periods)))
        lowp = prices * (1 - abs(rng.normal(0, 0.0045, periods)))
        vol = rng.integers(100, 200000, periods)
        df = pd.DataFrame({"Open": openp, "High": highp, "Low": lowp, "Close": prices, "Volume": vol}, index=dates)
        df.iloc[-1, df.columns.get_loc("Close")] = live
        # indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"]) 
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"]) 
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"]) 
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]
        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]
        df["ADX"] = adx(df["High"], df["Low"], df["Close"], period=14)
        df["HTF_Trend"] = 1
        return df

    def get_historical_accuracy(self, symbol, strategy):
        key = f"{symbol}_{strategy}"
        if key in self.backtest_engine.historical_accuracy:
            return self.backtest_engine.historical_accuracy[key]
        data = self.get_symbol_data(symbol, "15m")
        accuracy = self.backtest_engine.calculate_historical_accuracy(symbol, strategy, data)
        self.backtest_engine.historical_accuracy[key] = accuracy
        return accuracy

    def calculate_market_profile_signals(self, symbol):
        try:
            data_15m = self.get_symbol_data(symbol, "15m")
            if len(data_15m) < 50:
                return {"signal": "NEUTRAL", "confidence": 0.5, "reason": "Insufficient data"}
            data_5m = self.get_symbol_data(symbol, "5m")
            current_price_15m = float(data_15m["Close"].iloc[-1])
            current_price_5m = float(data_5m["Close"].iloc[-1]) if len(data_5m) > 0 else current_price_15m
            ema8_15m = float(data_15m["EMA8"].iloc[-1])
            ema21_15m = float(data_15m["EMA21"].iloc[-1])
            ema50_15m = float(data_15m["EMA50"].iloc[-1])
            rsi_val_15m = float(data_15m["RSI14"].iloc[-1])
            vwap_15m = float(data_15m["VWAP"].iloc[-1])
            if len(data_5m) > 0:
                rsi_val_5m = float(data_5m["RSI14"].iloc[-1])
                ema8_5m = float(data_5m["EMA8"].iloc[-1])
                ema21_5m = float(data_5m["EMA21"].iloc[-1])
            else:
                rsi_val_5m = rsi_val_15m
                ema8_5m = ema8_15m
                ema21_5m = ema21_15m
            bullish_score = 0
            bearish_score = 0
            if current_price_15m > ema8_15m > ema21_15m > ema50_15m:
                bullish_score += 3
            elif current_price_15m < ema8_15m < ema21_15m < ema50_15m:
                bearish_score += 3
            if current_price_5m > ema8_5m > ema21_5m:
                bullish_score += 2
            elif current_price_5m < ema8_5m < ema21_5m:
                bearish_score += 2
            if rsi_val_15m > 55 and rsi_val_5m > 50:
                bullish_score += 1
            elif rsi_val_15m < 45 and rsi_val_5m < 50:
                bearish_score += 1
            elif (rsi_val_15m > 55 and rsi_val_5m < 50) or (rsi_val_15m < 45 and rsi_val_5m > 50):
                bullish_score -= 1
                bearish_score -= 1
            if current_price_15m > vwap_15m and current_price_5m > vwap_15m:
                bullish_score += 2
            elif current_price_15m < vwap_15m and current_price_5m < vwap_15m:
                bearish_score += 2
            total_score = max(bullish_score + bearish_score, 1)
            bullish_ratio = (bullish_score + 5) / (total_score + 10)
            price_alignment = 1.0 if abs(current_price_15m - current_price_5m) / current_price_15m < 0.01 else 0.7
            final_confidence = min(0.95, bullish_ratio * price_alignment)
            if bullish_ratio >= 0.65:
                return {"signal": "BULLISH", "confidence": final_confidence, "reason": "Strong bullish alignment across timeframes"}
            elif bullish_ratio <= 0.35:
                return {"signal": "BEARISH", "confidence": final_confidence, "reason": "Strong bearish alignment across timeframes"}
            else:
                return {"signal": "NEUTRAL", "confidence": 0.5, "reason": "Mixed signals across timeframes"}
        except Exception as e:
            return {"signal": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}

    def should_run_rsi_scan(self):
        current_time = time.time()
        if self.last_rsi_scan is None:
            self.last_rsi_scan = current_time
            return True
        if current_time - self.last_rsi_scan >= 75:
            self.last_rsi_scan = current_time
            return True
        return False

# --- Backtest engine (kept largely the same) ---
class BacktestEngine:
    def __init__(self):
        self.historical_accuracy = {}

    def calculate_historical_accuracy(self, symbol, strategy, data):
        if len(data) < 100:
            default_accuracies = {
                "EMA_VWAP_Confluence": 0.68,
                "RSI_MeanReversion": 0.65,
                "Bollinger_Reversion": 0.62,
                "MACD_Momentum": 0.66,
                "Support_Resistance_Breakout": 0.60,
                "EMA_VWAP_Downtrend": 0.65,
                "RSI_Overbought": 0.63,
                "Bollinger_Rejection": 0.61,
                "MACD_Bearish": 0.64,
                "Trend_Reversal": 0.59
            }
            return default_accuracies.get(strategy, 0.65)
        wins = 0
        total_signals = 0
        for i in range(50, len(data)-3):
            current_data = data.iloc[:i+1]
            if len(current_data) < 30:
                continue
            signal_data = self.generate_signal_for_backtest(current_data, strategy)
            if signal_data and signal_data['action'] in ['BUY', 'SELL']:
                total_signals += 1
                entry_price = data.iloc[i]['Close']
                future_prices = data.iloc[i+1:i+4]['Close']
                if len(future_prices) > 0:
                    if signal_data['action'] == 'BUY':
                        max_future_price = future_prices.max()
                        if max_future_price > entry_price * 1.002:
                            wins += 1
                    else:
                        min_future_price = future_prices.min()
                        if min_future_price < entry_price * 0.998:
                            wins += 1
        if total_signals < 5:
            default_accuracies = {
                "EMA_VWAP_Confluence": 0.68,
                "RSI_MeanReversion": 0.65,
                "Bollinger_Reversion": 0.62,
                "MACD_Momentum": 0.66,
                "Support_Resistance_Breakout": 0.60,
                "EMA_VWAP_Downtrend": 0.65,
                "RSI_Overbought": 0.63,
                "Bollinger_Rejection": 0.61,
                "MACD_Bearish": 0.64,
                "Trend_Reversal": 0.59
            }
            accuracy = default_accuracies.get(strategy, 0.65)
        else:
            accuracy = wins / total_signals
        if accuracy >= 0.65:
            return max(0.65, min(0.85, accuracy))
        else:
            return 0.0

    def generate_signal_for_backtest(self, data, strategy):
        if len(data) < 30:
            return None
        try:
            current = data.iloc[-1]
            live = float(current['Close'])
            ema8 = float(current['EMA8'])
            ema21 = float(current['EMA21'])
            ema50 = float(current['EMA50'])
            rsi_val = float(current['RSI14'])
            atr = float(current['ATR'])
            macd_line = float(current['MACD'])
            macd_signal = float(current['MACD_Signal'])
            vwap = float(current['VWAP'])
            support = float(current['Support'])
            resistance = float(current['Resistance'])
            bb_upper = float(current['BB_Upper'])
            bb_lower = float(current['BB_Lower'])
            vol_latest = float(current['Volume'])
            vol_avg = float(data['Volume'].rolling(20).mean().iloc[-1])
            volume_spike = vol_latest > vol_avg * 1.3
            adx_val = float(current['ADX'])
            htf_trend = int(current['HTF_Trend'])
            # BUY
            if strategy == "EMA_VWAP_Confluence":
                if (ema8 > ema21 > ema50 and live > vwap and adx_val > 20 and htf_trend == 1):
                    return {'action': 'BUY', 'confidence': 0.82}
            elif strategy == "RSI_MeanReversion":
                rsi_prev = float(data.iloc[-2]['RSI14']) if len(data) > 1 else rsi_val
                if rsi_val < 30 and rsi_val > rsi_prev and live > support:
                    return {'action': 'BUY', 'confidence': 0.78}
            elif strategy == "Bollinger_Reversion":
                if live <= bb_lower and rsi_val < 35 and live > support:
                    return {'action': 'BUY', 'confidence': 0.75}
            elif strategy == "MACD_Momentum":
                if (macd_line > macd_signal and macd_line > 0 and ema8 > ema21 and live > vwap and adx_val > 22 and htf_trend == 1):
                    return {'action': 'BUY', 'confidence': 0.80}
            elif strategy == "Support_Resistance_Breakout":
                if (live > resistance and volume_spike and rsi_val > 50 and htf_trend == 1 and ema8 > ema21 and macd_line > macd_signal):
                    return {'action': 'BUY', 'confidence': 0.75}
            # SELL
            elif strategy == "EMA_VWAP_Downtrend":
                if (ema8 < ema21 < ema50 and live < vwap and adx_val > 20 and htf_trend == -1):
                    return {'action': 'SELL', 'confidence': 0.78}
            elif strategy == "RSI_Overbought":
                rsi_prev = float(data.iloc[-2]['RSI14']) if len(data) > 1 else rsi_val
                if rsi_val > 70 and rsi_val < rsi_prev and live < resistance:
                    return {'action': 'SELL', 'confidence': 0.72}
            elif strategy == "Bollinger_Rejection":
                if live >= bb_upper and rsi_val > 65 and live < resistance:
                    return {'action': 'SELL', 'confidence': 0.70}
            elif strategy == "MACD_Bearish":
                if (macd_line < macd_signal and macd_line < 0 and ema8 < ema21 and live < vwap and adx_val > 22 and htf_trend == -1):
                    return {'action': 'SELL', 'confidence': 0.75}
            elif strategy == "Trend_Reversal":
                if len(data) > 5:
                    prev_trend = 1 if data.iloc[-3]['EMA8'] > data.iloc[-3]['EMA21'] else -1
                    current_trend = -1 if ema8 < ema21 else 1
                    if prev_trend == 1 and current_trend == -1 and rsi_val > 60:
                        return {'action': 'SELL', 'confidence': 0.68}
        except Exception:
            return None
        return None

# --- Multi-strategy trader simplified for crypto ---
class MultiStrategyCryptoTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.symbol_trades = 0
        self.auto_trades_count = 0
        self.last_reset = now_utc().date()
        self.selected_market = "CRYPTO"
        self.auto_execution = False
        self.signal_history = []
        self.auto_close_triggered = False
        self.strategy_performance = {s: {"signals":0, "trades":0, "wins":0, "pnl":0.0} for s in TRADING_STRATEGIES}

    def reset_daily_counts(self):
        if now_utc().date() != self.last_reset:
            self.daily_trades = 0
            self.symbol_trades = 0
            self.auto_trades_count = 0
            self.last_reset = now_utc().date()

    def can_auto_trade(self):
        return (self.auto_trades_count < MAX_AUTO_TRADES and self.daily_trades < MAX_DAILY_TRADES)

    def calculate_support_resistance(self, symbol, current_price):
        try:
            data = data_manager.get_symbol_data(symbol, "15m")
            if data is None or len(data) < 20:
                return current_price * 0.98, current_price * 1.02
            return float(data["Support"].iloc[-1]), float(data["Resistance"].iloc[-1])
        except Exception:
            return current_price * 0.98, current_price * 1.02

    def calculate_intraday_target_sl(self, entry_price, action, atr, current_price, support, resistance):
        if atr <= 0 or np.isnan(atr):
            atr = max(entry_price * 0.005, 0.0001)
        if action == "BUY":
            sl = entry_price - (atr * 1.2)
            target = entry_price + (atr * 2.5)
            if target > resistance:
                target = min(target, resistance * 0.998)
            sl = max(sl, support * 0.995)
        else:
            sl = entry_price + (atr * 1.2)
            target = entry_price - (atr * 2.5)
            if target < support:
                target = max(target, support * 1.002)
            sl = min(sl, resistance * 1.005)
        rr = abs(target - entry_price) / max(abs(entry_price - sl), 1e-12)
        if rr < 2.0:
            if action == "BUY":
                target = entry_price + max((entry_price - sl) * 2.0, atr * 2.0)
            else:
                target = entry_price - max((sl - entry_price) * 2.0, atr * 2.0)
        return round(float(target), 6), round(float(sl), 6)

    def equity(self):
        total = float(self.cash)
        for symbol, pos in self.positions.items():
            if pos.get("status") == "OPEN":
                try:
                    data = data_manager.get_symbol_data(symbol, "5m")
                    price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                    total += pos["quantity"] * price
                except Exception:
                    total += pos["quantity"] * pos["entry_price"]
        return total

    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, win_probability=0.75, auto_trade=False, strategy=None):
        self.reset_daily_counts()
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        if self.symbol_trades >= MAX_SYMBOL_TRADES:
            return False, "Symbol trade limit reached"
        if auto_trade and self.auto_trades_count >= MAX_AUTO_TRADES:
            return False, "Auto trade limit reached"
        trade_value = float(quantity) * float(price)
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"
        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {"trade_id": trade_id, "symbol": symbol, "action": action, "quantity": int(quantity),
                  "entry_price": float(price), "stop_loss": float(stop_loss) if stop_loss else None,
                  "target": float(target) if target else None, "timestamp": now_utc(), "status": "OPEN",
                  "current_pnl": 0.0, "current_price": float(price), "win_probability": float(win_probability),
                  "closed_pnl": 0.0, "entry_time": now_utc().strftime("%H:%M:%S"), "auto_trade": auto_trade, "strategy": strategy}
        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2
            record["margin_used"] = margin
            self.positions[symbol] = record
            self.cash -= margin
        self.symbol_trades += 1
        self.trade_log.append(record)
        self.daily_trades += 1
        if auto_trade:
            self.auto_trades_count += 1
        if strategy and strategy in self.strategy_performance:
            self.strategy_performance[strategy]["trades"] += 1
        return True, f"{('[AUTO] ' if auto_trade else '')}{action} {int(quantity)} {symbol} @ ${price:.6f} | Strategy: {strategy}"

    def update_positions_pnl(self):
        for symbol, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue
            try:
                data = data_manager.get_symbol_data(symbol, "5m")
                if data is not None and len(data) > 0:
                    price = float(data["Close"].iloc[-1])
                    pos["current_price"] = price
                    entry = pos["entry_price"]
                    if pos["action"] == "BUY":
                        pnl = (price - entry) * pos["quantity"]
                    else:
                        pnl = (entry - price) * pos["quantity"]
                    pos["current_pnl"] = float(pnl)
                    pos["max_pnl"] = max(pos.get("max_pnl", 0.0), float(pnl))
                    sl = pos.get("stop_loss")
                    tg = pos.get("target")
                    if sl is not None:
                        if (pos["action"] == "BUY" and price <= sl) or (pos["action"] == "SELL" and price >= sl):
                            self.close_position(symbol, exit_price=sl)
                            continue
                    if tg is not None:
                        if (pos["action"] == "BUY" and price >= tg) or (pos["action"] == "SELL" and price <= tg):
                            self.close_position(symbol, exit_price=tg)
                            continue
            except Exception:
                continue

    def auto_close_all_positions(self):
        for sym in list(self.positions.keys()):
            self.close_position(sym)

    def close_position(self, symbol, exit_price=None):
        if symbol not in self.positions:
            return False, "Position not found"
        pos = self.positions[symbol]
        if exit_price is None:
            try:
                data = data_manager.get_symbol_data(symbol, "5m")
                exit_price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
            except Exception:
                exit_price = pos["entry_price"]
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            self.cash += pos.get("margin_used", 0) + (pos["quantity"] * pos["entry_price"])
        pos["status"] = "CLOSED"
        pos["exit_price"] = float(exit_price)
        pos["closed_pnl"] = float(pnl)
        pos["exit_time"] = now_utc()
        pos["exit_time_str"] = now_utc().strftime("%H:%M:%S")
        strategy = pos.get("strategy")
        if strategy and strategy in self.strategy_performance:
            if pnl > 0:
                self.strategy_performance[strategy]["wins"] += 1
            self.strategy_performance[strategy]["pnl"] += pnl
        try:
            del self.positions[symbol]
        except Exception:
            pass
        return True, f"Closed {symbol} @ ${exit_price:.6f} | P&L: ${pnl:+.2f}"

    def get_open_positions_data(self):
        self.update_positions_pnl()
        out = []
        for symbol, pos in self.positions.items():
            if pos.get("status") != "OPEN":
                continue
            try:
                data = data_manager.get_symbol_data(symbol, "5m")
                price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                if pos["action"] == "BUY":
                    pnl = (price - pos["entry_price"]) * pos["quantity"]
                else:
                    pnl = (pos["entry_price"] - price) * pos["quantity"]
                var = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
                sup, res = self.calculate_support_resistance(symbol, price)
                strategy = pos.get("strategy", "Manual")
                historical_accuracy = data_manager.get_historical_accuracy(symbol, strategy) if strategy != "Manual" else 0.65
                out.append({
                    "Symbol": symbol,
                    "Action": pos["action"],
                    "Quantity": pos["quantity"],
                    "Entry Price": f"${pos['entry_price']:.6f}",
                    "Current Price": f"${price:.6f}",
                    "P&L": f"${pnl:+.2f}",
                    "Variance %": f"{var:+.2f}%",
                    "Stop Loss": f"${pos.get('stop_loss', 0):.6f}",
                    "Target": f"${pos.get('target', 0):.6f}",
                    "Support": f"${sup:.6f}",
                    "Resistance": f"${res:.6f}",
                    "Historical Win %": f"{historical_accuracy:.1%}",
                    "Current Win %": f"{pos.get('win_probability', 0.75)*100:.1f}%",
                    "Entry Time": pos.get("entry_time"),
                    "Auto Trade": "Yes" if pos.get("auto_trade") else "No",
                    "Strategy": strategy,
                    "Status": pos.get("status")
                })
            except Exception:
                continue
        return out

    def get_trade_history_data(self):
        history_data = []
        for trade in self.trade_log:
            if trade.get("status") == "CLOSED":
                pnl = trade.get("closed_pnl", 0)
                trade_class = "trade-buy" if trade.get("action") == "BUY" else "trade-sell"
                history_data.append({
                    "Trade ID": trade.get("trade_id", ""),
                    "Symbol": trade.get("symbol", ""),
                    "Action": trade.get("action", ""),
                    "Quantity": trade.get("quantity", 0),
                    "Entry Price": f"${trade.get('entry_price', 0):.6f}",
                    "Exit Price": f"${trade.get('exit_price', 0):.6f}",
                    "P&L": f"${pnl:+.2f}",
                    "Entry Time": trade.get("entry_time", ""),
                    "Exit Time": trade.get("exit_time_str", ""),
                    "Strategy": trade.get("strategy", "Manual"),
                    "Auto Trade": "Yes" if trade.get("auto_trade") else "No",
                    "Duration": self.calculate_trade_duration(trade.get("entry_time"), trade.get("exit_time_str")),
                    "_row_class": trade_class
                })
        return history_data

    def calculate_trade_duration(self, entry_time_str, exit_time_str):
        try:
            if entry_time_str and exit_time_str:
                fmt = "%H:%M:%S"
                entry_time = datetime.strptime(entry_time_str, fmt).time()
                exit_time = datetime.strptime(exit_time_str, fmt).time()
                today = datetime.now().date()
                entry_dt = datetime.combine(today, entry_time)
                exit_dt = datetime.combine(today, exit_time)
                duration = (exit_dt - entry_dt).total_seconds() / 60
                return f"{int(duration)} min"
        except:
            pass
        return "N/A"

    def get_performance_stats(self):
        self.update_positions_pnl()
        closed = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        total_trades = len(closed)
        open_pnl = sum([p.get("current_pnl", 0) for p in self.positions.values() if p.get("status") == "OPEN"])
        if total_trades == 0:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0, "open_positions": len(self.positions), "open_pnl": open_pnl, "auto_trades": self.auto_trades_count}
        wins = len([t for t in closed if t.get("closed_pnl", 0) > 0])
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed])
        win_rate = wins / total_trades if total_trades else 0.0
        avg_pnl = total_pnl / total_trades if total_trades else 0.0
        auto_trades = [t for t in self.trade_log if t.get("auto_trade")]
        auto_closed = [t for t in auto_trades if t.get("status") == "CLOSED"]
        auto_win_rate = len([t for t in auto_closed if t.get("closed_pnl", 0) > 0]) / len(auto_closed) if auto_closed else 0.0
        return {"total_trades": total_trades, "win_rate": win_rate, "total_pnl": total_pnl, "avg_pnl": avg_pnl, "open_positions": len(self.positions), "open_pnl": open_pnl, "auto_trades": self.auto_trades_count, "auto_win_rate": auto_win_rate}

    def generate_strategy_signals(self, symbol, data):
        signals = []
        if data is None or len(data) < 30:
            return signals
        try:
            live = float(data["Close"].iloc[-1])
            ema8 = float(data["EMA8"].iloc[-1])
            ema21 = float(data["EMA21"].iloc[-1])
            ema50 = float(data["EMA50"].iloc[-1])
            rsi_val = float(data["RSI14"].iloc[-1])
            atr = float(data["ATR"].iloc[-1]) if "ATR" in data.columns else max(live*0.005,1e-6)
            macd_line = float(data["MACD"].iloc[-1])
            macd_signal = float(data["MACD_Signal"].iloc[-1])
            vwap = float(data["VWAP"].iloc[-1])
            support = float(data["Support"].iloc[-1])
            resistance = float(data["Resistance"].iloc[-1])
            bb_upper = float(data["BB_Upper"].iloc[-1])
            bb_lower = float(data["BB_Lower"].iloc[-1])
            vol_latest = float(data["Volume"].iloc[-1])
            vol_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data["Volume"]) >= 20 else float(data["Volume"].mean())
            volume_spike = vol_latest > vol_avg * 1.3
            adx_val = float(data["ADX"].iloc[-1]) if "ADX" in data.columns else 20
            htf_trend = int(data["HTF_Trend"].iloc[-1]) if "HTF_Trend" in data.columns else 1

            # Strategy checks (same as original, filtered by historical accuracy >= 65%)
          # EMA_VWAP_Confluence
if (ema8 > ema21 > ema50 and live > vwap and adx_val > 20 and htf_trend == 1):
    action = "BUY"
    confidence = 0.82
    score = 9
    strategy = "EMA_VWAP_Confluence"

    target, stop_loss = self.calculate_intraday_target_sl(
        entry_price=live,
        action=action,
        atr=atr,
        current_price=live,
        support=support,
        resistance=resistance
    )

    signals.append({
        "strategy": strategy,
        "action": action,
        "confidence": confidence,
        "score": score,
        "entry_price": live,
        "target": target,
        "stop_loss": stop_loss
    })
