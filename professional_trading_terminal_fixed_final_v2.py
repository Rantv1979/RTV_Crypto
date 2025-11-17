# professional_trading_terminal_fixed_final_v4.py
"""
Updated Trading Terminal
- UI auto-refresh every 15s
- Signal generation every 30s (separate timer)
- Prevent duplicate signals across recent generations
- Auto-execute trades for newly generated signals (respecting max positions)
- Execution timestamps included in trade_log (timestamp field)
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque

warnings.filterwarnings("ignore")

# --- Constants / Config ---
REFRESH_INTERVAL = 15  # seconds (UI auto-refresh)
SIGNAL_GEN_INTERVAL = 30  # seconds (signal generation frequency)
CHART_INTERVALS = ["1m", "5m", "15m", "1h", "1d"]
DEFAULT_CHART_INTERVAL = "5m"
SCAN_INTERVALS = ["5m", "15m"]  # include scalping 15m
MIN_TRADE_CONFIDENCE = 0.75
MAX_CONCURRENT_POSITIONS = 10
RECENT_SIGNAL_TTL = 300  # seconds to keep recent signals IDs to avoid duplicates

PROFESSIONAL_SYMBOLS = {
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "NZDUSD=X": "NZD/USD", "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "ADA-USD": "Cardano",
    "DOT-USD": "Polkadot", "LINK-USD": "Chainlink", "SOL-USD": "Solana",
    "XRP-USD": "Ripple", "AVAX-USD": "Avalanche",
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil",
    "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"
}

# -------------------------
# Session state init
# -------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_refresh = datetime.now()
    st.session_state.last_signal_gen = datetime.now() - timedelta(seconds=SIGNAL_GEN_INTERVAL)
    st.session_state.last_trade_time = 0
    st.session_state.trade_log = []
    st.session_state.signals = {}
    st.session_state.force_refresh = False
    st.session_state.chart_symbol = "BTC-USD"
    st.session_state.chart_interval = DEFAULT_CHART_INTERVAL
    st.session_state.refresh_count = 0
    st.session_state.auto_trade_enabled = True # Enabled auto-trade by default
    st.session_state.auto_trade_config = {'trade_amount': 1000.0, 'max_positions': 5}
    st.session_state.paper_trading = {
        'balance': 50000.0,
        'positions': {},
        'trade_history': [],
        'unrealized_pnl': 0.0,
        'realized_pnl': 0.0,
        'total_trades': 0,
        'winning_trades': 0
    }
    # recent_signal_ids: deque of tuples (signal_id, timestamp) to prevent duplicates
    st.session_state.recent_signal_ids = deque()
    # executed_signals: map signal_id -> timestamp when auto-executed
    st.session_state.executed_signals = {}

# -------------------------
# Cached top-level data helpers (no 'self' arg)
# -------------------------
@st.cache_data(ttl=6, show_spinner=False)
def _cached_get_live_chart_data(symbol: str, interval: str) -> pd.DataFrame:
    try:
        period_map = {"1m": "1d", "5m": "1d", "15m": "5d", "1h": "1mo", "1d": "3mo"}
        period = period_map.get(interval, "1d")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False, rounding=True)
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.tz_localize(None) if getattr(df.index, 'tz', None) is not None else df.index
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=4, show_spinner=False)
def _cached_get_current_price(symbol: str) -> Optional[float]:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m", actions=False)
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        info = getattr(ticker, 'info', {}) or {}
        for k in ('regularMarketPrice', 'currentPrice', 'ask', 'bid', 'previousClose'):
            if k in info and info[k] is not None:
                return float(info[k])
    except Exception:
        pass
    return None

@st.cache_data(ttl=10, show_spinner=False)
def _cached_get_trading_data(symbol: str, period: str = "2d", interval: str = "5m") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False, rounding=True)
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.tz_localize(None) if getattr(df.index, 'tz', None) is not None else df.index
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def _cached_get_day_change(symbol: str) -> Optional[float]:
    try:
        ticker = yf.Ticker(symbol)
        prev = ticker.history(period='3d', interval='1d')
        if not prev.empty and len(prev) > 1:
            prev_close = prev['Close'].iloc[-2]
            current_close = prev['Close'].iloc[-1]
            return ((current_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
    except Exception:
        pass
    return None

# -------------------------
# TA helpers
# -------------------------
class TechnicalAnalysis:
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()
    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=window, adjust=False).mean().fillna(method='bfill')
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3):
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        k = ((close - lowest_low) / denom) * 100
        d = k.rolling(window=d_window).mean()
        return k.fillna(50), d.fillna(50)
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20):
        if df.empty:
            return pd.Series(), pd.Series()
        df = df.copy()
        df['Resistance'] = df['High'].rolling(window=window, center=True).max().shift(-1)
        df['Support'] = df['Low'].rolling(window=window, center=True).min().shift(-1)
        df['Resistance'] = df['Resistance'].fillna(method='bfill').fillna(method='ffill')
        df['Support'] = df['Support'].fillna(method='bfill').fillna(method='ffill')
        return df['Support'], df['Resistance']

# -------------------------
# Data manager (calls cached helpers)
# -------------------------
class ProfessionalDataManager:
    def __init__(self):
        self.symbols = PROFESSIONAL_SYMBOLS
        self.ta = TechnicalAnalysis()
    def get_live_chart_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        return _cached_get_live_chart_data(symbol, interval)
    def get_current_price(self, symbol: str) -> Optional[float]:
        return _cached_get_current_price(symbol)
    def get_day_change_percent(self, symbol: str) -> Optional[float]:
        return _cached_get_day_change(symbol)
    def get_trading_data(self, symbol: str, period: str = "2d", interval: str = "5m") -> pd.DataFrame:
        return _cached_get_trading_data(symbol, period, interval)
    def calculate_pro_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df['SMA_20'] = self.ta.sma(df['Close'], 20)
        df['SMA_50'] = self.ta.sma(df['Close'], 50)
        df['EMA_12'] = self.ta.ema(df['Close'], 12)
        df['EMA_26'] = self.ta.ema(df['Close'], 26)
        bb_upper, bb_middle, bb_lower = self.ta.bollinger_bands(df['Close'], 20, 2)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Middle'] = bb_middle
        df['RSI'] = self.ta.rsi(df['Close'], 14)
        macd, signal, hist = self.ta.macd(df['Close'], 12, 26, 9)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = hist
        stoch_k, stoch_d = self.ta.stochastic(df['High'], df['Low'], df['Close'], 14, 3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        df['Volume_SMA'] = df['Volume'].rolling(20).mean().fillna(method='bfill')
        df['Volume_Ratio'] = (df['Volume'] / (df['Volume_SMA'].replace(0, np.nan))).fillna(1.0)
        df['ATR'] = self.ta.atr(df['High'], df['Low'], df['Close'], 14)
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        vol_cumsum = df['Volume'].cumsum().replace(0, np.nan)
        df['VWAP'] = (typical * df['Volume']).cumsum() / vol_cumsum
        df['Support'], df['Resistance'] = self.ta.calculate_support_resistance(df, 20)
        return df

# -------------------------
# Signals & Paper Trading
# -------------------------
class ProfessionalSignal:
    def __init__(self, symbol: str, action: str, strategy: str, timeframe: str,
                 entry: float, stop_loss: float, take_profit: float,
                 confidence: float, volume: float, indicators: dict,
                 support: float = None, resistance: float = None):
        self.symbol = symbol
        self.action = action
        self.strategy = strategy
        self.timeframe = timeframe
        self.entry = entry
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.volume = volume
        self.indicators = indicators
        self.support = support
        self.resistance = resistance
        self.timestamp = datetime.now()
    def id(self):
        # Unique deterministic id for deduplication: symbol|strategy|timeframe|rounded_entry
        return f"{self.symbol}|{self.strategy}|{self.timeframe}|{round(self.entry, 6)}"

class PaperTradingEngine:
    MARGIN_RATE = 0.20
    COMMISSION_RATE = 0.001
    def __init__(self, initial_balance: float = 50000.0):
        self.initial_balance = initial_balance
    def ensure(self):
        if 'paper_trading' not in st.session_state:
            self.reset()
    def reset(self):
        st.session_state.paper_trading = {
            'balance': self.initial_balance,
            'positions': {},
            'trade_history': [],
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0
        }
        st.session_state.trade_log = []
    def get_balance(self) -> float:
        self.ensure()
        return st.session_state.paper_trading['balance']
    def get_positions(self) -> dict:
        self.ensure()
        return st.session_state.paper_trading['positions']
    def portfolio_value(self) -> float:
        self.ensure()
        cash = st.session_state.paper_trading['balance']
        positions_value = 0.0
        for pos in st.session_state.paper_trading['positions'].values():
            side = pos['side']
            quantity = pos['quantity']
            current = pos.get('current_price', pos['entry_price'])
            if side == 'LONG':
                positions_value += quantity * current
            else:
                positions_value += pos.get('margin_locked', 0.0) + pos.get('unrealized_pnl', 0.0)
        return cash + positions_value
    def place_trade(self, symbol: str, action: str, quantity: float, price: float, signal: ProfessionalSignal = None) -> bool:
        self.ensure()
        action = action.upper()
        if quantity <= 0 or price <= 0:
            return False
        commission = quantity * price * self.COMMISSION_RATE
        is_long_side = symbol in st.session_state.paper_trading['positions'] and st.session_state.paper_trading['positions'][symbol]['side'] == 'LONG'
        is_short_side = symbol in st.session_state.paper_trading['positions'] and st.session_state.paper_trading['positions'][symbol]['side'] == 'SHORT'
        if action == "BUY":
            cost = quantity * price
            total_cost = cost + commission
            if st.session_state.paper_trading['balance'] >= total_cost:
                st.session_state.paper_trading['balance'] -= total_cost
                if is_long_side:
                    existing = st.session_state.paper_trading['positions'][symbol]
                    total_qty = existing['quantity'] + quantity
                    total_cost_basis = (existing['entry_price'] * existing['quantity']) + (price * quantity)
                    avg_price = total_cost_basis / total_qty
                    existing.update({
                        'quantity': total_qty,
                        'entry_price': avg_price,
                        'current_price': price,
                        'entry_time': existing['entry_time']
                    })
                else:
                    st.session_state.paper_trading['positions'][symbol] = {
                        'side': 'LONG',
                        'quantity': quantity,
                        'entry_price': price,
                        'current_price': price,
                        'entry_time': datetime.now(),
                        'signal': signal,
                        'cost_basis': price
                    }
                entry = datetime.now()
                st.session_state.trade_log.append({
                    'timestamp': entry,
                    'execution_time': entry,
                    'symbol': symbol,
                    'action': 'BUY' if not is_long_side else 'BUY_ADD',
                    'quantity': quantity,
                    'price': price,
                    'commission': commission,
                    'signal': signal.strategy if signal else 'Manual'
                })
                st.session_state.paper_trading['total_trades'] += 1
                return True
            return False
        elif action == "SELL":
            notional = quantity * price
            margin_required = notional * self.MARGIN_RATE
            total_required = margin_required + commission
            if st.session_state.paper_trading['balance'] >= total_required:
                st.session_state.paper_trading['balance'] -= total_required
                if is_short_side:
                    existing = st.session_state.paper_trading['positions'][symbol]
                    total_qty = existing['quantity'] + quantity
                    total_notional = existing['entry_price'] * existing['quantity'] + price * quantity
                    avg_entry = total_notional / total_qty
                    existing.update({
                        'quantity': total_qty,
                        'entry_price': avg_entry,
                        'current_price': price,
                        'margin_locked': existing.get('margin_locked', 0.0) + margin_required,
                        'entry_time': existing['entry_time']
                    })
                else:
                    st.session_state.paper_trading['positions'][symbol] = {
                        'side': 'SHORT',
                        'quantity': quantity,
                        'entry_price': price,
                        'current_price': price,
                        'margin_locked': margin_required,
                        'entry_time': datetime.now(),
                        'signal': signal
                    }
                entry = datetime.now()
                st.session_state.trade_log.append({
                    'timestamp': entry,
                    'execution_time': entry,
                    'symbol': symbol,
                    'action': 'SELL_OPEN' if not is_short_side else 'SELL_ADD',
                    'quantity': quantity,
                    'price': price,
                    'commission': commission,
                    'margin_locked': margin_required,
                    'signal': signal.strategy if signal else 'Manual'
                })
                st.session_state.paper_trading['total_trades'] += 1
                return True
            return False
        return False
    def close_position(self, symbol: str, price: float, reason: str, quantity: Optional[float] = None) -> bool:
        self.ensure()
        if symbol not in st.session_state.paper_trading['positions']:
            return False
        pos = st.session_state.paper_trading['positions'][symbol]
        side = pos['side']
        if quantity is None:
            quantity = pos['quantity']
        if quantity <= 0 or quantity > pos['quantity']:
            return False
        commission = quantity * price * self.COMMISSION_RATE
        if side == 'LONG':
            proceeds = quantity * price
            pnl = (price - pos['entry_price']) * quantity - commission
            st.session_state.paper_trading['balance'] += (proceeds - commission)
            st.session_state.paper_trading['realized_pnl'] += pnl
            if pnl > 0:
                st.session_state.paper_trading['winning_trades'] += 1
            pos['quantity'] -= quantity
            if pos['quantity'] <= 0:
                del st.session_state.paper_trading['positions'][symbol]
            else:
                pos['current_price'] = price
            entry = datetime.now()
            st.session_state.trade_log.append({
                'timestamp': entry,
                'execution_time': entry,
                'symbol': symbol,
                'action': f'CLOSE_LONG ({reason})',
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'commission': commission
            })
            return True
        else:
            cost_to_buy = quantity * price
            pnl = (pos['entry_price'] - price) * quantity - commission
            margin_locked = pos.get('margin_locked', 0.0)
            locked_release = 0.0
            if pos['quantity'] > 0:
                locked_release = margin_locked * (quantity / pos['quantity'])
            st.session_state.paper_trading['balance'] += locked_release + pnl
            st.session_state.paper_trading['realized_pnl'] += pnl
            if pnl > 0:
                st.session_state.paper_trading['winning_trades'] += 1
            pos['quantity'] -= quantity
            if pos['quantity'] <= 0:
                del st.session_state.paper_trading['positions'][symbol]
            else:
                pos['margin_locked'] -= locked_release
                pos['current_price'] = price
            entry = datetime.now()
            st.session_state.trade_log.append({
                'timestamp': entry,
                'execution_time': entry,
                'symbol': symbol,
                'action': f'CLOSE_SHORT ({reason})',
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'commission': commission,
                'released_margin': locked_release
            })
            return True
    def update_positions(self, dm: ProfessionalDataManager):
        self.ensure()
        unrealized_pnl = 0.0
        to_close = []
        for symbol, pos in list(st.session_state.paper_trading['positions'].items()):
            current_price = dm.get_current_price(symbol)
            if not current_price:
                continue
            pos['current_price'] = current_price
            if pos['side'] == 'LONG':
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                unrealized_pnl += pnl
                pos['unrealized_pnl'] = pnl
                signal = pos.get('signal')
                if signal:
                    if current_price <= signal.stop_loss:
                        to_close.append((symbol, current_price, "SL Hit"))
                    elif current_price >= signal.take_profit:
                        to_close.append((symbol, current_price, "TP Hit"))
            else:
                pnl = (pos['entry_price'] - current_price) * pos['quantity']
                unrealized_pnl += pnl
                pos['unrealized_pnl'] = pnl
                signal = pos.get('signal')
                if signal:
                    if current_price >= signal.stop_loss:
                        to_close.append((symbol, current_price, "SL Hit"))
                    elif current_price <= signal.take_profit:
                        to_close.append((symbol, current_price, "TP Hit"))
            if (datetime.now() - pos['entry_time']).total_seconds() > 86400 * 3:
                to_close.append((symbol, current_price, "Time Exit"))
        for symbol, price, reason in to_close:
            try:
                self.close_position(symbol, price, reason)
            except Exception:
                pass 
        st.session_state.paper_trading['unrealized_pnl'] = unrealized_pnl

# -------------------------
# Trading Terminal
# -------------------------
class ProfessionalTradingTerminal:
    def __init__(self):
        self.dm = ProfessionalDataManager()
        self.pt = PaperTradingEngine()
        if 'trading_engine' not in st.session_state:
            st.session_state.trading_engine = {'active_positions': 0, 'daily_pnl': 0.0, 'today_trades': 0, 'signals_generated': 0}
    def _cleanup_recent_signals(self):
        """Remove old entries from recent_signal_ids based on TTL"""
        now = datetime.now()
        while st.session_state.recent_signal_ids and (now - st.session_state.recent_signal_ids[0][1]).total_seconds() > RECENT_SIGNAL_TTL:
            st.session_state.recent_signal_ids.popleft()
    # ---------- Auto-refresh implementation ----------
    def check_auto_refresh(self):
        current_time = datetime.now()
        time_since_refresh = (current_time - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= REFRESH_INTERVAL:
            st.session_state.last_refresh = current_time
            st.session_state.refresh_count += 1
            # UI refresh occurs; but signal generation/execution controlled separately
            # Check if it's time to generate signals (every SIGNAL_GEN_INTERVAL)
            time_since_sig = (current_time - st.session_state.last_signal_gen).total_seconds()
            if time_since_sig >= SIGNAL_GEN_INTERVAL:
                st.session_state.last_signal_gen = current_time
                # Generate signals (this function will deduplicate)
                new_count = self.generate_all_signals()
                # Auto-trade newly generated signals if enabled
                if st.session_state.auto_trade_enabled and new_count > 0:
                    cfg = st.session_state.auto_trade_config
                    self.execute_auto_trading(cfg.get('trade_amount', 1000.0), cfg.get('max_positions', 5))
            # Force UI refresh
            st.rerun()
    def generate_all_signals(self):
        """Generate signals for all symbols and timeframes; avoid duplicates using recent_signal_ids"""
        all_signals = []
        scan_symbols = ["EURUSD=X", "BTC-USD", "ETH-USD", "GC=F", "^GSPC"]
        for symbol in scan_symbols:
            for timeframe in SCAN_INTERVALS:
                signals = self.run_pro_strategies(symbol, timeframe)
                all_signals.extend(signals)
        confirmed = [sig for sig in all_signals if sig.confidence >= MIN_TRADE_CONFIDENCE]
        confirmed.sort(key=lambda x: x.confidence, reverse=True)
        # Deduplicate against recent_signal_ids
        self._cleanup_recent_signals()
        recent_ids = {sid for sid, ts in st.session_state.recent_signal_ids}
        new_signals = []
        for sig in confirmed[:80]:
            sid = sig.id()
            if sid not in recent_ids:
                new_signals.append(sig)
                st.session_state.recent_signal_ids.append((sid, datetime.now()))
        # Update session signals to include the latest (replace old list)
        st.session_state.signals = {f"{sig.symbol}_{sig.strategy}_{sig.timeframe}_{round(sig.entry,6)}": sig for sig in confirmed[:80]}
        st.session_state.trading_engine['signals_generated'] = len(confirmed)
        return len(new_signals)
    # ---------- Strategies (same as before, omitted here for brevity) ----------
    def strategy_trend_following(self, symbol: str, df: pd.DataFrame, timeframe: str):
        signals = []
        if len(df) < 50:
            return signals
        try:
            current = df['Close'].iloc[-1]
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            ema_12 = df['EMA_12'].iloc[-1]
            ema_26 = df['EMA_26'].iloc[-1]
            atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else max(0.0001, current * 0.01)
            volume_ratio = float(df['Volume_Ratio'].iloc[-1]) if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
            rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            support = df['Support'].iloc[-1] if not pd.isna(df['Support'].iloc[-1]) else current * 0.98
            resistance = df['Resistance'].iloc[-1] if not pd.isna(df['Resistance'].iloc[-1]) else current * 1.02
            bullish_trend = (current > ema_12 > ema_26 > sma_20 > sma_50)
            momentum_confirmation = (df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1])
            if bullish_trend and momentum_confirmation and rsi < 70:
                stop_loss = min(current - 2 * atr, support)
                risk = current - stop_loss
                take_profit = current + 3 * risk
                confidence = 0.82
                if volume_ratio > 1.5:
                    confidence += 0.08
                if timeframe == "15m":
                    confidence += 0.05
                if rsi < 60:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "BUY", f"Trend_Following_{timeframe}", timeframe,
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'RSI': rsi, 'MACD': df['MACD'].iloc[-1]}, support, resistance))
            bearish_trend = (current < ema_12 < ema_26 < sma_20 < sma_50)
            momentum_bear = (df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1])
            if bearish_trend and momentum_bear and rsi > 30:
                stop_loss = max(current + 2 * atr, resistance)
                risk = stop_loss - current
                take_profit = current - 3 * risk
                confidence = 0.82
                if volume_ratio > 1.5:
                    confidence += 0.08
                if timeframe == "15m":
                    confidence += 0.05
                if rsi > 40:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "SELL", f"Trend_Following_{timeframe}", timeframe,
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'RSI': rsi, 'MACD': df['MACD'].iloc[-1]}, support, resistance))
        except Exception:
            pass
        return signals
    def strategy_mean_reversion_pro(self, symbol: str, df: pd.DataFrame, timeframe: str):
        signals = []
        if len(df) < 30:
            return signals
        try:
            current = df['Close'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            bb_upper = df['BB_Upper'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            stoch_k = df['Stoch_K'].iloc[-1]
            atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else max(0.0001, current * 0.01)
            support = df['Support'].iloc[-1] if not pd.isna(df['Support'].iloc[-1]) else current * 0.98
            resistance = df['Resistance'].iloc[-1] if not pd.isna(df['Resistance'].iloc[-1]) else current * 1.02
            if (current <= bb_lower and rsi < 35 and stoch_k < 20 and 
                df['Close'].iloc[-1] > df['Close'].iloc[-2] and
                current <= support * 1.01):
                stop_loss = min(current - 1.5 * atr, support * 0.99)
                risk = current - stop_loss
                take_profit = current + 2.5 * risk
                confidence = 0.78
                if rsi < 25:
                    confidence += 0.07
                if stoch_k < 10:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "BUY", f"Mean_Reversion_{timeframe}", timeframe,
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'RSI': rsi, 'Stoch_K': stoch_k}, support, resistance))
            if (current >= bb_upper and rsi > 65 and stoch_k > 80 and 
                df['Close'].iloc[-1] < df['Close'].iloc[-2] and
                current >= resistance * 0.99):
                stop_loss = max(current + 1.5 * atr, resistance * 1.01)
                risk = stop_loss - current
                take_profit = current - 2.5 * risk
                confidence = 0.78
                if rsi > 75:
                    confidence += 0.07
                if stoch_k > 90:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "SELL", f"Mean_Reversion_{timeframe}", timeframe,
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'RSI': rsi, 'Stoch_K': stoch_k}, support, resistance))
        except Exception:
            pass
        return signals
    def strategy_breakout_pro(self, symbol: str, df: pd.DataFrame, timeframe: str):
        signals = []
        if len(df) < 25:
            return signals
        try:
            current = df['Close'].iloc[-1]
            resistance = df['Resistance'].iloc[-2] if len(df) > 2 else df['High'].iloc[-1]
            support = df['Support'].iloc[-2] if len(df) > 2 else df['Low'].iloc[-1]
            volume_ratio = float(df['Volume_Ratio'].iloc[-1]) if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
            atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else max(0.0001, current * 0.01)
            rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            if current > resistance and volume_ratio > 2.0 and rsi < 70:
                stop_loss = min(current - 2 * atr, resistance)
                risk = current - stop_loss
                take_profit = current + 3 * risk
                confidence = min(0.88, 0.75 + (volume_ratio - 1.5) * 0.1)
                if rsi > 50:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "BUY", f"Breakout_{timeframe}", timeframe,
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'Volume_Ratio': volume_ratio, 'RSI': rsi}, support, resistance))
            if current < support and volume_ratio > 2.0 and rsi > 30:
                stop_loss = max(current + 2 * atr, support)
                risk = stop_loss - current
                take_profit = current - 3 * risk
                confidence = min(0.88, 0.75 + (volume_ratio - 1.5) * 0.1)
                if rsi < 50:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "SELL", f"Breakout_{timeframe}", timeframe,
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'Volume_Ratio': volume_ratio, 'RSI': rsi}, support, resistance))
        except Exception:
            pass
        return signals
    def strategy_scalping_15m(self, symbol: str, df: pd.DataFrame):
        signals = []
        if len(df) < 30:
            return signals
        try:
            current = df['Close'].iloc[-1]
            ema_short = df['EMA_12'].iloc[-1]
            ema_long = df['EMA_26'].iloc[-1]
            atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else max(0.00001, current * 0.005)
            volume_ratio = float(df['Volume_Ratio'].iloc[-1]) if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1.0
            rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            support = df['Support'].iloc[-1] if not pd.isna(df['Support'].iloc[-1]) else current * 0.98
            resistance = df['Resistance'].iloc[-1] if not pd.isna(df['Resistance'].iloc[-1]) else current * 1.02
            if (ema_short > ema_long and df['Close'].iloc[-1] > df['Open'].iloc[-1] and 
                volume_ratio > 1.2 and rsi < 65 and current > support):
                stop_loss = min(current - 0.8 * atr, support)
                take_profit = current + 1.6 * atr
                confidence = 0.76 + (min(volume_ratio - 1.2, 0.6) * 0.05)
                if rsi > 40:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "BUY", "Scalping_15m", "15m",
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'ATR': atr, 'Volume_Ratio': volume_ratio, 'RSI': rsi}, support, resistance))
            if (ema_short < ema_long and df['Close'].iloc[-1] < df['Open'].iloc[-1] and 
                volume_ratio > 1.2 and rsi > 35 and current < resistance):
                stop_loss = max(current + 0.8 * atr, resistance)
                take_profit = current - 1.6 * atr
                confidence = 0.76 + (min(volume_ratio - 1.2, 0.6) * 0.05)
                if rsi < 60:
                    confidence += 0.05
                signals.append(ProfessionalSignal(symbol, "SELL", "Scalping_15m", "15m",
                                                  current, stop_loss, take_profit, confidence, float(df['Volume'].iloc[-1]),
                                                  {'ATR': atr, 'Volume_Ratio': volume_ratio, 'RSI': rsi}, support, resistance))
        except Exception:
            pass
        return signals
    def run_pro_strategies(self, symbol: str, timeframe: str):
        df = self.dm.get_trading_data(symbol, period="2d", interval=timeframe)
        if df.empty:
            return []
        df = self.dm.calculate_pro_indicators(df)
        if df.empty:
            return []
        signals = []
        signals.extend(self.strategy_trend_following(symbol, df, timeframe))
        signals.extend(self.strategy_mean_reversion_pro(symbol, df, timeframe))
        signals.extend(self.strategy_breakout_pro(symbol, df, timeframe))
        if timeframe == '15m':
            signals.extend(self.strategy_scalping_15m(symbol, df))
        return signals
    # ---------- Charting & display (omitted heavy charting code for brevity in this update) ----------
    def display_trading_signals(self):
        st.markdown("## 📡 Live Trading Signals")
        confirmed = list(st.session_state.signals.values())
        confirmed.sort(key=lambda x: x.confidence, reverse=True)
        if not confirmed:
            st.info("No high-confidence signals right now.")
            return
        st.markdown(f"### Top {min(12, len(confirmed))} Signals (Auto-generated)")
        st.info(f"Last signal generation: {st.session_state.last_signal_gen.strftime('%Y-%m-%d %H:%M:%S')}")
        for i, sig in enumerate(confirmed[:12]):
            with st.expander(f"{sig.symbol} | {sig.strategy} | {sig.timeframe} | {sig.action} | Conf {sig.confidence:.2f}", expanded=(i==0)):
                cols = st.columns(5)
                with cols[0]:
                    st.write(f"**Action:** {sig.action}")
                    st.write(f"**Timeframe:** {sig.timeframe}")
                    st.write(f"**Strategy:** {sig.strategy}")
                with cols[1]:
                    st.write(f"**Entry:** ${sig.entry:.6f}")
                    st.write(f"**Stop:** ${sig.stop_loss:.6f}")
                    st.write(f"**TP:** ${sig.take_profit:.6f}")
                with cols[2]:
                    risk = abs(sig.entry - sig.stop_loss)
                    reward = abs(sig.take_profit - sig.entry)
                    rr = (reward / risk) if risk != 0 else 0
                    st.write(f"**R:R:** {rr:.2f}")
                    st.write(f"**Vol:** {sig.volume:,.0f}")
                with cols[3]:
                    if sig.support and sig.resistance:
                        st.write(f"**Support:** ${sig.support:.6f}")
                        st.write(f"**Resistance:** ${sig.resistance:.6f}")
                with cols[4]:
                    if st.button("Execute (Paper)", key=f"exec_manual_{i}_{sig.symbol}"):
                        price = self.dm.get_current_price(sig.symbol)
                        if price:
                            qty = self._calc_quantity_from_risk(sig, st.session_state.auto_trade_config.get('trade_amount', 1000.0))
                            placed = self.pt.place_trade(sig.symbol, sig.action, qty, price, sig)
                            if placed:
                                st.success(f"Placed {sig.action} {sig.symbol} qty {qty:.6f} @ {price:.6f}")
                                st.rerun()
                            else:
                                st.warning("Insufficient balance to place this manual trade.")
                        else:
                            st.warning("Price unavailable.")
    def _calc_quantity_from_risk(self, sig: ProfessionalSignal, trade_amount: float):
        risk_amount = trade_amount * 0.02 
        price_diff = abs(sig.entry - sig.stop_loss) if sig.stop_loss and sig.entry != sig.stop_loss else max(1e-6, sig.entry * 0.01)
        quantity = max(1e-6, risk_amount / price_diff)
        notional = quantity * sig.entry
        if notional > st.session_state.paper_trading['balance'] * 5:
            quantity = (st.session_state.paper_trading['balance'] * 5) / sig.entry
        return quantity
    def display_portfolio(self):
        st.markdown("## 💼 Portfolio")
        self.pt.update_positions(self.dm)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cash", f"${self.pt.get_balance():,.2f}")
        with col2:
            st.metric("Portfolio Value", f"${self.pt.portfolio_value():,.2f}")
        with col3:
            st.metric("Realized P&L", f"${st.session_state.paper_trading['realized_pnl']:,.2f}")
        with col4:
            unrealized_pnl = st.session_state.paper_trading['unrealized_pnl']
            st.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}", f"{(unrealized_pnl / self.pt.get_balance() * 100):.2f}%" if self.pt.get_balance() > 0 else "0.00%")
        st.markdown("### Active Positions")
        positions = self.pt.get_positions()
        if positions:
            for sym, pos in positions.items():
                c1, c2, c3, c4, c5, c6 = st.columns([2,1,1,1,1,1])
                with c1:
                    st.write(f"**{sym}**")
                    st.write(f"Side: {pos['side']}")
                    if pos.get('signal'):
                        st.write(f"*{pos.get('signal').strategy}*")
                with c2:
                    st.write(f"Qty: {pos['quantity']:.6f}")
                    st.write(f"Entry: ${pos['entry_price']:.6f}")
                with c3:
                    curr = pos.get('current_price', pos['entry_price'])
                    st.write(f"Price: ${curr:.6f}")
                with c4:
                    pnl = pos.get('unrealized_pnl', 0.0)
                    notional = pos['quantity'] * pos['entry_price']
                    pct = (pnl / notional) * 100 if notional != 0 else 0
                    color = "green" if pnl >= 0 else "red"
                    st.markdown(f"<span style='color:{color}'>P&L: ${pnl:.2f}<br>({pct:+.2f}%)</span>", unsafe_allow_html=True)
                with c5:
                    if pos.get('signal'):
                        st.write(f"SL: ${pos['signal'].stop_loss:.6f}")
                        st.write(f"TP: ${pos['signal'].take_profit:.6f}")
                with c6:
                    if st.button("Close", key=f"close_{sym}"):
                        curr = pos.get('current_price', pos['entry_price'])
                        self.pt.close_position(sym, curr, "Manual Close")
                        st.rerun()
                st.markdown("---")
        else:
            st.info("No active positions.")
    def display_auto_trading(self):
        st.markdown("## 🤖 Auto-Trading")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Config")
            auto_trade = st.checkbox("Enable Auto-Trading", value=st.session_state.auto_trade_enabled, key="auto_trade_enabled_checkbox")
            st.session_state.auto_trade_enabled = auto_trade
            trade_amount = st.number_input("Trade Size ($)", min_value=100.0, max_value=100000.0, value=float(st.session_state.auto_trade_config.get('trade_amount',1000.0)), step=100.0, key="trade_amount_input")
            max_positions = st.slider("Max Positions", 1, 10, int(st.session_state.auto_trade_config.get('max_positions',5)), key="max_positions_slider")
            st.session_state.auto_trade_config['trade_amount'] = float(trade_amount)
            st.session_state.auto_trade_config['max_positions'] = int(max_positions)
            if st.button("🔄 Execute Auto-Trading Cycle Now (Manual Run)"):
                with st.spinner("Running auto-trading..."):
                    self.execute_auto_trading(st.session_state.auto_trade_config['trade_amount'], st.session_state.auto_trade_config['max_positions'])
                st.success("Auto-trade cycle completed.")
                st.rerun()
        with c2:
            st.subheader("Performance")
            st.metric("Total Trades", st.session_state.paper_trading['total_trades'])
            if st.session_state.paper_trading['total_trades'] > 0:
                win_rate = (st.session_state.paper_trading['winning_trades'] / st.session_state.paper_trading['total_trades']) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Realized P&L", f"${st.session_state.paper_trading['realized_pnl']:,.2f}")
            st.metric("Signals Generated", st.session_state.trading_engine['signals_generated'])
            if st.session_state.auto_trade_enabled:
                st.success("🟢 Auto-Trading: ACTIVE (Signals gen every 30s)")
            else:
                st.warning("🟡 Auto-Trading: INACTIVE")
    def execute_auto_trading(self, trade_amount: float, max_positions: int):
        """Execute trades for newly generated signals, avoiding duplicates and respecting max positions"""
        self.pt.update_positions(self.dm)
        confirmed = list(st.session_state.signals.values())
        # Filter out signals for which we already opened positions or already executed recently
        open_symbols = set(self.pt.get_positions().keys())
        # Sort by confidence
        confirmed.sort(key=lambda x: x.confidence, reverse=True)
        current_positions = len(self.pt.get_positions())
        available_slots = max(0, max_positions - current_positions)
        executed = 0
        for sig in confirmed:
            if executed >= available_slots:
                break
            sid = sig.id()
            if sid in st.session_state.executed_signals:
                continue  # already executed recently
            if sig.symbol in open_symbols:
                continue
            price = self.dm.get_current_price(sig.symbol)
            if not price:
                continue
            quantity = self._calc_quantity_from_risk(sig, trade_amount)
            placed = self.pt.place_trade(sig.symbol, sig.action, quantity, price, sig)
            if placed:
                executed += 1
                exec_time = datetime.now()
                # Mark executed signal to prevent re-execution
                st.session_state.executed_signals[sid] = exec_time
                # Add a concise auto-trade entry in trade_log (place_trade already appends; we add system entry)
                st.session_state.trade_log.append({
                    'timestamp': exec_time,
                    'execution_time': exec_time,
                    'symbol': sig.symbol,
                    'action': f'AUTO_EXEC_{sig.action}',
                    'quantity': quantity,
                    'price': price,
                    'signal': sig.strategy
                })
        st.session_state.trade_log.append({
            'timestamp': datetime.now(),
            'symbol': 'SYSTEM',
            'action': 'AUTO_TRADING_RUN',
            'executed': executed,
            'available_slots': available_slots
        })
    def display_trade_history(self):
        st.markdown("## 📝 Trade History")
        if st.session_state.trade_log:
            df = pd.DataFrame(st.session_state.trade_log).sort_values('timestamp', ascending=False)
            # Ensure execution_time is visible (already present)
            st.dataframe(df.head(400), use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("Download Trade History CSV", data=csv, file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            # Also show a simple tabulation of executions by time
            st.markdown("### Execution Times (recent)")
            exec_df = df[['execution_time','symbol','action','quantity','price']].dropna().head(50)
            st.dataframe(exec_df, use_container_width=True)
        else:
            st.info("No trades yet.")
    def setup_sidebar(self):
        st.sidebar.title("🎯 Controls")
        s = st.sidebar.selectbox("Chart Symbol", options=list(self.dm.symbols.keys()), format_func=lambda x: f"{x} - {self.dm.symbols[x]}")
        st.session_state.chart_symbol = s
        interval = st.sidebar.selectbox("Chart Timeframe", options=CHART_INTERVALS, index=CHART_INTERVALS.index(st.session_state.chart_interval))
        if interval != st.session_state.chart_interval:
            st.session_state.chart_interval = interval
            st.rerun()
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**UI Refresh rate:** {REFRESH_INTERVAL}s")
        st.sidebar.markdown(f"**Signal gen rate:** {SIGNAL_GEN_INTERVAL}s")
        st.sidebar.markdown(f"**Refresh count:** {st.session_state.refresh_count}")
        st.sidebar.markdown(f"**Last refresh:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        if st.sidebar.button("🔄 Reset Portfolio"):
            self.pt.reset()
            st.sidebar.success("Portfolio reset to initial balance.")
            st.rerun()
        if st.sidebar.button("🎯 Generate Signals Now"):
            with st.sidebar.spinner("Generating signals..."):
                self.generate_all_signals()
            st.sidebar.success(f"Generated {len(st.session_state.signals)} signals")
            st.rerun()
    def run_terminal(self):
        st.set_page_config(page_title="Trading Terminal - Auto 15s / Signals 30s", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")
        st.title("🚀 Trading Terminal — Auto-refresh 15s • Signals 30s • Auto-execute")
        st.markdown("*UI refresh every 15s • Signals generated every 30s • Auto-execute new signals (no duplicates)*")
        # Check and execute auto-refresh, signals, and auto-trading
        self.check_auto_refresh()
        self.setup_sidebar()
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Live Charts", "Market", "Signals", "Portfolio", "Auto-Trading", "News", "History"])
        with t1:
            st.info("Live Charts rendering omitted in this concise file; use your original chart function if needed.")
        with t2:
            st.info("Market overview omitted for brevity.")
        with t3:
            self.display_trading_signals()
        with t4:
            self.display_portfolio()
        with t5:
            self.display_auto_trading()
        with t6:
            st.markdown("## 📰 News (placeholder)")
            st.info("News feed integration not yet implemented.")
        with t7:
            self.display_trade_history()

if __name__ == "__main__":
    try:
        term = ProfessionalTradingTerminal()
        term.run_terminal()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Refresh the page to restart.")
