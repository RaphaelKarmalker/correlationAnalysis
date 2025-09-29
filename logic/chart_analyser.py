from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

try:
    # Reuse robust timestamp parser if available
    from data_loader import _parse_any_timestamp as _to_dt
except Exception:
    def _to_dt(s: pd.Series) -> pd.Series:
        dt = pd.to_datetime(s, errors='coerce')
        if dt.isna().any():
            num = pd.to_numeric(s, errors='coerce')
            dt_ns = pd.to_datetime(num, unit='ns', errors='coerce')
            dt = dt.fillna(dt_ns)
            if dt.isna().any():
                dt_ms = pd.to_datetime(num, unit='ms', errors='coerce')
                dt = dt.fillna(dt_ms)
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
        return dt

class ChartAnalyser:
    """
    Load OHLCV CSV and compute common chart indicators.
    Returns a DataFrame with 'timestamp' and indicator columns.
    """

    def __init__(self, csv_path: str, timestamp_col: str = "timestamp"):
        self.csv_path = csv_path
        self.timestamp_col = timestamp_col
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> "ChartAnalyser":
        df = pd.read_csv(self.csv_path)
        expected = {'open', 'high', 'low', 'close', 'volume', self.timestamp_col}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Ensure numeric
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Parse timestamp to naive datetime and format
        dt = _to_dt(df[self.timestamp_col])
        if dt.isna().all():
            raise ValueError("Could not parse timestamp column to datetime.")
        df[self.timestamp_col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")

        # Keep a datetime index internally for rolling ops
        df = df.set_index(pd.to_datetime(dt, errors='coerce')).sort_index()
        self.df = df
        return self()

    def __call__(self) -> "ChartAnalyser":
        return self

    # ---------- Indicator implementations ----------

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).mean()

    @staticmethod
    def _std(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).std(ddof=0)

    @staticmethod
    def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr

    @classmethod
    def _atr(cls, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        tr = cls._tr(high, low, close)
        atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        return atr

    @staticmethod
    def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        tp = (high + low + close) / 3.0
        cum_vp = (tp * volume).cumsum()
        cum_vol = volume.cumsum().replace(0, np.nan)
        return cum_vp / cum_vol

    @classmethod
    def _bollinger(cls, close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
        ma = cls._sma(close, n)
        sd = cls._std(close, n)
        upper = ma + k * sd
        lower = ma - k * sd
        return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower})

    @classmethod
    def _macd(cls, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        ema_fast = cls._ema(close, fast)
        ema_slow = cls._ema(close, slow)
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = macd - sig
        return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": hist})

    @staticmethod
    def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
        ll = low.rolling(window=k, min_periods=k).min()
        hh = high.rolling(window=k, min_periods=k).max()
        denom = (hh - ll).replace(0, np.nan)
        k_pct = 100 * (close - ll) / denom
        d_pct = k_pct.rolling(window=d, min_periods=d).mean()
        return pd.DataFrame({"stoch_k": k_pct, "stoch_d": d_pct})

    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
        tp = (high + low + close) / 3.0
        rmf = tp * volume
        prev_tp = tp.shift(1)
        pos_flow = rmf.where(tp > prev_tp, 0.0)
        neg_flow = rmf.where(tp < prev_tp, 0.0)
        pos = pos_flow.rolling(n, min_periods=n).sum()
        neg = neg_flow.rolling(n, min_periods=n).sum()
        mr = pos / neg.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mr))
        return mfi

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        sign = np.sign(close.diff()).fillna(0.0)
        return (sign * volume).cumsum()

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(n, min_periods=n).mean()
        mad = (tp - sma_tp).abs().rolling(n, min_periods=n).mean()
        cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
        return cci

    @staticmethod
    def _roc(close: pd.Series, n: int = 12) -> pd.Series:
        return (close / close.shift(n) - 1.0) * 100.0

    # ---------- Public API ----------

    def compute_all(self, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Compute a set of common indicators.
        config can override parameters, e.g.:
        {
          "rsi_periods": [5, 10, 15, 14],
          "sma_periods": [5, 10, 20, 50, 200],
          "ema_periods": [8, 21, 50, 200],
          "atr_period": 14,
          "bb_period": 20, "bb_k": 2.0,
          "macd": {"fast":12, "slow":26, "signal":9},
          "stoch": {"k":14, "d":3},
          "mfi_period": 14,
          "cci_period": 20,
          "roc_period": 12
        }
        """
        if self.df is None:
            self.load()
        df = self.df.copy()

        # Defaults
        cfg = {
            "rsi_periods": [5, 10, 14, 15],
            "sma_periods": [5, 10, 20, 50, 200],
            "ema_periods": [8, 21, 50, 200],
            "atr_period": 14,
            "bb_period": 20, "bb_k": 2.0,
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "stoch": {"k": 14, "d": 3},
            "mfi_period": 14,
            "cci_period": 20,
            "roc_period": 12,
        }
        if config:
            # shallow merge
            for k, v in config.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v

        out = pd.DataFrame(index=df.index)

        # RSI multiple
        for n in cfg["rsi_periods"]:
            out[f"rsi_{n}"] = self._rsi(df['close'], n=int(n))

        # SMA/EMA
        for n in cfg["sma_periods"]:
            out[f"sma_{n}"] = self._sma(df['close'], n=int(n))
        for n in cfg["ema_periods"]:
            out[f"ema_{n}"] = self
