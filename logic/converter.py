import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def zscore(s: pd.Series) -> pd.Series:
	mean = s.mean()
	std = s.std(ddof=1)
	return (s - mean) / std if std and std > 0 else pd.Series(np.zeros(len(s), dtype=float), index=s.index)

def robust_zscore(s: pd.Series) -> pd.Series:
	med = s.median()
	mad = (s - med).abs().median()
	return 0.6745 * (s - med) / mad if mad and mad > 0 else pd.Series(np.zeros(len(s), dtype=float), index=s.index)

def rolling_zscore(s: pd.Series, window: int = 20, min_periods: Optional[int] = None, leak_safe: bool = False) -> pd.Series:
	"""
	Rolling z-score. If leak_safe=True, compute stats on s.shift(1) so only past data is used.
	"""
	if min_periods is None:
		min_periods = max(5, window // 2)
	base = s.shift(1) if leak_safe else s
	roll_mean = base.rolling(window=window, min_periods=min_periods).mean()
	roll_std = base.rolling(window=window, min_periods=min_periods).std(ddof=1)
	out = (s - roll_mean) / roll_std
	return out.replace([np.inf, -np.inf], np.nan)

def minmax(s: pd.Series) -> pd.Series:
	vmin, vmax = s.min(), s.max()
	den = vmax - vmin
	return (s - vmin) / den if den and den != 0 else pd.Series(np.zeros(len(s), dtype=float), index=s.index)

def normalize(s: pd.Series) -> pd.Series:
	return (s - s.mean()) / (s.max() - s.min())

def rank_pct(s: pd.Series) -> pd.Series:
	return s.rank(method='average', pct=True)

def log1p_signed(s: pd.Series) -> pd.Series:
	return np.sign(s) * np.log1p(np.abs(s))

# NEW: leakage-safe expanding/EWM/rolling helpers
def expanding_zscore(s: pd.Series, min_periods: int = 20) -> pd.Series:
	"""
	Time-safe z-score using expanding mean/std (uses only history up to t-1).
	"""
	mean = s.shift(1).expanding(min_periods=min_periods).mean()
	std = s.shift(1).expanding(min_periods=min_periods).std(ddof=1)
	out = (s - mean) / std
	return out.replace([np.inf, -np.inf], np.nan)

def ewm_zscore(s: pd.Series, span: int = 50, leak_safe: bool = True) -> pd.Series:
	"""
	EWM-based z-score; with leak_safe=True, EWM stats are computed on s.shift(1).
	"""
	base = s.shift(1) if leak_safe else s
	mu = base.ewm(span=span, adjust=False, min_periods=max(5, span // 2)).mean()
	# EWM variance via E[X^2] - (E[X])^2
	m2 = (base.pow(2)).ewm(span=span, adjust=False, min_periods=max(5, span // 2)).mean()
	var = (m2 - mu.pow(2)).clip(lower=0)
	std = np.sqrt(var)
	out = (s - mu) / std
	return out.replace([np.inf, -np.inf], np.nan)

def rolling_robust_zscore(s: pd.Series, window: int = 50, min_periods: Optional[int] = None, leak_safe: bool = True) -> pd.Series:
	"""
	Robust rolling z-score using rolling median and MAD; stats computed on shifted series if leak_safe.
	"""
	if min_periods is None:
		min_periods = max(10, window // 2)
	base = s.shift(1) if leak_safe else s
	med = base.rolling(window=window, min_periods=min_periods).median()
	mad = (base - med).abs().rolling(window=window, min_periods=min_periods).median()
	z = 0.6745 * (s - med) / mad
	return z.replace([np.inf, -np.inf], np.nan)

def rolling_minmax(s: pd.Series, window: int = 100, min_periods: Optional[int] = None, leak_safe: bool = True) -> pd.Series:
	"""
	Rolling min-max normalization; stats computed on s.shift(1) if leak_safe.
	"""
	if min_periods is None:
		min_periods = max(10, window // 2)
	base = s.shift(1) if leak_safe else s
	vmin = base.rolling(window=window, min_periods=min_periods).min()
	vmax = base.rolling(window=window, min_periods=min_periods).max()
	den = (vmax - vmin).replace(0, np.nan)
	out = (s - vmin) / den
	return out.fillna(0.0).clip(0.0, 1.0)

def expanding_minmax(s: pd.Series, min_periods: int = 20) -> pd.Series:
	"""
	Expanding min-max normalization using only history up to t-1.
	"""
	vmin = s.shift(1).expanding(min_periods=min_periods).min()
	vmax = s.shift(1).expanding(min_periods=min_periods).max()
	den = (vmax - vmin).replace(0, np.nan)
	out = (s - vmin) / den
	return out.fillna(0.0).clip(0.0, 1.0)

def winsorize_rolling(s: pd.Series, window: int = 250, lower: float = 0.01, upper: float = 0.99, leak_safe: bool = True) -> pd.Series:
	"""
	Rolling winsorization by clipping to rolling quantiles; quantiles computed on s.shift(1) if leak_safe.
	"""
	base = s.shift(1) if leak_safe else s
	low = base.rolling(window=window, min_periods=max(10, window // 5)).quantile(lower)
	high = base.rolling(window=window, min_periods=max(10, window // 5)).quantile(upper)
	return s.clip(lower=low, upper=high)

def pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
	"""
	Percentage change based on past value only (standard pct_change is already safe).
	"""
	return s.pct_change(periods=periods)

def diff_n(s: pd.Series, periods: int = 1) -> pd.Series:
	return s.diff(periods=periods)

def log_return(s: pd.Series, periods: int = 1) -> pd.Series:
	"""
	Log return using past value only.
	"""
	return np.log(s / s.shift(periods))

_TRANSFORMS = {
	'zscore': lambda s, **kw: zscore(s),
	'robust_zscore': lambda s, **kw: robust_zscore(s),
	'rolling_zscore': lambda s, **kw: rolling_zscore(s, window=int(kw.get('window', 20)), min_periods=kw.get('min_periods'), leak_safe=bool(kw.get('leak_safe', False))),
	'minmax': lambda s, **kw: minmax(s),
	'normalize': lambda s, **kw: normalize(s),
	'rank_pct': lambda s, **kw: rank_pct(s),
	'log1p_signed': lambda s, **kw: log1p_signed(s),

	# NEW safe modes
	'expanding_zscore': lambda s, **kw: expanding_zscore(s, min_periods=int(kw.get('min_periods', 20))),
	'ewm_zscore': lambda s, **kw: ewm_zscore(s, span=int(kw.get('span', 50)), leak_safe=bool(kw.get('leak_safe', True))),
	'rolling_robust_zscore': lambda s, **kw: rolling_robust_zscore(s, window=int(kw.get('window', 50)), min_periods=kw.get('min_periods'), leak_safe=bool(kw.get('leak_safe', True))),
	'rolling_minmax': lambda s, **kw: rolling_minmax(s, window=int(kw.get('window', 100)), min_periods=kw.get('min_periods'), leak_safe=bool(kw.get('leak_safe', True))),
	'expanding_minmax': lambda s, **kw: expanding_minmax(s, min_periods=int(kw.get('min_periods', 20))),
	'winsorize_rolling': lambda s, **kw: winsorize_rolling(s, window=int(kw.get('window', 250)), lower=float(kw.get('lower', 0.01)), upper=float(kw.get('upper', 0.99)), leak_safe=bool(kw.get('leak_safe', True))),
	'pct_change': lambda s, **kw: pct_change_safe(s, periods=int(kw.get('periods', 1))),
	'diff': lambda s, **kw: diff_n(s, periods=int(kw.get('periods', 1))),
	'log_return': lambda s, **kw: log_return(s, periods=int(kw.get('periods', 1))),
}

def apply_feature_transforms(
    df: pd.DataFrame,
    transforms: Optional[Dict[str, Any]],
    inplace: bool = True,
    default_output: str = 'replace',
    default_leak_safe: bool = False  # NEW: switch to favor leakage-safe variants globally
) -> pd.DataFrame:
    """
    Applies per-feature transforms on df. transforms can be:
      - {'rsi': 'zscore'}
      - {'rsi': {'mode': 'rolling_zscore', 'window': 10}}
      - {'rsi': {'mode': 'rolling_zscore', 'window': 10, 'output': 'suffix', 'suffix': 'rsi_rz10'}}
      - {'ema_50': {'mode': 'zscore', 'safe': True}}  # will map to expanding_zscore (no leakage)
    Supported modes: zscore, robust_zscore, rolling_zscore, minmax, rank_pct, log1p_signed
                     and leakage-safe: expanding_zscore, ewm_zscore, rolling_robust_zscore, rolling_minmax, expanding_minmax, winsorize_rolling
    output: 'replace' (default) replaces original column; 'suffix' creates new column.
    The 'safe' or 'no_leak' flag (or default_leak_safe=True) switches to non-leaking variants.
    """
    if not transforms:
        return df

    target_df = df if inplace else df.copy()

    for feat, cfg in transforms.items():
        if feat not in target_df.columns:
            continue

        series = pd.to_numeric(target_df[feat], errors='coerce')

        # Parse config
        if isinstance(cfg, str):
            mode = cfg
            params: Dict[str, Any] = {}
            output = default_output
            suffix = None
            safe_flag = default_leak_safe
        elif isinstance(cfg, dict):
            mode = cfg.get('mode', cfg.get('transform', None))
            params = {k: v for k, v in cfg.items() if k not in ('mode', 'transform', 'output', 'suffix', 'safe', 'no_leak')}
            output = cfg.get('output', default_output)
            suffix = cfg.get('suffix')
            safe_flag = bool(cfg.get('safe') or cfg.get('no_leak') or default_leak_safe)
        else:
            continue

        if mode is None:
            continue

        # Remap to leakage-safe counterparts if requested
        if safe_flag:
            if mode == 'zscore':
                mode = 'expanding_zscore'
            elif mode == 'robust_zscore':
                # prefer rolling robust with leak-safe stats
                mode = 'rolling_robust_zscore'
                params.setdefault('window', 50)
                params['leak_safe'] = True
            elif mode == 'minmax':
                mode = 'expanding_minmax'
            elif mode == 'rolling_zscore':
                params['leak_safe'] = True
            elif mode == 'rolling_minmax':
                params['leak_safe'] = True
            elif mode == 'ewm_zscore':
                params['leak_safe'] = True

        if mode not in _TRANSFORMS:
            continue

        transformed = _TRANSFORMS[mode](series, **params)
        transformed = transformed.replace([np.inf, -np.inf], np.nan)

        if output == 'suffix':
            if suffix:
                col_name = suffix
            else:
                if mode in ('rolling_zscore', 'rolling_robust_zscore'):
                    win = int(params.get('window', 20))
                    col_name = f"{feat}_rz{win}" if 'robust' not in mode else f"{feat}_rrz{win}"
                elif mode in ('expanding_zscore', 'ewm_zscore'):
                    col_name = f"{feat}_{mode}"
                elif mode in ('rolling_minmax', 'expanding_minmax'):
                    col_name = f"{feat}_{mode}"
                else:
                    col_name = f"{feat}_{mode}"
            target_df[col_name] = transformed
        else:
            target_df[feat] = transformed

    return target_df

# NEW: convenience pre-normalization (leakage-safe) for multiple columns
def pre_normalize_features(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    mode: str = 'expanding_zscore',
    inplace: bool = True,
    params: Optional[Dict[str, Any]] = None,
    exclude: Optional[list] = None
) -> pd.DataFrame:
    """
    Apply leakage-safe normalization to a set of numeric columns.
    - mode: one of ['expanding_zscore','ewm_zscore','rolling_robust_zscore','rolling_minmax','expanding_minmax']
    - params: optional dict with transform-specific params (e.g., {'span':50} for ewm_zscore)
    - exclude: list of columns to skip
    """
    if params is None:
        params = {}
    target_df = df if inplace else df.copy()

    if columns is None:
        columns = [c for c in target_df.columns if pd.api.types.is_numeric_dtype(target_df[c])]
    if exclude:
        columns = [c for c in columns if c not in set(exclude)]

    if mode not in _TRANSFORMS:
        raise ValueError(f"Unknown mode '{mode}' for pre-normalization.")

    for col in columns:
        s = pd.to_numeric(target_df[col], errors='coerce')
        transformed = _TRANSFORMS[mode](s, **params).replace([np.inf, -np.inf], np.nan)
        target_df[col] = transformed

    return target_df
