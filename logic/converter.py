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

def rolling_zscore(s: pd.Series, window: int = 20, min_periods: Optional[int] = None) -> pd.Series:
	if min_periods is None:
		min_periods = max(5, window // 2)
	roll_mean = s.rolling(window=window, min_periods=min_periods).mean()
	roll_std = s.rolling(window=window, min_periods=min_periods).std(ddof=1)
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

_TRANSFORMS = {
	'zscore': lambda s, **kw: zscore(s),
	'robust_zscore': lambda s, **kw: robust_zscore(s),
	'rolling_zscore': lambda s, **kw: rolling_zscore(s, window=int(kw.get('window', 20)), min_periods=kw.get('min_periods')),
	'minmax': lambda s, **kw: minmax(s),
	'normalize': lambda s, **kw: normalize(s),
	'rank_pct': lambda s, **kw: rank_pct(s),
	'log1p_signed': lambda s, **kw: log1p_signed(s),
}


def apply_feature_transforms(
    df: pd.DataFrame,
    transforms: Optional[Dict[str, Any]],
    inplace: bool = True,
    default_output: str = 'replace'
) -> pd.DataFrame:
    """
    Applies per-feature transforms on df. transforms can be:
      - {'rsi': 'zscore'}
      - {'rsi': {'mode': 'rolling_zscore', 'window': 10}}
      - {'rsi': {'mode': 'rolling_zscore', 'window': 10, 'output': 'suffix', 'suffix': 'rsi_rz10'}}
    Supported modes: zscore, robust_zscore, rolling_zscore, minmax, rank_pct, log1p_signed
    output: 'replace' (default) replaces original column; 'suffix' creates new column.
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
        elif isinstance(cfg, dict):
            mode = cfg.get('mode')
            params = {k: v for k, v in cfg.items() if k not in ('mode', 'output', 'suffix')}
            output = cfg.get('output', default_output)
            suffix = cfg.get('suffix')
        else:
            continue

        if mode not in _TRANSFORMS:
            continue

        transformed = _TRANSFORMS[mode](series, **params)

        if output == 'suffix':
            if suffix:
                col_name = suffix
            else:
                if mode == 'rolling_zscore':
                    win = int(params.get('window', 20))
                    col_name = f"{feat}_rz{win}"
                else:
                    col_name = f"{feat}_{mode}"
            target_df[col_name] = transformed
        else:
            target_df[feat] = transformed

    return target_df
