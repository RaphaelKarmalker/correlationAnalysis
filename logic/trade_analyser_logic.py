import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class TradePerformanceAnalyser:
    """Analyzes trade performance against feature values."""

    def __init__(self, n_bins: int = 10, verbose: bool = False):
        self.n_bins = n_bins
        self.verbose = verbose

    def analyse(
        self,
        merged_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analysiert Trade-PnL gegen bereits gemergte Feature-Spalten.
        Erwartet merged_df mit Spalte 'realized_pnl' und Feature-Spalten.
        Keine Merge-Logik hier (verhindert Redundanz).
        """
        if self.verbose:
            print(f"Starting trade performance analysis on merged dataset with {len(merged_df)} rows and {self.n_bins} bins.")

        if 'realized_pnl' not in merged_df.columns:
            raise ValueError("merged_df muss die Spalte 'realized_pnl' enthalten.")

        # Determine which features to analyze
        if feature_columns is None:
            feature_columns = [
                c for c in merged_df.columns
                if c != 'realized_pnl' and pd.api.types.is_numeric_dtype(merged_df[c])
            ]

        # Coerce selected features to numeric
        for col in feature_columns:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

        # Pre-compute trade durations (in minutes) if both timestamps are available
        duration_series = None
        try:
            idx_name = merged_df.index.name
            if idx_name == 'timestamp' and 'closed_timestamp' in merged_df.columns:
                duration_series = (merged_df['closed_timestamp'] - merged_df.index.to_series()).dt.total_seconds() / 60.0
            elif idx_name == 'closed_timestamp' and 'timestamp' in merged_df.columns:
                duration_series = (merged_df.index.to_series() - merged_df['timestamp']).dt.total_seconds() / 60.0
        except Exception:
            duration_series = None

        results: Dict[str, pd.DataFrame] = {}
        for feature in feature_columns:
            if feature not in merged_df.columns:
                continue

            try:
                s = pd.to_numeric(merged_df[feature], errors='coerce')
                if s.dropna().nunique() < 2:
                    if self.verbose:
                        print(f"Skipping feature '{feature}' due to low variability.")
                    continue

                bin_col = f'{feature}_bin'
                bins_cat = None
                if bin_col in merged_df.columns:
                    bin_codes = merged_df[bin_col]
                else:
                    bins_cat = pd.cut(s, bins=self.n_bins, include_lowest=True, duplicates='drop')
                    bin_codes = bins_cat.cat.codes
                    merged_df[bin_col] = bin_codes

                valid = bin_codes >= 0
                df_feat = merged_df.loc[valid, ['realized_pnl', bin_col, feature]].copy()

                # Attach duration if available
                if duration_series is not None:
                    df_feat['duration_min'] = duration_series.loc[df_feat.index]

                # Base aggregates
                binned = df_feat.groupby(bin_col).agg(
                    mean_pnl=('realized_pnl', 'mean'),
                    total_pnl=('realized_pnl', 'sum'),
                    trade_count=('realized_pnl', 'count'),
                )

                wins = df_feat[df_feat['realized_pnl'] > 0].groupby(bin_col)['realized_pnl'].agg(sum='sum', count='count')
                losses = df_feat[df_feat['realized_pnl'] < 0].groupby(bin_col)['realized_pnl'].agg(sum='sum')

                wins = wins.reindex(binned.index)
                losses = losses.reindex(binned.index)

                binned['win_rate'] = (wins['count'].fillna(0) / binned['trade_count'].replace(0, np.nan) * 100).fillna(0)
                denom = (-losses['sum']).replace(0, np.nan)
                pf = (wins['sum'].fillna(0) / denom).replace([np.inf, -np.inf], np.nan).fillna(0)
                binned['profit_factor'] = pf

                # Extra advanced metrics per bin
                extra_stats: Dict[int, Dict[str, float]] = {}
                for code in binned.index:
                    g = df_feat[df_feat[bin_col] == code].sort_index()
                    pnl = g['realized_pnl'].astype(float)

                    n = len(pnl)
                    mean = pnl.mean()
                    std = pnl.std(ddof=1)
                    med = pnl.median()
                    skew = pnl.skew()
                    kurt = pnl.kurt()

                    pos = pnl[pnl > 0]
                    neg = pnl[pnl < 0]
                    gross_profit = pos.sum() if len(pos) else 0.0
                    gross_loss = neg.sum() if len(neg) else 0.0
                    avg_win = pos.mean() if len(pos) else np.nan
                    avg_loss = neg.mean() if len(neg) else np.nan
                    payoff_ratio = (avg_win / abs(avg_loss)) if (pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0) else np.nan

                    win_rate_dec = (len(pos) / n) if n > 0 else 0.0
                    expectancy = (win_rate_dec * (avg_win if pd.notna(avg_win) else 0.0)) + ((1 - win_rate_dec) * (avg_loss if pd.notna(avg_loss) else 0.0))

                    sharpe = (mean / std) if (std and std > 0) else np.nan
                    semidev = float(np.sqrt(np.mean(np.minimum(pnl.values, 0.0) ** 2))) if n > 0 else np.nan
                    sortino = (mean / semidev) if (semidev and semidev > 0) else np.nan

                    p5 = pnl.quantile(0.05) if n > 0 else np.nan
                    p95 = pnl.quantile(0.95) if n > 0 else np.nan
                    cvar5 = pnl[pnl <= p5].mean() if pd.notna(p5) else np.nan  # Expected shortfall (left tail)

                    # Max drawdown on cumulative PnL sequence within bin
                    eq = pnl.cumsum()
                    dd = eq - eq.cummax()
                    max_drawdown = dd.min() if len(eq) else np.nan  # negative (worst drop)

                    # Max consecutive streaks
                    max_win_streak = 0
                    max_loss_streak = 0
                    cw = cl = 0
                    for v in pnl:
                        if v > 0:
                            cw += 1
                            cl = 0
                        else:
                            cl += 1
                            cw = 0
                        if cw > max_win_streak:
                            max_win_streak = cw
                        if cl > max_loss_streak:
                            max_loss_streak = cl

                    # Durations
                    avg_dur = med_dur = np.nan
                    if 'duration_min' in g.columns:
                        avg_dur = g['duration_min'].mean()
                        med_dur = g['duration_min'].median()

                    extra_stats[int(code)] = {
                        'median_pnl': med,
                        'pnl_std': std,
                        'pnl_skew': skew,
                        'pnl_kurtosis': kurt,
                        'gross_profit': gross_profit,
                        'gross_loss': gross_loss,
                        'avg_win': avg_win if pd.notna(avg_win) else 0.0,
                        'avg_loss': avg_loss if pd.notna(avg_loss) else 0.0,
                        'payoff_ratio': payoff_ratio if pd.notna(payoff_ratio) else 0.0,
                        'expectancy': expectancy,
                        'sharpe': sharpe if pd.notna(sharpe) else 0.0,
                        'sortino': sortino if pd.notna(sortino) else 0.0,
                        'var_5pct': p5 if pd.notna(p5) else 0.0,
                        'cvar_5pct': cvar5 if pd.notna(cvar5) else 0.0,
                        'pnl_p95': p95 if pd.notna(p95) else 0.0,
                        'max_drawdown': float(max_drawdown) if pd.notna(max_drawdown) else 0.0,
                        'max_win_streak': int(max_win_streak),
                        'max_loss_streak': int(max_loss_streak),
                        'avg_duration_min': avg_dur if pd.notna(avg_dur) else 0.0,
                        'median_duration_min': med_dur if pd.notna(med_dur) else 0.0,
                    }

                extra_df = pd.DataFrame.from_dict(extra_stats, orient='index').reindex(binned.index)
                binned = binned.join(extra_df)

                # Build readable bin range labels with 3-dec rounding
                stats = df_feat.groupby(bin_col)[feature].agg(bin_min='min', bin_max='max').reindex(binned.index)

                def _fmt(v: float) -> str:
                    try:
                        return f"{float(v):.3f}".rstrip('0').rstrip('.')
                    except Exception:
                        return str(v)

                binned['bin_label'] = stats.apply(lambda r: f"{_fmt(r['bin_min'])} – {_fmt(r['bin_max'])}", axis=1)

                results[feature] = (
                    binned.reset_index()
                          .rename(columns={bin_col: 'bin'})
                          .sort_values('bin')  # keep natural bin order
                          .reset_index(drop=True)
                )

            except Exception as e:
                if self.verbose:
                    print(f"Could not analyze feature '{feature}': {e}")

        return results
