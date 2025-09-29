from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any
import numpy as np

from data_loader import load_trades, load_features, merge_trade_features
from trade_analyser_logic import TradePerformanceAnalyser
from converter import apply_feature_transforms
from vis import (
    plot_trade_performance_by_feature,
    plot_trade_risk_metrics_by_feature,
    plot_trade_expectancy_payoff_by_feature,
    plot_trade_streaks_drawdown_by_feature,
    plot_trade_duration_by_feature,
    plot_trade_gross_profit_loss_by_feature,
    plot_trade_long_short_by_feature,
)
import matplotlib.pyplot as plt

def run_trade_analysis(
    feature_csv_paths: List[str],
    features: List[str],
    trades_csv_path: str,
    num_bins: int = 10,
    group_by: str = "size",  # "size" => equal-width, "amount" => equal-count
    save_dir: str = "./data/output",
    verbose: bool = False,
    merge_tolerance: Optional[str] = None,
    feature_transforms: Optional[Dict[str, Any]] = None
):

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Trades
    trades_df = load_trades(trades_csv_path)

    # 2) Features
    feature_df = load_features(feature_csv_paths, features)

    # 3) Merge (forward-only: gleicher oder nächstfolgender Feature-Zeitpunkt)
    merged_dataset = merge_trade_features(trades_df, feature_df, tolerance=merge_tolerance)

    # 3a) Optional feature transforms (e.g., zscore, rolling_zscore, robust_zscore)
    features_to_analyse = list(features)
    plot_name_map: Dict[str, str] = {}
    if feature_transforms:
        if verbose:
            print(f"Applying feature transformations: {feature_transforms}")
        pre_cols = set(merged_dataset.columns)
        merged_dataset = apply_feature_transforms(
            merged_dataset,
            feature_transforms,
            inplace=True,
            default_output='replace'
        )
        added_cols = [c for c in merged_dataset.columns if c not in pre_cols]
        if added_cols:
            features_to_analyse.extend(added_cols)

        # Build pretty display names (so you see zscore/rolling in titles)
        for feat, cfg in feature_transforms.items():
            mode = cfg if isinstance(cfg, str) else cfg.get("mode")
            output = 'replace' if isinstance(cfg, str) else cfg.get('output', 'replace')
            # For replace: keep same column, but annotate in title
            if output == 'replace':
                pretty = mode
                if isinstance(cfg, dict) and mode == 'rolling_zscore':
                    win = int(cfg.get('window', 20))
                    pretty = f"rolling_zscore w={win}"
                plot_name_map[feat] = f"{feat} ({pretty})"
            else:
                # For suffix: infer actual created column name to map to a pretty title
                if isinstance(cfg, dict):
                    suffix = cfg.get('suffix')
                    if suffix:
                        col_name = suffix
                    else:
                        if mode == 'rolling_zscore':
                            win = int(cfg.get('window', 20))
                            col_name = f"{feat}_rz{win}"
                        else:
                            col_name = f"{feat}_{mode}"
                    pretty = mode
                    if mode == 'rolling_zscore':
                        pretty = f"rolling_zscore w={int(cfg.get('window', 20))}"
                    plot_name_map[col_name] = f"{feat} ({pretty})"

    # Pre-bin features per requested strategy
    def _assign_bins(df: pd.DataFrame, feat: str) -> None:
        if feat not in df.columns:
            return
        s = pd.to_numeric(df[feat], errors='coerce')
        if s.dropna().nunique() < 2:
            df[f"{feat}_bin"] = -1
            return
        if group_by.lower() == "amount":
            # Equal-count bins via qcut, fallback to rank-based chunking
            q = max(1, min(num_bins, int(s.dropna().nunique())))
            try:
                df[f"{feat}_bin"] = pd.qcut(s, q=q, labels=False, duplicates='drop')
            except Exception:
                order = s.sort_values().index
                chunks = np.array_split(order, q)
                codes = pd.Series(index=df.index, dtype="float")
                for i, idx in enumerate(chunks):
                    codes.loc[idx] = i
                df[f"{feat}_bin"] = codes.astype("Int64")
        else:
            # Equal-width bins over [min, max]
            bins_cat = pd.cut(s, bins=num_bins, include_lowest=True, duplicates='drop')
            df[f"{feat}_bin"] = bins_cat.cat.codes

    for feat in features_to_analyse:
        _assign_bins(merged_dataset, feat)

    print("\n=== Starting Trade Performance Analysis ===")
    if verbose:
        print(f"Trades: {len(trades_df)} | Feature rows: {len(feature_df)} | Merged: {len(merged_dataset)}")
        print(f"Binning: group_by='{group_by}', num_bins={num_bins}")
        if feature_transforms:
            print(f"Analysing columns: {features_to_analyse}")

    # Save merged per-trade dataset (with timestamp column) into cross-feature folder
    merged_path_dir = out / "all_features"
    merged_path_dir.mkdir(parents=True, exist_ok=True)
    merged_path = merged_path_dir / "trades_features_merged.csv"
    try:
        merged_dataset.reset_index().to_csv(merged_path, index=False)
        print(f"Merged dataset saved to: {merged_path.resolve()}")
    except Exception as e:
        print(f"Could not save merged dataset: {e}")

    # 4) Analyse
    trade_analyser = TradePerformanceAnalyser(n_bins=num_bins, verbose=verbose)
    trade_analysis_results = trade_analyser.analyse(merged_dataset, feature_columns=features_to_analyse)

    total_saved = 0
    for feature, analysis_df in trade_analysis_results.items():
        print(f"\n--- Performance by {feature} ---")
        print(analysis_df.to_string(index=False))
        display_name = plot_name_map.get(feature, feature)

        # Per-feature output directory
        feature_dirname = "".join(c for c in feature if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        feature_out = out / feature_dirname
        feature_out.mkdir(parents=True, exist_ok=True)

        # Core performance plot
        fig = plot_trade_performance_by_feature(analysis_df, display_name)
        figs = [fig] if fig else []

        # Additional plots
        for f in (
            plot_trade_risk_metrics_by_feature(analysis_df, display_name),
            plot_trade_expectancy_payoff_by_feature(analysis_df, display_name),
            plot_trade_streaks_drawdown_by_feature(analysis_df, display_name),
            plot_trade_duration_by_feature(analysis_df, display_name),
            plot_trade_gross_profit_loss_by_feature(analysis_df, display_name),
            plot_trade_long_short_by_feature(analysis_df, display_name),
        ):
            if f:
                figs.append(f)

        # Save all figures for this feature into its folder
        for fig in figs:
            if fig is not None:
                title = fig._suptitle.get_text() if fig._suptitle else ''
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
                fname = f"{safe_title}.png" if safe_title else f"{feature_dirname}.png"
                try:
                    fig.savefig(feature_out / fname, dpi=150, bbox_inches='tight')
                    total_saved += 1
                except Exception as e:
                    print(f"Could not save figure '{fname}': {e}")
                finally:
                    plt.close(fig)

    print(f"\nSaved {total_saved} plots grouped by feature under '{out.resolve()}'.")
    print("Trade performance analysis complete.")

if __name__ == "__main__":
    run_trade_analysis(
        feature_csv_paths=[
            "./data/fear_greed_clean.csv",
        ],
        features=["fear_greed", "rsi", "alt_coin_season_index"],
        trades_csv_path="./data/trades.csv",
        num_bins=10,
        group_by="amount",  # "size" for equal-width, "amount" for equal-count
        save_dir="./data/output/trade_performance",
        verbose=True,
        merge_tolerance=None,  # e.g., '1h' to restrict forward match distance
        feature_transforms={
    "rsi": "zscore",  # replace with standard z-score
    "fear_greed": {"mode": "rolling_zscore", "window": 10, "output": "suffix"},  # adds 'fear_greed_rz10'
    "alt_coin_season_index": "robust_zscore"  # replace with robust z-score
}
    )