from pathlib import Path
import pandas as pd
from typing import Optional, List
import numpy as np

from data_loader import load_trades, load_features, merge_trade_features
from trade_analyser_logic import TradePerformanceAnalyser
from vis import plot_trade_performance_by_feature
import matplotlib.pyplot as plt

def run_trade_analysis(
    feature_csv_paths: List[str],
    features: List[str],
    trades_csv_path: str,
    num_bins: int = 10,
    group_by: str = "size",  # "size" => equal-width, "amount" => equal-count
    save_dir: str = "./data/output",
    verbose: bool = False,
    merge_tolerance: Optional[str] = None
):

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Trades
    trades_df = load_trades(trades_csv_path)

    # 2) Features
    feature_df = load_features(feature_csv_paths, features)

    # 3) Merge (forward-only: gleicher oder nächstfolgender Feature-Zeitpunkt)
    merged_dataset = merge_trade_features(trades_df, feature_df, tolerance=merge_tolerance)

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

    for feat in features:
        _assign_bins(merged_dataset, feat)

    print("\n=== Starting Trade Performance Analysis ===")
    if verbose:
        print(f"Trades: {len(trades_df)} | Feature rows: {len(feature_df)} | Merged: {len(merged_dataset)}")
        print(f"Binning: group_by='{group_by}', num_bins={num_bins}")

    # Save merged per-trade dataset (with timestamp column)
    merged_path = out / "trades_features_merged.csv"
    try:
        merged_dataset.reset_index().to_csv(merged_path, index=False)
        print(f"Merged dataset saved to: {merged_path.resolve()}")
    except Exception as e:
        print(f"Could not save merged dataset: {e}")

    # 4) Analyse
    trade_analyser = TradePerformanceAnalyser(n_bins=num_bins, verbose=verbose)
    trade_analysis_results = trade_analyser.analyse(merged_dataset, feature_columns=features)

    all_figures = []
    for feature, analysis_df in trade_analysis_results.items():
        print(f"\n--- Performance by {feature} ---")
        print(analysis_df.to_string(index=False))
        fig = plot_trade_performance_by_feature(analysis_df, feature)
        if fig:
            all_figures.append(fig)

    print(f"\nSaving {len(all_figures)} trade performance plots to '{out.resolve()}'...")
    for fig in all_figures:
        if fig is not None:
            title = fig._suptitle.get_text() if fig._suptitle else ''
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            fname = f"{safe_title}.png"
            try:
                fig.savefig(out / fname, dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"Could not save figure '{fname}': {e}")
            finally:
                plt.close(fig)

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
        merge_tolerance=None  # e.g., '1h' to restrict forward match distance
    )
