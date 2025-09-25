from pathlib import Path
import pandas as pd
from data_loader import load_dataset
from correlation_analyser import AdvancedCorrelationAnalyser as CorrelationAnalyser
from visualise import (quick_dashboard, plot_selected_cross_correlations, 
                      plot_rolling_correlations, plot_lag_analysis, 
                      plot_correlation_comparison)
import matplotlib.pyplot as plt

def run(
    csv_path: str,
    target: str,
    features=None,
    max_lag: int = 30,
    rolling_window: int = 50,
    top_n_plots: int = 10,
    show_plots: bool = True,
    save_dir: str = None,
    compute_distance: bool = True,
    compute_partial: bool = True,
    compute_cross: bool = True,
    compute_rolling: bool = True,
    sample_size: int = None,
    distance_sample_size: int = 4000,
    verbose: bool = False
):
    X, y = load_dataset(csv_path, target, feature_columns=features)
    analyser = CorrelationAnalyser(
        max_lag=max_lag,
        rolling_window=rolling_window,
        compute_distance=compute_distance,
        compute_partial=compute_partial,
        compute_cross=compute_cross,
        compute_rolling=compute_rolling,
        sample_size=sample_size,
        distance_sample_size=distance_sample_size,
        verbose=verbose
    )
    results = analyser.analyse(X, y)

    print("=== Correlation Analysis Summary ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Target: {target}")
    print(f"Features analyzed: {len(X.columns)}")

    print("\n=== Summary (Top 20 by abs Pearson) ===")
    summary_display = results.summary.copy()
    for col in summary_display.columns:
        if col not in ['feature', 'best_lag'] and summary_display[col].dtype in ['float64', 'float32']:
            summary_display[col] = summary_display[col].round(4)
    
    print(summary_display.reindex(
        summary_display.pearson.abs().sort_values(ascending=False).index
    ).head(20).to_string(index=False))

    if show_plots:
        quick_dashboard(results, top_features=top_n_plots)
        plot_correlation_comparison(results, top_n=top_n_plots)
        
        if compute_cross and results.cross_correlation_curves:
            plot_selected_cross_correlations(results, top_by="pearson", top_n=min(5, top_n_plots))
            plot_lag_analysis(results, top_n=min(8, top_n_plots))
            
        if compute_rolling and results.rolling_correlations:
            plot_rolling_correlations(results.rolling_correlations,
                                      feature_list=results.summary.head(min(5, top_n_plots)).feature.tolist())
        plt.show()

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        results.summary.to_csv(out / "correlation_summary.csv", index=False)
        results.feature_corr_matrix_pearson.to_csv(out / "feature_corr_pearson.csv")
        results.feature_corr_matrix_spearman.to_csv(out / "feature_corr_spearman.csv")
        # Optionally: save figures programmatically (requires modifying plotting funcs to return fig)

if __name__ == "__main__":
    # Direkt hier Parameter definieren (statt CLI Flags)
    # Einfach anpassen:
    run(
        csv_path="./data/matched_data_filtered.csv ",   # z.B. r"C:\data\prices.csv"
        target="close",               # z.B. "close"
        features=["open","high","low","volume"],                        # oder Liste: ["open","high","low","volume"]
        max_lag=30,
        rolling_window=50,
        top_n_plots=15,
        show_plots=True,
        save_dir=None,
        compute_distance=True,
        compute_partial=True,
        compute_cross=True,
        compute_rolling=True,
        sample_size=None,
        distance_sample_size=4000,
        verbose=False
    )


