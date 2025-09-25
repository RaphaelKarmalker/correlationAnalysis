import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from correlation_analyser import CorrelationResults

def plot_feature_target_bar(summary: pd.DataFrame, metric: str = "pearson", top: Optional[int] = None):
    df = summary.copy().sort_values(metric, key=lambda s: s.abs(), ascending=False)
    if top:
        df = df.head(top)
    plt.figure(figsize=(10, 0.5 * len(df)))
    sns.barplot(x=metric, y="feature", data=df, palette="vlag")
    plt.title(f"{metric.capitalize()} correlation with target")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()

def plot_heatmap(matrix: pd.DataFrame, title: str):
    plt.figure(figsize=(min(0.6 * len(matrix.columns) + 4, 22),
                        min(0.6 * len(matrix.columns) + 4, 22)))
    sns.heatmap(matrix, cmap="coolwarm", center=0, annot=False, linewidths=.2)
    plt.title(title)
    plt.tight_layout()

def plot_cross_correlation(curve: pd.Series, feature: str):
    plt.figure(figsize=(8, 4))
    curve.plot()
    plt.axhline(0, color="black", linewidth=0.8)
    best_lag = curve.abs().idxmax()
    best_val = curve.loc[best_lag]
    plt.scatter([best_lag], [best_val], color="red")
    plt.title(f"Cross-Correlation (lead/lag) {feature} (best lag={best_lag}, r={best_val:.3f})")
    plt.xlabel("Lag (negative=feature leads)")
    plt.ylabel("Correlation")
    plt.tight_layout()

def plot_rolling_correlations(rolling_dict, feature_list=None):
    if not rolling_dict:
        return
    if feature_list is None:
        feature_list = list(rolling_dict.keys())
    for f in feature_list:
        series = rolling_dict[f]
        if series is None or series.empty:
            continue
        plt.figure(figsize=(9, 3))
        series.plot()
        plt.title(f"Rolling correlation: {f}")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()

def quick_dashboard(results: CorrelationResults, top_features: int = 15):
    """Create a comprehensive dashboard of correlation results"""
    plot_feature_target_bar(results.summary, "pearson", top=top_features)
    if "spearman" in results.summary.columns:
        plot_feature_target_bar(results.summary, "spearman", top=top_features)
    if "distance_corr" in results.summary.columns:
        plot_feature_target_bar(results.summary, "distance_corr", top=top_features)
    if "mutual_info_norm" in results.summary.columns:
        plot_feature_target_bar(results.summary, "mutual_info_norm", top=top_features)
    if "partial_corr" in results.summary.columns:
        plot_feature_target_bar(results.summary, "partial_corr", top=top_features)
    plot_heatmap(results.feature_corr_matrix_pearson, "Feature Pearson Correlation Matrix")
    plot_heatmap(results.feature_corr_matrix_spearman, "Feature Spearman Correlation Matrix")

def plot_selected_cross_correlations(results: CorrelationResults, features=None, top_by="pearson", top_n=5):
    if not results.cross_correlation_curves:
        return
    if features is None:
        sel = results.summary.sort_values(top_by, key=lambda s: s.abs(), ascending=False).head(top_n)
        features = sel.feature.tolist()
    for f in features:
        curve = results.cross_correlation_curves.get(f)
        if curve is not None and not curve.empty:
            plot_cross_correlation(curve, f)

def plot_lag_analysis(results: CorrelationResults, top_n: int = 5):
    if "best_lag" not in results.summary.columns:
        return
    top_features = results.summary.nlargest(top_n, 'pearson')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x="best_lag", y="feature", data=top_features, palette="viridis")
    plt.title("Best Lag by Feature (Top by Pearson)")
    plt.xlabel("Lag (neg = feature leads)")
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=top_features, x="best_lag", y="best_cross_corr",
                    size="pearson", sizes=(50, 200), alpha=0.7)
    plt.title("Lag vs Cross-Corr Strength")
    plt.xlabel("Best Lag")
    plt.ylabel("Cross-Corr @ Best Lag")
    plt.tight_layout()

def plot_correlation_comparison(results: CorrelationResults, top_n: int = 15):
    metrics = ["pearson", "spearman"]
    for m in ["distance_corr", "mutual_info_norm", "partial_corr"]:
        if m in results.summary.columns:
            metrics.append(m)
    top_features = results.summary.nlargest(top_n, 'pearson')
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        if metric in top_features.columns:
            sns.barplot(x=metric, y="feature", data=top_features, palette="viridis", ax=ax)
            ax.set_title(f"{metric.replace('_',' ').title()} Correlation")
            ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()


