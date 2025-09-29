import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from correlation_analyser import CorrelationResults
import numpy as np

def plot_feature_target_bar(summary: pd.DataFrame, metric: str = "pearson", top: Optional[int] = None):
    df = summary.copy().sort_values(metric, key=lambda s: s.abs(), ascending=False)
    if top:
        df = df.head(top)
    plt.figure(figsize=(10, 0.5 * len(df)))
    sns.barplot(x=metric, y="feature", data=df, palette="vlag")
    plt.title(f"{metric.capitalize()} correlation with target")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    return plt.gcf()

def plot_heatmap(matrix: pd.DataFrame, title: str):
    plt.figure(figsize=(min(0.6 * len(matrix.columns) + 4, 22),
                        min(0.6 * len(matrix.columns) + 4, 22)))
    sns.heatmap(matrix, cmap="coolwarm", center=0, annot=False, linewidths=.2)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

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
    return plt.gcf()

def plot_rolling_correlations(rolling_dict, feature_list=None):
    if not rolling_dict:
        return []
    if feature_list is None:
        feature_list = list(rolling_dict.keys())
    figures = []
    for f in feature_list:
        series = rolling_dict[f]
        if series is None or series.empty:
            continue
        plt.figure(figsize=(9, 3))
        series.plot()
        plt.title(f"Rolling correlation: {f}")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        figures.append(plt.gcf())
    return figures

def quick_dashboard(results: CorrelationResults, top_features: int = 15):
    """Create a comprehensive dashboard of correlation results"""
    figures = []
    figures.append(plot_feature_target_bar(results.summary, "pearson", top=top_features))
    if "spearman" in results.summary.columns:
        figures.append(plot_feature_target_bar(results.summary, "spearman", top=top_features))
    if "distance_corr" in results.summary.columns:
        figures.append(plot_feature_target_bar(results.summary, "distance_corr", top=top_features))
    if "mutual_info_norm" in results.summary.columns:
        figures.append(plot_feature_target_bar(results.summary, "mutual_info_norm", top=top_features))
    if "partial_corr" in results.summary.columns:
        figures.append(plot_feature_target_bar(results.summary, "partial_corr", top=top_features))
    figures.append(plot_heatmap(results.feature_corr_matrix_pearson, "Feature Pearson Correlation Matrix"))
    figures.append(plot_heatmap(results.feature_corr_matrix_spearman, "Feature Spearman Correlation Matrix"))
    return figures

def plot_selected_cross_correlations(results: CorrelationResults, features=None, top_by="pearson", top_n=5):
    if not results.cross_correlation_curves:
        return []
    if features is None:
        sel = results.summary.sort_values(top_by, key=lambda s: s.abs(), ascending=False).head(top_n)
        features = sel.feature.tolist()
    figures = []
    for f in features:
        curve = results.cross_correlation_curves.get(f)
        if curve is not None and not curve.empty:
            figures.append(plot_cross_correlation(curve, f))
    return figures

def plot_lag_analysis(results: CorrelationResults, top_n: int = 5):
    if "best_lag" not in results.summary.columns:
        return None
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
    return plt.gcf()

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
    return fig

def plot_trade_performance_by_feature(analysis_df: pd.DataFrame, feature_name: str):
    """
    Creates a multi-metric plot for trade performance vs. a binned feature.
    Uses bin range labels on the x-axis for readability.
    """
    if analysis_df.empty:
        return None

    # Determine labels and plotting positions
    df = analysis_df.copy()
    df = df.sort_values('bin' if 'bin' in df.columns else df.index).reset_index(drop=True)
    labels = df['bin_label'].tolist() if 'bin_label' in df.columns else df.get('bin_mid', df.get(feature_name, df.index)).tolist()
    df['pos'] = np.arange(len(df))

    # Adaptive figure width to reduce label overlap
    fig_width = max(10, min(24, 0.9 * len(df) + 8))
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 14), sharex=True)
    fig.suptitle(f'Trade Performance vs. {feature_name.upper()} Bins', fontsize=14)

    # Plot 1: Mean PnL (bar) + Total PnL (line) using positions
    ax1 = axes[0]
    sns.barplot(x='pos', y='mean_pnl', data=df, ax=ax1, color='skyblue', label='Mean PnL')
    ax1.set_ylabel('Mean PnL', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax1b = ax1.twinx()
    ax1b.plot(df['pos'], df['total_pnl'], color='green', marker='o', label='Total PnL')
    ax1b.set_ylabel('Total PnL', color='green')
    ax1b.tick_params(axis='y', labelcolor='green')
    ax1.set_title('PnL Analysis')

    # Plot 2: Win Rate (bar) + Profit Factor (line)
    ax2 = axes[1]
    sns.barplot(x='pos', y='win_rate', data=df, ax=ax2, color='coral', label='Win Rate (%)')
    ax2.set_ylabel('Win Rate (%)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    ax2b = ax2.twinx()
    ax2b.plot(df['pos'], df['profit_factor'], color='purple', marker='x', label='Profit Factor')
    ax2b.set_ylabel('Profit Factor', color='purple')
    ax2b.tick_params(axis='y', labelcolor='purple')
    ax2.set_title('Win Rate & Profit Factor')

    # Plot 3: Trade Count
    ax3 = axes[2]
    sns.barplot(x='pos', y='trade_count', data=df, ax=ax3, color='grey')
    ax3.set_title('Trade Distribution')
    ax3.set_ylabel('Number of Trades')

    # Apply x tick labels once at the bottom axis
    ax3.set_xticks(df['pos'])
    ax3.set_xticklabels(labels, rotation=30, ha='right')
    ax3.set_xlabel(f'{feature_name} bin (range)')

    # Improve layout to avoid overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.22)

    return fig

def _prepare_binned_positions(analysis_df: pd.DataFrame, feature_name: str):
    df = analysis_df.copy()
    df = df.sort_values('bin' if 'bin' in df.columns else df.index).reset_index(drop=True)
    labels = df['bin_label'].tolist() if 'bin_label' in df.columns else df.get('bin_mid', df.get(feature_name, df.index)).tolist()
    df['pos'] = np.arange(len(df))
    return df, labels

def plot_trade_risk_metrics_by_feature(analysis_df: pd.DataFrame, feature_name: str):
    needed = ['median_pnl', 'pnl_std', 'var_5pct', 'cvar_5pct', 'sharpe', 'sortino']
    if any(col not in analysis_df.columns for col in needed):
        return None
    df, labels = _prepare_binned_positions(analysis_df, feature_name)

    fig_width = max(10, min(24, 0.9 * len(df) + 8))
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 14), sharex=True)
    fig.suptitle(f'Trade Risk Metrics vs. {feature_name.upper()} Bins', fontsize=14)

    # 1) Median PnL (bar) + StdDev (line)
    ax1 = axes[0]
    sns.barplot(x='pos', y='median_pnl', data=df, ax=ax1, color='steelblue', label='Median PnL')
    ax1.set_ylabel('Median PnL', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1b = ax1.twinx()
    ax1b.plot(df['pos'], df['pnl_std'], color='orange', marker='o', label='PnL StdDev')
    ax1b.set_ylabel('PnL StdDev', color='orange')
    ax1b.tick_params(axis='y', labelcolor='orange')
    ax1.set_title('Median PnL & Volatility')

    # 2) VaR/CVaR (bars)
    ax2 = axes[1]
    width = 0.4
    ax2.bar(df['pos'] - width/2, df['var_5pct'], width=width, color='firebrick', label='VaR 5%')
    ax2.bar(df['pos'] + width/2, df['cvar_5pct'], width=width, color='darkred', alpha=0.7, label='CVaR 5%')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Risk (negative)')
    ax2.set_title('Tail Risk: VaR and CVaR (5%)')
    ax2.legend(loc='best')

    # 3) Sharpe / Sortino (lines)
    ax3 = axes[2]
    ax3.plot(df['pos'], df['sharpe'], color='purple', marker='o', label='Sharpe')
    ax3.plot(df['pos'], df['sortino'], color='teal', marker='x', label='Sortino')
    ax3.set_ylabel('Ratio')
    ax3.set_title('Risk-Adjusted Returns')
    ax3.legend(loc='best')

    ax3.set_xticks(df['pos'])
    ax3.set_xticklabels(labels, rotation=30, ha='right')
    ax3.set_xlabel(f'{feature_name} bin (range)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.22)
    return fig

def plot_trade_expectancy_payoff_by_feature(analysis_df: pd.DataFrame, feature_name: str):
    needed = ['expectancy', 'payoff_ratio', 'avg_win', 'avg_loss']
    if any(col not in analysis_df.columns for col in needed):
        return None
    df, labels = _prepare_binned_positions(analysis_df, feature_name)

    fig_width = max(10, min(24, 0.9 * len(df) + 8))
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 10), sharex=True)
    fig.suptitle(f'Expectancy & Payoff vs. {feature_name.upper()} Bins', fontsize=14)

    # 1) Expectancy (bar) + Payoff Ratio (line)
    ax1 = axes[0]
    sns.barplot(x='pos', y='expectancy', data=df, ax=ax1, color='slateblue', label='Expectancy')
    ax1.set_ylabel('Expectancy', color='slateblue')
    ax1.tick_params(axis='y', labelcolor='slateblue')
    ax1b = ax1.twinx()
    ax1b.plot(df['pos'], df['payoff_ratio'], color='darkgreen', marker='o', label='Payoff Ratio')
    ax1b.set_ylabel('Payoff Ratio', color='darkgreen')
    ax1b.tick_params(axis='y', labelcolor='darkgreen')
    ax1.set_title('Expectancy & Payoff')

    # 2) Average Win/Loss bars (side-by-side)
    ax2 = axes[1]
    width = 0.4
    ax2.bar(df['pos'] - width/2, df['avg_win'], width=width, color='seagreen', label='Avg Win')
    ax2.bar(df['pos'] + width/2, df['avg_loss'], width=width, color='indianred', label='Avg Loss')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('PnL')
    ax2.set_title('Average Win vs. Loss')
    ax2.legend(loc='best')

    ax2.set_xticks(df['pos'])
    ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.set_xlabel(f'{feature_name} bin (range)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.22)
    return fig

def plot_trade_streaks_drawdown_by_feature(analysis_df: pd.DataFrame, feature_name: str):
    needed = ['max_win_streak', 'max_loss_streak', 'max_drawdown']
    if any(col not in analysis_df.columns for col in needed):
        return None
    df, labels = _prepare_binned_positions(analysis_df, feature_name)

    fig_width = max(10, min(24, 0.9 * len(df) + 8))
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 10), sharex=True)
    fig.suptitle(f'Streaks & Drawdown vs. {feature_name.upper()} Bins', fontsize=14)

    # 1) Win/Loss streaks (grouped bars)
    ax1 = axes[0]
    width = 0.4
    ax1.bar(df['pos'] - width/2, df['max_win_streak'], width=width, color='forestgreen', label='Max Win Streak')
    ax1.bar(df['pos'] + width/2, df['max_loss_streak'], width=width, color='firebrick', label='Max Loss Streak')
    ax1.set_ylabel('Streak length')
    ax1.set_title('Max Consecutive Wins/Losses')
    ax1.legend(loc='best')

    # 2) Max Drawdown (line)
    ax2 = axes[1]
    ax2.plot(df['pos'], df['max_drawdown'], color='black', marker='o', label='Max Drawdown')
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.set_ylabel('PnL')
    ax2.set_title('Max Drawdown (cumulative within bin)')
    ax2.legend(loc='best')

    ax2.set_xticks(df['pos'])
    ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.set_xlabel(f'{feature_name} bin (range)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.22)
    return fig

def plot_trade_duration_by_feature(analysis_df: pd.DataFrame, feature_name: str):
    needed = ['avg_duration_min', 'median_duration_min']
    if any(col not in analysis_df.columns for col in needed):
        return None
    df, labels = _prepare_binned_positions(analysis_df, feature_name)

    fig_width = max(10, min(24, 0.9 * len(df) + 8))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 5), sharex=True)
    fig.suptitle(f'Trade Duration vs. {feature_name.upper()} Bins', fontsize=14)

    ax.bar(df['pos'], df['avg_duration_min'], color='royalblue', alpha=0.7, label='Avg Duration (min)')
    ax.plot(df['pos'], df['median_duration_min'], color='navy', marker='o', label='Median Duration (min)')
    ax.set_ylabel('Minutes')
    ax.set_title('Trade Duration')
    ax.legend(loc='best')

    ax.set_xticks(df['pos'])
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_xlabel(f'{feature_name} bin (range)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.22)
    return fig

def plot_trade_gross_profit_loss_by_feature(analysis_df: pd.DataFrame, feature_name: str):
    needed = ['gross_profit', 'gross_loss']
    if any(col not in analysis_df.columns for col in needed):
        return None
    df, labels = _prepare_binned_positions(analysis_df, feature_name)

    fig_width = max(10, min(24, 0.9 * len(df) + 8))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 5), sharex=True)
    fig.suptitle(f'Gross Profit/Loss vs. {feature_name.upper()} Bins', fontsize=14)

    width = 0.4
    ax.bar(df['pos'] - width/2, df['gross_profit'], width=width, color='seagreen', label='Gross Profit')
    ax.bar(df['pos'] + width/2, df['gross_loss'], width=width, color='indianred', label='Gross Loss')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('PnL')
    ax.set_title('Gross Profit and Gross Loss')
    ax.legend(loc='best')

    ax.set_xticks(df['pos'])
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_xlabel(f'{feature_name} bin (range)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.22)
    return fig

def plot_trade_long_short_by_feature(analysis_df: pd.DataFrame, feature_name: str):
    """
    Plot long vs short PnL per bin with trade counts.
    Requires columns: long_total_pnl, short_total_pnl, long_trade_count, short_trade_count.
    Returns None if not available.
    """
    needed = ['long_total_pnl', 'short_total_pnl', 'long_trade_count', 'short_trade_count']
    if any(col not in analysis_df.columns for col in needed):
        return None
    if analysis_df[["long_total_pnl", "short_total_pnl"]].isna().all().all():
        return None

    df, labels = _prepare_binned_positions(analysis_df, feature_name)

    fig_width = max(10, min(24, 0.9 * len(df) + 8))
    fig, ax1 = plt.subplots(1, 1, figsize=(fig_width, 6), sharex=True)
    fig.suptitle(f'Long vs Short PnL by {feature_name.upper()} Bins', fontsize=14)

    width = 0.4
    ax1.bar(df['pos'] - width/2, df['long_total_pnl'], width=width, color='seagreen', label='Long Total PnL')
    ax1.bar(df['pos'] + width/2, df['short_total_pnl'], width=width, color='indianred', label='Short Total PnL')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_ylabel('Total PnL')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(df['pos'], df['long_trade_count'], color='darkgreen', marker='o', label='Long Trades')
    ax2.plot(df['pos'], df['short_trade_count'], color='darkred', marker='x', label='Short Trades')
    ax2.set_ylabel('Trade Count')

    # Build combined legend
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels1 + labels2, loc='upper right')

    ax1.set_xticks(df['pos'])
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.set_xlabel(f'{feature_name} bin (range)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.22)
    return fig

def plot_indicator_with_ohlcv(ohlcv: pd.DataFrame, indicator: pd.Series, indicator_name: str, title: Optional[str] = None, last_n: Optional[int] = 300):
    """
    Draw simple OHLC candlesticks with an indicator overlaid (second y-axis).
    - ohlcv: DataFrame with index=datetime and columns ['open','high','low','close','volume']
    - indicator: datetime-indexed Series to align to ohlcv
    """
    needed_cols = {'open','high','low','close','volume'}
    if ohlcv is None or not needed_cols.issubset(set(ohlcv.columns)):
        return None
    if indicator is None or indicator.empty:
        return None

    df = ohlcv.copy().sort_index()
    ind = indicator.reindex(df.index).ffill()

    if last_n and len(df) > last_n:
        df = df.iloc[-last_n:]
        ind = ind.iloc[-last_n:]

    # integer x positions with datetime labels
    df = df.copy()
    df['pos'] = np.arange(len(df))
    pos = df['pos'].values
    dates = df.index

    fig, ax = plt.subplots(1, 1, figsize=(max(10, min(24, 0.04 * len(df) + 10)), 6))
    fig.suptitle(title or f'OHLCV + {indicator_name}', fontsize=14)

    # Candlestick (simple)
    width = 0.6
    up = df['close'] >= df['open']
    down = ~up
    # Wicks
    ax.vlines(pos, df['low'], df['high'], color='black', linewidth=1)
    # Bodies
    ax.bar(pos[up], (df.loc[up, 'close'] - df.loc[up, 'open']), width=width, bottom=df.loc[up, 'open'], color='#2ca02c', edgecolor='black')
    ax.bar(pos[down], (df.loc[down, 'close'] - df.loc[down, 'open']), width=width, bottom=df.loc[down, 'open'], color='#d62728', edgecolor='black')

    ax.set_ylabel('Price')

    ax2 = ax.twinx()
    ax2.plot(pos, ind.values, color='royalblue', linewidth=1.3, label=indicator_name)
    ax2.set_ylabel(indicator_name, color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')

    ax.set_xticks(pos[::max(1, len(pos)//10)])
    ax.set_xticklabels([d.strftime("%Y-%m-%d %H:%M") for d in dates[::max(1, len(pos)//10)]], rotation=30, ha='right')
    ax.set_xlabel('Time')

    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


