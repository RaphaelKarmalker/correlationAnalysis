import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    from dcor import distance_correlation
    DCOR_AVAILABLE = True
except ImportError:
    DCOR_AVAILABLE = False

@dataclass
class CorrelationResults:
    """Container for all correlation analysis results"""
    summary: pd.DataFrame
    feature_corr_matrix_pearson: pd.DataFrame
    feature_corr_matrix_spearman: pd.DataFrame
    partial_correlations: Optional[pd.DataFrame] = None
    cross_correlation_curves: Optional[Dict[str, pd.Series]] = None
    rolling_correlations: Optional[Dict[str, pd.Series]] = None
    distance_correlations: Optional[pd.DataFrame] = None

    def save_results(self, output_dir: str):
        """Save all correlation results to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        self.summary.to_csv(output_path / "correlation_summary.csv", index=False)
        
        # Save correlation matrices
        self.feature_corr_matrix_pearson.to_csv(output_path / "feature_correlation_pearson.csv")
        self.feature_corr_matrix_spearman.to_csv(output_path / "feature_correlation_spearman.csv")
        
        # Save partial correlations if available
        if self.partial_correlations is not None:
            self.partial_correlations.to_csv(output_path / "partial_correlations.csv")
        
        # Save cross-correlation curves if available
        if self.cross_correlation_curves:
            cross_corr_df = pd.DataFrame(self.cross_correlation_curves)
            cross_corr_df.to_csv(output_path / "cross_correlation_curves.csv")
        
        # Save rolling correlations if available
        if self.rolling_correlations:
            # Convert rolling correlations dict to DataFrame
            rolling_df_data = {}
            max_len = max(len(series) if series is not None else 0 
                         for series in self.rolling_correlations.values())
            
            for feature, series in self.rolling_correlations.items():
                if series is not None:
                    # Pad with NaN to match max length
                    padded_series = series.reindex(range(max_len))
                    rolling_df_data[feature] = padded_series
                else:
                    rolling_df_data[feature] = pd.Series([np.nan] * max_len)
            
            if rolling_df_data:
                rolling_df = pd.DataFrame(rolling_df_data)
                rolling_df.to_csv(output_path / "rolling_correlations.csv")
        
        # Save distance correlations if available
        if self.distance_correlations is not None:
            self.distance_correlations.to_csv(output_path / "distance_correlations.csv")

class AdvancedCorrelationAnalyser:
    """Advanced correlation analyzer with multiple metrics and lag analysis"""
    
    def __init__(
        self,
        max_lag: int = 30,
        rolling_window: int = 50,
        compute_distance: bool = True,
        compute_partial: bool = True,
        compute_cross: bool = True,
        compute_rolling: bool = True,
        sample_size: Optional[int] = None,
        distance_sample_size: int = 4000,
        verbose: bool = False
    ):
        self.max_lag = max_lag
        self.rolling_window = rolling_window
        self.compute_distance = compute_distance and DCOR_AVAILABLE
        self.compute_partial = compute_partial
        self.compute_cross = compute_cross
        self.compute_rolling = compute_rolling
        self.sample_size = sample_size
        self.distance_sample_size = distance_sample_size
        self.verbose = verbose
        
        if compute_distance and not DCOR_AVAILABLE:
            print("Warning: dcor package not available. Distance correlation will be skipped.")

    def analyse(self, X: pd.DataFrame, y: pd.Series) -> CorrelationResults:
        """Main analysis method"""
        if self.verbose:
            print(f"Starting correlation analysis with {len(X.columns)} features and {len(X)} samples")
        
        # Sample data if requested
        if self.sample_size and len(X) > self.sample_size:
            idx = np.random.choice(len(X), self.sample_size, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]
            if self.verbose:
                print(f"Sampled to {len(X)} rows")

        # Basic correlations
        summary_data = self._compute_basic_correlations(X, y)
        
        # Feature correlation matrices
        feature_corr_pearson = X.corr(method='pearson')
        feature_corr_spearman = X.corr(method='spearman')
        
        # Advanced metrics
        partial_corrs = None
        if self.compute_partial:
            partial_corrs = self._compute_partial_correlations(X, y)
            if partial_corrs is not None:
                summary_data['partial_corr'] = partial_corrs
        
        # Distance correlations
        if self.compute_distance:
            dist_corrs = self._compute_distance_correlations(X, y)
            if dist_corrs is not None:
                summary_data['distance_corr'] = dist_corrs
        
        # Mutual information
        mi_scores = self._compute_mutual_information(X, y)
        summary_data['mutual_info'] = mi_scores
        summary_data['mutual_info_norm'] = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
        
        # Cross-correlations (lead/lag analysis)
        cross_corr_curves = None
        if self.compute_cross:
            cross_corr_curves = self._compute_cross_correlations(X, y)
            # Add best lag info to summary
            best_lags = {}
            best_cross_corrs = {}
            for feature, curve in cross_corr_curves.items():
                if curve is not None and not curve.empty:
                    best_idx = curve.abs().idxmax()
                    best_lags[feature] = best_idx
                    best_cross_corrs[feature] = curve.loc[best_idx]
            
            summary_data['best_lag'] = pd.Series(best_lags)
            summary_data['best_cross_corr'] = pd.Series(best_cross_corrs)
        
        # Rolling correlations
        rolling_corrs = None
        if self.compute_rolling:
            rolling_corrs = self._compute_rolling_correlations(X, y)
        
        # Create summary DataFrame
        summary = pd.DataFrame(summary_data)
        summary.index.name = 'feature'
        summary = summary.reset_index()
        
        return CorrelationResults(
            summary=summary,
            feature_corr_matrix_pearson=feature_corr_pearson,
            feature_corr_matrix_spearman=feature_corr_spearman,
            partial_correlations=partial_corrs,
            cross_correlation_curves=cross_corr_curves,
            rolling_correlations=rolling_corrs
        )
    
    def _compute_basic_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.Series]:
        """Compute Pearson and Spearman correlations"""
        pearson_corrs = X.corrwith(y, method='pearson')
        spearman_corrs = X.corrwith(y, method='spearman')
        
        return {
            'pearson': pearson_corrs,
            'spearman': spearman_corrs
        }
    
    def _compute_partial_correlations(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.Series]:
        """Compute partial correlations"""
        try:
            from sklearn.linear_model import LinearRegression
            
            partial_corrs = {}
            X_scaled = StandardScaler().fit_transform(X)
            
            for i, feature in enumerate(X.columns):
                # Remove current feature
                X_others = np.delete(X_scaled, i, axis=1)
                
                if X_others.shape[1] == 0:  # Only one feature
                    partial_corrs[feature] = pearsonr(X.iloc[:, i], y)[0]
                    continue
                
                # Regress y on other features
                reg_y = LinearRegression().fit(X_others, y)
                y_residual = y - reg_y.predict(X_others)
                
                # Regress current feature on other features
                reg_x = LinearRegression().fit(X_others, X_scaled[:, i])
                x_residual = X_scaled[:, i] - reg_x.predict(X_others)
                
                # Correlation of residuals
                if np.std(x_residual) > 1e-10 and np.std(y_residual) > 1e-10:
                    partial_corr = pearsonr(x_residual, y_residual)[0]
                else:
                    partial_corr = 0.0
                
                partial_corrs[feature] = partial_corr
            
            return pd.Series(partial_corrs)
        
        except Exception as e:
            if self.verbose:
                print(f"Partial correlation computation failed: {e}")
            return None
    
    def _compute_distance_correlations(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.Series]:
        """Compute distance correlations"""
        if not DCOR_AVAILABLE:
            return None
        
        try:
            # Sample for performance if needed
            if len(X) > self.distance_sample_size:
                idx = np.random.choice(len(X), self.distance_sample_size, replace=False)
                X_sample = X.iloc[idx]
                y_sample = y.iloc[idx]
            else:
                X_sample = X
                y_sample = y
            
            dist_corrs = {}
            for feature in X_sample.columns:
                try:
                    # Remove NaN values
                    mask = ~(np.isnan(X_sample[feature]) | np.isnan(y_sample))
                    if mask.sum() < 10:  # Need minimum samples
                        dist_corrs[feature] = 0.0
                        continue
                    
                    x_clean = X_sample[feature][mask]
                    y_clean = y_sample[mask]
                    
                    dist_corr = distance_correlation(x_clean, y_clean)
                    dist_corrs[feature] = dist_corr
                    
                except Exception:
                    dist_corrs[feature] = 0.0
            
            return pd.Series(dist_corrs)
        
        except Exception as e:
            if self.verbose:
                print(f"Distance correlation computation failed: {e}")
            return None
    
    def _compute_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute mutual information scores"""
        try:
            # Handle NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                return pd.Series(0.0, index=X.columns)
            
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            return pd.Series(mi_scores, index=X.columns)
        
        except Exception as e:
            if self.verbose:
                print(f"Mutual information computation failed: {e}")
            return pd.Series(0.0, index=X.columns)
    
    def _compute_cross_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.Series]:
        """Compute cross-correlations for lead/lag analysis"""
        cross_corr_curves = {}
        
        for feature in X.columns:
            try:
                # Remove NaN values
                df_temp = pd.DataFrame({'x': X[feature], 'y': y}).dropna()
                if len(df_temp) < 2 * self.max_lag:
                    cross_corr_curves[feature] = None
                    continue
                
                x_series = df_temp['x']
                y_series = df_temp['y']
                
                correlations = []
                lags = range(-self.max_lag, self.max_lag + 1)
                
                for lag in lags:
                    if lag == 0:
                        corr = pearsonr(x_series, y_series)[0]
                    elif lag > 0:
                        # Feature leads (positive lag)
                        if len(x_series) > lag:
                            corr = pearsonr(x_series[:-lag], y_series[lag:])[0]
                        else:
                            corr = 0.0
                    else:
                        # Target leads (negative lag)
                        if len(y_series) > abs(lag):
                            corr = pearsonr(x_series[abs(lag):], y_series[:lag])[0]
                        else:
                            corr = 0.0
                    
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                
                cross_corr_curves[feature] = pd.Series(correlations, index=lags)
                
            except Exception as e:
                if self.verbose:
                    print(f"Cross-correlation failed for {feature}: {e}")
                cross_corr_curves[feature] = None
        
        return cross_corr_curves
    
    def _compute_rolling_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.Series]:
        """Compute rolling correlations"""
        rolling_corrs = {}
        
        if len(X) < self.rolling_window:
            return rolling_corrs
        
        for feature in X.columns:
            try:
                # Combine and remove NaN
                df_temp = pd.DataFrame({'x': X[feature], 'y': y}).dropna()
                if len(df_temp) < self.rolling_window:
                    rolling_corrs[feature] = None
                    continue
                
                rolling_corr = df_temp['x'].rolling(window=self.rolling_window).corr(df_temp['y'])
                rolling_corrs[feature] = rolling_corr.dropna()
                
            except Exception as e:
                if self.verbose:
                    print(f"Rolling correlation failed for {feature}: {e}")
                rolling_corrs[feature] = None
        
        return rolling_corrs
