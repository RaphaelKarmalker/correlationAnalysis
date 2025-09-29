from __future__ import annotations
import pandas as pd
from typing import List, Optional, Tuple

def load_dataset(
    path: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    parse_dates: Optional[List[str]] = None,
    index_col: Optional[str] = None,
    dropna: bool = True,
    fill_method: Optional[str] = "ffill"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Lädt CSV, wählt Feature + optional Target Spalten.
    - target_column=None => nur Features werden geladen, Target ist None.
    - feature_columns=None => alle außer target (falls target gesetzt), sonst alle.
    - dropna True: Wenn target gesetzt ist, entferne Zeilen mit NaN im Target.
                   Ohne target: entferne nur Zeilen, in denen alle Features NaN sind.
    - fill_method: ffill/bfill/None für Features.
    """
    df = pd.read_csv(path, parse_dates=parse_dates, index_col=index_col)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column] if target_column else list(df.columns)

    expected_cols = set(feature_columns) | ({target_column} if target_column else set())
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten im CSV: {missing}")

    target = df[target_column] if target_column else None
    features = df[feature_columns].copy()

    if fill_method in ("ffill", "bfill"):
        features = features.fillna(method=fill_method)

    if dropna:
        if target is not None:
            mask = ~target.isna()
            target = target[mask]
            features = features.loc[mask]
        else:
            # Ohne Target: entferne nur Zeilen, in denen alle Feature-Werte NaN sind
            features = features.dropna(how="all")

    # Align nur wenn Target vorhanden
    if target is not None:
        features, target = features.align(target, join="inner", axis=0)

    return features, target

def _parse_any_timestamp(s: pd.Series) -> pd.Series:
    """Best-effort conversion of timestamp column to naive datetime."""
    dt = pd.to_datetime(s, errors='coerce')
    if dt.isna().any():
        # Try numeric epoch in ns, then ms
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

def load_trades(path: str, analyse: str = "open") -> pd.DataFrame:
    """
    Lädt und bereinigt die Trade-Daten aus einer CSV-Datei.
    - Unterstützt Analyse-Modus: analyse='open' nutzt 'timestamp', analyse='close' nutzt 'closed_timestamp'.
    - Konvertiert gewählten Zeitstempel (Epoch ns/ms oder String) nach datetime.
    - Extrahiert numerischen PnL-Wert aus 'realized_pnl'.
    - Sortiert nach gewähltem Zeitstempel und setzt ihn als Index (Indexname entspricht der Spalte).
    """
    trades = pd.read_csv(path)

    if 'timestamp' not in trades.columns:
        raise ValueError("Spalte 'timestamp' fehlt in Trades-CSV.")
    trades['timestamp'] = _parse_any_timestamp(trades['timestamp'])

    if 'closed_timestamp' in trades.columns:
        trades['closed_timestamp'] = _parse_any_timestamp(trades['closed_timestamp'])

    if 'realized_pnl' not in trades.columns:
        raise ValueError("Spalte 'realized_pnl' fehlt in Trades-CSV.")
    trades['realized_pnl'] = trades['realized_pnl'].astype(str).str.extract(r'([-\d.]+)').astype(float)

    mode = analyse.lower().strip()
    if mode not in ("open", "close"):
        raise ValueError("analyse muss 'open' oder 'close' sein.")

    ts_col = 'timestamp' if mode == 'open' else 'closed_timestamp'
    if ts_col not in trades.columns:
        raise ValueError(f"Spalte '{ts_col}' fehlt für analyse='{analyse}'.")

    # Drop bad rows, sort and index by chosen timestamp column
    trades = trades.dropna(subset=[ts_col, 'realized_pnl'])
    trades = trades.sort_values(ts_col).set_index(ts_col)
    trades.index.name = ts_col

    return trades

def load_features(
    csv_paths: List[str],
    feature_columns: List[str],
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Lädt Feature-Spalten aus mehreren CSVs und baut ein gemeinsames Feature-DataFrame.
    - csv_paths: Liste von CSV-Dateien.
    - feature_columns: gewünschte Feature-Spalten.
    - Falls ein Feature mehrfach vorkommt, wird die erste gefundene Spalte verwendet.
    """
    collected = pd.DataFrame()
    seen = set()

    for path in csv_paths:
        df = pd.read_csv(path)
        if timestamp_col not in df.columns:
            raise ValueError(f"Spalte '{timestamp_col}' fehlt in {path}")
        df[timestamp_col] = _parse_any_timestamp(df[timestamp_col])
        df = df.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()

        for feat in feature_columns:
            if feat in seen:
                continue
            if feat in df.columns:
                # Take the column and outer-join onto collected
                col_df = df[[feat]].copy()
                if collected.empty:
                    collected = col_df
                else:
                    collected = collected.join(col_df, how='outer')
                seen.add(feat)

    # Keep only requested features order
    collected = collected[[c for c in feature_columns if c in collected.columns]]
    collected = collected.sort_index()

    # NEW: Forward-fill to propagate low-frequency data (e.g., daily fear/greed)
    # This ensures that when merging, the last known value is available.
    collected = collected.ffill()

    return collected

def merge_trade_features(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    tolerance: Optional[pd.Timedelta | str] = None  # Keep parameter but ignore it
) -> pd.DataFrame:
    """
    Merge features into trades DataFrame without leakage:
    - For each trade timestamp, find the LAST available feature value (backward looking)
    - This means one feature entry can be used for MULTIPLE trades
    - Example: Fear & Greed from Jan 1st is used for all trades on Jan 1st, 2nd, 3rd... until next update
    - This is leak-safe because we only use past/current feature values
    """
    if not isinstance(trades.index, pd.DatetimeIndex):
        raise TypeError("trades muss einen DatetimeIndex haben.")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features muss einen DatetimeIndex haben.")

    print(f"\n=== Merging Trade Features ===")
    print(f"Trades: {len(trades)} records from {trades.index.min()} to {trades.index.max()}")
    print(f"Features: {len(features)} records from {features.index.min()} to {features.index.max()}")

    trades = trades.sort_index()
    features = features.sort_index()

    # Check time overlap
    trades_start, trades_end = trades.index.min(), trades.index.max()
    features_start, features_end = features.index.min(), features.index.max()
    
    overlap_start = max(trades_start, features_start)
    overlap_end = min(trades_end, features_end)
    
    print(f"Time overlap: {overlap_start} to {overlap_end}")
    
    if overlap_start >= overlap_end:
        print("WARNING: No time overlap between trades and features!")
        return trades

    # Reset indexes for merge_asof
    trades_reset = trades.reset_index()
    features_reset = features.reset_index()
    
    # Get the name of the timestamp column
    ts_col = trades.index.name or 'timestamp'
    
    # Use merge_asof with direction='backward' 
    # This finds the most recent feature value that is <= trade timestamp
    # This is exactly what you want: use the LAST available feature value for each trade
    merged_reset = pd.merge_asof(
        trades_reset.sort_values(ts_col),
        features_reset.sort_values(ts_col),
        left_on=ts_col,
        right_on=ts_col,
        direction='backward',  # Use most recent past feature value
        allow_exact_matches=True
    )
    
    # Set timestamp back as index
    merged_final = merged_reset.set_index(ts_col)
    
    # Count successful matches
    feature_cols = list(features.columns)
    if feature_cols:
        successful_matches = merged_final[feature_cols].notna().any(axis=1).sum()
        print(f"Merge result: {successful_matches}/{len(trades)} trades matched with features ({successful_matches/len(trades)*100:.1f}%)")
        
        # Detailed match info per feature
        for col in feature_cols:
            valid_count = merged_final[col].notna().sum()
            print(f"  {col}: {valid_count}/{len(trades)} trades ({valid_count/len(trades)*100:.1f}%)")
            
            # Show sample of matched values and their reuse
            sample_data = merged_final[merged_final[col].notna()][col].head(10)
            if len(sample_data) > 0:
                print(f"    Sample values: {sample_data.tolist()}")
                
                # Show how many times each feature value is reused
                value_counts = merged_final[col].value_counts().head(3)
                print(f"    Most used values: {value_counts.to_dict()}")
    
    return merged_final
