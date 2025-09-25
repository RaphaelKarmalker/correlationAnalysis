from __future__ import annotations
import pandas as pd
from typing import List, Optional, Tuple

def load_dataset(
    path: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    parse_dates: Optional[List[str]] = None,
    index_col: Optional[str] = None,
    dropna: bool = True,
    fill_method: Optional[str] = "ffill"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Lädt CSV, wählt Feature + Target Spalten.
    - feature_columns=None => alle außer target.
    - dropna True: Zeilen mit NaN im Target werden entfernt.
    - fill_method: ffill/bfill/None für Features.
    """
    df = pd.read_csv(path, parse_dates=parse_dates, index_col=index_col)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]
    missing = set([target_column] + feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten im CSV: {missing}")
    target = df[target_column]
    features = df[feature_columns].copy()

    if fill_method in ("ffill", "bfill"):
        features = features.fillna(method=fill_method)
    if dropna:
        mask = ~target.isna()
        target = target[mask]
        features = features.loc[mask]

    # Align indices (Sicherheit)
    features, target = features.align(target, join="inner", axis=0)
    return features, target
