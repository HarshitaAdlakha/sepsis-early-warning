"""
Preprocessing pipeline for ICU time-series data.

Supports two model families:
  - XGBoost: encodes each time series as statistical features
  - RNN (LSTM/GRU): forward-fills + pads sequences to fixed length
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import joblib

ALL_FEATURES = [
    "heart_rate", "systolic_bp", "diastolic_bp", "mean_arterial_bp", "resp_rate",
    "spo2", "temperature", "gcs_total", "urine_output", "fio2", "peep",
    "tidal_volume", "rr_set", "plateau_pressure", "driving_pressure",
    "wbc", "hemoglobin", "hematocrit", "platelets", "sodium", "potassium",
    "chloride", "bicarbonate", "bun", "creatinine", "glucose", "lactate",
    "bilirubin", "alt", "ast", "albumin", "inr", "ptt", "ph", "pao2",
    "paco2", "base_excess", "troponin", "procalcitonin", "crp",
    "fibrinogen", "d_dimer", "bnp", "ferritin",
]


def _compute_slope(series: pd.Series) -> float:
    """Least-squares slope over time for non-null observations."""
    valid = series.dropna()
    if len(valid) < 2:
        return 0.0
    x = np.arange(len(valid), dtype=float)
    y = valid.values.astype(float)
    x -= x.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


STAT_FUNCTIONS = {
    "count":  lambda x: x.notna().sum(),
    "mean":   lambda x: x.mean(),
    "std":    lambda x: x.std(),
    "min":    lambda x: x.min(),
    "max":    lambda x: x.max(),
    "q25":    lambda x: x.quantile(0.25),
    "q50":    lambda x: x.quantile(0.50),
    "q75":    lambda x: x.quantile(0.75),
    "last":   lambda x: x.dropna().iloc[-1] if x.notna().any() else np.nan,
    "slope":  _compute_slope,
}


# --------------------------------------------------------------------------- #
# XGBoost feature engineering                                                  #
# --------------------------------------------------------------------------- #

def encode_time_series_xgboost(
    series_list: List[pd.DataFrame],
    features: List[str] = ALL_FEATURES,
) -> np.ndarray:
    """
    Convert a list of per-patient DataFrames into a 2-D feature matrix
    suitable for XGBoost.

    Each variable is encoded into count / mean / std / min / max /
    q25 / q50 / q75 / last / slope — giving 10 × 44 = 440 features,
    plus 1 'series_length' column = 441 total.
    """
    rows = []
    for df in series_list:
        row = {"series_length": len(df)}
        for feat in features:
            col = df[feat] if feat in df.columns else pd.Series(dtype=float)
            for stat_name, fn in STAT_FUNCTIONS.items():
                try:
                    val = fn(col)
                except Exception:
                    val = np.nan
                row[f"{feat}_{stat_name}"] = val
        rows.append(row)
    matrix = pd.DataFrame(rows).values.astype(np.float32)
    # Replace NaN/Inf with 0
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    return matrix


def fit_xgboost_scaler(X_train: np.ndarray, scaler_path: Optional[str] = None) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return scaler


def apply_xgboost_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


# --------------------------------------------------------------------------- #
# RNN sequence preprocessing                                                   #
# --------------------------------------------------------------------------- #

MASK_VALUE = -999.0


def forward_fill_series(df: pd.DataFrame, features: List[str] = ALL_FEATURES) -> pd.DataFrame:
    """Forward fill then fill remaining NaN with column median (global median=0 fallback)."""
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            df[feat] = np.nan
        df[feat] = df[feat].ffill()
        median = df[feat].median()
        df[feat] = df[feat].fillna(median if not np.isnan(median) else 0.0)
    return df


def pad_sequences(
    series_list: List[pd.DataFrame],
    max_length: int,
    features: List[str] = ALL_FEATURES,
    mask_value: float = MASK_VALUE,
) -> np.ndarray:
    """
    Pad / truncate each patient series to max_length.
    Returns array of shape (N, max_length, n_features).
    Positions beyond the actual length are filled with mask_value.
    """
    n_feats = len(features)
    result = np.full((len(series_list), max_length, n_feats), mask_value, dtype=np.float32)

    for i, df in enumerate(series_list):
        df_filled = forward_fill_series(df, features)
        seq = df_filled[features].values.astype(np.float32)
        length = min(len(seq), max_length)
        result[i, :length, :] = seq[:length]

    return result


def compute_rnn_feature_stats(
    X_train: np.ndarray,
    mask_value: float = MASK_VALUE,
    stats_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and std from non-masked training values
    for z-score normalisation before RNN training.
    """
    mask = X_train != mask_value
    n_feats = X_train.shape[2]
    means = np.zeros(n_feats, dtype=np.float32)
    stds  = np.ones(n_feats, dtype=np.float32)

    for f in range(n_feats):
        vals = X_train[:, :, f][mask[:, :, f]]
        if len(vals) > 0:
            means[f] = vals.mean()
            stds[f]  = vals.std() if vals.std() > 0 else 1.0

    if stats_path:
        np.savez(stats_path, means=means, stds=stds)

    return means, stds


def normalise_rnn_sequences(
    X: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    mask_value: float = MASK_VALUE,
) -> np.ndarray:
    """Z-score normalise, preserving mask positions."""
    X_norm = X.copy()
    mask = X != mask_value
    X_norm[mask] = (X[mask] - np.broadcast_to(means, X.shape)[mask]) / \
                   np.broadcast_to(stds,  X.shape)[mask]
    return X_norm


# --------------------------------------------------------------------------- #
# Label helpers                                                                 #
# --------------------------------------------------------------------------- #

def compute_class_weights(labels: np.ndarray) -> dict:
    """Return balanced class weights for imbalanced binary labels."""
    n_total = len(labels)
    n_pos = labels.sum()
    n_neg = n_total - n_pos
    return {
        0: n_total / (2 * n_neg),
        1: n_total / (2 * n_pos),
    }
