"""Tests for data preprocessing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    encode_time_series_xgboost,
    pad_sequences,
    forward_fill_series,
    compute_class_weights,
    ALL_FEATURES,
    MASK_VALUE,
)


def _make_series(n_hours=20, seed=0):
    rng = np.random.default_rng(seed)
    data = {feat: rng.normal(0, 1, n_hours) for feat in ALL_FEATURES}
    data["hour"] = list(range(n_hours))
    return pd.DataFrame(data)


class TestXGBoostEncoding:
    def test_shape(self):
        series = [_make_series(20), _make_series(30)]
        X = encode_time_series_xgboost(series)
        n_stats = 10  # count/mean/std/min/max/q25/q50/q75/last/slope
        expected_cols = 1 + len(ALL_FEATURES) * n_stats
        assert X.shape == (2, expected_cols)

    def test_no_nan_in_output(self):
        series = [_make_series(10)]
        X = encode_time_series_xgboost(series)
        assert not np.isnan(X).any()

    def test_with_missing_values(self):
        df = _make_series(15)
        df.iloc[::3, 1] = np.nan  # introduce NaNs
        X = encode_time_series_xgboost([df])
        assert not np.isnan(X).any()


class TestPadSequences:
    def test_shape(self):
        series = [_make_series(10), _make_series(50), _make_series(30)]
        X = pad_sequences(series, max_length=40, features=ALL_FEATURES)
        assert X.shape == (3, 40, len(ALL_FEATURES))

    def test_mask_value_present(self):
        short = _make_series(5)
        X = pad_sequences([short], max_length=20, features=ALL_FEATURES)
        assert (X[0, 5:, :] == MASK_VALUE).all()

    def test_truncation(self):
        long = _make_series(100)
        X = pad_sequences([long], max_length=20, features=ALL_FEATURES)
        assert X.shape == (1, 20, len(ALL_FEATURES))
        assert not (X == MASK_VALUE).any()


class TestForwardFill:
    def test_fills_nans(self):
        df = pd.DataFrame({feat: [1.0, np.nan, np.nan] for feat in ALL_FEATURES})
        df["hour"] = [0, 1, 2]
        filled = forward_fill_series(df)
        assert not filled[ALL_FEATURES].isna().any().any()


class TestClassWeights:
    def test_imbalanced(self):
        labels = np.array([0] * 90 + [1] * 10)
        weights = compute_class_weights(labels)
        assert weights[1] > weights[0]

    def test_balanced(self):
        labels = np.array([0, 1] * 50)
        weights = compute_class_weights(labels)
        assert abs(weights[0] - weights[1]) < 0.01
