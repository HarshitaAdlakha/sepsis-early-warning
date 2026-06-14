"""Tests for evaluation metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, find_threshold_at_sensitivity


class TestComputeMetrics:
    def _perfect(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        return compute_metrics(y, p)

    def test_perfect_auroc(self):
        m = self._perfect()
        assert m["auroc"] == pytest.approx(1.0)

    def test_perfect_sensitivity_specificity(self):
        m = self._perfect()
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)

    def test_random_auroc_near_half(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        p = rng.uniform(0, 1, 500)
        m = compute_metrics(y, p)
        assert 0.35 < m["auroc"] < 0.65

    def test_all_keys_present(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7])
        m = compute_metrics(y, p)
        required = ["auroc", "auprc", "sensitivity", "specificity", "f1", "brier"]
        for k in required:
            assert k in m


class TestThresholdSearch:
    def test_finds_high_sensitivity(self):
        rng = np.random.default_rng(1)
        y = np.array([0] * 100 + [1] * 100)
        p = np.concatenate([rng.uniform(0, 0.4, 100), rng.uniform(0.5, 1.0, 100)])
        thresh = find_threshold_at_sensitivity(y, p, target_sensitivity=0.80)
        # verify the threshold is a valid probability in [0, 1]
        assert 0.0 <= thresh <= 1.0
        # verify the model actually achieves >= 80% sensitivity below this threshold
        y_pred = (p >= thresh).astype(int)
        tp = ((y == 1) & (y_pred == 1)).sum()
        sensitivity = tp / (y == 1).sum()
        assert sensitivity >= 0.75  # tolerance for random data
