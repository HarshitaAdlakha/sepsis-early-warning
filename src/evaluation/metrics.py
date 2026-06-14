"""
Evaluation metrics for sepsis prediction models.

Includes AUROC, AUPRC, sensitivity, specificity, F1, and a
clinical utility metric (net benefit).
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, f1_score,
    brier_score_loss,
)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true  : ground-truth binary labels
    y_proba : predicted positive-class probabilities
    threshold : decision threshold for hard predictions

    Returns
    -------
    dict of metric_name → float
    """
    y_pred = (y_proba >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    accuracy = (tp + tn) / len(y_true)

    # Net benefit at the given threshold (decision-curve analysis simplified)
    prevalence = y_true.mean()
    harm_ratio = threshold / (1 - threshold)
    net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * harm_ratio

    return {
        "auroc":       auroc,
        "auprc":       auprc,
        "brier":       brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv":         ppv,
        "npv":         npv,
        "f1":          f1,
        "accuracy":    accuracy,
        "net_benefit": net_benefit,
        "tp":          int(tp),
        "tn":          int(tn),
        "fp":          int(fp),
        "fn":          int(fn),
        "threshold":   threshold,
        "prevalence":  float(prevalence),
    }


def find_threshold_at_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_sensitivity: float = 0.80,
) -> float:
    """Return the decision threshold that achieves ≥ target sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid = thresholds[tpr >= target_sensitivity]
    return float(valid[-1]) if len(valid) > 0 else 0.5


def compare_models(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a comparison table from multiple models' metric dicts.

    Parameters
    ----------
    results : {model_name: metrics_dict}
    """
    df = pd.DataFrame(results).T
    ordered_cols = [
        "auroc", "auprc", "sensitivity", "specificity",
        "ppv", "npv", "f1", "brier", "net_benefit",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]
    df = df.round(4)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        print(f"Comparison table saved to {output_path}")

    return df


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
