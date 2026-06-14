"""
Plotting utilities for model evaluation and EDA.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.metrics import roc_curve, precision_recall_curve


PALETTE = {
    "xgboost": "#E8613C",
    "lstm":    "#4C72B0",
    "gru":     "#55A868",
}


def _save_or_show(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_roc_curves(
    model_results: Dict[str, Dict],
    title: str = "ROC Curves — Sepsis Prediction",
    output_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves for multiple models on one axis.

    Parameters
    ----------
    model_results : {model_name: {"y_true": ..., "y_proba": ..., "auroc": ...}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.50)")

    for name, res in model_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        color = PALETTE.get(name, None)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name.upper()}  (AUROC={res['auroc']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_pr_curves(
    model_results: Dict[str, Dict],
    title: str = "Precision-Recall Curves — Sepsis Prediction",
    output_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in model_results.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        color = PALETTE.get(name, None)
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{name.upper()}  (AUPRC={res['auprc']:.3f})")

    prevalence = float(np.array(list(model_results.values())[0]["y_true"]).mean())
    ax.axhline(prevalence, color="k", linestyle="--", lw=1,
               label=f"No-skill baseline ({prevalence:.3f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_feature_importance(
    importance: Dict[str, float],
    top_n: int = 20,
    title: str = "Top Feature Importances (XGBoost)",
    output_path: Optional[str] = None,
) -> None:
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names  = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(names, values, color=PALETTE["xgboost"], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_training_history(
    history_dict: dict,
    title: str = "Training History",
    output_path: Optional[str] = None,
) -> None:
    """Plot Keras training history (loss + AUROC)."""
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(history_dict["loss"],     label="Train loss")
    ax1.plot(history_dict["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(history_dict["auroc"],     label="Train AUROC")
    ax2.plot(history_dict["val_auroc"], label="Val AUROC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.set_title("AUROC")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    model_name: str = "Model",
    output_path: Optional[str] = None,
) -> None:
    """Reliability diagram (calibration plot)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_true = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() > 0:
            bin_means.append(y_proba[mask].mean())
            bin_true.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.scatter(bin_means, bin_true, s=60, color=PALETTE.get(model_name.lower(), "#555"),
               zorder=3, label=model_name)
    ax.plot(bin_means, bin_true, color=PALETTE.get(model_name.lower(), "#555"), lw=1.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — {model_name}")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_vitals_comparison(
    sepsis_df: pd.DataFrame,
    control_df: pd.DataFrame,
    variables: List[str],
    output_path: Optional[str] = None,
) -> None:
    """Side-by-side boxplot of selected vitals for sepsis vs control."""
    import pandas as pd

    n = len(variables)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).ravel()

    for i, var in enumerate(variables):
        ax = axes[i]
        data_to_plot = []
        labels_to_plot = []
        for df, label in [(control_df, "Control"), (sepsis_df, "Sepsis")]:
            if var in df.columns:
                data_to_plot.append(df[var].dropna().values)
                labels_to_plot.append(label)
        ax.boxplot(data_to_plot, labels=labels_to_plot,
                   patch_artist=True,
                   boxprops=dict(facecolor="#c6def1"))
        ax.set_title(var.replace("_", " ").title(), fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Clinical Variable Distribution: Sepsis vs Control", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, output_path)
