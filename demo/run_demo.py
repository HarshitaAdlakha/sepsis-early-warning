"""
End-to-end demo: generate synthetic data → train XGBoost → evaluate.

Runs in ~2-3 minutes on CPU with the default settings.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_demo_data import generate_dataset
from src.data.preprocessing import (
    encode_time_series_xgboost,
    fit_xgboost_scaler,
    apply_xgboost_scaler,
    compute_class_weights,
)
from src.models.xgboost_model import SepsisXGBModel
from src.evaluation.metrics import compute_metrics
from src.visualization.plots import plot_roc_curves, plot_pr_curves


def main():
    print("=" * 60)
    print("  Sepsis Early Warning System — Demo")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n[1/4] Generating synthetic ICU dataset...")
    dataset = generate_dataset(
        n_sepsis=200, n_control=800, seed=42,
        output_dir="data/processed",
    )
    (train_series, y_train) = dataset["train"]
    (val_series,   y_val)   = dataset["val"]
    (test_series,  y_test)  = dataset["test"]

    # 2. Feature engineering
    print("\n[2/4] Encoding time series into statistical features...")
    X_train = encode_time_series_xgboost(train_series)
    X_val   = encode_time_series_xgboost(val_series)
    X_test  = encode_time_series_xgboost(test_series)

    scaler  = fit_xgboost_scaler(X_train)
    X_train = apply_xgboost_scaler(X_train, scaler)
    X_val   = apply_xgboost_scaler(X_val,   scaler)
    X_test  = apply_xgboost_scaler(X_test,  scaler)

    cw = compute_class_weights(y_train)

    # 3. Train XGBoost
    print("\n[3/4] Training XGBoost classifier...")
    model = SepsisXGBModel(params={"n_estimators": 200, "learning_rate": 0.05})
    model.fit(X_train, y_train, X_val, y_val,
              scale_pos_weight=cw[1] / cw[0])

    # 4. Evaluate
    print("\n[4/4] Evaluating on held-out test set...")
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_proba)

    print("\n" + "=" * 40)
    print("  TEST SET RESULTS")
    print("=" * 40)
    key_metrics = ["auroc", "auprc", "sensitivity", "specificity", "f1", "brier"]
    for k in key_metrics:
        bar = "#" * int(metrics[k] * 20)
        print(f"  {k:<14} {metrics[k]:.4f}  {bar}")

    # 5. Plots
    Path("results/demo").mkdir(parents=True, exist_ok=True)
    results_dict = {
        "xgboost": {
            "y_true": y_test, "y_proba": y_proba,
            "auroc": metrics["auroc"], "auprc": metrics["auprc"],
        }
    }
    plot_roc_curves(results_dict, output_path="results/demo/roc_curve.png")
    plot_pr_curves( results_dict, output_path="results/demo/pr_curve.png")
    print("\nPlots saved to results/demo/")
    print("Demo complete!")


if __name__ == "__main__":
    main()
