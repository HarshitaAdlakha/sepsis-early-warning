"""
XGBoost training script for sepsis prediction.

Usage:
  python src/training/train_xgboost.py --mode tune   # Optuna search
  python src/training/train_xgboost.py --mode train  # Train with best/default params
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_split
from src.data.preprocessing import (
    encode_time_series_xgboost,
    fit_xgboost_scaler,
    apply_xgboost_scaler,
    compute_class_weights,
)
from src.models.xgboost_model import SepsisXGBModel, tune_xgboost
from src.evaluation.metrics import compute_metrics, save_metrics
from src.visualization.plots import plot_feature_importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["tune", "train"], default="train")
    parser.add_argument("--data-dir",  default="data/processed")
    parser.add_argument("--model-dir", default="results/models")
    parser.add_argument("--output-dir",default="results/xgboost")
    parser.add_argument("--n-trials",  type=int, default=30,
                        help="Optuna trials (tune mode only)")
    args = parser.parse_args()

    print("Loading data...")
    train_series, y_train, _ = load_split(args.data_dir, "train")
    val_series,   y_val,   _ = load_split(args.data_dir, "val")
    test_series,  y_test,  _ = load_split(args.data_dir, "test")

    print("Encoding time series (XGBoost features)...")
    X_train = encode_time_series_xgboost(train_series)
    X_val   = encode_time_series_xgboost(val_series)
    X_test  = encode_time_series_xgboost(test_series)

    scaler_path = str(Path(args.model_dir) / "xgboost_scaler.joblib")
    scaler = fit_xgboost_scaler(X_train, scaler_path=scaler_path)
    X_train = apply_xgboost_scaler(X_train, scaler)
    X_val   = apply_xgboost_scaler(X_val,   scaler)
    X_test  = apply_xgboost_scaler(X_test,  scaler)

    cw = compute_class_weights(y_train)
    scale_pos_weight = cw[1] / cw[0]

    params_path = str(Path(args.output_dir) / "best_params.json")

    if args.mode == "tune":
        print(f"\nRunning Optuna hyperparameter search ({args.n_trials} trials)...")
        best_params = tune_xgboost(
            X_train, y_train, X_val, y_val,
            n_trials=args.n_trials,
            output_path=params_path,
        )
    else:
        if Path(params_path).exists():
            with open(params_path) as f:
                best_params = json.load(f)
            print(f"Loaded tuned params from {params_path}")
        else:
            best_params = {}
            print("Using default XGBoost params.")

    print("\nTraining final XGBoost model...")
    model = SepsisXGBModel(params=best_params)
    model.fit(X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos_weight)

    model_path = str(Path(args.model_dir) / "xgboost.json")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    print("\nEvaluating on test set...")
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_proba)
    save_metrics(metrics, str(Path(args.output_dir) / "test_metrics.json"))

    print("\n=== Test Set Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<15} {v:.4f}")

    # Feature importance plot
    from src.data.preprocessing import ALL_FEATURES, STAT_FUNCTIONS
    feat_names = ["series_length"] + [
        f"{feat}_{stat}" for feat in ALL_FEATURES for stat in STAT_FUNCTIONS
    ]
    importance = model.feature_importance(feature_names=feat_names)
    plot_feature_importance(
        importance, top_n=25,
        output_path=str(Path(args.output_dir) / "feature_importance.png"),
    )
    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
