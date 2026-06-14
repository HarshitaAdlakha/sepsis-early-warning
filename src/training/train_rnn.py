"""
RNN (LSTM / GRU) training script for sepsis prediction.

Usage:
  python src/training/train_rnn.py --rnn-type lstm
  python src/training/train_rnn.py --rnn-type gru
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_split
from src.data.preprocessing import (
    pad_sequences,
    compute_rnn_feature_stats,
    normalise_rnn_sequences,
    compute_class_weights,
    ALL_FEATURES,
)
from src.models.rnn_model import SepsisRNNModel
from src.evaluation.metrics import compute_metrics, save_metrics
from src.visualization.plots import plot_training_history, plot_roc_curves, plot_pr_curves


MAX_LENGTH = 72   # hours — sequences truncated / padded to this


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn-type",  choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--data-dir",  default="data/processed")
    parser.add_argument("--model-dir", default="results/models")
    parser.add_argument("--output-dir",default="results/rnn")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch-size",type=int, default=64)
    parser.add_argument("--n-units",   type=int, default=64)
    parser.add_argument("--n-layers",  type=int, default=2)
    parser.add_argument("--dropout",   type=float, default=0.3)
    args = parser.parse_args()

    print("Loading data...")
    train_series, y_train, _ = load_split(args.data_dir, "train")
    val_series,   y_val,   _ = load_split(args.data_dir, "val")
    test_series,  y_test,  _ = load_split(args.data_dir, "test")

    print(f"Padding sequences to {MAX_LENGTH} time steps...")
    X_train = pad_sequences(train_series, MAX_LENGTH, ALL_FEATURES)
    X_val   = pad_sequences(val_series,   MAX_LENGTH, ALL_FEATURES)
    X_test  = pad_sequences(test_series,  MAX_LENGTH, ALL_FEATURES)

    stats_path = str(Path(args.model_dir) / f"{args.rnn_type}_feature_stats.npz")
    means, stds = compute_rnn_feature_stats(X_train, stats_path=stats_path)
    X_train = normalise_rnn_sequences(X_train, means, stds)
    X_val   = normalise_rnn_sequences(X_val,   means, stds)
    X_test  = normalise_rnn_sequences(X_test,  means, stds)

    class_weight = compute_class_weights(y_train)
    print(f"Class weights: {class_weight}")

    print(f"\nBuilding {args.rnn_type.upper()} model...")
    rnn = SepsisRNNModel.build(
        n_timesteps=MAX_LENGTH,
        n_features=len(ALL_FEATURES),
        rnn_type=args.rnn_type,
        n_units=args.n_units,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    rnn.model.summary()

    log_dir = str(Path("results") / "logs" / args.rnn_type)
    print(f"\nTraining for up to {args.epochs} epochs (early stopping)...")
    history = rnn.fit(
        X_train, y_train,
        X_val, y_val,
        class_weight=class_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_dir=log_dir,
    )

    model_path = str(Path(args.model_dir) / f"{args.rnn_type}_model")
    rnn.save(model_path)
    print(f"\nModel saved to {model_path}")

    out_dir = Path(args.output_dir) / args.rnn_type
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        history.history,
        title=f"{args.rnn_type.upper()} Training History",
        output_path=str(out_dir / "training_history.png"),
    )

    print("\nEvaluating on test set...")
    y_proba = rnn.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_proba)
    save_metrics(metrics, str(out_dir / "test_metrics.json"))

    print(f"\n=== Test Set Results ({args.rnn_type.upper()}) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<15} {v:.4f}")


if __name__ == "__main__":
    main()
