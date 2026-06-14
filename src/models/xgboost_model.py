"""
XGBoost-based sepsis prediction model.

Wraps xgboost.XGBClassifier with early stopping, class-weight handling,
hyperparameter search via Optuna, and serialization.
"""

from __future__ import annotations

import json
import numpy as np
import optuna
import xgboost as xgb
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_PARAMS = {
    "objective":          "binary:logistic",
    "eval_metric":        "auc",
    "use_label_encoder":  False,
    "tree_method":        "hist",
    "n_estimators":       500,
    "learning_rate":      0.05,
    "max_depth":          6,
    "min_child_weight":   3,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "gamma":              0.1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "random_state":       42,
    "n_jobs":             -1,
    "verbosity":          0,
}


class SepsisXGBModel:
    """XGBoost classifier for sepsis prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 30,
    ) -> "SepsisXGBModel":
        params = {**self.params}
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight

        self.model = xgb.XGBClassifier(
            **params, early_stopping_rounds=early_stopping_rounds
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted."
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "SepsisXGBModel":
        obj = cls()
        obj.model = xgb.XGBClassifier()
        obj.model.load_model(path)
        return obj

    def feature_importance(self, feature_names=None) -> dict:
        scores = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(scores))]
        return dict(sorted(zip(feature_names, scores), key=lambda x: -x[1]))


# --------------------------------------------------------------------------- #
# Optuna hyperparameter search                                                 #
# --------------------------------------------------------------------------- #

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search; return best params dict."""

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "binary:logistic",
            "eval_metric":       "auc",
            "tree_method":       "hist",
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  scale_pos_weight,
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"Best XGBoost params saved to {output_path}")

    print(f"Best XGBoost AUROC: {study.best_value:.4f}")
    return best
