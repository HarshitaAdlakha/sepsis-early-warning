"""
LSTM and GRU models for sequential sepsis prediction.

Architecture: bidirectional recurrent layers → dense classification head.
Masking layer ignores padded time-steps during training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Literal
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


MASK_VALUE = -999.0


def build_rnn_model(
    n_timesteps: int,
    n_features: int,
    rnn_type: Literal["lstm", "gru"] = "lstm",
    n_units: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.1,
    dense_units: Tuple[int, ...] = (32,),
    l2_reg: float = 1e-4,
    bidirectional: bool = True,
    learning_rate: float = 1e-3,
    mask_value: float = MASK_VALUE,
) -> keras.Model:
    """
    Build an LSTM or GRU model with a masking layer.

    Parameters
    ----------
    rnn_type : 'lstm' | 'gru'
    n_units  : hidden units per recurrent layer
    n_layers : number of stacked recurrent layers
    """
    inp = keras.Input(shape=(n_timesteps, n_features), name="sequence_input")
    x = layers.Masking(mask_value=mask_value)(inp)

    RNNCls = layers.LSTM if rnn_type == "lstm" else layers.GRU

    for i in range(n_layers):
        return_sequences = (i < n_layers - 1)
        rnn_layer = RNNCls(
            units=n_units,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{rnn_type}_{i}",
        )
        if bidirectional:
            x = layers.Bidirectional(rnn_layer, name=f"bi_{rnn_type}_{i}")(x)
        else:
            x = rnn_layer(x)

    for j, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f"dense_{j}")(x)
        x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=f"sepsis_{rnn_type}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auroc"),
            keras.metrics.AUC(name="auprc", curve="PR"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


class SepsisRNNModel:
    """Wrapper around a Keras RNN model with train / predict / save / load."""

    def __init__(self, model: Optional[keras.Model] = None):
        self.model = model

    @classmethod
    def build(cls, **kwargs) -> "SepsisRNNModel":
        return cls(model=build_rnn_model(**kwargs))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: Optional[dict] = None,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 8,
        log_dir: Optional[str] = None,
    ) -> keras.callbacks.History:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auroc", patience=patience,
                mode="max", restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auroc", factor=0.5, patience=4,
                mode="max", min_lr=1e-6, verbose=0,
            ),
        ]
        if log_dir:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return self.model.predict(X, batch_size=batch_size, verbose=0).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "SepsisRNNModel":
        model = keras.models.load_model(path)
        return cls(model=model)
