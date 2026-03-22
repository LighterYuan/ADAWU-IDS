from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class _FallbackConfig:
    SEQUENCE_LENGTH = 10
    LSTM_UNITS = 64
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 1e-3
    ADAPTATION_RATE = 1e-4
    EPOCHS = 5
    BATCH_SIZE = 256
    MODEL_DIR = "results/models"


try:
    from config import Config  # type: ignore
except Exception:
    Config = _FallbackConfig


class LSTMIDModel:
    def __init__(self, input_shape=None, num_classes=2):
        self.model = None
        self.input_shape = input_shape or (getattr(Config, "SEQUENCE_LENGTH", 10), 80)
        self.num_classes = num_classes
        self.history = None

    def build_model(self):
        model = Sequential([
            LSTM(
                getattr(Config, "LSTM_UNITS", 64),
                return_sequences=True,
                input_shape=self.input_shape,
                dropout=getattr(Config, "DROPOUT_RATE", 0.2),
                recurrent_dropout=0.0,
            ),
            BatchNormalization(),
            LSTM(
                max(getattr(Config, "LSTM_UNITS", 64) // 2, 16),
                return_sequences=False,
                dropout=getattr(Config, "DROPOUT_RATE", 0.2),
                recurrent_dropout=0.0,
            ),
            BatchNormalization(),
            Dense(getattr(Config, "DENSE_UNITS", 64), activation="relu"),
            Dropout(getattr(Config, "DROPOUT_RATE", 0.2)),
            BatchNormalization(),
            Dense(self.num_classes, activation="softmax"),
        ])

        model.compile(
            optimizer=Adam(learning_rate=getattr(Config, "LEARNING_RATE", 1e-3)),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def build_adaptive_model(self):
        inputs = Input(shape=self.input_shape, name="input_layer")

        x = LSTM(
            getattr(Config, "LSTM_UNITS", 64),
            return_sequences=True,
            dropout=getattr(Config, "DROPOUT_RATE", 0.2),
            recurrent_dropout=0.0,
            name="lstm1",
        )(inputs)
        x = BatchNormalization(name="bn1")(x)

        x = LSTM(
            max(getattr(Config, "LSTM_UNITS", 64) // 2, 16),
            return_sequences=False,
            dropout=getattr(Config, "DROPOUT_RATE", 0.2),
            recurrent_dropout=0.0,
            name="lstm2",
        )(x)
        x = BatchNormalization(name="bn2")(x)

        x = Dense(
            getattr(Config, "DENSE_UNITS", 64),
            activation="relu",
            name="feature_extractor",
        )(x)
        x = Dropout(getattr(Config, "DROPOUT_RATE", 0.2), name="dropout1")(x)
        x = BatchNormalization(name="bn3")(x)

        outputs = Dense(self.num_classes, activation="softmax", name="classifier")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=getattr(Config, "LEARNING_RATE", 1e-3)),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        if self.model is None:
            self.build_model()

        epochs = epochs or getattr(Config, "EPOCHS", 5)
        batch_size = batch_size or getattr(Config, "BATCH_SIZE", 256)

        model_dir = Path(getattr(Config, "MODEL_DIR", "results/models"))
        model_dir.mkdir(parents=True, exist_ok=True)

        monitor_loss = "val_loss" if X_val is not None and y_val is not None else "loss"
        monitor_acc = "val_accuracy" if X_val is not None and y_val is not None else "accuracy"

        callbacks = [
            EarlyStopping(monitor=monitor_loss, patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor_loss, factor=0.5, patience=3, min_lr=1e-7),
            ModelCheckpoint(
                filepath=str(model_dir / "best_lstm_model.keras"),
                monitor=monitor_acc,
                save_best_only=True,
                mode="max",
            ),
        ]

        fit_kwargs = dict(
            x=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        if X_val is not None and y_val is not None:
            fit_kwargs["validation_data"] = (X_val, y_val)

        self.history = self.model.fit(**fit_kwargs)
        return self.history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.evaluate(X_test, y_test, verbose=0)

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

    def get_feature_extractor(self):
        if self.model is None:
            raise ValueError("Model not trained")
        try:
            return Model(inputs=self.model.input, outputs=self.model.get_layer("feature_extractor").output)
        except Exception:
            return Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

    def adaptive_update(self, X_new, y_new, learning_rate=None):
        if self.model is None:
            raise ValueError("Model not trained")

        lr = learning_rate or getattr(Config, "ADAPTATION_RATE", 1e-4)
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(
            X_new,
            y_new,
            epochs=1,
            batch_size=getattr(Config, "BATCH_SIZE", 256),
            verbose=0,
        )
