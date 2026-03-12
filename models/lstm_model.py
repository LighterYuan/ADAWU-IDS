"""
LSTM Network Intrusion Detection Model
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from config import Config


class LSTMIDModel:
    """LSTM-based Network Intrusion Detection Model"""

    def __init__(self, input_shape=None, num_classes=2):
        self.model = None
        # 强制使用78个特征
        self.input_shape = input_shape or (Config.SEQUENCE_LENGTH, 78)
        self.num_classes = num_classes
        self.history = None

    def build_model(self):
        """Build LSTM model architecture"""
        print(f"构建LSTM模型，输入形状: {self.input_shape}")
        print(f"特征维度: {self.input_shape[1]} (强制为78)")

        model = Sequential([
            # First LSTM layer
            LSTM(Config.LSTM_UNITS,
                 return_sequences=True,
                 input_shape=self.input_shape,
                 dropout=Config.DROPOUT_RATE,
                 recurrent_dropout=Config.DROPOUT_RATE),
            BatchNormalization(),

            # Second LSTM layer
            LSTM(Config.LSTM_UNITS // 2,
                 return_sequences=False,
                 dropout=Config.DROPOUT_RATE,
                 recurrent_dropout=Config.DROPOUT_RATE),
            BatchNormalization(),

            # Fully connected layer
            Dense(Config.DENSE_UNITS, activation='relu'),
            Dropout(Config.DROPOUT_RATE),
            BatchNormalization(),

            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def build_adaptive_model(self):
        """Build adaptive LSTM model (for concept drift adaptation)"""
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_layer')

        # LSTM layers
        lstm1 = LSTM(Config.LSTM_UNITS,
                    return_sequences=True,
                    dropout=Config.DROPOUT_RATE,
                    recurrent_dropout=Config.DROPOUT_RATE,
                    name='lstm1')(inputs)
        bn1 = BatchNormalization(name='bn1')(lstm1)

        lstm2 = LSTM(Config.LSTM_UNITS // 2,
                    return_sequences=False,
                    dropout=Config.DROPOUT_RATE,
                    recurrent_dropout=Config.DROPOUT_RATE,
                    name='lstm2')(bn1)
        bn2 = BatchNormalization(name='bn2')(lstm2)

        # Feature extraction layer
        feature_extractor = Dense(Config.DENSE_UNITS,
                                activation='relu',
                                name='feature_extractor')(bn2)
        dropout1 = Dropout(Config.DROPOUT_RATE, name='dropout1')(feature_extractor)
        bn3 = BatchNormalization(name='bn3')(dropout1)

        # Classification layer
        classifier = Dense(self.num_classes,
                          activation='softmax',
                          name='classifier')(bn3)

        # Create model
        model = Model(inputs=inputs, outputs=classifier)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        """
        Train model

        Args:
            X_train, y_train: training data
            X_val, y_val: validation data
            epochs: number of epochs
            batch_size: batch size
        """
        if self.model is None:
            self.build_model()

        epochs = epochs or Config.EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss',
                         patience=10,
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss',
                             factor=0.5,
                             patience=5,
                             min_lr=1e-7),
            ModelCheckpoint(
                filepath=os.path.join(Config.MODEL_DIR, 'best_lstm_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained, please call train method first")

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        if self.model is None:
            raise ValueError("Model not trained, please call train method first")

        return self.model.evaluate(X_test, y_test, verbose=0)

    def save_model(self, filepath):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained, please call train method first")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def get_feature_extractor(self):
        """Get feature extractor (for concept drift detection)"""
        if self.model is None:
            raise ValueError("Model not trained, please call train method first")

        # Create feature extraction model
        feature_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('feature_extractor').output
        )

        return feature_model

    def adaptive_update(self, X_new, y_new, learning_rate=None):
        """
        Adaptive model parameter update (for concept drift adaptation)

        Args:
            X_new: new data features
            y_new: new data labels
            learning_rate: learning rate
        """
        if self.model is None:
            raise ValueError("Model not trained, please call train method first")

        # Use smaller learning rate for fine-tuning
        lr = learning_rate or Config.ADAPTATION_RATE
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train for a few epochs on new data
        self.model.fit(X_new, y_new, epochs=1, verbose=0)

    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "Model not built"

        return self.model.summary()