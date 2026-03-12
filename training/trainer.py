"""
Model Training and Evaluation Module
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from models.lstm_model import LSTMIDModel
from data.data_processor import CICIDSDataProcessor
from config import Config

# 可选的drift相关导入
try:
    from drift.concept_drift_detector import ConceptDriftDetector
    from drift.adaptive_learning import AdaptiveLearningSystem
    DRIFT_AVAILABLE = True
except ImportError:
    DRIFT_AVAILABLE = False
    ConceptDriftDetector = None
    AdaptiveLearningSystem = None

class ModelTrainer:
    """Model Trainer"""

    def __init__(self):
        self.model = None
        self.drift_detector = None
        self.adaptive_system = None
        self.data_processor = CICIDSDataProcessor()
        self.training_history = None
        self.evaluation_results = None

    def prepare_data(self, data_dir=None, use_processed=True, sample_fraction=None):
        """
        Prepare training data with consistent dimension handling
        """
        processed_dir = Config.PROCESSED_DATA_DIR

        # Check if processed data exists
        processed_data_exists = (
                use_processed and
                os.path.exists(processed_dir) and
                os.path.exists(os.path.join(processed_dir, 'X_train.npy')) and
                os.path.exists(os.path.join(processed_dir, 'y_train.npy'))
        )

        if processed_data_exists and sample_fraction is None:
            # Load preprocessed data
            print("Loading preprocessed data...")
            try:
                train_data, val_data, test_data = self.data_processor.load_processed_data(processed_dir)

                # 确保所有数据都是78维
                train_data, val_data, test_data = self._ensure_78_features(train_data, val_data, test_data)

                return train_data, val_data, test_data
            except Exception as e:
                print(f"Error loading preprocessed data: {e}")
                print("Falling back to raw data processing...")

        # If we get here, either use_processed is False, processed data doesn't exist, or sampling is requested
        if data_dir and os.path.exists(data_dir):
            # Preprocess from raw data
            print("Preprocessing from raw data...")
            raw_data = self.data_processor.load_all_data(data_dir, sample_fraction)

            if raw_data is not None:
                # Split data
                train_data, val_data, test_data = self.data_processor.split_data(raw_data)

                # 确保所有数据都是78维
                train_data, val_data, test_data = self._ensure_78_features(train_data, val_data, test_data)

                # Save preprocessed data (only if no sampling)
                if sample_fraction is None:
                    self.data_processor.save_processed_data(
                        train_data, val_data, test_data, processed_dir
                    )

                return train_data, val_data, test_data
            else:
                raise ValueError("Unable to load raw data")
        else:
            raise ValueError(f"Data directory not found: {data_dir}")

    def _ensure_78_features(self, train_data, val_data, test_data):
        """确保所有数据都是78个特征维度"""

        def truncate_features(data):
            X, y = data
            if X.shape[2] != 78:
                print(f"截断特征维度: {X.shape[2]} -> 78")
                X = X[:, :, :78]
            return X, y

        train_data = truncate_features(train_data)
        val_data = truncate_features(val_data)
        test_data = truncate_features(test_data)

        return train_data, val_data, test_data

    def train_basic_model(self, train_data, val_data, class_weight=None, epochs=None, batch_size=None):
        """
        Train basic LSTM model
        """
        print("Starting basic LSTM model training...")

        # 验证输入维度
        actual_features = train_data[0].shape[2]
        print(f"实际数据特征维度: {actual_features}")

        if actual_features != 78:
            print(f"⚠ 警告: 数据特征维度为 {actual_features}，期望78")
            print("强制截断为78个特征...")
            # 强制截断特征维度
            train_data = (train_data[0][:, :, :78], train_data[1])
            val_data = (val_data[0][:, :, :78], val_data[1])
            print(f"截断后训练数据形状: {train_data[0].shape}")

        # 创建模型，强制使用78个特征
        self.model = LSTMIDModel(input_shape=(Config.SEQUENCE_LENGTH, 78))
        self.model.build_model()

        # Calculate class weights (handle imbalanced data)
        if class_weight is None:
            try:
                classes = np.unique(train_data[1])
                class_weights = compute_class_weight(
                    'balanced', classes=classes, y=train_data[1]
                )
                class_weight = dict(zip(classes, class_weights))
                print(f"Class weights: {class_weight}")
            except Exception as e:
                print(f"Error calculating class weights: {e}")
                class_weight = None

        # Use provided parameters or default from config
        train_epochs = epochs or Config.EPOCHS
        train_batch_size = batch_size or Config.BATCH_SIZE

        # Train model
        self.training_history = self.model.train(
            train_data[0], train_data[1],
            val_data[0], val_data[1],
            epochs=train_epochs,
            batch_size=train_batch_size
        )

        print("Basic model training completed")
        return self.training_history.history

    def evaluate_model(self, test_data, save_results=True):
        """
        Evaluate model performance
        """
        if self.model is None:
            raise ValueError("Model not trained")

        print("Starting model evaluation...")

        # 在评估时也需要截断测试数据
        X_test = test_data[0]
        y_test = test_data[1]

        # 检查测试数据维度并截断
        if X_test.shape[2] != 78:
            print(f"评估阶段: 测试数据维度为 {X_test.shape[2]}，截断为78")
            X_test = X_test[:, :, :78]

        # Predict
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = y_test

        # Calculate evaluation metrics
        accuracy = np.mean(y_pred == y_true)

        try:
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            auc_score = 0.5

        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        self.evaluation_results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist(),
            'true_labels': y_true.tolist()
        }

        print(f"Model accuracy: {accuracy:.4f}")
        print(f"AUC score: {auc_score:.4f}")

        if save_results:
            self._save_evaluation_results()

        return self.evaluation_results

    def _plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(Config.RESULTS_DIR, f"roc_curve_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {save_path}")
        plt.show()

        return roc_auc

    def setup_drift_detection(self, reference_data):
        """
        Setup concept drift detection

        Args:
            reference_data: Reference data
        """
        if not DRIFT_AVAILABLE:
            print("警告: drift模块不可用，跳过概念漂移检测设置")
            return
            
        print("Setting up concept drift detection...")

        # Create drift detector
        self.drift_detector = ConceptDriftDetector(
            window_size=Config.DRIFT_WINDOW_SIZE,
            threshold=Config.DRIFT_THRESHOLD,
            adaptation_rate=Config.ADAPTATION_RATE
        )

        # Add reference data
        self.drift_detector.add_reference_data(reference_data[0], reference_data[1])

        # Create adaptive learning system
        self.adaptive_system = AdaptiveLearningSystem(
            self.model, self.drift_detector, Config.ADAPTATION_RATE
        )

        print("Concept drift detection setup completed")

    def stream_evaluation(self, test_data, batch_size=1000):
        """
        Stream evaluation - simulate online learning environment

        Args:
            test_data: Test data
            batch_size: Batch size

        Returns:
            dict: Stream evaluation results
        """
        if not DRIFT_AVAILABLE:
            raise ValueError("drift模块不可用，无法进行流式评估")
            
        if self.model is None or self.drift_detector is None:
            raise ValueError("Model or drift detector not initialized")

        print("Starting stream evaluation...")

        X_test, y_test = test_data
        num_batches = len(X_test) // batch_size

        stream_results = {
            'batch_accuracies': [],
            'drift_detections': [],
            'adaptations': [],
            'performance_trend': []
        }

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_test))

            X_batch = X_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]

            print(f"Processing batch {batch_idx + 1}/{num_batches}")

            # Detect concept drift
            drift_result = self.drift_detector.detect_drift(X_batch, y_batch, self.model)

            # Convert to serializable format
            serializable_drift_result = self._convert_to_serializable(drift_result)

            if drift_result['is_drift']:
                print(f"Concept drift detected (confidence: {drift_result['confidence']:.3f})")

                # Perform adaptive learning
                adaptation_result = self.adaptive_system.adaptive_learning_pipeline(
                    X_batch, y_batch, drift_result
                )

                stream_results['adaptations'].append(self._convert_to_serializable(adaptation_result))
            else:
                stream_results['adaptations'].append(None)

            # Evaluate current batch performance
            batch_accuracy = self._evaluate_batch(X_batch, y_batch)
            stream_results['batch_accuracies'].append(float(batch_accuracy))

            # Record drift detection results
            stream_results['drift_detections'].append(serializable_drift_result)

            # Update performance trend
            stream_results['performance_trend'].append({
                'batch': int(batch_idx),
                'accuracy': float(batch_accuracy),
                'drift_detected': bool(drift_result['is_drift']),
                'drift_confidence': float(drift_result['confidence'])
            })
     
        # Save stream evaluation results
        self._save_stream_results(stream_results)

        print("Stream evaluation completed")
        # 在stream_evaluation方法中增加详细记录

        return stream_results

    def _evaluate_batch(self, X_batch, y_batch):
        """Evaluate single batch performance"""
        y_pred_proba = self.model.predict(X_batch)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = float(np.mean(y_pred == y_batch))  # Convert to Python float
        return accuracy

    def _save_evaluation_results(self):
        """Save evaluation results"""
        if self.evaluation_results is None:
            return

        results_dir = Config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)

        # Save confusion matrix image
        self._plot_confusion_matrix(
            self.evaluation_results['confusion_matrix'],
            os.path.join(results_dir, f"confusion_matrix_{timestamp}.png")
        )

        print(f"Evaluation results saved to {results_dir}")

    def _save_stream_results(self, stream_results):
        """Save stream evaluation results"""
        results_dir = Config.RESULTS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert to JSON serializable format
        serializable_results = self._convert_to_serializable(stream_results)

        # Save results
        results_file = os.path.join(results_dir, f"stream_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # Plot performance trend
        self._plot_performance_trend(
            stream_results['performance_trend'],
            os.path.join(results_dir, f"performance_trend_{timestamp}.png")
        )

        print(f"Stream evaluation results saved to {results_dir}")

    def _convert_to_serializable(self, obj):
        """
        Convert object to JSON serializable format

        Args:
            obj: Object to convert

        Returns:
            JSON serializable object
        """
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        else:
            return obj

    def _plot_confusion_matrix(self, conf_matrix, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_trend(self, performance_trend, save_path):
        """Plot performance trend"""
        batches = [item['batch'] for item in performance_trend]
        accuracies = [item['accuracy'] for item in performance_trend]
        drift_detected = [item['drift_detected'] for item in performance_trend]

        plt.figure(figsize=(12, 6))

        # Plot accuracy trend
        plt.subplot(2, 1, 1)
        plt.plot(batches, accuracies, 'b-', label='Accuracy')
        plt.title('Model Performance Trend')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot drift detection
        plt.subplot(2, 1, 2)
        drift_batches = [batch for batch, drift in zip(batches, drift_detected) if drift]
        plt.scatter(drift_batches, [1] * len(drift_batches), c='red', s=50, label='Drift Detected')
        plt.ylim(0, 2)
        plt.xlabel('Batch')
        plt.ylabel('Drift Detection')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path=None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained")

        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(Config.MODEL_DIR, f"lstm_model_{timestamp}.h5")

        self.model.save_model(model_path)
        return model_path

    def load_model(self, model_path):
        """Load model"""
        if self.model is None:
            self.model = LSTMIDModel()

        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def get_training_summary(self):
        """Get training summary"""
        if self.training_history is None:
            return "Model not trained"

        history = self.training_history.history
        final_accuracy = history['val_accuracy'][-1]
        final_loss = history['val_loss'][-1]

        return {
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'total_epochs': len(history['loss']),
            'best_accuracy': max(history['val_accuracy']),
            'best_loss': min(history['val_loss'])
        }