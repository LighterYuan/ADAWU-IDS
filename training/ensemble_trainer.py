"""
集成训练器 - 动态集成学习 + 概念漂移检测主线
Ensemble Trainer - Dynamic Ensemble Learning + Concept Drift Detection (Mainline)
"""
import os
import numpy as np
from datetime import datetime
import json
import tensorflow as tf


from models.lstm_model import LSTMIDModel
from config import Config

try:
    from drift.concept_drift_detector import ConceptDriftDetector
    from drift.dynamic_ensemble import DynamicEnsemble
    from drift.adaptive_learning import AdaptiveLearningSystem
    DRIFT_AVAILABLE = True
except ImportError:
    DRIFT_AVAILABLE = False


class EnsembleTrainer:
    """
    集成训练器 - 支持动态集成学习和自适应权重调整
    """
    
    def __init__(self, use_ensemble: bool = None, ensemble_type: str = None):
        """
        Args:
            use_ensemble: 是否使用集成学习（None则使用Config中的设置）
            ensemble_type: 集成类型（None则使用Config中的设置）
        """
        self.use_ensemble = use_ensemble if use_ensemble is not None else Config.USE_ENSEMBLE
        # 精简后仅支持 dynamic 集成
        self.ensemble_type = 'dynamic'
        
        self.base_model = None
        self.ensemble = None
        self.adaptive_system = None
        self.drift_detector = None
        
        self.training_history = []
        self.drift_history = []
        self.ensemble_history = []
        
    def create_base_model(self, input_shape=None):
        """创建基础模型"""
        if input_shape is None:
            input_shape = (Config.SEQUENCE_LENGTH, 78)
        
        self.base_model = LSTMIDModel(input_shape=input_shape)
        self.base_model.build_model()
        return self.base_model

    def setup_ensemble(self, n_models: int = 3):
        """
        设置集成学习系统（DynamicEnsemble + DAWU）

        关键修改：
        - 不再 deepcopy 同一个 base_model
        - 每个基模型重新独立初始化
        - 使用不同随机种子，提升模型多样性
        """
        if not DRIFT_AVAILABLE:
            print("警告: drift模块不可用，无法设置集成学习")
            return

        if self.base_model is None:
            raise ValueError("请先创建基础模型")

        # 创建多个独立初始化的基模型
        base_models = []
        input_shape = self.base_model.input_shape
        num_classes = self.base_model.num_classes

        print(f"正在创建 {n_models} 个独立初始化的基模型...")

        for i in range(n_models):
            seed = 42 + i
            np.random.seed(seed)
            tf.random.set_seed(seed)

            model = LSTMIDModel(input_shape=input_shape, num_classes=num_classes)
            model.build_model()
            base_models.append(model)

            print(f"[INFO] Base model {i + 1} initialized with seed={seed}")

        # 创建漂移检测器（统一使用综合检测器 + MSDI）
        self.drift_detector = ConceptDriftDetector(
            window_size=Config.DRIFT_WINDOW_SIZE,
            threshold=Config.DRIFT_THRESHOLD,
            adaptation_rate=Config.ADAPTATION_RATE
        )

        # 创建集成（DynamicEnsemble + DAWU）
        self.ensemble = DynamicEnsemble(
            base_models,
            weight_update_method='dawu',
            decay_factor=Config.WEIGHT_DECAY_FACTOR,
            min_weight=Config.MIN_MODEL_WEIGHT
        )

        # 创建自适应学习系统（传入已创建的集成系统）
        self.adaptive_system = AdaptiveLearningSystem(
            self.base_model,
            self.drift_detector,
            adaptation_rate=Config.ADAPTATION_RATE,
            use_ensemble=self.use_ensemble,
            ensemble=self.ensemble if self.use_ensemble else None
        )

        print(f"集成学习系统已设置: {self.ensemble_type}, {n_models}个基模型")

    def train_initial_models(self, train_data, val_data, epochs=50, batch_size=256):
        """
        训练初始模型

        关键修改：
        - 每个基模型使用不同 bootstrap 子样本训练
        - 让 ensemble 中的模型产生可区分差异
        """
        if self.ensemble is None:
            # 不使用集成，只训练单个模型
            if self.base_model is None:
                self.create_base_model()

            X_train, y_train = train_data
            X_val, y_val = val_data

            history = self.base_model.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size
            )
            self.training_history.append(history.history)
        else:
            # 训练集成中的每个模型
            X_train, y_train = train_data
            X_val, y_val = val_data

            n_samples = len(X_train)

            for i, model in enumerate(self.ensemble.base_models):
                print(f"训练模型 {i + 1}/{len(self.ensemble.base_models)}")

                # 为每个模型生成不同的 bootstrap 子样本
                rng = np.random.RandomState(100 + i)
                sample_idx = rng.choice(n_samples, size=n_samples, replace=True)

                X_sub = X_train[sample_idx]
                y_sub = y_train[sample_idx]

                unique_ratio = len(np.unique(sample_idx)) / len(sample_idx)
                print(f"[INFO] Model {i + 1} bootstrap unique ratio = {unique_ratio:.3f}")

                history = model.train(
                    X_sub, y_sub, X_val, y_val,
                    epochs=epochs, batch_size=batch_size
                )

                if hasattr(history, "history"):
                    self.training_history.append({
                        "model_index": i,
                        "history": history.history
                    })

            # 添加参考数据到漂移检测器
            if self.drift_detector:
                self.drift_detector.add_reference_data(X_train, y_train)

        print("初始模型训练完成")

    def adaptive_evaluation(self, test_data_stream, batch_size=1000):
        """
        自适应评估 - 在流式数据上评估并自适应调整
        
        Args:
            test_data_stream: 测试数据流（可以是生成器或列表）
            batch_size: 批次大小
        """
        if self.adaptive_system is None:
            raise ValueError("请先设置集成学习系统")
        
        results = {
            'batch_accuracies': [],
            'drift_detections': [],
            'adaptations': [],
            'ensemble_weights': []
        }
        
        if isinstance(test_data_stream, (list, tuple)):
            # 如果是列表，转换为批次
            X_test, y_test = test_data_stream
            n_batches = len(X_test) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_test))
                
                X_batch = X_test[start_idx:end_idx]
                y_batch = y_test[start_idx:end_idx]
                
                # 自适应学习
                adaptation_result = self.adaptive_system.adaptive_learning_pipeline(
                    X_batch, y_batch
                )
                
                # 评估性能
                y_pred = self.adaptive_system.predict(X_batch)
                if len(y_pred.shape) > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                
                accuracy = np.mean(y_pred == y_batch)
                
                # 记录结果
                results['batch_accuracies'].append(float(accuracy))
                results['drift_detections'].append(adaptation_result['drift_detected'])
                results['adaptations'].append(adaptation_result)
                
                if self.use_ensemble and self.ensemble:
                    results['ensemble_weights'].append(self.ensemble.get_weights().tolist())
                
                print(f"批次 {batch_idx+1}/{n_batches}: "
                      f"准确率={accuracy:.4f}, "
                      f"漂移={adaptation_result['drift_detected']}, "
                      f"策略={adaptation_result['adaptation_strategy']}")
        
        return results
    
    def evaluate_model(self, test_data, use_ensemble=None):
        """
        评估模型
        
        Args:
            test_data: (X_test, y_test)
            use_ensemble: 是否使用集成（None则使用self.use_ensemble）
        """
        X_test, y_test = test_data
        
        use_ens = use_ensemble if use_ensemble is not None else self.use_ensemble
        
        if use_ens and self.ensemble:
            y_pred = self.ensemble.predict(X_test)
        elif self.base_model:
            y_pred = self.base_model.predict(X_test)
        else:
            raise ValueError("没有可用的模型")
        
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        accuracy = np.mean(y_pred == y_test)
        
        return {
            'accuracy': float(accuracy),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }

    def save_ensemble(self, save_dir=None):
        """保存集成模型（保存每个基学习器 + 集成元信息）"""
        if save_dir is None:
            save_dir = Config.MODEL_DIR

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_model_paths = []

        # 1) 优先保存真正的集成基模型
        if self.ensemble and hasattr(self.ensemble, "base_models") and self.ensemble.base_models:
            for i, model_obj in enumerate(self.ensemble.base_models):
                model_path = os.path.join(save_dir, f"ensemble_base_{i}_{timestamp}.h5")

                try:
                    # 若是项目里的 LSTMIDModel 包装器
                    if hasattr(model_obj, "save_model"):
                        model_obj.save_model(model_path)

                    # 若直接是 keras model
                    elif hasattr(model_obj, "save"):
                        model_obj.save(model_path)
                        print(f"Model saved to: {model_path}")

                    # 若对象里有 .model
                    elif hasattr(model_obj, "model") and hasattr(model_obj.model, "save"):
                        model_obj.model.save(model_path)
                        print(f"Model saved to: {model_path}")

                    else:
                        raise ValueError(f"模型 {i} 不支持保存")

                    saved_model_paths.append(model_path)

                except Exception as e:
                    print(f"[WARN] 保存第 {i} 个基模型失败: {e}")

        # 2) 如果没有 ensemble，则回退保存单模型
        elif self.base_model:
            model_path = os.path.join(save_dir, f"ensemble_base_0_{timestamp}.h5")
            self.base_model.save_model(model_path)
            saved_model_paths.append(model_path)

        # 3) 保存集成信息
        ensemble_info = {
            "ensemble_type": self.ensemble_type,
            "use_ensemble": bool(self.use_ensemble),
            "n_models": int(len(saved_model_paths)) if saved_model_paths else 0,
            "timestamp": timestamp,
            "model_paths": [os.path.basename(p) for p in saved_model_paths],
            "weights": None,
            "model_info": None
        }

        if self.ensemble:
            try:
                ensemble_info["weights"] = self.ensemble.get_weights().tolist()
            except Exception:
                ensemble_info["weights"] = None

            try:
                ensemble_info["model_info"] = self.ensemble.get_model_info()
            except Exception:
                ensemble_info["model_info"] = None

        info_path = os.path.join(save_dir, f"ensemble_info_{timestamp}.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(ensemble_info, f, indent=2, ensure_ascii=False)

        print(f"集成模型已保存到: {save_dir}")
        print(f"[INFO] Saved {len(saved_model_paths)} base model(s)")
        for p in saved_model_paths:
            print(f"  - {os.path.basename(p)}")
        print(f"[INFO] Ensemble info: {os.path.basename(info_path)}")

    def get_status(self):
        """获取系统状态"""
        status = {
            'use_ensemble': self.use_ensemble,
            'ensemble_type': self.ensemble_type,
            'has_base_model': self.base_model is not None,
            'has_ensemble': self.ensemble is not None,
            'has_adaptive_system': self.adaptive_system is not None
        }
        
        if self.adaptive_system:
            status['adaptive_status'] = self.adaptive_system.get_status()
        
        if self.ensemble:
            status['ensemble_info'] = self.ensemble.get_model_info()
        
        return status

