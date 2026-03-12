"""
集成训练器 - 动态集成学习 + 概念漂移检测主线
Ensemble Trainer - Dynamic Ensemble Learning + Concept Drift Detection (Mainline)
"""

import os
import numpy as np
from datetime import datetime
import json
import copy

from models.lstm_model import LSTMIDModel
from data.data_processor import CICIDSDataProcessor
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
        
        Args:
            n_models: 初始模型数量
        """
        if not DRIFT_AVAILABLE:
            print("警告: drift模块不可用，无法设置集成学习")
            return
        
        if self.base_model is None:
            raise ValueError("请先创建基础模型")
        
        # 创建多个基模型
        base_models = []
        for i in range(n_models):
            model = copy.deepcopy(self.base_model)
            # 每个模型使用不同的随机种子初始化（如果需要）
            base_models.append(model)
        
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
        
        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            epochs: 训练轮数
            batch_size: 批次大小
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
            
            for i, model in enumerate(self.ensemble.base_models):
                print(f"训练模型 {i+1}/{len(self.ensemble.base_models)}")
                history = model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=epochs, batch_size=batch_size
                )
            
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
        """保存集成模型"""
        if save_dir is None:
            save_dir = Config.MODEL_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存基础模型
        if self.base_model:
            model_path = os.path.join(save_dir, f"ensemble_base_{timestamp}.h5")
            self.base_model.save_model(model_path)
        
        # 保存集成信息
        if self.ensemble:
            ensemble_info = {
                'ensemble_type': self.ensemble_type,
                'n_models': self.ensemble.n_models,
                'weights': self.ensemble.get_weights().tolist(),
                'model_info': self.ensemble.get_model_info()
            }
            
            info_path = os.path.join(save_dir, f"ensemble_info_{timestamp}.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(ensemble_info, f, indent=2, ensure_ascii=False)
            
            print(f"集成模型已保存到: {save_dir}")
    
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

