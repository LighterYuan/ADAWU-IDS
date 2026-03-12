"""
动态集成学习框架（精简版）
Dynamic Ensemble Learning Framework (Simplified)

当前项目主线仅保留：
- DynamicEnsemble + DAWU 漂移感知权重更新
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import copy
from abc import ABC, abstractmethod


class BaseEnsemble(ABC):
    """集成学习基类"""
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """增量学习"""
        pass


class DynamicEnsemble(BaseEnsemble):
    """
    动态集成学习框架
    支持多个基模型的动态权重调整
    """
    
    def __init__(self, base_models: List, initial_weights: Optional[List[float]] = None,
                 weight_update_method: str = 'dawu',
                 decay_factor: float = 0.9, min_weight: float = 0.01):
        """
        Args:
            base_models: 基模型列表
            initial_weights: 初始权重（None则均匀分配）
            weight_update_method: 权重更新方法
                - 'dawu': 漂移感知权重更新（项目主线使用）
            decay_factor: 时间衰减因子
            min_weight: 最小权重阈值
        """
        self.base_models = base_models
        self.n_models = len(base_models)
        
        if initial_weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(initial_weights)
            self.weights = self.weights / self.weights.sum()  # 归一化
        
        self.weight_update_method = weight_update_method
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.time_step = 0
        
        # DAWU 参数（默认值可由外部覆盖）
        self.dawu_alpha = getattr(self, 'dawu_alpha', 0.6)
        self.dawu_beta = getattr(self, 'dawu_beta', 0.3)
        self.dawu_gamma = getattr(self, 'dawu_gamma', 0.1)
        self.dawu_lambda = getattr(self, 'dawu_lambda', 0.05)

        # 性能历史记录
        self.performance_history = [deque(maxlen=100) for _ in range(self.n_models)]
        self.prediction_history = []
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测（加权投票）
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果（类别或概率）
        """
        predictions = []
        
        for model in self.base_models:
            try:
                pred = model.predict(X)
                # 如果是概率输出，取最大概率类别
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    pred = np.argmax(pred, axis=1)
                predictions.append(pred)
            except Exception as e:
                # 模型预测失败，使用随机预测
                predictions.append(np.random.randint(0, 2, size=len(X)))
        
        predictions = np.array(predictions)  # shape: (n_models, n_samples)
        
        # 加权投票
        if len(predictions.shape) == 2:
            # 类别预测：加权投票
            weighted_votes = np.zeros((len(X), 2))  # 假设二分类
            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                for j, class_label in enumerate([0, 1]):
                    weighted_votes[:, j] += weight * (pred == class_label)
            
            final_pred = np.argmax(weighted_votes, axis=1)
        else:
            # 概率预测：加权平均
            final_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return final_pred
    
    def update_weights_dawu(self, performance_scores: List[float],
                            msdi_score: float,
                            drift_confidence: float):
        """漂移感知权重更新 (DAWU)"""
        self.time_step += 1
        time_factor = np.exp(-self.dawu_lambda * self.time_step)
        drift_penalty = (1.0 - drift_confidence)
        msdi_penalty = (1.0 - min(1.0, msdi_score))

        updated_weights = []
        for score in performance_scores:
            combined = (
                self.dawu_alpha * score +
                self.dawu_beta * msdi_penalty +
                self.dawu_gamma * time_factor * drift_penalty
            )
            updated_weights.append(max(self.min_weight, combined))

        self.weights = np.array(updated_weights)
        self.weights = self.weights / (self.weights.sum() + 1e-10)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   drift_detections: Optional[List[bool]] = None,
                   msdi_score: Optional[float] = None,
                   drift_confidence: float = 0.0):
        """
        增量学习
        
        Args:
            X: 新数据特征
            y: 新数据标签
            drift_detections: 每个模型的漂移检测结果
        """
        # 更新每个基模型（使用不同的采样策略以增加多样性）
        for i, model in enumerate(self.base_models):
            try:
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X, y)
                elif hasattr(model, 'fit'):
                    # 如果没有partial_fit，使用少量数据微调
                    # 使用不同的随机种子为每个模型采样不同的数据，增加多样性
                    if len(X) > 100:
                        # Use different seed per model to ensure different samples
                        np.random.seed(42 + i * 1000)
                        sample_size = min(200, len(X))  # Increased from 100 to 200 for better training
                        sample_indices = np.random.choice(len(X), sample_size, replace=False)
                        X_sample = X[sample_indices]
                        y_sample = y[sample_indices]
                        # Use smaller learning rate for fine-tuning to avoid overfitting
                        if hasattr(model.optimizer, 'learning_rate'):
                            original_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate
                            model.optimizer.learning_rate.assign(original_lr * 0.1)  # 10% of original LR
                        model.fit(X_sample, y_sample, epochs=1, verbose=0, batch_size=min(32, len(X_sample)))
                        if hasattr(model.optimizer, 'learning_rate'):
                            model.optimizer.learning_rate.assign(original_lr)  # Restore original LR
            except Exception as e:
                print(f"Warning: Model {i} partial_fit failed: {e}")
        
        # 计算性能得分（基于更新后的模型）
        performance_scores = []
        for i, model in enumerate(self.base_models):
            predictions = None
            try:
                predictions = model.predict(X)
                if len(predictions.shape) > 1:
                    predictions = np.argmax(predictions, axis=1)
                accuracy = np.mean(predictions == y)
            except Exception:
                accuracy = 0.5
            self.performance_history[i].append(accuracy)
            # Use recent performance (last 10 batches) for more responsive weight updates
            recent_perf = list(self.performance_history[i])[-10:] if len(self.performance_history[i]) >= 10 else list(self.performance_history[i])
            performance = np.mean(recent_perf) if recent_perf else 0.5
            performance_scores.append(float(performance))

        # 更新权重（项目主线：仅使用 DAWU）
        self.update_weights_dawu(
            performance_scores,
            msdi_score if msdi_score is not None else 0.0,
            drift_confidence
        )
    
    def add_model(self, new_model, initial_weight: Optional[float] = None):
        """添加新模型到集成"""
        self.base_models.append(new_model)
        if initial_weight is None:
            initial_weight = 1.0 / (self.n_models + 1)
        
        # 重新分配权重
        self.weights = self.weights * (1 - initial_weight)
        self.weights = np.append(self.weights, initial_weight)
        self.n_models += 1
    
    def get_weights(self) -> np.ndarray:
        """获取当前权重"""
        return self.weights.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'n_models': self.n_models,
            'weights': self.weights.tolist(),
            'weight_update_method': self.weight_update_method,
            'average_performance': [
                np.mean(list(perf)) if perf else 0.0 
                for perf in self.performance_history
            ]
        }

