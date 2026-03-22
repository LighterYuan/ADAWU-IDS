"""
自适应学习系统 - 概念漂移检测 + 动态集成 (DynamicEnsemble + DAWU)
Adaptive Learning System - Concept Drift Detection + Dynamic Ensemble (DAWU)
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from .concept_drift_detector import ConceptDriftDetector
from .dynamic_ensemble import DynamicEnsemble
import copy


class AdaptiveLearningSystem:
    """
    自适应学习系统
    结合概念漂移检测和动态集成学习
    """
    
    def __init__(self, base_model, drift_detector: ConceptDriftDetector,
                 adaptation_rate: float = 0.01,
                 use_ensemble: bool = True,
                 ensemble: Optional[DynamicEnsemble] = None):
        """
        Args:
            base_model: 基础模型
            drift_detector: 概念漂移检测器
            adaptation_rate: 适应率
            use_ensemble: 是否使用集成学习（使用 DynamicEnsemble + DAWU）
            ensemble: 外部提供的集成系统（如果为None且use_ensemble=True，则创建新的）
        """
        self.base_model = base_model
        self.drift_detector = drift_detector
        self.adaptation_rate = adaptation_rate
        
        # 创建集成（如果需要），使用 DynamicEnsemble + DAWU 作为唯一主线
        self.use_ensemble = use_ensemble
        if use_ensemble:
            if ensemble is not None:
                # 使用外部提供的集成系统
                self.ensemble = ensemble
            else:
                # 创建新的集成系统
                self.ensemble = DynamicEnsemble(
                    [copy.deepcopy(base_model)],
                    weight_update_method='dawu'
                )
        else:
            self.ensemble = None
        
        self.drift_history = []
        self.adaptation_history = []
        
    def adaptive_learning_pipeline(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """
        自适应学习流水线
        
        Args:
            X_new: 新数据特征
            y_new: 新数据标签
            
        Returns:
            适应结果字典
        """
        # 1. 检测概念漂移
        drift_result = self.drift_detector.detect_drift(X_new, y_new, self.base_model)
        self.drift_history.append(drift_result)
        msdi_info = drift_result.get('details', {}).get('msdi', {})
        msdi_score = msdi_info.get('msdi_score', 0.0)
        
        result = {
            'drift_detected': drift_result['is_drift'],
            'drift_confidence': drift_result.get('confidence', 0.0),
            'adaptation_strategy': None,
            'ensemble_updated': False,
            'msdi_score': msdi_score,
            'msdi_groups': msdi_info.get('group_scores', {})
        }
        
        # 2. 根据漂移检测结果选择适应策略
        if drift_result['is_drift']:
            confidence = drift_result.get('confidence', 0.0)
            
            if confidence > 0.7:
                # 严重漂移：创建新模型
                result['adaptation_strategy'] = 'create_new_model'
                self._create_new_model(X_new, y_new)
            elif confidence > 0.4:
                # 中等漂移：渐进式适应
                result['adaptation_strategy'] = 'progressive_adaptation'
                self._progressive_adaptation(X_new, y_new)
            else:
                # 轻微漂移：微调
                result['adaptation_strategy'] = 'fine_tuning'
                self._fine_tuning(X_new, y_new)
        else:
            # 无漂移：正常更新
            result['adaptation_strategy'] = 'normal_update'
            self._normal_update(X_new, y_new)
        
        # 3. 更新集成（如果使用）
        if self.use_ensemble and self.ensemble:
            drift_detections = [drift_result['is_drift']]
            self.ensemble.partial_fit(
                X_new,
                y_new,
                drift_detections=drift_detections,
                msdi_score=msdi_score,
                drift_confidence=drift_result.get('confidence', 0.0)
            )
            result['ensemble_updated'] = True
            result['ensemble_weights'] = self.ensemble.get_weights().tolist()
        
        self.adaptation_history.append(result)
        return result
    
    def _create_new_model(self, X_new: np.ndarray, y_new: np.ndarray):
        """创建新模型（严重漂移时）"""
        # 创建新模型副本
        new_model = copy.deepcopy(self.base_model)
        
        # 在新数据上训练
        try:
            if hasattr(new_model, 'fit'):
                # 使用较小学习率
                original_lr = None
                if hasattr(new_model.model, 'optimizer'):
                    original_lr = new_model.model.optimizer.learning_rate.numpy()
                    new_model.model.optimizer.learning_rate = self.adaptation_rate
                
                new_model.fit(X_new, y_new, epochs=5, verbose=0)
                
                if original_lr is not None:
                    new_model.model.optimizer.learning_rate = original_lr
        except Exception as e:
            print(f"Warning: Failed to create new model: {e}")
        
        # 添加到集成
        if self.use_ensemble and self.ensemble:
            self.ensemble.add_model(new_model, initial_weight=0.3)
        else:
            # 如果不使用集成，直接替换
            self.base_model = new_model
    
    def _progressive_adaptation(self, X_new: np.ndarray, y_new: np.ndarray):
        """渐进式适应（中等漂移时）"""
        try:
            if hasattr(self.base_model, 'adaptive_update'):
                self.base_model.adaptive_update(X_new, y_new, 
                                              learning_rate=self.adaptation_rate)
            elif hasattr(self.base_model, 'fit'):
                # 使用较小学习率和少量epoch
                original_lr = None
                if hasattr(self.base_model.model, 'optimizer'):
                    original_lr = self.base_model.model.optimizer.learning_rate.numpy()
                    self.base_model.model.optimizer.learning_rate = self.adaptation_rate
                
                self.base_model.fit(X_new, y_new, epochs=3, verbose=0)
                
                if original_lr is not None:
                    self.base_model.model.optimizer.learning_rate = original_lr
        except Exception as e:
            print(f"Warning: Progressive adaptation failed: {e}")
    
    def _fine_tuning(self, X_new: np.ndarray, y_new: np.ndarray):
        """微调（轻微漂移时）"""
        try:
            if hasattr(self.base_model, 'adaptive_update'):
                self.base_model.adaptive_update(X_new, y_new, 
                                              learning_rate=self.adaptation_rate * 0.5)
            elif hasattr(self.base_model, 'fit'):
                original_lr = None
                if hasattr(self.base_model.model, 'optimizer'):
                    original_lr = self.base_model.model.optimizer.learning_rate.numpy()
                    self.base_model.model.optimizer.learning_rate = self.adaptation_rate * 0.5
                
                self.base_model.fit(X_new, y_new, epochs=1, verbose=0)
                
                if original_lr is not None:
                    self.base_model.model.optimizer.learning_rate = original_lr
        except Exception as e:
            print(f"Warning: Fine-tuning failed: {e}")
    
    def _normal_update(self, X_new: np.ndarray, y_new: np.ndarray):
        """正常更新（无漂移时）"""
        try:
            if hasattr(self.base_model, 'partial_fit'):
                self.base_model.partial_fit(X_new, y_new)
            elif hasattr(self.base_model, 'fit'):
                # 少量数据微调
                if len(X_new) > 50:
                    sample_indices = np.random.choice(len(X_new), 50, replace=False)
                    X_sample = X_new[sample_indices]
                    y_sample = y_new[sample_indices]
                    
                    original_lr = None
                    if hasattr(self.base_model.model, 'optimizer'):
                        original_lr = self.base_model.model.optimizer.learning_rate.numpy()
                        self.base_model.model.optimizer.learning_rate = self.adaptation_rate * 0.1
                    
                    self.base_model.fit(X_sample, y_sample, epochs=1, verbose=0)
                    
                    if original_lr is not None:
                        self.base_model.model.optimizer.learning_rate = original_lr
        except Exception as e:
            print(f"Warning: Normal update failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.use_ensemble and self.ensemble:
            return self.ensemble.predict(X)
        else:
            return self.base_model.predict(X)
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'drift_detections': len([h for h in self.drift_history if h['is_drift']]),
            'total_updates': len(self.adaptation_history),
            'use_ensemble': self.use_ensemble
        }
        
        if self.use_ensemble and self.ensemble:
            status['ensemble_info'] = self.ensemble.get_model_info()
        
        return status

