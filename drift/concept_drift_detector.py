"""
概念漂移检测模块（精简版，仅保留 MSDI + 综合检测器）
Concept Drift Detection Module (Simplified: MSDI + Aggregated Detector)
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from typing import Tuple, Optional, Dict, Any
import warnings
from config import Config


class MultiScaleDriftIndex:
    """
    多尺度漂移指数 (MSDI)
    结合特征组、类别和整体统计，输出 [0,1] 范围的漂移强度
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.feature_groups = self._build_feature_groups()

    def _build_feature_groups(self):
        groups = {
            'flow_stats': [
                ' Flow Duration', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                'Flow Bytes/s', ' Flow Packets/s'
            ],
            'packet_size': [
                ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
                ' Min Packet Length', ' Max Packet Length', ' Average Packet Size'
            ],
            'directional_packets': [
                ' Total Fwd Packets', ' Total Backward Packets', 'Fwd Packets/s', ' Bwd Packets/s',
                ' Total Length of Fwd Packets', ' Total Length of Bwd Packets'
            ],
            'flags': [
                'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
                ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count'
            ],
            'temporal_behavior': [
                'Active Mean', ' Active Std', ' Active Max', ' Active Min',
                'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
            ]
        }

        # 过滤不存在的特征
        filtered = {}
        for group, features in groups.items():
            filtered[group] = [f for f in features if f in self.feature_names]
        return filtered

    def compute(self, X_ref: np.ndarray, X_new: np.ndarray,
                y_ref: Optional[np.ndarray] = None,
                y_new: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算MSDI
        """
        ref_flat = X_ref.reshape(-1, X_ref.shape[-1])
        new_flat = X_new.reshape(-1, X_new.shape[-1])

        group_scores = {}
        feature_scores = []

        for group, features in self.feature_groups.items():
            if not features:
                continue
            distances = []
            for feature in features:
                idx = self.feature_names.index(feature)
                ref_feature = ref_flat[:, idx]
                new_feature = new_flat[:, idx]

                # 采样以降低计算量
                ref_sample = self._sample(ref_feature)
                new_sample = self._sample(new_feature)

                dist = wasserstein_distance(ref_sample, new_sample)
                norm = np.std(ref_sample) + 1e-6
                normalized_dist = min(1.0, dist / (norm + 1e-6))
                distances.append(normalized_dist)
                feature_scores.append(normalized_dist)

            group_scores[group] = float(np.mean(distances)) if distances else 0.0

        # 类别层面的漂移（正常/异常）
        class_scores = {}
        if y_ref is not None and y_new is not None:
            for label in [0, 1]:
                mask_ref = y_ref == label
                mask_new = y_new == label
                if mask_ref.sum() > 100 and mask_new.sum() > 100:
                    ref_subset = X_ref[mask_ref]
                    new_subset = X_new[mask_new]
                    ref_vec = ref_subset.reshape(-1, ref_subset.shape[-1])
                    new_vec = new_subset.reshape(-1, new_subset.shape[-1])
                    dist = np.linalg.norm(
                        np.mean(ref_vec, axis=0) - np.mean(new_vec, axis=0)
                    )
                    norm = np.linalg.norm(np.mean(ref_vec, axis=0)) + 1e-6
                    class_scores[f'class_{label}'] = min(1.0, dist / (norm + 1e-6))

        msdi_score = float(np.mean(feature_scores)) if feature_scores else 0.0

        return {
            'msdi_score': msdi_score,
            'group_scores': group_scores,
            'class_scores': class_scores
        }

    @staticmethod
    def _sample(arr, max_samples=2000):
        if len(arr) <= max_samples:
            return arr
        indices = np.random.choice(len(arr), max_samples, replace=False)
        return arr[indices]


class ConceptDriftDetector:
    """
    基础概念漂移检测器
    Base Concept Drift Detector
    """
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.05, 
                 adaptation_rate: float = 0.01):
        """
        Args:
            window_size: 检测窗口大小
            threshold: 漂移阈值
            adaptation_rate: 适应率
        """
        self.window_size = window_size
        self.threshold = threshold
        self.adaptation_rate = adaptation_rate
        self.reference_data = None
        self.msdi = MultiScaleDriftIndex(Config.FEATURE_COLUMNS)
        self.drift_history = []
        
    def add_reference_data(self, X: np.ndarray, y: np.ndarray):
        """添加参考数据"""
        self.reference_data = (X, y)
        
    def detect_drift(self, X_new: np.ndarray, y_new: np.ndarray, 
                    model=None) -> Dict[str, Any]:
        """
        检测概念漂移
        
        Returns:
            dict: 包含is_drift, confidence, method等信息
        """
        if self.reference_data is None:
            warnings.warn("No reference data provided")
            return {'is_drift': False, 'confidence': 0.0, 'method': 'none'}
        
        # 使用多种方法检测
        results = {
            'is_drift': False,
            'confidence': 0.0,
            'method': 'ensemble',
            'details': {}
        }
        
        # 1. 基于准确率的检测
        if model is not None:
            acc_result = self._detect_by_accuracy(X_new, y_new, model)
            results['details']['accuracy_based'] = acc_result
        
        # 2. 基于特征分布的检测
        dist_result = self._detect_by_distribution(X_new)
        results['details']['distribution_based'] = dist_result

        # 3. 多尺度漂移指数
        try:
            X_ref, y_ref = self.reference_data
            msdi_result = self.msdi.compute(X_ref, X_new, y_ref=y_ref, y_new=y_new)
        except Exception as e:
            warnings.warn(f"MSDI computation failed: {e}")
            msdi_result = {'msdi_score': 0.0, 'group_scores': {}, 'class_scores': {}}

        results['details']['msdi'] = msdi_result
        results['msdi_score'] = msdi_result.get('msdi_score', 0.0)
        
        # 3. 集成判断
        drift_votes = 0
        total_confidence = 0.0
        
        if 'accuracy_based' in results['details']:
            if results['details']['accuracy_based']['is_drift']:
                drift_votes += 1
            total_confidence += results['details']['accuracy_based']['confidence']
        
        if 'distribution_based' in results['details']:
            if results['details']['distribution_based']['is_drift']:
                drift_votes += 1
            total_confidence += results['details']['distribution_based']['confidence']

        # MSDI加入投票
        msdi_score = results['details']['msdi'].get('msdi_score', 0.0)
        if msdi_score > self.threshold * 1.5:  # 漂移强度超过阈值的1.5倍
            drift_votes += 1
            total_confidence += min(1.0, msdi_score)
        
        # 投票机制：至少一种方法检测到漂移
        results['is_drift'] = drift_votes >= 1
        results['confidence'] = total_confidence / max(len(results['details']), 1)
        
        return results
    
    def _detect_by_accuracy(self, X_new: np.ndarray, y_new: np.ndarray, 
                           model) -> Dict[str, Any]:
        """基于准确率的漂移检测"""
        try:
            y_pred = model.predict(X_new)
            y_pred_classes = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred
            accuracy = np.mean(y_pred_classes == y_new)
            
            # 如果准确率显著下降，认为发生漂移
            is_drift = accuracy < (1.0 - self.threshold)
            confidence = max(0.0, 1.0 - accuracy)
            
            return {
                'is_drift': is_drift,
                'confidence': confidence,
                'accuracy': float(accuracy)
            }
        except Exception as e:
            warnings.warn(f"Accuracy-based detection failed: {e}")
            return {'is_drift': False, 'confidence': 0.0, 'accuracy': 0.0}
    
    def _detect_by_distribution(self, X_new: np.ndarray) -> Dict[str, Any]:
        """基于特征分布的漂移检测（KS检验）"""
        if self.reference_data is None:
            return {'is_drift': False, 'confidence': 0.0}
        
        X_ref = self.reference_data[0]
        
        # 计算特征均值
        ref_means = X_ref.reshape(-1, X_ref.shape[-1]).mean(axis=0)
        new_means = X_new.reshape(-1, X_new.shape[-1]).mean(axis=0)
        
        # KS检验
        try:
            # 对每个特征进行KS检验
            p_values = []
            for i in range(min(len(ref_means), len(new_means))):
                ref_feature = X_ref.reshape(-1, X_ref.shape[-1])[:, i]
                new_feature = X_new.reshape(-1, X_new.shape[-1])[:, i]
                
                # 采样以避免计算量过大
                if len(ref_feature) > 1000:
                    ref_feature = np.random.choice(ref_feature, 1000, replace=False)
                if len(new_feature) > 1000:
                    new_feature = np.random.choice(new_feature, 1000, replace=False)
                
                statistic, p_value = stats.ks_2samp(ref_feature, new_feature)
                p_values.append(p_value)
            
            # 如果大部分特征的p值小于阈值，认为发生漂移
            significant_changes = sum(1 for p in p_values if p < self.threshold)
            drift_ratio = significant_changes / len(p_values) if p_values else 0.0
            
            is_drift = drift_ratio > 0.3  # 30%以上特征显著变化
            confidence = drift_ratio
            
            return {
                'is_drift': is_drift,
                'confidence': confidence,
                'drift_ratio': drift_ratio,
                'mean_p_value': np.mean(p_values) if p_values else 1.0
            }
        except Exception as e:
            warnings.warn(f"Distribution-based detection failed: {e}")
            return {'is_drift': False, 'confidence': 0.0}



