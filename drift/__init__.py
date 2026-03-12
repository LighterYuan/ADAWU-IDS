"""
概念漂移检测和动态集成学习模块
Concept Drift Detection and Dynamic Ensemble Learning Module
"""

try:
    from .concept_drift_detector import ConceptDriftDetector, MultiScaleDriftIndex
    from .dynamic_ensemble import DynamicEnsemble
    from .adaptive_learning import AdaptiveLearningSystem

    __all__ = [
        'ConceptDriftDetector',
        'MultiScaleDriftIndex',
        'DynamicEnsemble',
        'AdaptiveLearningSystem'
    ]
except ImportError as e:
    print(f"Warning: Failed to import drift modules: {e}")
    __all__ = []
