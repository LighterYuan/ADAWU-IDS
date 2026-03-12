"""
配置文件 - 网络入侵检测模型参数设置
"""

import os

class Config:
    """模型配置类"""

    # 数据相关配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CIC_IDS_DIR = os.path.join(DATA_DIR, 'cicids2017')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'cicids2017')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')  # 预处理后的数据保存目录
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # 模型相关配置
    SEQUENCE_LENGTH = 10  # LSTM输入序列长度
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 0.002
    DROPOUT_RATE = 0.3
    
    # LSTM模型参数
    LSTM_UNITS = 128
    DENSE_UNITS = 64
    NUM_CLASSES = 2  # 正常/异常
    
    # 概念漂移检测参数
    DRIFT_WINDOW_SIZE = 1000  # 漂移检测窗口大小
    DRIFT_THRESHOLD = 0.05    # 漂移阈值
    ADAPTATION_RATE = 0.01    # 模型适应率

    # 动态集成学习参数（项目主线：DynamicEnsemble + DAWU）
    USE_ENSEMBLE = True           # 是否使用集成学习
    ENSEMBLE_TYPE = 'dynamic'     # 固定为 dynamic 集成
    WEIGHT_UPDATE_METHOD = 'dawu' # 固定使用 DAWU 漂移感知权重更新
    MIN_MODEL_WEIGHT = 0.01       # 最小模型权重阈值
    WEIGHT_DECAY_FACTOR = 0.9     # 权重衰减因子

    # DAWU（Drift-Aware Weight Update）参数
    DAWU_ALPHA = 0.6              # 性能权重系数
    DAWU_BETA = 0.3               # 漂移抑制系数
    DAWU_GAMMA = 0.1              # 时间衰减系数
    DAWU_LAMBDA = 0.05            # 时间衰减速度
    
    # 特征选择
    FEATURE_COLUMNS = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
        'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
        ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max',
        ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
        ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total',
        ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
        ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags',
        ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
        ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
        ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
        ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
        ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length',
        'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
        ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
        ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
        ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min',
        'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR, cls.CIC_IDS_DIR, cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR, cls.RESULTS_DIR, cls.LOGS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

