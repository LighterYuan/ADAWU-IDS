"""
动态集成学习框架使用示例
Dynamic Ensemble Learning Framework Usage Example
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from training.ensemble_trainer import EnsembleTrainer
from data.data_processor import CICIDSDataProcessor


def main():
    """主函数 - 演示动态集成学习框架的使用"""
    
    print("=" * 60)
    print("动态集成学习框架 - 概念漂移检测示例")
    print("=" * 60)
    
    # 1. 创建集成训练器
    print("\n1. 创建集成训练器...")
    trainer = EnsembleTrainer(
        use_ensemble=True,
        ensemble_type='dynamic'  # 可选: 'dynamic', 'dwm', 'online_bagging'
    )
    
    # 2. 加载数据
    print("\n2. 加载预处理数据...")
    processor = CICIDSDataProcessor()
    processed_dir = Config.PROCESSED_DATA_DIR
    
    # 使用Monday文件作为训练数据
    monday_file = 'Monday-WorkingHours.pcap_ISCX'
    X_train_path = os.path.join(processed_dir, f'{monday_file}_X.npy')
    y_train_path = os.path.join(processed_dir, f'{monday_file}_y.npy')
    
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print("错误: 预处理数据不存在，请先运行 python main.py --mode preprocess")
        return
    
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    
    # 确保特征维度为78
    if X_train.shape[2] != 78:
        X_train = X_train[:, :, :78]
    
    # 分割训练和验证集
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    train_data = (X_train_split, y_train_split)
    val_data = (X_val_split, y_val_split)
    
    print(f"训练数据形状: {train_data[0].shape}")
    print(f"验证数据形状: {val_data[0].shape}")
    
    # 3. 创建基础模型
    print("\n3. 创建基础模型...")
    trainer.create_base_model(input_shape=(Config.SEQUENCE_LENGTH, 78))
    
    # 4. 设置集成学习系统
    print("\n4. 设置集成学习系统...")
    trainer.setup_ensemble(n_models=3, drift_detector_type='ensemble')
    
    # 5. 训练初始模型
    print("\n5. 训练初始模型...")
    trainer.train_initial_models(
        train_data, val_data,
        epochs=20,  # 减少epoch数以加快演示
        batch_size=Config.BATCH_SIZE
    )
    
    # 6. 自适应评估 - 按时间顺序评估后续文件
    print("\n6. 自适应评估 - 检测概念漂移...")
    ordered_files = processor.get_ordered_csv_files(Config.CIC_IDS_DIR)
    
    # 跳过Monday（已用于训练），评估其他文件
    for file_name in ordered_files[1:4]:  # 只评估前几个文件作为演示
        print(f"\n评估文件: {file_name}")
        file_name_base = os.path.splitext(file_name)[0]
        
        X_test_path = os.path.join(processed_dir, f'{file_name_base}_X.npy')
        y_test_path = os.path.join(processed_dir, f'{file_name_base}_y.npy')
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            print(f"  跳过: 预处理数据不存在")
            continue
        
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        if X_test.shape[2] != 78:
            X_test = X_test[:, :, :78]
        
        # 自适应评估
        results = trainer.adaptive_evaluation(
            (X_test, y_test),
            batch_size=1000
        )
        
        # 显示结果
        avg_accuracy = np.mean(results['batch_accuracies'])
        drift_count = sum(results['drift_detections'])
        
        print(f"  平均准确率: {avg_accuracy:.4f}")
        print(f"  检测到漂移次数: {drift_count}/{len(results['drift_detections'])}")
        
        if results['ensemble_weights']:
            final_weights = results['ensemble_weights'][-1]
            print(f"  最终集成权重: {[f'{w:.3f}' for w in final_weights]}")
    
    # 7. 保存模型
    print("\n7. 保存集成模型...")
    trainer.save_ensemble()
    
    # 8. 显示系统状态
    print("\n8. 系统状态:")
    status = trainer.get_status()
    print(f"  使用集成: {status['use_ensemble']}")
    print(f"  集成类型: {status['ensemble_type']}")
    if 'ensemble_info' in status:
        print(f"  模型数量: {status['ensemble_info']['n_models']}")
        print(f"  模型权重: {status['ensemble_info']['weights']}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

