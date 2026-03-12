"""
Main Program - LSTM Network Intrusion Detection Model with Temporal Evaluation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from training.trainer import ModelTrainer
from data.data_processor import CICIDSDataProcessor

# 尝试导入集成训练器
try:
    from training.ensemble_trainer import EnsembleTrainer
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    EnsembleTrainer = None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LSTM Network Intrusion Detection Model')
    parser.add_argument('--mode', type=str, 
                       choices=['preprocess', 'train', 'evaluate_temporal', 'visualize', 'ensemble_train', 'ensemble_evaluate'],
                       default='train', help='Run mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--ensemble_type', type=str, 
                       choices=['dynamic', 'dwm', 'online_bagging'],
                       default='dynamic', help='Ensemble type for ensemble modes')
    parser.add_argument('--n_models', type=int, default=3, help='Number of base models in ensemble')

    args = parser.parse_args()

    # Create necessary directories
    Config.create_directories()

    print("=" * 60)
    print("LSTM Network Intrusion Detection Model")
    print("=" * 60)
    print(f"Run mode: {args.mode}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.mode == 'preprocess':
        preprocess_mode()
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate_temporal':
        evaluate_temporal_mode(args)
    elif args.mode == 'visualize':
        visualize_mode()
    elif args.mode == 'ensemble_train':
        ensemble_train_mode(args)
    elif args.mode == 'ensemble_evaluate':
        ensemble_evaluate_mode(args)

    print(f"\nCompletion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def preprocess_mode():
    """数据预处理模式"""
    print("开始数据预处理...")
    from preprocess_data import main as preprocess_main
    preprocess_main()

def visualize_mode():
    """数据可视化模式"""
    print("开始数据可视化...")
    from visualize_processed_data import main as visualize_main
    visualize_main()

def train_mode(args):
    """训练模式"""
    print("Starting training mode...")

    # 创建训练器
    trainer = ModelTrainer()

    # 准备训练数据（使用第一个文件作为训练集）
    print("准备训练数据...")
    processor = CICIDSDataProcessor()
    ordered_files = processor.get_ordered_csv_files(Config.CIC_IDS_DIR)
    
    if not ordered_files:
        print("错误: 未找到CSV文件")
        return
    
    # 使用第一个文件（Monday）作为训练集
    first_file = ordered_files[0]
    print(f"使用 {first_file} 作为训练数据...")
    
    # 加载预处理后的数据
    processed_dir = Config.PROCESSED_DATA_DIR
    first_file_name = os.path.splitext(first_file)[0]
    
    X_train_path = os.path.join(processed_dir, f'{first_file_name}_X.npy')
    y_train_path = os.path.join(processed_dir, f'{first_file_name}_y.npy')
    
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print(f"错误: 预处理数据不存在，请先运行 --mode preprocess")
        return
    
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    
    # 确保特征维度为78
    if X_train.shape[2] != 78:
        print(f"调整特征维度: {X_train.shape[2]} -> 78")
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

    # 训练模型
    print("\n开始模型训练...")
    history = trainer.train_basic_model(
        train_data, val_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # 保存模型
    model_path = trainer.save_model()
    print(f"\n模型已保存到: {model_path}")

    print("训练完成！")

def ensemble_train_mode(args):
    """集成训练模式 - 使用动态集成学习框架"""
    if not ENSEMBLE_AVAILABLE:
        print("错误: 集成训练器不可用，请检查drift模块是否正确安装")
        return
    
    print("=" * 60)
    print("动态集成学习框架 - 训练模式")
    print("=" * 60)
    print(f"集成类型: {args.ensemble_type}")
    print(f"基模型数量: {args.n_models}")
    print()
    
    # 创建集成训练器
    trainer = EnsembleTrainer(
        use_ensemble=True,
        ensemble_type=args.ensemble_type
    )
    
    # 准备训练数据
    print("准备训练数据...")
    processor = CICIDSDataProcessor()
    ordered_files = processor.get_ordered_csv_files(Config.CIC_IDS_DIR)
    
    if not ordered_files:
        print("错误: 未找到CSV文件")
        return
    
    # 使用Monday文件作为训练数据
    first_file = ordered_files[0]
    print(f"使用 {first_file} 作为训练数据...")
    
    processed_dir = Config.PROCESSED_DATA_DIR
    first_file_name = os.path.splitext(first_file)[0]
    
    X_train_path = os.path.join(processed_dir, f'{first_file_name}_X.npy')
    y_train_path = os.path.join(processed_dir, f'{first_file_name}_y.npy')
    
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print(f"错误: 预处理数据不存在，请先运行 --mode preprocess")
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
    
    # 创建基础模型
    print("\n创建基础模型...")
    trainer.create_base_model(input_shape=(Config.SEQUENCE_LENGTH, 78))
    
    # 设置集成学习系统
    print(f"\n设置集成学习系统 ({args.ensemble_type}, {args.n_models}个模型)...")
    trainer.setup_ensemble(n_models=args.n_models)
    
    # 训练初始模型
    print("\n开始训练初始模型...")
    trainer.train_initial_models(
        train_data, val_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # 保存集成模型
    print("\n保存集成模型...")
    trainer.save_ensemble()
    
    # 显示系统状态
    print("\n系统状态:")
    status = trainer.get_status()
    print(f"  使用集成: {status['use_ensemble']}")
    print(f"  集成类型: {status['ensemble_type']}")
    if 'ensemble_info' in status:
        print(f"  模型数量: {status['ensemble_info']['n_models']}")
        print(f"  模型权重: {[f'{w:.3f}' for w in status['ensemble_info']['weights']]}")
    
    print("\n集成训练完成！")

def ensemble_evaluate_mode(args):
    """集成评估模式 - 自适应评估和概念漂移检测"""
    if not ENSEMBLE_AVAILABLE:
        print("错误: 集成训练器不可用，请检查drift模块是否正确安装")
        return
    
    print("=" * 60)
    print("动态集成学习框架 - 自适应评估模式")
    print("=" * 60)
    
    # 创建集成训练器
    trainer = EnsembleTrainer(
        use_ensemble=True,
        ensemble_type=args.ensemble_type
    )
    
    # 加载模型（查找最新的集成模型）
    model_files = [f for f in os.listdir(Config.MODEL_DIR) 
                   if f.startswith('ensemble_base_') and f.endswith('.h5')]
    if not model_files:
        print("错误: 未找到训练好的集成模型，请先运行 --mode ensemble_train")
        return
    
    latest_model = max(model_files, key=lambda f: os.path.getmtime(
        os.path.join(Config.MODEL_DIR, f)))
    model_path = os.path.join(Config.MODEL_DIR, latest_model)
    
    print(f"加载模型: {latest_model}")
    
    # 加载基础模型
    trainer.create_base_model()
    trainer.base_model.load_model(model_path)
    
    # 尝试加载集成信息（如果存在）
    info_files = [f for f in os.listdir(Config.MODEL_DIR) 
                  if f.startswith('ensemble_info_') and f.endswith('.json')]
    if info_files:
        # 使用最新的集成信息文件
        latest_info = max(info_files, key=lambda f: os.path.getmtime(
            os.path.join(Config.MODEL_DIR, f)))
        info_path = os.path.join(Config.MODEL_DIR, latest_info)
        print(f"加载集成信息: {latest_info}")
        
        try:
            import json
            with open(info_path, 'r', encoding='utf-8') as f:
                ensemble_info = json.load(f)
            n_models_from_info = ensemble_info.get('n_models', args.n_models)
            print(f"从集成信息中读取模型数量: {n_models_from_info}")
            args.n_models = n_models_from_info
        except Exception as e:
            print(f"警告: 无法加载集成信息，使用默认参数: {e}")
    
    # 设置集成系统
    trainer.setup_ensemble(n_models=args.n_models)
    
    # 获取按时间顺序排列的文件列表
    processor = CICIDSDataProcessor()
    ordered_files = processor.get_ordered_csv_files(Config.CIC_IDS_DIR)
    
    if not ordered_files:
        print("错误: 未找到CSV文件")
        return
    
    # 加载训练时的参考数据（Monday数据）用于漂移检测
    processed_dir = Config.PROCESSED_DATA_DIR
    first_file = ordered_files[0]
    first_file_name = os.path.splitext(first_file)[0]
    X_train_path = os.path.join(processed_dir, f'{first_file_name}_X.npy')
    y_train_path = os.path.join(processed_dir, f'{first_file_name}_y.npy')
    
    if os.path.exists(X_train_path) and os.path.exists(y_train_path):
        X_ref = np.load(X_train_path)
        y_ref = np.load(y_train_path)
        if X_ref.shape[2] != 78:
            X_ref = X_ref[:, :, :78]
        
        # 设置参考数据到漂移检测器
        if trainer.drift_detector:
            trainer.drift_detector.add_reference_data(X_ref, y_ref)
            print(f"已设置参考数据（{first_file_name}）用于漂移检测")
    
    print(f"\n将按时间顺序进行自适应评估...")
    print("文件顺序:", [os.path.splitext(f)[0] for f in ordered_files])
    print()
    results_summary = []
    
    # 跳过第一个文件（已用于训练），评估后续文件
    for idx, file_name in enumerate(ordered_files[1:], 1):
        print(f"\n[{idx}/{len(ordered_files)-1}] 评估文件: {file_name}")
        
        file_name_base = os.path.splitext(file_name)[0]
        X_test_path = os.path.join(processed_dir, f'{file_name_base}_X.npy')
        y_test_path = os.path.join(processed_dir, f'{file_name_base}_y.npy')
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            print(f"  警告: 预处理数据不存在，跳过")
            continue
        
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        if X_test.shape[2] != 78:
            X_test = X_test[:, :, :78]
        
        # 自适应评估
        print("  进行自适应评估（检测概念漂移并自动调整）...")
        results = trainer.adaptive_evaluation(
            (X_test, y_test),
            batch_size=5000
        )
        
        # 统计结果
        avg_accuracy = np.mean(results['batch_accuracies'])
        drift_count = sum(results['drift_detections'])
        total_batches = len(results['batch_accuracies'])
        
        results_summary.append({
            'file': file_name_base,
            'accuracy': avg_accuracy,
            'drift_count': drift_count,
            'total_batches': total_batches,
            'drift_ratio': drift_count / total_batches if total_batches > 0 else 0.0
        })
        
        print(f"  平均准确率: {avg_accuracy:.4f}")
        print(f"  检测到漂移: {drift_count}/{total_batches} 批次 ({drift_count/total_batches*100:.1f}%)")
        
        if results['ensemble_weights']:
            final_weights = results['ensemble_weights'][-1]
            print(f"  最终集成权重: {[f'{w:.3f}' for w in final_weights]}")
    
    # 打印汇总结果
    print("\n" + "=" * 60)
    print("自适应评估结果汇总")
    print("=" * 60)
    print(f"{'文件':<50} {'准确率':<10} {'漂移检测':<15}")
    print("-" * 75)
    for r in results_summary:
        print(f"{r['file']:<50} {r['accuracy']:<10.4f} {r['drift_count']}/{r['total_batches']} ({r['drift_ratio']*100:.1f}%)")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.RESULTS_DIR, f"ensemble_evaluation_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_file}")
    
    print("\n自适应评估完成！")

def evaluate_temporal_mode(args):
    """按时间顺序评估模式 - 观察模型性能随时间推移的变化"""
    print("开始按时间顺序评估模型性能...")
    
    # 加载模型
    trainer = ModelTrainer()
    
    # 查找最新的模型文件
    model_files = [f for f in os.listdir(Config.MODEL_DIR) if f.endswith('.h5')]
    if not model_files:
        print("错误: 未找到训练好的模型，请先运行 --mode train")
        return
    
    # 使用最新的模型
    latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(Config.MODEL_DIR, f)))
    model_path = os.path.join(Config.MODEL_DIR, latest_model)
    
    print(f"加载模型: {model_path}")
    trainer.load_model(model_path)
    
    # 获取按时间顺序排列的文件列表
    processor = CICIDSDataProcessor()
    ordered_files = processor.get_ordered_csv_files(Config.CIC_IDS_DIR)
    
    if not ordered_files:
        print("错误: 未找到CSV文件")
        return
    
    print(f"\n将按时间顺序评估 {len(ordered_files)} 个测试集...")
    print("文件顺序:", [os.path.splitext(f)[0] for f in ordered_files])
    print()
    
    # 存储每个测试集的性能指标
    results = {
        'file_names': [],
        'accuracies': [],
        'f1_scores': [],
        'details': []
    }
    
    processed_dir = Config.PROCESSED_DATA_DIR
    
    for idx, file_name in enumerate(ordered_files, 1):
        print(f"[{idx}/{len(ordered_files)}] 评估文件: {file_name}")
        
        # 加载预处理后的数据
        file_name_base = os.path.splitext(file_name)[0]
        X_test_path = os.path.join(processed_dir, f'{file_name_base}_X.npy')
        y_test_path = os.path.join(processed_dir, f'{file_name_base}_y.npy')
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            print(f"  警告: 预处理数据不存在，跳过 {file_name}")
            continue
        
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        # 确保特征维度为78
        if X_test.shape[2] != 78:
            X_test = X_test[:, :, :78]
        
        # 评估模型
        test_data = (X_test, y_test)
        evaluation_results = trainer.evaluate_model(test_data, save_results=False)
        
        # 提取指标
        accuracy = evaluation_results['accuracy']
        f1 = evaluation_results['classification_report'].get('1', {}).get('f1-score', 0.0)
        if f1 == 0.0:
            # 如果没有类别1，尝试使用加权平均
            f1 = evaluation_results['classification_report'].get('weighted avg', {}).get('f1-score', 0.0)
        
        results['file_names'].append(file_name_base)
        results['accuracies'].append(accuracy)
        results['f1_scores'].append(f1)
        results['details'].append(evaluation_results)
        
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print()
    
    # 打印性能变化趋势
    print("=" * 60)
    print("模型性能随时间推移的变化:")
    print("=" * 60)
    print(f"{'文件':<50} {'准确率':<10} {'F1-Score':<10}")
    print("-" * 70)
    for i, (name, acc, f1) in enumerate(zip(results['file_names'], results['accuracies'], results['f1_scores'])):
        print(f"{name:<50} {acc:<10.4f} {f1:<10.4f}")
    
    # 绘制性能变化图
    plot_temporal_performance(results)
    
    # 保存结果
    save_temporal_results(results)
    
    print("\n时间序列评估完成！")

def plot_temporal_performance(results):
    """绘制性能随时间变化的图表"""
    if len(results['file_names']) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x_pos = range(len(results['file_names']))
    file_labels = [name.split('.')[0] for name in results['file_names']]
    
    # 准确率变化
    ax1.plot(x_pos, results['accuracies'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('测试集（按时间顺序）', fontsize=12, fontfamily='sans-serif')
    ax1.set_ylabel('准确率 (Accuracy)', fontsize=12, fontfamily='sans-serif')
    ax1.set_title('模型准确率随时间推移的变化', fontsize=14, fontweight='bold', pad=15, fontfamily='sans-serif')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(file_labels, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 添加数值标签
    for i, acc in enumerate(results['accuracies']):
        ax1.annotate(f'{acc:.3f}', (i, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # F1-Score变化
    ax2.plot(x_pos, results['f1_scores'], 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('测试集（按时间顺序）', fontsize=12, fontfamily='sans-serif')
    ax2.set_ylabel('F1-Score', fontsize=12, fontfamily='sans-serif')
    ax2.set_title('模型F1-Score随时间推移的变化', fontsize=14, fontweight='bold', pad=15, fontfamily='sans-serif')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(file_labels, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 添加数值标签
    for i, f1 in enumerate(results['f1_scores']):
        ax2.annotate(f'{f1:.3f}', (i, f1), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # 调整布局，确保标题和标签不被遮挡
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(Config.RESULTS_DIR, f"temporal_performance_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n性能变化图表已保存到: {save_path}")
    plt.close()

def save_temporal_results(results):
    """保存时间序列评估结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.RESULTS_DIR, f"temporal_results_{timestamp}.txt")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("模型性能随时间推移的变化 - 详细报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"{'文件':<50} {'准确率':<12} {'F1-Score':<12}\n")
        f.write("-" * 70 + "\n")
        
        for name, acc, f1 in zip(results['file_names'], results['accuracies'], results['f1_scores']):
            f.write(f"{name:<50} {acc:<12.4f} {f1:<12.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("详细分类报告\n")
        f.write("=" * 70 + "\n\n")
        
        for i, (name, detail) in enumerate(zip(results['file_names'], results['details'])):
            f.write(f"\n文件 {i+1}: {name}\n")
            f.write("-" * 70 + "\n")
            if 'classification_report' in detail:
                report = detail['classification_report']
                f.write(f"准确率: {detail['accuracy']:.4f}\n")
                f.write(f"AUC分数: {detail.get('auc_score', 'N/A')}\n")
                f.write("\n分类报告:\n")
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                        f.write(f"  类别 {class_name}: "
                               f"精确率={metrics.get('precision', 0):.4f}, "
                               f"召回率={metrics.get('recall', 0):.4f}, "
                               f"F1={metrics.get('f1-score', 0):.4f}\n")
    
    print(f"详细结果已保存到: {results_file}")

if __name__ == "__main__":
    main()
