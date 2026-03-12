"""
可视化预处理后的数据文件
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from data.data_processor import CICIDSDataProcessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def visualize_single_file(file_name_base, processed_dir):
    """可视化单个预处理文件"""
    X_path = os.path.join(processed_dir, f'{file_name_base}_X.npy')
    y_path = os.path.join(processed_dir, f'{file_name_base}_y.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"文件不存在: {file_name_base}")
        return None
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"\n{'='*60}")
    print(f"文件: {file_name_base}")
    print(f"{'='*60}")
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"特征维度: {X.shape[2]}")
    print(f"序列长度: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    # 标签分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n标签分布:")
    for label, count in zip(unique, counts):
        label_name = "正常" if label == 0 else "异常"
        percentage = count / len(y) * 100
        print(f"  {label_name} (类别 {label}): {count:,} ({percentage:.2f}%)")
    
    # 特征统计
    X_flat = X.reshape(-1, X.shape[-1])
    print(f"\n特征统计信息:")
    print(f"  均值范围: [{X_flat.mean(axis=0).min():.4f}, {X_flat.mean(axis=0).max():.4f}]")
    print(f"  标准差范围: [{X_flat.std(axis=0).min():.4f}, {X_flat.std(axis=0).max():.4f}]")
    print(f"  最小值: {X_flat.min():.4f}")
    print(f"  最大值: {X_flat.max():.4f}")
    
    return X, y

def plot_data_distribution(X, y, file_name_base, save_dir):
    """绘制数据分布图"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 标签分布饼图
    ax1 = plt.subplot(2, 3, 1)
    unique, counts = np.unique(y, return_counts=True)
    labels = ['正常', '异常'] if len(unique) == 2 else [f'类别{i}' for i in unique]
    colors = ['#66b3ff', '#ff9999'] if len(unique) == 2 else None
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f'{file_name_base}\n标签分布', fontsize=12, fontweight='bold')
    
    # 2. 标签分布柱状图
    ax2 = plt.subplot(2, 3, 2)
    bars = plt.bar(labels, counts, color=['#66b3ff', '#ff9999'] if len(unique) == 2 else None)
    plt.title('标签数量分布', fontsize=12, fontweight='bold')
    plt.ylabel('样本数量')
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')
    
    # 3. 特征均值分布
    ax3 = plt.subplot(2, 3, 3)
    X_flat = X.reshape(-1, X.shape[-1])
    feature_means = X_flat.mean(axis=0)
    plt.hist(feature_means, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('特征均值分布', fontsize=12, fontweight='bold')
    plt.xlabel('特征均值')
    plt.ylabel('频数')
    plt.grid(True, alpha=0.3)
    
    # 4. 特征标准差分布
    ax4 = plt.subplot(2, 3, 4)
    feature_stds = X_flat.std(axis=0)
    plt.hist(feature_stds, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.title('特征标准差分布', fontsize=12, fontweight='bold')
    plt.xlabel('特征标准差')
    plt.ylabel('频数')
    plt.grid(True, alpha=0.3)
    
    # 5. 序列长度分布（每个样本的序列值范围）
    ax5 = plt.subplot(2, 3, 5)
    # 计算每个序列的均值
    seq_means = X.mean(axis=(1, 2))
    plt.hist(seq_means, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title('序列均值分布', fontsize=12, fontweight='bold')
    plt.xlabel('序列均值')
    plt.ylabel('频数')
    plt.grid(True, alpha=0.3)
    
    # 6. 类别特征对比（前10个特征的均值对比）
    ax6 = plt.subplot(2, 3, 6)
    if len(unique) == 2:
        normal_indices = np.where(y == 0)[0]
        anomaly_indices = np.where(y == 1)[0]
        
        if len(normal_indices) > 0 and len(anomaly_indices) > 0:
            normal_features = X[normal_indices].mean(axis=(0, 1))[:10]
            anomaly_features = X[anomaly_indices].mean(axis=(0, 1))[:10]
            
            x_pos = np.arange(10)
            width = 0.35
            plt.bar(x_pos - width/2, normal_features, width, label='正常', color='#66b3ff', alpha=0.7)
            plt.bar(x_pos + width/2, anomaly_features, width, label='异常', color='#ff9999', alpha=0.7)
            plt.title('前10个特征均值对比', fontsize=12, fontweight='bold')
            plt.xlabel('特征索引')
            plt.ylabel('特征均值')
            plt.legend()
            plt.xticks(x_pos, [f'F{i+1}' for i in range(10)])
            plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f'{file_name_base}_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存: {save_path}")
    plt.close()

def plot_feature_correlation(X, file_name_base, save_dir, max_features=20):
    """绘制特征相关性热力图（只显示前max_features个特征）"""
    X_flat = X.reshape(-1, X.shape[-1])
    
    # 只使用前max_features个特征
    n_features = min(max_features, X_flat.shape[1])
    X_subset = X_flat[:, :n_features]
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(X_subset.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=False, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": 0.8})
    plt.title(f'{file_name_base}\n特征相关性热力图 (前{n_features}个特征)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('特征索引')
    plt.ylabel('特征索引')
    
    save_path = os.path.join(save_dir, f'{file_name_base}_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"相关性热力图已保存: {save_path}")
    plt.close()

def visualize_all_files():
    """可视化所有预处理文件"""
    processor = CICIDSDataProcessor()
    processed_dir = Config.PROCESSED_DATA_DIR
    
    if not os.path.exists(processed_dir):
        print(f"错误: 预处理数据目录不存在: {processed_dir}")
        print("请先运行: python main.py --mode preprocess")
        return
    
    # 获取所有npy文件
    npy_files = [f for f in os.listdir(processed_dir) if f.endswith('_X.npy')]
    
    if not npy_files:
        print(f"错误: 在 {processed_dir} 中未找到预处理数据文件")
        print("请先运行: python main.py --mode preprocess")
        return
    
    # 提取文件名
    file_names = [f.replace('_X.npy', '') for f in npy_files]
    
    print(f"找到 {len(file_names)} 个预处理文件")
    print("="*60)
    
    # 创建保存目录
    save_dir = os.path.join(Config.RESULTS_DIR, 'data_visualization')
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化每个文件
    all_stats = []
    
    for idx, file_name in enumerate(file_names, 1):
        print(f"\n[{idx}/{len(file_names)}] 处理文件: {file_name}")
        result = visualize_single_file(file_name, processed_dir)
        
        if result is not None:
            X, y = result
            plot_data_distribution(X, y, file_name, save_dir)
            plot_feature_correlation(X, file_name, save_dir)
            
            # 收集统计信息
            unique, counts = np.unique(y, return_counts=True)
            all_stats.append({
                'file': file_name,
                'samples': len(y),
                'normal': counts[0] if len(counts) > 0 else 0,
                'anomaly': counts[1] if len(counts) > 1 else 0,
                'normal_pct': counts[0]/len(y)*100 if len(counts) > 0 else 0,
                'anomaly_pct': counts[1]/len(y)*100 if len(counts) > 1 else 0
            })
    
    # 绘制总体统计对比图
    if all_stats:
        plot_summary_comparison(all_stats, save_dir)
    
    print(f"\n{'='*60}")
    print(f"所有可视化结果已保存到: {save_dir}")
    print(f"{'='*60}")

def plot_summary_comparison(all_stats, save_dir):
    """绘制所有文件的统计对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    files = [s['file'] for s in all_stats]
    samples = [s['samples'] for s in all_stats]
    normal_counts = [s['normal'] for s in all_stats]
    anomaly_counts = [s['anomaly'] for s in all_stats]
    normal_pcts = [s['normal_pct'] for s in all_stats]
    anomaly_pcts = [s['anomaly_pct'] for s in all_stats]
    
    # 简化文件名显示
    file_labels = [f.split('.')[0] for f in files]
    
    # 1. 样本数量对比
    ax1 = axes[0, 0]
    x_pos = np.arange(len(files))
    width = 0.35
    ax1.bar(x_pos - width/2, normal_counts, width, label='正常', color='#66b3ff', alpha=0.7)
    ax1.bar(x_pos + width/2, anomaly_counts, width, label='异常', color='#ff9999', alpha=0.7)
    ax1.set_xlabel('文件')
    ax1.set_ylabel('样本数量')
    ax1.set_title('各文件样本数量对比', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(file_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 总样本数
    ax2 = axes[0, 1]
    ax2.bar(file_labels, samples, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('文件')
    ax2.set_ylabel('总样本数')
    ax2.set_title('各文件总样本数', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(samples):
        ax2.text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # 3. 异常比例
    ax3 = axes[1, 0]
    ax3.plot(file_labels, anomaly_pcts, marker='o', linewidth=2, markersize=8, color='#ff9999')
    ax3.set_xlabel('文件（按时间顺序）')
    ax3.set_ylabel('异常样本比例 (%)')
    ax3.set_title('各文件异常样本比例变化', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    for i, v in enumerate(anomaly_pcts):
        ax3.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # 4. 类别分布堆叠图
    ax4 = axes[1, 1]
    x_pos = np.arange(len(files))
    ax4.bar(x_pos, normal_pcts, label='正常', color='#66b3ff', alpha=0.7)
    ax4.bar(x_pos, anomaly_pcts, bottom=normal_pcts, label='异常', color='#ff9999', alpha=0.7)
    ax4.set_xlabel('文件')
    ax4.set_ylabel('比例 (%)')
    ax4.set_title('各文件类别分布（堆叠）', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(file_labels, rotation=45, ha='right')
    ax4.legend()
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'summary_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n总体对比图已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("="*60)
    print("预处理数据可视化工具")
    print("="*60)
    print(f"数据目录: {Config.PROCESSED_DATA_DIR}")
    print(f"结果保存目录: {Config.RESULTS_DIR}/data_visualization")
    print()
    
    visualize_all_files()

if __name__ == "__main__":
    main()

