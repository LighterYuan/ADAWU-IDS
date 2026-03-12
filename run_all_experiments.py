"""
完整实验执行脚本 - 用于SCI论文实验
Complete Experiment Script for SCI Paper

执行所有必要的实验：
1. 基线实验（静态模型）
2. 提出方法实验（DAWU）
3. 消融实验
4. 参数敏感性分析
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_command(cmd, description):
    """运行命令并记录结果"""
    print("\n" + "="*60)
    print(f"执行: {description}")
    print("="*60)
    print(f"命令: {cmd}")
    print("-"*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ 成功完成")
        return True
    else:
        print("✗ 执行失败")
        print("错误信息:")
        print(result.stderr)
        return False

def main():
    """主函数"""
    print("="*60)
    print("SCI论文完整实验执行脚本")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 实验配置
    epochs = 50
    n_models = 3
    batch_size = 1000
    
    experiments = []
    
    # ========== 阶段1：数据预处理 ==========
    print("\n【阶段1：数据预处理】")
    if not os.path.exists("data/processed/Monday-WorkingHours.pcap_ISCX_X.npy"):
        success = run_command(
            "python main.py --mode preprocess",
            "数据预处理"
        )
        experiments.append(("数据预处理", success))
    else:
        print("✓ 预处理数据已存在，跳过")
        experiments.append(("数据预处理", True))
    
    # ========== 阶段2：基线实验 ==========
    print("\n【阶段2：基线实验】")
    
    # 2.1 静态模型训练
    success = run_command(
        f"python main.py --mode train --epochs {epochs} --batch_size {batch_size}",
        "静态模型训练"
    )
    experiments.append(("静态模型训练", success))
    
    # 2.2 静态模型时间序列评估
    if success:
        success = run_command(
            "python main.py --mode evaluate_temporal",
            "静态模型时间序列评估"
        )
        experiments.append(("静态模型评估", success))
    
    # ========== 阶段3：提出方法实验 ==========
    print("\n【阶段3：提出方法实验（DAWU）】")
    
    # 3.1 DAWU集成训练
    success = run_command(
        f"python main.py --mode ensemble_train --epochs {epochs} --n_models {n_models} --batch_size {batch_size}",
        "DAWU集成模型训练"
    )
    experiments.append(("DAWU集成训练", success))
    
    # 3.2 DAWU自适应评估（核心实验）
    if success:
        success = run_command(
            f"python main.py --mode ensemble_evaluate --n_models {n_models} --ensemble_type dynamic",
            "DAWU自适应评估（核心实验）"
        )
        experiments.append(("DAWU自适应评估", success))
    
    # ========== 阶段4：结果汇总 ==========
    print("\n" + "="*60)
    print("实验执行汇总")
    print("="*60)
    
    for exp_name, success in experiments:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{exp_name:<30} {status}")
    
    print("\n" + "="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 检查输出文件
    print("\n输出文件检查:")
    output_files = [
        "results/temporal_results_*.txt",
        "results/ensemble_evaluation_*.json",
        "models/ensemble_base_*.h5",
        "models/ensemble_info_*.json"
    ]
    
    import glob
    for pattern in output_files:
        files = glob.glob(pattern)
        if files:
            print(f"✓ {pattern}: 找到 {len(files)} 个文件")
        else:
            print(f"✗ {pattern}: 未找到文件")
    
    print("\n提示:")
    print("1. 静态模型结果: results/temporal_results_*.txt")
    print("2. DAWU评估结果: results/ensemble_evaluation_*.json")
    print("3. 下一步: 运行结果分析脚本生成图表和表格")

if __name__ == "__main__":
    main()

