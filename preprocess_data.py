"""
修复版数据预处理脚本
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_processor import CICIDSDataProcessor
from config import Config

def inspect_dataset():
    """检查数据集结构"""
    print("检查数据集结构...")
    data_dir = Config.CIC_IDS_DIR

    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            print(f"\n检查文件: {file}")

            try:
                # 只读取前几行来检查结构
                df_sample = pd.read_csv(file_path, nrows=5)
                print(f"列名: {df_sample.columns.tolist()}")
                print(f"形状: {df_sample.shape}")

                # 检查可能的标签列
                object_columns = df_sample.select_dtypes(include=['object']).columns
                if len(object_columns) > 0:
                    print(f"文本列: {object_columns.tolist()}")
                    for col in object_columns:
                        unique_vals = df_sample[col].unique()
                        print(f"  {col} 的取值: {unique_vals}")
            except Exception as e:
                print(f"检查文件时出错: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("CIC-IDS2017 数据集预处理 - 分别处理8个CSV文件")
    print("=" * 60)

    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建必要目录
    Config.create_directories()

    # 初始化数据处理器
    processor = CICIDSDataProcessor()

    print(f"数据目录: {Config.CIC_IDS_DIR}")
    print(f"输出目录: {Config.PROCESSED_DATA_DIR}")
    print(f"特征数量: {len(Config.FEATURE_COLUMNS)}")
    print()

    try:
        # 分别预处理所有CSV文件
        print("开始分别预处理8个CSV文件...")
        processed_data = processor.preprocess_all_files(
            data_dir=Config.CIC_IDS_DIR,
            output_dir=Config.PROCESSED_DATA_DIR
        )

        if processed_data is None or len(processed_data) == 0:
            print("错误: 预处理失败或没有成功处理任何文件")
            return

        print("\n" + "=" * 60)
        print("数据预处理完成!")
        print(f"成功处理 {len(processed_data)} 个文件")
        print(f"预处理后的数据保存在: {Config.PROCESSED_DATA_DIR}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    except Exception as e:
        print(f"预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()