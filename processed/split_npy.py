import numpy as np
import os

def split_large_npy(file_path, max_size_gb=1.9):
    """
    将大npy文件拆分为多个小于max_size_gb的小文件，确保符合GitHub LFS限制
    :param file_path: 原始大文件绝对路径
    :param max_size_gb: 拆分阈值，默认1.9GB（留安全余量）
    """
    # 计算文件大小（KB）
    file_size_kb = os.path.getsize(file_path) / 1024
    file_name = os.path.basename(file_path)
    
    if file_size_kb <= max_size_gb * 1024 * 1024:
        print(f"✅ {file_name} 大小正常，无需拆分")
        return

    # 加载原始数据
    print(f"🔄 正在拆分 {file_name}，原始大小：{file_size_kb/1024/1024:.2f} GiB")
    data = np.load(file_path)
    
    # 计算拆分份数，确保每份<1.9GB
    num_splits = int(np.ceil(file_size_kb / (max_size_gb * 1024 * 1024)))
    split_indices = np.array_split(np.arange(len(data)), num_splits)

    # 拆分并保存（保存到原文件所在目录）
    base_name = os.path.splitext(file_path)[0]
    for i, idx in enumerate(split_indices):
        split_data = data[idx]
        output_path = os.path.join(os.path.dirname(file_path), f"{base_name}_part{i+1}.npy")
        np.save(output_path, split_data)
        split_size_kb = os.path.getsize(output_path) / 1024
        print(f"   ✅ 已生成：{os.path.basename(output_path)}，大小：{split_size_kb/1024/1024:.2f} GiB")

    # 删除原始超大文件
    os.remove(file_path)
    print(f"🗑️  已删除原始文件：{file_name}\n")


# ======================
# 配置你的文件绝对路径（必须改！填你实际的文件路径）
# ======================
# 替换成你实际的文件绝对路径（复制粘贴你的文件完整路径）
files_to_split = [
    "E:\\project\\datasets\\processed\\Wednesday-workingHours.pcap_ISCX_X.npy",
    "E:\\project\\datasets\\processed\\Monday-WorkingHours.pcap_ISCX_X.npy",
    "E:\\project\\datasets\\processed\\Tuesday-WorkingHours.pcap_ISCX_X.npy"
]

# 执行拆分
for file_path in files_to_split:
    if os.path.exists(file_path):
        split_large_npy(file_path)
    else:
        print(f"❌ 文件不存在：{file_path}\n")

print("🎉 所有超限文件拆分完成！")