import numpy as np
import os
import glob

def merge_files(prefix):
    print(f"正在合并：{prefix}")
    
    parts = sorted(glob.glob(f"processed/{prefix}_part*.npy"))
    if not parts:
        print(f"未找到 {prefix} 的拆分文件")
        return

    data_parts = [np.load(p) for p in parts]
    full_data = np.concatenate(data_parts, axis=0)
    np.save(f"processed/{prefix}.npy", full_data)
    print(f"✅ 合并完成：processed/{prefix}.npy\n")

# 自动合并你三个大文件
merge_files("Monday-WorkingHours.pcap_ISCX_X")
merge_files("Tuesday-WorkingHours.pcap_ISCX_X")
merge_files("Wednesday-workingHours.pcap_ISCX_X")

print("🎉 所有文件合并完成！")