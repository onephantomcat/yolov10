import os
from collections import Counter
# 指向你处理后的标签目录
LABEL_DIR = r"C:\Users\13238\datasets\UA-Finetune-Lite-V2\labels\train"

def count_labels(label_path):
    # 类别映射表（与你代码中一致）
    class_names = {0: 'car', 1: 'bus', 2: 'van', 3: 'truck(others)'}
    stats = Counter()

    if not os.path.exists(label_path):
        print(f"错误：路径不存在 {label_path}")
        return

    label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    print(f"正在统计 {len(label_files)} 个标签文件...")

    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls_id = int(line.split()[0])
                stats[cls_id] += 1

    print("\n--- 类别统计结果 ---")
    total = sum(stats.values())
    for cls_id in sorted(class_names.keys()):
        count = stats[cls_id]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{class_names[cls_id]} (ID {cls_id}): {count} 个 ({percentage:.2f}%)")
    print(f"总计实例数: {total}")

if __name__ == "__main__":
    count_labels(LABEL_DIR)