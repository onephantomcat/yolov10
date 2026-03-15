import os
import shutil
from tqdm import tqdm

# --- 配置部分 ---
# 指向你当前已经生成的 YOLO 数据集根目录
DATASET_ROOT = r"C:\Users\13238\datasets\UA-Finetune-Lite-V2"
# 仅对训练集进行扩充（验证集应保持真实分布，不建议扩充）
SUBSET = "train"

# 设定扩充倍数 (原本有1份，设为3则再复制2份，总共3份)
EXPAND_FACTORS = {
    1: 2,  # bus (ID 1): 总量变为 2 倍
    2: 2,  # van (ID 2): 总量变为 2 倍
    3: 5,  # truck (ID 3): 总量变为 5 倍 (卡车最少，给最高倍数)
}

def oversample():
    img_dir = os.path.join(DATASET_ROOT, "images", SUBSET)
    lbl_dir = os.path.join(DATASET_ROOT, "labels", SUBSET)

    if not os.path.exists(lbl_dir):
        print(f"错误：找不到标签目录 {lbl_dir}")
        return

    label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    print(f"开始扫描 {SUBSET} 集，包含 {len(label_files)} 个原始文件...")

    new_files_count = 0

    for lbl_name in tqdm(label_files):
        lbl_path = os.path.join(lbl_dir, lbl_name)
        img_name = lbl_name.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        # 读取标签内容，判断包含哪些类别
        frame_classes = set()
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    frame_classes.add(int(parts[0]))

        # 决定该帧的扩充倍数（取该帧中包含类别的最高扩充倍数）
        max_factor = 1
        for cls_id, factor in EXPAND_FACTORS.items():
            if cls_id in frame_classes:
                if factor > max_factor:
                    max_factor = factor

        # 如果需要扩充 (factor > 1)
        if max_factor > 1:
            for i in range(1, max_factor):
                prefix = f"aug{i}_"
                new_lbl_name = prefix + lbl_name
                new_img_name = prefix + img_name

                # 物理复制文件
                shutil.copy(lbl_path, os.path.join(lbl_dir, new_lbl_name))
                shutil.copy(img_path, os.path.join(img_dir, new_img_name))
                new_files_count += 1

    print(f"\n扩充完成！")
    print(f"新增文件数: {new_files_count} 组 (图片+标签)")
    print(f"当前 {SUBSET} 集总图片数: {len(os.listdir(img_dir))}")

if __name__ == "__main__":
    oversample()