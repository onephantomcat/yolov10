import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
import random

# --- 路径配置 ---
XML_DIR = r"C:\Users\13238\datasets\DETRAC-Train-Annotations-XML"
IMG_ROOT = r"C:\Users\13238\datasets\Insight-MVT_Annotation_Train"
OUTPUT_DIR = r"C:\Users\13238\datasets\UA-Lite-Balanced"

# --- 极速缩减策略配置 ---
FRAME_STRIDE = 10  # 【核心1】步长：每10帧取1帧。直接将冗余度降低90%
GLOBAL_REDUCE_RATE = 0.5  # 【核心2】在抽帧基础上再随机保留50%，进一步精简

# 类别筛选概率（由于已经抽帧，这里可以设置得大一些保证稀有类存在）
KEEP_RARE_RATE = 1.0  # 含有 Bus/Van/Truck 的帧 100% 进入概率池
KEEP_CAR_ONLY_RATE = 0.0  # 纯轿车帧 0% 保留 (混合帧里的轿车足够多了)

VAL_SPLIT = 0.2  # 【核心3】必须划分验证集，否则无法判断过拟合

CLASS_MAPPING = {'car': 0, 'bus': 1, 'van': 2, 'others': 3}


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    return ((box[0] + box[2] / 2.0) * dw, (box[1] + box[3] / 2.0) * dh,
            box[2] * dw, box[3] * dh)


def main():
    # 创建 train 和 val 目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]

    # 用来临时存储所有待保存的数据，最后统一划分
    data_pool = []

    print(f"开始处理：抽帧步长={FRAME_STRIDE}, 全局保留率={GLOBAL_REDUCE_RATE}")

    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(XML_DIR, xml_file))
        root = tree.getroot()
        seq_name = root.attrib['name']

        seq_img_dir = os.path.join(IMG_ROOT, "train", seq_name)
        if not os.path.exists(seq_img_dir):
            seq_img_dir = os.path.join(IMG_ROOT, seq_name)
            if not os.path.exists(seq_img_dir): continue

        for frame in root.findall('frame'):
            frame_num = int(frame.attrib['num'])

            # --- 策略1：时间步长过滤 ---
            if frame_num % FRAME_STRIDE != 0:
                continue

            target_list = frame.find('target_list')
            if target_list is None: continue

            current_objs = []
            frame_classes = set()
            for target in target_list.findall('target'):
                attr = target.find('attribute')
                if attr is None: continue
                v_type = attr.attrib.get('vehicle_type', '').lower()
                if v_type in CLASS_MAPPING:
                    cls_id = CLASS_MAPPING[v_type]
                    box = target.find('box')
                    b = [float(box.attrib['left']), float(box.attrib['top']),
                         float(box.attrib['width']), float(box.attrib['height'])]
                    if b[2] > 10 and b[3] > 10:
                        frame_classes.add(cls_id)
                        xywh = convert_box((960, 540), b)
                        current_objs.append(f"{cls_id} {' '.join(map(str, xywh))}")

            if not current_objs: continue

            # --- 策略2：类别与全局概率过滤 ---
            has_rare = any(c in frame_classes for c in [1, 2, 3])

            should_save = False
            if has_rare:
                if random.random() < KEEP_RARE_RATE: should_save = True
            else:
                if random.random() < KEEP_CAR_ONLY_RATE: should_save = True

            # 增加全局二度缩减，防止数据量过载
            if should_save and random.random() < GLOBAL_REDUCE_RATE:
                img_filename = f"img{frame_num:05d}.jpg"
                src_path = os.path.join(seq_img_dir, img_filename)
                if os.path.exists(src_path):
                    data_pool.append({
                        'src': src_path,
                        'name': f"{seq_name}_{img_filename}",
                        'label': '\n'.join(current_objs)
                    })

    # --- 策略3：打乱并划分训练/验证集 ---
    random.shuffle(data_pool)
    val_size = int(len(data_pool) * VAL_SPLIT)

    print(f"\n正在写入文件... 总数: {len(data_pool)} 帧 (预计 Train: {len(data_pool) - val_size}, Val: {val_size})")

    for i, item in enumerate(tqdm(data_pool, desc="写入磁盘")):
        split = 'val' if i < val_size else 'train'
        dst_img = os.path.join(OUTPUT_DIR, 'images', split, item['name'])
        dst_lbl = os.path.join(OUTPUT_DIR, 'labels', split, item['name'].replace('.jpg', '.txt'))

        shutil.copy(item['src'], dst_img)
        with open(dst_lbl, 'w') as f:
            f.write(item['label'])

    print(f"完成！数据已存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()