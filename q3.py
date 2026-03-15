import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
import random

# --- 路径配置 ---
XML_DIR = r"C:\Users\13238\datasets\DETRAC-Train-Annotations-XML"
IMG_ROOT = r"C:\Users\13238\datasets\Insight-MVT_Annotation_Train"
OUTPUT_DIR = r"C:\Users\13238\datasets\UA-Finetune-Lite-V2"  # 极简版目录

# --- 极度缩减策略 ---
FRAME_STRIDE = 5  # 【关键】每隔5帧取1帧，消除视频冗余
KEEP_RATES = {
    'has_truck': 1.0,  # 含有卡车的帧：全部保留
    'has_bus_van': 0.05,  # 含有公交/面包车的帧：仅保留 5% (大幅缩减总体量)
    'only_car': 0.0  # 纯轿车帧：0% 保留 (混合帧里的轿车足够多了)
}

CLASS_MAPPING = {'car': 0, 'bus': 1, 'van': 2, 'others': 3}


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    return ((box[0] + box[2] / 2.0) * dw, (box[1] + box[3] / 2.0) * dh,
            box[2] * dw, box[3] * dh)


def main():
    out_img_dir = os.path.join(OUTPUT_DIR, 'images', 'train')
    out_lbl_dir = os.path.join(OUTPUT_DIR, 'labels', 'train')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    print(f"执行极度缩减策略：抽帧步长={FRAME_STRIDE}, Bus/Van保留率={KEEP_RATES['has_bus_van'] * 100}%")

    final_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    saved_images_count = 0

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

            # 【核心逻辑1】抽帧：只处理 1, 6, 11, 16... 帧，跳过中间重复帧
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

            # 【核心逻辑2】概率筛选
            should_save = False
            dice = random.random()

            if 3 in frame_classes:  # 只要有卡车，必须留
                should_save = True
            elif (1 in frame_classes) or (2 in frame_classes):  # 只要有公交或面包车
                if dice < KEEP_RATES['has_bus_van']:
                    should_save = True
            else:  # 只有轿车
                if dice < KEEP_RATES['only_car']:
                    should_save = True

            if should_save:
                img_filename = f"img{frame_num:05d}.jpg"
                src_path = os.path.join(seq_img_dir, img_filename)
                if not os.path.exists(src_path): continue

                new_name = f"{seq_name}_{img_filename}"
                dst_img = os.path.join(out_img_dir, new_name)
                dst_lbl = os.path.join(out_lbl_dir, new_name.replace('.jpg', '.txt'))

                shutil.copy(src_path, dst_img)
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(current_objs))

                # 统计
                for obj_str in current_objs:
                    cls_id = int(obj_str.split()[0])
                    final_stats[cls_id] += 1
                saved_images_count += 1

    print(f"\n极速瘦身完成！最终保存图片: {saved_images_count} 帧 (目标：5000-8000帧)")
    print("--- 最终实例分布 ---")
    names = {0: 'Car', 1: 'Bus', 2: 'Van', 3: 'Truck(others)'}
    for k, v in final_stats.items():
        print(f"{names[k]}: {v} 个")


if __name__ == "__main__":
    main()