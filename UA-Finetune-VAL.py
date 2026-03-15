import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm

# XML 文件夹：请修改为你存放验证集 XML 的路径 (通常是 DETRAC-Test-Annotations-XML)
XML_DIR = r"C:\Users\13238\datasets\DETRAC-Test-Annotations-XML"
# 图片文件夹：原始验证集图片路径
IMG_ROOT = r"C:\Users\13238\datasets\UA-DETRAC-VAL\images\val"
# 输出路径：新的验证集位置
OUTPUT_DIR = r"C:\Users\13238\yolov10\datasets\UA-Finetune-VAL"

CLASS_MAPPING = {'car': 0, 'bus': 1, 'van': 2, 'others': 3}  # others -> truck

def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    return ((box[0] + box[2] / 2.0) * dw, (box[1] + box[3] / 2.0) * dh, box[2] * dw, box[3] * dh)


def main():
    img_dir = os.path.join(OUTPUT_DIR, 'images', 'val')
    lbl_dir = os.path.join(OUTPUT_DIR, 'labels', 'val')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    print(f"开始转换验证集，共 {len(xml_files)} 个序列...")

    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(XML_DIR, xml_file))
        root = tree.getroot()
        seq_name = root.attrib['name']

        for frame in root.findall('frame'):
            frame_num = int(frame.attrib['num'])
            # 寻找对应的图片
            src_img_name = f"{seq_name}__img{frame_num:05d}.jpg"
            src_img_path = os.path.join(IMG_ROOT, src_img_name)

            if not os.path.exists(src_img_path):
                continue  # 图片不存在则跳过

            target_list = frame.find('target_list')
            if target_list is None: continue

            label_lines = []
            for target in target_list.findall('target'):
                attr = target.find('attribute')
                if attr is None: continue
                v_type = attr.attrib.get('vehicle_type', '').lower()

                if v_type in CLASS_MAPPING:
                    cls_id = CLASS_MAPPING[v_type]
                    box = target.find('box')
                    xywh = convert_box((960, 540), [
                        float(box.attrib['left']), float(box.attrib['top']),
                        float(box.attrib['width']), float(box.attrib['height'])
                    ])
                    label_lines.append(f"{cls_id} {' '.join(map(str, xywh))}")

            # 只有当这张图里有我们关心的物体时，才作为验证集
            if label_lines:
                # 复制图片
                new_name = f"{seq_name}_{src_img_name}"
                dst_img_path = os.path.join(img_dir, new_name)
                shutil.copy(src_img_path, dst_img_path)

                # 写入标签
                with open(os.path.join(lbl_dir, new_name.replace('.jpg', '.txt')), 'w') as f:
                    f.write('\n'.join(label_lines))

    print(f"验证集生成完毕！路径: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()