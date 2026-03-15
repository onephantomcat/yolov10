import os
from ultralytics import YOLOv10
import torch
# 解决 Windows 路径问题
_original_load = torch.load

def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = safe_load_shim
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = r"C:\Users\13238\datasets\UA-Finetune-Lite-V2\images\train"
VAL_DIR = r"C:\Users\13238\yolov10\datasets\UA-Finetune-VAL"
YAML_PATH = os.path.join(BASE_DIR, "ua_finetune.yaml")

# 3. 生成配置文件
yaml_content = f"""
path: {os.path.dirname(TRAIN_DIR)} # 数据集根目录
train: {TRAIN_DIR}
val: {VAL_DIR}

nc: 4
names:
  0: car
  1: bus
  2: van
  3: truck
"""

with open(YAML_PATH, "w", encoding="utf-8") as f:
    f.write(yaml_content)

if __name__ == '__main__':
    model_name = "yolov10s.pt"
    print(f"使用官方权重重置训练: {model_name}")
    print("开始训练...")
    model = YOLOv10(model_name)

    model.train(
        data=YAML_PATH,
        workers=4,
        epochs=50,
        batch=8,
        imgsz=640,
        hsv_h=0.015,  # 色调微调
        hsv_s=0.7,  # 饱和度大幅随机 (模拟不同颜色鲜艳度)
        hsv_v=0.4,  # 亮度大幅随机 (模拟白天/阴天/阴影，强迫模型不依赖亮度)
        mosaic=0.5,  # 开启马赛克增强 (拼图)，这是解决遮挡的神器
        erasing=0.4,  # Random Erasing: 随机在物体框内擦除一小块区域 (填充噪声或灰色)
        device=0,
        lr0=0.0001,
        optimizer='AdamW',
        close_mosaic=10,  # 最后10轮关闭 Mosaic 增强，提高精度
        save=True,
        plots=True,
        name='ua_balanced_augmentation'
    )