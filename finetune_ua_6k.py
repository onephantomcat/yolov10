import os
from ultralytics import YOLOv10
import torch
_original_load = torch.load
def safe_load_shim(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_shim
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集路径
DATASET_DIR = r"C:\Users\13238\yolov10\datasets\UA-Finetune-6k"
# 验证集路径
VAL_DIR = r"C:\Users\13238\datasets\UA-DETRAC-VAL\images\val"

YAML_PATH = os.path.join(BASE_DIR, "ua_finetune.yaml")

yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: {VAL_DIR}  # 使用之前的完整验证集来评估，这样结果最公正

nc: 4
names:
  0: car
  1: bus
  2: van
  3: truck
"""

with open(YAML_PATH, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"配置文件已生成: {YAML_PATH}")

if __name__ == '__main__':
    model_path = r"C:\Users\13238\yolov10\runs\detect\finetune_ua6\weights\best.pt"

    if not os.path.exists(model_path):
        print("❌ 找不到预训练模型 best.pt，请检查路径！")
    else:
        print(f"🔥 加载预训练权重: {model_path}")
        print("🚀 开始在 UA-DETRAC (6k 精选集) 上进行微调...")
        model = YOLOv10(model_path)
        model.train(
            data=YAML_PATH,
            epochs=20,
            batch=8,
            imgsz=640,
            device=0,
            lr0=0.001,
            workers=0,
            cache=False,
            amp=True,
            name='finetune_ua'  # 结果保存在 runs/detect/finetune_ua
        )