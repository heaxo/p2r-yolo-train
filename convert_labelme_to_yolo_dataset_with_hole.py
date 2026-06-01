import argparse
import json
import math
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASSES = {
    "plate": 0,
    "paper": 1,
    "hole": 2,
}

CLASS_NAMES = {v: k for k, v in CLASSES.items()}
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
DEFAULT_TRAIN_SETTINGS = {
    "model": "yolov8s-seg.pt",
    "imgsz": 1280,
    "epochs": 300,
    "batch": 2,
    "patience": 80,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
}


def read_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image(json_path: Path, image_dir: Path, json_data: dict | None = None) -> Path | None:
    data = json_data if json_data is not None else read_json(json_path)

    image_path = data.get("imagePath")
    if image_path:
        candidate = image_dir / Path(image_path).name
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{json_path.stem}{ext}"
        if candidate.exists():
            return candidate

    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == json_path.stem:
            return p

    return None


def load_image_size(json_data: dict, image_path: Path) -> tuple[int, int]:
    w = json_data.get("imageWidth")
    h = json_data.get("imageHeight")

    if w and h:
        return int(w), int(h)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    return w, h


def clean_points(points: list) -> list[tuple[float, float]]:
    cleaned = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        cleaned.append((float(p[0]), float(p[1])))
    return cleaned


def rectangle_to_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    (x1, y1), (x2, y2) = points[:2]
    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def circle_to_polygon(points: list[tuple[float, float]], segments: int = 48) -> list[tuple[float, float]]:
    (cx, cy), (ex, ey) = points[:2]
    radius = math.hypot(ex - cx, ey - cy)
    if radius <= 0:
        return []

    return [
        (
            cx + radius * math.cos(2 * math.pi * i / segments),
            cy + radius * math.sin(2 * math.pi * i / segments),
        )
        for i in range(segments)
    ]


def shape_to_polygon(shape: dict) -> tuple[list[tuple[float, float]], str | None]:
    shape_type = (shape.get("shape_type") or "polygon").strip().lower()
    points = clean_points(shape.get("points", []))

    if shape_type == "polygon":
        if len(points) < 3:
            return [], "polygon has fewer than 3 points"
        return points, None

    if shape_type == "rectangle":
        if len(points) < 2:
            return [], "rectangle has fewer than 2 points"
        return rectangle_to_polygon(points), None

    if shape_type == "circle":
        if len(points) < 2:
            return [], "circle has fewer than 2 points"
        return circle_to_polygon(points), None

    return [], f"unsupported shape_type={shape_type}"


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def normalize_polygon(points: list[tuple[float, float]], w: int, h: int) -> list[float]:
    values = []
    for x, y in points:
        x = max(0.0, min(float(x), w - 1))
        y = max(0.0, min(float(y), h - 1))
        values.append(x / w)
        values.append(y / h)
    return values


def convert_one_labelme_json(json_path: Path, image_path: Path):
    data = read_json(json_path)
    w, h = load_image_size(data, image_path)
    lines = []
    counter = Counter()
    warnings = []

    for shape in data.get("shapes", []):
        label = (shape.get("label") or "").strip()
        if label not in CLASSES:
            warnings.append(f"{json_path.name}: ignored unknown label {label!r}")
            continue

        points, reason = shape_to_polygon(shape)
        if reason:
            warnings.append(f"{json_path.name}: ignored {label!r}, {reason}")
            continue

        if polygon_area(points) < 1.0:
            warnings.append(f"{json_path.name}: ignored {label!r}, polygon area is too small")
            continue

        cls_id = CLASSES[label]
        coords = normalize_polygon(points, w, h)
        line = str(cls_id) + " " + " ".join(f"{v:.8f}" for v in coords)
        lines.append(line)
        counter[label] += 1

    return lines, counter, warnings


def draw_visual_check(image_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None or not label_path.exists():
        return

    h, w = img.shape[:2]
    overlay = img.copy()
    alpha = 0.25

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    colors = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
    }

    for line in lines:
        arr = line.split()
        if len(arr) < 7:
            continue

        cls_id = int(arr[0])
        coords = list(map(float, arr[1:]))
        pts = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)
        color = colors.get(cls_id, (255, 255, 255))
        name = CLASS_NAMES.get(cls_id, str(cls_id))

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 3)

        if len(pts) > 0:
            x0, y0 = pts[0]
            cv2.putText(
                img,
                name,
                (x0, max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def write_plate_yaml(out_dir: Path):
    yaml_text = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: plate
  1: paper
  2: hole
"""

    for name in ("plate.yaml", "data.yaml"):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            f.write(yaml_text)

    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES)) + "\n")


def write_train_config(out_dir: Path):
    text = f"""# 当前数据集的 YOLOv8 分割推荐训练配置。
# 数据集现状：
# - 144 张 LabelMe 标注图
# - 任务是实例分割，不只是目标检测
# - plate/paper 样本充足，hole 极少
# - 钢板边缘、孔洞、细轮廓需要较高输入分辨率
# Generated by convert_labelme_to_yolo_dataset_with_hole.py

# 当前最推荐 yolov8s-seg：比 nano 对边缘和分割 mask 更稳，
# 又不会像 m/l/x 那样对 144 张小数据集和显存压力过大。
model: {DEFAULT_TRAIN_SETTINGS["model"]}

# 数据集配置文件，指向 images/train 和 images/val。
data: plate.yaml

# 1280 能保留小孔和细钢板边缘的像素细节。
# 如果显存不够，优先把 batch 降到 1，不要先降 imgsz。
imgsz: {DEFAULT_TRAIN_SETTINGS["imgsz"]}

# 144 张图属于小数据集，300 轮给模型足够学习次数；
# 后面的 patience 会自动早停，避免无意义训练和过拟合。
epochs: {DEFAULT_TRAIN_SETTINGS["epochs"]}

# batch=2 是 1280 分割训练的稳妥起点，兼顾梯度稳定和显存。
# 如果 CUDA out-of-memory，改成 batch=1。
batch: {DEFAULT_TRAIN_SETTINGS["batch"]}

# 验证集 80 轮没有提升就停止。
# 小数据集允许模型多等一会儿，但不让它长期死记训练集。
patience: {DEFAULT_TRAIN_SETTINGS["patience"]}

# Windows 上 workers=0 最稳定；144 张本地图片读取不是瓶颈。
workers: 0

# 使用第 0 张 GPU。没有 CUDA 时建议运行 Python 脚本，它会自动回退到 CPU。
device: 0

# 训练结果保存在当前数据集目录内，方便整体移动和复现。
project: runs/segment

# 固定实验名，预测脚本能直接找到 weights/best.pt。
name: plate_seg_best

# 不覆盖旧实验，方便对比不同训练结果。
exist_ok: false

# 只有 144 张图，必须用预训练权重；从零训练效果通常很差。
pretrained: true

# 混合精度能省显存，通常还能加快训练。
amp: true

# 不缓存到内存，避免 1280 图像占用过多 RAM；小数据集直接读盘即可。
cache: false

# AdamW 对小数据集微调更稳，正则化也更清晰，比 SGD 更省调参。
optimizer: {DEFAULT_TRAIN_SETTINGS["optimizer"]}

# 0.001 是保守初始学习率，适合微调预训练模型，降低 mask 发散风险。
lr0: {DEFAULT_TRAIN_SETTINGS["lr0"]}

# 最终学习率比例较低，训练后期更利于细化边缘 mask。
lrf: {DEFAULT_TRAIN_SETTINGS["lrf"]}

# 余弦退火让学习率平滑下降，适合 300 轮这种较长微调。
cos_lr: true

# 权重衰减用于抑制 144 张小数据集过拟合。
weight_decay: {DEFAULT_TRAIN_SETTINGS["weight_decay"]}

# 前 3 轮 warmup，避免一开始学习率过猛破坏预训练特征。
warmup_epochs: 3

# 小角度旋转模拟拍照角度变化，同时不明显破坏钢板几何。
degrees: 8

# 轻微平移增强目标在画面中不同位置的鲁棒性。
translate: 0.08

# 中等缩放模拟不同拍摄距离，同时不过度损坏分割轮廓。
scale: 0.35

# 小 shear 模拟相机倾斜；过大会让钢板形状不真实。
shear: 2.0

# 极小透视增强处理拍照透视；数值大了会扭曲边缘和孔洞。
perspective: 0.0005

# 轻微色相扰动处理不同相机白平衡。
hsv_h: 0.015

# 饱和度扰动处理材质反光和光照差异。
hsv_s: 0.45

# 明度扰动处理阴影、眩光、曝光变化。
hsv_v: 0.45

# 左右翻转有效，因为钢板方向不是类别定义的一部分。
fliplr: 0.5

# 上下翻转对俯拍/旋转钢板图片有效，能增加小数据集变化。
flipud: 0.5

# 低强度 mosaic 增加场景变化，但不大量破坏精细分割边界。
mosaic: 0.25

# 最后 40 轮关闭 mosaic，让最终 mask 回到真实图片上精修。
close_mosaic: 40

# mixup 关闭：混合图像会让钢板边界变模糊，不利于分割精度。
mixup: 0.0

# copy_paste 关闭：hole 太少，随机粘贴容易制造不真实孔洞样本。
copy_paste: 0.0

# 保存权重和训练日志。
save: true

# 每 20 轮保存一次，便于回退、对比和继续训练。
save_period: 20

# 保存训练曲线和验证图，便于判断过拟合和分割质量。
plots: true
"""
    with open(out_dir / "train_plate_seg_best.yaml", "w", encoding="utf-8") as f:
        f.write(text)


def write_train_requirements(out_dir: Path):
    text = """ultralytics>=8.3.0
opencv-python
torch
"""
    with open(out_dir / "requirements-train.txt", "w", encoding="utf-8") as f:
        f.write(text)


def write_train_script(out_dir: Path):
    text = '''# -*- coding: utf-8 -*-
from pathlib import Path

import torch
from ultralytics import YOLO


def pick_local_model(script_dir: Path) -> str:
    candidates = [
        script_dir / "yolov8s-seg.pt",
        script_dir.parent / "yolov8s-seg.pt",
        script_dir / "yolov8n-seg.pt",
        script_dir.parent / "yolov8n-seg.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "yolov8s-seg.pt"


def main():
    script_dir = Path(__file__).resolve().parent
    data_yaml = script_dir / "plate.yaml"
    model_path = pick_local_model(script_dir)
    device = 0 if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", model_path)
    print("Data:", data_yaml)

    model = YOLO(model_path)
    model.train(
        # 数据集配置文件，由转换脚本生成，指向 images/train 和 images/val。
        data=str(data_yaml),

        # 1280 是当前数据集的最佳默认值：小孔和钢板边缘很细，降分辨率会直接丢细节。
        # 如果显存不足，优先把 batch 降到 1，不要先降 imgsz。
        imgsz=1280,

        # 144 张图需要更多重复学习，300 轮能给模型足够收敛空间；patience 会负责早停。
        epochs=300,

        # batch=2 在 1280 分割训练下兼顾梯度稳定和显存占用，是稳妥起点。
        batch=2,

        # Windows 上 workers=0 更稳定；当前只有 144 张图，读取速度不是主要瓶颈。
        workers=0,

        # 有 CUDA 就用第 0 张 GPU，否则自动回退 CPU；1280 分割训练强烈建议用 GPU。
        device=device,

        # 训练结果放在数据集目录里，方便整体移动、复现和查找。
        project=str(script_dir / "runs" / "segment"),

        # 固定实验名，预测脚本可以直接定位到 weights/best.pt。
        name="plate_seg_best",

        # 不覆盖旧实验，方便比较不同训练结果。
        exist_ok=False,

        # 小数据集必须用预训练权重；从零训练很难学到稳定边缘和 mask。
        pretrained=True,

        # 混合精度可以节省显存，通常还能提升训练速度。
        amp=True,

        # 不缓存图片，避免 1280 图像占用过多内存；当前数据量直接读盘足够。
        cache=False,

        # 验证集 80 轮无提升就停止，给小数据集充分尝试，同时控制过拟合。
        patience=80,

        # AdamW 对小数据集微调更稳，配合 weight_decay 更容易控制过拟合。
        optimizer="AdamW",

        # 0.001 是保守学习率，适合微调预训练模型，不容易把 mask 训练发散。
        lr0=0.001,

        # 最终学习率较低，后期更适合精修边缘和孔洞轮廓。
        lrf=0.01,

        # 余弦学习率下降更平滑，适合 300 轮长微调。
        cos_lr=True,

        # 权重衰减用于降低 144 张小样本训练的记忆化风险。
        weight_decay=0.0005,

        # 前 3 轮 warmup，避免训练初期学习率过猛破坏预训练特征。
        warmup_epochs=3,

        # 小角度旋转适配拍摄角度变化，同时不让钢板几何形状失真。
        degrees=8,

        # 轻微平移增强目标在画面不同位置时的鲁棒性。
        translate=0.08,

        # 中等缩放模拟不同拍摄距离，同时保留 mask 精度。
        scale=0.35,

        # 小 shear 模拟相机倾斜；数值过大会产生不真实钢板形状。
        shear=2.0,

        # 极小透视增强处理拍照透视，但避免严重扭曲边缘和孔洞。
        perspective=0.0005,

        # 色相扰动处理不同相机/环境下的白平衡差异。
        hsv_h=0.015,

        # 饱和度扰动处理钢板材质反光和光照差异。
        hsv_s=0.45,

        # 明度扰动处理阴影、眩光、曝光变化。
        hsv_v=0.45,

        # 左右翻转有效，因为左右方向不会改变 plate/paper/hole 的语义。
        fliplr=0.5,

        # 上下翻转适合俯拍或旋转钢板图片，能增加小数据集变化。
        flipud=0.5,

        # 低强度 mosaic 增加背景/上下文变化，但不过度破坏精细分割边界。
        mosaic=0.25,

        # 最后 40 轮关闭 mosaic，让模型在真实图像上精修最终 mask。
        close_mosaic=40,

        # 关闭 mixup：混合图像会让边界变模糊，不利于精确分割。
        mixup=0.0,

        # 关闭 copy_paste：hole 只有少量实例，随机粘贴容易生成不真实样本。
        copy_paste=0.0,

        # 保存最终权重、最佳权重和训练日志。
        save=True,

        # 每 20 轮保存一次检查点，便于回退、比较和继续训练。
        save_period=20,

        # 保存训练曲线和验证图片，方便检查过拟合和分割质量。
        plots=True,
    )


if __name__ == "__main__":
    main()
'''

    with open(out_dir / "train_plate_seg_best.py", "w", encoding="utf-8") as f:
        f.write(text)


def write_predict_script(out_dir: Path):
    text = '''import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    script_dir = Path(__file__).resolve().parent
    default_model = script_dir / "runs" / "segment" / "plate_seg_best" / "weights" / "best.pt"
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_model
    source = sys.argv[2] if len(sys.argv) > 2 else str(script_dir / "images" / "val")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    model.predict(
        source=source,
        imgsz=1280,
        conf=0.25,
        save=True,
        project=str(script_dir / "predict_out"),
        name="vis",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
'''

    with open(out_dir / "predict_plate_seg.py", "w", encoding="utf-8") as f:
        f.write(text)


def label_presence(json_path: Path) -> set[str]:
    data = read_json(json_path)
    labels = set()
    for shape in data.get("shapes", []):
        label = (shape.get("label") or "").strip()
        if label in CLASSES:
            labels.add(label)
    return labels


def split_json_files(json_files: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0.0, 1.0)")

    files = list(json_files)
    rng = random.Random(seed)
    rng.shuffle(files)

    if len(files) <= 1 or val_ratio == 0:
        return files, []

    val_count = max(1, round(len(files) * val_ratio))
    val_count = min(val_count, len(files) - 1)
    val_jsons = files[:val_count]
    train_jsons = files[val_count:]

    presence = {p: label_presence(p) for p in files}
    for label in CLASSES:
        all_with_label = [p for p in files if label in presence[p]]
        if len(all_with_label) < 2:
            continue

        train_has = any(label in presence[p] for p in train_jsons)
        val_has = any(label in presence[p] for p in val_jsons)

        if train_has and val_has:
            continue

        if not val_has:
            move_to_val = next((p for p in train_jsons if label in presence[p]), None)
            move_to_train = next((p for p in val_jsons if label not in presence[p]), val_jsons[0])
            if move_to_val and move_to_train:
                train_jsons.remove(move_to_val)
                val_jsons.append(move_to_val)
                val_jsons.remove(move_to_train)
                train_jsons.append(move_to_train)

        if not train_has:
            move_to_train = next((p for p in val_jsons if label in presence[p]), None)
            move_to_val = next((p for p in train_jsons if label not in presence[p]), train_jsons[0])
            if move_to_train and move_to_val:
                val_jsons.remove(move_to_train)
                train_jsons.append(move_to_train)
                train_jsons.remove(move_to_val)
                val_jsons.append(move_to_val)

    return sorted(train_jsons), sorted(val_jsons)


def zip_dir(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir.parent))


def clean_output_dir(out_dir: Path):
    resolved = out_dir.resolve()
    if resolved.anchor == str(resolved):
        raise RuntimeError(f"Refusing to clean unsafe output path: {resolved}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def write_report(
    out_dir: Path,
    total_jsons: int,
    converted_images: int,
    split_counts: dict[str, int],
    object_counts: Counter,
    files_with_class: Counter,
    warnings: list[str],
):
    lines = [
        "# Dataset report",
        "",
        f"- LabelMe JSON files: {total_jsons}",
        f"- Converted images: {converted_images}",
        f"- Train images: {split_counts.get('train', 0)}",
        f"- Val images: {split_counts.get('val', 0)}",
        "",
        "## Object counts",
        "",
    ]

    for name in CLASSES:
        lines.append(f"- {name}: {object_counts.get(name, 0)} objects in {files_with_class.get(name, 0)} images")

    lines.extend(
        [
            "",
            "## Recommended training command",
            "",
            "```powershell",
            "cd yolo-out",
            "python .\\train_plate_seg_best.py",
            "```",
            "",
            "## Notes",
            "",
            "- The generated training profile targets a 144-image small segmentation dataset.",
            "- `hole` has very few samples if its count is below 30; metrics for this class will be unstable.",
            "- Keep `imgsz=1280` for small holes and thin contours. If GPU memory is insufficient, lower `batch` to 1 before lowering `imgsz`.",
            "- Install training dependencies with `pip install -r requirements-train.txt` inside the environment used for training.",
            "- Review several images in `visual_check` before training.",
        ]
    )

    if warnings:
        lines.extend(["", "## Conversion warnings", ""])
        for warning in warnings[:200]:
            lines.append(f"- {warning}")
        if len(warnings) > 200:
            lines.append(f"- ... {len(warnings) - 200} more warnings")

    with open(out_dir / "dataset_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def convert_dataset(args):
    image_dir = Path(args.image_dir)
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out)

    if not image_dir.exists():
        raise RuntimeError(f"Image directory not found: {image_dir}")
    if not json_dir.exists():
        raise RuntimeError(f"JSON directory not found: {json_dir}")

    if args.clean:
        clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found: {json_dir}")

    train_jsons, val_jsons = split_json_files(json_files, args.val_ratio, args.seed)
    split_map = {"train": train_jsons, "val": val_jsons}

    total_counter = Counter()
    files_with_class = Counter()
    warnings = []
    converted_images = 0
    split_counts = defaultdict(int)

    for split, files in split_map.items():
        img_out_dir = out_dir / "images" / split
        label_out_dir = out_dir / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for json_path in files:
            json_data = read_json(json_path)
            image_path = find_image(json_path, image_dir, json_data)
            if image_path is None:
                warnings.append(f"{json_path.name}: image not found")
                print(f"[SKIP] image not found: {json_path.name}")
                continue

            lines, counter, item_warnings = convert_one_labelme_json(json_path, image_path)
            warnings.extend(item_warnings)

            for k, v in counter.items():
                total_counter[k] += v
            for k, v in counter.items():
                if v > 0:
                    files_with_class[k] += 1

            if not lines:
                warnings.append(f"{json_path.name}: no valid annotations")
                print(f"[WARN] no valid annotations: {json_path.name}")

            target_image = img_out_dir / image_path.name
            target_label = label_out_dir / f"{image_path.stem}.txt"

            shutil.copy2(image_path, target_image)
            with open(target_label, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            visual_path = out_dir / "visual_check" / split / f"{image_path.stem}_overlay.jpg"
            draw_visual_check(target_image, target_label, visual_path)

            converted_images += 1
            split_counts[split] += 1
            print(
                f"[OK] {split}: {image_path.name} "
                f"plate={counter['plate']} paper={counter['paper']} hole={counter['hole']}"
            )

    write_plate_yaml(out_dir)
    write_train_config(out_dir)
    write_train_requirements(out_dir)
    write_train_script(out_dir)
    write_predict_script(out_dir)
    write_report(
        out_dir=out_dir,
        total_jsons=len(json_files),
        converted_images=converted_images,
        split_counts=dict(split_counts),
        object_counts=total_counter,
        files_with_class=files_with_class,
        warnings=warnings,
    )

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[ZIP] {zip_path.resolve()}")

    print("\nStats:")
    for name in CLASSES:
        print(f"  {name}: {total_counter[name]}")
    print(f"  train images: {split_counts.get('train', 0)}")
    print(f"  val images: {split_counts.get('val', 0)}")

    if warnings:
        print(f"\nWarnings: {len(warnings)}. See {out_dir / 'dataset_report.md'}")

    print(f"\nDone: {out_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to a YOLOv8 segmentation dataset with recommended training files."
    )
    parser.add_argument("--image-dir", default="./raw_images", help="Raw image directory")
    parser.add_argument("--json-dir", default="./labelme_json", help="LabelMe JSON directory")
    parser.add_argument("--out", default="./yolo-out", help="Output YOLO dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument("--zip", action="store_true", help="Also create a zip package")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before conversion")
    return parser.parse_args()


def main():
    convert_dataset(parse_args())


if __name__ == "__main__":
    main()
