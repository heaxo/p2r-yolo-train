import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


CLASSES = {
    "plate": 0,   # 钢板外轮廓
    "paper": 1,   # A4纸
    "hole": 2,    # 内孔 / 内轮廓 / 镂空 / 槽
}

CLASS_NAMES = {v: k for k, v in CLASSES.items()}
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(json_path: Path, image_dir: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = data.get("imagePath")
    if image_path:
        p = image_dir / Path(image_path).name
        if p.exists():
            return p

    for ext in IMAGE_EXTS:
        p = image_dir / f"{json_path.stem}{ext}"
        if p.exists():
            return p

    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == json_path.stem:
            return p

    return None


def load_image_size(json_data: dict, image_path: Path):
    w = json_data.get("imageWidth")
    h = json_data.get("imageHeight")

    if w and h:
        return int(w), int(h)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    return w, h


def normalize_polygon(points, w, h):
    values = []

    for x, y in points:
        x = max(0.0, min(float(x), w - 1))
        y = max(0.0, min(float(y), h - 1))
        values.append(x / w)
        values.append(y / h)

    return values


def convert_one_labelme_json(json_path: Path, image_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    w, h = load_image_size(data, image_path)
    lines = []
    counter = {name: 0 for name in CLASSES.keys()}

    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        if label not in CLASSES:
            print(f"[忽略未知类别] {json_path.name}: {label}")
            continue

        if shape_type != "polygon":
            print(f"[忽略非polygon] {json_path.name}: {label}, type={shape_type}")
            continue

        if len(points) < 3:
            print(f"[忽略点数不足] {json_path.name}: {label}")
            continue

        cls_id = CLASSES[label]
        coords = normalize_polygon(points, w, h)
        line = str(cls_id) + " " + " ".join(f"{v:.8f}" for v in coords)
        lines.append(line)
        counter[label] += 1

    return lines, counter


def draw_visual_check(image_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None or not label_path.exists():
        return

    h, w = img.shape[:2]
    overlay = img.copy()

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    colors = {
        0: (0, 0, 255),      # plate 红色
        1: (0, 255, 0),      # paper 绿色
        2: (255, 0, 0),      # hole 蓝色
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

        cv2.polylines(overlay, [pts], True, color, 3)

        if len(pts) > 0:
            x0, y0 = pts[0]
            cv2.putText(
                overlay,
                name,
                (x0, max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def write_plate_yaml(out_dir: Path):
    yaml_text = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: plate
  1: paper
  2: hole
"""

    with open(out_dir / "plate.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)

    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("plate\npaper\nhole\n")


def write_train_script(out_dir: Path):
    text = '''from ultralytics import YOLO
import torch

# yolov8	YOLO 第 8 代模型
# n	nano，最小模型，速度快，占显存少
# seg	segmentation，分割模型，可以识别轮廓
# .pt	PyTorch 模型文件
# yolov8n-seg.pt  快，但效果最弱
# yolov8s-seg.pt  推荐起步，效果明显更好
# yolov8m-seg.pt  更慢，效果通常更好
# yolov8l-seg.pt  更吃显存
# yolov8x-seg.pt  最重
def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # model = YOLO("yolov8n-seg.pt")
    model = YOLO("yolov8s-seg.pt")

    model.train(
        # 告诉 YOLO，训练图片在哪里，验证图片在哪，有哪些类别
        data="plate.yaml",
        # 训练时把图片缩放到 1024 x 1024 左右再训练
        # 越大：细节更多，轮廓可能更准
        # 越大：越吃显存，训练越慢
        # 越小：显存压力小，速度快
        # 越小： 小孔、细边、凹凸细节可能识别差
        imgsz=1280,
        # 完整训练数据集 100 轮。
        # 表示把 1000 张图片重复训练 100 遍
        # 太少：模型还没学会
        # 合适：模型效果逐渐变好
        # 太多：可能过拟合，只记住训练图片，泛化变差
        epochs=100,
        # 一次同时拿 n 张图片给显卡训练。
        # 越大：训练可能更稳定，速度可能更快
        # 越大：越吃显存
        # 越小：显存压力小
        # 太小：训练速度慢一些
        batch=2,
        # 加载训练图片时，使用几个子进程并行读取图片。
        # workers 越大，读取图片越快
        # 但 Windows 上太大有时不稳定
        workers=0,
        device=0,
        # 训练结果保存到哪个大目录。
        # 训练日志
        # 训练曲线
        # 模型权重
        # 验证图片效果
        project="C:\\\\Projects\\\\yolo-out\\\\runs\\\\segment\\\\runs_plate2",
        # 训练任务的名字。
        name="plate_seg_v2",
        # 混合精度训练，省显存
        amp=True,
        # 不把数据集缓存到内存
        cache=False,
        # 如果连续 100 轮效果没提升，就提前停止
        patience=100,
        save_period=-1
    )


if __name__ == "__main__":
    main()
'''

    with open(out_dir / "train_plate_seg.py", "w", encoding="utf-8") as f:
        f.write(text)


def write_predict_script(out_dir: Path):
    text = '''from ultralytics import YOLO
import sys


def main():
    model_path = "runs_plate/plate_seg_v1/weights/best.pt"
    source = sys.argv[1] if len(sys.argv) > 1 else "images/val"

    model = YOLO(model_path)

    model.predict(
        source=source,
        imgsz=1280,
        conf=0.25,
        save=True,
        project="predict_out",
        name="vis",
    )


if __name__ == "__main__":
    main()
'''

    with open(out_dir / "predict_plate_seg.py", "w", encoding="utf-8") as f:
        f.write(text)


def zip_dir(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir.parent))


def clean_output_dir(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=False, default="./raw_images", help="原始图片目录")
    parser.add_argument("--json-dir", required=False, default="./labelme_json", help="LabelMe JSON目录")
    parser.add_argument("--out", required=False, default="./yolo-out", help="输出YOLO数据集目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip", action="store_true", help="是否打包zip")
    parser.add_argument("--clean", action="store_true", help="转换前是否清空输出目录")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out)

    if args.clean:
        clean_output_dir(out_dir)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"没有找到 JSON 文件: {json_dir}")

    random.seed(args.seed)
    random.shuffle(json_files)

    val_count = max(1, int(len(json_files) * args.val_ratio)) if len(json_files) > 1 else 0
    val_jsons = json_files[:val_count]
    train_jsons = json_files[val_count:]

    total_counter = {name: 0 for name in CLASSES.keys()}
    split_map = {"train": train_jsons, "val": val_jsons}

    for split, files in split_map.items():
        img_out_dir = out_dir / "images" / split
        label_out_dir = out_dir / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for json_path in files:
            image_path = find_image(json_path, image_dir)
            if image_path is None:
                print(f"[跳过] 找不到对应图片: {json_path.name}")
                continue

            lines, counter = convert_one_labelme_json(json_path, image_path)

            for k, v in counter.items():
                total_counter[k] += v

            if not lines:
                print(f"[警告] 没有有效标注: {json_path.name}")

            target_image = img_out_dir / image_path.name
            target_label = label_out_dir / f"{image_path.stem}.txt"

            shutil.copy2(image_path, target_image)

            with open(target_label, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            visual_path = out_dir / "visual_check" / f"{image_path.stem}_overlay.jpg"
            draw_visual_check(target_image, target_label, visual_path)

            print(
                f"[OK] {split}: {image_path.name} "
                f"plate={counter['plate']} paper={counter['paper']} hole={counter['hole']}"
            )

    write_plate_yaml(out_dir)
    write_train_script(out_dir)
    write_predict_script(out_dir)

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        zip_dir(out_dir, zip_path)
        print(f"[ZIP] {zip_path.resolve()}")

    print("\n统计:")
    for name, count in total_counter.items():
        print(f"  {name}: {count}")

    print(f"\n完成: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
