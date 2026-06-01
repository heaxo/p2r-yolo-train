import argparse
import json
import math
import random
import shutil
from pathlib import Path


DEFAULT_CLASSES = ["plate", "paper", "hole"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def read_classes(root: Path) -> list[str]:
    classes_path = root / "classes.txt"
    if not classes_path.exists():
        return DEFAULT_CLASSES

    names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return names or DEFAULT_CLASSES


def find_image(raw_dir: Path, json_data: dict, stem: str) -> Path | None:
    image_path = json_data.get("imagePath")
    candidates: list[Path] = []

    if image_path:
        path = Path(image_path)
        candidates.append(path if path.is_absolute() else raw_dir / path.name)

    for ext in IMAGE_EXTS:
        candidates.append(raw_dir / f"{stem}{ext}")
        candidates.append(raw_dir / f"{stem}{ext.upper()}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_points(points: list[list[float]], width: int, height: int) -> list[float]:
    normalized: list[float] = []
    for x, y in points:
        normalized.append(clamp01(float(x) / width))
        normalized.append(clamp01(float(y) / height))
    return normalized


def circle_to_polygon(points: list[list[float]], segments: int = 36) -> list[list[float]]:
    if len(points) < 2:
        return []

    cx, cy = points[0]
    px, py = points[1]
    radius = math.hypot(px - cx, py - cy)
    if radius <= 0:
        return []

    polygon = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        polygon.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
    return polygon


def rectangle_to_polygon(points: list[list[float]]) -> list[list[float]]:
    if len(points) < 2:
        return []

    x1, y1 = points[0]
    x2, y2 = points[1]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def shape_points(shape: dict) -> list[list[float]]:
    points = shape.get("points") or []
    shape_type = shape.get("shape_type") or "polygon"

    if shape_type == "polygon":
        return points
    if shape_type == "circle":
        return circle_to_polygon(points)
    if shape_type == "rectangle":
        return rectangle_to_polygon(points)

    return []


def convert_json(json_path: Path, class_to_id: dict[str, int]) -> tuple[list[str], int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"{json_path} missing valid imageWidth/imageHeight")

    lines: list[str] = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in class_to_id:
            raise ValueError(f"{json_path} has unknown label: {label!r}")

        points = shape_points(shape)
        if len(points) < 3:
            continue

        normalized = normalize_points(points, width, height)
        values = " ".join(f"{value:.8f}" for value in normalized)
        lines.append(f"{class_to_id[label]} {values}")

    return lines, width, height


def clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_plate_yaml(root: Path, names: list[str]) -> None:
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    lines.append("")
    (root / "plate.yaml").write_text("\n".join(lines), encoding="utf-8")


def split_items(items: list[tuple[str, Path, list[str]]], val_ratio: float, seed: int):
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)

    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items


def copy_items(items: list[tuple[str, Path, list[str]]], image_dir: Path, label_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for stem, image_path, label_lines in items:
        target_image = image_dir / image_path.name
        target_label = label_dir / f"{stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def assert_no_overlap(train_items, val_items) -> None:
    train_names = {stem for stem, _, _ in train_items}
    val_names = {stem for stem, _, _ in val_items}
    overlap = train_names & val_names
    if overlap:
        examples = ", ".join(sorted(overlap)[:10])
        raise RuntimeError(f"train/val overlap detected: {examples}")


def build_dataset(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    raw_dir = root / args.raw_dir
    json_dir = root / args.json_dir
    images_dir = root / args.images_dir
    labels_dir = root / args.labels_dir

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw image directory not found: {raw_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"labelme json directory not found: {json_dir}")

    names = read_classes(root)
    class_to_id = {name: index for index, name in enumerate(names)}

    items: list[tuple[str, Path, list[str]]] = []
    missing_images: list[str] = []
    empty_labels: list[str] = []

    for json_path in sorted(json_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = find_image(raw_dir, data, json_path.stem)
        if image_path is None:
            missing_images.append(json_path.name)
            continue

        label_lines, _, _ = convert_json(json_path, class_to_id)
        if not label_lines:
            empty_labels.append(json_path.name)
            continue

        items.append((json_path.stem, image_path, label_lines))

    if len(items) < 2:
        raise RuntimeError(f"not enough matched labeled images: {len(items)}")

    train_items, val_items = split_items(items, args.val_ratio, args.seed)
    assert_no_overlap(train_items, val_items)

    if args.clean:
        for path in [
            images_dir / "train",
            images_dir / "val",
            labels_dir / "train",
            labels_dir / "val",
        ]:
            clear_dir(path)

    copy_items(train_items, images_dir / "train", labels_dir / "train")
    copy_items(val_items, images_dir / "val", labels_dir / "val")

    if args.write_yaml:
        write_plate_yaml(root, names)

    print(f"classes: {names}")
    print(f"matched labeled images: {len(items)}")
    print(f"train: {len(train_items)} images, {len(train_items)} labels")
    print(f"val: {len(val_items)} images, {len(val_items)} labels")
    print("train/val overlap: 0")

    if missing_images:
        print(f"missing images skipped: {len(missing_images)}")
        print("  " + ", ".join(missing_images[:10]))
    if empty_labels:
        print(f"empty labels skipped: {len(empty_labels)}")
        print("  " + ", ".join(empty_labels[:10]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO segmentation images/labels from raw_images and LabelMe JSON files."
    )
    parser.add_argument("--root", default=".", help="project root directory")
    parser.add_argument("--raw-dir", default="raw_images", help="source image directory under root")
    parser.add_argument("--json-dir", default="labelme_json", help="LabelMe JSON directory under root")
    parser.add_argument("--images-dir", default="images", help="output images directory under root")
    parser.add_argument("--labels-dir", default="labels", help="output labels directory under root")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation split ratio")
    parser.add_argument("--seed", type=int, default=0, help="random split seed")
    parser.add_argument("--no-clean", action="store_true", help="do not clear existing train/val outputs first")
    parser.add_argument("--no-yaml", action="store_true", help="do not rewrite plate.yaml")
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    args.clean = not args.no_clean
    args.write_yaml = not args.no_yaml
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
