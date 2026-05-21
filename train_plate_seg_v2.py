from ultralytics import YOLO
import torch


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="plate.yaml",

        imgsz=1280,
        epochs=300,
        batch=1,
        workers=0,
        device=0,

        project="C:\\Projects\\yolo-out\\runs\\segment\\runs_plate3",
        name="plate_seg_v3",
        exist_ok=False,

        pretrained=True,
        amp=True,
        cache=False,

        patience=80,

        # 优化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # 小数据集建议增强
        degrees=8,
        translate=0.08,
        scale=0.35,
        shear=2,
        perspective=0.0005,

        # 光照增强，针对车间反光、阴影、锈斑
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.5,

        # 钢板上下左右都可能出现，所以可以开
        fliplr=0.5,
        flipud=0.5,

        # 分割任务、小数据集，mosaic 不要太猛
        mosaic=0.3,
        close_mosaic=30,

        # 不建议开 mixup，钢板边界会被混合得更奇怪
        mixup=0.0,

        # 保存中间模型
        save=True,
        save_period=20,
    )


if __name__ == "__main__":
    main()