from ultralytics import YOLO
import torch


# ============================================================
# 钢板分割模型训练脚本
# 任务：YOLOv8 segmentation / 实例分割
# 模型：yolov8n-seg.pt
# 数据集配置：plate.yaml
# ============================================================

# yolov8：YOLO 第 8 代模型
# n：nano，最小模型，速度快，占显存少
# seg：segmentation，分割模型，可以识别轮廓
# .pt：PyTorch 模型权重文件
#
# 常见选择：
# yolov8n-seg.pt：最快，占显存最少，但效果最弱
# yolov8s-seg.pt：推荐起步，效果明显更好，显存压力中等
# yolov8m-seg.pt：更慢，效果通常更好
# yolov8l-seg.pt：更吃显存
# yolov8x-seg.pt：最重，对数据和显存要求最高


def print_device_info():
    """打印当前 CUDA / GPU 信息，方便确认是否用上显卡。"""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def main():
    print_device_info()

    # 加载预训练分割模型。
    # 当前使用 yolov8n-seg.pt，适合显存较小、先快速验证流程。
    # 如果后续追求效果，可以改成 yolov8s-seg.pt。
    model = YOLO("yolov8n-seg.pt")
    # model = YOLO("yolov8s-seg.pt")

    model.train(
        # -------------------------
        # 基础训练配置
        # -------------------------

        # 数据集配置文件，里面配置 train/val 路径和类别名称。
        data="plate.yaml",

        # 训练轮数。300 表示把整个训练集重复训练 300 遍。
        epochs=300,

        # 早停轮数。连续 80 轮验证效果没有提升，就提前停止训练。
        patience=80,

        # 输入图片尺寸。960 越大轮廓细节越多，但越吃显存。
        imgsz=960,

        # 批大小。batch=1 最省显存，适合 4GB 左右显卡。
        batch=1,

        # 使用第 0 张 GPU。没有 GPU 时可改成 device="cpu"。
        device=0,

        # Windows 下 workers=0 最稳，避免多进程读取数据出问题。
        workers=0,

        # 使用预训练权重继续训练，而不是从零开始。
        pretrained=True,

        # 优化器。AdamW 对小数据集通常比较稳。
        optimizer="AdamW",

        # 初始学习率。太大容易震荡，太小收敛慢。
        lr0=0.001,

        # 最终学习率比例。最终学习率 = lr0 * lrf。
        lrf=0.01,

        # 动量参数，影响梯度更新的平滑程度。
        momentum=0.937,

        # 权重衰减，用于抑制过拟合。
        weight_decay=0.0005,

        # 预热轮数，前 3 轮逐步升高学习率，训练更稳定。
        warmup_epochs=3.0,

        # 预热阶段的动量。
        warmup_momentum=0.8,

        # 预热阶段的 bias 学习率。
        warmup_bias_lr=0.1,

        # 是否使用余弦学习率。False 表示使用默认学习率策略。
        cos_lr=False,

        # 使用混合精度训练，省显存、速度更快。
        amp=True,

        # 是否启用确定性训练。True 更利于复现实验结果，但可能略慢。
        deterministic=True,

        # 随机种子，保证同配置下结果尽量可复现。
        seed=0,

        # 是否只按单类别训练。False 表示按 plate/paper/hole 等多类别训练。
        single_cls=False,

        # 是否使用矩形训练。False 更通用，True 有时更省显存。
        rect=False,

        # 是否恢复上一次训练。False 表示从当前模型重新开始。
        resume=False,

        # 是否只使用部分数据。1.0 表示使用全部训练数据。
        fraction=1.0,

        # 是否冻结部分网络层。None 表示不冻结。
        freeze=None,

        # 是否开启多尺度训练。0.0 表示关闭。
        multi_scale=0.0,

        # 是否编译模型加速。False 更稳，尤其是 Windows 环境。
        compile=False,

        # 是否进行性能分析。False 即可。
        profile=False,

        # -------------------------
        # 保存与输出配置
        # -------------------------

        # 训练结果保存的大目录。
        project=r"C:\Projects\yolo-out\runs\segment\runs_plate2",

        # 本次训练任务名称。
        name="plate_seg_v3-4",

        # 如果目录已存在是否覆盖。False 表示自动新建递增目录。
        exist_ok=False,

        # 是否保存最终权重。
        save=True,

        # 每隔 20 轮额外保存一次权重，方便回退到中间结果。
        save_period=20,

        # 是否生成训练曲线、PR 曲线、混淆矩阵等图片。
        plots=True,

        # 是否输出详细训练日志。
        verbose=True,

        # 是否缓存图片到内存/磁盘。False 更省内存，但训练读取会慢一点。
        cache=False,

        # -------------------------
        # 验证配置
        # -------------------------

        # 训练过程中是否执行验证。
        val=True,

        # 使用 data.yaml 中的 val 数据集进行验证。
        split="val",

        # 验证阶段的 IoU 阈值。
        iou=0.7,

        # 单张图片最多保留 300 个目标。
        max_det=300,

        # 是否保存 COCO JSON 结果。普通训练不用开。
        save_json=False,

        # 置信度阈值。None 表示使用默认值。
        conf=None,

        # 验证/推理时是否使用 FP16。训练已有 amp，这里保持 False。
        half=False,

        # 是否使用 OpenCV DNN 推理。训练时不需要。
        dnn=False,

        # 是否使用类别无关 NMS。False 表示不同类别分开处理。
        agnostic_nms=False,

        # 是否只训练/验证指定类别。None 表示使用全部类别。
        classes=None,

        # 是否使用高分辨率 mask。False 更省显存，True 轮廓可能更细。
        retina_masks=False,

        # -------------------------
        # 分割 mask 配置
        # -------------------------

        # 重叠目标的 mask 是否允许重叠。分割任务一般保持 True。
        overlap_mask=True,

        # mask 下采样比例。4 是默认常用值，越小 mask 越细但越吃显存。
        mask_ratio=4,

        # -------------------------
        # 损失权重配置
        # -------------------------

        # 框回归损失权重，影响定位框学习。
        box=7.5,

        # 分类损失权重，类别少时不宜过大。
        cls=0.5,

        # DFL 损失权重，影响边界框精细定位。
        dfl=1.5,

        # 姿态任务损失权重，分割任务基本不关心。
        pose=12.0,

        # 关键点目标损失权重，分割任务基本不关心。
        kobj=1.0,

        # RLE 相关权重，保留当前训练配置。
        rle=1.0,

        # 角度相关权重，保留当前训练配置。
        angle=1.0,

        # 名义 batch size，用于学习率归一化。
        nbs=64,

        # Dropout 比例。0.0 表示不开启。
        dropout=0.0,

        # -------------------------
        # 数据增强配置
        # -------------------------

        # 色相增强幅度。钢板颜色变化不大，保持较小。
        hsv_h=0.015,

        # 饱和度增强幅度，用于模拟不同光照/材质颜色。
        hsv_s=0.5,

        # 亮度增强幅度，用于模拟明暗变化、阴影。
        hsv_v=0.5,

        # 随机旋转角度。钢板拍摄角度变化时有帮助。
        degrees=8,

        # 随机平移比例。
        translate=0.08,

        # 随机缩放比例。
        scale=0.35,

        # 随机错切角度。
        shear=2,

        # 轻微透视增强，模拟拍照角度变化。
        perspective=0.0005,

        # 上下翻转概率。俯拍钢板一般可以开启。
        flipud=0.5,

        # 左右翻转概率。
        fliplr=0.5,

        # BGR 通道增强概率。0.0 表示关闭。
        bgr=0.0,

        # Mosaic 增强概率。小数据集有帮助，但过大可能影响真实轮廓。
        mosaic=0.3,

        # 最后 30 轮关闭 Mosaic，让模型回到真实图片分布。
        close_mosaic=30,

        # MixUp 增强概率。0.0 表示关闭，避免轮廓混乱。
        mixup=0.0,

        # CutMix 增强概率。0.0 表示关闭。
        cutmix=0.0,

        # Copy-Paste 增强概率。0.0 表示关闭。
        copy_paste=0.0,

        # Copy-Paste 的模式，保留默认 flip。
        copy_paste_mode="flip",

        # 自动增强策略。randaugment 是默认常用策略。
        auto_augment="randaugment",

        # 随机擦除比例。用于提升遮挡鲁棒性，但过大可能影响轮廓学习。
        erasing=0.4,

        # 自定义配置文件。None 表示不用额外 cfg。
        cfg=None,
    )


if __name__ == "__main__":
    main()
