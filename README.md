# mmyolo_yolo26

把 YOLO26 用 mmyolo 的代码风格重新写了一遍。

## 为什么做这个

Ultralytics 官方在 2026 年 1 月发布 YOLO26 时只给了 ultralytics 风格的实现，trainer / cfg / model 三者是耦合在一起的，习惯 mmyolo 写法的人迁过去要改不少东西。这个仓库做的事情就是把 YOLO26 的网络和训练流程，按 mmyolo 的模块拆分方式和命名习惯重新实现一遍——backbone、neck、head、assigner、loss 都拆成独立模块，detector 只负责把前后向串起来。同时不依赖 mmcv / mmyolo / ultralytics 任何一个，只依赖 PyTorch 和少量基础库。

适合以下情况：

- 项目本来基于 mmyolo 二开，想加 YOLO26 又不想换框架；
- 想读 mmyolo 风格代码弄懂 YOLO26 内部实现；
- 受限环境装不上 mmcv，但又想跑 YOLO26 训练。

只覆盖目标检测。实例分割、姿态、OBB 这些 YOLO26 也支持的任务不在范围内。

## YOLO26 相对前代的改动

本仓库实际复现的改动：

- **Backbone**：CSP-Darknet 整体保留，stage 内的 block 从 C2f/C3k 换成 `C3K2v2`，最后一阶段接 `SPPF26` 和 `C2PSA`（带 PSA 注意力的 CSP 模块）。
- **Neck**：PAFPN 结构，top-down 和 bottom-up 都用 `C3k2For26`，最高层 P5 的 bottom-up 块开 PSA 注意力，reduce 层为 Identity，不再额外加 1×1 conv。
- **Head（标准版）**：解耦头 + DFL（reg_max=16），分类分支用 DepthwiseSeparableConv，整体沿用 YOLOv11 风格。
- **Head（NMS-Free 版）**：双分支结构，one2many 分支 topk=10 提供密集监督，one2one 分支 topk=7 + topk2=1 负责推理端的端到端输出；回归直接预测 ltrb，不再走 DFL。
- **Assigner**：`BatchTaskAlignedAssigner`，相比普通 TAL 多了 STAL——按 stride 放大候选框后再做一轮 topk2 过滤，提升小目标分配质量。
- **ProgLoss**：one2many 的 loss 权重随训练 epoch 线性衰减（默认 0.8 → 0.1），one2one 权重同步上升，训练后期模型能力逐步迁移到端到端分支。
- **训练侧**：SGD + paramwise lr 分组（weight / bn / bias 三组 lr 不同）、linear warmup + cosine、EMA、AMP。这些是 YOLO 系一贯做法，不是 YOLO26 新增，但一并实现了。

官方在 COCO 上的参考精度（仅用于说明 YOLO26 本身的性能水平，本仓库未复现该精度）：

| 模型 | mAP50-95 | T4 TensorRT | 参数量 |
|------|----------|-------------|--------|
| YOLO26n | 40.9 | 1.7 ms | 2.4 M |
| YOLO26s | 48.6 | 2.5 ms | 9.5 M |
| YOLO26l | 55.0 | 6.2 ms | 24.8 M |
| YOLO26x | 57.5 | 11.8 ms | 55.7 M |

## 项目结构

```
mmyolo_yolo26/
├── configs/
│   ├── det_tiny.yaml             # 标准NMS版, n (widen=0.25)
│   ├── det_small.yaml            # 标准NMS版, s (widen=0.5)
│   ├── det_large.yaml            # 标准NMS版, l (widen=1.0)
│   ├── det_tiny_nmsfree.yaml     # NMS-Free版, n
│   ├── det_small_nmsfree.yaml    # NMS-Free版, s
│   └── det_large_nmsfree.yaml    # NMS-Free版, l
├── yolo26/
│   ├── models/
│   │   ├── backbone.py           # YOLO26CSPDarknet
│   │   ├── neck.py               # YOLO26PAFPN
│   │   ├── head.py               # YOLOv11HeadModule / YOLO26NMSFreeHeadModule
│   │   ├── detectors.py          # YOLO26 / YOLO26NMSFree + YOLO26Head / YOLO26NMSFreeHead
│   │   ├── assigner.py           # BatchTaskAlignedAssigner (STAL topk2)
│   │   ├── losses.py             # IoULoss(CIoU) / DistributionFocalLoss / BCE
│   │   ├── coder.py              # MlvlPointGenerator / DistancePointBBoxCoder
│   │   ├── common.py             # Conv / C3K2v2 / SPPF26 / C2PSA / C3k2For26 等基础模块
│   │   └── postprocess.py        # multiclass_nms
│   ├── data/
│   │   ├── dataset.py            # DetDataset + collate_fn
│   │   └── transforms.py         # JointResize / JointRandomCrop / Normalize 等
│   └── utils/
│       ├── checkpoint.py
│       └── metric.py             # COCO风格mAP
├── tools/
│   ├── train.py                  # 训练入口
│   └── smoke_test.py             # 前向/反向/推理冒烟测试
└── requirements.txt
```

## 安装

```bash
git clone <your-repo-url> mmyolo_yolo26
cd mmyolo_yolo26
pip install -r requirements.txt
```

依赖只有 `torch`、`torchvision`、`numpy`、`opencv-python`、`pyyaml`、`tqdm`。

## 数据格式

标注按 labelme 风格的 JSON 文件 + 一个 CSV 索引组织，目录结构如下：

```
dataset_root/
├── Annotations/
│   ├── class_names_list.txt     # 每行一个类名, 会自动追加 "ok" 类
│   ├── train.csv                # file_name, anno_label, anno_file
│   ├── val.csv
│   ├── xxx.json                 # labelme 风格, shapes[].points / bbox
│   └── ...
└── images/
    └── xxx.png
```

`train.csv` 每行三列：`图片相对路径, 标签类型(defect/ok), 标注JSON相对路径`。label 为 `ok` 的样本在训练集默认丢弃、在验证集保留用于计算负样本召回。需要换 COCO / VOC 等格式的话，自己写个 dataset 继承一下即可。

## 使用

训练（标准 NMS 版）：

```bash
python tools/train.py \
    --config configs/det_tiny.yaml \
    --data-path /path/to/dataset \
    --work-dir /path/to/work \
    --epochs 100 --batch-size 8 --img-size 1024 1024 --lr 0.01
```

训练（NMS-Free 版）：

```bash
python tools/train.py \
    --config configs/det_tiny_nmsfree.yaml \
    --data-path /path/to/dataset \
    --work-dir /path/to/work \
    --epochs 100 --batch-size 8 --img-size 1024 1024 --lr 0.01
```

## 各规模参数量

实测参数量（`num_classes=5`）：

| 配置 | 类型 | 参数量 |
|------|------|--------|
| det_tiny | YOLO26 (NMS) | 2.71 M |
| det_small | YOLO26 (NMS) | 9.85 M |
| det_large | YOLO26 (NMS) | 24.79 M |
| det_tiny_nmsfree | YOLO26NMSFree | 2.52 M |
| det_small_nmsfree | YOLO26NMSFree | 9.96 M |
| det_large_nmsfree | YOLO26NMSFree | 26.18 M |

n / s / l 三档和官方 YOLO26 对应（官方 YOLO26n 2.4M、YOLO26s 9.5M、YOLO26l 24.8M，差异主要来自 head 的 `num_classes` 和 DFL 通道数）。
