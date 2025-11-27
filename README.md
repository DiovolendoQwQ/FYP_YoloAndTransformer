# FYP：YOLOv8 + 注意力增强（CBAM / Transformer）训练与评估

本项目基于 Ultralytics YOLOv8，针对小目标检测场景集成了注意力增强模块（CBAM 与 Transformer），并提供可复现实验管线（训练、验证、阈值扫描、目录命名与结果汇总）。

## 主要特性
- 注意力模块可选：`--attn cbam` 或 `--attn transformer`
- 可参数化插入位置：在指定 `C2f` 层（默认 `model.18,model.21`，高分辨率特征）插入注意力
- 训练与验证统一 CLI：`--data/--epochs/--batch/--imgsz/--device/--conf/--iou`（默认 `conf=0.205`）
- 阈值扫描：`--sweep/--sweep_confs`，对 `conf` 进行扫描并输出关键指标与 `predictions.json`
- 评估目录修复：基线与增强模型的预测各自保存与读取，避免“最新目录覆盖”带来的混淆
- 目录命名自动化：
  - 单模型：`YYYY-MM-DD_训练模型_第N次训练`
  - 双模型统一根目录：启用 `--dual_out` 后生成 `YYYY-MM-DD_2模型训练_第N次训练`，子目录为 `yolo_epoch{N}`、`yolo+transform_epoch{N}`
- 数据集根路径：`--dataset_root f:/Project/FYP_YoloAndTransformer/datasets`，脚本自动生成并使用 `datasets/runtime_coco128.yaml`
- 仅注意力训练：`--only_attn` 跳过基线，直接训练并验证注意力版本

## 目录结构
- `scripts/inject_cbam_yolov8.py`：训练与评估主脚本（训练/验证/阈值扫描/目录命名/结果汇总）
- `modules/cbam.py`：CBAM 模块与 `C2fCBAMWrapper`（残差缩放可配，`reduction/min_size/residual_scale`）
- `modules/transformer.py`：Transformer 模块与 `C2fTransformerWrapper`（`heads/ff/dropout/min_size/residual_scale`）
- `exp/`：训练输出目录（权重与日志），自动命名为 `YYYY-MM-DD_训练模型_第N次训练`
- `runs/detect/`：验证输出目录（曲线与 `predictions.json`），与 `exp` 命名同步
- `更新日志.md`：每日更新记录（日期为标题，内容列出改动与进展）

## 环境准备（Windows）
- 创建并激活 GPU 环境（建议）：
  - `conda create -n yolo-cuda python=3.10 -y`
  - `conda activate yolo-cuda`
  - `conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
  - `pip install ultralytics pycocotools scipy`
- 验证 CUDA：
  - `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.__version__)"`

## 快速开始
- 激活环境：
  - `conda activate yolo-cuda`
- 基线训练（YOLO，默认 `conf=0.205`）：
  - `python scripts/inject_cbam_yolov8.py --epochs 100 --batch 16 --imgsz 640 --device 0 --dataset_root f:/Project/FYP_YoloAndTransformer/datasets`
- Transformer 增强：
  - `python scripts/inject_cbam_yolov8.py --epochs 100 --batch 16 --imgsz 640 --device 0 --attn transformer --cbam_layers model.18,model.21 --tr_heads 8 --tr_ff 256 --tr_dropout 0.1 --tr_min_size 64 --tr_residual_scale 1.0 --dataset_root f:/Project/FYP_YoloAndTransformer/datasets`
- 双模型统一输出（同参对比）：
  - `python scripts/inject_cbam_yolov8.py --epochs 200 --device 0 --attn transformer --dual_out --dataset_root f:/Project/FYP_YoloAndTransformer/datasets`
- 仅注意力训练：
  - `python scripts/inject_cbam_yolov8.py --epochs 200 --device 0 --attn transformer --only_attn --dataset_root f:/Project/FYP_YoloAndTransformer/datasets`
- 阈值扫描（验证集）：
  - `python scripts/inject_cbam_yolov8.py --epochs 1 --imgsz 640 --device 0 --sweep --sweep_confs 0.18,0.195,0.205,0.22 --dataset_root f:/Project/FYP_YoloAndTransformer/datasets`

## 训练与验证输出
- 单模型：`exp/YYYY-MM-DD_训练模型_第N次训练/`
  - `weights/best.pt`、`weights/last.pt`、`args.yaml`、`results.csv`
- 双模型统一输出：`exp/YYYY-MM-DD_2模型训练_第N次训练/`
  - 子目录：`yolo_epoch{E}`、`yolo+transform_epoch{E}`（`E` 为训练轮次）
  - 验证目录：`yolo_epoch{E}_val`、`yolo+transform_epoch{E}_val`
  - 根目录自动生成对比图：
    - `compare_train_box_loss.png / compare_val_box_loss.png`
    - `compare_train_cls_loss.png / compare_val_cls_loss.png`
    - `compare_train_dfl_loss.png / compare_val_dfl_loss.png`
    - `compare_precision.png / compare_recall.png / compare_map50.png / compare_map50-95.png`
  - 验证曲线与矩阵：各 `_val` 目录含 `BoxPR_curve.png / BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png / confusion_matrix.png / confusion_matrix_normalized.png`

## 注意力模块
- CBAM：
  - 封装：`modules/cbam.py:4-41`
  - 关键参数：`--cbam_reduction`（表达-参数折中）、`--cbam_min_size`（仅在高分辨率层启用）、`--cbam_residual_scale`（残差缩放，训练更稳）
- Transformer：
  - 封装：`modules/transformer.py:1-53`
  - 关键参数：`--tr_heads`（需整除通道数）、`--tr_ff`（前馈维度）、`--tr_dropout`、`--tr_min_size`、`--tr_residual_scale`
- 插入层位：使用 `--cbam_layers model.18,model.21` 指定（高分辨率层更利于小目标）

## 评估与阈值
- 默认 `conf=0.205`（在 `coco128` 上 F1 峰值约 `conf≈0.205`）
- 验证自动保存：基线与增强模型的 `predictions.json` 分别保存到各自目录，避免目录覆盖
- 阈值扫描：在验证集上扫 `conf`（如 `0.18/0.195/0.205/0.22`），选择最优 `F1/mAP_small`

## 提升路线（建议）
- 数据侧：小目标过采样、Copy-Paste 增强、提高分辨率（如 800/960）
- 训练侧：小数据 `epochs=50–100`，中/大数据 `epochs=100–300`；同参对比确保公平
- 模型侧：仅在 P3 或 P3+P4 插入注意力，做 `reduction/heads/ff/dropout/residual_scale` 消融，记录 `mAP/mAP_small/FPS`

## 更新日志
- 详见 `更新日志.md`（日期为标题，列出改动与进展）

## 参考代码位置
- 训练与评估主逻辑：`scripts/inject_cbam_yolov8.py`
- CBAM 模块：`modules/cbam.py:4-41`
- Transformer 模块：`modules/transformer.py:1-53`

---
如需将 CBAM 目录名区分为 `yolo+cbam`，或自动将每次训练摘要写入 `更新日志.md`，可以在脚本末尾追加写入逻辑。
