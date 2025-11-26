## 目标
- 修复 `AP_small` 计算的目录混淆，确保基线与 CBAM 评估各自读取对应的 `predictions.json`。
- 将脚本扩展为可参数化的“真实训练+评估”管线（数据、训练轮次、阈值、CBAM 位置/强度）。
- 以小目标为重点，优化 CBAM 的插入层、reduction、残差友好性与应用条件。
- 给出可直接运行的指令与消融实验流程，产出可信的对比与结论。

## 修复评估基线（AP_small 目录问题）
- 核心问题：`scripts/inject_cbam_yolov8.py:92–128` 使用 `latest_val_dir()`，在先后两次 `val()` 后，两个 `AP_small` 很可能都指向最新一次（CBAM）的 `predictions.json`。
- 修改方案：
  - 在 `val()` 时强制不同的 `project/name`：
    - 基线：`base.val(..., project='runs/detect', name='val_base', save_json=True)`（`scripts/inject_cbam_yolov8.py:79`）
    - CBAM：`cbam.val(..., project='runs/detect', name='val_cbam', save_json=True)`（`scripts/inject_cbam_yolov8.py:80`）
  - 直接使用返回对象的保存目录（若存在）：
    - `base_dt = os.path.join(getattr(mb, 'save_dir', 'runs/detect/val_base'), 'predictions.json')`
    - `cbam_dt = os.path.join(getattr(mc, 'save_dir', 'runs/detect/val_cbam'), 'predictions.json')`
  - 移除 `latest_val_dir()` 相关逻辑（`scripts/inject_cbam_yolov8.py:92–128`），避免目录竞态。
- 结果：`base_aps` 与 `cbam_aps` 分别且只读取各自的 `predictions.json`，对比可信。

## 扩展 CLI 与训练管线
- 在 `main()` 增加参数：
  - `--data`（默认不再强绑定 `coco128.yaml`）、`--epochs`、`--batch`、`--imgsz`、`--device`
  - 小目标相关评估阈值：`--conf`、`--iou`（用于 `val()` 与推理阶段扫阈值）
  - CBAM 控制：`--cbam_reduction`、`--cbam_min_size`、`--cbam_layers`（逗号分隔，如 `model.18,model.21`）、`--cbam_residual_scale`
- 训练调用改为真实数据：
  - 基线：`base.train(data=args.data, imgsz=args.imgsz, device=args.device, epochs=args.epochs, batch=args.batch, ...)`
  - CBAM：`cbam.train(... 同上 ...)`
- 验证统一传参与保存：
  - `base.val(data=args.data, imgsz=args.imgsz, device=args.device, conf=args.conf, iou=args.iou, project='runs/detect', name='val_base', save_json=True)`
  - `cbam.val(... name='val_cbam' ...)`

## CBAM 设计与实现微调
- 位置可配：将 `TARGET_C2F` 由常量改为 CLI 可传入（默认仍为 `{ 'model.18','model.21' }`）。
  - `scripts/inject_cbam_yolov8.py:11–21` 中 `replace_c2f_with_cbam` 读取 CLI 指定的集合。
- reduction 与 min_size：
  - `modules/cbam.py:24–41` 中 `C2fCBAMWrapper` 已支持 `reduction`（当前默认 64）；改为由 CLI 注入，默认 16；`min_size` 默认下调至能覆盖 P3（如 64），避免高分辨率层被误关。
- 残差友好性：
  - 在 `CBAM` 中加入残差缩放：`return x + s * (x * att - x)`，其中 `s` 来自 CLI（如 `--cbam_residual_scale=0.5`），或最简单 `return x + att_out`。
  - 目的：训练早期稳态，避免注意力过强破坏原表示。
- 轻量化开关：
  - 提供 `--cbam_layers=P3` 或 `P3,P4` 的语义映射（落到具体 `model.xx`），便于仅在最高分辨率层插入。

## 数据侧建议（执行步骤）
- 标签清洗：随机抽查 `train/val` 若干图，确认无漏标、类目一致、坐标规范。
- 小目标过采样：在 dataloader 中对含小目标的样本加权，或重复文件路径。
- Copy-Paste 增强：将小目标贴到大图；配合更强的随机缩放与多尺度训练。
- 分辨率提升：`imgsz=800/960`（视显存）；注意 FPS 下降，先离线冲精度。

## 训练策略（落地参数）
- 训练轮数：
  - 小数据集：`epochs=50–100`
  - 中/大数据集：`epochs=100–300`（可加 early stopping）
- 优化器与学习率：沿用 Ultralytics 默认（SGD/AdamW + cosine）；公平对比仅改结构不改超参。
- 小目标侧：适度提高 `box/cls` 权重；NMS 参数试验：`conf≈0.25`，`iou=0.5–0.6`。

## 推理与阈值扫描
- 在验证集上扫 `conf`：`0.20/0.25/0.30/0.35`，记录 `F1/mAP_small`；部署时按需求取更高 `conf`（少误报）或更低（少漏检）。
- 可在脚本中加入简单 sweep：循环调用 `model.val(conf=val, iou=args.iou, ...)`，输出曲线关键点。

## 消融与对比产出
- 先跑基线 YOLOv8（真实数据、`epochs≈100`），记录：`mAP/mAP_small/Precision/Recall/FPS`。
- 插入当前 CBAM（默认仅 P3/P4 两层），保持同参再跑一遍；
- 调 CBAM 超参与位置：`reduction=16→8→4`；位置 `P3` vs `P3+P4`；记录“消融表”；
- 从数据与训练侧再拉一次：提高分辨率（如 800）、做小目标采样/增强、调整 `conf/iou`，复评曲线；
- 以以上结果形成“精度提升趋势 + 定量对比”的阶段结论。

## 可直接执行的示例指令
- 基线：`python scripts/inject_cbam_yolov8.py --data path/to/data.yaml --epochs 100 --batch 16 --imgsz 640 --conf 0.25 --iou 0.6`
- CBAM：`python scripts/inject_cbam_yolov8.py --data path/to/data.yaml --epochs 100 --batch 16 --imgsz 640 --conf 0.25 --iou 0.6 --cbam_reduction 8 --cbam_layers model.18,model.21 --cbam_min_size 64 --cbam_residual_scale 0.5`

## 代码定位（便于修改）
- `scripts/inject_cbam_yolov8.py:79–80` 调整 `val()` 的 `project/name` 与 `save_json`；
- `scripts/inject_cbam_yolov8.py:92–128` 移除 `latest_val_dir()`，改为 `mb.save_dir` / `mc.save_dir` 读取两份 `predictions.json`；
- `scripts/inject_cbam_yolov8.py:60–66` 扩展 CLI；`scripts/inject_cbam_yolov8.py:11–21` 支持 `TARGET_C2F` 由 CLI 传入；
- `modules/cbam.py:4–23` 增加残差友好输出与缩放；`modules/cbam.py:24–41` 由 CLI 注入 `reduction/min_size`，默认对齐小目标场景。

## 交付与验证
- 修改脚本后：在真实数据集跑两套（基线/CBAM），核查 `predictions.json` 分别位于 `runs/detect/val_base` 与 `val_cbam`，并且 `coco_aps_small(...)` 读取对应文件。
- 输出统一 `summary`，包含 `mAP50-95/mAP50/mAP_small/Params/FPS`，并补充阈值扫描的 `F1` 最佳点。
- 完成后如需剪枝/量化，可在精度定稿后进行。