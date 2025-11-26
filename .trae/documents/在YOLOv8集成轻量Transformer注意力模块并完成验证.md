## 目标与交付物

* 集成一个轻量注意力模块（优先 CBAM，备选 Swin-Tiny）到 YOLOv8。

* 完成可运行的训练与推理，并与原生 YOLOv8 做小目标性能对比（如 APs/mAP\_small）。

* 交付：自定义模块代码、模型 YAML、训练与推理日志、基线对比结论。

## 环境准备（Windows / F 盘）(我已经安装了Anaconda)

* 创建独立环境：`conda create -p F:\envs\yolov8_cbam python=3.10 -y`；激活：`conda activate F:\envs\yolov8_cbam`。

* 安装 PyTorch CUDA 12.x：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`。

* 安装 Ultralytics：`pip install ultralytics`。

* 验证 GPU：在 Python 中运行 `import torch; torch.cuda.is_available(); torch.version.cuda; torch.cuda.get_device_name(0)`。

## 模块选择策略

* 首选 CBAM：参数与计算开销极低，插拔简单，适合在高分辨率层试验小目标增益。

* 备选 Swin-Tiny：在部分高分辨率层用极浅的 Stage（如单个窗口注意力块），观察速度影响后再决定是否保留。

## 模块实现（PyTorch nn.Module）

* 新增 `CBAM`，支持 `in_channels` 与 `reduction`，包含通道注意力与空间注意力，输出尺寸与输入一致，便于包裹 `C2f` 输出。

* 示例代码（后续将落地为文件）：

```python
import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False))
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        b, c, h, w = x.size()
        avg = torch.mean(x, dim=(2, 3), keepdim=False)
        maxv, _ = torch.max(x.view(b, c, -1), dim=2)
        mc = torch.sigmoid(self.mlp(avg) + self.mlp(maxv)).view(b, c, 1, 1)
        xs = torch.cat((torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]), dim=1)
        ms = torch.sigmoid(self.spatial(xs))
        return x * mc * ms
```

## 结构修改与嵌入位置

* 目标位置：

  * Backbone：在关键 `C2f` 或 `Bottleneck` 后加入 `CBAM`，优先靠近高分辨率特征流。

  * Neck（PAN-FPN）：在特征融合后的高分辨率输出（P3/P2，对应 80×80、160×160）处插入。小目标通常受益于此处的注意力。

* 两种集成方式：

  * 直接在 YAML 将 `CBAM` 作为层插入于 `C2f` 之后。

  * 定义 `C2f_CBAM` 包裹模块，在 `forward` 中先 `C2f` 再 `CBAM`。

## YAML 引用与注册

* 在 `ultralytics/nn/modules` 中注册 `CBAM`，确保 YAML 可识别。

* 自定义模型 YAML（示例片段，具体通道数按原模型对应）：

```yaml
backbone:
  - {from: -1, number: 1, module: Conv, args: [64, 3, 2]}
  - {from: -1, number: 1, module: C2f, args: [64, True]}
  - {from: -1, number: 1, module: CBAM, args: [64]}
  - {from: -1, number: 2, module: C2f, args: [128, True]}
neck:
  - {from: [x], number: 1, module: C2f, args: [96]}
  - {from: -1, number: 1, module: CBAM, args: [96]}
```

* 如出现 YAML 识别问题，则改用 `C2f_CBAM` 模块并在 YAML 中直接替换 `C2f` 为 `C2f_CBAM`。

## 正确调用与基本验证

* 预热推理：`yolo predict model=path\to\yolov8n-cbam.yaml source=path\to\images imgsz=640 device=0`，检查张量形状与速度。

* 模型构建自检：在 Python 导入自定义模块并 `model = YOLO('yolov8n-cbam.yaml')`，运行一次 dummy 输入 `torch.randn(1,3,640,640).cuda()`。

## 小规模训练与收敛观察

* 基线：原生 `yolov8n.yaml`，相同超参；试验模型：`yolov8n-cbam.yaml`。

* 统一设置：`imgsz=640 batch=16 epochs=10 workers=4 device=0 seed=0`。

* 命令：

  * 基线：`yolo train model=yolov8n.yaml data=path\to\data.yaml imgsz=640 batch=16 epochs=10 device=0 project=exp name=base`。

  * CBAM：`yolo train model=yolov8n-cbam.yaml data=path\to\data.yaml imgsz=640 batch=16 epochs=10 device=0 project=exp name=cbam`。

* 观察：损失曲线是否正常下降；GPU 显存与 step/s 速度是否无明显恶化。

## 小目标性能对比

* 若数据为 COCO 格式，使用 COCO evaluator 的 `APs`（小目标）作为 mAP\_small。

* Ultralytics `val`：`yolo val model=... data=... split=val device=0`，提取 `APs/APm/APl`（需要开启 COCO 面积分段评估；若默认未展示，则使用 pycocotools 在评估脚本中设定 `areaRng=[[0, 1024]]` 等分段）。

* 输出对比：记录基线与 CBAM 的 `APs`，并展示增益或持平结论；同时附上速度与显存变化。

## 稳定性与性能监控

* 训练过程中记录：`GPU memory`, `iter/s`, `epoch time`。

* 推理压力测：`yolo benchmark model=... imgsz=640 device=0` 或自行计时多张图片。

* 模型复杂度：使用 `thop` 或 Ultralytics 内置 `profile` 获取 FLOPs 与参数量，评估开销。

## 风险与回退策略

* 若 CBAM 引入与 YAML 注册出现报错，优先改用包裹式 `C2f_CBAM`。

* 若速度显著下降：减少插入点数量，仅在 P3 层保留；或降低 `reduction`。

* 若增益不明显：尝试在 Neck 的上采样前后插入；或切换 Swin-Tiny 的单窗口注意力块，仅在高分辨率层使用。

## 备选：Swin-Tiny 轻量集成路径

* 定义极简 `SwinBlockTiny`（单窗口注意力 + MLP），仅在 P3 层插入 1 次。

* 先验证形状与速度，再视情况替换 CBAM 或混合使用。

## 最终产出物

* `cbam.py` 或 `modules/attention/cbam.py`。

* 自定义 `yolov8n-cbam.yaml`。

* 训练与验证日志（含 APs、小目标结论、速度与显存对比）。

* 推理 Demo 图片或视频，用于肉眼检查小目标检测效果。

## 验证通过条件

* 训练、推理无报错；速度与显存在可接受范围；APs 有改善或至少持平；小目标可见的质变或稳态提升。

