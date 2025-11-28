import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import time
import json
import re
import torch
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.cbam import C2fCBAMWrapper
from modules.transformer import C2fTransformerWrapper

TARGET_C2F = { 'model.18', 'model.21' }

def replace_c2f_with_cbam(m, target_set, reduction, min_size, residual_scale, prefix=''):
    for name, child in list(m.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if child.__class__.__name__ == 'C2f' and full in target_set:
            setattr(m, name, C2fCBAMWrapper(child, min_size=min_size, reduction=reduction, residual_scale=residual_scale))
        else:
            replace_c2f_with_cbam(child, target_set, reduction, min_size, residual_scale, full)

def build_cbam_model(layers, reduction, min_size, residual_scale):
    model = YOLO('yolov8n.pt')
    replace_c2f_with_cbam(model.model, layers, reduction, min_size, residual_scale)
    return model

def replace_c2f_with_transformer(m, target_set, heads, dim_ff, dropout, min_size, residual_scale, prefix=''):
    for name, child in list(m.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if child.__class__.__name__ == 'C2f' and full in target_set:
            setattr(m, name, C2fTransformerWrapper(child, heads=heads, dim_ff=dim_ff, dropout=dropout, min_size=min_size, residual_scale=residual_scale))
        else:
            replace_c2f_with_transformer(child, target_set, heads, dim_ff, dropout, min_size, residual_scale, full)

def build_transformer_model(layers, heads, dim_ff, dropout, min_size, residual_scale):
    model = YOLO('yolov8n.pt')
    replace_c2f_with_transformer(model.model, layers, heads, dim_ff, dropout, min_size, residual_scale)
    return model

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def measure_fps(model, imgsz=640, device=0, iters=50, batch=8):
    device_str = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    model.model.to(device_str).eval()
    x = torch.randn(batch, 3, imgsz, imgsz, device=device_str)
    with torch.no_grad():
        for _ in range(5):
            _ = model.model(x)
        start = time.time()
        for _ in range(iters):
            _ = model.model(x)
        elapsed = time.time() - start
    total_imgs = iters * batch
    fps = total_imgs / elapsed if elapsed > 0 else 0.0
    return fps

def coco_aps_small(gt_json, dt_json):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        cocoGt = COCO(gt_json)
        cocoDt = cocoGt.loadRes(dt_json)
        e = COCOeval(cocoGt, cocoDt, 'bbox')
        e.params.imgIds = sorted(cocoGt.getImgIds())
        e.evaluate(); e.accumulate(); e.summarize()
        return float(e.stats[3])
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='coco128.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--conf', type=float, default=0.2)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--cbam_layers', type=str, default='model.10,model.13,model.18')
    parser.add_argument('--cbam_reduction', type=int, default=16)
    parser.add_argument('--cbam_min_size', type=int, default=64)
    parser.add_argument('--cbam_residual_scale', type=float, default=0.0)
    parser.add_argument('--attn', type=str, default='cbam')
    parser.add_argument('--tr_heads', type=int, default=8)
    parser.add_argument('--tr_ff', type=int, default=None)
    parser.add_argument('--tr_dropout', type=float, default=0.1)
    parser.add_argument('--tr_min_size', type=int, default=32)
    parser.add_argument('--tr_residual_scale', type=float, default=1.0)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--sweep_confs', type=str, default='0.2,0.25,0.3,0.35')
    parser.add_argument('--only_attn', action='store_true', default=True)
    parser.add_argument('--hyp', type=str, default=None)
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--dual_out', action='store_true')
    args = parser.parse_args()
    device_arg = args.device if torch.cuda.is_available() else 'cpu'

    base = YOLO('yolov8n.pt')
    cbam_layers = {s.strip() for s in args.cbam_layers.split(',') if s.strip()}
    if args.attn == 'transformer':
        cbam = build_transformer_model(cbam_layers, args.tr_heads, args.tr_ff, args.tr_dropout, args.tr_min_size, args.tr_residual_scale)
    else:
        cbam = build_cbam_model(cbam_layers, args.cbam_reduction, args.cbam_min_size, args.cbam_residual_scale)

    device_str = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(1, 3, args.imgsz, args.imgsz).to(device_str)
    cbam.model.to(device_str)
    with torch.no_grad():
        _ = cbam.model(x)

    def next_train_name(model_label):
        date_str = time.strftime('%Y-%m-%d')
        prefix = f"{date_str}_{model_label}_第"
        project_dir = os.path.join(os.getcwd(), 'exp')
        idx = 1
        if os.path.isdir(project_dir):
            for d in os.listdir(project_dir):
                if d.startswith(f"{date_str}_{model_label}_第"):
                    m = re.search(r"第(\d+)次训练", d)
                    if m:
                        try:
                            idx = max(idx, int(m.group(1)) + 1)
                        except Exception:
                            pass
        return f"{date_str}_{model_label}_第{idx}次训练"

    def next_dual_root():
        date_str = time.strftime('%Y-%m-%d')
        project_dir = os.path.join(os.getcwd(), 'exp')
        idx = 1
        if os.path.isdir(project_dir):
            for d in os.listdir(project_dir):
                if d.startswith(f"{date_str}_2模型训练_第"):
                    m = re.search(r"第(\d+)次训练", d)
                    if m:
                        try:
                            idx = max(idx, int(m.group(1)) + 1)
                        except Exception:
                            pass
        return f"{date_str}_2模型训练_第{idx}次训练"

    base_name = next_train_name('yolo')
    attn_label = 'yolo+transform' if args.attn == 'transformer' else 'yolo'
    cbam_name = next_train_name(attn_label)

    root_project = os.path.join(os.getcwd(), 'exp', next_dual_root()) if args.dual_out else os.path.join(os.getcwd(), 'exp')
    yolo_run_name = f"yolo_epoch{args.epochs}" if args.dual_out else base_name
    attn_run_name = f"{attn_label}_epoch{args.epochs}" if args.dual_out else cbam_name

    data_arg = args.data
    if args.dataset_root:
        root = os.path.normpath(args.dataset_root)
        coco_path = os.path.join(root, 'coco128')
        yaml_path = os.path.join(os.getcwd(), 'datasets', 'runtime_coco128.yaml')
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        names = [
            'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
            'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
            'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
            'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
            'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
            'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
        ]
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {coco_path}\n")
            f.write("train: images/train2017\n")
            f.write("val: images/train2017\n")
            f.write("names:\n")
            for i, n in enumerate(names):
                f.write(f"  {i}: {n}\n")
        data_arg = yaml_path

    extra_kwargs = {}
    if args.hyp:
        try:
            import yaml
            with open(args.hyp, 'r', encoding='utf-8') as f:
                hv = yaml.safe_load(f) or {}
            if isinstance(hv, dict):
                extra_kwargs.update(hv)
        except Exception:
            pass
    if args.only_attn:
        cbam.train(data=data_arg, imgsz=args.imgsz, device=device_arg, epochs=args.epochs, batch=args.batch, project=root_project, name=attn_run_name, plots=True, **extra_kwargs)
    else:
        base.train(data=data_arg, imgsz=args.imgsz, device=device_arg, epochs=args.epochs, batch=args.batch, project=root_project, name=yolo_run_name, plots=True, **extra_kwargs)
        cbam.train(data=data_arg, imgsz=args.imgsz, device=device_arg, epochs=args.epochs, batch=args.batch, project=root_project, name=attn_run_name, plots=True, **extra_kwargs)

    mb = None
    if not args.only_attn:
        mb = base.val(data=data_arg, imgsz=args.imgsz, device=device_arg, split='val', conf=args.conf, iou=args.iou, save_json=True, project=root_project, name=f'{yolo_run_name}_val', plots=True)
    mc = cbam.val(data=data_arg, imgsz=args.imgsz, device=device_arg, split='val', conf=args.conf, iou=args.iou, save_json=True, project=root_project, name=f'{attn_run_name}_val', plots=True)

    base_params = count_params(base.model)
    cbam_params = count_params(cbam.model)
    base_fps = None if args.only_attn else measure_fps(base, imgsz=args.imgsz, device=args.device, iters=args.iters, batch=args.batch)
    cbam_fps = measure_fps(cbam, imgsz=args.imgsz, device=args.device, iters=args.iters, batch=args.batch)

    base_map = float(getattr(mb.box, 'map', 0.0)) if (mb is not None and hasattr(mb, 'box')) else 0.0
    base_map50 = float(getattr(mb.box, 'map50', 0.0)) if (mb is not None and hasattr(mb, 'box')) else 0.0
    cbam_map = float(getattr(mc.box, 'map', 0.0)) if hasattr(mc, 'box') else 0.0
    cbam_map50 = float(getattr(mc.box, 'map50', 0.0)) if hasattr(mc, 'box') else 0.0

    base_dt = None
    cbam_dt = None
    if mb is not None and hasattr(mb, 'save_dir'):
        p = os.path.join(mb.save_dir, 'predictions.json')
        if os.path.isfile(p):
            base_dt = p
    else:
        p = os.path.join('runs', 'detect', 'val_base', 'predictions.json')
        if os.path.isfile(p):
            base_dt = p
    if hasattr(mc, 'save_dir'):
        p2 = os.path.join(mc.save_dir, 'predictions.json')
        if os.path.isfile(p2):
            cbam_dt = p2
    else:
        p2 = os.path.join('runs', 'detect', 'val_cbam', 'predictions.json')
        if os.path.isfile(p2):
            cbam_dt = p2

    # Try common coco128 gt path
    gt_json_path = None
    for fname in (
        os.path.join('datasets', 'coco128', 'annotations', 'instances_train2017.json'),
        os.path.join('datasets', 'coco128', 'annotations', 'instances_val2017.json'),
    ):
        if os.path.isfile(fname):
            gt_json_path = fname
            break

    base_aps = None
    cbam_aps = None
    if gt_json_path and base_dt:
        base_aps = coco_aps_small(gt_json_path, base_dt)
    if gt_json_path and cbam_dt:
        cbam_aps = coco_aps_small(gt_json_path, cbam_dt)

    sweep_results = None
    if args.sweep:
        sweep_results = {
            'baseline': [],
            'cbam': []
        }
        confs = [float(s.strip()) for s in args.sweep_confs.split(',') if s.strip()]
        for c in confs:
            if not args.only_attn:
                rb = base.val(data=data_arg, imgsz=args.imgsz, device=device_arg, split='val', conf=c, iou=args.iou, save_json=True, project='runs/detect', name=f'{base_name}_c{int(c*100)}')
                bd = os.path.join(getattr(rb, 'save_dir', os.path.join('runs','detect',f'{base_name}_c{int(c*100)}')) , 'predictions.json')
                sm = None
                if gt_json_path and os.path.isfile(bd):
                    sm = coco_aps_small(gt_json_path, bd)
                sweep_results['baseline'].append({'conf': c, 'mAP50-95': float(getattr(rb.box,'map',0.0)) if hasattr(rb,'box') else 0.0, 'mAP50': float(getattr(rb.box,'map50',0.0)) if hasattr(rb,'box') else 0.0, 'mAP_small': sm})
            rc = cbam.val(data=data_arg, imgsz=args.imgsz, device=device_arg, split='val', conf=c, iou=args.iou, save_json=True, project='runs/detect', name=f'{cbam_name}_c{int(c*100)}')
            cd = os.path.join(getattr(rc, 'save_dir', os.path.join('runs','detect',f'{cbam_name}_c{int(c*100)}')) , 'predictions.json')
            smc = None
            if gt_json_path and os.path.isfile(cd):
                smc = coco_aps_small(gt_json_path, cd)
            sweep_results['cbam'].append({'conf': c, 'mAP50-95': float(getattr(rc.box,'map',0.0)) if hasattr(rc,'box') else 0.0, 'mAP50': float(getattr(rc.box,'map50',0.0)) if hasattr(rc,'box') else 0.0, 'mAP_small': smc})

    summary = {
        'settings': {
            'data': args.data,
            'imgsz': args.imgsz,
            'device': args.device,
            'epochs': args.epochs,
            'batch': args.batch,
            'conf': args.conf,
            'iou': args.iou,
            'cbam_layers': list(cbam_layers),
            'cbam_reduction': args.cbam_reduction,
            'cbam_min_size': args.cbam_min_size,
            'cbam_residual_scale': args.cbam_residual_scale,
            'attn': args.attn,
            'tr_heads': args.tr_heads,
            'tr_ff': args.tr_ff,
            'tr_dropout': args.tr_dropout,
            'tr_min_size': args.tr_min_size,
            'tr_residual_scale': args.tr_residual_scale,
        },
        'baseline': {
            'mAP50-95': base_map,
            'mAP50': base_map50,
            'mAP_small': base_aps,
            'Params': base_params,
            'FPS': base_fps,
        },
        'cbam': {
            'mAP50-95': cbam_map,
            'mAP50': cbam_map50,
            'mAP_small': cbam_aps,
            'Params': cbam_params,
            'FPS': cbam_fps,
        },
        'sweep': sweep_results
    }
    print(json.dumps(summary, indent=2))

    def _read_results_csv(csv_path):
        xs = {}
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                for k, v in row.items():
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    xs.setdefault(k, []).append(val)
        return xs

    def _plot_two_curves(out_png, title, ylabel, x_values, y1, y2, label1='yolo', label2='yolo+transform'):
        plt.figure(figsize=(8,5))
        plt.plot(x_values, y1, label=label1)
        plt.plot(x_values, y2, label=label2)
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

    if args.dual_out and (not args.only_attn):
        yolo_dir = os.path.join(root_project, yolo_run_name)
        attn_dir = os.path.join(root_project, attn_run_name)
        y_csv = os.path.join(yolo_dir, 'results.csv')
        a_csv = os.path.join(attn_dir, 'results.csv')
        if os.path.isfile(y_csv) and os.path.isfile(a_csv):
            y = _read_results_csv(y_csv)
            a = _read_results_csv(a_csv)
            epochs_list = list(range(1, 1 + min(len(y.get('train/box_loss', [])), len(a.get('train/box_loss', [])))))
            out_dir = Path(root_project)
            _plot_two_curves(out_dir / 'compare_train_box_loss.png', 'Train Box Loss', 'loss', epochs_list, y.get('train/box_loss', []), a.get('train/box_loss', []))
            _plot_two_curves(out_dir / 'compare_val_box_loss.png', 'Val Box Loss', 'loss', epochs_list, y.get('val/box_loss', []), a.get('val/box_loss', []))
            _plot_two_curves(out_dir / 'compare_train_cls_loss.png', 'Train Cls Loss', 'loss', epochs_list, y.get('train/cls_loss', []), a.get('train/cls_loss', []))
            _plot_two_curves(out_dir / 'compare_val_cls_loss.png', 'Val Cls Loss', 'loss', epochs_list, y.get('val/cls_loss', []), a.get('val/cls_loss', []))
            _plot_two_curves(out_dir / 'compare_train_dfl_loss.png', 'Train DFL Loss', 'loss', epochs_list, y.get('train/dfl_loss', []), a.get('train/dfl_loss', []))
            _plot_two_curves(out_dir / 'compare_val_dfl_loss.png', 'Val DFL Loss', 'loss', epochs_list, y.get('val/dfl_loss', []), a.get('val/dfl_loss', []))
            _plot_two_curves(out_dir / 'compare_precision.png', 'Precision vs Epoch', 'precision', epochs_list, y.get('metrics/precision', []), a.get('metrics/precision', []))
            _plot_two_curves(out_dir / 'compare_recall.png', 'Recall vs Epoch', 'recall', epochs_list, y.get('metrics/recall', []), a.get('metrics/recall', []))
            _plot_two_curves(out_dir / 'compare_map50.png', 'mAP50 vs Epoch', 'mAP50', epochs_list, y.get('metrics/mAP50', []), a.get('metrics/mAP50', []))
            _plot_two_curves(out_dir / 'compare_map50-95.png', 'mAP50-95 vs Epoch', 'mAP50-95', epochs_list, y.get('metrics/mAP50-95', []), a.get('metrics/mAP50-95', []))

        # copy val visual plots into common root
        for name, label in ((f'{yolo_run_name}_val', 'yolo'), (f'{attn_run_name}_val', 'yolo+transform')):
            val_dir = Path(root_project) / name
            if val_dir.exists():
                for src, dst in (
                    ('PR_curve.png', f'{label}_PR_curve.png'),
                    ('F1_curve.png', f'{label}_F1_conf_curve.png'),
                    ('P_curve.png', f'{label}_Precision_conf_curve.png'),
                    ('R_curve.png', f'{label}_Recall_conf_curve.png'),
                    ('confusion_matrix.png', f'{label}_confusion_matrix.png'),
                    ('confusion_matrix_raw.png', f'{label}_confusion_matrix_raw.png'),
                ):
                    p = val_dir / src
                    if p.exists():
                        try:
                            (Path(root_project) / dst).write_bytes(p.read_bytes())
                        except Exception:
                            pass

if __name__ == '__main__':
    main()
