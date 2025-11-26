import argparse
import os
import sys
import time
import json
import re
import torch
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
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--cbam_layers', type=str, default='model.18,model.21')
    parser.add_argument('--cbam_reduction', type=int, default=16)
    parser.add_argument('--cbam_min_size', type=int, default=64)
    parser.add_argument('--cbam_residual_scale', type=float, default=0.0)
    parser.add_argument('--attn', type=str, default='cbam')
    parser.add_argument('--tr_heads', type=int, default=8)
    parser.add_argument('--tr_ff', type=int, default=None)
    parser.add_argument('--tr_dropout', type=float, default=0.1)
    parser.add_argument('--tr_min_size', type=int, default=64)
    parser.add_argument('--tr_residual_scale', type=float, default=1.0)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--sweep_confs', type=str, default='0.2,0.25,0.3,0.35')
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

    base_name = next_train_name('yolo')
    attn_label = 'yolo+transform' if args.attn == 'transformer' else 'yolo'
    cbam_name = next_train_name(attn_label)

    base.train(data=args.data, imgsz=args.imgsz, device=device_arg, epochs=args.epochs, batch=args.batch, project='exp', name=base_name, plots=False)
    cbam.train(data=args.data, imgsz=args.imgsz, device=device_arg, epochs=args.epochs, batch=args.batch, project='exp', name=cbam_name, plots=False)

    mb = base.val(data=args.data, imgsz=args.imgsz, device=device_arg, split='val', conf=args.conf, iou=args.iou, save_json=True, project='runs/detect', name=base_name)
    mc = cbam.val(data=args.data, imgsz=args.imgsz, device=device_arg, split='val', conf=args.conf, iou=args.iou, save_json=True, project='runs/detect', name=cbam_name)

    base_params = count_params(base.model)
    cbam_params = count_params(cbam.model)
    base_fps = measure_fps(base, imgsz=args.imgsz, device=args.device, iters=args.iters, batch=args.batch)
    cbam_fps = measure_fps(cbam, imgsz=args.imgsz, device=args.device, iters=args.iters, batch=args.batch)

    base_map = float(getattr(mb.box, 'map', 0.0)) if hasattr(mb, 'box') else 0.0
    base_map50 = float(getattr(mb.box, 'map50', 0.0)) if hasattr(mb, 'box') else 0.0
    cbam_map = float(getattr(mc.box, 'map', 0.0)) if hasattr(mc, 'box') else 0.0
    cbam_map50 = float(getattr(mc.box, 'map50', 0.0)) if hasattr(mc, 'box') else 0.0

    base_dt = None
    cbam_dt = None
    if hasattr(mb, 'save_dir'):
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
            rb = base.val(data=args.data, imgsz=args.imgsz, device=device_arg, split='val', conf=c, iou=args.iou, save_json=True, project='runs/detect', name=f'{base_name}_c{int(c*100)}')
            rbc = None
            bd = os.path.join(getattr(rb, 'save_dir', os.path.join('runs','detect',f'{base_name}_c{int(c*100)}')) , 'predictions.json')
            sm = None
            if gt_json_path and os.path.isfile(bd):
                sm = coco_aps_small(gt_json_path, bd)
            sweep_results['baseline'].append({'conf': c, 'mAP50-95': float(getattr(rb.box,'map',0.0)) if hasattr(rb,'box') else 0.0, 'mAP50': float(getattr(rb.box,'map50',0.0)) if hasattr(rb,'box') else 0.0, 'mAP_small': sm})
            rc = cbam.val(data=args.data, imgsz=args.imgsz, device=device_arg, split='val', conf=c, iou=args.iou, save_json=True, project='runs/detect', name=f'{cbam_name}_c{int(c*100)}')
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

if __name__ == '__main__':
    main()
