import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.transformer import C2fTransformerWrapper


def replace_c2f_with_transformer(m, target_set, heads, dim_ff, dropout, min_size, residual_scale, prefix=''):
    for name, child in list(m.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if child.__class__.__name__ == 'C2f' and full in target_set:
            setattr(m, name, C2fTransformerWrapper(child, heads=heads, dim_ff=dim_ff, dropout=dropout, min_size=min_size, residual_scale=residual_scale))
        else:
            replace_c2f_with_transformer(child, target_set, heads, dim_ff, dropout, min_size, residual_scale, full)


def build_transformer_model(base_weights, layers, heads, dim_ff, dropout, min_size, residual_scale):
    model = YOLO(base_weights)
    replace_c2f_with_transformer(model.model, layers, heads, dim_ff, dropout, min_size, residual_scale)
    return model


def load_image(path):
    img = cv2.imread(path)
    return img


def boxes_area_xyxy(boxes):
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    return wh[:, 0] * wh[:, 1]


def filter_big_boxes(xyxy, img_w, img_h, area_ratio, abs_thresh):
    areas = boxes_area_xyxy(xyxy)
    thresh = abs_thresh if abs_thresh is not None else img_w * img_h * area_ratio
    keep = areas >= thresh
    return xyxy[keep]


def pad_boxes(xyxy, img_w, img_h, padding):
    out = xyxy.clone()
    out[:, 0] = torch.clamp(out[:, 0] - padding, min=0, max=img_w - 1)
    out[:, 1] = torch.clamp(out[:, 1] - padding, min=0, max=img_h - 1)
    out[:, 2] = torch.clamp(out[:, 2] + padding, min=0, max=img_w - 1)
    out[:, 3] = torch.clamp(out[:, 3] + padding, min=0, max=img_h - 1)
    return out


def make_mask(img, boxes):
    h, w = img.shape[:2]
    m = np.ones((h, w), dtype=np.uint8) * 255
    for b in boxes.cpu().numpy().astype(np.int32):
        x1, y1, x2, y2 = b.tolist()
        cv2.rectangle(m, (x1, y1), (x2, y2), 0, -1)
    return m


def generate_tiles(img_w, img_h, tile_size, stride):
    tiles = []
    for y in range(0, max(img_h - tile_size, 0) + 1, stride):
        for x in range(0, max(img_w - tile_size, 0) + 1, stride):
            tiles.append((x, y, x + tile_size, y + tile_size))
    if img_w < tile_size or img_h < tile_size:
        tiles = [(0, 0, img_w, img_h)]
    return tiles


def iou_xyxy(a, b):
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = torch.clamp(br - tl, min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = boxes_area_xyxy(a)
    area_b = boxes_area_xyxy(b)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / torch.clamp(union, min=1e-6)


def skip_tiles_by_overlap(tiles, big_boxes, thresh):
    kept = []
    if len(big_boxes) == 0:
        return tiles
    b = torch.tensor(tiles, dtype=torch.float32, device=big_boxes.device)
    i = iou_xyxy(b, big_boxes)
    mx = i.max(dim=1).values
    for idx, t in enumerate(tiles):
        if float(mx[idx].cpu()) <= thresh:
            kept.append(t)
    return kept


def predict_yolo(model, img, imgsz, conf, iou, device):
    return model.predict(img, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)


def results_to_tensors(res):
    r = res[0]
    xyxy = r.boxes.xyxy
    scores = r.boxes.conf
    cls = r.boxes.cls
    return xyxy, scores, cls


def offset_boxes(xyxy, dx, dy):
    out = xyxy.clone()
    out[:, 0] += dx
    out[:, 1] += dy
    out[:, 2] += dx
    out[:, 3] += dy
    return out


def batched_nms_per_class(xyxy, scores, cls, iou_thresh):
    c = cls.int()
    keep = torchvision.ops.batched_nms(xyxy, scores, c, iou_thresh)
    return keep


def draw_boxes(img, boxes, color=(0, 255, 0)):
    out = img.copy()
    for b in boxes.cpu().numpy().astype(np.int32):
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), color, 2)
    return out


def run_inference_pipeline(
    source,
    weights_yolo='yolov8n.pt',
    use_transformer=False,
    weights_trans_base='yolov8n.pt',
    layers='model.10,model.13,model.18',
    tile_size=640,
    tile_stride=320,
    imgsz=640,
    conf=0.25,
    iou=0.6,
    area_ratio=0.1,
    abs_area=None,
    padding=8,
    tile_skip_iou=0.4,
    device=None,
    half=False,
    heads=8,
    dim_ff=None,
    dropout=0.1,
    min_size=64,
    residual_scale=1.0,
    use_mask=False,
    residual_imgsz=640,
    tile_batch=4
):
    dev = device if device else (f"cuda:0" if torch.cuda.is_available() else "cpu")
    base = YOLO(weights_yolo)

    if half and torch.cuda.is_available():
        base.model.half()

    if isinstance(source, str):
        img = load_image(source)
    else:
        img = source # Assume numpy array

    if img is None:
        raise ValueError("Could not load image")

    h, w = img.shape[:2]
    r = predict_yolo(base, img, imgsz, conf, iou, dev)
    xyxy1, s1, c1 = results_to_tensors(r)
    big = filter_big_boxes(xyxy1, w, h, area_ratio, abs_area)
    big = pad_boxes(big, w, h, padding)

    tiles = generate_tiles(w, h, tile_size, tile_stride)
    tiles = skip_tiles_by_overlap(tiles, big, tile_skip_iou)

    xyxy_all = []
    scores_all = []
    cls_all = []

    xyxy_all.append(xyxy1)
    scores_all.append(s1)
    cls_all.append(c1)

    trans = None
    if use_transformer:
        layer_set = {s.strip() for s in layers.split(',') if s.strip()}
        trans = build_transformer_model(weights_trans_base, layer_set, heads, dim_ff, dropout, min_size, residual_scale)
        if half and torch.cuda.is_available():
            trans.model.half()

    if trans is not None:
        if use_mask:
            m = make_mask(img, big)
            residual = cv2.bitwise_and(img, img, mask=m)
            sz = residual_imgsz if residual_imgsz else imgsz
            rr = predict_yolo(trans, residual, sz, conf, iou, dev)
            xyxy2, s2, c2 = results_to_tensors(rr)
            if xyxy2.shape[0] > 0:
                xyxy_all.append(xyxy2)
                scores_all.append(s2)
                cls_all.append(c2)
        else:
            if len(tiles) > 0:
                batch = []
                coords = []
                for t in tiles:
                    x1, y1, x2, y2 = t
                    crop = img[y1:y2, x1:x2]
                    batch.append(crop)
                    coords.append((x1, y1))
                    if len(batch) == tile_batch:
                        rr = predict_yolo(trans, batch, tile_size, conf, iou, dev)
                        for k, res_k in enumerate(rr):
                            xyxy2, s2, c2 = results_to_tensors([res_k])
                            if xyxy2.shape[0] == 0:
                                continue
                            dx, dy = coords[k]
                            xyxy2 = offset_boxes(xyxy2, dx, dy)
                            xyxy_all.append(xyxy2)
                            scores_all.append(s2)
                            cls_all.append(c2)
                        batch = []
                        coords = []
                if len(batch) > 0:
                    rr = predict_yolo(trans, batch, tile_size, conf, iou, dev)
                    for k, res_k in enumerate(rr):
                        xyxy2, s2, c2 = results_to_tensors([res_k])
                        if xyxy2.shape[0] == 0:
                            continue
                        dx, dy = coords[k]
                        xyxy2 = offset_boxes(xyxy2, dx, dy)
                        xyxy_all.append(xyxy2)
                        scores_all.append(s2)
                        cls_all.append(c2)

    if len(xyxy_all) == 0:
        return img, []

    xyxy = torch.cat(xyxy_all, dim=0)
    scores = torch.cat(scores_all, dim=0)
    cls = torch.cat(cls_all, dim=0)

    keep = batched_nms_per_class(xyxy, scores, cls, iou)
    xyxy = xyxy[keep]
    scores = scores[keep]
    cls = cls[keep]

    vis = draw_boxes(img, xyxy)
    return vis, xyxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--weights_yolo', type=str, default='yolov8n.pt')
    parser.add_argument('--use_transformer', action='store_true')
    parser.add_argument('--weights_trans_base', type=str, default='yolov8n.pt')
    parser.add_argument('--layers', type=str, default='model.10,model.13,model.18')
    parser.add_argument('--tile_size', type=int, default=640)
    parser.add_argument('--tile_stride', type=int, default=320)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--area_ratio', type=float, default=0.1)
    parser.add_argument('--abs_area', type=int, default=None)
    parser.add_argument('--padding', type=int, default=8)
    parser.add_argument('--tile_skip_iou', type=float, default=0.4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dim_ff', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--min_size', type=int, default=64)
    parser.add_argument('--residual_scale', type=float, default=1.0)
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--residual_imgsz', type=int, default=640)
    parser.add_argument('--tile_batch', type=int, default=4)
    args = parser.parse_args()

    vis, _ = run_inference_pipeline(
        source=args.img,
        weights_yolo=args.weights_yolo,
        use_transformer=args.use_transformer,
        weights_trans_base=args.weights_trans_base,
        layers=args.layers,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        area_ratio=args.area_ratio,
        abs_area=args.abs_area,
        padding=args.padding,
        tile_skip_iou=args.tile_skip_iou,
        device=args.device,
        half=args.half,
        heads=args.heads,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        min_size=args.min_size,
        residual_scale=args.residual_scale,
        use_mask=args.use_mask,
        residual_imgsz=args.residual_imgsz,
        tile_batch=args.tile_batch
    )
    
    out_dir = args.out if args.out else os.path.join(os.getcwd(), 'runs', 'two_stage')
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join(out_dir, f'{base_name}_two_stage.jpg')
    cv2.imwrite(out_path, vis)
    print(out_path)


if __name__ == '__main__':
    main()
