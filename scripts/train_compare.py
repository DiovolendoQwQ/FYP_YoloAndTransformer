import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
from ultralytics import YOLO

# Check CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available. Training will run on CPU.")

# Ensure project root and scripts directory are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import build_transformer_model from two_stage_infer
try:
    from two_stage_infer import build_transformer_model
except ImportError:
    try:
        from scripts.two_stage_infer import build_transformer_model
    except ImportError as e:
        print(f"Error importing two_stage_infer: {e}")
        sys.exit(1)

def train_yolo(data_config, epochs=200, project='runs/train', name='yolo_baseline', batch=16, device=None):
    print(f"\n{'='*20} Starting YOLO Baseline Training {'='*20}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device if device else 'Auto'}")
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data=data_config, 
        epochs=epochs, 
        project=project, 
        name=name,
        batch=batch,
        device=device,
        exist_ok=True # Overwrite if exists
    )
    print(f"YOLO Baseline training completed. Results saved to {project}/{name}")
    return results

def train_transformer(data_config, epochs=200, project='runs/train', name='yolo_transformer', batch=16, device=None):
    print(f"\n{'='*20} Starting YOLO+Transformer Training {'='*20}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device if device else 'Auto'}")
    
    # Transformer parameters (matching defaults in two_stage_infer.py)
    layers = 'model.10,model.13,model.18'
    heads = 8
    dim_ff = None
    dropout = 0.1
    min_size = 64
    residual_scale = 1.0
    
    # Build model
    print("Building Transformer model...")
    model = build_transformer_model(
        base_weights='yolov8n.pt',
        layers=layers,
        heads=heads,
        dim_ff=dim_ff,
        dropout=dropout,
        min_size=min_size,
        residual_scale=residual_scale
    )
    
    # Train
    # Note: The model returned by build_transformer_model is a YOLO object
    results = model.train(
        data=data_config, 
        epochs=epochs, 
        project=project, 
        name=name,
        batch=batch,
        device=device,
        exist_ok=True
    )
    print(f"YOLO+Transformer training completed. Results saved to {project}/{name}")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO vs YOLO+Transformer models')
    parser.add_argument('--mode', type=str, default='all', choices=['yolo', 'transformer', 'all'], help='Which model to train')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Dataset config file')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device (e.g., 0 or 0,1,2,3 or cpu)')
    
    args = parser.parse_args()
    
    # Handle device
    device = args.device if args.device else None
    
    # Check if data file exists
    if not os.path.exists(args.data) and not os.path.exists(os.path.join(project_root, args.data)):
        # Try to find it in project root
        if os.path.exists(os.path.join(project_root, args.data)):
            args.data = os.path.join(project_root, args.data)
        else:
            print(f"Warning: Data config {args.data} not found.")
    
    if args.mode in ['yolo', 'all']:
        train_yolo(args.data, args.epochs, batch=args.batch, device=device)
        
    if args.mode in ['transformer', 'all']:
        train_transformer(args.data, args.epochs, batch=args.batch, device=device)
