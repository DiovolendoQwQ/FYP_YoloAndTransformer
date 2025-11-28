import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Add modules to path just in case
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_menu():
    print("\n===========================================")
    print("      小目标专属训练选项 (Small Object Training)")
    print("===========================================")
    print("说明: ")
    print("1. 使用公开数据集 (推荐) - 如 VisDrone, TinyPerson, DOTA")
    print("   适合情况: 若目标泛化性为主")
    print("2. 从当前数据集中筛选小目标")
    print("   说明: 筛出面积 < 32x32 的目标子集重新训练")
    print("   适合情况: 若你用的就是 COCO 或自己采集的数据")
    print("===========================================")

def handle_option_1():
    print("\n[选项 1] 使用公开数据集")
    print("1. VisDrone (推荐) - 无人机视角小目标")
    print("2. DOTA - 航空视角目标检测")
    print("3. TinyPerson - 远距离微小人体检测 (需手动配置)")
    
    ds_choice = input("请选择数据集 (1: VisDrone, 2: DOTA): ").strip()
    
    data_yaml = 'VisDrone.yaml'
    name = 'visdrone_small_obj'
    
    if ds_choice == '2':
        data_yaml = 'DOTAv1.yaml' # Ultralytics standard name
        name = 'dota_small_obj'
    
    choice = input(f"是否立即开始下载并训练 {data_yaml}? (y/n): ").lower()
    if choice == 'y':
        print(f"\n正在启动 {data_yaml} 训练...")
        model_name = 'yolov8n.pt' 
        print(f"使用模型: {model_name}")
        print("注意: 第一次运行会自动下载数据集，请确保网络通畅。")
        
        try:
            model = YOLO(model_name)
            # Default epochs 50, but can be overridden in main if we passed args, 
            # here we are in interactive mode, let's ask or default.
            epochs = input("请输入训练轮数 (默认 50): ").strip()
            if not epochs.isdigit():
                epochs = 50
            else:
                epochs = int(epochs)
                
            imgsz = input("请输入输入图像尺寸 (默认 640, 小目标建议 1024+): ").strip()
            if not imgsz.isdigit():
                imgsz = 640
            else:
                imgsz = int(imgsz)
                
            model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=16, project='runs/train', name=name)
        except Exception as e:
            print(f"训练启动失败: {e}")
    else:
        print(f"已取消。你可以手动运行: yolo train data={data_yaml}")

def filter_small_objects(source_yaml, output_dir_name='small_objects_dataset', force_overwrite=False):
    """
    Filter dataset for small objects (area < 32x32)
    """
    print(f"\n正在处理数据集: {source_yaml}")
    
    # Check if source yaml exists
    if not os.path.exists(source_yaml):
        print(f"错误: 找不到文件 {source_yaml}")
        return None

    # Load dataset config
    with open(source_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Resolve paths
    # data['path'] is usually the root dir for the dataset
    root_path = Path(data.get('path', ''))
    
    # If path is relative, we need to resolve it against something. 
    # Usually it's relative to the yaml file location if the yaml is inside datasets/
    # Or relative to YOLO execution dir.
    # Let's try to make it absolute if it exists
    if not root_path.is_absolute():
        # Try finding it
        p1 = Path(source_yaml).parent / root_path
        if p1.exists():
            root_path = p1.resolve()
        else:
             p2 = Path(os.getcwd()) / root_path
             if p2.exists():
                 root_path = p2.resolve()

    # Helper to resolve image path
    def resolve_image_path(split_path):
        # split_path could be a list or string
        if isinstance(split_path, list):
            split_path = split_path[0] # handle single path for now
        
        p = Path(split_path)
        if p.is_absolute():
            return p
            
        # Try combining with root_path
        p1 = root_path / p
        if p1.exists():
            return p1
        # Try combining with yaml directory
        p2 = Path(source_yaml).parent / p
        if p2.exists():
            return p2
        # Try as relative to CWD
        p3 = Path(os.getcwd()) / p
        if p3.exists():
            return p3
        # Try standard datasets folder
        p4 = Path(os.getcwd()) / 'datasets' / p
        if p4.exists():
            return p4
        return p

    splits = ['train', 'val']
    if 'test' in data:
        splits.append('test')

    # Prepare new dataset structure
    output_dir = Path(os.getcwd()) / 'datasets' / output_dir_name
    if output_dir.exists():
        print(f"警告: 输出目录 {output_dir} 已存在。")
        if force_overwrite:
            print("Force overwrite enabled. Removing existing directory.")
            shutil.rmtree(output_dir)
        else:
            overwrite = input("是否覆盖? (y/n): ").lower()
            if overwrite == 'y':
                shutil.rmtree(output_dir)
            else:
                print("请提供一个新的输出目录名称。")
                return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    new_data = data.copy()
    new_data['path'] = str(output_dir.absolute())
    
    # Stats
    total_images = 0
    kept_images = 0
    total_objects = 0
    small_objects = 0

    for split in splits:
        if split not in data:
            continue
            
        img_dir_path = resolve_image_path(data[split])
        if not img_dir_path.exists():
            print(f"警告: 找不到 {split} 集目录: {img_dir_path}")
            continue

        # Create new directories
        new_img_dir = output_dir / 'images' / split
        new_label_dir = output_dir / 'labels' / split
        new_img_dir.mkdir(parents=True, exist_ok=True)
        new_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Update yaml config
        new_data[split] = f"images/{split}"

        # List images
        # support common extensions
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        images = [p for p in img_dir_path.rglob('*') if p.suffix.lower() in valid_exts]
        
        print(f"正在处理 {split} 集 ({len(images)} 张图片)...")
        
        for img_path in tqdm(images):
            total_images += 1
            
            # Find corresponding label file
            # Assuming standard YOLO structure: .../images/train/name.jpg -> .../labels/train/name.txt
            # Or side-by-side if not in standard structure
            
            # Try standard structure replacement
            # Replace 'images' with 'labels' in path parts if it exists
            parts = list(img_path.parts)
            try:
                idx = len(parts) - 1 - parts[::-1].index('images')
                parts[idx] = 'labels'
                label_path = Path(*parts).with_suffix('.txt')
            except ValueError:
                # 'images' not in path, try simply replacing suffix in same folder (not standard but possible)
                label_path = img_path.with_suffix('.txt')
            
            if not label_path.exists():
                # Try finding it in a parallel 'labels' folder relative to image parent
                label_path = img_path.parent.parent / 'labels' / img_path.parent.name / img_path.with_suffix('.txt').name
                if not label_path.exists():
                     # Try simpler parallel
                    label_path = img_path.parent.parent / 'labels' / img_path.with_suffix('.txt').name
            
            if not label_path.exists():
                # If no label, skip or keep as empty?
                # For "filtering small objects", if no label, it has no small objects (it has NO objects).
                # Skipping empty images
                continue
                
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Read labels
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            has_small_object = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls_id = parts[0]
                # x_center, y_center, width, height (normalized)
                nw, nh = float(parts[3]), float(parts[4])
                
                # Convert to pixels
                pw = nw * w
                ph = nh * h
                
                total_objects += 1
                
                # Check if small (area < 32x32 = 1024)
                if (pw * ph) < (32 * 32):
                    new_lines.append(line)
                    small_objects += 1
                    has_small_object = True
            
            # If image has small objects, save it and the filtered labels
            if has_small_object:
                kept_images += 1
                # Copy image
                shutil.copy2(img_path, new_img_dir / img_path.name)
                # Write new labels
                with open(new_label_dir / label_path.name, 'w') as f:
                    f.writelines(new_lines)
    
    # Save new yaml
    new_yaml_path = output_dir / f"{Path(source_yaml).stem}_small.yaml"
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_data, f, sort_keys=False)
        
    print("\n筛选完成!")
    print(f"原图片总数: {total_images}")
    print(f"保留图片数 (含小目标): {kept_images}")
    print(f"原目标总数: {total_objects}")
    print(f"小目标总数 (<32x32): {small_objects}")
    print(f"新的数据集配置已保存至: {new_yaml_path}")
    
    return new_yaml_path

def handle_option_2():
    print("\n[选项 2] 从当前数据集中筛选小目标")
    
    default_yaml = 'coco128.yaml'
    source_yaml = input(f"请输入数据集 YAML 配置文件路径 (默认: {default_yaml}): ").strip()
    if not source_yaml:
        source_yaml = default_yaml
        
    # Check if we can find the yaml
    if not os.path.exists(source_yaml):
        # try to find in current dir or datasets dir
        if os.path.exists(os.path.join('datasets', source_yaml)):
            source_yaml = os.path.join('datasets', source_yaml)
        elif not os.path.exists(source_yaml): 
            # try to search? No, just fail
            pass
            
    if not os.path.exists(source_yaml):
        print(f"找不到文件: {source_yaml}")
        return

    new_yaml = filter_small_objects(source_yaml)
    
    if new_yaml:
        choice = input("\n是否立即使用新数据集开始训练? (y/n): ").lower()
        if choice == 'y':
            print("\n正在启动训练...")
            model_name = 'yolov8n.pt'
            try:
                model = YOLO(model_name)
                
                epochs = input("请输入训练轮数 (默认 50): ").strip()
                if not epochs.isdigit():
                    epochs = 50
                else:
                    epochs = int(epochs)

                imgsz = input("请输入输入图像尺寸 (默认 640, 小目标建议 1024+): ").strip()
                if not imgsz.isdigit():
                    imgsz = 640
                else:
                    imgsz = int(imgsz)

                model.train(data=str(new_yaml), epochs=epochs, imgsz=imgsz, batch=16, project='runs/train', name='custom_small_obj')
            except Exception as e:
                print(f"训练启动失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='Small Object Training Options')
    parser.add_argument('--mode', type=str, choices=['1', '2'], help='Mode: 1 for Public Datasets, 2 for Filtering Custom Dataset')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Data YAML for mode 2')
    parser.add_argument('--auto_train', action='store_true', help='Auto start training')
    parser.add_argument('--force', action='store_true', help='Force overwrite output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    args = parser.parse_args()

    if args.mode:
        choice = args.mode
    else:
        print_menu()
        choice = input("请输入选项 (1 或 2): ").strip()
    
    if choice == '1':
        if args.mode and args.auto_train:
             # Automation for mode 1
             print("\n[选项 1] 使用公开数据集")
             print("自动启动 VisDrone 训练...")
             try:
                model = YOLO('yolov8n.pt')
                model.train(data='VisDrone.yaml', epochs=args.epochs, imgsz=args.imgsz, batch=16, project='runs/train', name='visdrone_small_obj')
             except Exception as e:
                print(f"训练启动失败: {e}")
        else:
            handle_option_1()
    elif choice == '2':
        if args.mode:
             # Automation for mode 2
             source_yaml = args.data
             if not os.path.exists(source_yaml) and os.path.exists(os.path.join('datasets', source_yaml)):
                 source_yaml = os.path.join('datasets', source_yaml)
                 
             new_yaml = filter_small_objects(source_yaml, force_overwrite=args.force)
             
             if new_yaml and args.auto_train:
                print("\n自动启动训练...")
                try:
                    model = YOLO('yolov8n.pt')
                    model.train(data=str(new_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=16, project='runs/train', name='custom_small_obj')
                except Exception as e:
                    print(f"训练启动失败: {e}")
        else:
            handle_option_2()
    else:
        print("无效选项，退出。")

if __name__ == "__main__":
    main()
