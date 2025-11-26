from ultralytics import YOLO

def main():
    m = YOLO('yolov8n.pt').model
    for name, child in m.named_modules():
        if child.__class__.__name__ == 'C2f':
            print('C2f:', name)
            for attr in ['cv1','cv2','c','c2','m','n']:
                if hasattr(child, attr):
                    v = getattr(child, attr)
                    print(' -', attr, type(v))
            break

if __name__ == '__main__':
    main()

