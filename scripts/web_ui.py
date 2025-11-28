import streamlit as st
import os
import sys
import numpy as np
import cv2
from PIL import Image
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import inference pipeline
from scripts.two_stage_infer import run_inference_pipeline

st.set_page_config(page_title="YOLO Two-Stage Inference", layout="wide")

st.title("YOLO Two-Stage Inference UI")
st.markdown("上传图片并调整参数以运行两阶段目标检测推理。")

# Sidebar for parameters
st.sidebar.header("模型设置 (Model Settings)")
weights_yolo = st.sidebar.text_input("YOLO 权重路径 (Base Model)", "yolov8n.pt")
use_transformer = st.sidebar.checkbox("使用 Transformer (Two-Stage)", value=False)
weights_trans_base = st.sidebar.text_input("Transformer 基础权重", "yolov8n.pt", disabled=not use_transformer)
layers = st.sidebar.text_input("Transformer 替换层 (e.g. model.10,model.13)", "model.10,model.13,model.18", disabled=not use_transformer)

st.sidebar.header("推理参数 (Inference Params)")
conf = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25)
iou = st.sidebar.slider("IOU 阈值", 0.0, 1.0, 0.6)
imgsz = st.sidebar.number_input("推理图像尺寸 (Image Size)", value=640, step=32)

st.sidebar.header("切片设置 (Tiling Settings)")
tile_size = st.sidebar.number_input("切片大小 (Tile Size)", value=640, step=32)
tile_stride = st.sidebar.number_input("切片步长 (Tile Stride)", value=320, step=32)
tile_skip_iou = st.sidebar.slider("切片跳过 IOU (Tile Skip IOU)", 0.0, 1.0, 0.4)
area_ratio = st.sidebar.slider("大目标面积比例 (Big Box Ratio)", 0.0, 1.0, 0.1)

# Main area
uploaded_file = st.file_uploader("上传图片 (Upload Image)", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Convert to cv2 image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图片 (Original)")
        st.image(img, channels="BGR", use_column_width=True)

    if st.button("开始推理 (Run Inference)"):
        with st.spinner("正在推理中... (Running Inference...)"):
            try:
                # Run inference
                vis, detections = run_inference_pipeline(
                    source=img,
                    weights_yolo=weights_yolo,
                    use_transformer=use_transformer,
                    weights_trans_base=weights_trans_base,
                    layers=layers,
                    tile_size=int(tile_size),
                    tile_stride=int(tile_stride),
                    imgsz=int(imgsz),
                    conf=conf,
                    iou=iou,
                    area_ratio=area_ratio,
                    tile_skip_iou=tile_skip_iou,
                    device=None, # Auto
                )
                
                # Display result
                with col2:
                    st.subheader("检测结果 (Result)")
                    st.image(vis, channels="BGR", use_column_width=True)
                
                st.success(f"检测完成! 发现 {len(detections)} 个目标。")
                
                # Show detection list (optional)
                if len(detections) > 0 and st.checkbox("显示详细数据"):
                    st.write(detections)
                    
            except Exception as e:
                st.error(f"推理出错: {e}")
                import traceback
                st.text(traceback.format_exc())

else:
    st.info("请在左侧上传图片以开始。")

st.markdown("---")
st.caption("Powered by Streamlit & YOLOv8")
