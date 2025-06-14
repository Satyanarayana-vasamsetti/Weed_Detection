import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Title
st.title("🌿 Weed Detection")
# st.markdown("Upload an image to detect weeds using a trained YOLOv8 model (`weed.pt`).")

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO("weed.pt")

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # 1 means color

    # Run YOLOv8 inference
    results = model(img)

    # Annotate the image
    annotated_frame = results[0].plot()

    # Show result
    st.image(annotated_frame, channels="BGR", caption="🟩 Detected Weeds", use_container_width=True)


    # Save to temp file for download
    temp_dir = tempfile.mkdtemp()
    result_path = os.path.join(temp_dir, "annotated_image.jpg")
    cv2.imwrite(result_path, annotated_frame)

    # Download button
    with open(result_path, "rb") as file:
        btn = st.download_button(
            label="📥 Download Annotated Image",
            data=file,
            file_name="weed_detection_result.jpg",
            mime="image/jpeg"
        )
