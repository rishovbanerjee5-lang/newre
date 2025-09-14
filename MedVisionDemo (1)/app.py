import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("MedVision - Healthcare CV Demo")
st.write("Upload a medical image (e.g., X-ray) for anomaly detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image with PIL
    image = Image.open(uploaded_file)

    # Convert PIL image to numpy array for OpenCV if needed
    img_array = np.array(image)

    # Optional: convert to grayscale (example preprocessing step)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Display original image
    st.image(image, caption='Uploaded Medical Image', use_column_width=True)

    # Display processed image (for demo)
    st.image(gray, caption='Processed (Grayscale) Image', use_column_width=True, channels="GRAY")

    # Dummy prediction (replace with actual model inference)
    st.write("Analyzing image...")
    st.success("Result: Possible abnormality detected in highlighted region.")
