import streamlit as st
from streamlit.components.v1 import html
import cv2
import numpy as np
from PIL import Image
import io
import time
from selenium import webdriver
from transformers import pipeline
import torch
import torch.nn.functional as F
from ollama import Client
import base64
from io import BytesIO
import yaml
import psutil
import threading
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av

OPENCV_AVFOUNDATION_SKIP_AUTH=1

def main():
    st.title('SpatialSense')
    st.write('Github: https://github.com/kabir12345/SpatialSense')

    # Initialize the depth-estimation pipeline
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    class VideoTransformer:
        def __init__(self):
            self.pipe = pipe

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            pil_img = Image.fromarray(image)

            # Perform depth estimation
            depth_mask = apply_depth_estimation(self.pipe, pil_img)

            # Convert PIL Image to NumPy array for display in Streamlit
            depth_mask_np = np.array(depth_mask)
            return av.VideoFrame.from_ndarray(depth_mask_np, format="bgr24")

    # Streamlit-WebRTC component
    webrtc_ctx = webrtc_streamer(key="example",
                                 mode=WebRtcMode.SENDRECV,
                                 video_frame_callback=VideoTransformer().transform)




def apply_depth_estimation(pipe, pil_img):
    # Assume the rest of your depth estimation logic is defined here
    original_width, original_height = pil_img.size
    depth = pipe(pil_img)["depth"]
    depth_tensor = torch.from_numpy(np.array(depth)).unsqueeze(0).unsqueeze(0).float()
    depth_resized = F.interpolate(depth_tensor, size=(original_height, original_width), mode='bilinear', align_corners=False)[0, 0]

    depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
    depth_normalized_np = depth_normalized.byte().cpu().numpy()
    colored_depth = cv2.applyColorMap(depth_normalized_np, cv2.COLORMAP_INFERNO)
    colored_depth_rgb = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
    colored_depth_image = Image.fromarray(colored_depth_rgb)
    
    return colored_depth_image

def encode_image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")  # You can change to "PNG" if you prefer
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handle_user_query(query, image_path, text_placeholder):
    if query:
        client = Client(host='http://localhost:11434')
        response = client.chat(model='llava:7b-v1.5-q2_K', messages=[
            {
                'role': 'user',
                'content': query,
                'images': [image_path]  # Pass the path to the temporary file
            },
        ])
        # Assuming response returns correctly, extract the response content if necessary
        response_content = str(response['message']['content'])  # Adjust based on how the response content is structured
        text_placeholder.text(response_content)

def update_cpu_usage():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        st.session_state.cpu_usage = f"CPU Usage: {cpu_usage}%"
        time.sleep(5)  # Update every 5 seconds, adjust as needed


if __name__ == "__main__":
    main()