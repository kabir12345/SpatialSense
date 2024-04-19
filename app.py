import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
import av
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline
import torch
import torch.nn.functional as F

def main():
    st.title('SpatialSense')
    st.write('Github: https://github.com/kabir12345/SpatialSense')

    # Initialize the depth-estimation pipeline
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.pipe = pipe

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            pil_img = Image.fromarray(image)

            # Perform depth estimation
            depth_mask = apply_depth_estimation(self.pipe, pil_img)

            # Convert PIL Image to NumPy array for display in Streamlit
            depth_mask_np = np.array(depth_mask)

            # Concatenate original image and depth mask side by side
            combined_frame = np.concatenate((image, depth_mask_np), axis=1)

            return av.VideoFrame.from_ndarray(combined_frame, format="bgr24")

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

if __name__ == "__main__":
    main()
