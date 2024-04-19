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
from ZoeDepth.zoedepth.models.builder import build_model
from ZoeDepth.zoedepth.utils.config import get_config

def main():
    st.title('SpatialSense')
    st.write('Github: https://github.com/kabir12345/SpatialSense')
    cpu_usage = psutil.cpu_percent(interval=1)
    st.metric(label="CPU Usage", value=f"{cpu_usage} %")
    st.write("Original Video")
    channel_name = 'kabirjaiswal900'
    temp_image_path = "temp_image.jpg"
    temp_image_depth_path = "temp_image_depth.jpg"
    with open("/Users/kabir/Downloads/SpatialSense/utils/prompt.yml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    if channel_name:
        embed_twitch_stream(channel_name)
    else:
        st.error("Please enter a valid Twitch channel name.")

    st.write("Depth Anything Model Running Locally in Near Real-Time")
    placeholder = st.empty()
    text_placeholder = st.empty()
  
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)

    driver.get('http://localhost:8502/')

    frame_rate = 1
    wait_time = 1 / frame_rate

    start_time = time.time()
    messages = []

    try:
        while True:
            screenshot = driver.get_screenshot_as_png()
            nparr = np.frombuffer(screenshot, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cpu_usage = psutil.cpu_percent(interval=1)
            pil_img = Image.fromarray(img_np)
            pil_img.save(temp_image_path, "JPEG")
            depth_mask = apply_depth_estimation(pipe, pil_img)

            depth_mask_cv = np.array(depth_mask)

            height, width, _ = depth_mask_cv.shape
            for i in range(1, 3):
                depth_mask_cv = cv2.line(depth_mask_cv, (width // 3 * i, 0), (width // 3 * i, height), (255, 255, 255), 2)
                depth_mask_cv = cv2.line(depth_mask_cv, (0, height // 3 * i), (width, height // 3 * i), (255, 255, 255), 2)

            depth_mask_with_grid = Image.fromarray(depth_mask_cv)
            depth_mask_with_grid.save(temp_image_depth_path, "JPEG")
            placeholder.image(depth_mask_with_grid, channels="RGB")

            if time.time() - start_time > 10:  # Adjusted to every 15 seconds for testing
                # No need to encode to base64 since Ollama client expects a file path
                client = Client(host='http://localhost:11434')
                response = client.chat(model='llava:7b-v1.5-q2_K', messages=[
                {
                    'role': 'user',
                    'content': cfg['generator_prompt'],
                    'images': [temp_image_path,temp_image_depth_path]  # Correctly passing the file path
                },
                ])
                # Assuming response returns correctly, extract the response content if necessary
                # Append a string message to the list
                messages.append(str(response['message']['content']))  # Adjust based on how the response content is structured
                # Convert the list of messages into a single string
                messages_str = "\n".join(messages)
                text_placeholder.text_area("Navigational Output", messages_str, height=300)
                start_time = time.time()

            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("Stopping stream...")

    finally:
        driver.quit()

def embed_twitch_stream(channel_name):
    embed_url = f"https://player.twitch.tv/?channel={channel_name}&parent=localhost"
    html_code = f'<iframe src="{embed_url}" height="394" width="700" frameborder="0" allowfullscreen="true" scrolling="no" allow="autoplay; fullscreen"></iframe>'
    html(html_code, height=400)

def apply_depth_estimation(pipe, pil_img):
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
    pil_img.save(buffered, format="JPEG")  
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_zoedepth(image_path):
        config = get_config("/Users/kabir/Downloads/SpatialSense/ZoeDepth/configs/zoe.yaml")
        model = build_model(config)
        model.load_state_dict(torch.load("/Users/kabir/Downloads/SpatialSense/ZoeDepth/checkpoints/zoe.pth"))
        model.eval()
        image = Image.open(image_path)
        depth_map = Zoe(model, image)
        zoedepth = np.mean(depth_map)
        return zoedepth

def handle_user_query(query, image_path, text_placeholder):
    if query:
        client = Client(host='http://localhost:11434')
        response = client.chat(model='llava:7b-v1.5-q2_K', messages=[
            {
                'role': 'user',
                'content': query,
                'images': [image_path] 
            },
        ])
        
        response_content = str(response['message']['content'])  

def update_cpu_usage():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        st.session_state.cpu_usage = f"CPU Usage: {cpu_usage}%"
        time.sleep(5)  # Update every 5 seconds, adjust as needed


if __name__ == "__main__":
    main()

