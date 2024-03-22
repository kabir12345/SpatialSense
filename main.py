from transformers import pipeline
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


image_path = '//Users/kabir/Downloads/SpatialSense/samples/images/img1.jpg'  
image = Image.open(image_path)
original_width, original_height = image.size

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
depth = pipe(image)["depth"]
depth_tensor = torch.from_numpy(np.array(depth)).unsqueeze(0).unsqueeze(0).float()
depth_resized = F.interpolate(depth_tensor, size=(original_height, original_width), mode='bilinear', align_corners=False)[0, 0]

depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
depth_normalized_np = depth_normalized.byte().cpu().numpy()  

colored_depth = cv2.applyColorMap(depth_normalized_np, cv2.COLORMAP_INFERNO)
colored_depth_rgb = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
colored_depth_image = Image.fromarray(colored_depth_rgb)
colored_depth_image_path = '/Users/kabir/Downloads/SpatialSense/samples/images/new3.png'  
colored_depth_image.save(colored_depth_image_path)

sam = sam_model_registry["vit_h"](checkpoint="/Users/kabir/Downloads/SpatialSense/model_checkpoints/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
colored_depth_np = np.array(colored_depth_image)
masks = mask_generator.generate(colored_depth_np)

image_np = np.array(colored_depth_image)


image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

for mask in masks:
    
    bbox = mask['bbox']  
    segmentation = mask['segmentation']  

    seg_color = np.zeros_like(image_np)
    seg_color[segmentation] = [0, 255, 0] 
    cv2.addWeighted(seg_color, 0.5, image_np, 0.5, 0, image_np)
    start_point = (bbox[0], bbox[1])
    end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    cv2.rectangle(image_np, start_point, end_point, (255, 0, 0), 2)  

image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
marked_image = Image.fromarray(image_np)
marked_image_path = '/Users/kabir/Downloads/SpatialSense/samples/images/new3_SAM.png'  
marked_image.save(marked_image_path)
