import cv2
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image_path, img_size=(256, 256)):
    """
    Reads an image, resizes it, and converts it to a PyTorch tensor.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    
    # Normalize to 0-1 and convert to Tensor (C, H, W)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform(img).unsqueeze(0), img  # Return tensor and original numpy img for display