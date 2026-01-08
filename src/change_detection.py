import torch
import cv2
import numpy as np

def detect_changes(model, pre_img_tensor, post_img_tensor, threshold=0.5):
    """
    Predicts masks for both images and subtracts them to find new water.
    """
    model.eval()
    with torch.no_grad():
        # Predict water mask for Pre-flood
        pre_pred = model(pre_img_tensor)
        pre_mask = (pre_pred > threshold).float().cpu().numpy()[0][0]

        # Predict water mask for Post-flood
        post_pred = model(post_img_tensor)
        post_mask = (post_pred > threshold).float().cpu().numpy()[0][0]

    # LOGIC: Flood = (Water in Post) AND (No Water in Pre)
    # This captures the change.
    flood_change_mask = np.maximum(post_mask - pre_mask, 0)
    
    return pre_mask, post_mask, flood_change_mask

def create_overlay(original_img, mask, color=(0, 0, 255)):
    """
    Overlays the flood mask (red) onto the original image.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    colored_mask = np.zeros_like(original_img)
    colored_mask[:] = color # Red fill
    
    # Apply mask
    masked_colored = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_uint8)
    
    # Blend
    overlay = cv2.addWeighted(original_img, 0.7, masked_colored, 0.3, 0)
    return overlay