import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
import os
from PIL import Image, ImageEnhance

app = Flask(__name__)

# --- CONFIGURATION ---
DATA_DIR = os.path.join("data", "raw")
Image.MAX_IMAGE_PIXELS = None 

# --- REGION DATA ---
REGION_FILES = {
    "Assam (Silchar)":      {"pre": "Assam_Pre.tif",   "post": "Assam_Post.tif",   "coords": [24.83, 92.78]},
    "Kerala (Kuttanad)":    {"pre": "Kerala_Pre.tif",  "post": "Kerala_Post.tif",  "coords": [9.42, 76.46]},
    "Maharashtra (Mumbai)": {"pre": "Mumbai_Pre.tif",  "post": "Mumbai_Post.tif",  "coords": [19.07, 72.87]},
    "Delhi (Yamuna)":       {"pre": "Delhi_Pre.tif",   "post": "Delhi_Post.tif",   "coords": [28.66, 77.22]},
    "Tamil Nadu (Chennai)": {"pre": "Chennai_Pre.tif", "post": "Chennai_Post.tif", "coords": [13.08, 80.27]},
    "Bihar (Kosi River)":   {"pre": "Bihar_Pre.tif",   "post": "Bihar_Post.tif",   "coords": [25.54, 87.13]}
}

# --- HELPER FUNCTIONS ---
def image_to_base64(img_np):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def mask_to_base64(mask_np):
    h, w = mask_np.shape
    bgra_mask = np.zeros((h, w, 4), dtype=np.uint8)
    # Red Color for Flood
    bgra_mask[mask_np == 255, 0] = 0    # Blue
    bgra_mask[mask_np == 255, 1] = 0    # Green
    bgra_mask[mask_np == 255, 2] = 255  # Red
    bgra_mask[mask_np == 255, 3] = 180  # Alpha
    _, buffer = cv2.imencode('.png', bgra_mask)
    return base64.b64encode(buffer).decode('utf-8')

def load_image(source, is_file_upload=False, subfolder=None):
    try:
        if is_file_upload:
            img = Image.open(source.stream).convert("RGB")
        else:
            path = os.path.join(DATA_DIR, subfolder, source)
            if not os.path.exists(path): return None
            img = Image.open(path).convert("RGB")
            
        img_np = np.array(img)
        # Stronger Auto-Contrast to make water visible against land
        if img_np.mean() < 60:
            img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
            
        return cv2.resize(img_np, (512, 512))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def generate_white_theme_plot(flood_area, total_area, region_name):
    safe_area = total_area - flood_area
    flood_color = '#dc3545' if flood_area > (total_area * 0.05) else '#fd7e14'
    
    plt.figure(figsize=(6, 4), facecolor='white')
    ax = plt.axes()
    ax.set_facecolor('#f8f9fa')

    labels = ['Flooded', 'Safe']
    values = [flood_area, safe_area]
    colors = [flood_color, '#28a745']

    bars = plt.bar(labels, values, color=colors, width=0.5)
    
    plt.title(f'Flood Impact: {region_name}', color='#333', fontsize=12, fontweight='bold')
    plt.ylabel('Area (sq km)', color='#555')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#ccc')
    ax.spines['left'].set_color('#ccc')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{round(height, 1)}', ha='center', va='bottom', color='black')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def process_images(pre_img, post_img, region_name, coords):
    if pre_img is None or post_img is None: return jsonify({"error": "Image Load Failed"}), 500

    # 1. Convert to Grayscale
    gray_pre = cv2.cvtColor(pre_img, cv2.COLOR_RGB2GRAY)
    gray_post = cv2.cvtColor(post_img, cv2.COLOR_RGB2GRAY)

    # 2. VALID MASK: Strict Border Removal
    # Ignore pixels that are very dark ( < 10 ) in BOTH images.
    # This cleans up the black box around the rotated satellite images.
    mask_pre = cv2.threshold(gray_pre, 10, 255, cv2.THRESH_BINARY)[1]
    mask_post = cv2.threshold(gray_post, 10, 255, cv2.THRESH_BINARY)[1]
    valid_mask = cv2.bitwise_and(mask_pre, mask_post)

    # 3. NO BRIGHTNESS MATCHING (Removed)
    # Correcting brightness was hiding real floods in dark images.
    # We compare raw values now.

    # 4. Blur (Minimal)
    blur_pre = cv2.GaussianBlur(gray_pre, (3, 3), 0)
    blur_post = cv2.GaussianBlur(gray_post, (3, 3), 0)
    
    # 5. Difference Calculation
    diff = cv2.absdiff(blur_pre, blur_post)
    
    # 6. THRESHOLD: 30 (High Sensitivity)
    # Any change greater than 30 intensity (out of 255) is marked as potential flood.
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 7. Apply Valid Mask (Cut borders)
    mask = cv2.bitwise_and(mask, mask, mask=valid_mask)

    # 8. Noise Removal
    # 2x2 Kernel removes single-pixel specs but keeps small rivers/floods
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Stats
    flood_pixels = np.count_nonzero(mask == 255)
    total_valid_pixels = np.count_nonzero(valid_mask == 255)
    
    if total_valid_pixels == 0: total_valid_pixels = 1
        
    percentage = (flood_pixels / total_valid_pixels) * 100
    total_area_km = 500
    flood_area_km = (percentage / 100) * total_area_km

    # Generate Visuals
    heatmap = np.zeros_like(pre_img)
    heatmap[:, :, 0] = mask
    final_output = cv2.addWeighted(post_img, 0.7, heatmap, 0.4, 0)
    
    plot_b64 = generate_white_theme_plot(flood_area_km, total_area_km, region_name)

    return jsonify({
        "result_image": f"data:image/png;base64,{image_to_base64(final_output)}",
        "mask_image": f"data:image/png;base64,{mask_to_base64(mask)}",
        "plot_image": f"data:image/png;base64,{plot_b64}",
        "stats": {"percentage": round(percentage, 2), "area": round(flood_area_km, 2)},
        "coords": coords
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_region', methods=['POST'])
def analyze_region():
    try:
        region = request.json.get('region')
        if region not in REGION_FILES: return jsonify({"error": "Region not found"}), 400
        
        info = REGION_FILES[region]
        pre_img = load_image(info['pre'], False, "pre_flood")
        post_img = load_image(info['post'], False, "post_flood")
        
        return process_images(pre_img, post_img, region, info['coords'])
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/analyze_upload', methods=['POST'])
def analyze_upload():
    try:
        pre_file = request.files.get('pre_image')
        post_file = request.files.get('post_image')
        
        pre_img = load_image(pre_file, True)
        post_img = load_image(post_file, True)
        
        return process_images(pre_img, post_img, "Uploaded Region", [20.0, 78.0])
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)