import ee
import geemap
import os

# --- CONFIGURATION ---
# ✅ FIX: Put your Project ID inside quotes
MY_PROJECT_ID = 'flood-detection-483508' 

# Define Output Folders
pre_dir = "data/raw/pre_flood"
post_dir = "data/raw/post_flood"
os.makedirs(pre_dir, exist_ok=True)
os.makedirs(post_dir, exist_ok=True)

# --- AUTHENTICATION ---
try:
    # ✅ FIX: Use 'project=' parameter with the variable
    ee.Initialize(project=MY_PROJECT_ID)
    print("✅ Successfully connected to Google Earth Engine!")
except Exception as e:
    print(f"⚠️ Authentication needed: {e}")
    print("Opening browser for login...")
    ee.Authenticate()
    # ✅ FIX: Use 'project=' here too
    ee.Initialize(project=MY_PROJECT_ID)

# --- DEFINE REGION: ASSAM, INDIA (Silchar) ---
roi = ee.Geometry.Rectangle([92.7, 24.7, 92.9, 24.9])

# --- DATES (Assam Floods June 2022) ---
pre_start = '2022-04-01'
pre_end = '2022-04-15'
post_start = '2022-06-20'
post_end = '2022-06-30'

# --- HELPER FUNCTION ---
def get_s1_image(roi, start, end):
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate(start, end)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .select(['VV']))
    return collection.mosaic().clip(roi)

print(f"⏳ Fetching Sentinel-1 Data for Assam, India...")

# 1. Fetch Images
pre_img = get_s1_image(roi, pre_start, pre_end)
post_img = get_s1_image(roi, post_start, post_end)

# 2. Download
print("⬇️ Downloading Pre-Flood Image...")
geemap.ee_export_image(
    pre_img, 
    filename=os.path.join(pre_dir, "assam_pre.tif"), 
    scale=20, 
    region=roi, 
    file_per_band=False
)

print("⬇️ Downloading Post-Flood Image...")
geemap.ee_export_image(
    post_img, 
    filename=os.path.join(post_dir, "assam_post.tif"), 
    scale=20, 
    region=roi, 
    file_per_band=False
)

print("✅ Download Complete! Images saved in 'data/raw/'.")