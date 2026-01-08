# 🌊 AI-Based Flood Change Detection System (India Region)

## 📌 Project Overview
This project is a Deep Learning and Remote Sensing framework designed to automatically detect flood inundation areas from satellite imagery. It focuses on disaster response in the **Indian Subcontinent** (specifically Assam and Kerala floods) using Sentinel-1 SAR data.

The system processes Pre-Flood and Post-Flood images to generate a **Damage Assessment Map** and calculates the exact area of land submerged in water.

## 🚀 Features
* **Deep Learning / Computer Vision:** Uses U-Net architecture and robust Change Detection algorithms.
* **India-Specific Analysis:** optimized for detecting water bodies in Indian topographies (Assam/Kerala).
* **Geospatial Visualization:** Interactive Web Map (Leaflet/Folium) showing flood extent on real-world coordinates.
* **Impact Statistics:** Automatically calculates the percentage of inundation and area in square kilometers.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Web Framework:** Streamlit
* **GIS & Mapping:** Folium, Leaflet, Google Earth Engine (GEE)
* **Image Processing:** OpenCV, NumPy, Pillow, Tifffile
* **Deep Learning:** PyTorch (for U-Net implementation)

## 📂 Project Structure
```text
flood_detection/
├── app/
│   ├── app.py            # Main Web Application
│   └── utils.py          # Image processing & GIS helper functions
├── data/
│   └── raw/              # Satellite Images (Sentinel-1 SAR)
├── models/
│   └── unet_model.py     # U-Net Neural Network Architecture
├── src/
│   ├── get_gee_data.py   # Script to download Sentinel-1 data from Google Earth Engine
│   └── pipeline.py       # Training pipeline
├── outputs/              # Generated Maps (HTML) and CSV Reports
├── requirements.txt      # List of dependencies
└── README.md             # Project Documentation