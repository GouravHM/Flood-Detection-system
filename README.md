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
Flood-Detection-system/
│
├── data/
│   └── raw/
│       ├── pre_flood/     # Your Pre-flood TIF images go here
│       └── post_flood/    # Your Post-flood TIF images go here
│
├── static/
│   ├── style.css          # Styles
│   └── script.js          # Frontend Logic
│
├── templates/
│   └── index.html         # The Website
│
├── server.py              # THE BRAIN (Runs the whole app)
├── requirements.txt       # Dependencies
└── README.md              # Documentation
