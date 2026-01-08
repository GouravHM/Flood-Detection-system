import numpy as np

def calculate_flood_statistics(flood_mask, pixel_resolution=10):
    """
    Calculates percentage of area flooded.
    pixel_resolution: meters per pixel (Sentinel-2 is approx 10m)
    """
    total_pixels = flood_mask.size
    flood_pixels = np.count_nonzero(flood_mask)
    
    flood_percentage = (flood_pixels / total_pixels) * 100
    
    # Area calculation (approximate)
    area_sq_km = (flood_pixels * pixel_resolution * pixel_resolution) / 1_000_000
    
    return {
        "Flood Percentage": round(flood_percentage, 2),
        "Flooded Area (sq km)": round(area_sq_km, 3)
    }