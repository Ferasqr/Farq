"""
Example script demonstrating spectral index calculations using Farq.
Shows how to calculate and analyze various vegetation and water indices from Landsat imagery.

This example demonstrates:
- Loading and handling multiple Landsat bands
- Calculating various spectral indices
- Visualizing and analyzing index distributions
"""
import farq
import os
from typing import Dict, Any

def load_landsat_bands(base_dir: str) -> Dict[str, Any]:
    """
    Load Landsat 8 bands from the specified directory.
    
    Args:
        base_dir: Base directory containing Landsat bands
        
    Returns:
        Dict containing band data with keys: 'blue', 'green', 'red', 'nir'
    """
    # Define band files
    band_files = {
        'blue': "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B2.TIF",
        'green': "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B3.TIF",
        'red': "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B4.TIF",
        'nir': "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B5.TIF"
    }
    
    # Load each band
    data = {}
    
    for band_name, filename in band_files.items():
        filepath = os.path.join(base_dir, filename)
        print(f"Loading {band_name} band...")
        data[band_name], _ = farq.read(filepath)
        
        # Print band statistics
        print(f"\n{band_name.upper()} Band Statistics:")
        print(f"  Min: {farq.min(data[band_name]):.2f}")
        print(f"  Max: {farq.max(data[band_name]):.2f}")
        print(f"  Mean: {farq.mean(data[band_name]):.2f}")
        print(f"  Std: {farq.std(data[band_name]):.2f}")
    
    return data

def main():
    try:
        # Load Landsat bands
        print("Loading Landsat bands...")
        base_dir = r"examples\2024 Forest Example"
        bands = load_landsat_bands(base_dir)
        
        # Calculate indices
        print("\nCalculating spectral indices...")
        
        # NDVI (Normalized Difference Vegetation Index)
        ndvi = farq.ndvi(bands['red'], bands['nir'])
        
        # NDWI (Normalized Difference Water Index)
        ndwi = farq.ndwi(bands['green'], bands['nir'])
        
        # EVI (Enhanced Vegetation Index)
        evi = farq.evi(bands['red'], bands['nir'], bands['blue'])
        
        # Visualize results
        print("\nVisualizing indices...")
        
        # Plot NDVI
        farq.plot(
            ndvi,
            title="NDVI Analysis",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1
        )
        farq.plt.show()
        
        # Plot NDWI
        farq.plot(
            ndwi,
            title="NDWI Analysis",
            cmap="RdYlBu",
            vmin=-1,
            vmax=1
        )
        farq.plt.show()
        
        # Plot EVI
        farq.plot(
            evi,
            title="EVI Analysis",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1
        )
        farq.plt.show()
        
        # Calculate coverage statistics
        print("\nAnalyzing index distributions...")
        
        # Calculate vegetation coverage from NDVI
        veg_mask = ndvi > 0.2  # Common threshold for vegetation
        veg_pixels = farq.sum(veg_mask)
        veg_percentage = (veg_pixels / veg_mask.size) * 100
        
        # Calculate water coverage from NDWI
        water_mask = ndwi > 0
        water_pixels = farq.sum(water_mask)
        water_percentage = (water_pixels / water_mask.size) * 100
        
        # Calculate enhanced vegetation coverage from EVI
        evi_mask = evi > 0.2
        evi_pixels = farq.sum(evi_mask)
        evi_percentage = (evi_pixels / evi_mask.size) * 100
        
        print("\nIndex Statistics:")
        print(f"NDVI - Vegetation coverage: {veg_percentage:.2f}%")
        print(f"NDWI - Water coverage: {water_percentage:.2f}%")
        print(f"EVI - Enhanced vegetation coverage: {evi_percentage:.2f}%")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 