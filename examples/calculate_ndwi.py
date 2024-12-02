"""
Example script demonstrating NDWI calculation and water body analysis using Farq.

This example demonstrates:
- Loading and preprocessing Landsat bands
- Calculating and validating NDWI
- Analyzing water bodies
- Visualizing results with input validation
"""
import farq
import os
from typing import Dict, Tuple, Any

def load_bands(base_dir: str) -> Tuple[Any, Any, Dict]:
    """
    Load and validate green and NIR bands from Landsat data.
    
    Args:
        base_dir: Base directory containing Landsat bands
        
    Returns:
        Tuple containing (green band, NIR band, metadata)
    """
    # Load the bands using farq's read function
    green_path = os.path.join(base_dir, "LC08_L2SP_174038_20241123_20241127_02_T1_SR_B3.TIF")
    nir_path = os.path.join(base_dir, "LC08_L2SP_174038_20241123_20241127_02_T1_SR_B5.TIF")
    
    print("Loading green band...")
    green, meta = farq.read(green_path)
    
    print("Loading NIR band...")
    nir, _ = farq.read(nir_path)
    
    # Print band statistics
    print("\nBand Statistics:")
    print("Green Band:")
    print(f"  Min: {farq.min(green):.2f}")
    print(f"  Max: {farq.max(green):.2f}")
    print(f"  Mean: {farq.mean(green):.2f}")
    print(f"  Std: {farq.std(green):.2f}")
    
    print("\nNIR Band:")
    print(f"  Min: {farq.min(nir):.2f}")
    print(f"  Max: {farq.max(nir):.2f}")
    print(f"  Mean: {farq.mean(nir):.2f}")
    print(f"  Std: {farq.std(nir):.2f}")
    
    return green, nir, meta

def main():
    try:
        # Load the bands
        print("Loading Landsat bands...")
        base_dir = r"examples\2024 Water"
        green, nir, meta = load_bands(base_dir)
        
        # Calculate NDWI
        print("\nCalculating NDWI...")
        ndwi = farq.ndwi(green, nir)
        
        # Create water mask (NDWI > 0 indicates water)
        print("\nAnalyzing water distribution...")
        water_mask = ndwi > 0
        
        # Calculate water statistics
        water_pixels = farq.sum(water_mask)
        total_pixels = water_mask.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        # Create visualization
        print("\nCreating visualization...")
        farq.plot(
            ndwi,
            title="NDWI Analysis",
            cmap="RdYlBu",
            vmin=-1,
            vmax=1
        )
        farq.plt.show()
        print("\nAnalysis complete!")
        print(f"Potential water pixels: {water_pixels}")
        print(f"Water coverage: {water_percentage:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 