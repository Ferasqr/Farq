"""
Example script demonstrating water body analysis using Farq.
Shows how to analyze and visualize water bodies from Landsat imagery.
"""
import farq
import os

def main():
    try:
        # Define paths
        base_dir = r"examples\chad lake"
        
        # Load Landsat bands for two time periods
        print("Loading raster datasets...")
        
        # 1985 data
        green_85, meta_85 = farq.read(os.path.join(base_dir, "1985", "LT05_L2SP_185051_19850501_20200918_02_T1_SR_B2.TIF"))
        nir_85, _ = farq.read(os.path.join(base_dir, "1985", "LT05_L2SP_185051_19850501_20200918_02_T1_SR_B4.TIF"))
        
        # 2024 data
        green_24, meta_24 = farq.read(os.path.join(base_dir, "2024 8", "LC08_L2SP_185051_20230323_20230404_02_T1_SR_B3.TIF"))
        nir_24, _ = farq.read(os.path.join(base_dir, "2024 8", "LC08_L2SP_185051_20230323_20230404_02_T1_SR_B5.TIF"))
        
        # Print original shapes
        print(f"\nOriginal shapes:")
        print(f"1985 image: {green_85.shape}")
        print(f"2024 image: {green_24.shape}")
        
        # Resample 2024 data to match 1985 dimensions
        print("\nResampling images to match...")
        green_24 = farq.resample(green_24, green_85.shape)
        nir_24 = farq.resample(nir_24, nir_85.shape)
        
        print(f"After resampling:")
        print(f"1985 image: {green_85.shape}")
        print(f"2024 image: {green_24.shape}")
        
        # Calculate NDWI for both periods
        print("\nCalculating NDWI...")
        ndwi_85 = farq.ndwi(green_85, nir_85)
        ndwi_24 = farq.ndwi(green_24, nir_24)
        
        # Analyze water distribution
        print("\nAnalyzing water distribution...")
        
        # Calculate 1985 water coverage
        water_mask_85 = ndwi_85 > 0
        water_pixels_85 = farq.sum(water_mask_85)
        water_percentage_85 = (water_pixels_85 / water_mask_85.size) * 100
        
        # Calculate 2024 water coverage
        water_mask_24 = ndwi_24 > 0
        water_pixels_24 = farq.sum(water_mask_24)
        water_percentage_24 = (water_pixels_24 / water_mask_24.size) * 100
        
        print("\nWater Coverage Analysis:")
        print(f"1985 Water coverage: {water_percentage_85:.2f}%")
        print(f"2024 Water coverage: {water_percentage_24:.2f}%")
        
        change = water_percentage_24 - water_percentage_85
        print(f"Change in water coverage: {change:.2f}%")
        
        # Visualize results
        print("\nVisualizing results...")
        farq.compare(
            ndwi_85, ndwi_24,
            title1="NDWI 1985",
            title2="NDWI 2024",
            cmap="RdYlBu",
            vmin=-1,
            vmax=1
        )
        farq.plt.show()
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 