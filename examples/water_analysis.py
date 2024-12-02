import farq
import numpy as np

def main():
    # Define the full paths to the raster files
    base_dir = r"examples\chad lake"
    
    # Load Landsat bands for two time periods
    print("Loading raster datasets...")
    
    # 1985 data
    green_85, meta_85 = farq.read(farq.os.path.join(base_dir, "1985\LT05_L2SP_185051_19850501_20200918_02_T1_SR_B2.TIF"))
    nir_85, _ = farq.read(farq.os.path.join(base_dir, "1985\LT05_L2SP_185051_19850501_20200918_02_T1_SR_B4.TIF"))
    
    # 2024 data
    green_24, meta_24 = farq.read(farq.os.path.join(base_dir, "2024 8\LC08_L2SP_185051_20230323_20230404_02_T1_SR_B3.TIF"))
    nir_24, _ = farq.read(farq.os.path.join(base_dir, "2024 8\LC08_L2SP_185051_20230323_20230404_02_T1_SR_B5.TIF"))
    
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
    
    # Visualize NDWI results
    print("\nPlotting NDWI results...")
    farq.compare(
        ndwi_85, ndwi_24,
        title1="NDWI 1985",
        title2="NDWI 2024",
        cmap="RdYlBu",
        vmin=-1,
        vmax=1
    )
    farq.plt.show()
    
    # Create water masks (NDWI > 0)
    water_mask_85 = ndwi_85 > 0
    water_mask_24 = ndwi_24 > 0
    
    # Calculate basic water statistics
    print("\nCalculating water statistics...")
    stats_85 = farq.water_stats(water_mask_85, pixel_size=30.0)
    stats_24 = farq.water_stats(water_mask_24, pixel_size=30.0)
    
    print("\n1985 Water Statistics:")
    print(f"Total water area: {stats_85['total_area']:.2f} km²")
    print(f"Water coverage: {stats_85['coverage_percent']:.2f}%")
    
    print("\n2024 Water Statistics:")
    print(f"Total water area: {stats_24['total_area']:.2f} km²")
    print(f"Water coverage: {stats_24['coverage_percent']:.2f}%")
    
    # Analyze changes between periods
    print("\nAnalyzing temporal changes...")
    changes = farq.water_change(water_mask_85, water_mask_24, pixel_size=30.0)
    
    print("\nChange Statistics:")
    print(f"New water area: {changes['gained_area']:.2f} km²")
    print(f"Lost water area: {changes['lost_area']:.2f} km²")
    print(f"Net change: {changes['net_change']:.2f} km²")
    print(f"Percent change: {changes['change_percent']:.2f}%")
    
    # Compare water masks
    print("\nVisualizing water extent changes...")
    farq.compare(
        water_mask_85, water_mask_24,
        title1="Water Extent 1985",
        title2="Water Extent 2024",
        cmap="Blues",
        vmin=0,
        vmax=1
    )
    farq.plt.show()
    
    # Get and analyze individual water bodies
    print("\nAnalyzing individual water bodies...")
    labeled_85, areas_85 = farq.get_water_bodies(water_mask_85, pixel_size=30.0, min_area=9000)
    labeled_24, areas_24 = farq.get_water_bodies(water_mask_24, pixel_size=30.0, min_area=9000)
    
    # Print detailed water body statistics
    print("\nDetailed Water Body Analysis:")
    print("\n1985:")
    print(f"Number of water bodies: {len(areas_85)}")
    if areas_85:
        print(f"Average water body size: {np.mean(list(areas_85.values())):.2f} km²")
        print(f"Largest water body: {max(areas_85.values()):.2f} km²")
    
    print("\n2024:")
    print(f"Number of water bodies: {len(areas_24)}")
    if areas_24:
        print(f"Average water body size: {np.mean(list(areas_24.values())):.2f} km²")
        print(f"Largest water body: {max(areas_24.values()):.2f} km²")
    
    # Visualize individual water bodies
    print("\nVisualizing individual water bodies...")
    farq.compare(
        labeled_85, labeled_24,
        title1="Water Bodies 1985",
        title2="Water Bodies 2024",
        cmap="tab20"
    )
    farq.plt.show()
    
    # Plot water body size distributions
    if areas_85 and areas_24:
        print("\nPlotting water body size distributions...")
        farq.plt.figure(figsize=(12, 6))
        farq.plt.subplot(121)
        farq.plt.hist(list(areas_85.values()), bins=30, color='blue', alpha=0.6)
        farq.plt.title("Water Body Size Distribution 1985")
        farq.plt.xlabel("Area (km²)")
        farq.plt.ylabel("Count")
        
        farq.plt.subplot(122)
        farq.plt.hist(list(areas_24.values()), bins=30, color='blue', alpha=0.6)
        farq.plt.title("Water Body Size Distribution 2024")
        farq.plt.xlabel("Area (km²)")
        farq.plt.ylabel("Count")
        
        farq.plt.tight_layout()
        farq.plt.show()

if __name__ == "__main__":
    main() 