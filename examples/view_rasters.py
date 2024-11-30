import farq
import matplotlib.pyplot as plt
import os

def main():
    # Define the full paths to the raster files
    base_dir = r"D:\feras\Study\4th Sem\GEOSPATIAL PROCESSING\PROJECT\Farq\Farq\examples"
    raster1_path = os.path.join(base_dir, "LT05_L2SP_174038_19850824_20200918_02_T1_SR_B5.TIF")
    raster2_path = os.path.join(base_dir, "LC08_L2SP_174038_20240312_20240401_02_T1_SR_B5.TIF")

    # Load the two raster datasets
    print("Loading raster datasets...")
    print(f"Loading first raster from: {raster1_path}")
    print(f"Loading second raster from: {raster2_path}")
    
    raster1, meta1 = farq.load_raster(raster1_path)
    raster2, meta2 = farq.load_raster(raster2_path)

    # Print basic information about the rasters
    print("\nRaster Information:")
    print("First raster shape:", raster1.shape)
    print("Second raster shape:", raster2.shape)
    print("\nFirst raster metadata:")
    for key, value in meta1.items():
        print(f"{key}: {value}")

    # Visualize individual rasters
    print("\nPlotting individual rasters...")
    farq.plot_raster(raster1, title="Landsat 5 (1985) - Band 5", cmap="viridis")
    plt.show()

    farq.plot_raster(raster2, title="Landsat 8 (2024) - Band 5", cmap="viridis")
    plt.show()

    # Visualize both rasters side by side
    print("\nPlotting rasters side by side...")
    farq.plot_raster_pair(
        raster1, raster2,
        title1="Landsat 5 (1985) - Band 5",
        title2="Landsat 8 (2024) - Band 5",
        cmap="viridis"
    )
    plt.show()

if __name__ == "__main__":
    main() 