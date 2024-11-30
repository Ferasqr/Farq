import farq

def main():
    # Define the full paths to the raster files
    base_dir = r"examples\2024 Water"
    
    # Load Landsat 8 bands
    green_path = farq.os.path.join(base_dir, "LC08_L2SP_174038_20241123_20241127_02_T1_SR_B3.TIF")
    nir_path = farq.os.path.join(base_dir, "LC08_L2SP_174038_20241123_20241127_02_T1_SR_B5.TIF")
    
    print("Loading raster datasets...")
    green, green_meta = farq.read(green_path)
    nir, nir_meta = farq.read(nir_path)
    
    # Calculate NDWI
    print("Calculating NDWI...")
    ndwi = farq.ndwi(green, nir)
    
    # Create water mask (NDWI > 0)
    water_mask = ndwi > 0
    
    # Visualize the input bands
    print("\nPlotting input bands...")
    farq.compare(
        green, nir,
        title1="Green Band",
        title2="NIR Band",
        cmap="viridis"
    )
    farq.plt.show()
    
    # Visualize NDWI
    print("\nPlotting NDWI...")
    farq.plot(
        ndwi,
        title="NDWI",
        cmap="RdYlBu",  # Blue for water, red for non-water
        vmin=-1,
        vmax=1
    )
    farq.plt.show()
    
    # Visualize water mask
    print("\nPlotting Water Mask...")
    farq.plot(
        water_mask.astype(farq.np.float32),
        title="Water Mask",
        cmap="Blues",  # Blue for water, white for non-water
        vmin=0,
        vmax=1
    )
    farq.plt.show()
    
    # Print basic statistics
    print("\nNDWI Statistics:")
    ndwi_stats = farq.stats(ndwi)
    print(f"Mean: {ndwi_stats['mean']:.4f}")
    print(f"Min: {ndwi_stats['min']:.4f}")
    print(f"Max: {ndwi_stats['max']:.4f}")
    print(f"\nWater Coverage: {farq.np.mean(water_mask) * 100:.2f}%")
    
    # Plot NDWI histogram
    farq.plt.figure(figsize=(10, 6))
    farq.hist(ndwi, bins=100, title="NDWI Distribution")
    farq.plt.axvline(x=0, color='r', linestyle='--', label='Water Threshold')
    farq.plt.legend()
    farq.plt.show()

if __name__ == "__main__":
    main() 