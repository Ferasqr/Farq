import farq

def main():
    # Define the full paths to the raster files
    base_dir = r"examples\2024 Forest Example"
    
    # Load Landsat 8 bands
    blue_path = farq.os.path.join(base_dir, "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B2.TIF")
    green_path = farq.os.path.join(base_dir, "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B3.TIF")
    red_path = farq.os.path.join(base_dir, "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B4.TIF")
    nir_path = farq.os.path.join(base_dir, "LC08_L2SP_227065_20240705_20240712_02_T1_SR_B5.TIF")
    
    
    print("Loading raster datasets...")
    blue, _ = farq.read(blue_path)
    green, _ = farq.read(green_path)
    red, _ = farq.read(red_path)
    nir, _ = farq.read(nir_path)
    
    
    # Calculate indices
    print("\nCalculating spectral indices...")
    
    # Vegetation indices
    ndvi_result = farq.ndvi(nir, red)
    savi_result = farq.savi(nir, red)
    evi_result = farq.evi(nir, red, blue)
    
    # Water indices
    ndwi_result = farq.ndwi(green, nir)
    
    
  
    
    # Visualize all indices
    indices = {
        "NDVI (Vegetation)": (ndvi_result, "RdYlGn"),  # Red-Yellow-Green
        "SAVI (Soil-Adjusted Vegetation)": (savi_result, "RdYlGn"),
        "EVI (Enhanced Vegetation)": (evi_result, "RdYlGn"),
        "NDWI (Water)": (ndwi_result, "RdYlBu")  # Red-Yellow-Blue
        
    }
    
    # Plot each index
    for title, (data, cmap) in indices.items():
        print(f"\nPlotting {title}...")
        farq.plot(
            data,
            title=title,
            cmap=cmap,
            vmin=-1,
            vmax=1
        )
        farq.plt.show()
        
        # Print basic statistics
        print(f"\n{title} Statistics:")
        index_stats = farq.stats(data)
        print(f"Mean: {index_stats['mean']:.4f}")
        print(f"Min: {index_stats['min']:.4f}")
        print(f"Max: {index_stats['max']:.4f}")
        
        # Plot histogram
        farq.hist(data, bins=100, title=f"{title} Distribution")
        farq.plt.show()

if __name__ == "__main__":
    main() 