import rasterio
import numpy as np
import os

input_root = "output_geotiff"
output_suffix = "_nodata"
nodata_val = -9999.0  # use a clearly invalid value as NoData

for dirpath, _, filenames in os.walk(input_root):
    for fname in filenames:
        if not fname.lower().endswith(".tif"):
            continue

        input_path = os.path.join(dirpath, fname)
        output_path = os.path.join("output_transparent_geotiff", fname.replace(".tif", f"{output_suffix}.tif"))

        with rasterio.open(input_path) as src:
            data = src.read(1)
            profile = src.profile.copy()

        # Set values < 0.12 to NoData
        data = np.where(data < 0.1, nodata_val, data)
        profile.update(nodata=nodata_val, dtype=rasterio.float32)

        # Save updated raster
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)

        print(f"✅ {output_path} saved with NoData for < 0.1")
