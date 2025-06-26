import rasterio
import numpy as np
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling


input_root = "output_geotiff"
output_suffix = "_4326_system"
nodata_val = -9999.0  # use a clearly invalid value as NoData


def reproject_raster(input_path, output_path, dst_crs="EPSG:4326"):
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


for dirpath, _, filenames in os.walk(input_root):
    for fname in filenames:
        if not fname.lower().endswith(".tif"):
            continue

        input_path = os.path.join(dirpath, fname)
        output_path = os.path.join("output_4326_system_tiff", fname.replace(".tif", f"{output_suffix}.tif"))

        reproject_raster(input_path, output_path)

        print(f"✅ {output_path} saved with NoData for < 0.1")


