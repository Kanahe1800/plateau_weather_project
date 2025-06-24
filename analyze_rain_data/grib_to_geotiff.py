import pygrib
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Load GRIB2 file
file_name = "2025-06-24-11-30"
grbs = pygrib.open(f'{file_name}.bin')
grb = grbs.message(1)
data, lats, lons = grb.data()
data = data.filled(np.nan)  # convert MaskedArray to ndarray, filling masked values with NaN


# Convert level to mm/h
def level_to_mm_per_hr(level):
    if level == 0:
        return np.nan
    elif level == 1:
        return 0.0
    elif 2 <= level <= 20:
        return 0.1 + 0.1 * (level - 2)
    elif 21 <= level <= 32:
        return 2.0 + 0.25 * (level - 21)
    elif 33 <= level <= 42:
        return 5.0 + 0.5 * (level - 33)
    elif 43 <= level <= 212:
        return 10.0 + 1.0 * (level - 43)
    elif 213 <= level <= 250:
        return 180.0 + 2.0 * (level - 213)
    elif level == 251:
        return 260.0
    return np.nan

mm_per_hr = np.vectorize(level_to_mm_per_hr)(data)

# Calculate resolution and transform
lat_res = abs(lats[1, 0] - lats[0, 0])
lon_res = abs(lons[0, 1] - lons[0, 0])
transform = from_origin(lons[0, 0], lats[0, 0], lon_res, lat_res)

# Save as GeoTIFF
with rasterio.open(f'{file_name}.tif', "w", driver="GTiff",
                   height=mm_per_hr.shape[0], width=mm_per_hr.shape[1],
                   count=1, dtype=mm_per_hr.dtype,
                   crs="EPSG:4326", transform=transform,
                   nodata=np.nan) as dst:
    dst.write(mm_per_hr, 1)
