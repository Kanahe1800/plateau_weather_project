import numpy as np
import rasterio
from datetime import datetime, timedelta
from scipy.stats import linregress
import os

def load_rainfall_stack(start_time_str, interval_minutes=5, count=12, folder="."):
    rainfall_stack = []
    times = []

    start_time = datetime.strptime(start_time_str, "%Y-%m-%d-%H-%M")
    for i in range(count):
        time = start_time + timedelta(minutes=i * interval_minutes)
        times.append(time)
        filename = f"{time.strftime('%Y-%m-%d-%H-%M')}.tif"
        filepath = os.path.join(folder, filename)

        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            nodata_val = src.nodata
            if nodata_val is not None:
                data[data == nodata_val] = 0.0
            data = np.nan_to_num(data, nan=0.0)
            rainfall_stack.append(data)

            if i == 0:
                profile = src.profile.copy()  # keep metadata safely


    rainfall_stack = np.stack(rainfall_stack, axis=0)  # shape: (12, H, W)
    print("finish creating rainfall stack")
    return rainfall_stack, times, profile

def compute_trend(rainfall_stack, interval_minutes=5):
    def fast_linear_trend(rain_stack, time_steps):
        T, H, W = rain_stack.shape
        X = np.asarray(time_steps)
        
        # Center x and precompute denominator
        x_mean = X.mean()
        x_centered = X - x_mean
        denom = np.sum(x_centered ** 2)

        # Flatten spatially
        Y = rain_stack.reshape(T, -1)  # shape (T, N)
        y_mean = Y.mean(axis=0)
        y_centered = Y - y_mean

        # Compute slopes for each pixel
        slopes = np.dot(x_centered, y_centered) / denom
        return slopes.reshape(H, W)

    time_steps = np.arange(rainfall_stack.shape[0]) * interval_minutes
    print("📈 Using vectorized slope calculation...")
    slopes = fast_linear_trend(rainfall_stack, time_steps)
    print("✅ Finished slope calculation.")
    return slopes

def compute_cumulative(rainfall_stack, steps):
    return rainfall_stack[-steps:, :, :].sum(axis=0)

def build_input_tensor(rainfall_stack, risk_attr_path=None):
    # Compute trend

    trend = compute_trend(rainfall_stack)
    print("finish analyzing trend")
    # Compute cumulative rainfall maps
    cum15 = compute_cumulative(rainfall_stack, 3)
    cum30 = compute_cumulative(rainfall_stack, 6)
    cum60 = compute_cumulative(rainfall_stack, 12)
    print("finish cumulative analysis")
    channels = list(rainfall_stack)  # 12 channels

    # Optional: add risk attribution layer
    if risk_attr_path:
        try:
            with rasterio.open(risk_attr_path) as src:
                risk_attr = src.read(1)
                print("Risk shape:", risk_attr.shape)
                channels.append(risk_attr)  # channel 13
        except Exception as e:
            print(f"⚠️ Failed to load risk layer: {e}")
            dummy_risk = np.zeros_like(rainfall_stack[0])
            channels.append(dummy_risk)
    else:
        print("⚠️ No risk attribution provided. Using rainfall only.")
        dummy_risk = np.zeros_like(rainfall_stack[0])
        channels.append(dummy_risk)

    channels.append(trend)   # channel 14
    channels.append(cum15)   # channel 15
    channels.append(cum30)   # channel 16
    channels.append(cum60)   # channel 17

    # Final tensor: shape = (H, W, 16)
    stacked = np.stack(channels, axis=-1)
    return stacked

# === USAGE ===

# Folder with the GeoTIFFs
tiff_folder = "./rainfall_data"  # change to your path
start_time = "2025-06-26-13-10"

# Load 12 rainfall files
rainfall_stack, times, profile = load_rainfall_stack(start_time, folder=tiff_folder)

# Build the full input tensor
input_tensor = build_input_tensor(rainfall_stack)

# Save as NumPy array (optional)
np.save("input_tensor_2025-06-26-14-10.npy", input_tensor)
print("✅ 16-channel input tensor shape:", input_tensor.shape)
