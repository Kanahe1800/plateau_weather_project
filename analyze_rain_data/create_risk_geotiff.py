import numpy as np
import torch
from torch import nn
import torch.optim as optim
import rasterio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime

# === Load and Prepare Input ===
tensor = np.load("input_tensor_2025-06-24-11-30.npy")  # shape: (H, W, 16)
H, W, C = tensor.shape

# Transpose to (C, H, W) for CNN
tensor = tensor.transpose(2, 0, 1)  # shape: (16, H, W)

# Normalize each channel independently
scaler = StandardScaler()
tensor_reshaped = tensor.reshape(C, -1).T  # shape: (H*W, C)
tensor_scaled = scaler.fit_transform(tensor_reshaped).T.reshape(C, H, W)

# ✅ Auto-crop to make divisible by patch size
patch_size = 64
new_H = (H // patch_size) * patch_size
new_W = (W // patch_size) * patch_size

if new_H != H or new_W != W:
    print(f"⚠️ Cropping input from ({H}, {W}) → ({new_H}, {new_W}) for 64×64 patching")
    tensor_scaled = tensor_scaled[:, :new_H, :new_W]
    H, W = new_H, new_W

# Convert to torch tensor
tensor_torch = torch.tensor(tensor_scaled, dtype=torch.float32)

# Divide into non-overlapping 64×64 patches
tensor_patches = tensor_torch.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
# shape: (C, H//64, W//64, 64, 64)
tensor_patches = tensor_patches.permute(1, 2, 0, 3, 4).reshape(-1, C, patch_size, patch_size)
print(f"📦 Extracted {tensor_patches.shape[0]} patches of size {patch_size}×{patch_size}")

# === Define CNN Autoencoder ===
class CNNAutoencoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# === Model & Training ===
model = CNNAutoencoder(channels=C)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

patches = tensor_patches.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
if torch.isnan(patches).any():
    print("❌ Found NaNs in input patches — please check input tensor!")
    exit(1)
    
print("🚀 Training CNN autoencoder...")
for epoch in range(11):
    model.train()
    optimizer.zero_grad()
    
    recon = model(patches)
    loss = loss_fn(recon, patches)
    
    # 🔍 Check for NaNs before backward
    if torch.isnan(loss):
        print(f"❌ NaN loss at epoch {epoch}, aborting training.")
        break

    loss.backward()
    
    # ✅ Clip gradients to avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

    # ⏱️ Log epoch timing and loss
    now = datetime.datetime.now()
    print(f"{epoch:02d} done at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Loss: {loss.item():.6f}")

# === Inference: Compute Anomaly Map ===
model.eval()
with torch.no_grad():
    recon_patches = model(patches)

errors = ((recon_patches - patches) ** 2).mean(dim=1).cpu().numpy()  # shape: (N, 64, 64)

# Reconstruct full image from patches
nH = H // patch_size
nW = W // patch_size
risk_map = errors.reshape(nH, nW, patch_size, patch_size).transpose(0, 2, 1, 3).reshape(H, W)

# Normalize 0–1
risk_map_norm = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min())

# === Save GeoTIFF ===
example_tif = "rainfall_data/2025-06-24-10-35.tif"
with rasterio.open(example_tif) as src:
    profile = src.profile.copy()
    transform = src.transform

# Crop transform if necessary
profile.update(dtype=rasterio.float32, count=1, height=H, width=W)
with rasterio.open("flood_risk_cnn.tif", "w", **profile) as dst:
    dst.write(risk_map_norm.astype(np.float32), 1)

print("✅ Flood risk map saved as flood_risk_cnn.tif")

# === Visualize ===
plt.figure(figsize=(8, 6))
plt.imshow(risk_map_norm, cmap="hot")
plt.title("🔥 CNN-Based Flood Risk Map")
plt.colorbar(label="Flood Risk Score")
plt.tight_layout()
plt.show()
