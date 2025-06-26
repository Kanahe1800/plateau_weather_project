import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---- Step 1: Define Dataset ----
class RainfallLandslideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, C)
        self.y = torch.tensor(y, dtype=torch.long)     # shape: (N,)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---- Step 2: Define Model ----
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---- Step 3: Load data ----
input_tensor = np.load("input_tensor_2025-06-26-14-10.npy")  # shape (H, W, 17)
with rasterio.open("aligned_landslide_to_rain_cleaned.tif") as src:
    label = src.read(1).astype(np.uint8)  # shape (H, W)
    profile = src.profile.copy()

H, W, C = input_tensor.shape
assert label.shape == (H, W)

# Filter valid data (where label is in 0, 1, 2, 3)
valid_mask = label <= 3
input_tensor = input_tensor[valid_mask]
label = label[valid_mask]  # values: 0 (no risk), 1-3 (risk levels)

# ---- Step 4: Train/test split ----
X_train, X_val, y_train, y_val = train_test_split(input_tensor, label, test_size=0.2, random_state=42)

train_dataset = RainfallLandslideDataset(X_train, y_train)
val_dataset = RainfallLandslideDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

# ---- Step 5: Train ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(input_dim=17, num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training started...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).long()

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("Training finished.")
torch.save(model.state_dict(), "model.pth")

# ---- Step 6: Inference for whole raster and save GeoTIFF ----
input_tensor = np.load("input_tensor_2025-06-26-14-10.npy")  # reload full
H, W, C = input_tensor.shape
input_flat = input_tensor.reshape(-1, C)

model.eval()
with torch.no_grad():
    preds = []
    for i in range(0, input_flat.shape[0], 1024):
        batch = input_flat[i:i+1024]
        batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
        out = model(batch_tensor)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        preds.append(pred)

pred_map = np.concatenate(preds).reshape(H, W).astype(np.uint8)

# ---- Step 7: Save as GeoTIFF ----
profile.update({"dtype": "uint8", "count": 1, "compress": "DEFLATE"})
with rasterio.open("predicted_landslide_risk.tif", "w", **profile) as dst:
    dst.write(pred_map, 1)

print("Prediction saved to predicted_landslide_risk.tif")
