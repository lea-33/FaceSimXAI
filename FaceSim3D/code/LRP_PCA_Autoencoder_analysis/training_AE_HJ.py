"""
Trains an Autoencoder on the LRP heatmaps from the VGG-Hum model.
Learns a compressed latent representation of the relevance maps for structural analysis and clustering.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from tqdm import tqdm
import random
import csv
from facesim3d import local_paths

# --- Settings ---
image_size = 128
batch_size = 128
latent_dim = 32
epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Set to " + ("cuda" if torch.cuda.is_available() else "cpu"))
random.seed(42)

output_dir = local_paths.DIR_PCA_AE_RESULTS_HJ

# --- Gaussian Noise Transform ---
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


# --- Thresholding Relevance Transform ---
class RelevanceThreshold:
    def __init__(self, threshold_ratio=0.3):
        self.threshold_ratio = threshold_ratio

    def __call__(self, tensor):
        max_val = tensor.max().item()
        threshold = max_val * self.threshold_ratio
        return torch.where(tensor >= threshold, tensor, torch.tensor(0.0, device=tensor.device))


# --- Dataset ---
class HeatmapDataset(Dataset):
    def __init__(self, root_glob, transform=None):
        self.image_paths = glob.glob(os.path.join(root_glob, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # single-channel
        if self.transform:
            image = self.transform(image)
        return image, img_path  # return path 

# Gaussian Blur Method
transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.02),
])

dataset = HeatmapDataset(local_paths.DIR_SINGLE_HJ_HEATMAPS, transform)

# Leave one image out for reconstruction
total_size = len(dataset)
indices = list(range(total_size))
random.shuffle(indices)

# take 5% for holdout
holdout_size = round(0.05 * total_size) 
train_size = total_size - holdout_size

# Split
train_indices = indices[:train_size]
holdout_indices = indices[train_size:]

# Create subsets
train_dataset = Subset(dataset, train_indices)
holdout_dataset = Subset(dataset, holdout_indices)


dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)


# --- Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, latent_dim)         # From conv output to latent
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),        # From latent to conv shape
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# --- Training ---
train_losses = []

print("Starting training...")
model.train()
for epoch in range(epochs):
    total_loss = 0
    for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon, _ = model(imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss) 
    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

# --- Extract Latent Codes ---
print("Extracting latent representations...")
model.eval()
all_latents = []
all_paths = []

with torch.no_grad():
    for imgs, paths in tqdm(DataLoader(dataset, batch_size=256), desc="Encoding"):
        imgs = imgs.to(device)
        _, z = model(imgs)
        all_latents.append(z.cpu())
        all_paths.extend(paths)

latents = torch.cat(all_latents, dim=0).numpy()

# Save important data to perform the latent space visualization
np.save(os.path.join(output_dir,"latent_codes_HJ.npy"), latents)
np.save(os.path.join(output_dir,"image_paths_HJ.npy"), all_paths)


# --- Save training log to CSV ---
log_filename =  os.path.join(output_dir,"train_log_AE_HJ.csv")
with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "AverageLoss"])
    for epoch_num, loss_val in enumerate(train_losses, start=1):
        writer.writerow([epoch_num, loss_val])
print(f"Training log saved to {log_filename}")

# --- Save model weights ---
model_save_path = os.path.join(output_dir,"AE_weights_HJ.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")

# ---- VALIDATION ----
# Check whether the AutoEncoder works
import matplotlib.pyplot as plt

print("Starting validation...")

# Set model to eval mode
model.eval()

# Build a small DataLoader for the 20-image holdout set ---
holdout_loader = DataLoader(holdout_dataset, batch_size=20, shuffle=False)

val_loss_sum = 0.0
num_samples = 0


with torch.no_grad():
    for imgs, _ in holdout_loader:
        imgs = imgs.to(device)
        recon, _ = model(imgs)
        loss = criterion(recon, imgs)  # mean over batch
        # accumulate properly even if batch < 20
        val_loss_sum += loss.item() * imgs.size(0)
        num_samples += imgs.size(0)

val_loss = val_loss_sum / max(1, num_samples)
print(f"Holdout (50 images) reconstruction loss: {val_loss:.6f}")

# --- 3) Visualize ONLY the first holdout sample (index 0) ---
print("Saving reconstruction example of first holdout image...")

# get the first (image, path/label) from the 20
img_tensor, _ = holdout_dataset[0]
img_tensor = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    recon_tensor, _ = model(img_tensor)

# move to CPU and remove batch dim
orig = img_tensor.squeeze().cpu().numpy()
recon = recon_tensor.squeeze().cpu().numpy()

# Plot side-by-side 
plt.figure(figsize=(6, 3)) 

plt.subplot(1, 2, 1) 
plt.imshow(orig, cmap='seismic') 
plt.title("Original") 
plt.axis('off') 

plt.subplot(1, 2, 2) 
plt.imshow(recon, cmap='seismic') 
plt.title("Reconstructed") 
plt.axis('off')

plt.tight_layout() 
plt.savefig(os.path.join(output_dir,"reconstruction_AE_HJ.png"))
