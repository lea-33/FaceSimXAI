"""
Calculates the explained variance of the PCA on averaged LRP heatmaps.
Visualizes the principal components to interpret the main axes of variation in the model's relevance structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import os
import re
import glob
from facesim3d import local_paths

# ==== CONFIG ====

model_type = "MaxP"    # else: HumJudge
print("Model Type:", model_type)

if model_type == "HumJudge":
    input_dir = local_paths.DIR_AVERAGE_HEATMAPS_HUM
    output_dir = local_paths.DIR_PCA_AE_RESULTS_HJ
    exclude_file = "overall_average_odd_one_out_VGGFaceHumanJudgment.npy"

else:
    input_dir = local_paths.DIR_AVERAGE_HEATMAPS_MaxP
    output_dir = local_paths.DIR_PCA_AE_RESULTS_MaxP
    exclude_file = "overall_average_odd_one_out_VGGFace_Maxp5_3_Sim"
    
os.makedirs(output_dir, exist_ok=True)


npy_files = [
    f for f in sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not f.endswith(exclude_file)
]
print(f"Found {len(npy_files)} heatmaps")

# ==== LOAD ALL HEATMAPS ====
avg_heatmaps = np.stack([np.load(f) for f in npy_files])
print("Loaded heatmap array:", avg_heatmaps.shape)

# ==== FLATTEN ====
flat_heatmaps = avg_heatmaps.reshape(avg_heatmaps.shape[0], -1)
print("Flattened shape:", flat_heatmaps.shape)

# ==== PCA – compute full variance spectrum ====
pca = PCA(n_components=min(flat_heatmaps.shape), random_state=42)
pca.fit(flat_heatmaps)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

# ==== FIND 95% THRESHOLD ====
threshold = 0.95
n_components_95 = np.argmax(cumulative >= threshold) + 1

print("\n===== PCA Variance Summary =====")
print(f"Total components     : {len(explained)}")
print(f"Components for 95%   : {n_components_95}")
print(f"Cumulative variance  : {cumulative[n_components_95-1]:.4f}")

# (Optional) print table
for i in range(1, 51):
    print(f"PC {i}: cumulative variance = {cumulative[i-1]:.4f}")


# Get spatial shape
H, W = avg_heatmaps.shape[1], avg_heatmaps.shape[2]

# All PCA components (each is 1 × D flattened)
components = pca.components_[:n_components_95]     # shape: [K, H*W]

# Reshape components → 2D heatmaps
pc_maps = components.reshape(n_components_95, H, W)  # shape: [K, H, W]

# ==== PLOT ALL PCS IN ONE FIGURE ====
n_cols = 5
n_rows = int(np.ceil(n_components_95 / n_cols))

plt.figure(figsize=(3*n_cols, 3*n_rows))

for i, pc in enumerate(pc_maps):
    var = pca.explained_variance_ratio_[i]
    plt.subplot(n_rows, n_cols, i+1)
    amax = np.max(np.abs(pc))   # symmetric
    plt.imshow(pc, cmap='seismic', vmin=-amax, vmax=amax)
    plt.title(f"PC {i+1}, Explained Variance: {var:.3f}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir,"PCA_avg_components_up_to_95variance.png"), dpi=300)
plt.close()

print(f"Saved all PCA components (1–{n_components_95}) in one figure.")
