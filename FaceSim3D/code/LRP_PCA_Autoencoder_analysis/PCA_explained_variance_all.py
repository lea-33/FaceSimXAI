"""
Performs Incremental PCA on the flattened heatmaps of the entire dataset.
Calculates the explained variance and transforms the heatmaps into the PCA space for structural analysis.
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
#input_dir = local_paths.DIR_SINGLE_HJ_HEATMAPS   # VGG-Hum
input_dir = local_paths.DIR_SINGLE_MaxP_HEATMAPS   # VGG-MaxP

output_dir = local_paths.DIR_PCA_AE_RESULTS_MaxP
#output_dir = local_paths.DIR_PCA_AE_RESULTS_HJ
os.makedirs(output_dir, exist_ok=True)

npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
print(f"Found {len(npy_files)} heatmaps")


#mm_path = os.path.join(output_dir, "heatmaps_memmap_HJ.dat")    # VGG-Hum
mm_path = os.path.join(output_dir, "heatmaps_memmap_MaxP.dat")    # VGG-MaxP

flat_dim = 224 * 224   # flatten size
N = len(npy_files)

# Load again
X = np.memmap(mm_path, dtype="float32", mode="r", shape=(N, flat_dim))

from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

n_components = 1000
batch_size = 2000

ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

for start in tqdm(range(0, N, batch_size), desc="Fitting PCA"):
    ipca.partial_fit(X[start:start + batch_size])

X_pca = np.zeros((N, n_components), dtype="float32")

for start in tqdm(range(0, N, batch_size), desc="Transforming"):
    X_pca[start:start + batch_size] = ipca.transform(X[start:start + batch_size])

import numpy as np

# cumulative explained variance
cumvar = np.cumsum(ipca.explained_variance_ratio_)

for i, cv in enumerate(cumvar, start=1):
    print(f"PC {i}: cumulative variance = {cv:.3f}")

#with open(os.path.join(output_dir,"pca_exVAR_HJ.txt"), "w") as f:    # VGG-Hum
with open(os.path.join(output_dir,"pca_exVAR_MaxP.txt"), "w") as f:    # VGG-MaxP
    f.write(f"Found {len(npy_files)} heatmaps\n")
    for i, cv in enumerate(cumvar, start=1):
        f.write(f"PC {i}: cumulative variance = {cv:.3f}\n")
