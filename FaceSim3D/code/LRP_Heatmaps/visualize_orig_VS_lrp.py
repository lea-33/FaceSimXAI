"""
Script to visualize original 3D face reconstructions alongside LRP maps from different models.
Facilitates a qualitative visual comparison of model attention patterns.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from facesim3d import local_paths
SAVE_DIR = local_paths.DIR_HEATMAP_COMPARISON
os.makedirs(SAVE_DIR, exist_ok=True)

# === FOLDER PATHS  ===
folder_3d_reconstruction = local_paths.DIR_FRONTAL_VIEW_HEADS
folder_lrp_vgg_face      = local_paths.DIR_VGG_FACE_HEATMAPS
folder_lrp_vgg_hum_judge = local_paths.DIR_AVERAGE_HEATMAPS_HUM
folder_lrp_vgg_maxp_sim  = local_paths.DIR_AVERAGE_HEATMAPS_MaxP

# === HEAD IDs TO PLOT ===
head_ids = [i for i in range(0, 100)]

# === LOOP OVER HEADS ===
for head_id in head_ids:
    print(f"Processing Head {head_id:03d}...")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle(f"Head {head_id:03d}", fontsize=14)

    # File paths
    path_3d = os.path.join(folder_3d_reconstruction, f"head-{head_id:03d}.png")
    path_face = os.path.join(folder_lrp_vgg_face, f"head_{head_id:03d}.npy")
    path_hum = os.path.join(folder_lrp_vgg_hum_judge, f"head_{head_id:03d}_average_odd_one_out_VGGFaceHumanJudgment.npy")
    path_sim = os.path.join(folder_lrp_vgg_maxp_sim, f"head_{head_id:03d}_average_odd_one_out_VGGFace_Maxp5_3_Sim.npy")

    # Load images (LRP maps can be .npy or .png)
    def load_img(path):
        if path.endswith(".npy"):
            arr = np.load(path)
            return arr
        else:
            return np.array(Image.open(path))

    imgs = [load_img(p) for p in [path_3d, path_face, path_hum, path_sim]]
    titles = ["3D Reconstruction", "LRP VGG-Face", "LRP Human-Judgement", "LRP Maxp5_3-Sim"]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap="seismic" if "LRP" in title else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    save_path = os.path.join(SAVE_DIR, f"Head_{head_id:03d}_comparison.png")
    plt.savefig(save_path)
    plt.close()
