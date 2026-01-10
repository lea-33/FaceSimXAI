"""
Script to single LRP heatmaps for the odd-one-out decision task.
The model used is the VGGFaceHumanJudgment model.
"""

import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from zennit.composites import EpsilonAlpha2Beta1Flat
from functools import partial
from facesim3d.modeling.VGG.vgg_predict import prepare_data_for_human_judgment_model
from facesim3d.modeling.VGG.models import (
    load_trained_vgg_face_human_judgment_model,
    VGGFaceHumanjudgmentFrozenCoreWithLegs,
)
from facesim3d.lrp_utils import Gradient2, feed_attr_output_fn, get_orig_dataset
from facesim3d import local_paths
# ===============================
# === CONFIGURATION PARAMETERS ==
# ===============================
SESSION = "3D"
DATA_MODE = "3d-reconstructions"
LAST_CORE_LAYER = "fc7-relu"
BATCH_SIZE = 1
DTYPE = torch.float32
SHUFFLE = True
SEED = 42

MODEL_TYPE = "VGGFaceHumanJudgment"
DECISION = "odd_one_out"

SAVE_DIR = local_paths.DIR_SINGLE_HJ_HEATMAPS
os.makedirs(SAVE_DIR, exist_ok=True)

# Directory for single heatmaps + metadata CSV
SINGLE_MAPS_DIR = os.path.join(SAVE_DIR, "single_heatmaps")
os.makedirs(SINGLE_MAPS_DIR, exist_ok=True)
SINGLE_META_FILE = os.path.join(SAVE_DIR, "single_heatmaps_metadata.csv")

DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===========================================
# === Model, Data, Composite
# ===========================================
if SESSION == "2D":
    model_name_vgg = "2023-11-15_04-44_VGGFaceHumanjudgmentFrozenCore"
else:
    model_name_vgg = "2023-12-11_19-46_VGGFaceHumanjudgmentFrozenCore"

model_vgg = load_trained_vgg_face_human_judgment_model(
    session=SESSION,
    model_name=model_name_vgg,
    exclusive_gender_trials=None,
    device=DEVICE,
)
model = VGGFaceHumanjudgmentFrozenCoreWithLegs(frozen_top_model=model_vgg)
model.name = model_name_vgg + "WithLegs"
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.to(DEVICE)
print(f"Loaded pretrained model: {model.name}")

train_dl, val_dl, test_dl = prepare_data_for_human_judgment_model(
    session=SESSION,
    frozen_core=False,
    last_core_layer=LAST_CORE_LAYER,
    data_mode=DATA_MODE,
    split_ratio=(0, 0, 1),    # use all data
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    dtype=DTYPE,
)
dataset = get_orig_dataset(test_dl)
composite = EpsilonAlpha2Beta1Flat()

triplets = dataset.triplets.copy()
random.shuffle(triplets)
shuffled_indices = list(range(len(dataset)))
random.shuffle(shuffled_indices)

# ===========================================
# === Helper: compute heatmap for dataset index
# ===========================================
def compute_heatmap_for_dataset_index(idx, model, composite, dataset, device):
    """Compute LRP heatmap for the odd-one-out of triplet idx."""
    sample = dataset[idx]
    x1 = sample["image1"].to(device).unsqueeze(0)
    x2 = sample["image2"].to(device).unsqueeze(0)
    x3 = sample["image3"].to(device).unsqueeze(0)
    y = sample["choice"].to(device)  # true odd-one-out index
    inputs = (x1, x2, x3)

    t = dataset.triplets[idx]
    h1, h2, h3, choice_idx = int(t["head1"]), int(t["head2"]), int(t["head3"]), int(t["choice_idx"])
    gt_head_id = [h1, h2, h3][choice_idx]

    attributor = Gradient2(model, composite)

    with torch.no_grad():
        preds = model(*inputs)
        pred_class = preds.argmax(dim=1).item()

    output_relevance = partial(feed_attr_output_fn, target=pred_class)
    print(f"Processing idx={idx}: triplet=({h1},{h2},{h3}), model_pred={pred_class}")

    with attributor:
        _, relevance = attributor(input=inputs, attr_output=output_relevance)

    rel = relevance[pred_class].detach().cpu().numpy().squeeze()  # select relevance for predicted class
    if rel.ndim == 3:
        rel = rel.sum(axis=0)

    correct = (pred_class == y.item())
    return gt_head_id, rel, correct, pred_class

# ===========================================
# === Prepare metadata CSV
# ===========================================
with open(SINGLE_META_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "dataset_idx", "head1", "head2", "head3",
        "gt_odd_head", "model_pred", "human_choice_index", "correct",
        "heatmap_file"
    ])

# ===========================================
# === Main loop: save every individual LRP map
# ===========================================
skipped_errors = 0
total_processed = 0

pbar = tqdm(shuffled_indices, desc="Computing & saving single LRP heatmaps")
for idx in pbar:
    try:
        gt_head_id, heatmap, correct, pred_class = compute_heatmap_for_dataset_index(
            idx, model, composite, dataset, DEVICE
        )

        t = dataset.triplets[idx]
        h1, h2, h3, choice_idx = int(t["head1"]), int(t["head2"]), int(t["head3"]), int(t["choice_idx"])

        # get head ID from the prediction
        pred_head_id = [h1, h2, h3][pred_class]

        # Save heatmap (.npy)
        heatmap_fname = (f"triplet{idx:05d}_pred_head{pred_head_id:03d}.npy")
        heatmap_path = os.path.join(SINGLE_MAPS_DIR, heatmap_fname)
        np.save(heatmap_path, heatmap)

        # Also save PNG for quick visualization
        amax = np.max(np.abs(heatmap))
        vmin, vmax = -amax, amax
        plt.imsave(
            heatmap_path.replace(".npy", ".png"),
            heatmap,
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )

        # Append metadata
        with open(SINGLE_META_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx, h1, h2, h3, gt_head_id,
                pred_class, choice_idx, int(correct),
                heatmap_fname,
            ])

        total_processed += 1
        if total_processed % 50 == 0:
            pbar.set_postfix({"saved_maps": total_processed})

    except Exception as e:
        skipped_errors += 1
        if skipped_errors <= 5:
            print(f"[Warn] Error at idx={idx}: {e}")
        continue

print("\n=== DONE ===")
print(f"Total heatmaps saved: {total_processed}")
print(f"Skipped errors: {skipped_errors}")
print(f"Saved in: {SINGLE_MAPS_DIR}")
print(f"Metadata CSV: {SINGLE_META_FILE}")
