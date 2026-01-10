"""
Heatmap averaging script for the VGG-MaxP model.
Computes aggregated relevance maps to compare the computational model's focus against the human-aligned model.
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
from facesim3d.modeling.VGG.prepare_data import prepare_data_for_maxp5_3_similarity_model
from facesim3d.lrp_utils import Gradient2, feed_attr_output_fn, get_orig_dataset, load_model_for_LRP

# ===============================
# === CONFIGURATION PARAMETERS ==
# ===============================
SESSION = "3D"                         # "2D" or "3D"
DATA_MODE = "3d-reconstructions"       # used by prepare_data_for_human_judgment_model
LAST_CORE_LAYER = "fc7-relu"
BATCH_SIZE = 1
DTYPE = torch.float32
SHUFFLE = True

TARGET_HEATMAPS_PER_HEAD = 100         # stop once every head has >= this many heatmaps
PLAN_B_MIN_HEATMAPS_PER_HEAD = 20      # if data runs out, still save heads with >= this
SEED = 42

MODEL_TYPE = "VGGFace_Maxp5_3_Sim"
DECISION = "odd_one_out"
METHOD = "relative"  # or "centroid"

from facesim3d import local_paths

SAVE_DIR = local_paths.DIR_AVERAGE_HEATMAPS_MaxP
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, f"run_log_efficient_average_triplet_heatmaps_{MODEL_TYPE}.csv")

DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===========================================
# === Model, Data, Composite
# ===========================================
model, model_name = load_model_for_LRP(method=METHOD, session=SESSION, device=DEVICE)
model.to(DEVICE)

model.eval()
for param in model.parameters():
    param.requires_grad = False
model.to(DEVICE)
print(f"Loaded pretrained model: {model_name}")

# Data: a single big split is fine (we just need access to the dataset)
train_dl, val_dl, test_dl, set_length = prepare_data_for_maxp5_3_similarity_model(
    session=SESSION,
    method=METHOD,
    frozen_core=False,
    last_core_layer=LAST_CORE_LAYER,
    data_mode="3d-reconstructions",
    split_ratio=(0.98, 0.01, 0.01),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    dtype=DTYPE,
    size=None,
)

dataset = get_orig_dataset(test_dl)
composite = EpsilonAlpha2Beta1Flat()

# Heads present in the dataset (dataset.session_data has columns head1, head2, head3, head_odd)
all_heads = np.unique(dataset.session_data[['head1','head2','head3']].to_numpy().flatten()).astype(int).tolist()

# Storage for heatmaps per (odd) head
head_heatmaps = {h: [] for h in all_heads}

# Precompute triplet list (each has head1, head2, head3, choice_idx)
triplets = dataset.triplets.copy()
random.shuffle(triplets)

# Index map from triplet tuple to dataset index (optional; only needed if you want idx-based retrieval)
# Here, we’ll just use dataset[...] via a running index since dataset[i] corresponds to session_data row i.
# But dataset.triplets is derived from session_data order, so we need to keep an index mapping.
# Build a list of indices in a shuffled order to iterate efficiently.
shuffled_indices = list(range(len(dataset)))
random.shuffle(shuffled_indices)

# ===========================================
# === Helper: compute heatmap for a dataset index (patched + debug)
# ===========================================
def compute_heatmap_for_dataset_index(idx, model, composite, dataset, device):
    """
    Compute LRP heatmap for the odd-one-out of the triplet at dataset index `idx`.
    Returns (odd_head_id, heatmap) or (None, None) on failure.
    """
    sample = dataset[idx]
    x1 = sample["image1"].to(device).unsqueeze(0)
    x2 = sample["image2"].to(device).unsqueeze(0)
    x3 = sample["image3"].to(device).unsqueeze(0)
    y = sample["choice"].to(device)
    inputs = (x1, x2, x3)

    t = dataset.triplets[idx]
    h1, h2, h3, choice_idx = int(t["head1"]), int(t["head2"]), int(t["head3"]), int(t["choice_idx"])
    odd_head_id = [h1, h2, h3][choice_idx]

    # Lightweight progress info every few triplets
    if idx % 10000 == 0:
        print(f"[Debug] idx={idx}, triplet=({h1},{h2},{h3}), odd={odd_head_id}")

    attributor = Gradient2(model, composite)
    output_relevance = partial(feed_attr_output_fn, target=y.item())

    with attributor:
        model_output, relevance = attributor(input=inputs, attr_output=output_relevance)

    # --- Safe relevance extraction (handles list/tuple/tensor) ---
    if isinstance(relevance, (list, tuple)):
        rel = relevance[y.item()]
    else:
        rel = relevance

    rel = rel.detach().cpu().numpy().squeeze()
    if rel.ndim == 3:
        rel = rel.sum(axis=0)

    return odd_head_id, rel

# ===========================================
# === Main loop: iterate only over VALID triplets
# ===========================================
total_processed = 0
skipped_errors = 0

# running_mean[head] = (mean_array, count)
running_means = {h: [None, 0] for h in all_heads}

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["head_id", "collected_heatmaps", "target", "note"])

pbar = tqdm(shuffled_indices, desc="Processing triplets (LRP)")
for idx in pbar:
    # Stop if all heads reached target
    if all(v[1] >= TARGET_HEATMAPS_PER_HEAD for v in running_means.values()):
        pbar.set_description("Reached target for all heads")
        break

    t = dataset.triplets[idx]
    h1, h2, h3, choice_idx = int(t["head1"]), int(t["head2"]), int(t["head3"]), int(t["choice_idx"])
    odd_head = [h1, h2, h3][choice_idx]

    if running_means[odd_head][1] >= TARGET_HEATMAPS_PER_HEAD:
        continue

    try:
        odd_head_id, heatmap = compute_heatmap_for_dataset_index(idx, model, composite, dataset, DEVICE)
        if heatmap is None:
            skipped_errors += 1
            continue

        mean, count = running_means[odd_head_id]

        # --- Initialize or update running mean ---
        if mean is None:
            running_means[odd_head_id][0] = heatmap
            running_means[odd_head_id][1] = 1
        else:
            new_count = count + 1
            running_means[odd_head_id][0] = mean + (heatmap - mean) / new_count
            running_means[odd_head_id][1] = new_count

        total_processed += 1

        if total_processed % 100 == 0:
            done = sum(v[1] >= TARGET_HEATMAPS_PER_HEAD for v in running_means.values())
            pbar.set_postfix({
                "heads_done": f"{done}/{len(all_heads)}",
                "processed": total_processed,
                "skipped": skipped_errors
            })

    except Exception as e:
        skipped_errors += 1
        if skipped_errors <= 10:
            print(f"[Warn] idx={idx}: {e}")
        continue

# ===========================================
# === Save per-head averages (streaming mean version)
# ===========================================
def save_head_average(head_id, mean_map, decision, model_type, save_dir):
    np.save(os.path.join(save_dir, f"head_{head_id:03d}_average_{decision}_{model_type}.npy"), mean_map)
    plt.imsave(os.path.join(save_dir, f"head_{head_id:03d}_average_{decision}_{model_type}.png"), mean_map, cmap="seismic")


saved_heads = []
for h in sorted(all_heads):
    mean, count = running_means[h]
    note = ""

    if count >= TARGET_HEATMAPS_PER_HEAD:
        save_head_average(h, mean, DECISION, MODEL_TYPE, SAVE_DIR)
        saved_heads.append(h)
        note = "target_met"
    elif count >= PLAN_B_MIN_HEATMAPS_PER_HEAD:
        save_head_average(h, mean, DECISION, MODEL_TYPE, SAVE_DIR)
        saved_heads.append(h)
        note = f"planB_min={PLAN_B_MIN_HEATMAPS_PER_HEAD}"
    else:
        note = "insufficient_maps"

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([h, count, TARGET_HEATMAPS_PER_HEAD, note])

# Move summary print OUTSIDE the loop
total_heatmaps = sum(v[1] for v in running_means.values())

print(f"\n=== SUMMARY ===")
print(f"Total heatmaps processed: {total_processed}")
print(f"Skipped due to errors: {skipped_errors}")
print(f"Total accumulated heatmaps: {total_heatmaps}")
print(f"Heads saved (>= {PLAN_B_MIN_HEATMAPS_PER_HEAD} maps): {len(saved_heads)}/{len(all_heads)}")
done_full = sum(v[1] >= TARGET_HEATMAPS_PER_HEAD for v in running_means.values())
print(f"Heads meeting full target ({TARGET_HEATMAPS_PER_HEAD}): {done_full}/{len(all_heads)}")
print(f"Averages saved to: {SAVE_DIR}")
