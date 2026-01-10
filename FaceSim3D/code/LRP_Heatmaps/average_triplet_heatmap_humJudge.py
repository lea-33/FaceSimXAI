"""
Heatmap averaging script for the VGG-Hum model.
Computes aggregated relevance maps for triplet decisions to identify consistent facial features driving model performance.
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

MODEL_TYPE = "VGGFaceHumanJudgment"
DECISION = "odd_one_out"

from facesim3d import local_paths

SAVE_DIR = local_paths.DIR_AVERAGE_HEATMAPS_HUM
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, f"run_log_efficient_average_triplet_heatmaps_{MODEL_TYPE}.csv")

DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===========================================
# === Model, Data, Composite
# ===========================================
if SESSION == "2D":
    model_name_vgg = "2023-11-15_04-44_VGGFaceHumanjudgmentFrozenCore"
else: #3D
    model_name_vgg = "2023-12-11_19-46_VGGFaceHumanjudgmentFrozenCore"    # this has the overall best TEST ACCURACY


model_vgg = load_trained_vgg_face_human_judgment_model(
    session=SESSION,
    model_name=model_name_vgg,
    exclusive_gender_trials=None,
    device=DEVICE,
)
model = VGGFaceHumanjudgmentFrozenCoreWithLegs(frozen_top_model=model_vgg)
model.name = model_name = model_name_vgg + "WithLegs"

model.eval()
for param in model.parameters():
    param.requires_grad = False
model.to(DEVICE)
print(f"Loaded pretrained model: {model_name}")

# Data: a single big split is fine (we just need access to the dataset)
train_dl, val_dl, test_dl = prepare_data_for_human_judgment_model(
    session=SESSION,
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
# === Helper: compute heatmap for a dataset index
# ===========================================
def compute_heatmap_for_dataset_index(idx, model, composite, dataset, device):
    """
    Compute LRP heatmap for the odd-one-out of the triplet at dataset index `idx`.
    Returns (odd_head_id, heatmap) or (None, None) on failure.
    """
    # Pull images & choice
    sample = dataset[idx]
    x1 = sample["image1"].to(device).unsqueeze(0)
    x2 = sample["image2"].to(device).unsqueeze(0)
    x3 = sample["image3"].to(device).unsqueeze(0)
    y = sample["choice"].to(device)  # int tensor in {0,1,2}
    inputs = (x1, x2, x3)

    # figure out odd head id from dataset.triplets (same order as session_data)
    t = dataset.triplets[idx]
    h1, h2, h3, choice_idx = int(t["head1"]), int(t["head2"]), int(t["head3"]), int(t["choice_idx"])
    odd_head_id = [h1, h2, h3][choice_idx]
    print(f"Processing dataset idx={idx}: triplet=({h1},{h2},{h3}), odd_head={odd_head_id}")

    # LRP: attribution towards the chosen class
    attributor = Gradient2(model, composite)
    output_relevance = partial(feed_attr_output_fn, target=y.item())

    with attributor:
        model_output, relevance = attributor(input=inputs, attr_output=output_relevance)

    # Extract relevance for the odd one
    rel = relevance[y.item()].detach().cpu().numpy().squeeze()
    if rel.ndim == 3:
        rel = rel.sum(axis=0)  # sum channels if present

    return odd_head_id, rel

# ===========================================
# === Main loop: iterate only over VALID triplets
# ===========================================
total_processed = 0
skipped_errors = 0

# CSV header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["head_id", "collected_heatmaps", "target", "note"])

# We may need multiple passes if some heads lag behind (rare, but Plan B handles exhaustion)
pbar = tqdm(shuffled_indices, desc="Processing valid study triplets")
for idx in pbar:
    # Quick exit if everyone met target
    if all(len(maps) >= TARGET_HEATMAPS_PER_HEAD for maps in head_heatmaps.values()):
        pbar.set_description("Reached target for all heads ✅")
        break

    # Peek the odd head for this triplet; skip if already at target to save compute
    t = dataset.triplets[idx]
    h1, h2, h3, choice_idx = int(t["head1"]), int(t["head2"]), int(t["head3"]), int(t["choice_idx"])
    odd_head = [h1, h2, h3][choice_idx]
    if len(head_heatmaps[odd_head]) >= TARGET_HEATMAPS_PER_HEAD:
        continue

    try:
        odd_head_id, heatmap = compute_heatmap_for_dataset_index(idx, model, composite, dataset, DEVICE)
        if odd_head_id is None or heatmap is None:
            skipped_errors += 1
            continue

        head_heatmaps[odd_head_id].append(heatmap)
        total_processed += 1

        # progress text
        if total_processed % 50 == 0:
            done = sum(len(v) >= TARGET_HEATMAPS_PER_HEAD for v in head_heatmaps.values())
            pbar.set_postfix({"heads_done": f"{done}/{len(all_heads)}", "processed": total_processed})

    except Exception as e:
        skipped_errors += 1
        # Optional: log errors sparsely to avoid spam
        if skipped_errors <= 5:
            print(f"[Warn] error at idx={idx}: {e}")
        continue

# ===========================================
# === Save per-head averages (with Plan B)
# ===========================================
def save_head_average(head_id, maps, decision, model_type, save_dir):
    avg_map = np.mean(np.stack(maps, axis=0), axis=0)
    np.save(os.path.join(save_dir, f"head_{head_id:03d}_average_{decision}_{model_type}.npy"), avg_map)
    plt.imsave(os.path.join(save_dir, f"head_{head_id:03d}_average_{decision}_{model_type}.png"), avg_map, cmap="seismic")
    return avg_map

saved_heads = []
for h in sorted(all_heads):
    n_maps = len(head_heatmaps[h])
    note = ""
    if n_maps >= TARGET_HEATMAPS_PER_HEAD:
        save_head_average(h, head_heatmaps[h], DECISION, MODEL_TYPE, SAVE_DIR)
        saved_heads.append(h)
        note = "target_met"
    elif n_maps >= PLAN_B_MIN_HEATMAPS_PER_HEAD:
        # Plan B: still save if enough signal
        save_head_average(h, head_heatmaps[h], DECISION, MODEL_TYPE, SAVE_DIR)
        saved_heads.append(h)
        note = f"planB_min={PLAN_B_MIN_HEATMAPS_PER_HEAD}"
    else:
        note = "insufficient_maps"

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([h, n_maps, TARGET_HEATMAPS_PER_HEAD, note])

# ===========================================
# === Overall average
# ===========================================
head_files = [
    os.path.join(SAVE_DIR, f"head_{h:03d}_average_{DECISION}_{MODEL_TYPE}.npy")
    for h in sorted(all_heads)
    if os.path.exists(os.path.join(SAVE_DIR, f"head_{h:03d}_average_{DECISION}_{MODEL_TYPE}.npy"))
]

if len(head_files) == 0:
    print("No per-head averages were saved. Exiting without overall average.")
else:
    all_heatmaps = [np.load(f) for f in head_files]
    overall_average = np.mean(np.stack(all_heatmaps, axis=0), axis=0)
    np.save(os.path.join(SAVE_DIR, f"overall_average_{DECISION}_{MODEL_TYPE}.npy"), overall_average)
    plt.imsave(os.path.join(SAVE_DIR, f"overall_average_{DECISION}_{MODEL_TYPE}.png"), overall_average, cmap="seismic")

    # Final summary prints
    print("\n=== SUMMARY ===")
    print(f"Total valid triplets processed: {total_processed}")
    print(f"Skipped due to errors: {skipped_errors}")
    print(f"Heads saved (>= {PLAN_B_MIN_HEATMAPS_PER_HEAD} maps): {len(saved_heads)}/{len(all_heads)}")
    done_full = sum(len(v) >= TARGET_HEATMAPS_PER_HEAD for v in head_heatmaps.values())
    print(f"Heads meeting full target ({TARGET_HEATMAPS_PER_HEAD}): {done_full}/{len(all_heads)}")
    print(f"Averages saved to: {SAVE_DIR}")
