"""
Analyzes the distribution of 'Odd-One-Out' selections across the dataset to understand potential biases or frequencies in the model's choices..
"""

# %% check_ooo_distribution.py
import torch
import pandas as pd
from facesim3d.modeling.VGG.vgg_predict import prepare_data_for_human_judgment_model
from facesim3d.modeling.VGG.prepare_data import prepare_data_for_maxp5_3_similarity_model
from facesim3d.lrp_utils import get_orig_dataset

#FLAG
HUM_JUDGE = True     # else is maxp5_3

# -----------------------------
# Configuration
# -----------------------------
SESSION = "3D"             # or "2D"
DATA_MODE = "3d-reconstructions"
LAST_CORE_LAYER = "fc7-relu"
BATCH_SIZE = 1
NUM_WORKERS = 0
DTYPE = torch.float32
SHUFFLE = True
DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# -----------------------------
# Load dataset (no model needed)
# -----------------------------
if HUM_JUDGE:
    train_dl, val_dl, test_dl = prepare_data_for_human_judgment_model(
        session=SESSION,
        frozen_core=False,
        last_core_layer=LAST_CORE_LAYER,
        data_mode=DATA_MODE,
        split_ratio=(0.98, 0.01, 0.01),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        dtype=DTYPE,
        size=None,
    )
else:
    METHOD = "relative"
    train_dl, val_dl, test_dl, set_lengths = prepare_data_for_maxp5_3_similarity_model(
        session=SESSION,
        method=METHOD,
        frozen_core=False,
        last_core_layer="fc7-relu",
        data_mode="3d-reconstructions",
        split_ratio=(0.98, 0.01, 0.01),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        dtype=DTYPE,
        size=None,
    )
dataset = get_orig_dataset(test_dl)

# -----------------------------
# Compute OOO counts per head
# -----------------------------
if HUM_JUDGE:
    counts = dataset.session_data['head_odd'].value_counts().sort_index()
    print("\n=== Odd-One-Out distribution across heads for Human Judgement Model===")
else:
    df = dataset.session_data.copy()
    if df["head_odd"].max() <= 2:  # detect that it's 0,1,2 instead of real IDs
        df["head_odd"] = df.apply(lambda r: [r["head1"], r["head2"], r["head3"]][r["head_odd"]], axis=1)
    counts = df["head_odd"].value_counts().sort_index()

    print("\n=== Odd-One-Out distribution across heads for Maxp5_3 Similarity Model===")
print(counts.describe())
print(f"Min OOO count: {counts.min()} | Max OOO count: {counts.max()}")
print(f"Total triplets: {len(dataset)} | Unique heads: {counts.shape[0]}")

# plot distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
counts.plot(kind='bar')
plt.title('Odd-One-Out Distribution Across Heads (Human Judgement Model)'if HUM_JUDGE else 'Odd-One-Out Distribution Across Heads (Maxp5_3 Sim Model)')
plt.xlabel('Head ID')
plt.ylabel('Number of Times as Odd-One-Out')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
