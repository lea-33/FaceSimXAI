"""
Quantifies relevance scores for specific facial regions (e.g., eyes, nose, mouth) using MediaPipe landmarks on LRP heatmaps from the VGG-MaxP model.
Enables a quantitative regional analysis of the model's decision-making strategy.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import defaultdict
import csv
from tqdm import tqdm
from facesim3d import local_paths

# ==========================
# CONFIGURATION
# ==========================
images_dir = local_paths.DIR_FRONTAL_VIEW_HEADS
heatmaps_dir = local_paths.DIR_AVERAGE_HEATMAPS_MAXP
results_dir = local_paths.DIR_REGION_ANALYSIS_RESULTS
csv_path = os.path.join(results_dir, "landmark_annotations.csv")
output_csv = os.path.join(results_dir, "relevance_per_region_dataset_maxpSim.csv")


# ==========================
# LOAD REGION MAP (region → indices)
# ==========================
region_map = defaultdict(list)
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["landmark_index"])
        region = row["region"].strip()
        region_map[region].append(idx)

# ==========================
# INITIALIZE MEDIAPIPE
# ==========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# ==========================
# REGION RELEVANCE FUNCTION
# ==========================
def get_region_sum(gray_image, points, region_indices):
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    region_points = points[region_indices]
    hull = cv2.convexHull(region_points)
    cv2.fillConvexPoly(mask, hull, 255)
    masked = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    return int(np.sum(masked))

# ==========================
# MAIN PROCESSING LOOP
# ==========================
records = []

for idx,filename in tqdm(enumerate(sorted(os.listdir(images_dir)))):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(images_dir, filename)
    heatmap_path = os.path.join(heatmaps_dir, f"head_{idx-1:03d}_average_odd_one_out_VGGFace_Maxp5_3_Sim.npy")
    if not os.path.exists(heatmap_path):
        print(f"Skipping {filename} — no matching heatmap found.")
        continue

    # --- Load image and get landmarks ---
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("No face detected.")
    
    h, w = gray.shape
    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

    # --- Load LRP heatmap (grayscale intensity) ---
    import numpy as np
    # --- Load LRP heatmap (numeric values) ---
    heatmap = np.load(heatmap_path)
    
    if heatmap.ndim == 3:
        # Convert to grayscale if it's RGB-like
        gray_lrp = cv2.cvtColor(heatmap.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray_lrp = heatmap.astype(np.float32)

    # --- Compute per-region relevance sums and pixel counts ---
    region_sums_lrp = {}
    region_pixel_counts = {}
    
    for name, idxs in region_map.items():
        mask = np.zeros(gray_lrp.shape[:2], dtype=np.uint8)
        region_points = points[idxs]
        if len(region_points) < 3:   # 3 points needed to span a hull
            region_sums_lrp[name] = 0.0
            region_pixel_counts[name] = 1.0  # avoid division by zero
            continue
        hull = cv2.convexHull(region_points)
        cv2.fillConvexPoly(mask, hull, 255)
    
        region_pixel_counts[name] = np.count_nonzero(mask)
        masked = gray_lrp * (mask / 255.0)
        region_sums_lrp[name] = float(np.sum(masked))
    
    # --- Remove contour before normalization ---
    region_sums_lrp.pop("contour", None)
    region_pixel_counts.pop("contour", None)
    
    # --- Normalize per region by pixel count ---
    region_sums_norm = {}
    for r in region_sums_lrp:
        count = region_pixel_counts[r]
        if count == 0:
            region_sums_norm[r] = 0.0
        else:
            region_sums_norm[r] = region_sums_lrp[r] / count

    # --- Store result for this image ---
    region_sums_norm["image_name"] = filename
    records.append(region_sums_norm)

# ==========================
# SAVE RESULTS
# ==========================
df = pd.DataFrame(records)
df = df.fillna(0.0)
df.to_csv(output_csv, index=False)

print(f"\nDone! Saved normalized region relevance scores for {len(df)} images → {output_csv}")
print(df.head())
