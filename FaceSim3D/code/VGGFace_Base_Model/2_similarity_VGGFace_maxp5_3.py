"""
Script to compute similarity scores of internal VGGFace representation of all possible Triplets.
Features from the maxpool5_3 layer are used and cosine similarity is computed.
All similarity scores are saved to be used for training the model later on.
"""
# Imports
import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os

from facesim3d import local_paths

data_folder = local_paths.DIR_VGGFACE_MAXP5_3_DATA

# Load fc7 outputs
maxp5_3_outputs = np.load(os.path.join(data_folder, "vggface_maxp5_3_outputs.npy"))  # shape (100, 512, 7, 7)
num_images = maxp5_3_outputs.shape[0]

def compute_triplet_similarities(maxp5_3_outputs: np.ndarray, method: str, num_images: int = 100) -> np.ndarray:
    """
    Compute per-face triplet scores (unique triplets only).

    Args:
        maxp5_3_outputs : np.ndarray
            Face embeddings, shape (N, C, H, W)
        method : str
            - "relative"
            - "centroid"
        num_images : int
            Number of images (N)

    Returns:
        np.ndarray of shape (num_triplets, 6):
            (i, j, k, score_i, score_j, score_k)
    """

    # Flatten conv outputs (N, 512, 7, 7) → (N, 25088)
    embeddings = maxp5_3_outputs.reshape(num_images, -1)
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    results = []

    for (i, j, k) in tqdm(combinations(range(num_images), 3), desc="Processing triplets"):
        # pairwise cosine similarities
        sim_ij = np.dot(embeddings[i], embeddings[j])
        sim_ik = np.dot(embeddings[i], embeddings[k])
        sim_jk = np.dot(embeddings[j], embeddings[k])


        if method == "relative":
            score_i = (sim_ij + sim_ik) / 2 - sim_jk
            score_j = (sim_ij + sim_jk) / 2 - sim_ik
            score_k = (sim_ik + sim_jk) / 2 - sim_ij

            # most dissimilar face index is the one with the lowest score
            choice_idx = np.argmin([score_i, score_j, score_k])

        # Each face’s score is its distance (1 – cosine similarity) from the triplet’s centroid embedding.
        elif method == "centroid":
            centroid = embeddings[i] + embeddings[j] + embeddings[k]
            # normalize centroid
            centroid /= np.linalg.norm(centroid)

            score_i = 1 - np.dot(embeddings[i], centroid)
            score_j = 1 - np.dot(embeddings[j], centroid)
            score_k = 1 - np.dot(embeddings[k], centroid)

            # most dissimilar face index is the one with the highest score
            choice_idx = np.argmax([score_i, score_j, score_k])

        else:
            raise ValueError(f"Unknown method: {method}")



        # append also the index of the most dissimilar face
        results.append({"head1": i, "head2": j, "head3": k, "choice_idx": choice_idx})

    return np.array(results)


# Compute similarities
method = "centroid"  # "relative" or "centroid"
similarity_scores = compute_triplet_similarities(maxp5_3_outputs, method=method, num_images=num_images)
print("Computed similarity scores shape:", similarity_scores.shape)  # (161700, )

# Save similarity scores
np.save(os.path.join(data_folder, f"vggface_maxp5_3_triplet_similarity_scores_method={method}.npy"), similarity_scores, allow_pickle=True)
print(f"Saved similarity scores to vggface_maxp5_3_triplet_similarity_scores_method={method}.npy")



#### PLOTTING ####

# Load your saved similarity scores
method = "centroid"  # "centroid" or "relative"
scores = np.load(os.path.join(data_folder, f"vggface_maxp5_3_triplet_similarity_scores_method={method}.npy"), allow_pickle=True)

# Convert to array of shape (num_triplets, 3)
triplet_scores = np.array([[s["head1"], s["head2"], s["head3"], s["choice_idx"]] for s in scores])
print("Loaded triplets:", triplet_scores.shape)

# Extract number of triplets
num_triplets = triplet_scores.shape[0]

choice_matrix = np.zeros((num_triplets, 3))
for i, s in enumerate(scores):
    choice_matrix[i, s["choice_idx"]] = 1  # mark the odd-one-out

# Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    choice_matrix,
    cmap="coolwarm",
    xticklabels=[0, 1, 2],
    yticklabels=False
)
plt.xlabel("Face in triplet")
plt.ylabel("Triplet index")
plt.title(f"Method: {method}\nTriplet similarity scores from VGGFace maxpool5_3 layer")
plt.tight_layout()
plt.show()



