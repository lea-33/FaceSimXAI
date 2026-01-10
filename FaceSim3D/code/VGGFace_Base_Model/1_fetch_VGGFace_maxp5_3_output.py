"""
Computes and saves the activations of the `fc7` layer (maxp5_3) from the VGG-Face model for a set of input images.
These feature embeddings serve as the basis for the MaxP similarity model calculations.
"""

# Imports
import cv2
import torch
from facesim3d.modeling.VGG.vgg_face import VGGFace
import os
import numpy as np
from tqdm import tqdm
from facesim3d import local_paths

results_folder = local_paths.DIR_VGGFACE_MAXP5_3_DATA
os.makedirs(results_folder, exist_ok=True)

# Set up model
# === Load full VGGFace model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Set to " + ("cuda" if torch.cuda.is_available() else "cpu"))

model_path = local_paths.VGG_FACE_PTH
model = VGGFace(save_layer_output=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Access index of the max5_3 layer
for idx, layer_name in enumerate(model.layer_names):
    if "maxp_5_3" in layer_name:
        maxp5_3_index = idx
        print(f"Index of maxpool5_3 layer: {idx}")
        break


# Define Input Images
image_folder = local_paths.DIR_FRONTAL_VIEW_HEADS
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

def load_image_for_model(image_path: str, dtype: torch.float64, subtract_mean: bool = True) -> torch.Tensor:
    """Load an image for the `VGG` model."""
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, dsize=(224, 224))
    image = torch.Tensor(image).permute(2, 0, 1).view(1, 3, 224, 224).to(dtype).to(device)
    if subtract_mean:
        # this subtraction should be the average pixel value of the training set of the original VGGFace
        image -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).to(dtype).view(1, 3, 1, 1).to(device)
    return image


# Write function to fetch the fc7 layer output
def get_maxp5_3_output(image_tensor: torch.Tensor) -> torch.Tensor:
    """Get the maxp5_3 layer output for a given image tensor."""
    model.reset_layer_output()
    _ = model(image_tensor)
    output = model.layer_output[maxp5_3_index]
    return output


outputs = []

# Test for the first image
for img_file in tqdm(image_files):
    img_path = os.path.join(image_folder, img_file)
    img_tensor = load_image_for_model(img_path, dtype=torch.float32)
    maxp5_3_output = get_maxp5_3_output(img_tensor)
    # Append output as shape (1, 4096)
    outputs.append(maxp5_3_output.detach().numpy())

print("Shape of outputs:", np.array(outputs).shape)  # Shape (100, 1, 512, 7, 7)

# Save the outputs as a numpy array
outputs_array = np.array(outputs).squeeze()  # Shape (100, 512, 7, 7)
print("Shape of saved outputs_array:", outputs_array.shape)
np.save(os.path.join(results_folder, "vggface_maxp5_3_outputs.npy"), outputs_array)

