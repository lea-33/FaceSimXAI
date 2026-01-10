import os
import numpy as np
import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonAlpha2Beta1Flat
from facesim3d.modeling.VGG.vgg_face import VGGFace, VGGFaceTruncated
import matplotlib.pyplot as plt
import cv2
from facesim3d import local_paths

def load_image_for_model(image_path: str, dtype: torch.float64, subtract_mean: bool = True) -> torch.Tensor:
    """Load an image for the `VGG` model."""
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, dsize=(224, 224))
    image = torch.Tensor(image).permute(2, 0, 1).view(1, 3, 224, 224).to(dtype).to(device)
    if subtract_mean:
        # this subtraction should be the average pixel value of the training set of the original VGGFace
        image -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).to(dtype).view(1, 3, 1, 1).to(device)
    return image


def target_output_fn(output):
    # Create a tensor of zeros with the same shape as output
    target = torch.zeros_like(output)
    # Set the target class logit to 1
    target[0, pred_class] = 1.0
    return target


# === Load full VGGFace model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Set to " + ("cuda" if torch.cuda.is_available() else "cpu"))


save_dir= local_paths.DIR_VGG_FACE_HEATMAPS

model_path = local_paths.VGG_FACE_PTH
model_full = VGGFace().to(device)
model_full.load_state_dict(torch.load(model_path, map_location=device))
model_full.eval()

#model = VGGFaceTruncated(model_full).to(device)
#model.eval()

# Set up LRP properties
composite = EpsilonAlpha2Beta1Flat()
attributor = Gradient(model_full, composite)

# Head-ID images directory
head_id_images_dir = local_paths.DIR_FRONTAL_VIEW_HEADS 
head_images = sorted([f for f in os.listdir(head_id_images_dir) if f.endswith(".png")])
print(f"Found {len(head_images)} head ID images.")


for i,face in enumerate(head_images):
    path = os.path.join(head_id_images_dir, face)
    input_tensor = load_image_for_model(path, dtype=torch.float32)

    # Choose target class (e.g., the predicted class)
    with torch.no_grad():
        output = model_full(input_tensor)
    pred_class = output.argmax(dim=1).item()

    print(f"Processing {face}, predicted class: {pred_class}")
 
    input_tensor.requires_grad_()

    # compute LRP relevance for the predicted class
    with attributor:
        _, relevance = attributor(input=input_tensor, attr_output=target_output_fn)

    # Extract relevance map and convert to numpy
    relevance_map = relevance.detach().cpu().numpy().squeeze()
    if relevance_map.ndim == 3:
        relevance_map = relevance_map.sum(axis=0)  # sum channels if present

    # Show plot
    plt.figure(figsize=(8, 4))
    plt.title("LRP Relevance Map")
    plt.imshow(relevance_map, cmap="seismic")
    plt.axis("off")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"heatmap_head_{i+1:03d}.png")
    plt.savefig(save_path)
    plt.close()
    
    # also save as .npy
    npy_save_path = save_path.replace(".png", ".npy")
    np.save(npy_save_path, relevance_map)
    
