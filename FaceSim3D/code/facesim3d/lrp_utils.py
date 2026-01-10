"""
Contains core utility functions for loading models and computing LRP heatmaps.
Supports the regional analysis scripts by standardizing the relevance propagation process across different experiments.
"""
# %% Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from facesim3d.modeling.VGG.models import (load_trained_vgg_face_human_judgment_model, VGGFaceHumanjudgmentFrozenCoreWithLegs)
from facesim3d.modeling.VGG.vgg_predict import prepare_data_for_human_judgment_model
from functools import partial
from zennit.composites import EpsilonAlpha2Beta1Flat
from zennit.attribution import Gradient

device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def ensure_array(x):
    """
    Ensure input is a numpy array of type float32.
    """
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = np.array(x.tolist())  # flatten nested objects
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
    return x.astype(np.float32)

class Gradient2(Gradient):
    """
    Gradient class for multi-input models.

    Overwrite the original forward function of the Gradient class to allow for a custom
    attribution function.
    """

    def forward(self, input, attr_output_fn):
        if not isinstance(input, tuple):
            raise TypeError("Use Gradient for single input models.")

        # Ensure inputs require grad
        inputs = tuple(ipt.detach().requires_grad_(True) for ipt in input)
        for i, inp in enumerate(inputs):
            print(f"Input {i} shape:", inp.shape)

        # Forward pass
        output = self.model(*inputs)

        # Compute attribution signal for output
        attr_out = attr_output_fn(output.detach())

        # Repeat attribution for each input
        grad_outputs = (attr_out,) * len(inputs)

        print("Model output:", output.shape)
        print("Attribution target shape:", attr_out.shape)

        # Backward
        gradients = torch.autograd.grad(
            outputs=(output,),
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=self.create_graph,
            retain_graph=self.retain_graph,
        )

        return output, gradients


def feed_attr_output_fn(output, target):
    """
    Feed the output relevance to the model's output for the target label.

    :param output: model output
    :param target: target label
    :return: output times one-hot encoding of the target labels of size (len(target), n_outputs)
    """
    # output times one-hot encoding of the target labels of size (len(target), n_outputs)
    eye = torch.eye(n=output.shape[1], device=device)
    return output * eye[target]


def get_orig_dataset(loader):
    """Unwrap Subset/DataLoader to get back the original dataset."""
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def load_model_for_LRP(method, session, device):
    """
    Load the pretrained VGGFaceHumanjudgmentFrozenCoreWithLegs model for LRP analysis.

    Parameters
    ----------
    method : str
        Similarity method used ("relative" or "centroid").
    session : str
        Session type ("2D" or "3D").

    Returns
    -------
    model : torch.nn.Module
        The loaded model ready for LRP.
    model_name : str
        Name of the loaded model.

    """

    # Load pretrained FrozenCore model
    if method == "relative":
        model_name_vgg = "2025-10-09_17-35_VGGFaceHumanjudgmentFrozenCore_maxp5_3_SIM_method-relative_final.pth"
    else:
        model_name_vgg = "2025-10-09_12-04_VGGFaceHumanjudgmentFrozenCore_maxp5_3_SIM_method-centroid_final.pth"


    model_vgg = load_trained_vgg_face_human_judgment_model(
        session=session,
        model_name=model_name_vgg,
        exclusive_gender_trials=None,
        device=device,
        method=method,
    )

    model = VGGFaceHumanjudgmentFrozenCoreWithLegs(frozen_top_model=model_vgg)
    model.name = model_name = model_name_vgg + "WithLegs"
    model.eval()

    # Disable requires_grad for all parameters, we do not need their modified gradients
    for param in model.parameters():
        param.requires_grad = False

    return model, model_name


def compute_triplet_heatmap(i, j, k, model, composite, dataset, device, visualize=False):
    """
    Compute the LRP relevance heatmap for the 'odd-one-out' head (i) in triplet (i, j, k).

    Parameters
    ----------
    i, j, k : int
        Indices of the heads forming the triplet.
    model : torch.nn.Module
        The trained VGGFaceHumanjudgmentFrozenCoreWithLegs model.
    composite : zennit.Composite
        Zennit composite for LRP relevance computation.
    dataset : VGGFaceMaxp5_3_Dataset
        Dataset object containing triplets and their computed odd one out choice (based on embedding similarity).
    device : str
        Device to use ("cpu" or "cuda").
    visualize : bool
        If True, shows the resulting 2D heatmap for debugging.

    Returns
    -------
    np.ndarray or None
        2D numpy array of shape (224, 224) representing the heatmap for the odd-one-out image
    """

    model.eval()

    # --- Find a matching triplet from the dataset ---
    matching_triplet = None
    for idx, sample in enumerate(dataset.triplets):
        h1, h2, h3 = map(int, (sample["head1"], sample["head2"], sample["head3"]))
        if set((i, j, k)) == set((h1, h2, h3)):
            matching_triplet = dataset[idx]
            y_loc = int(sample["choice_idx"])
            odd_head_ID = [h1, h2, h3][y_loc]
            break

    if matching_triplet is None:
        print(f"No matching triplet found for ({i},{j},{k}). Skipping.")
        return "no_match"

    if odd_head_ID != i:
        print(f"Skipping triplet ({i},{j},{k}) — head {i} not odd one out (odd={odd_head_ID}).")
        return "not_ooo"

    x1, x2, x3, y = (
        matching_triplet["image1"].to(device).unsqueeze(0),
        matching_triplet["image2"].to(device).unsqueeze(0),
        matching_triplet["image3"].to(device).unsqueeze(0),
        matching_triplet["choice"].to(device),
    )

    inputs = (x1, x2, x3)

    # --- Check that i is the odd one out ---
    if [i, j, k][y.item()] != i:
        print(f"Skipping triplet ({i},{j},{k}) — head {i} not odd one out (y={y}).")
        return None

    # --- Compute relevance using LRP ---
    attributor = Gradient2(model, composite)
    output_relevance = partial(feed_attr_output_fn, target=y.item())

    with attributor:
        model_output, relevance = attributor(input=inputs, attr_output=output_relevance)

    # --- Extract relevance for the odd one ---
    relevance_odd = relevance[y.item()].detach().cpu().numpy().squeeze()

    if relevance_odd.ndim == 3:
        relevance_odd = relevance_odd.sum(axis=0)

    if visualize:
        plt.figure(figsize=(4, 4))
        plt.imshow(relevance_odd, cmap="bwr", vmin=-0.05, vmax=0.05)
        plt.title(f"Head {i} odd-one-out | Triplet ({i},{j},{k})")
        plt.axis("off")
        plt.show()

    return relevance_odd

def compute_triplet_heatmap_hj(i, j, k, model, composite, dataset, device, visualize=False):
    """
    Compute the LRP relevance heatmap for the 'odd-one-out' head (i) in triplet (i, j, k).

    Parameters
    ----------
    i, j, k : int
        Indices of the heads forming the triplet.
    model : torch.nn.Module
        The trained VGGFaceHumanjudgmentFrozenCoreWithLegs model.
    composite : zennit.Composite
        Zennit composite for LRP relevance computation.
    dataset : VGGFaceDataset
        Dataset object containing frozen-core embeddings (vgg_core_output).
    device : str
        Device to use ("cpu" or "cuda").
    visualize : bool
        If True, shows the resulting 2D heatmap for debugging.

    Returns
    -------
    np.ndarray or None
        2D numpy array of shape (224, 224) representing the heatmap for the odd-one-out image
    """

    # --- Find a matching triplet from the dataset ---
    matching_triplet = None

    for idx, sample in enumerate(dataset.triplets):
        h1, h2, h3 = map(int, (sample["head1"], sample["head2"], sample["head3"]))
        if set((i, j, k)) == set((h1, h2, h3)):
            matching_triplet = dataset[idx]
            y_loc = int(sample["choice_idx"])
            odd_head_ID = [h1, h2, h3][y_loc]
            break

    if matching_triplet is None:
        print(f"No matching triplet found for ({i},{j},{k}). Skipping.")
        return "no_match"

    if odd_head_ID != i:
        print(f"Skipping triplet ({i},{j},{k}) — head {i} not odd one out (odd={odd_head_ID}).")
        return "not_ooo"

    x1, x2, x3, y = (
        matching_triplet["image1"].to(device).unsqueeze(0),
        matching_triplet["image2"].to(device).unsqueeze(0),
        matching_triplet["image3"].to(device).unsqueeze(0),
        matching_triplet["choice"].to(device),
    )

    inputs = (x1, x2, x3)

    # --- Compute relevance using LRP ---
    attributor = Gradient2(model, composite)
    output_relevance = partial(feed_attr_output_fn, target=y.item())

    with attributor:
        model_output, relevance = attributor(input=inputs, attr_output=output_relevance)

    # --- Extract relevance for the odd one ---
    relevance_odd = relevance[y.item()].detach().cpu().numpy().squeeze()

    if relevance_odd.ndim == 3:
        relevance_odd = relevance_odd.sum(axis=0)

    if visualize:
        plt.figure(figsize=(4, 4))
        plt.imshow(relevance_odd, cmap="bwr", vmin=-0.05, vmax=0.05)
        plt.title(f"Head {i} odd-one-out | Triplet ({i},{j},{k})")
        plt.axis("off")
        plt.show()

    return relevance_odd
