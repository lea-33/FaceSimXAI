"""
This script executes the full LLM pipeline, prompting the model to identify the odd-one-out face and explicitly state the facial regions driving its decision.
It bridges the cross-model comparison with regional analysis by extracting qualitative explanations from the LLM, which can be compared to LRP heatmaps.
"""

import json
import os.path
from openai import OpenAI
import matplotlib.pyplot as plt
from facesim3d.modeling.VGG.vgg_predict import prepare_data_for_human_judgment_model
import torch
import io
import base64
from PIL import Image
import random
from facesim3d.modeling.VGG.models import load_trained_vgg_face_human_judgment_model, VGGFaceHumanjudgmentFrozenCoreWithLegs
from tqdm import tqdm
from facesim3d import local_paths


# ==== DEFINITIONS ====
def tensor_to_base64_image(tensor: torch.Tensor) -> str:
    """Convert a torch image tensor (C, H, W) or (H, W, C) into base64 PNG string."""
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # (C, H, W)
        tensor = tensor.permute(1, 2, 0)  # -> (H, W, C)
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype("uint8")
    img = Image.fromarray(tensor)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def show_triplet(img1, img2, img3, title=None, face_ids=None):
    """
    Display 3 head images side by side.
    Works with torch tensors (C,H,W), numpy arrays (H,W,C), or PIL images.
    """
    imgs = [img1, img2, img3]
    n = len(imgs)

    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if title:
        fig.suptitle(title, fontsize=14)

    for i, (ax, img) in enumerate(zip(axs, imgs)):
        # Convert torch tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()
        # Normalize to 0–1 range
        if img.max() > 1.0:
            img = img / 255.0

        ax.imshow(img)
        label = f"Head {face_ids[i]}" if face_ids else f"Image {i}"
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def get_orig_dataset(loader):
    """Unwrap Subset/DataLoader to get back the original dataset."""
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds

# ==== SET UP =====
my_api_key = "your-api-key-here"   # Replace with your API key
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
method = "relative"  # or "centroid"

save_dir = local_paths.DIR_LLM_ANALYSIS_RESULTS
attempt_limit = 3
attempt_delay = 1  # seconds

# OpenAI client
openai_client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=my_api_key)
for model in openai_client.models.list().data:
    llm_model_name = model.id
    if "Qwen/Qwen2" in llm_model_name:
        print(llm_model_name)

# ==== FACE REGIONS =====
REGIONS = [
    "mouth", "nose_tip", "nose_bridge", "left_eye", "right_eye",
    "left_eyebrow", "right_eyebrow", "forehead", "left_cheek",
    "right_cheek", "chin", "space between eyes"
]

system_prompt = "You are a skilled AI assistant that is asked to participate in a study about face similarity."

final_prompt = (
    "These are three 3D-reconstructed human faces (A, B, C)."
    "You have 2 tasks:"
    "1) First, choose the face that appears most dissimilar to the other two."
    "In this study we are interested in the similarity between faces."
    "During the experiment you will see a variety of faces. They come in a set of 3."
    "In each round we ask you to choose the one face which is the most dissimilar to the other two faces."
    "We are interested in how you perceive the similarity of faces."
    "Caution: In some trials two faces are identical, here you must choose the respective third, dissimilar face."
    "Your task is to decide which face is most dissimilar to the other two."
    "Do not rush, but also do not overthink: Decide intuitively."
    "2) Second, choose up to 3 facial regions IN THE DISSIMILAR FACE that make it most different."
    f"Here are the possible facial regions you can choose from: {', '.join(REGIONS)}."
    "Choose those regions that are most relevant for choosing the dissimilar face."
    "Answer in VALID JSON format:\n"
    "{\n"
    '  "odd_one_out": index,\n'
    '  "regions": ["x", "y", "z"]\n'
    "}\n"
    "index should be the POSITION of the most DISSIMILAR head of the triplet (use zero-indexing)."
    "only possible answers you can give are the numbers: 0, 1, 2"
    "regions should be a list of up to 3 chosen facial regions from the provided list."
    f"only possible region names: {', '.join(REGIONS)}"
    "ONLY output the valid json format. No explanations, no other words or anything."

)



# ==== LOAD DATASET =====
# Initialize the dataloader
train_dl, val_dl, test_dl = prepare_data_for_human_judgment_model(
    session="3D",
    frozen_core=False,
    last_core_layer="fc7-relu",
    data_mode="3d-reconstructions",
    split_ratio=(0.98, 0.01, 0.01),
    batch_size=1,
    shuffle=True,
    dtype=torch.float32,
    size=None,
)
dataset = get_orig_dataset(test_dl)

# create list of random indices from the dataset
num_samples = 10000
index_list = random.sample(range(len(dataset)), num_samples)

# ==== INIT MODEL FOR HUMAN JUDGEMENTS ====
hj_model_name = "2023-12-11_19-46_VGGFaceHumanjudgmentFrozenCore"    # this has the overall best TEST ACCURACY


hj_model = load_trained_vgg_face_human_judgment_model(
    session="3D",
    model_name=hj_model_name,
    exclusive_gender_trials=None,
    device=str(device),
)
hj_model_legs = VGGFaceHumanjudgmentFrozenCoreWithLegs(frozen_top_model=hj_model)

# ==== INIT MODEL FOR MAXP5_3 SIMILARITY ====
if method == "relative":
    max_model_name = "2025-10-09_17-35_VGGFaceHumanjudgmentFrozenCore_maxp5_3_SIM_method-relative_final.pth"
else:
    max_model_name = "2025-10-09_12-04_VGGFaceHumanjudgmentFrozenCore_maxp5_3_SIM_method-centroid_final.pth"

max_model = load_trained_vgg_face_human_judgment_model(
        session="3D",
        model_name=max_model_name,
        exclusive_gender_trials=None,
        device=str(device),
        method=method,
    )
max_model_legs = VGGFaceHumanjudgmentFrozenCoreWithLegs(frozen_top_model=max_model)

# prepare models for inference
hj_model_legs.to(device)
hj_model_legs.eval()
max_model_legs.to(device)
max_model_legs.eval()

# ==== INIT CSV LOG FILE =====
with open(os.path.join(save_dir, f"llm_pipeline_#{num_samples}.csv"), mode="w") as f:
    f.write("index,ground_truth,llm_answer,llm_regions,human_judgement,maxp_sim,llm_prediction_id\n")

accuracy = 0

for idx in tqdm(index_list, desc="Evaluating samples"):
    # sample from the dataset for testing
    sample = dataset[idx]

    img1, img2, img3 = sample["image1"], sample["image2"], sample["image3"]

    # Get their head IDs from session_data
    h1, h2, h3, _ = dataset.session_data.iloc[idx].to_numpy()
    heads = [h1, h2, h3]
    # Determine ground truth answer
    gt = sample["choice"].item()

    #show_triplet(img1, img2, img3, title=f"Triplet #{idx}", face_ids=[h1, h2, h3])

    # Convert images to base64 strings
    img1_b64 = tensor_to_base64_image(sample["image1"])
    img2_b64 = tensor_to_base64_image(sample["image2"])
    img3_b64 = tensor_to_base64_image(sample["image3"])

    message = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img3_b64}"}},
            ],
        },
    ]

    for attempt in range(attempt_limit):

        # Send request
        response = openai_client.chat.completions.create(
            model=llm_model_name,
            messages=message
        )
        raw_response = response.choices[0].message.content.strip()

        try:
            llm_json = json.loads(raw_response)
            llm_answer = str(llm_json.get("odd_one_out", ""))
            llm_regions = llm_json.get("regions", [])
            llm_prediction_id = heads[int(llm_answer)] if llm_answer in ["0","1","2"] else "NA"

        except:
            llm_answer, llm_regions = "", []

        if llm_answer in ["0", "1", "2"]:
            break
        elif attempt == attempt_limit- 1:
            print(f"Skipping sample {idx}: Invalid LLM response after {attempt_limit} tries.")
            llm_answer, llm_regions = "NA", []


    if llm_answer == str(gt):
        accuracy += 1

    inputs = (sample["image1"].unsqueeze(0).to(device),
              sample["image2"].unsqueeze(0).to(device),
              sample["image3"].unsqueeze(0).to(device),)

    # Evaluate Human Judgment Model
    with torch.no_grad():
        outputs_hj = hj_model_legs(*inputs)
        _, predicted_hj = torch.max(outputs_hj, 1)
        hj_result = predicted_hj.item()

    with torch.no_grad():
        outputs_max = max_model_legs(*inputs)
        _, predicted_max = torch.max(outputs_max, 1)
        maxp_sim_result = predicted_max.item()

    with open(os.path.join(save_dir, f"llm_pipeline_#{num_samples}.csv"), mode="a") as f:
        f.write(f"{idx},{gt},{llm_answer},\"{';'.join(llm_regions)}\",{hj_result},{maxp_sim_result},{llm_prediction_id}\n")


print(f"Final Accuracy over {len(index_list)} samples: {accuracy}/{len(index_list)} = {accuracy/len(index_list)*100:.2f}%")

# end
print("Results saved to", f"llm_pipeline_#{num_samples}.csv")
