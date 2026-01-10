# Explainable AI methods for human-aligned face perception

**Master’s Thesis — Lea Gihlein, University of Leipzig, 2026**

*Investigating the underlying decision-making processes of a human-aligned face similarity model using Explainable AI (XAI).*

![Last update](https://img.shields.io/badge/last_update-Jan_10,_2026-green)
![version](https://img.shields.io/badge/version-v.1-blue)
[![Original Project](https://img.shields.io/badge/Based_on-FaceSim3D-violet)](https://github.com/SHEscher/FaceSim3D)

---

## Project Description

This project, **Explainable AI methods for human-aligned face perception**, is a Master's thesis built as an extension of the **[FaceSim3D](https://github.com/SHEscher/FaceSim3D)** project by Hofmann et al. (2024).

While the original work focused on testing the effect of space and time on face similarity judgments and training human-aligned encoding models, this thesis applies **Explainable AI (XAI)** methods - specifically **Layer-wise Relevance Propagation (LRP)** - to investigate *why* these models make specific similarity decisions. We analyze the models' reliance on facial regions, the structure of their internal representations, and compare their reasoning with Vision Language Models (VLMs).

---

## Models

This work investigates two variations of VGG-based models.
The base model is the **VGG-Face** model (Parkhi et al., 2015), a deep convolutional network originally trained for facial recognition. It serves as the backbone for the other models used in this study.

### 1. VGG Human Judgment Model (VGG-Hum)
Developed by Hofmann et al. (2024), the **VGG-Hum** is an extension of the VGG-Face model fine-tuned on human behavioral data. It determines the most dissimilar face in a triplet (Odd-One-Out task) based on human choices collected in a dynamic 3D viewing condition.

### 2. VGG Computational Similarity Model (VGG-MaxP)
A new model introduced in this thesis, the **VGG-MaxP**, shares the identical architecture with VGG-Hum but was trained solely on computational similarity metrics (derived from the VGG-Face `maxpooling5-3` layer) rather than human judgments. This allows for a controlled comparison between human-aligned and purely computational decision strategies.

---

## Main Analyses

The core of this thesis fits into three main storylines:

### 1. Regional Analysis
We employ **Layer-wise Relevance Propagation (LRP)** (Bach et al., 2015) to generate relevance heatmaps for the models' decisions.
*   **Visual Explanation**: By projecting facial landmarks (using MediaPipe; Lugaresi et al., 2019) onto the heatmaps, we quantify which specific facial regions (e.g., eyes, nose, mouth) contribute most to the decision of which face is the "odd one out."
*   **Comparison**: Evaluating if VGG-Hum focuses on different features compared to the unaligned VGG-MaxP.

### 2. Structural Analysis
We investigate the latent structure of the relevance maps to see if systematic clusters emerge from the model's internal representation of facial similarity.
*   **PCA**: Applied to flattened heatmaps to identify principal components explaining the variance in relevance distribution.
*   **Autoencoder**: A convolutional autoencoder compresses heatmaps into a lower-dimensional latent space. We analyze this space using techniques like UMAP and t-SNE to visualize how the model organizes faces based on relevance patterns.

### 3. Cross-Model Comparison
To further evaluate the interpretability and validity of the VGG models' reasoning, we compare their results against a state-of-the-art **Vision Language Model (VLM) (Qwen2-VL-72B-Instruct)** (Wang et al., 2024).
*   **Accuracy**: Assessing if the VLM can perform the same triplet odd-one-out task.
*   **Agreement**: Comparing the VLM's selected "important regions" (extracted via prompting) with the LRP-derived important regions of the VGG models.
*   **Decoding Heatmaps**: Using the VLM to decode the heatmaps and compare the results with the LRP-derived important regions.

---

## Environment Setup

This project uses multiple Python environments.

By default, scripts should be run using `face_sim_env`.  
Some groups of scripts require alternative environments due to incompatible dependencies. 

Detailed setup instructions and environment specifications can be found in  
[`environments/README.md`](environments/README.md).

For the code to run, locate to the `FaceSimXAI/FaceSim3D` directory and run the scripts from there.
```shell
cd FaceSimXAI/FaceSim3D/
python code/example-folder/example-script.py
```

---

## Data Accessibility

The data used in this project, as well as all resulting files, can be found on Zenodo with the DOI `10.5281/zenodo.18200352` and should be downloaded and unzipped in the `FaceSim3D/data` and `FaceSim3D/results` directory respectively. 

However, the single heatmaps (created in scripts `Save_single_LRP_Heatmaps_HJ.py` and `Save_single_LRP_Heatmaps_MaxP.py`) needed to run some of the scripts are not included in the Zenodo repository due to their large size. They can be retrieved from additional Zenodo records with DOIs `10.5281/zenodo.18152382` for the VGG-Hum model and `10.5281/zenodo.18153823` for the VGG-MaxP model.

Some data used can not be published because it originated from the Chicago Face Database (CFD) which can not be redistributed. After downloading the data, those folders are missing the actual data due to this restriction:
  - `FaceSim3D/data/frontal_view_heads`
  - `FaceSim3D/experiment/FaceSimExp/Assets/Faces/CFD`

For memory reasons the `.gitignore` is listing the .png files of all folders. The `.npy` files are kept and can be used to load the data.

---

## License

- **Code** (except where noted) is licensed under the **BSD 3-Clause License** (see `LICENSE.txt`).
- **Documentation and text** are licensed under **CC BY 4.0** (see `LICENSE-CC-BY.txt`).
- This repository includes **third-party code** from Hofmann et al. (2024) under the **MIT License** (see `LICENSE-MIT.txt` and headers in corresponding files).

---

## Citation

### This Thesis (XAI Extension)
Lea Gihlein. Explainable AI methods for human-aligned face perception. Master's Thesis, University Leipzig, 2025

### Original FaceSim3D Project
If you use the base code or data, please cite the original paper by Hofmann et al.:
[Hofmann, S. M., Ciston, A. B., Koushik, A., Klotzsche, F., Hebart, M. N., Müller, K., … Gaebler, M. (2025, August 11). Dynamic presentation in 3D modulates face similarity judgments – A human-aligned encoding model approach. https://doi.org/10.31234/osf.io/f62pw_v4](https://osf.io/preprints/psyarxiv/f62pw_v4)


### Methodological References

If you build upon the specific methods used in this thesis, please consider citing:

**Layer-wise Relevance Propagation (LRP)**
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek, W. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 10(7), e0130140.

**Zennit (LRP Implementation)**
Anders, C. J., Neumann, D., Samek, W., Müller, K. R., & Lapuschkin, S. (2026). Software for dataset-wide XAI: from local explanations to global insights with Zennit, CoRelAy, and ViRelAy. PloS one, 21(1), e0336683.

**MediaPipe**
Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019, June). Mediapipe: A framework for perceiving and processing reality. In Third workshop on computer vision for AR/VR at IEEE computer vision and pattern recognition (CVPR) (Vol. 2019).

**VGG-Face (Base Model)**
Parkhi, O., Vedaldi, A., & Zisserman, A. (2015). Deep face recognition. In BMVC 2015-Proceedings of the British Machine Vision Conference 2015. British Machine Vision Association.

**Qwen2-VL (Large Language Model)**
Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., ... & Lin, J. (2024). Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

**Chicago Face Database (CFD)**
Ma, D. S., Correll, J., & Wittenbrink, B. (2015). The Chicago face database: A free stimulus set of faces and norming data. Behavior research methods, 47(4), 1122-1135.

---

## Contributors

**Thesis Author:**
*   [Lea Gihlein](https://github.com/lea-33)

**Supervisors:**
*   [Dr. Nico Scherf](https://neural-data-science-lab.github.io/team/nico-scherf)
*   [Dr. Robert Haase](https://haesleinhuepf.github.io)

**Original FaceSim3D Authors:**
*   [Simon M. Hofmann*](https://bsky.app/profile/smnhfmnn.bsky.social)
*   Anthony Ciston
*   Abhay Koushik
*   Felix Klotzsche
*   Martin N. Hebart
*   Klaus-Robert Müller
*   Arno Villringer
*   Nico Scherf
*   Anna Hilsmann
*   Vadim V. Nikulin
*   Michael Gaebler

*\* corresponding author of original paper*

