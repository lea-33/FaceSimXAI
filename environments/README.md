# Computational Environments

This project uses **three separate Python environments**.
Most of the codebase is designed to run in a single **default environment**, while a small number of specialized scripts require alternative environments due to incompatible dependencies.

> **Default rule:**  
> If it is **not explicitly stated otherwise**, scripts should be executed using **`face_sim_env`**.

---

## `face_sim_env` (Default Environment)

**Purpose**  
The primary environment for this project.  
All core scripts and analyses are implemented and tested using `face_sim_env`.
This environment is based on the [FaceSim3D](https://github.com/SHEscher/FaceSim3D) package. Additionally, the [zennit package](https://github.com/chr5tphr/zennit) is required to implement the LRP method.

Create a virtual environment to install the package:
```shell
conda create -n face_sim_env python=3.10.4
```

Activate the environment:
```shell
conda activate face_sim_env
```

Install the packages
```shell
# Go to root folder of FaceSim3D
cd FaceSim3D/
pip install -e .
pip install zennit openai
```


## `latent_analysis_env` (Specialized for Latent Space Analysis)

**Purpose**  
This environment is required only for scripts that perform large-scale clustering, dimensionality reduction, or GPU-accelerated analysis. 

**When to use**  
Use `latent_analysis_env` **only** for scripts or jupyter notebooks within the LRP_PCA_Autoencoder_analysis folder.

Create a virtual environment using the `latent_analysis_env.yml` file:
```shell
conda env create -n latent_analysis_env -f latent_analysis_env.yml
```

Activate the environment:
```shell
conda activate latent_analysis_env
```

Install the FaceSim3D Package (without its dependencies):
```shell
# Go to root folder of FaceSim3D
pip install --no-deps -e .
```


## `mediapipe_env` (Specialized for Regional Analysis)

**Purpose**  
This environment is required only for scripts that use the Mediapipe facial landmark tools to define regions of interest in face images.

**When to use**  
Use `mediapipe_env` **only** for scripts or jupyter notebooks within the LRP_Region_analysis folder.

Create a virtual environment using the `mediapipe_env.yml` file:
```shell
conda env create -n mediapipe_env -f mediapipe_env.yml
```

Activate the environment:
```shell
conda activate mediapipe_env
```

Install the FaceSim3D Package (without its dependencies):
```shell
# Go to root folder of FaceSim3D
pip install --no-deps -e .
```