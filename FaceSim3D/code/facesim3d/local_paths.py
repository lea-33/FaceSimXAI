from pathlib import Path

"""
The following paths are used to store the data and results of the project. 
They can be changed to match the local file structure of the user.

"""

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =============================================================================
# 1. SOURCE DATA (Images & Metadata)
# =============================================================================
DIR_FRONTAL_VIEW_HEADS = str(PROJECT_ROOT / "FaceSim3D/data/frontal_view_heads")
HEAD_MAPPING = str(PROJECT_ROOT / "FaceSim3D/data/faces")
HEAD_N_MAP = str(PROJECT_ROOT / "FaceSim3D/experiment/FaceSimExp/Assets/Faces/CFD/headnmap.csv")

# =============================================================================
# 2. MODELS (Weights & Scripts)
# =============================================================================
VGG_FACE_PTH = str(PROJECT_ROOT / "FaceSim3D/data/models/vgg_face_torch/vggface.pth")
DIR_VGGFACE_BASE_MODEL = str(PROJECT_ROOT / "FaceSim3D/code/VGGFace_Base_Model")

# =============================================================================
# 3. INTERMEDIATE DATA (Heatmaps & Features)
# =============================================================================
# Computed features of the VGG-Face MaxPooling5_3 layer
DIR_VGGFACE_MAXP5_3_DATA = str(PROJECT_ROOT / "FaceSim3D/data/VGGFace_MaxPSim_Model")

# LRP Heatmaps (Original VGG-Face)
DIR_VGG_FACE_HEATMAPS = str(PROJECT_ROOT / "FaceSim3D/results/VGGFace_Base_Heatmaps")

# Averaged heatmaps (per Face-ID) for VGG-Hum and VGG-MaxP
DIR_AVERAGE_HEATMAPS_HUM = str(PROJECT_ROOT / "FaceSim3D/data/heatmaps/averaged_heatmaps_humJudgement")
DIR_AVERAGE_HEATMAPS_MaxP = str(PROJECT_ROOT / "FaceSim3D/data/heatmaps/averaged_heatmaps_maxp5_3")

# Single instance heatmaps for VGG-Hum and VGG-MaxP
DIR_SINGLE_HJ_HEATMAPS = str(PROJECT_ROOT / "FaceSim3D/data/heatmaps/single_LRP_HJ_heatmaps_pred_ID")
DIR_SINGLE_MaxP_HEATMAPS = str(PROJECT_ROOT / "FaceSim3D/data/heatmaps/single_LRP_MAXP_heatmaps_pred_ID")

# =============================================================================
# 4. ANALYSIS RESULTS (Final Outputs)
# =============================================================================
# LLM Analysis Results
DIR_LLM_ANALYSIS_RESULTS = str(PROJECT_ROOT / "FaceSim3D/results/LRP_LLM_analysis")

# Region Analysis Results
DIR_REGION_ANALYSIS_RESULTS = str(PROJECT_ROOT / "FaceSim3D/results/LRP_Region_analysis")

# PCA & Autoencoder Results
DIR_PCA_AE_RESULTS_HJ = str(PROJECT_ROOT / "FaceSim3D/results/LRP_PCA_Autoencoder_analysis/Hum_Judge")
DIR_PCA_AE_RESULTS_MaxP = str(PROJECT_ROOT / "FaceSim3D/results/LRP_PCA_Autoencoder_analysis/MaxP")
DIR_PCA_AE_RESULTS_betaVAE = str(PROJECT_ROOT / "FaceSim3D/results/LRP_PCA_Autoencoder_analysis/betaVAE")
# Heatmap Comparison 
DIR_HEATMAP_COMPARISON = str(PROJECT_ROOT / "FaceSim3D/code/LRP_regional_analysis/Comparison_per_Head")

