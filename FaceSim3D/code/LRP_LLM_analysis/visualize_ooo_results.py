"""
Script to visualize the odd-one-out results from the study_task_by_llm.py script.
It reads in the results table and creates different plots to analyze the LLM performance compared to the human ground truth, Human Judgment Model and MaxP5_3 Similarity model.
This supports the cross-model comparison by quantifying alignment between different decision-making systems (Cohen's Kappa).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from facesim3d import local_paths

# read in the results csv
results_dir = local_paths.DIR_LLM_ANALYSIS_RESULTS 
results_df = pd.read_csv(os.path.join(results_dir, "llm_pipeline_#10000.csv"))
print(f"Loaded results for {len(results_df)} samples.")

# Calculate accuracies
results_df['llm_correct'] = results_df['llm_answer'] == results_df['ground_truth']
llm_accuracy = results_df['llm_correct'].mean()
print("LLM Accuracy: {:.2f}%".format(llm_accuracy * 100))

results_df['hj_correct'] = results_df['human_judgement'] == results_df['ground_truth']
hj_accuracy = results_df['hj_correct'].mean()
print("Human Judgment Model Accuracy: {:.2f}%".format(hj_accuracy * 100))

results_df['maxp_sim_correct'] = results_df['maxp_sim'] == results_df['ground_truth']
maxp_sim_accuracy = results_df['maxp_sim_correct'].mean()
print("MaxP5_3 Similarity Model Accuracy: {:.2f}%".format(maxp_sim_accuracy * 100))

# ==== Visualizations ====
# Show histogram of distribution of choices by LLM, GT, Human Judgment Model and MaxP5_3 Similarity Model
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.countplot(x='llm_answer', data=results_df)
plt.title('LLM Choices Distribution')
plt.subplot(2, 2, 2)
sns.countplot(x='ground_truth', data=results_df)
plt.title('Ground Truth Choices Distribution')
plt.subplot(2, 2, 3)
sns.countplot(x='human_judgement', data=results_df)
plt.title('Human Judgment Model Choices Distribution')
plt.subplot(2, 2, 4)
sns.countplot(x='maxp_sim', data=results_df)
plt.title('MaxP5_3 Similarity Model Choices Distribution')
plt.tight_layout()
plt.show()

# ==== AGREEMENT ANALYSIS ====
# Test whether there is a strong or week agreement between llm, gt, hj and maxp sim
from sklearn.metrics import cohen_kappa_score

kappa_llm_hj = cohen_kappa_score(results_df['llm_answer'], results_df['human_judgement'])
kappa_llm_maxp = cohen_kappa_score(results_df['llm_answer'], results_df['maxp_sim'])
kappa_hj_maxp = cohen_kappa_score(results_df['human_judgement'], results_df['maxp_sim'])
kappa_gt_hj = cohen_kappa_score(results_df['ground_truth'], results_df['human_judgement'])
kappa_gt_maxp = cohen_kappa_score(results_df['ground_truth'], results_df['maxp_sim'])
kappa_gt_llm = cohen_kappa_score(results_df['ground_truth'], results_df['llm_answer'])
print("Cohen's Kappa Scores:")
print(f"LLM vs Human Judgment Model: {kappa_llm_hj:.2f}")
print(f"LLM vs MaxP5_3 Similarity Model: {kappa_llm_maxp:.2f}")
print(f"Human Judgment Model vs MaxP5_3 Similarity Model: {kappa_hj_maxp:.2f}")
print(f"Ground Truth vs Human Judgment Model: {kappa_gt_hj:.2f}")
print(f"Ground Truth vs MaxP5_3 Similarity Model: {kappa_gt_maxp:.2f}")
print(f"Ground Truth vs LLM: {kappa_gt_llm:.2f}")




