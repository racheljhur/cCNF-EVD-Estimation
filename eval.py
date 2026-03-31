'''Evaluation script for assessing losses, comparing distributions, etc.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import data
import random
import torch
import os
from datetime import date

d = date.today().isoformat()
version = '0'

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In case you want to loop through all the patterns
patterns = ['F30', 'FA30', 'FAAA30', 'FC30', 'FCCC30',
            'F40', 'FA40', 'FAAA40', 'FC40', 'FCCC40',
            'F50', 'FA50', 'FAAA50', 'FC50', 'FCCC50']

# Load in dictionaries, where all values are grouped by class
data_dir=f'Organized_Results/{d}/{version}/'
EVs_train_gr_truth = pk.load(open(data_dir+'EVs_train.pkl','rb'))
train_samples = pk.load(open(data_dir+'train_samples.pkl','rb'))

EVs_test_gr_truth = pk.load(open(data_dir+'EVs_test.pkl','rb'))
test_samples = pk.load(open(data_dir+'test_samples.pkl','rb'))

#--- TRAIN HISTORGRAMS ---#
# **to visualize select classes, since I have like 15 classes...
# make sure you unstandardize samples before plotting

select_patterns=['F30', 'FAAA37', 'FCCC50']
fig, axes = plt.subplots(1,3,figsize=(12, 4))
axes = axes.flatten()

for i, p in enumerate(select_patterns):
    ax = axes[i]
    train_gr_truth=EVs_train_gr_truth[p].flatten()
    train_samples=train_samples[p].flatten()

    # Compute 1-W distance
    # Normalize the distance by the std dev of the ground truth
    mean=train_gr_truth.mean()
    std=train_gr_truth.std()

    # This is mainly to interpret our 1-W distance (>2 is likely not good)
    train_gr_truth_norm=(train_gr_truth-mean)/std
    train_samples_norm=(train_samples-mean)/std

    W1_dist = wasserstein_distance(train_gr_truth_norm, train_samples_norm)

    ax.hist(train_gr_truth, density=True, bins=20, alpha=0.5, label='Ground Truth', color='blue')
    ax.hist(train_samples, density=True, bins=20, alpha=0.5, label='cCNF Samples', color='orange')
    ax.set_xlim(0.5e6, 3.0e6)
    ax.set_xticks(np.linspace(0.5e6, 3.0e6, 7))
    ax.set_xlabel("MP Stress (Pa)")
    ax.set_ylabel("Density")
    ax.set_title(f'Comparing Distributions for Class {p}')
    ax.legend(
        labels=[
            "Ground Truth (FEM)", 
            "cCNF Samples (Train)", 
            f"Wasserstein Distance: {W1_dist:.3f}"
        ], 
        prop=dict(size=12), loc='upper right'
        )

plt.tight_layout()

# Assess 1-W distance across all classes
W1_dist_train=[]
for p in patterns:
    train_gr_truth=EVs_train_gr_truth[p].flatten()
    train_samples=train_samples[p].flatten()

    # Compute 1-W distance
    # Normalize the distance by the std dev of the ground truth
    mean=train_gr_truth.mean()
    std=train_gr_truth.std()

    # This is mainly to interpret our 1-W distance (>2 is likely not good)
    train_gr_truth_norm=(train_gr_truth-mean)/std
    train_samples_norm=(train_samples-mean)/std

    W1_dist = wasserstein_distance(train_gr_truth_norm, train_samples_norm)
    W1_dist_train.append(W1_dist)

W1_dist_train=np.array(W1_dist_train)
print(W1_dist_train.shape)

print('#-------- TRAIN METRICS (1-W Distance) --------#')

row=np.where(W1_dist_train==W1_dist_train.max())[0]
print(f'Maximum W1 Distance, {row}')
print(W1_dist_train.max())

row=np.where(W1_dist_train==W1_dist_train.min())[0]
print(f'Minimum W1 Distance, {row}')
print(W1_dist_train.min())

print('Average W1 Distance')
print(W1_dist_train.mean())

# TO DO:
# --- TRAIN HISTOGRAMS ---#

fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()

for i, (label, (start_idx, end_idx)) in enumerate(ranges_dict.items()):

    ax = axes[i]

    gr_truth_vals = gr_truth[start_idx + 1:end_idx].flatten()
    # gr_truth_vals_scaled = data.input_scaler.transform(gr_truth_vals.reshape(-1, 1))
    # gr_truth_vals = gr_truth_vals_scaled.flatten()

    rand_post_samples = np.array(class_pools_train[label]).flatten()

    combined_values = np.concatenate((rand_post_samples, gr_truth_vals))
    mean = combined_values.mean()
    std = combined_values.std()

    normalized_samples = (rand_post_samples - mean) / std
    normalized_ground_truth = (gr_truth_vals - mean) / std

    # compute scaled Wasserstein distance
    w_dist = wasserstein_distance(normalized_samples, normalized_ground_truth)

    ax.hist(rand_post_samples, density=True, bins=20, alpha=0.5, label="Posterior Samples", color='orange')
    ax.hist(gr_truth_vals, density=True, bins=20, alpha=0.4, label="Ground Truth", color='blue')

    ax.set_title(f"Histogram Comparison for Class: {label}")
    ax.set_xlabel("MP Stress (Pa)")
    ax.set_ylabel("Density")
    ax.set_xlim(1.0e6, 4.5e6)
    ax.set_xticks(np.linspace(1.0e6, 4.5e6, 7))

    ax.legend(
        labels=[
            "Posterior Samples (Train)", 
            "Ground Truth", 
            f"Wasserstein Distance: {w_dist:.3f}"
        ], 
        prop=dict(size=12), loc='upper right'
    )

plt.tight_layout()
plt.savefig(f'histograms/{d}/histogram_train_compare_{d}_{version}.png', dpi=300)

#--- RECONSTRUCTION HISTOGRAMS ---#

fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()

for i, (label, (start_idx, end_idx)) in enumerate(ranges_dict.items()):

    ax = axes[i]    
    
    train_shuffled = np.array(class_shuffled_train[label])
    rand_post_samples = np.array(class_pools_obs[label]).flatten()

    combined_values = np.concatenate((rand_post_samples, train_shuffled))
    mean = combined_values.mean()
    std = combined_values.std()

    normalized_samples = (rand_post_samples - mean) / std
    normal_shuffled = (train_shuffled - mean) / std

    # compute scaled Wasserstein distance
    w_dist = wasserstein_distance(normalized_samples, normal_shuffled)

    ax.hist(rand_post_samples, density=True, bins=20, alpha=0.5, label="reconstructed samples", color='orange')
    ax.hist(train_shuffled, density=True, bins=20, alpha=0.4, label="x train", color='blue')

    ax.set_title(f"Reconstruction for Class: {label}")
    ax.set_xlabel("MP Stress (Pa)")
    ax.set_ylabel("Density")
    ax.set_xlim(1.0e6, 4.5e6)
    ax.set_xticks(np.linspace(1.0e6, 4.5e6, 7))

    ax.legend(
        labels=[
            "Reconstructed Observations (Train)", 
            "Ground Truth", 
            f"Wasserstein Distance: {w_dist:.3f}"
        ], 
        prop=dict(size=12), loc='upper right'
    )

plt.tight_layout()
plt.savefig(f'histograms/{d}/latent_reconstruction_{d}_{version}.png', dpi=300)

#--- evaluating the F35 class ---#

# load in gr truth
gr_truth_f35 = data.gr_truth_f35 # shape (200,56)

# load in posterior samples (all in the same class)
post_samples_f35 = np.load(f'post_samples/post_samples_f35_{d}_{version}.npy', allow_pickle=True) # shape (11200*20, 56, 1)

#--- class pooling over f35 posterior samples ---#

class_pools_f35 = []

# iterate over the data points
for i in range(len(post_samples_f35)):

    # get the current row in the posterior sample array
    sample = post_samples_f35[i]  

    # sample one random value from the current post_sample row (one random column value)
    sampled_value = random.choice(sample).item()

    # Add the sampled value to the corresponding class pool
    class_pools_f35.append(sampled_value)

    # Check if the class pool has enough samples as per gr_truth length
    # Remove 'break' to ensure that all classes are sampled
    if len(class_pools_f35) >= int(200*56):
        break

print('shape of sample pool for the f35 class', np.array(class_pools_f35).shape)

#--- F35 Histogram ---#

# Flatten posterior samples and ground truth arrays
class_pools_f35 = np.array(class_pools_f35) # shape (11200,)
class_pools_f35 = data.unnormalize_x(class_pools_f35)

# output scaler is for the evs
norm_samples = data.output_scaler.fit_transform(class_pools_f35.reshape(-1,1)).flatten()
norm_truth = data.output_scaler.fit_transform(data.f35_evs.detach().cpu()).flatten()

rand_post_samples = class_pools_f35.flatten()
gr_truth_f35_flat = gr_truth_f35.flatten() # gr_truth_f35 is loaded in as (200,56)

# Compute Wasserstein distance using normalized and unnormalized values
w_dist_norm = wasserstein_distance(norm_samples, norm_truth)
w_dist = wasserstein_distance(rand_post_samples, gr_truth_f35_flat)

w_dist = w_dist / 1e6

# Create a single figure and axis
plt.figure()

# Plot histograms
plt.hist(rand_post_samples, density=True, bins=20, alpha=0.5, label="Posterior Samples", color='orange')
plt.hist(gr_truth_f35_flat, density=True, bins=20, alpha=0.4, label="Gr Truth", color='blue')

# adjustments to show histogram patches for the first two labels only    
posterior_patch = mpatches.Patch(color='orange', alpha=0.5, label='Posterior Samples')
ground_truth_patch = mpatches.Patch(color='blue', alpha=0.4, label='Ground Truth')

legend_handles = [posterior_patch, ground_truth_patch]
legend_labels = [
    'Posterior Samples',
    'Ground Truth'
]

plt.legend(legend_handles, legend_labels, prop=dict(size=14), loc='upper right')

plt.text(0.5, 0.7,
    f'W Dist: {w_dist:.3f}',
    verticalalignment='top', horizontalalignment='right',
    transform=ax.transAxes, fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

plt.title("Histograms for the F35 Class", fontsize=12)
plt.xlabel("MP Stress (Pa)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(1.0e6, 4.5e6)
plt.xticks(np.linspace(1.0e6, 4.5e6, 7))

plt.tight_layout()
plt.savefig(f'histograms/{d}/histogram_f35_{d}_{version}.png', dpi=300)
plt.show()
