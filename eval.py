import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import data
import random
import torch

# change date and version here
date = '01_23_25'
version ='0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--- data prep ---#

# prepare posterior samples and corresponding class labels
post_samples_test = np.load(f'post_samples/post_samples_test_{date}_{version}.npy', allow_pickle=True)
post_samples_train = np.load(f'post_samples/post_samples_train_{date}_{version}.npy', allow_pickle=True)
post_samples_all = np.concatenate((post_samples_train, post_samples_test), axis=0)

labels_test = np.load('class_labels/test_set_labels.npy')
# labels_train = np.load('class_labels/train_set_labels.npy')

labels_test = np.repeat(labels_test, 56, axis=0) # this is (54152,)
# labels_train = np.repeat(labels_train, 56, axis=0)

# labels_all = np.concatenate((labels_train, labels_test), axis=0)

print('shape of concatenated posterior samples:', post_samples_all.shape)
# print('shape of concatenated labels:', labels_all.shape)

# load raw 56 extreme values, which will be superimposed on the histograms
gr_truth = pd.read_csv('../forward_model_LMC_gp/raw_data/results_T4909_filtered_ordered_N56.csv')
gr_truth = gr_truth.iloc[:, 2:].to_numpy().astype(np.float32)
nan_rows = np.isnan(gr_truth).any(axis=1)
gr_truth = gr_truth[~nan_rows] # shape (4832, 56)

print('shape of gr_truth values:', gr_truth.shape)

# label ordering of the original ground truth dataset
ranges_dict = {
    'F_0.3': (0, 467),
    'FA_0.3': (468, 927),
    'FAAA_0.3': (928, 1281),
    'FC_0.3': (1282, 1448),
    'FCCC_0.3': (1449, 1730),
    'F_0.4': (1731, 2048),
    'FA_0.4': (2049, 2404),
    'FAAA_0.4': (2405, 2805),
    'FC_0.4': (2806, 3062),
    'FCCC_0.4': (3063, 3269),
    'F_0.5': (3270, 3458),
    'FA_0.5': (3459, 3835),
    'FAAA_0.5': (3836, 4183),
    'FC_0.5': (4184, 4547),
    'FCCC_0.5': (4548, 4832)
}

#--- create class pools from posterior samples ---#

class_pools_test = {label: [] for label in ranges_dict}
post_samples_test = data.unnormalize_x(post_samples_test)

# iterate over the data points
for i in range(len(post_samples_test)):
    sample = post_samples_test[i]
    label = labels_test[i]

    # Map label to the range in gr_truth
    range_start, range_end = ranges_dict[label]
    num_samples_in_class = range_end - range_start + 1
    num_samples_in_class = num_samples_in_class * 56

    # Sample one random value from the current post_sample row
    sampled_value = random.choice(sample)

    # Add the sampled value to the corresponding class pool
    class_pools_test[label].append(sampled_value)

    # Check if the class pool has enough samples
    if len(class_pools_test[label]) >= num_samples_in_class:
        continue

#--- checking reconstructed observations ---#

recon_obs = np.load(f'post_samples/reconstructed_obs_{date}_{version}.npy', allow_pickle=True)
# load in the training extreme values used for comparing reconstruction data
train_shuffled = np.load(f'randomly_shuffled_data/train_shuffled_for_recon_comparison_{date}_{version}.npy', allow_pickle=True)

# unnormalize
recon_obs = data.unnormalize_x(recon_obs)
train_shuffled = data.unnormalize_x(train_shuffled)

# check the shape of the reconstruction and training data
print('reconstructed observations shape:', recon_obs.shape)
print('randomly shuffled training data shape:', train_shuffled.shape)

# only take the reconstructed observations from the last training epoch
# and only select the training observations from the last training epoch
recon_obs = recon_obs[-216496:]
train_shuffled = train_shuffled[-216496:]

# load in the shuffled_labels from the last training epoch (this should only be used for reconstruction comparison)
shuffled_labels = np.load(f'class_labels/shuffled_set_labels_last_epoch_{date}_{version}.npy')
print('randomly shuffled training class labels shape:', shuffled_labels.shape)

print('shape of reconstructed observations (last training epoch):', recon_obs.shape)
print('shape of pc score train dataset (last training epoch):', train_shuffled.shape)

class_pools_obs = {label: [] for label in ranges_dict}
class_shuffled_train = {label: [] for label in ranges_dict}

# iterate over the data points
for i in range(len(recon_obs)):
    sample = recon_obs[i]
    sample_train = train_shuffled[i]
    # use the order of labels for the shuffled set (randomly shuffled training order in last epoch of training)
    label = shuffled_labels[i]

    # Map label to the range in gr_truth
    range_start, range_end = ranges_dict[label]
    num_samples_in_class = range_end - range_start + 1

    # Sample one random value from the current post_sample row
    sampled_value = random.choice(sample)
    sampled_train_value = random.choice(sample_train)

    # Add the sampled value to the corresponding class pool
    class_pools_obs[label].append(sampled_value)
    class_shuffled_train[label].append(sampled_train_value)

    # Check if the class pool has enough samples as per gr_truth length
    # Remove 'break' to ensure that all classes are sampled
    if len(class_pools_obs[label]) >= num_samples_in_class:
        continue

#--- class pooling over train posterior samples ---#

# load in labels corresponding with the shuffled training data (used for posterior train generation, NOT reconstruction)
shuffled_labels_train = np.load(f'class_labels/shuffled_post_train_labels_{date}_{version}.npy')

class_pools_train = {label: [] for label in ranges_dict}
post_samples_train = data.unnormalize_x(post_samples_train)

# iterate over the data points
for i in range(len(post_samples_train)):
    sample = post_samples_train[i]  # Get the row from post_samples
    label = shuffled_labels_train[i]  # Get the corresponding label

    # Map label to the range in gr_truth
    range_start, range_end = ranges_dict[label]
    num_samples_in_class = range_end - range_start + 1

    # Sample one random value from the current post_sample row
    sampled_value = random.choice(sample)

    # Add the sampled value to the corresponding class pool
    class_pools_train[label].append(sampled_value)

    # Check if the class pool has enough samples as per gr_truth length
    # Remove 'break' to ensure that all classes are sampled
    if len(class_pools_train[label]) >= num_samples_in_class:
        continue  # Just continue to the next iteration of the loop

#--- TEST HISTORGRAMS ---#

fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()

for i, (label, (start_idx, end_idx)) in enumerate(ranges_dict.items()):

    ax = axes[i]

    gr_truth_vals = gr_truth[start_idx + 1:end_idx].flatten()
    # gr_truth_vals_scaled = data.input_scaler.transform(gr_truth_vals.reshape(-1, 1))
    # gr_truth_vals = gr_truth_vals_scaled.flatten()

    rand_post_samples = np.array(class_pools_test[label]).flatten()

    combined_values = np.concatenate((rand_post_samples, gr_truth_vals))
    mean = combined_values.mean()
    std = combined_values.std()

    normalized_samples = (rand_post_samples - mean) / std
    normalized_ground_truth = (gr_truth_vals - mean) / std

    # compute scaled Wasserstein distance
    w_dist = wasserstein_distance(normalized_samples, normalized_ground_truth)

    ax.hist(rand_post_samples, density=True, bins=20, alpha=0.5, label="Posterior Samples", color='orange')
    # ax.hist(gr_truth_vals, density=True, bins=20, alpha=0.4, label="Ground Truth", color='blue')

    ax.set_title(f"Histogram Comparison for Class: {label}")
    ax.set_xlabel("MP Stress (Pa)")
    ax.set_ylabel("Density")
    ax.set_xlim(1.0e6, 4.5e6)
    ax.set_xticks(np.linspace(1.0e6, 4.5e6, 7))

    ax.legend(
        labels=[
            "Posterior Samples (Test)", 
            "Ground Truth", 
            f"Wasserstein Distance: {w_dist:.3f}"
        ], 
        prop=dict(size=12), loc='upper right'
    )

plt.tight_layout()
plt.savefig(f'histograms/{date}/histogram_test_compare_{date}_{version}.png', dpi=300)

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
plt.savefig(f'histograms/{date}/histogram_train_compare_{date}_{version}.png', dpi=300)

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
plt.savefig(f'histograms/{date}/latent_reconstruction_{date}_{version}.png', dpi=300)

#--- evaluating the F35 class ---#

# load in gr truth
gr_truth_f35 = data.gr_truth_f35 # shape (200,56)

# load in posterior samples (all in the same class)
post_samples_f35 = np.load(f'post_samples/post_samples_f35_{date}_{version}.npy', allow_pickle=True) # shape (11200*20, 56, 1)

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
plt.savefig(f'histograms/{date}/histogram_f35_{date}_{version}.png', dpi=300)
plt.show()
