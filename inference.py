''' Main Inference Script '''
import matplotlib.pyplot as plt
import torch
import data
from rich.progress import track
import numpy as np
import zuko
import time
import os
from datetime import date

d = date.today().isoformat()
version = '0'

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model type and configuration
flow=zuko.flows.continuous.CNF(features=1, context=10, hidden_features=(64, 128, 256))
flow.to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)

# Load learned hyperparameters
PATH=f'stored_state/flow_{d}_{version}.pth'
flow_bis = torch.load(PATH, weights_only=False)

#--- Conditional Sampling using Train Labels ---#

train_samples=[]
shuffled_pc_scores_train = []
shuffled_extreme_vals_train=[]

for i, (x, c) in enumerate(data.train_loader):
    # This contains shuffled numerical PC vectors, which will be used for RVE assignment (for validation)
    shuffled_pc_scores_train.append(c)
    # This is the baseline FE extreme values for comparison
    shuffled_extreme_vals_train.append(x)

    # To evaluate model performance on the train set, I am conditionally generating samples using train pc scores
    samples = flow_bis(c).sample((56,)).permute(1, 0, 2).cpu().numpy()  # (batch_size, 56, features=1)
    train_samples.append(samples)

# Organize appended data and convert to suitable data types
shuffled_pc_scores_train = torch.cat(shuffled_pc_scores_train, dim=0)
shuffled_pc_scores_train = shuffled_pc_scores_train.cpu().numpy()

shuffled_extreme_vals_train = torch.cat(shuffled_extreme_vals_train, dim=0)
shuffled_extreme_vals_train = shuffled_extreme_vals_train.cpu().numpy()

train_samples = np.concatenate(train_samples, axis=0)
train_samples = np.array(train_samples, dtype=object)

#--- Conditional Sampling using Test Labels ---#

start=time.time()
test_samples=[]
for i, (x, c) in enumerate(data.test_loader):
    #*** TO DO: sample this (1,) instead of (56,) and evaluate inference time.
    samples = flow_bis(c).sample((56,)).permute(1, 0, 2).cpu().numpy()  # (batch_size, 56, features=1)
    test_samples.append(samples)

end=time.time()
print('Elapsed sampling time:',start-end)

test_samples = np.concatenate(test_samples, axis=0)
test_samples = np.array(test_samples, dtype=object)

# To validate z_test in the latent space
z_test=[]
for x, c in data.test_loader:
    z_learned_test = flow_bis.transform(c)(x).detach()
    z_test.append(z_learned_test)
    loss = -flow_bis(c).log_prob(x).mean()
    print(loss)

z_test = torch.cat(z_test, dim=0)
z_test = np.array(z_test.detach().cpu())

#--- SAVING ---#
out_dir_test=f'testing_out/{d}/{version}/'
out_dir_train=f'training_out/{d}/{version}/'

if not os.path.exists(out_dir_test):
    os.makedirs(out_dir_test)
if not os.path.exists(out_dir_train):
    os.makedirs(out_dir_train)

# Testing
print('Shape of shuffled PC scores (Train)')
print(shuffled_pc_scores_train.shape)
print('Shape gr truth extreme values (Train)')
print(shuffled_extreme_vals_train.shape)
print('Shape of conditional sample dataset (Train)')
print(train_samples.shape)

# Write results
np.save(out_dir_train+'shuffled_pc_scores_train.npy', shuffled_pc_scores_train)
np.save(out_dir_train+'shuffled_extreme_vals_train.npy', shuffled_extreme_vals_train)
np.save(out_dir_train+'train_samples.npy', train_samples)

np.save(out_dir_test+'z_test.npy', z_test)
np.save(out_dir_test+'test_samples.npy', test_samples)

print('Finished writing all results.')
