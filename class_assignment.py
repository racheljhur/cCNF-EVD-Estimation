''' RVE Assignment Script '''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from rich.progress import track
import data
from datetime import date
import pickle as pk
import os

'''
When approximating an RVE by an ensemble of SVEs, you must group them by similar
target spatial statistics. Each SVE descriptor and its corresponding extreme response
in my training dataset has been randomly shuffled. We need to identify which RVE each
SVE belongs to s.t. we can correctly assess the model's performance on performance
characterization per RVE.

Accordingly, this script attempts to match numerical PC vectors in my shuffled training set to
the numerical values in the original PC dataset, whose rows are tied to RVE labels by a dictionary
in the data.py script.
'''

d = date.today().isoformat()
version='0'

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load in original pc_score dataset with known RVE assignment for row ranges
pc_scores = data.PCs_no_rep_scaled # (3000,10)

# First, I need to assign known 10-D PC vectors to their corresponding classes
patterns = ['F30', 'FA30', 'FAAA30', 'FC30', 'FCCC30',
            'F40', 'FA40', 'FAAA40', 'FC40', 'FCCC40',
            'F50', 'FA50', 'FAAA50', 'FC50', 'FCCC50']

# Make a dictionary with corresponding index range
patterns_dict = {
    'F30': (0, 199),
    'FA30': (200, 399),
    'FAAA30': (400, 599),
    'FC30': (600, 766),
    'FCCC30': (767, 966),
    'F40': (967, 1166),
    'FA40': (1167, 1366),
    'FAAA40': (1367, 1610),
    'FC40': (1611, 1810),
    'FCCC40': (1811, 2010),
    'F50': (2011, 2199),
    'FA50': (2200, 2399),
    'FAAA50': (2400, 2599),
    'FC50': (2600, 2799),
    'FCCC50': (2800, 2999)
}

pc_scores_labeled={}
for patt, (start, end) in patterns_dict.items():
    pc_scores_labeled[patt]=[]
    pc_scores_tmp=pc_scores[start:end+1]
    pc_scores_labeled[patt].append(pc_scores_tmp)
    pc_scores_labeled[patt]=np.array(pc_scores_labeled)
    # This should be (200ish,10)
    print(pc_scores_labeled[patt].shape)

# Write results
out_dir='../../data/pc_scores/'
with open(out_dir +'pc_scores_labeled_dictionary.pkl', 'wb') as f:
    pickle.dump(pc_scores_labeled, f)

'''
The idea is that I will be using the numerical values from the unshuffled, labeled dataset
and the numerical values from the shuffled, unlabeled dataset to get the shuffled RVE labels.
'''

# Load in all shuffled datasets
data_dir_train=f'training_out/{d}/{version}/'
data_dir_test=f'testing_out/{d}/{version}/'

pc_scores_shuffled=np.load(data_dir_train+'shuffled_pc_scores_train.npy')
extreme_vals_shuffled=np.load(data_dir_train+'shuffled_extreme_vals_train.npy')
ccnf_samples_train=np.load(data_dir_train+'train_samples.npy')

pc_scores_test=data.y_test
extreme_vals_test=data.x_test
ccnf_samples_test=np.load(data_dir_test+'test_samples.npy')

# Check if this saved across all batches or not
# Total training dataset size=0.8*831,600, n_batches=20, batch size=33,264
print(pc_scores_shuffled.shape)

# Take the last batch
n_training=0.8*831600
pc_scores_shuffled = pc_scores_shuffled[-n_training:]
# Dataset which will store [row, column, class]
matching_indices = np.empty((len(pc_scores_shuffled), 2), dtype=int)

'''
This is what the following code snippet does. 
For a given pattern, it checks all the rows and corresponding value in pc_scores_labeled[patt],
and checks if there is a match in pcs_shuffled_train. If there is, it marks the row in the shuffled
dataset and assigns it the pattern. It does this for all patterns until there is a match.

The rows EVs train and train_samples will be marked accordingly, since they are shuffled in the same manner
as pcs_shuffled_train. Note that there is a random_seed in place for the test/train split, but this is not
the case for shuffle=True in the training dataloader.
'''

# Prepare dictionaries for assignment
pcs_shuffled_train = {patt: [] for patt in pc_scores_labeled.keys()}
EVs_train = {patt: [] for patt in pc_scores_labeled.keys()}
train_samples = {patt: [] for patt in pc_scores_labeled.keys()}

# Loop through each row in the shuffled scores
for row in pc_scores_shuffled:
    found = False
    for patt, arr in pc_scores_labeled.items():
        # Find all matching rows
        match_idx = np.where((arr == row).all(axis=1))[0]
        if match_idx.size > 0:
            pcs_shuffled_train[patt].append(row)

            # Once we've found the row for a given pattern for which
            # there is a numerical match, use 'row' to grab the correct
            # values from these two datasets. Then append them to the dictionaries
            # with the pattern for which there was a match.

            EVs_train[patt].append(extreme_vals_shuffled[i])
            train_samples[patt].append(ccnf_samples_train[i])

            found = True
            # Stop checking once match is found
            break 
    if not found:
        print("Row not found in any labeled set:", row)

# Convert all lists to numpy arrays
pcs_shuffled_train = {k: np.array(v) for k, v in pcs_shuffled_train.items()}
EVs_train = {k: np.array(v) for k, v in EVs_train.items()}
train_samples = {k: np.array(v) for k, v in train_samples.items()}

# Repeat class assignment for test labels (since I performed a random 80/20 split, 
# I don't know what my test labels are, even if it isn't randomly shuffled)

pcs_shuffled_test = {patt: [] for patt in pc_scores_labeled.keys()}
EVs_test = {patt: [] for patt in pc_scores_labeled.keys()}
test_samples = {patt: [] for patt in pc_scores_labeled.keys()}

# Loop through each row in the shuffled scores
for row in pc_scores_test:
    found = False
    for patt, arr in pc_scores_labeled.items():
        # Find all matching rows
        match_idx = np.where((arr == row).all(axis=1))[0]
        if match_idx.size > 0:
            pcs_shuffled_test[patt].append(row)

            EVs_train[patt].append(extreme_vals_shuffled[i])
            train_samples[patt].append(ccnf_samples_train[i])

            found = True
            # Stop checking once match is found
            break 
    if not found:
        print("Row not found in any labeled set:", row)

pcs_shuffled_test = {k: np.array(v) for k, v in pc_shuffled_test.items()}
EVs_test = {k: np.array(v) for k, v in EVs_test.items()}
test_samples = {k: np.array(v) for k, v in test_samples.items()}

print('Shape of assigned labels (Train)')
for patt in patterns:
    print(pcs_shuffled_train[patt].shape)

print('Shape of assigned labels (Test)')
for patt in patterns:
    print(pcs_shuffled_test[patt].shape)

# Write results
out_dir=f'Organized_Results/{d}/{version}/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

pk.dump(pca, open(out_dir+'pcs_shuffled_train.pkl','wb'))
pk.dump(pca, open(out_dir+'EVs_train.pkl','wb'))
pk.dump(pca, open(out_dir+'train_samples.pkl','wb'))

pk.dump(pca, open(out_dir+'pcs_shuffled_test.pkl','wb'))
pk.dump(pca, open(out_dir+'EVs_test.pkl','wb'))
pk.dump(pca, open(out_dir+'test_samples.pkl','wb'))

# Now, pcs_shuffled_{dataset}, EVs_{dataset}, and {dataset}_samples, where dataset = train/test
# are all dictionaries, and you can easily assess model performance per class
# by accessing class labels.
