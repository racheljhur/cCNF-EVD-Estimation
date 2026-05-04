"""Main Training Script"""
import torch
import zuko
import numpy as np
import data
from rich.progress import track
import matplotlib.pyplot as plt
from datetime import date
import os
import sys

d=date.today().isoformat()
version="0"

mod_path=f"stored_state/flow_{d}_{version}.pth"
out_path=f"training_out/{d}/{version}/"

os.makesirs(mod_path,exist_ok=True)
os.makesirs(out_path,exist_ok=True)

dim=1
cdim=10

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

flow = zuko.flows.continuous.CNF(features=dim, context=cdim, hidden_features=(64, 128, 256))
flow.to(device)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)

z_train = []
iter_losses = []

n_iters=300

for i in track(range(n_iters)):
    losses = []
    for x, c in data.train_loader:

        optimizer.zero_grad()
        loss = -flow(c).log_prob(x).mean()
        loss.backward()
        optimizer.step()
        
        # This is to save the loss per batch, per training iteration
        losses.append(loss.detach())

        # Append relevant data for validation at the last training iteration to avoid memory accumulation
        if i==n_iters-1:
            # This will append across all batches at the final training iteration
            z_learned = flow.transform(c)(x).detach()
            z_train.append(z_learned)


    # Average loss across all 20 batches per iteration
    avg_loss = torch.stack(losses).mean().item()
    iter_losses.append(avg_loss)
    iter_losses = torch.stack(iter_losses)
    
    print(f"({i})", losses.mean().item(), "±", losses.std().item())

torch.save(flow, out_path)
print(f"Finished training for {n_iters} iterations")

iter_losses = np.array(iter_losses.detach().cpu())
iter_losses=np.array(iter_losses)

z_train=torch.cat(z_train, dim=0)
z_train=np.array(z_train.detach().cpu())

print('Shape of avg losses:', iter_losses.shape)
print('Shape of z_train:', z_train.shape)

# Write results
np.save(out_path+"z_train", z_train)
np.save(out_path+"avg_losses.npy",losses)

print("Finished writing results.")

"""
Training log
-------------------------------------------------------------------------------------------------------
Final configuration used in the paper:
08/15/2025 (0): continuous normalizing flow, (64,128,256), batch=8192, weight_decay=1e-5, lr=1e-3, n_iters=300, shuffle=true
-------------------------------------------------------------------------------------------------------
"""
