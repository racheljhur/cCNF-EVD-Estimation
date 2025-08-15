''' Main Training Script for Paper #2 '''
import torch
import zuko
import numpy as np
import data
from rich.progress import track
import matplotlib.pyplot as plt
from datetime import date
import os
import sys

d = date.today().isoformat()
version = '0'

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Training log
-------------------------------------------------------------------------------------------------------
Final configuration used in the paper:
08/15/2025 (0): CNF, (64,128,256), batch=8192, weight_decay=1e-5, lr=1e-3, epochs=500, shuffle=true
-------------------------------------------------------------------------------------------------------
'''

flow = zuko.flows.continuous.CNF(features=1, context=10, hidden_features=(64, 128, 256))
flow.to(device)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)

z_train = []
epoch_losses = []

n_epochs=500

for epoch in track(range(n_epochs)):
    losses = []
    for x, c in data.train_loader:

        optimizer.zero_grad()
        loss = -flow(c).log_prob(x).mean()
        loss.backward()
        optimizer.step()
        
        # This is to save the loss per batch, per epoch
        losses.append(loss.detach())

        # Append relevant data for validation at the last training epoch to avoid memory accumulation
        if epoch==n_epochs-1:
            # This will append across all batches at the final training epoch
            z_learned = flow.transform(c)(x).detach()
            z_train.append(z_learned)

    if epoch % 25 == 0:
        # Since I am running this via interactive bash, inactivity sometimes disconnects me.
        # This is to periodically perturb the terminal, so I don't disconnect.
        print(f"[Wake] Epoch {epoch}", file=sys.stderr, flush=True)


    # Average loss across all 20 batches per epoch
    avg_loss = torch.stack(losses).mean().item()
    epoch_losses.append(avg_loss)
    losses = torch.stack(losses)
    
    print(f"({epoch})", losses.mean().item(), "±", losses.std().item())

losses = np.array(losses.detach().cpu())

PATH=f'stored_state/flow_{d}_{version}.pth'
torch.save(flow, PATH)

print(f'Finished training for {n_epochs} epochs')

# Convert everything into appropriate data types
z_train=torch.cat(z_train, dim=0)
z_train=np.array(z_train.detach().cpu())

epoch_losses=np.array(epoch_losses)

print('Shape of z_train:', z_train.shape)
print('Shape of avg losses:', epoch_losses.shape)

out_dir=f'training_out/{d}/{version}/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Write results
np.save(out_dir+'z_train.npy', z_train)
np.save(out_dir+'avg_losses.npy', epoch_losses)

print(f'Finished writing results.')
