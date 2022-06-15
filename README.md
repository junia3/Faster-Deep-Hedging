# Faster-Deep-Hedging
Faster optimization for Asian option Deep Hedging

This is the source code for the paper 'Faster optimization for Asian option Deep Hedging'

---

To begin with, there is a jupyter notebook code, so you just can open it and copy.
```python
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Define current device (cpu or cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```

The requirements for the experiment is following.
- numpy
- pandas
- scipy
- pytorch
- matplotlib

## Calculate Black Scholes premium(Asian option)
```python
def bscall(S, K, T, r, sig):
    d1 = (np.log(S/K)+(r+(sig**2)/6)*(T/2))/(sig*np.sqrt(T/3))
    d2 = (np.log(S/K)+(r-(sig**2)/2)*(T/2))/(sig*np.sqrt(T/3))
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    
def bsput(S, K, T, r, sig):
    d1 = (np.log(S/K)+(r+(sig**2)/6)*(T/2))/(sig*np.sqrt(T/3))
    d2 = (np.log(S/K)+(r-(sig**2)/2)*(T/2))/(sig*np.sqrt(T/3))
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
```
The whole derivatives will be on the paper I wrote.

## Functions that train model one epoch with dataset
```python
def train_with_dataloader(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        model.train()
        S, S_mean, S_diff, premium, K = data
        S_diff = S_diff.to(device)
        premium = premium.to(device)
        geo_call = torch.maximum(S_mean-K, torch.zeros_like(S_mean)).to(device)

        optimizer.zero_grad()
        delta = model(S_diff)
        costs = torch.sum(delta.mul(S_diff), 1)

        loss = criterion(costs+geo_call, premium)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss/len(dataloader)
```

This is the code that only train one epoch the model

## Evaluation
```python
def evaluate_with_dataloader(model, dataloader):
    model.eval()
    prices, results = [], []
    with torch.no_grad():
            for i, data in enumerate(dataloader):
                S, S_mean, S_diff, premium, K = data
                S_diff = S_diff.to(device)
                premium = premium.to(device)
                geo_call = torch.maximum(S_mean-K, torch.zeros_like(S_mean)).to(device)

                delta = model(S_diff)
                costs = torch.sum(delta.mul(S_diff), 1)
                results += list((costs + geo_call-premium).cpu().numpy())
                prices += list(S[:, -1].cpu().numpy())
            
            plt.scatter(prices, results, color='black', s=2)
            plt.ylim([-2, 2])
            plt.savefig(f'slp_{str(K_val)}_dist.png')
            plt.show()
            plt.hist(results, color='black')
            plt.xlim([-6, 6])
            plt.savefig(f'slp_{str(K_val)}_hist.png')
            plt.show()
    
    return np.array(results)
```

Plot result graph in histogram type and distribution type.


--> To be updated later..
