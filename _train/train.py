import torch
import numpy as np
import matplotlib.pyplot as plt

def train_with_dataloader(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
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


def evaluate_with_dataloader(model, MODEL_NAME, dataloader, TARGET_DIR, K_val, device):
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
            plt.savefig(f'{TARGET_DIR}{MODEL_NAME}_{str(K_val)}_dist.png')
            plt.show()
            plt.hist(results, color='black')
            plt.xlim([-6, 6])
            plt.savefig(f'{TARGET_DIR}{MODEL_NAME}_{str(K_val)}_hist.png')
            plt.show()
    
    return np.array(results)