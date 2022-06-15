import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import torch
from torch.utils.data import Dataset
from _utils import utils

class PriceDataset(Dataset):
    def __init__(self, samples, args, price_option=None, strike_option=None):
        r, sig, T, N = args
        self.samples = samples
        dt = T/N
        rdt = r*dt
        sigsdt = sig * np.sqrt(dt)

        self.S = np.empty([self.samples,N+1])
        self.S_mean = np.empty([self.samples])
        self.S_diff = np.empty([self.samples, N])
        self.premium = np.empty([self.samples])
        self.K = np.empty([self.samples])
        self.price_option = price_option

        rv = np.random.normal(rdt, sigsdt, [self.samples,N])

        for i in range(self.samples):
            if price_option is None:
                self.S[i,0] = np.random.uniform(90, 110)
            else:
                self.S[i,0] = price_option

            for j in range(N):
                self.S[i,j+1] = self.S[i,j] * (1+rv[i,j])
                self.S_diff[i, j] = self.S[i, j+1]-self.S[i, j]

        for i in range(self.samples):
            self.S_mean[i] = np.exp(np.mean(np.log(self.S[i, :])))
            if strike_option is None:
                self.K[i] = np.random.uniform(90, 110)
                self.premium[i] = utils.bscall(self.S[i, 0], self.K[i], T, r, sig)
            else:
                self.K[i] = strike_option
                self.premium[i] = utils.bscall(self.S[i, 0], strike_option, T, r, sig)

    def __len__(self):
        return self.samples
  
    def __getitem__(self, idx):
        S = self.S[idx, :]
        mean = self.S_mean[idx]
        diff = self.S_diff[idx, :]
        premium = self.premium[idx]
        k = self.K[idx]

        return torch.from_numpy(S).float(), mean, torch.from_numpy(diff).float(), premium, k