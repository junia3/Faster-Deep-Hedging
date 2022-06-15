import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

# Black scholes call option premium and put option premium
def bscall(S, K, T, r, sig):
    d1 = (np.log(S/K)+(r+(sig**2)/6)*(T/2))/(sig*np.sqrt(T/3))
    d2 = (np.log(S/K)+(r-(sig**2)/2)*(T/2))/(sig*np.sqrt(T/3))
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    
def bsput(S, K, T, r, sig):
    d1 = (np.log(S/K)+(r+(sig**2)/6)*(T/2))/(sig*np.sqrt(T/3))
    d2 = (np.log(S/K)+(r-(sig**2)/2)*(T/2))/(sig*np.sqrt(T/3))
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

# Entropic loss measurements
def entropic_loss(pnl):
    pnl = torch.Tensor(pnl)
    return -torch.mean(-torch.exp(-pnl)).numpy()

def evaluate_model(Y):
    Y = pd.Series(Y)
    metric = {"Entropic Loss Measure (ERM)" : entropic_loss(Y)}
    return metric