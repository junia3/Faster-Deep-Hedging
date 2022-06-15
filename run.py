# You can utilize same code on jupyter notebook example
# Or colab ^~^
import os
import torch
import torch.nn as nn
import pandas as pd
import torchsummary
from _dataset import dataset
from _utils.utils import evaluate_model
from torch.utils.data import DataLoader
from _model.model import DeltaNet, ConvDeltaNet
from _train.train import train_with_dataloader, evaluate_with_dataloader

def generate_data(args, batch_size, K_val):
    # r, sig, T, N = args
    p_dataset = dataset.PriceDataset(5000, args, 100, K_val)
    v_dataset = dataset.PriceDataset(500, args, 100, K_val)
    train_dataloader = DataLoader(p_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(v_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def train_network(model, dataloader, epochs, device):
    # instantiate model
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, verbose=False)
    criterion = nn.L1Loss()
    loss_hist_delta = []
    print('<Training epochs : %d>'%epochs)
    for epoch in range(epochs):
        current_loss = train_with_dataloader(model, dataloader, optimizer, criterion, device)
        loss_hist_delta.append(current_loss)
        print('[epoch : %d] ===========> training loss: %.6f' % (epoch + 1, current_loss))
        scheduler.step()

    return model, loss_hist_delta

def eval_network(model, MODEL_NAME, dataloader, result, K_val, device):
    eval_result = evaluate_model(evaluate_with_dataloader(model, MODEL_NAME,
                                                          dataloader,
                                                          os.path.join(os.getcwd(),'results/'),
                                                          K_val, device))
    df = pd.DataFrame(eval_result, index=[f'{MODEL_NAME}: {str(K_val)}'])
    result = pd.concat([result, df])
    return result


'''
    This is the running on the main stream
    You can change any of the hyperparameters
'''
def example():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = pd.DataFrame()
    # You can change parameters
    r = 0.02
    sig = 0.2
    T = 100 / 365
    N = 500
    K_val = 100

    print('Generating datasets ...')
    # generate dataset, dataloder
    args = [r, sig, T, N]
    train_dataloader, val_dataloader = generate_data(args, 256, K_val=K_val)

    print('Instantiate model')
    # instantiate model to train and test
    model = DeltaNet(N)         # 'DeltaNet', 'ConvDeltaNet'
    MODEL_NAME = 'SLP'          # 'SLP', 'SCNN'
    input_shape = (256, N)      # '(256, N), '256, 1, N'
    torchsummary.summary(model, input_shape, device=str(device))

    print('Start training ...')
    model, _ = train_network(model, train_dataloader, 100, device)

    print('Evaluation ...')
    result = eval_network(model, MODEL_NAME, val_dataloader, result, K_val, device)

    print(result)

if __name__ == '__main__':
    example()