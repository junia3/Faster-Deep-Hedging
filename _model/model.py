import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaNet(nn.Module):
    def __init__(self, price_length):
        super(DeltaNet, self).__init__()
        self.layer1 = nn.Linear(price_length, price_length*2)
        self.layer2 = nn.Linear(price_length*2, price_length*4)
        self.layer3 = nn.Linear(price_length*4, price_length*2)
        self.layer4 = nn.Linear(2*price_length, price_length)

    def forward(self, x):
        delta = F.leaky_relu(self.layer1(x), 0.01, inplace=True)
        delta = F.leaky_relu(self.layer2(delta), 0.2, inplace=True)
        delta = F.leaky_relu(self.layer3(delta), 0.2, inplace=True)
        delta = torch.tanh(self.layer4(delta))

        return delta


class ConvDeltaNet(nn.Module):
    def __init__(self, price_length):
        super(ConvDeltaNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(32 * price_length, 4 * price_length),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4 * price_length, price_length),
        )

    def forward(self, x):
        delta = x.unsqueeze(1)
        delta = F.relu(self.conv1(delta), inplace=True)
        delta = F.relu(self.conv2(delta), inplace=True)
        delta = F.relu(self.conv3(delta), inplace=True)

        delta = delta.view(delta.size()[0], -1)
        delta = torch.tanh(self.classifier(delta))
        return delta