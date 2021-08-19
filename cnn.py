import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(14 * 14 * 6, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU()
        )

    def forward(self, x):
        feature = self.conv(x)
        out = self.fc(feature.view(feature.shape[0], -1))
        return out
