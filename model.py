import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Net(nn.Module):
    def __init__(self, activation="relu"):
        super(CIFAR10Net, self).__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def activate(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "leakyrelu":
            return F.leaky_relu(x, negative_slope=0.1)
        elif self.activation == "elu":
            return F.elu(x)
        else:
            return x  # fallback

    def forward(self, x):
        x = self.pool(self.activate(self.bn1(self.conv1(x))))
        x = self.pool(self.activate(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.activate(self.fc1(x))
        x = self.fc2(x)
        return x
