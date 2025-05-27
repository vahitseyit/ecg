import torch
import torch.nn as nn

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch4 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4_conv(self.branch4(x))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bn(out)
        return self.relu(out)

class InceptionTime1D(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.incept1 = InceptionBlock1D(in_channels, 32)
        self.incept2 = InceptionBlock1D(32*4, 64)
        self.incept3 = InceptionBlock1D(64*4, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128*4, n_classes)
    def forward(self, x):
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.incept3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))  # Multi-label için sigmoid
