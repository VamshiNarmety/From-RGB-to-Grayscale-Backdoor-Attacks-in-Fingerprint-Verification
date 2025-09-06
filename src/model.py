import os
import torch
import torch.nn as nn

class LeNet5Embedding(nn.Module):
    def __init__(self):
        super(LeNet5Embedding, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(120 * 49 * 49, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class SiameseLeNet5(nn.Module):
    def __init__(self):
        super(SiameseLeNet5, self).__init__()
        self.embedding_net = LeNet5Embedding()

    def forward(self, img1, img2):
        output1 = self.embedding_net(img1)
        output2 = self.embedding_net(img2)
        return output1, output2