import torch
import torch.nn as nn

class DQN2048(nn.Module):
    def __init__(self):
        super(DQN2048, self).__init__()
        
        # 2D Convolutions analyze structural tile clusters (like pairs or corners)
        self.conv1 = nn.Conv2d(16, 128, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)
        
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.out = nn.Linear(256, 4) # 4 Actions: UP, DOWN, LEFT, RIGHT

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten tensor
        x = torch.relu(self.fc1(x))
        return self.out(x)
