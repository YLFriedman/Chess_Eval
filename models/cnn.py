import torch.nn as nn

class ChessCNN(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            nn.Linear(1152, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)