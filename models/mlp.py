import torch.nn as nn

class ChessMLP(nn.Module):
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

class TwoLayerChessMLP(nn.Module):
    def __init__(self, layer1_width=1024, layer2_width=512, dropout_rate=0.2):
        super().__init__()
        
        # Input tensor is (Batch, 18, 8, 8) -> flattened to 1152
        self.network = nn.Sequential(
            nn.Flatten(),
            
            # Layer 1
            nn.Linear(1152, layer1_width),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(layer1_width, layer2_width),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output Layer (Single centipawn evaluation)
            nn.Linear(layer2_width, 1)
        )

    def forward(self, x):
        return self.network(x)