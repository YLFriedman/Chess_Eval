import torch.nn as nn

import torch
import torch.nn as nn

class ChessCNN(nn.Module):
    def __init__(self, mlp_hidden1=512, mlp_hidden2=128, dropout_rate=0.2):
        super().__init__()
        
        self.conv_stack = nn.Sequential(
            # Block 1: Input (20) -> 64
            nn.Conv2d(20, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Block 4: The 1x1 Bottleneck (256 -> 64)
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        # 64 channels * 8 rows * 8 cols = 4096 input features
        self.mlp = nn.Sequential(
            # Stage 1: Large transition layer with dropout
            nn.Linear(4096, mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Stage 2: Precise evaluation funnel (Unregularized)
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            
            # Output Layer: Scalar centipawn evaluation
            nn.Linear(mlp_hidden2, 1)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        return self.mlp(x)