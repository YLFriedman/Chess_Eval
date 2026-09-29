import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, fens_tensor, evals_tensor):
        self.fens_tensor = fens_tensor
        self.evals_tensor = evals_tensor

    def __len__(self):
        return len(self.fens_tensor)

    def __getitem__(self, idx):
        # Decode string and retrieve target, deferring tensor creation
        fen_bytes = self.fens_tensor[idx].numpy().tobytes()
        fen = fen_bytes.decode('utf-8').rstrip('\x00')
        
        target = self.evals_tensor[idx]
        y = target.unsqueeze(0)
        
        return fen, y