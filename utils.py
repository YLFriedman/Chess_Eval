import os
import torch
import numpy as np
import pandas as pd
import gc
from datetime import datetime
from dataclasses import dataclass, field
from torch.utils.data import random_split, DataLoader

from dataset import ChessDataset
from models.cnn import ChessCNN
from models.mlp import TwoLayerMLP, ThreeLayerMLP

PIECES = ['P', 'N', 'B', 'R', 'K', 'Q', 'p', 'n', 'b', 'r', 'k', 'q']
CASTLING = ['K', 'Q', 'k', 'q']

DEFAULT_CONFIG = {
    'model_type': 'ThreeLayerMLP',
    'layer1_width': 2048,
    'layer2_width': 1024,
    'layer3_width' : 256,
    'dropout_rate': 0.2
}

DEFAULT_SCHEDULER = {
    'mode': 'min', 
    'factor': 0.5, 
    'patience': 2,
    'threshold': 1e-3
}

@dataclass
class TrainingState:
    """Bundles all training objects and history."""
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    model_config: dict
    scheduler_config: dict
    start_epoch: int = 0
    best_val_loss: float = float('inf')
    train_history: list = field(default_factory=list)
    val_history: list = field(default_factory=list)
    lr_history: list = field(default_factory=list)
    test_loss: float = None

def batch_fen_to_tensor(fens):
    batch_size = len(fens)
    boards, turns, castles, eps = [], [], [], []
    
    for fen in fens:
        parts = fen.split(' ')
        boards.append(parts[0])
        turns.append(parts[1])
        castles.append(parts[2])
        eps.append(parts[3])

    for i in range(1, 9):
        repl = '.' * i
        boards = [b.replace(str(i), repl) for b in boards]
        
    boards = [b.replace('/', '') for b in boards]
    board_arr = np.array(boards, dtype='S64').view('S1').reshape(batch_size, 8, 8)
    tens = np.zeros((batch_size, 18, 8, 8), dtype=np.float32)
    
    for channel, piece in enumerate(PIECES):
        tens[:, channel, :, :] = (board_arr == piece.encode())

    turn_arr = np.array(turns) == 'w'
    tens[:, 12, :, :] = turn_arr[:, None, None]

    for i, c in enumerate(CASTLING):
        has_castle = np.array([c in cast for cast in castles])
        tens[:, 13 + i, :, :] = has_castle[:, None, None]

    ep_arr = np.array(eps)
    has_ep = ep_arr != '-'
    valid_ep_idx = np.where(has_ep)[0]
    
    if len(valid_ep_idx) > 0:
        valid_eps = ep_arr[has_ep]
        ep_bytes = valid_eps.astype('S2').view('S1').reshape(-1, 2)
        cols = ep_bytes[:, 0].view(np.uint8) - ord('a')
        rows = 8 - (ep_bytes[:, 1].view(np.uint8) - ord('0'))
        tens[valid_ep_idx, 17, rows, cols] = 1

    return torch.from_numpy(tens)

def custom_collate(batch):
    fens = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    x = batch_fen_to_tensor(fens)
    y = torch.stack(targets)
    return x, y

def get_chess_datasets(data_path, val_size=500_000, test_size=500_000, seed=42):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    fens_np = df['FEN'].values.astype('S128')
    fens_uint8 = np.ascontiguousarray(fens_np.view(np.uint8).reshape(len(df), 128))
    
    fens_tensor = torch.from_numpy(fens_uint8)
    evals_tensor = torch.from_numpy(np.ascontiguousarray(df['Target'].values.astype(np.float32)))
    
    fens_tensor.share_memory_()
    evals_tensor.share_memory_()
    
    dataset = ChessDataset(fens_tensor, evals_tensor)
    
    total_size = len(dataset)
    train_size = total_size - val_size - test_size
    
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    
    del df
    gc.collect()
    
    return train_dataset, val_dataset, test_dataset

def get_data_loaders(data, batch_size=1024):
    train_dataset, val_dataset, test_dataset = get_chess_datasets(data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4096, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )

    test_loader = DataLoader(
        test_dataset, batch_size=4096, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )
    return train_loader, val_loader, test_loader

def state_from_checkpoint(checkpoint, device='cpu'):
    # Safely extract configuration or fall back to defaults
    config = checkpoint.get('model_config', DEFAULT_CONFIG) if isinstance(checkpoint, dict) and 'model_config' in checkpoint else DEFAULT_CONFIG
    
    kwargs = {k: v for k, v in config.items() if k != 'model_type'}
    model_type = config.get('model_type', 'ThreeLayerMLP')
    
    if model_type == 'TwoLayerMLP':                    
        model = TwoLayerMLP(**kwargs)
    elif model_type == 'ThreeLayerMLP':
        model = ThreeLayerMLP(**kwargs) 
    elif model_type == 'ChessCNN':
        model = ChessCNN(**kwargs)
    else:
        model = ThreeLayerMLP(**kwargs)
        
    # Move parameters to device BEFORE passing to optimizer
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler_config = checkpoint.get('scheduler_config', DEFAULT_SCHEDULER) if isinstance(checkpoint, dict) else DEFAULT_SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)

    # Initialize standard State object
    state = TrainingState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_config=config,
        scheduler_config=scheduler_config
    )

    # Hydrate the State object if weights exist
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        state.start_epoch = checkpoint.get('epoch', 0)
        state.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        state.train_history = checkpoint.get('train_loss_history', [])
        state.val_history = checkpoint.get('val_loss_history', [])
        state.lr_history = checkpoint.get('lr_history', [])
        
    elif isinstance(checkpoint, dict) and len(checkpoint) > 0 and 'epoch' not in checkpoint:
        # Fallback for old raw state_dict checkpoints
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            pass
            
    return state

def write_record(state: TrainingState, path: str):
    """Appends the training run metrics to an Excel file"""
    model_name = type(state.model).__name__
    opt_name = type(state.optimizer).__name__
    
    # Format Layer Widths (e.g. "1024, 512, 256")
    widths = [str(v) for k, v in state.model_config.items() if 'width' in k]
    layer_width_str = ", ".join(widths)
    
    # Format Dropouts (handles single float or lists)
    dropouts = state.model_config.get('dropout_rate', '')
    if isinstance(dropouts, list):
        dropout_str = ", ".join(map(str, dropouts))
    else:
        num_layers = len(widths)
        dropout_str = ", ".join([str(dropouts)] * num_layers) if num_layers > 0 else str(dropouts)
        
    # Get Initial Learning Rate
    lr = state.lr_history[0] if state.lr_history else state.optimizer.param_groups[0]['lr']
    
    # Get Scheduler Info
    sched_name = type(state.scheduler).__name__ if state.scheduler else 'None'
    patience = state.scheduler_config.get('patience', '')
    ratio = state.scheduler_config.get('factor', '')
    threshold = state.scheduler_config.get('threshold', '')
    
    # Run Identifier
    run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Create the 'Runs' dataframe
    runs_new = pd.DataFrame([{
        'Model': model_name,
        'Optimizer': opt_name,
        'Layer Width': layer_width_str,
        'LayerDropout': dropout_str,
        'Learning rate': lr,
        'Sched Type': sched_name,
        'Sched Patience': patience,
        'Sched Ratio': ratio,
        'Sched Threshold': threshold,
        'Run_ID': run_id,
        'Test Loss': state.test_loss
    }])
    
    # 2. Create the 'Epochs' dataframe
    epochs_new = pd.DataFrame({
        'Run_Id': run_id,
        'Epoch #': range(1, len(state.train_history) + 1),
        'Train Loss': state.train_history,
        'Validation Loss': state.val_history,
        'LR': state.lr_history
    })
    
    # 3. Safely read and append if file exists
    if os.path.exists(path):
        try:
            runs_existing = pd.read_excel(path, sheet_name='Runs')
            runs_df = pd.concat([runs_existing, runs_new], ignore_index=True)
        except Exception:
            runs_df = runs_new
            
        try:
            epochs_existing = pd.read_excel(path, sheet_name='Epochs')
            epochs_df = pd.concat([epochs_existing, epochs_new], ignore_index=True)
        except Exception:
            epochs_df = epochs_new
    else:
        runs_df = runs_new
        epochs_df = epochs_new
        
    # 4. Write back to disk
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        runs_df.to_excel(writer, sheet_name='Runs', startrow=1, header=False, index=False)
        epochs_df.to_excel(writer, sheet_name='Epochs', startrow=1, header=False, index=False)






    
