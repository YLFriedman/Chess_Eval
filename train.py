import argparse
import torch
import torch.nn as nn

import time
from dataclasses import dataclass, field

from models.mlp import ChessMLP, TwoLayerMLP, ThreeLayerMLP
from utils import get_data_loaders, model_from_checkpoint

DEFAULT_CONFIG = {
    'model_type': 'ThreeLayerMLP',
    'layer1_width': 2048,
    'layer2_width': 1024,
    'layer3_width' : 256,
    'dropout_rate': 0.2
}

DEFAULT_SCHEDULER =  {
    'mode' :'min', 
    'factor':0.5, 
    'patience':2,
    'threshold' :1e-3
}

@dataclass
class TrainingState:
    """Bundles all training objects and history."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    model_config: dict
    scheduler_config: dict
    start_epoch: int = 0
    best_val_loss: float = float('inf')
    train_history: list = field(default_factory=list)
    val_history: list = field(default_factory=list)
    lr_history: list = field(default_factory=list)


def setup_resume(file_path, device):
    """Loads all components from a checkpoint and returns a TrainingState."""

    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model, model_config = model_from_checkpoint(checkpoint)  
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Fallback to defaults if scheduler config is missing from an old checkpoint
        scheduler_config = checkpoint.get('scheduler_config', DEFAULT_SCHEDULER)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return TrainingState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            scheduler_config=scheduler_config,
            start_epoch=checkpoint.get('epoch', 0),
            best_val_loss=checkpoint.get('best_val_loss', float('inf')),
            train_history=checkpoint.get('train_loss_history', []),
            val_history=checkpoint.get('val_loss_history', []),
            lr_history=checkpoint.get('lr_history', [])
        )
    else:
        raise ValueError("Invalid checkpoint format")
    

def setup_default(device):
    """Initializes default model and optimizers from scratch."""
    model = ThreeLayerMLP(**{k: v for k, v in DEFAULT_CONFIG.items() if k != 'model_type'})
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **DEFAULT_SCHEDULER) 
    
    return TrainingState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_config=DEFAULT_CONFIG,
        scheduler_config=DEFAULT_SCHEDULER
    )


def main():     
    parser = argparse.ArgumentParser(description="Train ChessMLP")
    parser.add_argument("--data", type=str, default="data/chessData_cleaned.parquet", help="Path to parquet data")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--resume", type=str, default='', help="Name of checkpoint file to resume from")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader = get_data_loaders(args.data, args.batch_size)
    
    # 2. Initialize Model, Optimizer, and Loss
    if args.resume:
        state = setup_resume(f"checkpoints/{args.resume}", device)
        print(f"Resumed from full checkpoint at epoch {state.start_epoch}.")
    else: 
        state = setup_default(device)

    criterion = nn.MSELoss()
    model_name = type(state.model).__name__
    
    # 3. Training Loop
    
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(state.start_epoch, args.epochs):
        state.model.train()
        running_train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 20)

        total_batches = len(train_loader)
        checkpoints = {
            max(1, total_batches // 4): 25,
            max(1, total_batches // 2): 50,
            max(1, 3 * total_batches // 4): 75,
            total_batches: 100
        }
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            state.optimizer.zero_grad()
            outputs = state.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            state.optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) in checkpoints:
                percent_complete = checkpoints[batch_idx + 1]
                print(f"[{percent_complete:>3.0f}%] Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")
    
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation Phase
        state.model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = state.model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        avg_val_loss = running_val_loss / len(val_loader.dataset)

        # Step the scheduler based on the validation loss
        state.scheduler.step(avg_val_loss)
        current_lr = state.optimizer.param_groups[0]['lr']
        
        state.train_history.append(avg_train_loss)
        state.val_history.append(avg_val_loss)
        state.lr_history.append(current_lr)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")
        
        # Base checkpoint dictionary
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': state.model.state_dict(),
            'optimizer_state_dict': state.optimizer.state_dict(),
            'scheduler_state_dict': state.scheduler.state_dict(),
            'model_config': state.model_config,
            'scheduler_config': state.scheduler_config,
            'train_loss_history': state.train_history,
            'val_loss_history': state.val_history,
            'lr_history': state.lr_history,
            'best_val_loss': state.best_val_loss
        }
        
        # Save optimal model
        if avg_val_loss < state.best_val_loss:
            state.best_val_loss = avg_val_loss
            checkpoint_dict['best_val_loss'] = state.best_val_loss
            torch.save(checkpoint_dict, f'checkpoints/{model_name}_best_model.pth')
            print("--> Validation loss improved! Saved optimal model checkpoint.")
        
        # Always save latest model
        torch.save(checkpoint_dict, f"checkpoints/{model_name}_latest_model.pth")
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\nTraining complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s.")

if __name__ == '__main__':
    main()