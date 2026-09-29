import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


from dataset import ChessDataset
from models.mlp import ChessMLP
from utils import batch_fen_to_tensor


def custom_collate(batch):
    fens = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    x = batch_fen_to_tensor(fens)
    y = torch.stack(targets)
    return x, y

def main():
    parser = argparse.ArgumentParser(description="Train ChessMLP")
    parser.add_argument("--data", type=str, default="data/chessData_cleaned.parquet", help="Path to parquet data")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to resume from if using an old state_dict checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    df = pd.read_parquet(args.data)
    
    # Convert FEN strings to fixed-length zero-padded byte arrays (S128)
    # Then view as uint8 for PyTorch tensor compatibility
    fens_np = df['FEN'].values.astype('S128')
    fens_uint8 = np.ascontiguousarray(fens_np.view(np.uint8).reshape(len(df), 128))
    
    fens_tensor = torch.from_numpy(fens_uint8)
    evals_tensor = torch.from_numpy(np.ascontiguousarray(df['Target'].values.astype(np.float32)))
    
    # Explicitly move to shared memory to prevent RAM duplication across workers
    fens_tensor.share_memory_()
    evals_tensor.share_memory_()
    
    dataset = ChessDataset(fens_tensor, evals_tensor)
    
    total_size = len(dataset)
    val_size = 500_000
    test_size = 500_000
    train_size = total_size - val_size - test_size
    
    # Set seed to ensure the random split is identical even if training is paused/resumed
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=custom_collate 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4096, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=custom_collate 
    )

    # 2. Initialize Model, Optimizer, and Loss
    model = ChessMLP(hidden_size=512).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    start_epoch = 0
    best_val_loss = float('inf')

    # 3. Handle Checkpoint Resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from full checkpoint at epoch {start_epoch}.")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = args.start_epoch
            print(f"Loaded raw model weights. Starting at manual epoch {start_epoch}.")

    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 20)

        total_batches = len(train_loader)
        print_interval = max(1, total_batches // 4)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_batches:
                percent_complete = 100.0 * (batch_idx + 1) / total_batches
                print(f"[{percent_complete:>3.0f}%] Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")
    
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save updated full checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, "best_chess_model.pth")
            print("--> Validation loss improved! Saved full model checkpoint.")
    
    print("\nTraining complete.")

if __name__ == '__main__':
    main()