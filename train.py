import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from models.mlp import ChessMLP, TwoLayerChessMLP
from utils import get_chess_datasets, custom_collate, model_from_checkpoint

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
    train_dataset, val_dataset, _ = get_chess_datasets(args.data)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4096, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=custom_collate
    )

    # 2. Initialize Model, Optimizer, and Loss

   
    criterion = nn.MSELoss()
    
    start_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    # 3. Handle Checkpoint Resuming
    if args.resume:
        print(f"Loading checkpoint from checkpoints/{args.resume}...")
        path = f"checkpoints/{args.resume}"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model, model_config = model_from_checkpoint(checkpoint)  
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)      
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            train_loss_history = checkpoint.get('train_loss_history', [])
            val_loss_history = checkpoint.get('val_loss_history', [])
            print(f"Resumed from full checkpoint at epoch {start_epoch}.")

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2,
                threshold = 1e-3)
            
            #Load scheduler_state_dict
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            start_epoch = 0
            print("Error resuming from checkpoint, beginning at Epoch 0")
    else:
        model_config = {
        'model_type': 'TwoLayerChessMLP',
        'layer1_width': 1024,
        'layer2_width': 512,
        'dropout_rate': 0.2
        }
        model = TwoLayerChessMLP(**{k: v for k, v in model_config.items() if k != 'model_type'})


        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2,
                threshold = 1e-3)
    # 4. Training Loop
    
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
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
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) in checkpoints:
                percent_complete = checkpoints[batch_idx + 1]
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

        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")
        
        # Save updated full checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss_history': train_loss_history,
                'model_config' : model_config,
                'val_loss_history': val_loss_history
            }
            model_name = type(model).__name__
            path = f'checkpoints/{model_name}_best_model.pth'
            torch.save(checkpoint, path)
            print("--> Validation loss improved! Saved full model checkpoint.")
    checkpoint_latest = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss, # Carry forward the best loss
            'model_config': model_config,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }
    torch.save(checkpoint_latest, f"checkpoints/{model_name}_latest_model.pth")
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\nTraining complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s.")

if __name__ == '__main__':
    main()