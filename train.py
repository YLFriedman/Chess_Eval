import argparse
import torch
import torch.nn as nn
import time

from utils import get_data_loaders, state_from_checkpoint

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
    train_loader, val_loader, _ = get_data_loaders(args.data, args.batch_size)
    
    # 2. Initialize Architecture via State Object
    
    if args.resume:
        checkpoint = torch.load(f"checkpoints/{args.resume}", map_location=device, weights_only=False)
        state = state_from_checkpoint(checkpoint, device)
        print(f"Resumed from full checkpoint at epoch {state.start_epoch}.")
    else: 
        # Passing an empty dict tells the function to fetch defaults from scratch
        state = state_from_checkpoint({}, device)

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