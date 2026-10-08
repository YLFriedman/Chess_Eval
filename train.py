import argparse
import torch
import torch.nn as nn
import time
import mlflow

from utils import get_data_loaders, state_from_checkpoint

def main():     
    parser = argparse.ArgumentParser(description="Train ChessMLP")
    parser.add_argument("--data", type=str, default="data/chessData_cleaned.parquet", help="Path to parquet data")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--resume", action='store_true', help="Resume mode")
    parser.add_argument("--resume_file", type=str, default="ChessCNN_best_model.pth", help="Path to resume checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, _ = get_data_loaders(args.data, args.batch_size)
    
    # 2. Initialize Architecture via State Object
    if args.resume:
        checkpoint = torch.load(f"checkpoints/{args.resume_file}", map_location=device, weights_only=False)
        state = state_from_checkpoint(checkpoint, device)
        print(f"Resumed from full checkpoint at epoch {state.start_epoch}.")
    else: 
        state = state_from_checkpoint({}, device)

    criterion = nn.MSELoss()
    model_name = type(state.model).__name__
    
    # 3. Setup MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Chess_Architecture_Search")
    
    # Resume the existing run if we have an ID, otherwise start a new one
    if state.mlflow_run_id:
        mlflow.start_run(run_id=state.mlflow_run_id)
        print(f"Resuming MLflow Run: {state.mlflow_run_id}")
    else:
        active_run = mlflow.start_run()
        state.mlflow_run_id = active_run.info.run_id
        
        # Log Hyperparameters for new runs
        mlflow.log_params(state.model_config)
        mlflow.log_param("optimizer", type(state.optimizer).__name__)
        mlflow.log_param("batch_size", args.batch_size)
        for k, v in state.scheduler_config.items():
            mlflow.log_param(f"sched_{k}", v)

    # 4. Training Loop
    print("\nStarting training...")
    start_time = time.time()

    scaler = torch.amp.GradScaler('cuda')

    try:
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
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                state.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = state.model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(state.optimizer)
                scaler.update()
                
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
            
            # Log Metrics to MLflow DB
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch+1)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch+1)
            mlflow.log_metric("learning_rate", current_lr, step=epoch+1)
            
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
                'best_val_loss': state.best_val_loss,
                'mlflow_run_id': state.mlflow_run_id
            }
            
            # Save optimal model
            if avg_val_loss < state.best_val_loss:
                state.best_val_loss = avg_val_loss
                checkpoint_dict['best_val_loss'] = state.best_val_loss
                torch.save(checkpoint_dict, f'checkpoints/{model_name}_best_model.pth')
                print("--> Validation loss improved! Saved optimal model checkpoint...")
            
            # Always save latest model
            print("Saving latest model...")
            torch.save(checkpoint_dict, f"checkpoints/{model_name}_latest_model.pth")
        
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print(f"\nTraining complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s.")
    
    except KeyboardInterrupt:
        print("\n\n[-] Training manually interrupted by user (Ctrl+C).")
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTraining Runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s.")

    finally:
        # Ensures the run gracefully ends even if you manual interrupt (Ctrl+C)
        mlflow.end_run()

if __name__ == '__main__':
    main()