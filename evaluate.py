import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.mlp import ChessMLP, TwoLayerChessMLP
from models.cnn import ChessCNN
from utils import get_chess_datasets, custom_collate, model_from_checkpoint

def plot_loss(train_losses, val_losses, figpath):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpath)
    plt.show()

def main():

    parser = argparse.ArgumentParser(description="Train ChessMLP")
    parser.add_argument("--figpath", type=str, default='figures/loss_curve.png', help="Path to save loss curve image to")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/', help="Path to checkpointfolder")
    parser.add_argument("--model", type=str, default='TwoLayerChessMLP', help="model to evaluate")
    args = parser.parse_args()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data (Extracting only the test set)
    _, _, test_dataset = get_chess_datasets("data/chessData_cleaned.parquet")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4096, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate
    )

    # 2. Load the trained model
    
    print("Loading checkpoint...")
    checkpoint = torch.load(f"{args.checkpoint}{args.model}_best_model.pth", map_location=device, weights_only=False)
    model, _ = model_from_checkpoint(checkpoint)

    criterion = nn.MSELoss()
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Evaluate Test Data
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    running_test_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 25 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} test batches...")

    avg_test_loss = running_test_loss / len(test_dataset)
    test_rmse = np.sqrt(avg_test_loss)
    
    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test MSE:  {avg_test_loss:.5f}")
    print(f"Test RMSE: {test_rmse:.5f} (~{test_rmse * 1000:.1f} centipawns)")

    # 4. Plot the Loss Curve
    train_history = checkpoint.get('train_loss_history', [])
    val_history = checkpoint.get('val_loss_history', [])
    
    if train_history and val_history:
        print("\nGenerating loss curve graph...")
        plot_loss(train_history, val_history, args.figpath)
    else:
        print("\nNo loss history found in checkpoint to plot.")

if __name__ == '__main__':
    main()