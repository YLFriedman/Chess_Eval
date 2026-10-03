import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils import state_from_checkpoint, write_record, get_data_loaders

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

    parser = argparse.ArgumentParser(description="Evaluate Chess Model")
    parser.add_argument("--figpath", type=str, default='loss_curve.png', help="file to save loss curve image to")
    parser.add_argument("--checkpoint", type=str, default='ThreeLayerMLP_best_model.pth', help="name of checkpoint file")
    parser.add_argument("--record_file", type = str, default='model_performance.xlsx', help="name of record file")
    parser.add_argument("-r", "--record", action="store_true", help="Save eval metrics to {record_file}")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data (Extracting only the test set)
    _, _, test_loader = get_data_loaders("data/chessData_cleaned.parquet")
    

    # 2. Load the trained model via State object
    print("Loading checkpoint...")
    checkpoint = torch.load(f'checkpoints/{args.checkpoint}', map_location=device, weights_only=False)
    state = state_from_checkpoint(checkpoint, device)

    criterion = nn.MSELoss()
    state.model.eval()

    # 3. Evaluate Test Data
    print(f"\nEvaluating on {len(test_loader.dataset)} test samples...")
    running_test_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = state.model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 25 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} test batches...")

    avg_test_loss = running_test_loss / len(test_loader.dataset)
    test_rmse = np.sqrt(avg_test_loss)
    
    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test MSE:  {avg_test_loss:.5f}")
    print(f"Test RMSE: {test_rmse:.5f} (~{test_rmse * 1000:.1f} centipawns)")

    # 4. Plot the Loss Curve
    train_history = state.train_history
    val_history = state.val_history
    
    if train_history and val_history:
        print("\nGenerating loss curve graph...")
        plot_loss(train_history, val_history, f'figures/{args.figpath}')
    else:
        print("\nNo loss history found in checkpoint to plot.")
    
    if args.record:
        print(f"\nSaving run metrics to {args.record_file}...")
        state.test_loss = avg_test_loss
        write_record(state, args.record_file)
        print("Record saved successfully.")

if __name__ == '__main__':
    main()