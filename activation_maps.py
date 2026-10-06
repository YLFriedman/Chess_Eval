import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import batch_fen_to_tensor, state_from_checkpoint

def plot_feature_maps(plot_type='bottleneck'):
    """
    Generates feature map visualizations for the CNN.
    
    Parameters:
    plot_type (str): 'bottleneck' (plots all 64 final channels) or 
                     'hierarchical' (plots top 8 channels across all 4 conv blocks)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_path = 'figures/heat_maps/'
    
    # Load the trained CNN model
    checkpoint = torch.load("checkpoints/ChessCNN_best_model.pth", map_location=device, weights_only=False)
    state = state_from_checkpoint(checkpoint, device)
    model = state.model
    model.eval()

    # Prepare the FEN with dummy castling/en passant markers
    fen = "r1bqkb1r/ppp2Npp/2n5/3np3/2B5/8/PPPP1PPP/RNBQK2R b - -"
    tensor_input = batch_fen_to_tensor([fen]).to(device)

    if plot_type == 'hierarchical':
        activations = {}

        def get_activation(layer_name):
            def hook(model, input, output):
                activations[layer_name] = output.detach().squeeze(0).cpu().numpy()
            return hook

        # Register hooks to the ReLUs in the architecture
        model.conv_stack[2].register_forward_hook(get_activation('Block 1 (64)'))
        model.conv_stack[5].register_forward_hook(get_activation('Block 2 (128)'))
        model.conv_stack[8].register_forward_hook(get_activation('Block 3 (256)'))
        model.conv_stack[11].register_forward_hook(get_activation('Bottleneck (64)'))

        with torch.no_grad():
            _ = model(tensor_input)

        fig, axes = plt.subplots(4, 8, figsize=(16, 9))
        fig.suptitle(
            "Hierarchical Feature Map Evolution (Top 8 Active Channels per Block)\n"
            "Position: r1bqkb1r/ppp2Npp/2n5/3np3/2B5/8/PPPP1PPP/RNBQK2R b", 
            fontsize=14, fontweight='bold', y=0.98
        )

        for row_idx, (layer_name, feature_maps) in enumerate(activations.items()):
            # Calculate the mean activation for each channel
            channel_means = feature_maps.mean(axis=(1, 2))
            top_8_indices = np.argsort(channel_means)[-8:][::-1]
            
            for col_idx, channel_idx in enumerate(top_8_indices):
                ax = axes[row_idx, col_idx]
                heatmap = feature_maps[channel_idx]
                
                ax.imshow(heatmap, cmap='magma', interpolation='nearest')
                
                if col_idx == 0:
                    ax.set_ylabel(layer_name, fontsize=12, fontweight='bold')
                    
                ax.set_title(f"Ch {channel_idx}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        save_path += 'hierarchical_activations.png'
        
    elif plot_type == 'bottleneck':
        # Execute the forward pass up to the MLP flattening step
        with torch.no_grad():
            feature_maps = model.conv_stack(tensor_input)

        activations = feature_maps.squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(8, 8, figsize=(16, 16))
        fig.suptitle(
            "64-Channel 1x1 Bottleneck Feature Maps\n"
            "Position: Fried Liver", 
            fontsize=16, fontweight='bold', y=0.95
        )

        for i, ax in enumerate(axes.flat):
            if i < 64:
                ax.imshow(activations[i], cmap='viridis', interpolation='nearest')
                ax.set_title(f"Ch {i}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        save_path += 'bottleneck_64_activations.png'
        
    else:
        raise ValueError("Invalid plot_type. Please choose 'hierarchical' or 'bottleneck'.")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # Toggle 'plot_type' here to switch between visualization modes
    plot_feature_maps(plot_type='bottleneck')