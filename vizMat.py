import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

def visualize_linear_sparsity(state_dict, output_dir, columns=3):
    """
    For each key in the state dict that corresponds to a 2D weight tensor 
    (assumed to be from an nn.Linear layer), plot a heatmap showing
    which weight entries are zero vs. non-zero (black vs. white only),
    and save the plot in the specified directory.

    Args:
        state_dict (OrderedDict): A state_dict from a PyTorch checkpoint.
        output_dir (str): Directory where the plots will be saved.
        columns (int): Number of columns for subplot arrangement.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter out keys that end with 'weight' and have 2 dimensions (likely from nn.Linear layers)
    linear_keys = []
    for key, tensor in state_dict.items():
        if key.endswith('weight') and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            linear_keys.append(key)
    
    n = len(linear_keys)
    if n == 0:
        print("No 2D weight tensors found in state_dict.")
        return

    # Figure out subplot layout
    rows = (n + columns - 1) // columns  # round up
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case it's 2D

    # Define a two-color (black/white) colormap
    two_color_cmap = ListedColormap(["black", "white"])

    for i, key in enumerate(linear_keys):
        ax = axes[i]
        # Convert the weights to a CPU numpy array
        w = state_dict[key].detach().cpu().numpy()
        
        # Build a mask: 1 = non-zero, 0 = zero
        mask = (w != 0).astype(np.uint8)
        
        # Plot the mask with only black/white
        im = ax.imshow(
            mask,
            cmap=two_color_cmap,
            aspect='auto',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )
        ax.set_title(f"{key}\nshape={w.shape}")
        ax.axis("tight")  # Keep axes snug around data
        
        # Optionally remove axis ticks for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots (if len(linear_keys) < rows * columns)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    # Save the figure to the specified output directory
    save_path = os.path.join(output_dir, "linear_sparsity_heatmaps.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved linear sparsity heatmaps to {save_path}")
