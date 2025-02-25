#Visualize a weight matrix to see the structure of non-zeros

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

def visualize_linear_sparsity(model, columns=3):
    """
    For each nn.Linear in the given model, plot a heatmap showing
    which weight entries are zero vs. non-zero.

    Args:
        model: A PyTorch model.
        columns (int): Number of columns for subplot arrangement.
    """
    # Collect all linear modules with names
    linear_layers = [(name, module) for name, module in model.named_modules() 
                     if isinstance(module, nn.Linear)]
    
    # Figure out subplot layout
    n = len(linear_layers)
    rows = (n + columns - 1) // columns  # round up
    fig, axes = plt.subplots(rows, columns, figsize=(5*columns, 4*rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case it's 2D

    for i, (layer_name, layer) in enumerate(linear_layers):
        ax = axes[i]
        # Convert the weights to CPU numpy
        w = layer.weight.detach().cpu().numpy()
        
        # Build a mask: 1 = non-zero, 0 = zero
        mask = (w != 0).astype(float)
        
        # Plot the mask (aspect='auto' so tall/skinny matrices aren't too distorted)
        im = ax.imshow(mask, cmap='gray', aspect='auto')
        ax.set_title(f"{layer_name}\nshape={w.shape}")
        
        # Optionally add a colorbar
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots (if len(linear_layers) < rows*columns)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()
