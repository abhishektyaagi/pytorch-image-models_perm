#Take a trained model and calculate the total number of non-zero weights. Calculate the overall sparsity

import torch
import torch.nn as nn
import numpy as np
import pdb

def threshold_linear_weights_state_dict(state_dict, threshold=1e-3):
    """
    Iterate over all items in the state dict, and for any weight tensor
    with 2 dimensions (assumed to be from a linear layer), set entries
    with absolute value below `threshold` to zero.
    """
    for key, weight in state_dict.items():
        # We assume a 2D weight is from a linear layer.
        if "weight" in key and weight.ndim == 2:
            # Clone to avoid modifying in-place if needed
            w = weight.clone()
            w[w.abs() < threshold] = 0.0
            state_dict[key] = w
    return state_dict

def calcNNZ(state_dict):
    total_params = 0
    total_nnz = 0

    for name, weight in state_dict.items():
        # Convert weight tensor to numpy if it's a tensor
        if torch.is_tensor(weight):
            w = weight.detach().cpu().numpy()
            mask = (w != 0).astype(float)
            nnz = np.sum(mask)
            total_params += w.size
            total_nnz += nnz
            print(f"{name}: {nnz} / {w.size} non-zero ({nnz / w.size:.2%})")
    
    print(f"Total non-zero weights: {total_nnz} / {total_params} ({total_nnz / total_params:.2%})")
    return total_nnz, total_params

def compare_nonzero_locations(state_dict1, state_dict2):
    """
    For every tensor in the state dicts, compare the locations of non-zero entries.
    Prints a summary per tensor.
    """
    for key in state_dict1.keys():
        weight1 = state_dict1[key]
        weight2 = state_dict2[key]
        if torch.is_tensor(weight1) and torch.is_tensor(weight2):
            w1 = weight1.detach().cpu().numpy()
            w2 = weight2.detach().cpu().numpy()
            
            # Create boolean masks: True where nonzero, False otherwise.
            mask1 = (w1 != 0)
            mask2 = (w2 != 0)
            
            # Check if the nonzero pattern is exactly the same.
            if np.array_equal(mask1, mask2):
                print(f"{key}: Non-zero pattern is exactly matching.")
            else:
                # Count nonzeros and common nonzeros.
                nonzeros1 = np.sum(mask1)
                nonzeros2 = np.sum(mask2)
                common_nonzeros = np.sum(np.logical_and(mask1, mask2))
                print(f"{key}:")
                print(f"  Model1 non-zeros: {nonzeros1} / {w1.size} ({nonzeros1 / w1.size:.2%})")
                print(f"  Model2 non-zeros: {nonzeros2} / {w2.size} ({nonzeros2 / w2.size:.2%})")
                print(f"  Common non-zero locations: {common_nonzeros} / {w1.size} ({common_nonzeros / w1.size:.2%})")
                if nonzeros1 > 0:
                    print(f"  Matching ratio relative to Model1: {common_nonzeros / nonzeros1:.2%}")
            print()

if __name__ == "__main__":
    checkpoint = torch.load("/localdisk/Abhishek/pytorch-image-models_perm/output/train/deit-tiny-permDiagI100_patchLinear_maskPerm_0.95_1/model_best.pth.tar")
    state_dict = checkpoint['state_dict']
    #state_dict = threshold_linear_weights_state_dict(state_dict, threshold=1e-1)
    #calcNNZ(state_dict)

    checkpoint2 = torch.load("/localdisk/Abhishek/pytorch-image-models_perm/output/train/deit-tiny-permDiagI100_patchLinear_maskPerm_0.95_2/model_best.pth.tar")
    state_dict2 = checkpoint2['state_dict']

    #Compare the two models to see if they have non-zeros in the same locations
    compare_nonzero_locations(state_dict, state_dict2)