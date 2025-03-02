#Take a trained model and calculate the total number of non-zero weights. Calculate the overall sparsity

import torch
import torch.nn as nn
import numpy as np
import pdb

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

if __name__ == "__main__":
    checkpoint = torch.load("/localdisk/Abhishek/pytorch-image-models_perm/output/train/deit-tiny-diagI100_patchLinear_maskInit_0.80/model_best.pth.tar")
    state_dict = checkpoint['state_dict']
    calcNNZ(state_dict)
