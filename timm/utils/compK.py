import torch
import numpy as np

# Let's call this function to compute k for each matrix
    def compute_k_for_param(mask_shape,sparsity):
        # shape is (out_features, in_features)
        # You might do something like:
        M, N = shape

        num_rows, num_cols = mask_shape
        # Diagonal length is the smaller dimension:
        diagLen = num_cols if num_rows >= num_cols else num_rows

        # Calculate total nonzero elements desired.
        elemCount = int((1 - sparsity) * num_rows * num_cols)
        # Number of full diagonals we need to cover elemCount:
        numDiag = elemCount // diagLen + 1

        return numDiag