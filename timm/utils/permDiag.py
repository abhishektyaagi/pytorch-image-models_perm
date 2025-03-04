import torch
import numpy as np
import random
import math
import time

#Generate permuted diagonals given an input matrix, sparsity 
def get_mask_one_diagonal_torch(mask_shape, diag_pos, experimentType="random", device='cuda'):

    # Create an array of zeros with the specified shape and boolean type
    mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    num_rows, num_cols = mask_shape

    if num_rows >= num_cols:
        # Case when there are more rows than columns
        diag_length = num_cols
        start_row = int(diag_pos)
        rows = (torch.arange(diag_length, device=device) + start_row) % num_rows
        cols = torch.arange(diag_length, device=device) % num_cols
    else:
        # Case when there are more columns than rows
        diag_length = num_rows
        start_col = int(diag_pos)
        rows = torch.arange(diag_length, device=device) % num_rows
        cols = (torch.arange(diag_length, device=device) + start_col) % num_cols

    mask[rows, cols] = True

    return mask

#Produces a mask with all the number of diagonals
def get_mask_diagonal_torch(mask_shape,sparsity, device='cuda'):
    """
    Computes a mask for a matrix of shape `mask_shape` where a specified fraction
    (1 - sparsity) of elements are nonzero. This is done by:
    
    1) Computing the number of nonzero elements as: elemCount = (1 - sparsity) * total_elems.
    2) Dividing elemCount by the diagonal length (min(num_rows, num_cols)) to get the number
       of full diagonals (numDiag) to add.
    3) Randomly choosing that many starting positions (rows if num_rows >= num_cols, otherwise columns).
    4) Creating a mask for each diagonal via get_mask_one_diagonal_torch and combining them.
    """
    num_rows, num_cols = mask_shape
    # Diagonal length is the smaller dimension:
    diagLen = num_cols if num_rows >= num_cols else num_rows

    # Calculate total nonzero elements desired.
    elemCount = int((1 - sparsity) * num_rows * num_cols)
    # Number of full diagonals we need to cover elemCount:
    print(elemCount)
    numDiag = elemCount // diagLen + 1
    print("NumDiag: ", numDiag)

    # Randomly choose starting positions from the available rows (or columns)
    if num_rows >= num_cols:
        diagPositions = torch.randperm(num_rows, device=device)[:numDiag]
    else:
        diagPositions = torch.randperm(num_cols, device=device)[:numDiag]

    # Initialize final mask as all False.
    final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    # For each chosen diagonal starting position, get the one-diagonal mask and combine.
    for diag_pos in diagPositions:
        one_diag = get_mask_one_diagonal_torch(mask_shape, int(diag_pos), device=device)
        final_mask |= one_diag  # logical OR to add this diagonal

    return final_mask

#Produce a mask with elements spread randomly in unstructured manner
def get_mask_unstructured_torch(mask_shape, sparsity, device='cuda'):
    """
    Computes a mask for a matrix of shape `mask_shape` where a specified fraction
    (1 - sparsity) of elements are nonzero. This is done by:
    
    1) Computing the number of nonzero elements as: elemCount = (1 - sparsity) * total_elems.
    2) Randomly choosing elemCount number of elements to set as True.
    """
    num_rows, num_cols = mask_shape
    diagLen = num_cols if num_rows >= num_cols else num_rows

    # Calculate total nonzero elements desired.
    elemCount = int((1 - sparsity) * num_rows * num_cols)
    numDiag = elemCount // diagLen + 1
    elemCount = numDiag * diagLen
    print(elemCount)
    
    # Randomly choose elemCount number of elements to set as True.
    mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    # Choose elemCount number of elements to set as True.
    mask.ravel()[torch.randperm(num_rows * num_cols)[:elemCount]] = True

    return mask

def generate_random_permutation_matrix(size, device='cpu'):
    """
    Generates a random permutation matrix of shape (size, size).
    """
    identity = torch.eye(size, device=device)
    permutation = identity[torch.randperm(size)]

    #Save the permutation matrix to a file
    torch.save(permutation, 'permutation.pt')

    return permutation

def generate_random_permutation_matrix_time_seed(size, device='cuda'):
    """
    Generates a random permutation matrix of shape (size, size).
    Uses the current time to seed a local RNG so that each call is unique,
    even if the global seed is fixed elsewhere.
    """
    # Create a local generator so we don't affect the global RNG
    gen = torch.Generator(device=device)

    # Use microseconds or nanoseconds for uniqueness
    # (int(time.time() * 1e6), for example)
    seed_val = int(time.time() * 1e6)
    gen.manual_seed(seed_val)

    identity = torch.eye(size, device=device)
    
    # Pass the local generator to randperm
    #permutation = identity[torch.randperm(size, generator=gen)]

    # IMPORTANT: specify the same device for randperm
    perm_indices = torch.randperm(size, generator=gen, device=device)
    permutation = identity[perm_indices]

    # Optionally save to a unique file
    #filename = f'permutation_{seed_val}.pt'
    #torch.save(permutation, filename)
    #print(f"Saved permutation to {filename} with seed {seed_val}")

    return permutation

def apply_permutation_to_mask(mask, permutation_matrix):
    """
    Applies a permutation matrix to a mask by matrix multiplication.
    """
    return torch.matmul(permutation_matrix.float(), mask.float()).bool()


def permDiag(mask, device='cuda', permute_rows=True, permute_cols=True):
    """
    Permute a rectangular mask of shape (M, N) in the row dimension,
    column dimension, or both.

    Args:
      mask: The original 2D mask (M x N).
      device: 'cpu' or 'cuda'.
      permute_rows: If True, permute the row dimension.
      permute_cols: If True, permute the column dimension.

    Returns:
      A permuted mask of shape (M, N).
    """
    M, N = mask.shape

    # Convert the mask to float for matmul, we'll cast back to bool later.
    result = mask.float().to(device)

    # --- 1) Permute rows (left-multiply) ---
    if permute_rows:
        P_rows = generate_random_permutation_matrix_time_seed(M, device=device)  # (M x M)
        result = P_rows.float().matmul(result)  # --> shape is (M, N)

    # --- 2) Permute columns (right-multiply) ---
    if permute_cols:
        P_cols = generate_random_permutation_matrix_time_seed(N, device=device)  # (N x N)
        result = result.matmul(P_cols.float())  # --> shape is (M, N)

    # Cast back to bool if your mask is boolean
    return result.bool()

#Produce a mask which abides by our sparsity pattern of permuted diagonalsz
