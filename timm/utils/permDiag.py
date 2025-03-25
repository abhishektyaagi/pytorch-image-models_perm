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


def permStruc(mask, device='cuda', permute_rows=True, permute_cols=True):
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

def get_mask_one_block_torch(mask_shape, start_row, start_col, block_size, device='cuda'):
    """
    Computes a mask for a matrix where a square block of size `block_size` x `block_size`
    starting at position (start_row, start_col) is nonzero.
    """
    # Initialize mask as all False
    mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    
    # Calculate rows and columns for the block
    rows = torch.arange(start_row, start_row + block_size, device=device)
    cols = torch.arange(start_col, start_col + block_size, device=device)
    
    # Create a block by setting elements at these positions to True
    row_indices, col_indices = torch.meshgrid(rows, cols, indexing='ij')
    mask[row_indices, col_indices] = True

    return mask

import torch

def get_mask_nm_torch(mask_shape, n, m, device='cuda'):
    """
    For each row, break it into segments of length m. In each segment,
    choose n random elements to be True. If the last segment is shorter
    than m, then pick a proportionally smaller number of elements to be True.

    Args:
        mask_shape (tuple): (num_rows, num_cols).
        n (int): Number of non-zero (True) elements per segment of length m.
                 For the final shorter segment, a proportional number is used.
        m (int): Length of the segment to consider per chunk in a row.
        device (str): Device to place the generated mask on (default 'cuda').

    Returns:
        torch.Tensor: A boolean mask of shape (num_rows, num_cols).
    """
    num_rows, num_cols = mask_shape
    final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)

    for row in range(num_rows):
        col_start = 0
        while col_start < num_cols:
            # Determine the chunk size for this segment
            segment_size = min(m, num_cols - col_start)

            # Calculate how many indices to pick in this segment
            # e.g., proportionally scale n for smaller segments
            # and clamp so it never exceeds segment_size
            sub_n = int(round(segment_size * n / m))
            sub_n = min(sub_n, segment_size)

            if sub_n > 0:
                # Pick sub_n random indices from this segment
                indices = torch.randperm(segment_size, device=device)[:sub_n]
                final_mask[row, col_start + indices] = True

            col_start += segment_size

    return final_mask


def get_mask_block_torch(mask_shape, sparsity, block_size, device='cuda'):
    """
    Computes a mask for a matrix where a specified fraction (1 - sparsity) of elements 
    are nonzero using square blocks of size `block_size` x `block_size`.
    
    Valid starting positions ensure that blocks fit entirely within the matrix boundaries.
    """
    num_rows, num_cols = mask_shape
    diagLen = num_cols if num_rows >= num_cols else num_rows

    # Calculate total nonzero elements desired.
    elemCount = int((1 - sparsity) * num_rows * num_cols)
    numDiag = elemCount // diagLen + 1
    elemCount = numDiag * diagLen
    # Number of full blocks needed (each block has block_size^2 elements)
    numBlocks = elemCount // (block_size * block_size) + 1
    #print(f"Number of blocks: {numBlocks}")
    
    # Calculate valid starting positions (ensuring blocks don't go out of bounds)
    valid_row_starts = num_rows - block_size + 1
    valid_col_starts = num_cols - block_size + 1
    
    # Check if there are enough valid positions
    if valid_row_starts <= 0 or valid_col_starts <= 0:
        raise ValueError(f"Block size {block_size} is too large for the mask shape {mask_shape}")
    
    # Initialize final mask as all False
    final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    
    # Generate random starting positions for blocks
    start_rows = torch.randint(0, valid_row_starts, (numBlocks,), device=device)
    start_cols = torch.randint(0, valid_col_starts, (numBlocks,), device=device)
    
    # For each chosen block starting position, get the one-block mask and combine
    for i in range(numBlocks):
        one_block = get_mask_one_block_torch(
            mask_shape, int(start_rows[i]), int(start_cols[i]), block_size, device=device
        )
        final_mask |= one_block  # logical OR to add this block

    return final_mask