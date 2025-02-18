import torch
import numpy as np
import random
import math

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
    numDiag = elemCount // diagLen

    # Randomly choose starting positions from the available rows (or columns)
    if num_rows >= num_cols:
        diagPositions = torch.randint(0, num_rows, (numDiag,), device=device)
    else:
        diagPositions = torch.randint(0, num_cols, (numDiag,), device=device)

    # Initialize final mask as all False.
    final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    # For each chosen diagonal starting position, get the one-diagonal mask and combine.
    for diag_pos in diagPositions:
        one_diag = get_mask_one_diagonal_torch(mask_shape, int(diag_pos), device=device)
        final_mask |= one_diag  # logical OR to add this diagonal

    return final_mask

def generate_random_permutation_matrix(size, device='cpu'):
    """
    Generates a random permutation matrix of shape (size, size).
    """
    identity = torch.eye(size, device=device)
    permutation = identity[torch.randperm(size)]

    #Save the permutation matrix to a file
    torch.save(permutation, 'permutation.pt')

    return permutation

def apply_permutation_to_mask(mask, permutation_matrix):
    """
    Applies a permutation matrix to a mask by matrix multiplication.
    """
    return torch.matmul(permutation_matrix.float(), mask.float()).bool()


#Take the diagonal matrix, and apply permutation to it
def permDiag(diagMask,device='cuda'):

    #NOTE: This works for a square matrix
    permutation = generate_random_permutation_matrix(diagMask.shape[0], device=device)

    #Apply the permutation to the mask
    mask = apply_permutation_to_mask(diagMask, permutation)

    return mask

#Produce a mask which abides by our sparsity pattern of permuted diagonals