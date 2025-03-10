import torch
import torch.nn as nn
import torch.optim as optim
from timm.layers import AutoShuffleLinear
# --------------------------------------------------------
# 1) Define the L1 - L2 penalty for a matrix M
#    P(M) = sum_i [ (sum_j |M[i,j]|) - sqrt(sum_j M[i,j]^2 ) ]
#          + sum_j [ (sum_i |M[i,j]|) - sqrt(sum_i M[i,j]^2 ) ]
# --------------------------------------------------------
def l1_l2_penalty(M: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1-L2 penalty P(M) row-wise and column-wise.

    M is assumed to be a 2D tensor of shape (N, N).
    """
    # Row-wise penalty
    #   row_l1 = sum_j |M[i,j]|
    #   row_l2 = sqrt( sum_j (M[i,j])^2 )
    #   row_penalty = sum_i [ row_l1[i] - row_l2[i] ]
    row_abs_sum = M.abs().sum(dim=1)        # shape [N]
    row_l2 = torch.sqrt((M**2).sum(dim=1))  # shape [N]
    row_penalty = (row_abs_sum - row_l2).sum()

    # Column-wise penalty
    #   col_l1 = sum_i |M[i,j]|
    #   col_l2 = sqrt( sum_i (M[i,j])^2 )
    #   col_penalty = sum_j [ col_l1[j] - col_l2[j] ]
    col_abs_sum = M.abs().sum(dim=0)        # shape [N]
    col_l2 = torch.sqrt((M**2).sum(dim=0))  # shape [N]
    col_penalty = (col_abs_sum - col_l2).sum()

    return row_penalty + col_penalty

def threshold_and_normalize(M: torch.Tensor) -> None:
    """
    - Threshold to ensure non-negative
    - Column-normalize
    - Row-normalize
    in-place on M.
    """
    with torch.no_grad():
        # clamp negatives
        M.clamp_(min=0.0)

        # column normalize
        col_sums = M.sum(dim=0, keepdim=True)
        col_sums = torch.where(col_sums == 0, torch.ones_like(col_sums), col_sums)
        M.div_(col_sums)

        # row normalize
        row_sums = M.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        M.div_(row_sums)

def auto_shuffle_penalty(module: AutoShuffleLinear) -> torch.Tensor:
    """
    Compute the L1-L2 penalty for P_left + P_right.
    """
    return l1_l2_penalty(module.P_left)# + l1_l2_penalty(module.P_right)