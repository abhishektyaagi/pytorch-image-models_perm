import torch
import torch.nn as nn

class RowColSparsityLoss(nn.Module):
    """
    Encourages each row and column in a weight matrix to have an L1 norm
    near some target. This is *one* way to softly push row/col distributions
    of non-zero parameters.

    row_target: desired L1 norm for each row
    col_target: desired L1 norm for each column
    row_weight, col_weight: weighting factors controlling the penalty strength
    """
    def __init__(self, row_target, col_target, row_weight=1.0, col_weight=1.0):
        super().__init__()
        self.row_target = row_target
        self.col_target = col_target
        self.rw = row_weight
        self.cw = col_weight

    def forward(self, W):
        """
        W: 2D weight tensor (out_features x in_features)
        """
        # sum of absolute values across rows => shape [out_features]
        row_l1 = W.abs().sum(dim=1)
        # sum of absolute values across columns => shape [in_features]
        col_l1 = W.abs().sum(dim=0)

        # MSE penalty: (row_l1 - row_target)^2
        # (We do MSE vs row_target, but many other choices are possible)
        row_penalty = ((row_l1 - self.row_target)**2).mean()
        col_penalty = ((col_l1 - self.col_target)**2).mean()

        return self.rw * row_penalty + self.cw * col_penalty
