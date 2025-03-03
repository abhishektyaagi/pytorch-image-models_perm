import torch
import torch.nn as nn

class KNonzeroRowColPenalty(nn.Module):
    """
    Encourages each row or column (or both) in a weight matrix W to have
    an L1 norm close to 'k'.  The shape of W determines whether we enforce
    row- or column- constraints or both:
      - MxM: apply row & column penalty => each row has L1 ~ k, each col has L1 ~ k
      - MxN with M > N: apply only column penalty => each col L1 ~ k
      - MxN with M < N: apply only row penalty => each row L1 ~ k

    This is a *soft* penalty, and does NOT guarantee exactly k nonzeros.
    You typically threshold small weights after training if needed.
    """

    def __init__(self, k: float, row_weight=1.0, col_weight=1.0):
        """
        Args:
            k: desired L1 norm for each relevant row/column
            row_weight, col_weight: penalty multipliers
        """
        super().__init__()
        self.k = k
        self.rw = row_weight
        self.cw = col_weight

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        W has shape (out_features, in_features) => (M, N).
        We assume M = number of rows, N = number of columns.
        """
        M, N = W.shape

        # sum of absolute values across each row => shape [M]
        row_l1 = W.abs().sum(dim=1)
        # sum of absolute values across each column => shape [N]
        col_l1 = W.abs().sum(dim=0)

        # We'll create 0-penalties so that we only enforce
        # row or column constraints if needed:
        row_pen = torch.tensor(0., device=W.device, dtype=W.dtype)
        col_pen = torch.tensor(0., device=W.device, dtype=W.dtype)

        if M == N:
            # Square matrix => enforce both row & column
            row_pen = ((row_l1 - self.k)**2).mean()
            col_pen = ((col_l1 - self.k)**2).mean()
        elif M > N:
            # More rows than columns => only enforce column constraint
            col_pen = ((col_l1 - self.k)**2).mean()
        else:
            # M < N => only enforce row constraint
            row_pen = ((row_l1 - self.k)**2).mean()

        return self.rw * row_pen + self.cw * col_pen
