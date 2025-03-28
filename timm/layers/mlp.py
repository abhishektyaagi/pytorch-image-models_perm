""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial

from torch import nn as nn
import torch.nn.functional as F
from .grn import GlobalResponseNorm
from .helpers import to_2tuple
from timm.utils import permDiag
import torch
from timm.utils.permDiag import get_mask_diagonal_torch, permStruc, get_mask_unstructured_torch, get_mask_block_torch, get_mask_nm_torch
#from timm.loss.autoshuffle_loss import l1_l2_penalty, threshold_and_normalize

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

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MaskedLinear(nn.Module):
    """
    A linear layer that multiplies its weight by a (fixed) binary mask each forward pass.
    """
    def __init__(self, in_features, out_features, bias=True, sparsity=0.8, sparsityType='random', n=2,m=4,block_size=2 ,device='cuda'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        print("In mask linear, sparsity type: ",sparsityType, " sparsity: ",sparsity)
        # Build your diagonal mask
        #diag_mask = get_mask_diagonal_torch((out_features, in_features), sparsity, device=device)
        #diag_mask = permDiag(diag_mask, device=device)
        if sparsityType == 'random':
            print("Random mask")
            diag_mask = get_mask_unstructured_torch((out_features, in_features), sparsity, device=device)
        elif sparsityType == 'diag':
            print("Diagonal mask")
            diag_mask = get_mask_diagonal_torch((out_features, in_features), sparsity, device=device)
        elif sparsityType == 'permDiag':
            print("Permuted Diagonal mask")
            diag_mask = get_mask_diagonal_torch((out_features, in_features), sparsity, device=device)
            diag_mask = permStruc(diag_mask, device=device)
        elif sparsityType == 'km':
            print("NM mask, with n,m: ",n,m)
            diag_mask = get_mask_nm_torch((out_features, in_features), sparsity, n, m, device=device)
        elif sparsityType == 'block':
            print("Block mask")
            diag_mask = get_mask_block_torch((out_features, in_features), sparsity, block_size=block_size, device=device)
        elif sparsityType == 'permkm':
            print("Permuted NM mask, with n,m: ",n,m)
            diag_mask = get_mask_nm_torch((out_features, in_features), sparsity, n, m, device=device)
            diag_mask = permStruc(diag_mask, device=device)
        elif sparsityType == 'permBlock':
            print("Permuted Block mask")
            diag_mask = get_mask_block_torch((out_features, in_features), sparsity, block_size=block_size, device=device)
            diag_mask = permStruc(diag_mask, device=device)
        else:
            raise ValueError('Invalid sparsityType')
       
        diag_mask = diag_mask.to(self.linear.weight.device)

        # Register the final mask as a buffer so that it does not update with gradients
        self.register_buffer('mask', diag_mask)

        # Permanently apply the mask on initialization
        with torch.no_grad():
            self.linear.weight.data.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the mask to the weight on-the-fly
        w = self.linear.weight * self.mask
        return F.linear(x, w, self.linear.bias)
    
    def apply_mask(self):
        """Reapply the mask to ensure the zeroed weights remain zero."""
        with torch.no_grad():
            self.linear.weight.data.mul_(self.mask)

class MaskedMLP(nn.Module):
    """
    Two-layer MLP with MaskedLinear. 
    """
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        drop=0., 
        sparsity=0.8, 
        bias=True,
        sparsityType='random',
        n=2,
        m=4,
        block_size=2,
        device='cuda'
    ):
        super().__init__()
        print("Chosen sparsity type: ",sparsityType)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = MaskedLinear(in_features, hidden_features, bias=True, sparsityType = sparsityType, sparsity=sparsity, n=n, m=m, block_size=block_size, device=device)
        self.act = act_layer()
        self.fc2 = MaskedLinear(hidden_features, out_features, bias=True, sparsityType = sparsityType, sparsity=sparsity, n=n, m=m, block_size=block_size, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AutoShuffleLinear(nn.Module):
    """
    A linear layer W' = P_left * W * P_right, where P_left, P_right are
    trainable NxN matrices that we relax to doubly-stochastic with L1-L2 penalty.

    If out_features = O, in_features = I, then:
      - P_left  is shape (O, O)
      - P_right is shape (I, I)
      - W       is shape (O, I)
    """
    def __init__(self, in_features, out_features, bias=True, sparsity=0.8, sparsityType='random', n=2, m=4, block_size=2, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.bias = self.linear.bias 

        if sparsityType == 'random':
            diag_mask = get_mask_unstructured_torch((out_features, in_features), sparsity, device=device)
        elif sparsityType == 'diag':
            diag_mask = get_mask_diagonal_torch((out_features, in_features), sparsity, device=device)
        elif sparsityType == 'permDiag':
            diag_mask = get_mask_diagonal_torch((out_features, in_features), sparsity, device=device)
            diag_mask = permStruc(diag_mask, device=device)
        elif sparsityType == 'km':
            diag_mask = get_mask_nm_torch((out_features, in_features), sparsity, n, m, device=device)
        elif sparsityType == 'block':
            diag_mask = get_mask_block_torch((out_features, in_features), sparsity, block_size=block_size, device=device)
        elif sparsityType == 'permkm':
            diag_mask = get_mask_nm_torch((out_features, in_features), sparsity, n, m, device=device)
            diag_mask = permStruc(diag_mask, device=device)
        elif sparsityType == 'permBlock':
            diag_mask = get_mask_block_torch((out_features, in_features), sparsity, block_size=block_size, device=device)
            diag_mask = permStruc(diag_mask, device=device)
        else:
            raise ValueError('Invalid sparsityType')
       
        diag_mask = diag_mask.to(self.linear.weight.device)

        # Register the final mask as a buffer so that it does not update with gradients
        self.register_buffer('mask', diag_mask)

        # Permanently apply the mask on initialization
        with torch.no_grad():
            self.linear.weight.data.mul_(self.mask)

        # The normal "base" weight
        #self.weight = nn.Parameter(torch.empty(out_features, in_features))
        #nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # or any init you like

        # Optional bias
        #if bias:
        #    self.bias = nn.Parameter(torch.zeros(out_features))
        #else:
        #    self.bias = None

        # Initialize the P_left and P_right as random nonnegative so we can clamp/normalize
        P_left_init = torch.rand(out_features, out_features)
        #P_right_init = torch.rand(in_features, in_features)

        # Make them nn.Parameters so they're trained
        self.P_left = nn.Parameter(P_left_init)
        #self.P_right = nn.Parameter(P_right_init)

    def forward(self, x):
        # The relaxed shuffle weight:
        #   w' = P_left @ weight
        w = self.linear.weight * self.mask
        w_dash = self.P_left @ w
        return F.linear(x, w_dash, self.bias)

    def post_update_clamp_and_norm(self):
        """
        Call this after each optimizer step to clamp negative entries
        and do row/column normalization for both P_left and P_right.
        """
        threshold_and_normalize(self.P_left)
        #threshold_and_normalize(self.P_right)
    
    def apply_mask(self):
        """Reapply the mask to ensure the zeroed weights remain zero."""
        with torch.no_grad():
            self.linear.weight.data.mul_(self.mask)

class AutoShuffleMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        sparsity=0.8, 
        bias=True,
        sparsityType='random',
        drop=0.,
        n=2,
        m=4,
        block_size=2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = AutoShuffleLinear(in_features, hidden_features, bias=True, sparsity=sparsity, sparsityType=sparsityType, n=n, m=m, block_size=block_size)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = AutoShuffleLinear(hidden_features, out_features, bias=True, sparsity=sparsity, sparsityType=sparsityType, n=n, m=m, block_size=block_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def post_update_clamp_and_norm(self):
        """
        Convenience method to clamp/normalize after each optimizer.step().
        """
        self.fc1.post_update_clamp_and_norm()
        self.fc2.post_update_clamp_and_norm()

def auto_shuffle_penalty(module: AutoShuffleLinear) -> torch.Tensor:
    """
    Compute the L1-L2 penalty for P_left + P_right.
    """
    return l1_l2_penalty(module.P_left) + l1_l2_penalty(module.P_right)

class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1.bias is not None:
            nn.init.ones_(self.fc1.bias[self.fc1.bias.shape[0] // 2:])
        nn.init.normal_(self.fc1.weight[self.fc1.weight.shape[0] // 2:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nn.SiLU, gate_last=False)


class SwiGLU(nn.Module):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            gate_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims (for 2D NCHW tensors)
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GlobalResponseNormMlp(nn.Module):
    """ MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d

    NOTE: Intended for '2D' NCHW (use_conv=True) or NHWC (use_conv=False, channels-last) tensor layouts
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.grn = GlobalResponseNorm(hidden_features, channels_last=not use_conv)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
