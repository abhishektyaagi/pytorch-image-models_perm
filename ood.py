#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from timm.models import create_model
#from timm.layers import AutoShuffleLinear
from torchvision import datasets, transforms

##############################################################################
# Utility Functions
##############################################################################

def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def validate(model, loader, loss_fn, device):
    """Validate the model on the given loader and return loss and accuracy metrics."""
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            total_loss += loss.item() * input.size(0)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_correct_top1 += (acc1 / 100) * input.size(0)
            total_correct_top5 += (acc5 / 100) * input.size(0)
            total_samples += input.size(0)
    
    avg_loss = total_loss / total_samples
    avg_top1 = (total_correct_top1 / total_samples) * 100
    avg_top5 = (total_correct_top5 / total_samples) * 100
    return {'loss': avg_loss, 'top1': avg_top1, 'top5': avg_top5}

def project_to_permutation(M):
    """
    Rounds a real-valued matrix M to a permutation matrix by 
    placing a 1 at the argmax of each row and 0 elsewhere.
    """
    with torch.no_grad():
        M_rounded = torch.zeros_like(M)
        row_indices = torch.argmax(M, dim=1)
        for i, j in enumerate(row_indices):
            M_rounded[i, j] = 1
    return M_rounded

def analyze_permutation_matrix(rounded_matrix, layer_name):
    """
    Analyze the number of 1s in each row and column of a rounded permutation matrix
    and print the results to the terminal.
    """
    rows, cols = rounded_matrix.shape
    row_sums = torch.sum(rounded_matrix, dim=1).int()
    col_sums = torch.sum(rounded_matrix, dim=0).int()
    hist = torch.bincount(col_sums)
    zero_ones = hist[0].item() if len(hist) > 0 else 0
    exact_one = hist[1].item() if len(hist) > 1 else 0
    more_than_one = sum(hist[2:]).item() if len(hist) > 2 else 0
    max_ones = col_sums.max().item()
    
    if not torch.all(row_sums == 1):
        print(f"Warning: Layer {layer_name} has rows with != 1 ones, which is unexpected.")
    
    print(f"Layer {layer_name}:")
    print(f"  Matrix shape: {rows} x {cols}")
    print(f"  Columns with 0 ones: {zero_ones}")
    print(f"  Columns with exactly 1 one: {exact_one}")
    print(f"  Columns with >1 ones: {more_than_one}")
    print(f"  Maximum ones in any column: {max_ones}")

##############################################################################
# Core Evaluation Function
##############################################################################

def parse_args():
    """Parse command-line arguments for OOD evaluation on CIFAR-10."""
    parser = argparse.ArgumentParser(description='OOD Evaluation on CIFAR-10')
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture name, e.g., deit_tiny_patch16_224')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to CIFAR-10 root directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation (default: 128)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of DataLoader workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--channels-last', action='store_true',
                        help='Use channels_last memory format')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin memory in DataLoader')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes the model was trained on.')
    parser.add_argument('--sparsity', type=float, default=0.97,
                        help='Sparsity level used during training (default: 0.97)')
    parser.add_argument('--sparsityType', type=str, default='diag',
                        help='Sparsity type used during training (default: diag)')
    parser.add_argument('--round-permutations', action='store_true',
                        help='Whether to round any AutoShuffleLinear permutations before evaluating.')
    parser.add_argument('--nm_n', type=int, default=4, metavar='N', help='n for N:M pattern')
    parser.add_argument('--nm_m', type=int, default=4, metavar='M', help='m for N:M pattern')
    parser.add_argument('--block_size', type=int, default=4, metavar='N', help='block size for block sparsity pattern')

    return parser.parse_args()

def evaluate_ood(args):
    """
    Evaluate the out-of-distribution performance on CIFAR-10 using the given arguments.
    Returns a dictionary with the metrics: loss, top1, top5.
    """
    # Prepare extra keyword arguments (if any)
    extra_kwargs = {}
    # Map the internal kwarg names to the args Namespace names
    mapping = {
        "sparsity": "sparsity",
        "sparsityType": "sparsityType",
        "n": "nm_n",
        "m": "nm_m",
        "block_size": "block_size",
    }
    for kw, arg_name in mapping.items():
        if hasattr(args, arg_name):
            extra_kwargs[kw] = getattr(args, arg_name)

    # Try to create the model with extra kwargs.
    # If the model doesn't accept them, catch the error and create without them.
    try:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            in_chans=3,
            **extra_kwargs
        )
    except TypeError as e:
        print("Warning: Model does not accept extra keyword arguments:", e)
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            in_chans=3,
        )
    model.eval()
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Send model to device
    device = torch.device(args.device)
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # Create CIFAR-10 dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize(224),      # adjust if model needs a different size
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    cifar10_val = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=True
    )

    loader_eval = torch.utils.data.DataLoader(
        cifar10_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
    )

    # Optionally round any AutoShuffleLinear layers
    if args.round_permutations:
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, AutoShuffleLinear):
                    if hasattr(module, 'P_left'):
                        left_rounded = project_to_permutation(module.P_left)
                        module.P_left.data.copy_(left_rounded)
                        analyze_permutation_matrix(left_rounded, name)

    # Evaluate on CIFAR-10 (OOD performance)
    val_loss_fn = nn.CrossEntropyLoss().to(device)
    metrics = validate(model, loader_eval, val_loss_fn, device)
    
    return metrics

##############################################################################
# Main entry point (for command-line use)
##############################################################################

def main():
    args = parse_args()
    metrics = evaluate_ood(args)
    print("====== OOD Evaluation on CIFAR-10 ======")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Top-1 Accuracy: {metrics['top1']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5']:.2f}%")
    return metrics

if __name__ == '__main__':
    main()
