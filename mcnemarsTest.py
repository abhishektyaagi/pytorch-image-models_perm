#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import numpy as np
import os

# If you need AutoShuffleLinear or other custom classes, import them here:
# from timm.layers import AutoShuffleLinear
from timm.models import create_model
from timm.data import create_dataset, create_loader, resolve_data_config
from timm import utils
# For McNemar's test
from statsmodels.stats.contingency_tables import mcnemar

##############################################################################
# Accuracy + McNemar Support
##############################################################################

def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy as a percentage."""
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

def mcnemar_test_on_models(modelA, modelB, loader, device='cuda', alpha=0.05):
    """
    Runs both models on the same validation loader, collects correctness patterns,
    and performs McNemar's test with significance level alpha.

    Args:
        modelA: First PyTorch model (already on `device`).
        modelB: Second PyTorch model (already on `device`).
        loader: A validation DataLoader.
        device (str): The device on which models and data reside.
        alpha (float): Significance threshold, default=0.05.

    Returns:
        A dictionary with keys:
          'statistic': McNemar’s test statistic,
          'pvalue': p-value,
          'significant': bool indicating if p-value < alpha,
          'table': 2x2 contingency table [ [n00, n01], [n10, n11] ].
    """
    # n00: both wrong, n11: both right
    # n10: A right, B wrong, n01: A wrong, B right
    n00 = n01 = n10 = n11 = 0

    modelA.eval()
    modelB.eval()

    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)

            # Model A predictions
            logitsA = modelA(input)
            predsA = torch.argmax(logitsA, dim=1)
            correctA = (predsA == target)

            # Model B predictions
            logitsB = modelB(input)
            predsB = torch.argmax(logitsB, dim=1)
            correctB = (predsB == target)

            # Move correctness to CPU for easy counting
            correctA = correctA.cpu()
            correctB = correctB.cpu()

            for ca, cb in zip(correctA, correctB):
                if ca and cb:
                    n11 += 1
                elif (not ca) and (not cb):
                    n00 += 1
                elif ca and (not cb):
                    n10 += 1
                else:
                    n01 += 1

    # Build 2x2 contingency table
    # Common arrangement: [[n00, n01], [n10, n11]]
    table = [[n00, n01],
             [n10, n11]]

    # Perform McNemar's test
    # exact=False uses the chi-square approximation,
    # exact=True uses binomial distribution (more accurate but can be slow)
    result = mcnemar(table, exact=False, correction=True)

    # Evaluate significance
    significant = (result.pvalue < alpha)
    return {
        'statistic': result.statistic,
        'pvalue': result.pvalue,
        'significant': significant,
        'table': table
    }

##############################################################################
# Argument Parsing and Main
##############################################################################

def parse_args():
    """Parse command-line arguments for McNemar's test script."""
    parser = argparse.ArgumentParser(description='McNemar’s Test on Two Models')
    # Model A
    parser.add_argument('--modelA', type=str, required=True,
                        help='Architecture name for first model (e.g., deit_tiny_patch16_224)')
    parser.add_argument('--checkpointA', type=str, required=True,
                        help='Path to first model checkpoint file')
    # Model B
    parser.add_argument('--modelB', type=str, required=True,
                        help='Architecture name for second model')
    parser.add_argument('--checkpointB', type=str, required=True,
                        help='Path to second model checkpoint file')

    # Dataset / DataLoader settings
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset type for timm (default: torch/imagenet style)')
    parser.add_argument('--val-split', type=str, default='validation',
                        help='Validation split name (default: validation)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads for data loading (default: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--channels-last', action='store_true',
                        help='Use channels_last memory format')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin memory in DataLoader')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes in the dataset (e.g., 100 for ImageNet100)')

    # Additional model creation arguments if needed (e.g. sparsity)
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help='Sparsity level if the model supports it (default: 0.0)')
    parser.add_argument('--sparsityType', type=str, default='diag',
                        help='Sparsity type if the model supports it (default: diag)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance threshold for McNemar’s test (default: 0.05)')

    return parser.parse_args()

def evaluatepval(args):

    # 1) Create the validation dataset
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.batch_size,
    )
    # Determine num_classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    elif hasattr(dataset_eval, 'classes'):
        num_classes = len(dataset_eval.classes)
    elif hasattr(dataset_eval, 'parser') and hasattr(dataset_eval.parser, 'classes'):
        num_classes = len(dataset_eval.parser.classes)
    else:
        raise ValueError("Cannot determine the number of classes from the dataset.")

    # Common extra kwargs you might have used in training
    # (You can add or remove these as needed.)
    extra_model_kwargs = {
        'pretrained': False,             # set to True if you want pretrained weights
        'in_chans': 3,                   # typical for RGB images
        'num_classes': num_classes,
        'drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'drop_block_rate': None,
        'global_pool': None,
        'bn_momentum': None,
        'bn_eps': None,
        'scriptable': False,
        'checkpoint_path': None,
        # Your custom arguments
        'sparsityType': args.sparsityType,
        'sparsity': args.sparsity,
        'n': getattr(args, 'nm_n', 4),
        'm': getattr(args, 'nm_m', 4),
        'block_size': getattr(args, 'block_size', 1),
    }

    # 2) Create models A and B
    #    If your model classes don’t accept 'sparsity' or 'sparsityType', remove them.
    # 2) Create models A and B using the same factory pattern
    modelA = create_model(args.modelA, **extra_model_kwargs)
    modelB = create_model(args.modelB, **extra_model_kwargs)

    device = torch.device(args.device)
    modelA.to(device)
    modelB.to(device)
    
    if args.channels_last:
        modelA.to(memory_format=torch.channels_last)
        modelB.to(memory_format=torch.channels_last)

    # 3) Load checkpoints
    checkpointA = torch.load(args.checkpointA, map_location='cpu')
    modelA.load_state_dict(checkpointA['state_dict'])
    checkpointB = torch.load(args.checkpointB, map_location='cpu')
    modelB.load_state_dict(checkpointB['state_dict'])

    # 4) Build the DataLoader
    data_config = resolve_data_config(vars(args), model=modelA)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # 5) Run McNemar’s test
    results = mcnemar_test_on_models(modelA, modelB, loader_eval, device, alpha=args.alpha)

    # 6) Print outcomes
    print("=== McNemar’s Test Results ===")
    print(f"Contingency Table: {results['table']}  (format: [[n00, n01],[n10,n11]])")
    print(f"Statistic        : {results['statistic']}")
    print(f"P-value          : {results['pvalue']:.6f}")
    if results['significant']:
        print(f"Models differ significantly at alpha={args.alpha}.")
    else:
        print(f"No significant difference at alpha={args.alpha}.")

def main():
    args = parse_args()

    # 1) Create the validation dataset
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.batch_size,
    )
    # Determine num_classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    elif hasattr(dataset_eval, 'classes'):
        num_classes = len(dataset_eval.classes)
    elif hasattr(dataset_eval, 'parser') and hasattr(dataset_eval.parser, 'classes'):
        num_classes = len(dataset_eval.parser.classes)
    else:
        raise ValueError("Cannot determine the number of classes from the dataset.")

    # 2) Create models A and B
    #    If your model classes don’t accept 'sparsity' or 'sparsityType', remove them.
    modelA = create_model(
        args.modelA,
        pretrained=False,
        num_classes=num_classes,
        sparsity=args.sparsity,
        sparsityType=args.sparsityType,
        in_chans=3,
    )
    modelB = create_model(
        args.modelB,
        pretrained=False,
        num_classes=num_classes,
        sparsity=args.sparsity,
        sparsityType=args.sparsityType,
        in_chans=3,
    )
    device = torch.device(args.device)
    modelA.to(device)
    modelB.to(device)

    if args.channels_last:
        modelA.to(memory_format=torch.channels_last)
        modelB.to(memory_format=torch.channels_last)

    # 3) Load checkpoints
    checkpointA = torch.load(args.checkpointA, map_location='cpu')
    modelA.load_state_dict(checkpointA['state_dict'])
    checkpointB = torch.load(args.checkpointB, map_location='cpu')
    modelB.load_state_dict(checkpointB['state_dict'])

    # 4) Build the DataLoader
    data_config = resolve_data_config(vars(args), model=modelA)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # 5) Run McNemar’s test
    results = mcnemar_test_on_models(modelA, modelB, loader_eval, device, alpha=args.alpha)

    # 6) Print outcomes
    print("=== McNemar’s Test Results ===")
    print(f"Contingency Table: {results['table']}  (format: [[n00, n01],[n10,n11]])")
    print(f"Statistic        : {results['statistic']}")
    print(f"P-value          : {results['pvalue']:.6f}")
    if results['significant']:
        print(f"Models differ significantly at alpha={args.alpha}.")
    else:
        print(f"No significant difference at alpha={args.alpha}.")

if __name__ == '__main__':
    main()
