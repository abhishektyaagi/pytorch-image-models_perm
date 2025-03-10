import torch
import numpy as numpy
import torch
import argparse
from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.layers import AutoShuffleLinear
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm import utils

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to evaluate (e.g., deit_tiny_patch16_224)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--dataset', type=str, default='', help='Dataset type (default: torch/imagenet)')
    parser.add_argument('--val-split', type=str, default='validation', help='Dataset validation split (default: validation)')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for evaluation (default: 512)')
    parser.add_argument('--workers', type=int, default=16, help='Number of data loader workers (default: 16)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--channels-last', action='store_true', help='Use channels_last memory format')
    parser.add_argument('--pin-mem', action='store_true', help='Pin memory in DataLoader')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes in the dataset (e.g., 100 for ImageNet100)')
    # Arguments from training script that might affect model creation
    parser.add_argument('--sparsity', type=float, default=0.97, help='Sparsity level (default: 0.97)')
    parser.add_argument('--sparsityType', type=str, default='diag', help='Sparsity type (default: diag)')
    args = parser.parse_args()
    return args

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

# Define accuracy if not imported from timm.utils
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

def project_to_permutation(M):
    # simple row-argmax approach
    with torch.no_grad():
        M_rounded = torch.zeros_like(M)
        row_indices = torch.argmax(M, dim=1)
        for i, j in enumerate(row_indices):
            M_rounded[i, j] = 1
    return M_rounded

def calcNNZLinears(model):
    total_params = 0
    total_nnz = 0

    for name, param in model.named_parameters():
        if 'weight' in name:  # e.g., 'proj.weight', 'fc.weight', etc.
            w = param.detach().cpu().numpy()
            nnz = np.count_nonzero(w)
            size = w.size
            total_params += size
            total_nnz += nnz
            print(f"{name}: {nnz} / {size} non-zero ({nnz/size:.2%})")

    print(f"Total non-zero weights: {total_nnz} / {total_params} ({total_nnz/total_params:.2%})")
    return total_nnz, total_params

def analyze_permutation_matrix(rounded_matrix, layer_name):
    """
    Analyze the number of 1s in each row and column of a rounded permutation matrix
    and print the results to the terminal.
    
    Args:
        rounded_matrix (torch.Tensor): The rounded permutation matrix (0s and 1s).
        layer_name (str): Name of the layer for identification in output.
    """
    # Number of rows and columns (assuming square matrix)
    N = rounded_matrix.shape[1]
    
    # Compute number of 1s per row (should be 1 for all rows)
    row_sums = torch.sum(rounded_matrix, dim=1).int()
    
    # Compute number of 1s per column
    col_sums = torch.sum(rounded_matrix, dim=0).int()
    
    # Histogram of column sums (how many columns have 0, 1, 2, ... 1s)
    hist = torch.bincount(col_sums)
    
    # Extract statistics
    zero_ones = hist[0].item() if len(hist) > 0 else 0
    exact_one = hist[1].item() if len(hist) > 1 else 0
    more_than_one = sum(hist[2:]).item() if len(hist) > 2 else 0
    max_ones = col_sums.max().item()
    
    # Verify row sums (optional, for debugging)
    if not torch.all(row_sums == 1):
        print(f"Warning: Layer {layer_name} has rows with != 1 ones, which is unexpected.")
    
    # Print statistics
    print(f"Layer {layer_name}:")
    print(f"  Total columns: {N}")
    print(f"  Columns with 0 ones: {zero_ones}")
    print(f"  Columns with exactly 1 one: {exact_one}")
    print(f"  Columns with >1 ones: {more_than_one}")
    print(f"  Maximum ones in any column: {max_ones}")


def main():
    args = parse_args()

    # Step 1: Create the validation dataset to infer num_classes
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.batch_size,
    )

    if args.num_classes is not None:
        num_classes = args.num_classes
    elif hasattr(dataset_eval, 'classes'):
        num_classes = len(dataset_eval.classes)
    elif hasattr(dataset_eval, 'parser') and hasattr(dataset_eval.parser, 'classes'):
        num_classes = len(dataset_eval.parser.classes)
    else:
        raise ValueError("Cannot determine the number of classes from the dataset, and '--num-classes' was not specified.")

    # Step 2: Create the model using parameters consistent with training
    model = create_model(
        args.model,
        pretrained=False,  # We'll load the checkpoint manually
        num_classes=num_classes,
        sparsity=args.sparsity,
        sparsityType=args.sparsityType,
        in_chans=3,  # Assuming RGB input as in training
        drop_rate=0.0,  # Default from timm, adjust if specified in training
        drop_path_rate=None,  # Adjust if used in training
        global_pool=None,  # Use model default unless specified
    )
    model.eval()

    # Move model to device and apply channels_last if specified
    device = torch.device(args.device)
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # Step 3: Load the state dict from the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # Step 5: Resolve data configuration and create the data loader
    data_config = resolve_data_config(vars(args), model=model)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=False,  # Simplify for evaluation
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    val_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    metrics = validate(model, loader_eval, val_loss_fn, device)
    print("Validation results without permutations rounded:", metrics)


    # Step 4: Round the permutations for AutoShuffleLinear layers
    with torch.no_grad():
        for name, module in model.named_modules():  # Use named_modules() to get layer names
            if isinstance(module, AutoShuffleLinear):
                import pdb
                if hasattr(module, 'P_left'):
                    #pdb.set_trace()
                    left_rounded = project_to_permutation(module.P_left)
                    module.P_left.data.copy_(left_rounded)
                    # Analyze and print statistics
                    analyze_permutation_matrix(left_rounded, name)


    # Step 6: Define loss function and validate
    val_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    metrics = validate(model, loader_eval, val_loss_fn, device)
    print("Validation results with permutations rounded:", metrics)

if __name__ == '__main__':
    main()