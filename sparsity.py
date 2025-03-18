import csv
import torch

def get_layerwise_sparsity(state_dict, output_file):
    """
    Scan the state_dict for 2D weight tensors (likely from nn.Linear layers), 
    compute their sparsity, and write the results to a CSV file with columns:
    layer_name, dimensions, sparsity.

    Args:
        state_dict (OrderedDict or dict): The PyTorch state_dict containing parameter tensors.
        output_file (str): Path to a CSV file where the sparsity info will be saved.
    """
    # Open/overwrite the CSV file
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["layer_name", "dimensions", "sparsity"])

        # Iterate over key-value pairs in the state_dict
        for key, tensor in state_dict.items():
            # Check for 2D weight tensors that likely correspond to linear layers
            if key.endswith("weight") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                # Move to CPU for safe operations
                w_cpu = tensor.detach().cpu()
                total_elems = w_cpu.numel()
                num_zeros = (w_cpu == 0).sum().item()
                sparsity = num_zeros / total_elems

                writer.writerow([key, list(w_cpu.shape), f"{sparsity:.6f}"])

    print(f"Sparsity data saved to {output_file}")
