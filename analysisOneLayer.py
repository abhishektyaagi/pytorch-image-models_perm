import numpy as np
import matplotlib.pyplot as plt 
import torch
from graphQualityMetric import von_neumann_graph_entropy
import pdb
from numpy.linalg import eigh

def extract_weights(model_dict):
        # Grab the base model's fc.linear.weight
        base_weight = model_dict['base_model_state_dict']['fc.linear.weight']
        
        # Grab each permutation's final_weight
        final_weights = []
        for perm_info in model_dict['permutations']:
            final_weights.append(perm_info['final_weight'])
        
        #Get accuracies for each permutation as well
        accuracies = []
        for perm_info in model_dict['permutations']:
            accuracies.append(perm_info['final_accuracy'])

        # Return the base weight and final weights
        return base_weight, final_weights, accuracies

def von_neumann_graph_entropy(R):
    """
    Compute von Neumann entropy of a bipartite graph whose adjacency
    is given by a (m x n) rectangular matrix R. The graph has:
      - 'm' nodes in set U
      - 'n' nodes in set V
      - Edges from i in U to j in V with adjacency R[i, j]
    
    Parameters
    ----------
    R : np.ndarray, shape (m, n)
        Rectangular matrix representing bipartite edges.
    
    Returns
    -------
    float
        The von Neumann graph entropy of the resulting bipartite graph.
    """
    R = np.asarray(R, dtype=float)
    m, n = R.shape
    
    # 1) Construct the (m+n) x (m+n) block adjacency matrix A
    A = np.zeros((m+n, m+n), dtype=float)
    #   top-right block = R
    A[:m, m:] = R
    #   bottom-left block = R^T
    A[m:, :m] = R.T
    
    # 2) Degree matrix D
    d = A.sum(axis=1)  # degrees for each of the (m+n) nodes
    D = np.diag(d)
    
    # 3) Laplacian: L = D - A
    L = D - A
    
    # 4) Normalized Laplacian: L_norm = D^{-1/2} * L * D^{-1/2}
    #    Handle zeros in degree to avoid division by zero
    d_inv_sqrt = np.zeros_like(d)
    nonzero_mask = (d > 1e-15)
    d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(d[nonzero_mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    # 5) Eigen-decomposition (L_norm is symmetric => eigh)
    w, _ = eigh(L_norm)
    
    # Clip negative eigenvalues to 0 for numerical stability
    w = np.clip(w, 0, None)
    
    # 6) Von Neumann entropy: - sum_i [lambda_i log(lambda_i)]
    mask = (w > 1e-15)
    S_vN = -np.sum(w[mask] * np.log(w[mask]))
    
    return S_vN

def calcVonNeuman(path_block, path_km, path_diag):

    # Load the dictionary from .pth file
    block_dict = torch.load(path_block)
    km_dict = torch.load(path_km)
    diag_dict = torch.load(path_diag)

    #Extract the weight matrix of the base model, and extract the permutations from the dict
    # Extract for block_dict
    base_weight_block, final_weights_block, accuracies_block = extract_weights(block_dict)
    
    # Extract for km_dict
    base_weight_km, final_weights_km, accuracies_km = extract_weights(km_dict)
    
    # Extract for diag_dict
    base_weight_diag, final_weights_diag, accuracies_diag = extract_weights(diag_dict)

    #convert each weight matrix to numpy array and copy to host as well
    # For block_dict
    base_weight_block = base_weight_block.cpu().numpy()
    final_weights_block = [weight.cpu().numpy() for weight in final_weights_block]

    # For km_dict
    base_weight_km = base_weight_km.cpu().numpy()
    final_weights_km = [weight.cpu().numpy() for weight in final_weights_km]

    # For diag_dict
    base_weight_diag = base_weight_diag.cpu().numpy()
    final_weights_diag = [weight.cpu().numpy() for weight in final_weights_diag]
    #pdb.set_trace()
    # Calculate the von Neumann graph entropy for the base weight and final weights
    # For block_dict
    base_weight_block_vn = von_neumann_graph_entropy(base_weight_block)
    final_weights_block_vn = [von_neumann_graph_entropy(weight) for weight in final_weights_block]

    # For km_dict
    base_weight_km_vn = von_neumann_graph_entropy(base_weight_km)
    final_weights_km_vn = [von_neumann_graph_entropy(weight) for weight in final_weights_km]

    # For diag_dict
    base_weight_diag_vn = von_neumann_graph_entropy(base_weight_diag)
    final_weights_diag_vn = [von_neumann_graph_entropy(weight) for weight in final_weights_diag]

    #Calculate the total number of non-zero elements in the base weight and final weights
    # For block_dict
    base_weight_block_nonzero = np.count_nonzero(base_weight_block)
    final_weights_block_nonzero = [np.count_nonzero(weight) for weight in final_weights_block]

    # For km_dict
    base_weight_km_nonzero = np.count_nonzero(base_weight_km)
    final_weights_km_nonzero = [np.count_nonzero(weight) for weight in final_weights_km]
    # For diag_dict
    base_weight_diag_nonzero = np.count_nonzero(base_weight_diag)
    final_weights_diag_nonzero = [np.count_nonzero(weight) for weight in final_weights_diag]
    # Print the total number of non-zero elements

    # Print the results
    print("Block Sparsity Experiment:")
    print(f"Base Model VN: {base_weight_block_vn}")
    print("Final Weights VN (Block):")
    
    for i, vn in enumerate(final_weights_block_vn):
        print(f"Permutation {i+1}: {vn}")
    print(f"Base Model Non-zero elements: {base_weight_block_nonzero}")
    """ print("Final Weights Non-zero elements (Block):")
    for i, nonzero in enumerate(final_weights_block_nonzero):
        print(f"Permutation {i+1}: {nonzero}") """
    print("Accuracies:")
    for i, acc in enumerate(accuracies_block):
        print(f"Permutation {i+1}: {acc}")

    print("\nKM Sparsity Experiment:")
    print(f"Base Model VN: {base_weight_km_vn}")
    print("Final Weights VN (KM):")
    for i, vn in enumerate(final_weights_km_vn):
        print(f"Permutation {i+1}: {vn}")
    print(f"Base Model Non-zero elements: {base_weight_km_nonzero}")
    """ print("Final Weights Non-zero elements (KM):")
    for i, nonzero in enumerate(final_weights_km_nonzero):
        print(f"Permutation {i+1}: {nonzero}") """
    print("Accuracies:")
    for i, acc in enumerate(accuracies_km):
        print(f"Permutation {i+1}: {acc}")

    print("\nDiagonal Sparsity Experiment:")
    print(f"Base Model VN: {base_weight_diag_vn}")
    print("Final Weights VN (Diagonal):")
    for i, vn in enumerate(final_weights_diag_vn):
        print(f"Permutation {i+1}: {vn}")
    print(f"Base Model Non-zero elements: {base_weight_diag_nonzero}")
    """ print("Final Weights Non-zero elements (Diagonal):")
    for i, nonzero in enumerate(final_weights_diag_nonzero):
        print(f"Permutation {i+1}: {nonzero}") """
    print("Accuracies:")
    for i, acc in enumerate(accuracies_diag):
        print(f"Permutation {i+1}: {acc}")


    # If you want to return them (instead of just printing)
    return {
        'block': (base_weight_block, final_weights_block),
        'km': (base_weight_km, final_weights_km),
        'diag': (base_weight_diag, final_weights_diag),
    }

    #pdb.set_trace()


if __name__ == "__main__":
    
    path_block = '/localdisk/Abhishek/pytorch-image-models_perm/block_sparsity_experiment_block.pth'
    path_km = '/localdisk/Abhishek/pytorch-image-models_perm/block_sparsity_experiment_km.pth'
    path_diag = '/localdisk/Abhishek/pytorch-image-models_perm/block_sparsity_experiment_diag.pth'
    
    calcVonNeuman(path_block, path_km, path_diag)