import numpy as np
from numpy.linalg import eigh
#from timm import utils
#from permDiag import get_mask_block_torch, get_mask_unstructured_torch, get_mask_diagonal_torch, get_mask_nm_torch
import torch
import random
import math
import time
import matplotlib.pyplot as plt
import pulp
import networkx as nx
import warnings

def all_subsets_of_vertices(vertices):
    """
    Generate all subsets (power set) of the given list 'vertices'.
    Returns each subset as a Python set.
    """
    for r in range(len(vertices)+1):
        for combo in itertools.combinations(vertices, r):
            yield set(combo)

def is_independent_set(adj_matrix, subset):
    """
    Check if 'subset' is an independent set in the graph with adjacency 'adj_matrix'.
    An independent set has no edges among any of its vertices.
    """
    for u in subset:
        for v in subset:
            if u < v and adj_matrix[u][v] == 1:
                return False
    return True

def fractional_chromatic_number(adj_matrix):
    """
    Compute the fractional chromatic number of the graph represented by 'adj_matrix'
    by building and solving a linear program.
    
    The graph here is assumed to be undirected and unweighted.
    Returns a floating-point value of chi_f(G).
    """
    n = len(adj_matrix)
    vertices = range(n)
    
    # 1) Enumerate all independent sets.
    #    For large n, this is 2^n enumeration, so only feasible for small graphs.
    independent_sets = []
    for subset in all_subsets_of_vertices(vertices):
        if is_independent_set(adj_matrix, subset):
            independent_sets.append(subset)

    # 2) Create LP:  min sum(x_S)  subject to sum_{S : v in S}( x_S ) >= 1 for every vertex v
    prob = pulp.LpProblem("FractionalColoring", pulp.LpMinimize)

    # Create one variable per independent set
    x = {}
    for i, s in enumerate(independent_sets):
        x[i] = pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpContinuous)

    # Objective: minimize sum of x_S
    prob += pulp.lpSum([x[i] for i in range(len(independent_sets))]), "MinimizeSumOfx_S"

    # Constraints: for each vertex v, the sum of x_S for all S containing v >= 1
    for v in vertices:
        prob += pulp.lpSum(x[i] for i, s in enumerate(independent_sets) if v in s) >= 1

    # Solve the LP
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # The objective value = sum x_S = fractional chromatic number
    chi_f = pulp.value(prob.objective)
    return chi_f

def compute_korner_entropy(adj_matrix):
    """
    Compute Shannon-type (Körner) Graph Entropy H(G) in base 2 for a graph
    with the given adjacency matrix 'adj_matrix'.
    
    H(G) = log2( chi_f( complement_of_G ) ), 
    where chi_f() is the fractional chromatic number.
    """
    n = len(adj_matrix)
    
    # Build the complement adjacency matrix
    complement_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                complement_matrix[i][j] = 1 - adj_matrix[i][j]
            else:
                # No self-loops in this representation
                complement_matrix[i][j] = 0
    
    # Compute fractional chromatic number of the complement
    chi_f_complement = fractional_chromatic_number(complement_matrix)
    
    # Körner's entropy is log2(chi_f( complement(G) ))
    # (Be careful with floating precision and edge cases.)
    return math.log2(chi_f_complement)


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
    print(elemCount)
    numDiag = elemCount // diagLen + 1
    print("NumDiag: ", numDiag)

    # Randomly choose starting positions from the available rows (or columns)
    if num_rows >= num_cols:
        diagPositions = torch.randperm(num_rows, device=device)[:numDiag]
    else:
        diagPositions = torch.randperm(num_cols, device=device)[:numDiag]

    # Initialize final mask as all False.
    final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    # For each chosen diagonal starting position, get the one-diagonal mask and combine.
    for diag_pos in diagPositions:
        one_diag = get_mask_one_diagonal_torch(mask_shape, int(diag_pos), device=device)
        final_mask |= one_diag  # logical OR to add this diagonal

    return final_mask

#Produce a mask with elements spread randomly in unstructured manner
def get_mask_unstructured_torch(mask_shape, sparsity, device='cuda'):
    """
    Computes a mask for a matrix of shape `mask_shape` where a specified fraction
    (1 - sparsity) of elements are nonzero. This is done by:
    
    1) Computing the number of nonzero elements as: elemCount = (1 - sparsity) * total_elems.
    2) Randomly choosing elemCount number of elements to set as True.
    """
    num_rows, num_cols = mask_shape
    diagLen = num_cols if num_rows >= num_cols else num_rows

    # Calculate total nonzero elements desired.
    elemCount = int((1 - sparsity) * num_rows * num_cols)
    numDiag = elemCount // diagLen + 1
    elemCount = numDiag * diagLen
    print(elemCount)
    
    # Randomly choose elemCount number of elements to set as True.
    mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    # Choose elemCount number of elements to set as True.
    mask.ravel()[torch.randperm(num_rows * num_cols)[:elemCount]] = True

    return mask

def generate_random_permutation_matrix(size, device='cpu'):
    """
    Generates a random permutation matrix of shape (size, size).
    """
    identity = torch.eye(size, device=device)
    permutation = identity[torch.randperm(size)]

    #Save the permutation matrix to a file
    torch.save(permutation, 'permutation.pt')

    return permutation

def generate_random_permutation_matrix_time_seed(size, device='cuda'):
    """
    Generates a random permutation matrix of shape (size, size).
    Uses the current time to seed a local RNG so that each call is unique,
    even if the global seed is fixed elsewhere.
    """
    # Create a local generator so we don't affect the global RNG
    gen = torch.Generator(device=device)

    # Use microseconds or nanoseconds for uniqueness
    # (int(time.time() * 1e6), for example)
    seed_val = int(time.time() * 1e6)
    gen.manual_seed(seed_val)

    identity = torch.eye(size, device=device)
    
    # Pass the local generator to randperm
    #permutation = identity[torch.randperm(size, generator=gen)]

    # IMPORTANT: specify the same device for randperm
    perm_indices = torch.randperm(size, generator=gen, device=device)
    permutation = identity[perm_indices]

    # Optionally save to a unique file
    #filename = f'permutation_{seed_val}.pt'
    #torch.save(permutation, filename)
    #print(f"Saved permutation to {filename} with seed {seed_val}")

    return permutation

def apply_permutation_to_mask(mask, permutation_matrix):
    """
    Applies a permutation matrix to a mask by matrix multiplication.
    """
    return torch.matmul(permutation_matrix.float(), mask.float()).bool()


def permStruc(mask, device='cuda', permute_rows=True, permute_cols=True):
    """
    Permute a rectangular mask of shape (M, N) in the row dimension,
    column dimension, or both.

    Args:
      mask: The original 2D mask (M x N).
      device: 'cpu' or 'cuda'.
      permute_rows: If True, permute the row dimension.
      permute_cols: If True, permute the column dimension.

    Returns:
      A permuted mask of shape (M, N).
    """
    M, N = mask.shape

    # Convert the mask to float for matmul, we'll cast back to bool later.
    result = mask.float().to(device)

    # --- 1) Permute rows (left-multiply) ---
    if permute_rows:
        P_rows = generate_random_permutation_matrix_time_seed(M, device=device)  # (M x M)
        result = P_rows.float().matmul(result)  # --> shape is (M, N)

    # --- 2) Permute columns (right-multiply) ---
    if permute_cols:
        P_cols = generate_random_permutation_matrix_time_seed(N, device=device)  # (N x N)
        result = result.matmul(P_cols.float())  # --> shape is (M, N)

    # Cast back to bool if your mask is boolean
    return result.bool()

def get_mask_one_block_torch(mask_shape, start_row, start_col, block_size, device='cuda'):
    """
    Computes a mask for a matrix where a square block of size `block_size` x `block_size`
    starting at position (start_row, start_col) is nonzero.
    """
    # Initialize mask as all False
    mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    
    # Calculate rows and columns for the block
    rows = torch.arange(start_row, start_row + block_size, device=device)
    cols = torch.arange(start_col, start_col + block_size, device=device)
    
    # Create a block by setting elements at these positions to True
    row_indices, col_indices = torch.meshgrid(rows, cols, indexing='ij')
    mask[row_indices, col_indices] = True

    return mask

import torch

def get_mask_nm_torch(mask_shape, sparsity='random', n=2, m=4, device='cuda'):
    """
    For each row, break it into segments of length m. In each segment,
    choose n random elements to be True. If the last segment is shorter
    than m, then pick a proportionally smaller number of elements to be True.

    Args:
        mask_shape (tuple): (num_rows, num_cols).
        n (int): Number of non-zero (True) elements per segment of length m.
                 For the final shorter segment, a proportional number is used.
        m (int): Length of the segment to consider per chunk in a row.
        device (str): Device to place the generated mask on (default 'cuda').

    Returns:
        torch.Tensor: A boolean mask of shape (num_rows, num_cols).
    """
    print("Inside get_mask_nm_torch, with n: ", n, " m: ", m)
    num_rows, num_cols = mask_shape
    final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)

    for row in range(num_rows):
        col_start = 0
        while col_start < num_cols:
            # Determine the chunk size for this segment
            segment_size = min(m, num_cols - col_start)

            # Calculate how many indices to pick in this segment
            # e.g., proportionally scale n for smaller segments
            # and clamp so it never exceeds segment_size
            sub_n = int(round(segment_size * n / m))
            sub_n = min(sub_n, segment_size)

            if sub_n > 0:
                # Pick sub_n random indices from this segment
                indices = torch.randperm(segment_size, device=device)[:sub_n]
                final_mask[row, col_start + indices] = True

            col_start += segment_size

    return final_mask


def get_mask_block_torch(mask_shape, sparsity, block_size, device='cuda'):
    """
    Computes a mask for a matrix where a specified fraction (1 - sparsity) of elements 
    are nonzero using square blocks of size `block_size` x `block_size`.
    
    Valid starting positions ensure that blocks fit entirely within the matrix boundaries.
    """
    num_rows, num_cols = mask_shape
    diagLen = num_cols if num_rows >= num_cols else num_rows

    # Calculate total nonzero elements desired.
    elemCount = int((1 - sparsity) * num_rows * num_cols)
    numDiag = elemCount // diagLen + 1
    elemCount = numDiag * diagLen
    # Number of full blocks needed (each block has block_size^2 elements)
    numBlocks = elemCount // (block_size * block_size) + 1
    #print(f"Number of blocks: {numBlocks}")
    
    # Calculate valid starting positions (ensuring blocks don't go out of bounds)
    valid_row_starts = num_rows - block_size + 1
    valid_col_starts = num_cols - block_size + 1
    
    # Check if there are enough valid positions
    if valid_row_starts <= 0 or valid_col_starts <= 0:
        raise ValueError(f"Block size {block_size} is too large for the mask shape {mask_shape}")
    
    # Initialize final mask as all False
    final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    
    # Generate random starting positions for blocks
    start_rows = torch.randint(0, valid_row_starts, (numBlocks,), device=device)
    start_cols = torch.randint(0, valid_col_starts, (numBlocks,), device=device)
    
    # For each chosen block starting position, get the one-block mask and combine
    for i in range(numBlocks):
        one_block = get_mask_one_block_torch(
            mask_shape, int(start_rows[i]), int(start_cols[i]), block_size, device=device
        )
        final_mask |= one_block  # logical OR to add this block

    return final_mask

def von_neumann_graph_entropy(A):
    """
    Compute the von Neumann graph entropy for an undirected graph
    given its adjacency matrix A.

    Parameters
    ----------
    A : np.ndarray (or similar array-like)
        Adjacency matrix of the graph (N x N). Can be dense or sparse
        but must be convertible to a dense array for this implementation.

    Returns
    -------
    float
        The von Neumann entropy of the graph.
    """
    # Convert A to dense if needed (for demonstration).
    # For large graphs, consider a sparse approach with eigsh or similar.
    A = np.asarray(A, dtype=float)

    # Number of nodes
    n = A.shape[0]

    # Degree vector: d_i = sum of row i in A
    d = A.sum(axis=1)

    # Construct the diagonal of D^{-1/2}, taking care of zeros in d
    # (for isolated vertices, set 1/sqrt(0) to 0 in those positions)
    d_inv_sqrt = np.zeros_like(d)
    nonzero_mask = (d > 0)
    d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(d[nonzero_mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # Standard (combinatorial) Laplacian: L = D - A
    # Normalized Laplacian: L_norm = D^{-1/2} * (D - A) * D^{-1/2}
    L = np.diag(d) - A
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Compute eigenvalues of L_norm (symmetric => eigh)
    w, _ = eigh(L_norm)

    # Numerical safeguard: clip small negative eigenvalues to 0
    w = np.clip(w, 0, None)

    # Von Neumann entropy = - sum_i [lambda_i * log(lambda_i)]
    # By convention, 0 * log(0) is treated as 0.
    mask = (w > 1e-15)
    S_vN = -np.sum(w[mask] * np.log(w[mask]))

    return S_vN

def random_permute_rows(W, seed=None):
    """
    Generates a random permutation matrix P of appropriate size
    and applies W' = P * W, reordering the rows of W.

    Parameters
    ----------
    W : np.ndarray
        2D array, shape (N, M). The matrix (or mask) to permute.
    seed : int, optional
        Random seed. If None, a time-based seed is used.

    Returns
    -------
    W_perm : np.ndarray
        The permuted matrix with shape (N, M).
    P : np.ndarray
        The (N x N) permutation matrix used, so that W_perm = P @ W.
    """
    if seed is None:
        # Use current nanosecond time, but reduce it to fit the 32-bit range:
        seed = int(time.time_ns() % (2**32 - 1))
        
    np.random.seed(seed)

    n = W.shape[0]
    perm = np.random.permutation(n)
    P = np.eye(n)[perm]

    #if seed is None:
        # Use current nanosecond time, but reduce it to fit the 32-bit range:
    seed2 = int(time.time_ns() % (2**32 - 1))
        
    np.random.seed(seed2)

    n2 = W.shape[0]
    perm2 = np.random.permutation(n2)
    P2 = np.eye(n)[perm2]

    W_perm = P @ W @ P2
    return W_perm, P

def normalized_laplacian(adj_matrix):
    """
    Compute the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    for an undirected graph with adjacency 'adj_matrix'.
    """
    n = len(adj_matrix)
    deg = np.sum(adj_matrix, axis=1)
    
    # Avoid division by zero for isolated vertices by treating them carefully.
    # (In practice, we might define an isolated vertex's row in D^{-1/2} as 0.)
    D_inv_sqrt = np.zeros((n, n))
    for i in range(n):
        if deg[i] > 0:
            D_inv_sqrt[i, i] = 1.0 / math.sqrt(deg[i])
    
    # L = I - D^(-1/2) A D^(-1/2)
    I = np.eye(n)
    D_inv_sqrt_AD_inv_sqrt = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return I - D_inv_sqrt_AD_inv_sqrt


def shannon_renyi_graph_entropy(adj_matrix, alpha=1.0, eps=1e-12):
    """
    Compute the Shannon–Rényi entropy for the undirected graph
    given by 'adj_matrix', using the normalized Laplacian spectrum.
    
    :param adj_matrix: 2D square numpy array (adjacency matrix).
    :param alpha: real parameter > 0. 
                  - alpha=1 -> Shannon (limit case)
                  - alpha!=1 -> Rényi
    :param eps: small epsilon to avoid log(0).
    :return: entropy value (float), in natural log base (ln).
             If you'd like log base 2, divide by ln(2).
    """
    # 1) Build normalized Laplacian
    L = normalized_laplacian(adj_matrix)
    
    # 2) Compute its eigenvalues
    #    Using 'eigvalsh' for a symmetric matrix.
    vals = np.linalg.eigvalsh(L)
    
    # 3) The sum of eigenvalues of L is n (for a connected or disconnected undirected graph).
    #    We'll form a probability distribution p_i = lambda_i / n.
    n = len(adj_matrix)
    # to be safe, let's explicitly sum them in case of floating precision
    total = np.sum(vals)
    # but theoretically total ~ n
    if total < eps:
        # This would be a weird edge case (like a graph with no edges, or numeric issues).
        # The "entropy" is effectively 0 if L is all zeros. We'll just return 0.
        return 0.0
    
    p = vals / total
    
    # 4) Shannon–Rényi formula
    if abs(alpha - 1.0) < 1e-9:
        # Shannon Entropy (limit alpha -> 1)
        # H = - sum_i p_i * ln(p_i)
        # handle zero p_i carefully
        return -np.sum(p * np.log(p + eps))
    else:
        # Rényi Entropy
        # H_alpha = (1 / (1 - alpha)) ln( sum_i p_i^alpha )
        sum_palpha = np.sum(np.power(p, alpha))
        return (1.0 / (1.0 - alpha)) * math.log(sum_palpha + eps)

def compute_clustering_and_path_length(G):
    """
    Compute the average clustering coefficient (C) and characteristic path length (L).
    If the graph is disconnected, we use the largest connected component to compute L.
    """
    # Clustering
    C = nx.average_clustering(G)
    
    # Characteristic path length: use the largest connected component if needed
    if not nx.is_connected(G):
        # Extract the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_sub)
    else:
        L = nx.average_shortest_path_length(G)
    
    return C, L

def generate_random_reference(n, m, num_reference=10):
    """
    Generate 'num_reference' Erdős–Rényi random graphs (G(n, p)) 
    matching the approximate edge density p = m / (n*(n-1)/2).
    
    Returns the average clustering and average path length across those random graphs.
    """
    # Probability of an edge
    possible_edges = n * (n - 1) / 2
    p = m / possible_edges if possible_edges > 0 else 0
    
    C_values = []
    L_values = []
    
    for _ in range(num_reference):
        # Generate random graph
        G_rand = nx.fast_gnp_random_graph(n, p)
        
        if G_rand.number_of_nodes() == 0 or G_rand.number_of_edges() == 0:
            # If empty or edgeless, skip
            continue
        
        C_rand, L_rand = compute_clustering_and_path_length(G_rand)
        C_values.append(C_rand)
        L_values.append(L_rand)
    
    if len(C_values) == 0 or len(L_values) == 0:
        # If we can't generate a valid random graph (edge probability too low, etc.), warn
        warnings.warn("Random reference graphs could not be generated properly.")
        return np.nan, np.nan
    
    return np.mean(C_values), np.mean(L_values)

def generate_lattice_reference(n, m, num_reference=1):
    """
    Generate a ring-lattice (or multiple ring-lattices) with the same average degree
    as the original graph. The original graph has n nodes and m edges, so the
    average degree is k = 2m / n. We'll round k to the nearest integer for simplicity.

    Returns the average (C, L) across those 'num_reference' lattice graphs.
    """

    if n == 0:
        return np.nan, np.nan

    # Average degree
    k_approx = 2 * m / float(n)
    k = int(round(k_approx))

    # For a ring-lattice, typically we connect each node to k/2 neighbors on each side.
    # If k is odd, we'll do k//2 on one side and k//2 + 1 on the other side, etc.
    # We'll generate 'num_reference' such lattices (though typically one might suffice).
    C_values = []
    L_values = []
    for _ in range(num_reference):
        # Build ring-lattice
        G_lat = nx.Graph()
        G_lat.add_nodes_from(range(n))

        # Each node i: connect to the next k//2 neighbors on each side (cyclic)
        for i in range(n):
            for j in range(1, k // 2 + 1):
                # wrap-around neighbors
                right = (i + j) % n
                left = (i - j) % n
                G_lat.add_edge(i, right)
                G_lat.add_edge(i, left)

            # If k is odd, connect one extra neighbor in one direction
            if k % 2 == 1:
                extra = (i + (k // 2 + 1)) % n
                G_lat.add_edge(i, extra)

        c_lat, l_lat = compute_clustering_and_path_length(G_lat)
        C_values.append(c_lat)
        L_values.append(l_lat)

    return np.mean(C_values), np.mean(L_values)

def small_world_omega(L, L_r, C, C_l):
    """
    Compute the small-world measure ω defined as:
    
        ω = (L_r / L) - (C / C_l)

    where:
      - L  : characteristic path length of the original network
      - L_r: characteristic path length of the random reference
      - C  : clustering coefficient of the original network
      - C_l: clustering coefficient of the lattice reference
    """
    # Watch out for divide-by-zero
    if (L == 0 or C_l == 0):
        return np.nan
    return (L_r / L) - (C / C_l)

def small_world_index(L, L_l, L_r, C, C_l, C_r):
    """
    Compute the Small World Index (SWI) defined as:
    
        SWI = ((L - L_l) / (L_r - L_l)) * ((C - C_r) / (C_l - C_r))

    where:
      - L   : characteristic path length of the original network
      - L_l : path length of the lattice reference
      - L_r : path length of the random reference
      - C   : clustering coefficient of the original network
      - C_l : clustering coefficient of the lattice reference
      - C_r : clustering coefficient of the random reference
    
    Both ω and SWI (sometimes called ω′ in some sources) can range between 0 and 1
    under ideal conditions, but interpretations vary.
    """
    # Safeguard denominators
    denom_L = (L_r - L_l)
    denom_C = (C_l - C_r)
    if denom_L == 0 or denom_C == 0:
        return np.nan

    return ((L - L_l) / denom_L) * ((C - C_r) / denom_C)

def small_worldness_humphries_gurney(C, L, C_rand, L_rand):
    """
    Standard definition from Humphries & Gurney (2008):
    
    S_HG = (C / C_rand) / (L / L_rand)
    
    Description:
    ------------
    - C: average clustering coefficient of the original graph
    - L: characteristic path length of the original graph
    - C_rand: average clustering of the random reference graphs
    - L_rand: average path length of the random reference graphs
    
    If S_HG > 1, this often indicates 'small-world' behavior.
    """
    if (C_rand == 0 or L_rand == 0):
        return np.nan
    return (C / C_rand) / (L / L_rand)

def small_worldness_diff(C, L, C_rand, L_rand):
    """
    Difference-based measure (one variant used in Telesford et al. 2011):
    
    S_diff = (C - C_rand) / (L - L_rand)
    
    Description:
    ------------
    - C: average clustering of original graph
    - L: average path length of original graph
    - C_rand: average clustering of the random reference
    - L_rand: average path length of the random reference
    
    Interpretation:
    ---------------
    - If S_diff > 0, your network is more 'small-world' than the random reference 
      (higher clustering or shorter path lengths).
    - Negative values suggest the opposite.
    """
    # Guard against zero denominators
    denom = (L - L_rand)
    if denom == 0:
        return np.nan
    return (C - C_rand) / denom

def compute_small_world_measures(adj_matrix, num_reference=10):
    """
    Given an adjacency matrix, compute:
      - C, L (original graph)
      - C_rand, L_rand (random reference)
      - small-worldness S_HG (Humphries & Gurney)
      - small-worldness S_diff (difference-based approach, Telesford variant)
    
    Returns a dictionary with all metrics.
    """
    # Construct Graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Compute metrics for the original network
    C, L = compute_clustering_and_path_length(G)
    
    # Compute reference values
    C_rand, L_rand = generate_random_reference(n, m, num_reference=num_reference)
    
    # Lattice reference metrics
    C_l, L_l = generate_lattice_reference(n, m, num_reference=1)

    # Compute small-worldness
    S_HG = small_worldness_humphries_gurney(C, L, C_rand, L_rand)
    S_diff = small_worldness_diff(C, L, C_rand, L_rand)
    omega_val = small_world_omega(L, L_rand, C, C_l)
    swi_val = small_world_index(L, L_l, L_rand, C, C_l, C_rand)
    
    return {
        'C': C,
        'L': L,
        'C_rand': C_rand,
        'L_rand': L_rand,
        'L_lattice': L_l,
        'C_lattice': C_l,
        'S_HG': S_HG,      # Humphries & Gurney ratio
        'S_diff': S_diff,   # Difference-based approach
        'omega': omega_val, # Small-world measure
        'SWI': swi_val      # Small World Index
    }

# Example usage:
""" if __name__ == "__main__":
    #Get a mask with block sparsity
    maskShape = (256,256)
    sparsitySel = 0.8
    maskBlock4 = get_mask_block_torch(mask_shape=maskShape, sparsity=sparsitySel, block_size=4, device='cuda')
    maskBlock8 = get_mask_block_torch(mask_shape=maskShape, sparsity=sparsitySel, block_size=8, device='cuda')
    maskBlock16 = get_mask_block_torch(mask_shape=maskShape, sparsity=sparsitySel, block_size=16, device='cuda')
    maskBlock32 = get_mask_block_torch(mask_shape=maskShape, sparsity=sparsitySel, block_size=32, device='cuda')

    #Get a mask with unstrucutred sparsity
    maskUnstructured = get_mask_unstructured_torch(mask_shape=maskShape, sparsity=sparsitySel, device='cuda')

    #Get a mask with diagonal sparsity
    maskDiagonal = get_mask_diagonal_torch(mask_shape=maskShape, sparsity=sparsitySel, device='cuda')

    #Get a mask with N:M sparsity
    maskNM = get_mask_nm_torch(mask_shape=maskShape, n=1, m=5, device='cuda')

    #Now, calculate the von nueman entropy for all the masks and print them out
    print("Von Neumann entropy for block size 4: ", von_neumann_graph_entropy(maskBlock4.cpu().numpy()))
    print("Von Neumann entropy for block size 8: ", von_neumann_graph_entropy(maskBlock8.cpu().numpy()))
    print("Von Neumann entropy for block size 16: ", von_neumann_graph_entropy(maskBlock16.cpu().numpy()))
    print("Von Neumann entropy for block size 32: ", von_neumann_graph_entropy(maskBlock32.cpu().numpy()))
    print("Von Neumann entropy for unstructured: ", von_neumann_graph_entropy(maskUnstructured.cpu().numpy()))
    print("Von Neumann entropy for diagonal: ", von_neumann_graph_entropy(maskDiagonal.cpu().numpy()))
    print("Von Neumann entropy for N:M: ", von_neumann_graph_entropy(maskNM.cpu().numpy()))

    #Apply random permutation to all masks above and then calculate the von_neuman_entropy
    maskBlock4_perm, P_block4 = random_permute_rows(maskBlock4.cpu().numpy())
    maskBlock8_perm, P_block8 = random_permute_rows(maskBlock8.cpu().numpy())
    maskBlock16_perm, P_block16 = random_permute_rows(maskBlock16.cpu().numpy())
    maskBlock32_perm, P_block32 = random_permute_rows(maskBlock32.cpu().numpy())

    maskUnstructured_perm, P_unstructured = random_permute_rows(maskUnstructured.cpu().numpy())
    maskDiagonal_perm, P_diagonal = random_permute_rows(maskDiagonal.cpu().numpy())
    maskNM_perm, P_nm = random_permute_rows(maskNM.cpu().numpy())

    print("Post permutation entropy")
    print("Von Neumann entropy for block size 4: ", von_neumann_graph_entropy(maskBlock4_perm))
    print("Von Neumann entropy for block size 8: ", von_neumann_graph_entropy(maskBlock8_perm))
    print("Von Neumann entropy for block size 16: ", von_neumann_graph_entropy(maskBlock16_perm))
    print("Von Neumann entropy for block size 32: ", von_neumann_graph_entropy(maskBlock32_perm))
    print("Von Neumann entropy for unstructured: ", von_neumann_graph_entropy(maskUnstructured_perm))
    print("Von Neumann entropy for diagonal: ", von_neumann_graph_entropy(maskDiagonal_perm))
    print("Von Neumann entropy for N:M: ", von_neumann_graph_entropy(maskNM_perm))
 """

def mainVonNeuman(num_iterations=100):
    # 1) We test multiple sparsities
    sparsities = [0.8, 0.9, 0.95, 0.99]
    
    # 2) Matrix sizes
    #matrix_sizes = [32, 64, 128, 256, 512, 1024]
    matrix_sizes = [60, 80,100,120,140,160,200,240,280,320,360,400,450, 500,600]
    
    # We'll define the mask types we want
    mask_types = [ "Block 16", "Block 32",
                  "Unstructured", "Diagonal", "N:M"]
    
    # For cleanliness, let's store everything in a nested dictionary:
    # results[sparsity]['before'][mask_type][size] = list of 100 entropy values
    # results[sparsity]['after'][mask_type][size]  = list of 100 entropy values
    results = {}
    for sp in sparsities:
        results[sp] = {
            'before': {mt: {sz: [] for sz in matrix_sizes} for mt in mask_types},
            'after':  {mt: {sz: [] for sz in matrix_sizes} for mt in mask_types}
        }
    
    # ---------------------------------------------------------
    # Gather data:
    # For each (sparsity, matrix size, mask type), run 100 permutations
    # ---------------------------------------------------------
    for sp in sparsities:
        for size in matrix_sizes:
            # Build each mask ONCE per (sp, size)
            mask_shape = (size, size)
            #mask_block4  = get_mask_block_torch(mask_shape, sp, block_size=4)
            #mask_block8  = get_mask_block_torch(mask_shape, sp, block_size=8)
            mask_block16 = get_mask_block_torch(mask_shape, sp, block_size=16)
            mask_block32 = get_mask_block_torch(mask_shape, sp, block_size=32)
            mask_unstruct= get_mask_unstructured_torch(mask_shape, sp)
            mask_diag    = get_mask_diagonal_torch(mask_shape, sp)
            mask_nm      = get_mask_nm_torch(mask_shape, n=1, m=5)
            
            # Convert to numpy
            #mb4_np  = mask_block4.cpu().numpy()
            #mb8_np  = mask_block8.cpu().numpy()
            mb16_np = mask_block16.cpu().numpy()
            mb32_np = mask_block32.cpu().numpy()
            mu_np   = mask_unstruct.cpu().numpy()
            md_np   = mask_diag.cpu().numpy()
            mnm_np  = mask_nm.cpu().numpy()
            
            # (Optional) compute "before" entropies only once per iteration,
            # but since the mask doesn't change across iterations, you could
            # also compute them just once. Here, we repeat each iteration
            # for clarity, though it's not strictly necessary.
            # ---- "before" entropies ----
            #ent_b4_block4  = float(von_neumann_graph_entropy(mb4_np))
            #ent_b4_block8  = float(von_neumann_graph_entropy(mb8_np))
            ent_b4_block16 = float(von_neumann_graph_entropy(mb16_np))
            ent_b4_block32 = float(von_neumann_graph_entropy(mb32_np))
            ent_b4_unstruct= float(von_neumann_graph_entropy(mu_np))
            ent_b4_diag    = float(von_neumann_graph_entropy(md_np))
            ent_b4_nm      = float(von_neumann_graph_entropy(mnm_np))
            
            #results[sp]['before']["Block 4"][size].append(ent_b4_block4)
            #results[sp]['before']["Block 8"][size].append(ent_b4_block8)
            results[sp]['before']["Block 16"][size].append(ent_b4_block16)
            results[sp]['before']["Block 32"][size].append(ent_b4_block32)
            results[sp]['before']["Unstructured"][size].append(ent_b4_unstruct)
            results[sp]['before']["Diagonal"][size].append(ent_b4_diag)
            results[sp]['before']["N:M"][size].append(ent_b4_nm)

            for _ in range(num_iterations):
                print("Iteration: ", _)
                
                # ---- "after" entropies: random permutation each iteration ----
                #mb4_perm, _  = random_permute_rows(mb4_np)
                #mb8_perm, _  = random_permute_rows(mb8_np)
                mb16_perm, _ = random_permute_rows(mb16_np)
                mb32_perm, _ = random_permute_rows(mb32_np)
                mu_perm, _   = random_permute_rows(mu_np)
                md_perm, _   = random_permute_rows(md_np)
                mnm_perm, _  = random_permute_rows(mnm_np)
                
                #ent_aft_block4  = float(von_neumann_graph_entropy(mb4_perm))
                #ent_aft_block8  = float(von_neumann_graph_entropy(mb8_perm))
                ent_aft_block16 = float(von_neumann_graph_entropy(mb16_perm))
                ent_aft_block32 = float(von_neumann_graph_entropy(mb32_perm))
                ent_aft_unstruct= float(von_neumann_graph_entropy(mu_perm))
                ent_aft_diag    = float(von_neumann_graph_entropy(md_perm))
                ent_aft_nm      = float(von_neumann_graph_entropy(mnm_perm))
                
                #results[sp]['after']["Block 4"][size].append(ent_aft_block4)
                #results[sp]['after']["Block 8"][size].append(ent_aft_block8)
                results[sp]['after']["Block 16"][size].append(ent_aft_block16)
                results[sp]['after']["Block 32"][size].append(ent_aft_block32)
                results[sp]['after']["Unstructured"][size].append(ent_aft_unstruct)
                results[sp]['after']["Diagonal"][size].append(ent_aft_diag)
                results[sp]['after']["N:M"][size].append(ent_aft_nm)
    
    # ----------------------------------------------------
    # Plotting: For each sparsity, produce 2 figures:
    #   1) "Before permutation" with 7 lines
    #   2) "After permutation"  with 7 lines
    # Each line shows mean ± std dev across the 100 runs
    # ----------------------------------------------------
    
    for sp in sparsities:
        # Prepare x-axis
        x_vals = matrix_sizes
        
        # =============== BEFORE permutation ===============
        plt.figure()
        for mt in mask_types:
            # Gather all 100 entropies for each matrix size
            means = []
            stds = []
            for sz in x_vals:
                arr = results[sp]['before'][mt][sz]
                means.append(np.mean(arr))
                stds.append(np.std(arr))
            
            # Plot the mean line
            plt.plot(x_vals, means, label=mt)
            
            # Shade +/- 1 std dev
            lower = [m - s for (m, s) in zip(means, stds)]
            upper = [m + s for (m, s) in zip(means, stds)]
            plt.fill_between(x_vals, lower, upper, alpha=0.2)
        
        plt.xticks(x_vals, [str(val) for val in x_vals])
        plt.xlabel("Matrix Size")
        plt.ylabel("Von Neumann Entropy")
        plt.title(f"Entropy Before Permutation (Sparsity={sp})")
        plt.legend()
        plt.savefig(f'Entropy_before_permutation_s{sp}.png')
        plt.close()
        
        # =============== AFTER permutation ===============
        plt.figure()
        for mt in mask_types:
            means = []
            stds = []
            for sz in x_vals:
                arr = results[sp]['after'][mt][sz]
                means.append(np.mean(arr))
                stds.append(np.std(arr))
            
            plt.plot(x_vals, means, label=mt)
            lower = [m - s for (m, s) in zip(means, stds)]
            upper = [m + s for (m, s) in zip(means, stds)]
            plt.fill_between(x_vals, lower, upper, alpha=0.2)
        
        plt.xticks(x_vals, [str(val) for val in x_vals])
        plt.xlabel("Matrix Size")
        plt.ylabel("Von Neumann Entropy")
        plt.title(f"Entropy After Permutation (Sparsity={sp})")
        plt.legend()
        plt.savefig(f'Entropy_after_permutation_s{sp}.png')
        plt.close()
    
    # (Optional) Save final results to disk for further processing:
    # for sp in sparsities:
    #     torch.save(results[sp]['before'], f'entropies_before_s{sp}.pt')
    #     torch.save(results[sp]['after'],  f'entropies_after_s{sp}.pt')

def smallWorldness():
    size = 100
    sp = 0.4
    mask_shape = (size, size)
    mask_block4  = get_mask_block_torch(mask_shape, sp, block_size=4)
    mask_block8  = get_mask_block_torch(mask_shape, sp, block_size=8)
    mask_block16 = get_mask_block_torch(mask_shape, sp, block_size=16)
    mask_block32 = get_mask_block_torch(mask_shape, sp, block_size=32)
    mask_unstruct= get_mask_unstructured_torch(mask_shape, sp)
    mask_diag    = get_mask_diagonal_torch(mask_shape, sp)
    mask_nm      = get_mask_nm_torch(mask_shape, n=1, m=5)

    #convert the masks to numpy
    mask_block4_np = mask_block4.cpu().numpy()
    mask_block8_np = mask_block8.cpu().numpy()
    mask_block16_np = mask_block16.cpu().numpy()
    mask_block32_np = mask_block32.cpu().numpy()
    mask_unstruct_np = mask_unstruct.cpu().numpy()
    mask_diag_np = mask_diag.cpu().numpy()
    mask_nm_np = mask_nm.cpu().numpy()


    results = compute_small_world_measures(mask_block32_np, num_reference=20)
    
    print("Computed Small-World Metrics:")
    for key, val in results.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    results = compute_small_world_measures(mask_block16_np, num_reference=20)
    
    print("Computed Small-World Metrics:")
    for key, val in results.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    results = compute_small_world_measures(mask_diag_np, num_reference=10)
    
    print("Computed Small-World Metrics:")
    for key, val in results.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    results = compute_small_world_measures(mask_nm_np, num_reference=10)
    
    print("Computed Small-World Metrics:")
    for key, val in results.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

if __name__ == "__main__":
    mainVonNeuman(200)  
    #smallWorldness()