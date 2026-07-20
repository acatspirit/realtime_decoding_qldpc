import numpy as np

'''
The functions "compute_lp_norm" , "compute_cluster_stats" and "compute_cluster_norm_fraction"
were created in https://github.com/seokhyung-lee/ldpc-post-selection/tree/main,
and some features were modified here for our purposes. 
All credits for these three functions go to the creators of https://github.com/seokhyung-lee/ldpc-post-selection/tree/main.

'''

def compute_lp_norm(values: np.ndarray, order: float, take_abs: bool = False) -> float:
    """
    Compute an L_p norm for 1D values with optional absolute values.

    Parameters
    ----------
    values : 1D numpy array of float
        Values for which the norm should be computed.
    order : float
        Order for the L_p norm (positive number or `np.inf`).
    take_abs : bool, optional
        If True, take absolute values before the norm calculation. Defaults to False.

    Returns
    -------
    norm_value : float
        Calculated norm of the provided values.
    """
    if values.size == 0:
        return 0.0

    processed = np.abs(values) if take_abs else values

    if processed.size == 0:
        return 0.0

    if order == 1:
        return float(np.sum(processed))
    if order == 2:
        return float(np.sqrt(np.sum(processed**2)))
    if np.isinf(order):
        return float(np.max(processed)) if processed.size > 0 else 0.0

    return float(np.sum(processed**order) ** (1.0 / order))


def compute_cluster_stats(clusters: np.ndarray):
    """
    Compute cluster statistics from cluster assignments using numpy native functions.

    Parameters
    ----------
    clusters : 1D numpy array of int
        Cluster assignments for each bit (0 = outside cluster, 1+ = cluster ID).


    Returns
    -------
    cluster_sizes : 1D numpy array of int
        Size of each cluster (index corresponds to cluster ID).

    """
    max_cluster_id = clusters.max()

    # Use bincount to efficiently compute cluster sizes and LLR sums
    cluster_sizes = np.bincount(clusters, minlength=max_cluster_id + 1).astype(np.int_)
    
    return cluster_sizes

def compute_cluster_norm_fraction(values: np.ndarray, order: float) -> float:
    """
    Compute norm fraction for given values and order.

    Parameters
    ----------
    values : 1D numpy array of float
        Values to compute norm fraction for (e.g., cluster sizes or LLRs).
        values[0] is assumed to be the outside region and is excluded.
    order : float
        Order for norm computation (can be a positive number or `np.inf`).

    Returns
    -------
    norm_fraction : float
        Norm fraction value for the given order.
    """
    # Get values excluding the outside region (index 0)
    inside_values = values[1:] if values.size > 1 else np.array([], dtype=values.dtype)
    if inside_values.size == 0:
        return 0.0

    total_sum = float(np.sum(values))
    if total_sum == 0.0:
        return 0.0

    inside_norm = compute_lp_norm(inside_values.astype(float, copy=False), order)
    return inside_norm / total_sum


def collect_cluster_norm(stats, num_faults_in_W, num_faults_in_F, order, decoder_type = 'bplsd'):
    '''
    Get the cluster norm for a particular window W that was decoded, but for which cluster are
    restricted into the commit region F.

    Inputs:
    stats: the stats from bplsd for the particular committed region / or directly the clusters array for uf 
          (clusters array: each value is cluster id of where error mechanism belongs to)
    num_faults_in_W: total # of faults in the entire window 
    num_faults_in_F: total # of faults in the commit region F
    order: order for cluster norm 
    decoder_type: type of decoder used. Either 'bplsd' or 'uf'

    Output:
    cluster_norm:  (\sum_{i clusters} |C_i|^{order})^{1/order} / |E|, where |E| is the size of the entire region
                   inside of which some clusters can exist. |C_i| is the cluster size of the i-th cluster (# of elements it contains).
    '''
    
    if decoder_type == 'bplsd':
        soft_output_stats = stats["individual_cluster_stats"]
        clusters = np.zeros(num_faults_in_W,dtype=np.int_)
        cluster_id = 1
        for data in soft_output_stats.values():
            if data.get("active",False):
                final_bits = data["final_bits"]
                clusters[final_bits] = cluster_id 
                cluster_id +=1

    elif decoder_type == 'uf':
        clusters = stats
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")

    #restrict the clusters to the num_faults_in_F
    clusters = clusters[:num_faults_in_F]

    cluster_sizes = compute_cluster_stats(clusters)

    soft_outputs = {}
    soft_outputs["clusters"] = clusters 
    soft_outputs["cluster_sizes"] = cluster_sizes 

    
    cluster_norm = compute_cluster_norm_fraction(cluster_sizes,order=order)
    

    return cluster_norm
