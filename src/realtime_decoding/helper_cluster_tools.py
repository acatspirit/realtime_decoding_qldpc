import numpy as np
from ldpc_post_selection.src.ldpc_post_selection.cluster_tools import *

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
    Inputs:
    stats: the stats from bplsd for the particular committed region
    num_faults_in_W: total # of faults in the entire window 
    num_faults_in_F: total # of faults in the commit region F
    order: order for cluster norm 
    decoder_type: type of decoder used. Either 'bplsd' or 'uf'

    Output:
    clusters: array of size # of faults. each entry has a unique identifier, that tells us in which cluster each fault mechanism belongs to.
    '''
    # modify to work with UF
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
