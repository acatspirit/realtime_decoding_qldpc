

import sys

import numpy as np
import ldpc.codes
from ldpc import BpOsdDecoder, UnionFindDecoder
from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.cluster_tools import compute_cluster_norm_fraction
# try UF from the other repo now too 
import py_wrapper.py_decoder as uf
from py_wrapper.some_codes import surface_code_non_periodic, get_bb
from quits.qldpc_code import HgpCode
import ldpc.codes as codes 
from realtime_decoding.decoder_switching import get_BB_circuit
from quits.qldpc_code import BbCode
from quits import ErrorModel

def get_bb_code_parity_and_logs(d):

    d_dict = {6:{'l':6, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},
                10: {'l':15, 'm':3, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [2,7], 'B_y_pows':[0]},
                12:{'l':12, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},}
    code_params = d_dict[d]
    bb = BbCode(
        l=code_params['l'],
        m=code_params['m'],
        A_x_pows=code_params['A_x_pows'],
        A_y_pows=code_params['A_y_pows'],
        B_x_pows=code_params['B_x_pows'],
        B_y_pows=code_params['B_y_pows'],
    )
    return bb.hx, bb.lx

def get_bb_codes_uf_decoder(d):
    l_dims = [[6, 6], [15, 3], [9, 6], [12, 6], [12, 12]]
    pow_a = [[3, 1, 2], [9, 1, 2], [3, 1, 2], [3, 1, 2], [3, 2, 7]]
    pow_b = [[3, 1, 2], [0, 2, 7], [3, 1, 2], [3, 1, 2], [3, 1, 2]]

    
    Hx, Hz = get_bb(l_dims[0][0], l_dims[0][1], pow_a[0], pow_b[0])  #.astype(np.uint8)
    return Hx, Hz
np.set_printoptions(threshold=sys.maxsize)

def get_cluster_norm(cluster_sizes, order=2, type="LSD"):
    if type == "LSD": # using LSD decoder
        return compute_cluster_norm_fraction(cluster_sizes, order=order) # this should include the largest cluster - i.e. whichever one doesn't have errors
    else: # using UF decoder
        total_clusters = sum(cluster_sizes)
        cluster_sizes_internal = cluster_sizes[:-1] # remove the outside cluster size, which is the largest one
        cluster_powers = np.power(cluster_sizes_internal, order)
        cluster_norm = np.sum(cluster_powers)**(1/order) / total_clusters
        return cluster_norm
d = 10

# H, L = surface_code_non_periodic(7)
H,L = get_bb_code_parity_and_logs(d=d)
# H,Hz = get_bb_codes_uf_decoder(d)
# print(f"H:\n{[(H.nonzero()[0][i],H.nonzero()[1][i]) for i in range(len(H.nonzero()[0]))]}")
# print(f"L:\n{L.nonzero()}")
# H = ldpc.codes.hamming_code(5)
p = 0.0158 # I think 6 is the smallest cluster size here

decoder_uf = uf.UFDecoder(H.astype(np.uint8))
# decoder_lsd = SoftOutputsBpLsdDecoder(H, p=p*np.ones(H.shape[1]), bp_method="minimum_sum", max_iter=20, schedule="serial", osd_method="osd_cs", osd_order=0)

# syndrome = np.random.randint(size=H.shape[0], low=0, high=2).astype(np.uint8)
error = np.random.binomial(n=1, p=p, size=H.shape[1]).astype(np.uint8)

syndrome = (H @ error % 2).astype(np.uint8)
erasures = np.zeros(shape=H.shape[1], dtype=np.uint8)
print(f"Error: {np.where(error == 1)[0]}")
print(f"Syndrome: {np.where(syndrome == 1)[0]}")
# print(f"H:\n{H}")``
clusters = decoder_uf.ldpc_decode(syndrome, erasures)
print(f"Decoding: {decoder_uf.correction}")
print(f"List of clusters: {clusters} with length {len(clusters)}")
# decoding_syndrome = H @ decoding % 2
# print(f"Decoding syndrome: {decoding_syndrome}")


# test cluster sizes
# found_cluster_sizes_uf = decoder_uf.ldpc_decode(syndrome, erasures)
# num_qubits = H.shape[1]
# clusters_including_outside_uf = np.append(found_cluster_sizes_uf[0], num_qubits - np.sum(found_cluster_sizes_uf[0])) # this includes the outside cluster, which is the largest one
# correction_uf = decoder_uf.correction
# cluster_norm_uf = get_cluster_norm(clusters_including_outside_uf, order=2, type="UF")
# print(f"Found cluster sizes: {clusters_including_outside_uf}")
# print(f"Cluster norm for uf: {cluster_norm_uf}")
# print("\n--------------------\n")
# correction_lsd, correction_bp, converge, soft_outputs_lsd = decoder_lsd.decode(syndrome)
# # print(f"Correction for lsd: {correction_lsd}")
# cluster_sizes = np.delete(soft_outputs_lsd['cluster_sizes'], np.argmax(soft_outputs_lsd['cluster_sizes']))
# cluster_norm_lsd = get_cluster_norm(soft_outputs_lsd['cluster_sizes'], order=2, type="LSD")
#  # remove the 0 cluster size (for unclustered qubits)
# print(f"cluster sizes(qubits) for lsd: {soft_outputs_lsd['cluster_sizes']}")
# print(f"Cluster norm for lsd: {cluster_norm_lsd}")


# # 2. Simulate an actual physical error pattern (e.g., flip 3 random qubits)
# num_qubits = H.shape[1]
# true_error = np.zeros(num_qubits, dtype=np.uint8)
# # Pick 3 random data qubits to suffer phase/bit flips
# # error_indices = np.random.choice(num_qubits, size=3, replace=False)
# error_indices = np.array([18, 23, 64]) # for testing purposes, do some clusters
# # error_indices = np.array([10,11,12]) # for testing purposes, do some clusters
# true_error[error_indices] = 1

# # Calculate valid syndrome from physical error indices [18, 23, 64]
# syndrome = (H @ true_error % 2).astype(np.uint8)
# erasures = np.zeros(num_qubits, dtype=np.uint8)

# # Run the updated decoder 
# sizes, cluster_membership = decoder.ldpc_decode(syndrome, erasures)

# print(f"Tracked Cluster Sizes: {sizes}")
# print("Cluster Membership Maps:")
# for cluster_idx, qubit_ids in cluster_membership.items():
#     print(f"  -> Cluster {cluster_idx} (Size {len(qubit_ids)}) contains Qubit IDs: {qubit_ids}")

# print("\n--------------------\n")
# # Setup your batch inputs (e.g. 5 repetitions)
# nrep = 5
# batch_syndromes = np.zeros(nrep * decoder.n_syndr, dtype=np.uint8)
# batch_erasures = np.zeros(nrep * decoder.n_qbt, dtype=np.uint8)

# # Populate individual shots with simulated data ...

# # Run the updated batch pipeline
# batch_data = decoder.ldpc_decode_batch(batch_syndromes, batch_erasures, nrep)

# # Unpack and verify results loop
# for shot_id, (sizes, membership) in enumerate(batch_data):
#     print(f"Shot {shot_id} -> Cluster Sizes found: {sizes}")
#     for c_idx, qubits in membership.items():
#         print(f"    Cluster {c_idx} contains Qubits: {qubits}")


"""
example output:
(base) ariannameinking@Ariannas-Fancy-MacBook-Pro-9 realtime_decoding_qldpc % /Users/ariannameinking/miniforge3/envs/realtime_decoding/bin/python /Users/ariannameinking/Documents/Brown_Research/realtime_decoding_qldpc/src/realtime_decoding/union_find.py
Syndrome: [0 0 0 0 1 0 1 0 0 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0]
Decoding: [0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0]
Number of clusters: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 34]
"""