

import sys

import numpy as np
import ldpc.codes
from ldpc import BpOsdDecoder, UnionFindDecoder
# try UF from the other repo now too 
import py_wrapper.py_decoder as uf
from py_wrapper.some_codes import surface_code_non_periodic
from quits.qldpc_code import HgpCode
import ldpc.codes as codes 
np.set_printoptions(threshold=sys.maxsize)

# I realized that their package has a toric code built in that I can compare to directly so I just used that instead

# def get_parity_toric(d):
#     H_rep = codes.rep_code(d).toarray().astype(int)

#     # build cyclic row
#     cyclic_row = np.zeros((1, d), dtype=int)
#     cyclic_row[0, 0] = 1
#     cyclic_row[0, -1] = 1

#     # append it
#     H_rep_cyclic = np.vstack([H_rep, cyclic_row])

#     code = HgpCode(H_rep_cyclic, H_rep_cyclic) # generate a toric code from the repetition code

#     return code.hx, code.hz, code.lx, code.lz

# d = 7
# Hx,Hz,Lx,Lz = get_parity_toric(d)
# H = Hx
# L = Lx

H, L = surface_code_non_periodic(7)
# print(f"H:\n{[(H.nonzero()[0][i],H.nonzero()[1][i]) for i in range(len(H.nonzero()[0]))]}")
# print(f"L:\n{L.nonzero()}")
# H = ldpc.codes.hamming_code(5)

decoder = uf.UFDecoder(H)

## The
# bp_osd = BpOsdDecoder(
#     H,
#     error_rate=0.1,
#     bp_method="product_sum",
#     max_iter=7,
#     schedule="serial",
#     osd_method="osd_cs",  #set to OSD_0 for fast solve
#     osd_order=2,
# )
#help(UnionFindDecoder)
# union_find = UnionFindDecoder(pcm=H, uf_method='True')

# #according to their docstring this is what I get for the two parameters we can pass in:
# """
# pcm : Union[np.ndarray, spmatrix]
#       The parity-check matrix (PCM) of the code. This should be either a dense matrix (numpy ndarray)
#       or a sparse matrix (scipy sparse matrix).
# uf_method : bool, optional
#       If True, the decoder operates in matrix solve mode. If False, it operates in peeling mode.
#       Default is False.
# """

# syndrome = np.random.randint(size=H.shape[0], low=0, high=2).astype(np.uint8)
# erasures = np.zeros(shape=H.shape[1], dtype=np.uint8)
# print(f"Syndrome: {syndrome}")
# decoding = union_find.decode(syndrome)
# print(f"Decoding: {decoding}")
# decoding_syndrome = H @ decoding % 2
# print(f"Decoding syndrome: {decoding_syndrome}")


# # test cluster sizes
# found_cluster_sizes = decoder.ldpc_decode(syndrome, erasures)
# print(f"Found cluster sizes: {found_cluster_sizes}")


# 2. Simulate an actual physical error pattern (e.g., flip 3 random qubits)
num_qubits = H.shape[1]
true_error = np.zeros(num_qubits, dtype=np.uint8)
# Pick 3 random data qubits to suffer phase/bit flips
# error_indices = np.random.choice(num_qubits, size=3, replace=False)
error_indices = np.array([18, 23, 64]) # for testing purposes, do some clusters
# error_indices = np.array([10,11,12]) # for testing purposes, do some clusters
true_error[error_indices] = 1

# Calculate valid syndrome from physical error indices [18, 23, 64]
syndrome = (H @ true_error % 2).astype(np.uint8)
erasures = np.zeros(num_qubits, dtype=np.uint8)

# Run the updated decoder 
sizes, cluster_membership = decoder.ldpc_decode(syndrome, erasures)

print(f"Tracked Cluster Sizes: {sizes}")
print("Cluster Membership Maps:")
for cluster_idx, qubit_ids in cluster_membership.items():
    print(f"  -> Cluster {cluster_idx} (Size {len(qubit_ids)}) contains Qubit IDs: {qubit_ids}")

print("\n--------------------\n")
# Setup your batch inputs (e.g. 5 repetitions)
nrep = 5
batch_syndromes = np.zeros(nrep * decoder.n_syndr, dtype=np.uint8)
batch_erasures = np.zeros(nrep * decoder.n_qbt, dtype=np.uint8)

# Populate individual shots with simulated data ...

# Run the updated batch pipeline
batch_data = decoder.ldpc_decode_batch(batch_syndromes, batch_erasures, nrep)

# Unpack and verify results loop
for shot_id, (sizes, membership) in enumerate(batch_data):
    print(f"Shot {shot_id} -> Cluster Sizes found: {sizes}")
    for c_idx, qubits in membership.items():
        print(f"    Cluster {c_idx} contains Qubits: {qubits}")