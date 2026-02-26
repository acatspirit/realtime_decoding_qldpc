

import numpy as np
import ldpc.codes
from ldpc import BpOsdDecoder, UnionFindDecoder

H = ldpc.codes.hamming_code(5)

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
union_find = UnionFindDecoder(pcm=H, uf_method='True')

#according to their docstring this is what I get for the two parameters we can pass in:
"""
pcm : Union[np.ndarray, spmatrix]
      The parity-check matrix (PCM) of the code. This should be either a dense matrix (numpy ndarray)
      or a sparse matrix (scipy sparse matrix).
uf_method : bool, optional
      If True, the decoder operates in matrix solve mode. If False, it operates in peeling mode.
      Default is False.
"""

syndrome = np.random.randint(size=H.shape[0], low=0, high=2).astype(np.uint8)
print(f"Syndrome: {syndrome}")
decoding = union_find.decode(syndrome)
print(f"Decoding: {decoding}")
decoding_syndrome = H @ decoding % 2
print(f"Decoding syndrome: {decoding_syndrome}")
