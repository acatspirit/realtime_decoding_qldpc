from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.stim_tools import remove_detectors_from_circuit
from ldpc_post_selection.cluster_tools import compute_cluster_norm_fraction
import stim
import numpy as np
from tqdm import tqdm
import collections

def get_cluster_soft_output_from_bplsd_glocal_decoding(circuit:stim.Circuit,  cluster_method: str, order, 
                                                       det_events, obs_flips, decoder=None,max_iter=30, bp_method = "minimum_sum", lsd_method="LSD_0", lsd_order=0, ms_scaling_factor=1.0 ):
    '''
    Get cluster size or cluster llr soft output from bplsd decoder. Note that if we run Z-memory (X-memory) we need to 
    restrict the circuit to Z (X) detectors only to get the confidence based only on the memory type we correct.
    Note that these cluster based metrics scale inversely to decoding confidence
    (e.g. small cluster_size better expected decoding performance).
    
    Inputs:

    circuit: the stim circuit
    cluster_method: "cluster_sizes" or "cluster_llrs" to choose the log-likelihood ratio or the cluster size
    order: 1 for 1-norm, 2 for 2-norm etc (positive int), and can also be np.inf 
    det_events: detection events from stim circuit to decode and compute the soft output metric
    obs_flips: observable flips from stim circuit

    Outputs:
    fails: list of True/False for BP-LSD decoding failures
    norm_fracs: list of norm fraction values per detection event
    '''

    if cluster_method!="cluster_sizes" and cluster_method!="cluster_llrs":
        raise ValueError("The cluster method can be cluster_sizes or cluster_llrs only.")
    

    # detector_coords = circuit.get_detector_coordinates()
    # det_ids_to_remove = []  # X-type detectors
    # for det_id, (x, y, z) in detector_coords.items():
    #     if (round(x) + round(y)) % 4 == 2:
    #         det_ids_to_remove.append(det_id)
    # circuit = remove_detectors_from_circuit(circuit, det_ids_to_remove)

    if not decoder:
        #Some of these parameters can change
        decoder = SoftOutputsBpLsdDecoder(
            circuit=circuit,
            max_iter=max_iter,
            bp_method=bp_method,
            lsd_method=lsd_method,
            lsd_order=lsd_order,
            ms_scaling_factor=ms_scaling_factor,
        )


    fails = []
    norm_fracs = []
    shots = np.shape(det_events)[0]

    for i_sample in tqdm(list(range(shots))):
        correction, _, _, soft_outputs = decoder.decode(det_events[i_sample])
        obs_correction = correction @ decoder.obs_matrix.T % 2
        fail = np.any(obs_flips[i_sample] != obs_correction)
        norm_frac = compute_cluster_norm_fraction(soft_outputs[cluster_method], order)
        fails.append(fail)
        norm_fracs.append(norm_frac)

    # errors = fails here, just use the comp gap conditioned failure directly 

    # Classify all shots by their error + gap.
    custom_counts = collections.Counter()
    norm_fracs  = np.round(norm_fracs).astype(dtype=np.int64)
    for k in range(len(norm_fracs)):
        g = norm_fracs[k]
        key = f'E{g}' if fails[k] else f'C{g}'
        custom_counts[key] += 1/shots

    # P_L(e | g) = E_g / (E_g + C_g) -> frac conditioned logical error rate

    frac_conditioned_PL = {}

    # collect all gap values that appear
    gaps = set()
    for key in custom_counts:
        gaps.add(int(key[1:]))

    for g in gaps:
        E = custom_counts.get(f'E{g}', 0.0)
        C = custom_counts.get(f'C{g}', 0.0)

        if E + C > 0:
            frac_conditioned_PL[g] = E / (E + C)
        else:
            frac_conditioned_PL[g] = np.nan

    return fails, norm_fracs, frac_conditioned_PL



