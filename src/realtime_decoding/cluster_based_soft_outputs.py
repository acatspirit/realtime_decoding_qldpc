from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.stim_tools import remove_detectors_from_circuit
from ldpc_post_selection.cluster_tools import compute_cluster_norm_fraction
import stim
import numpy as np
from tqdm import tqdm

def get_cluster_soft_output_from_bplsd_glocal_decoding(circuit:stim.Circuit,  cluster_method: str, order, 
                                                       det_events, obs_flips):
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

    
    #Some of these parameters can change
    bplsd = SoftOutputsBpLsdDecoder(
        circuit=circuit,
        max_iter=30,
        bp_method="minimum_sum",
        lsd_method="LSD_0",
        lsd_order=0,
        ms_scaling_factor=1.0,
    )


    fails = []
    norm_fracs = []
    shots = np.shape(det_events)[0]

    for i_sample in tqdm(list(range(shots))):
        correction, _, _, soft_outputs = bplsd.decode(det_events[i_sample])
        obs_correction = correction @ bplsd.obs_matrix.T % 2
        fail = np.any(obs_flips[i_sample] != obs_correction)
        norm_frac = compute_cluster_norm_fraction(soft_outputs[cluster_method], order)
        fails.append(fail)
        norm_fracs.append(norm_frac)

    return fails, norm_fracs


