import stim
import ldpc.codes as codes
from ldpc import BpOsdDecoder, UnionFindDecoder, BpDecoder
import numpy as np
from pymatching import Matching
from quits.decoder import detector_error_model_to_matrix

def BP_MWPM(syndrome, H, L, error_priors, max_iter, t, dem = None):
    print(H.shape)
    bp = BpDecoder(
                H,
                error_channel= error_priors,
                bp_method = 'product_sum',
                max_iter = max_iter,
                schedule = 'serial',
    )

    bp.decode(syndrome)
    llrs = bp.log_prob_ratios
    p_post = 1/(1+np.exp(llrs))

    error_p = (p_post > t).astype(np.uint8)
    syndrome_bp = H@error_p % 2
    syndrome_p = np.logical_xor(syndrome,syndrome_bp).astype(np.uint8)

    if not np.any(syndrome_p):
        total_correction = error_p
    else:
        matching = Matching(H) if dem is None else Matching.from_detector_error_model(dem)
        correction = matching.decode(syndrome_p)
        total_correction = np.logical_xor(error_p, correction).astype(np.uint8)
    return total_correction

def get_log_error_CL_BP_MWPM(p,d,t,max_iter, memory_type, shots):
    circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{memory_type}",rounds=d, distance=d,
                                    after_clifford_depolarization=p,
                                    before_round_data_depolarization=p,
                                    before_measure_flip_probability=p,
                                    after_reset_flip_probability=p) # noise model kinda like the paper I guess
    num_errors = 0

    for k in range(shots):
        if k % 1000 == 0:
            print(f"Shot {k}/{shots} BP+MWPM", flush=True)
        dem = circuit.detector_error_model() # not sure whether I should be decomposing errors etc
        detector_error_matrix, observables_matrix,priors = detector_error_model_to_matrix(dem)
        sampler = circuit.compile_detector_sampler()
        detection_events, obs_flips = sampler.sample(shots=shots, separate_observables=True)
        total_correction = BP_MWPM(detection_events[k,:], detector_error_matrix, observables_matrix, priors, max_iter=max_iter, t=t, dem=dem)
        predicted_flip = observables_matrix@total_correction % 2
        num_errors += not np.array_equal(predicted_flip, obs_flips[k,:]) # shot 0
    return num_errors / shots


def get_log_error_CL_MWPM(p,d,memory_type, shots):
    circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{memory_type}",rounds=d, distance=d,
                                    after_clifford_depolarization=p,
                                    before_round_data_depolarization=p,
                                    before_measure_flip_probability=p,
                                    after_reset_flip_probability=p) # noise model kinda like the paper I guess
    dem = circuit.detector_error_model()
    matching = Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    syndrome, obs_flips = sampler.sample(shots=shots, separate_observables=True)
    predictions = matching.decode_batch(syndrome)
    return np.sum(np.array(obs_flips) != np.array(predictions))/shots
