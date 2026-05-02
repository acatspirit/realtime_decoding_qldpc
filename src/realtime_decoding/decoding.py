import stim
import ldpc.codes as codes
from ldpc import BpOsdDecoder, UnionFindDecoder, BpDecoder
import numpy as np
from pymatching import Matching
from quits.decoder import detector_error_model_to_matrix
from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.stim_tools import remove_detectors_from_circuit
from ldpc_post_selection.cluster_tools import compute_cluster_norm_fraction


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


class DecoderSwitchingWrapper:
    def __init__(self, check_matrix, **params):
        """
        A generic wrapper that attempts BPLSD first and switches to 
        'strong_decoder_class' if cluster_gap is above the cutoff that is input in the dictionary.
        """
        # 1. Capture the spacetime priors provided by QUITS
        priors = params.get('priors')

        # 2. Initialize the Primary Decoder (BPLSD)
        bplsd_keys = ['max_iter', 'bp_method', 'lsd_method', 'lsd_order', 
                      'ms_scaling_factor', 'detector_time_coords']
        bplsd_params = {k: params[k] for k in bplsd_keys if k in params}
        # Explicitly map priors to the 'p' argument for BPLSD
        self.primary_decoder = SoftOutputsBpLsdDecoder(H=check_matrix, p=priors, **bplsd_params)
        
        # 3. Setup the strong_decoder 
        # We look for the class object in the params dict
        self.strong_decoder_class = params.get('strong_decoder_class')
        
        if self.strong_decoder_class:
            # Extract strong_decoder-specific arguments
            s_params = params.get('strong_decoder_params', {}).copy()
            
            # Most strong_decoder decoders (like MWPM or Relay-BP) need the priors
            # We inject them into the strong_decoder's parameters
            if 'priors' not in s_params:
                s_params['priors'] = priors
                
            # Initialize the strong_decoder decoder instance
            self.strong_decoder = self.strong_decoder_class(check_matrix, **s_params)

        # 4. Switching Logic Parameters
        self.cutoff = params.get('switching_cutoff', 0.001)
        self.metric_key = params.get('metric_key', 'cluster_llrs')
        self.verbose = params.get('verbose_switch', False)
        self.count_container = params.get('switch_count_container')
        self.norm_order = params.get('norm_order', 2)

    def decode(self, syndrome):
        """
        Attempt BPLSD; fall back to the secondary decoder if certainty is low.
        """
        # BPLSD returns: (correction, bp_correction, converged, soft_info)
        corr_bplsd, _, _, soft_info = self.primary_decoder.decode(syndrome)
        cluster_gap = compute_cluster_norm_fraction(soft_info[self.metric_key], self.norm_order)
        
        if self.strong_decoder is not None and cluster_gap > self.cutoff:
            # Increment the shared counter
            if self.count_container is not None:
                self.count_container[0] += 1
                
            if self.verbose:
                print(f"[Switch] cluster_gap {cluster_gap:.2f} > {self.cutoff}. Using strong_decoder.")
            
            # Call the strong_decoder's decode method
            return self.strong_decoder.decode(syndrome)
            
        return corr_bplsd