import stim
import ldpc.codes as codes
from ldpc import BpOsdDecoder, UnionFindDecoder, BpDecoder
import numpy as np
from pymatching import Matching
from quits.decoder import detector_error_model_to_matrix
import relay_bp
from tesseract_decoder import tesseract
import py_wrapper.py_decoder as uf
from scipy.sparse import csr_matrix
from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.stim_tools import remove_detectors_from_circuit
from ldpc_post_selection.cluster_tools import compute_cluster_norm_fraction
from realtime_decoding.tesseract_w_sliding_window import chk_obs_priors_to_dem, get_dems_per_window

#################################################################################
#
#
# Decoding LER extraction functions
#
#
#################################################################################

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


#################################################################################
#
#
# Decoder construction and switching wrapper classes
#
#
#################################################################################

class RelayBpWrapper:
    def __init__(self, check_matrix, **params):
        if not isinstance(check_matrix, csr_matrix):
            check_matrix = csr_matrix(check_matrix)
        priors = params.get('priors')

        self.decoder = relay_bp.RelayDecoderF32( # filter by detectors if you don't wanna do full XYZ decoding. defaults are for the gross code
            check_matrix, priors, 
            gamma0=params.get('gamma0', 0.125), # initial memory strength vector for relay legs, 0.35 RSC, 0.125 Gross code [[144,12,12]]
            gamma_dist_interval=params.get('gamma_dist_interval', (-0.175, 0.575)), # random dist to draw gamma from [center - w/2, center + w/2], gross w = 0.75, c = 0.2, RSC w=0.8, c=0.3
            num_sets=params.get('num_sets', 600), # number of relay ensemble elements to tweak, R= 601 in paper :0
            set_max_iter=params.get('set_max_iter', 30), # max number of iterations of each relay leg, init 80, otherwise 60, 30 could be fine
            pre_iter=params.get('pre_iter', 80), # number max bp iter for first ensemble, init 80
            stop_nconv=params.get('stop_nconv', 1) # number of relay solutions to find before stopping (choose best, run up to num_sets when picking parameters)

        )
    def decode(self, syndrome):
        binary_syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        return self.decoder.decode(binary_syndrome)

class TesseractWrapper:
    def __init__(self, check_matrix, **kwargs):
        """
        Standardized wrapper for Google's Tesseract decoder to make it 
        compatible with the QUITS sliding window interface.
        """
        self.check_matrix = check_matrix
        
        # 1. Extract priors (injected dynamically by the sliding window loop)
        priors_key = kwargs.get('priors_key', 'priors')
        self.priors = kwargs.get(priors_key)
        
        # 2. Extract the correct window observable matrix using the tracker
        window_observables = kwargs.get('window_observables')
        if window_observables is not None:
            if 'window_index' in kwargs:
                idx = kwargs['window_index']
            elif 'window_index_tracker' in kwargs:
                tracker = kwargs['window_index_tracker']
                idx = tracker[0]
                tracker[0] += 1  # Move tracker to the next window position
            else:
                raise ValueError("TesseractWrapper requires 'window_index' or 'window_index_tracker'.")
            
            self.obs = window_observables[idx]
        else:
            raise ValueError("TesseractWrapper requires 'window_observables' to build the DEM.")
            
        self.det_beam = kwargs.get('det_beam', 10)
        
        # 3. Assemble the DEM using your colleague's function
        self.dem = chk_obs_priors_to_dem(self.check_matrix, self.obs, self.priors)
        
        # 4. Compile the Tesseract instance
        config = tesseract.TesseractConfig(dem=self.dem, det_beam=self.det_beam)
        self.decoder = config.compile_decoder()


    def decode(self, syndrome):
        """Alias to guarantee interoperability inside standard switching wrappers."""
        decoded_error_inds = self.decoder.decode_to_errors(syndrome)
        num_events = self.dem.num_errors
        decoded_errors = np.zeros(num_events, dtype=bool)
        decoded_errors[decoded_error_inds] = True
        return decoded_errors

class BPLSDWrapper:
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
        self.cluster_sizes = None
        self.cluster_gap = None
        # Explicitly map priors to the 'p' argument for BPLSD
        self.decoder = SoftOutputsBpLsdDecoder(H=check_matrix, p=priors, **bplsd_params)

    def decode(self, syndrome):
        # BPLSD returns: (correction, bp_correction, converged, soft_info)
        corr_bplsd, _, _, soft_info = self.decoder.decode(syndrome)
        cluster_gap = compute_cluster_norm_fraction(soft_info[self.metric_key], self.norm_order)
        self.cluster_sizes = soft_info.get('cluster_sizes')
        self.cluster_gap = cluster_gap
            
        return corr_bplsd

class UnionFindWrapper:
    def __init__(self, check_matrix, **params):
        """
        A sliding-window compatible wrapper for the custom nbi-hyq Union Find decoder.
        
        Parameters
        ----------
        check_matrix : array_like or csr_matrix
            The parity-check matrix (H) associated with the current decoding window.
        """
        if not isinstance(check_matrix, csr_matrix):
            check_matrix = csr_matrix(check_matrix)
            
        self.check_matrix = check_matrix
        
        # Initialize the underlying C-wrapped Union Find decoder instance
        self.decoder = uf.UFDecoder(self.check_matrix)
        
        # Public instance attribute for DecoderSwitchingWrapper to read
        self.last_cluster_sizes = None
        
    def decode(self, syndrome):
        """
        Decodes a window syndrome vector and returns a correction vector.
        
        Parameters
        ----------
        syndrome : np.ndarray
            The syndrome vector generated inside the current sliding window.
            
        Returns
        -------
        correction : np.ndarray
            The predicted correction vector extracted from the decoder instance.
        """
        # 1. Enforce strict contiguous data alignment types for the underlying C structures
        binary_syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        
        # 2. Build a matching blank erasure vector corresponding to the window size's qubits
        erasures = np.zeros(shape=self.check_matrix.shape[1], dtype=np.uint8)
        erasures = np.ascontiguousarray(erasures, dtype=np.uint8)
        
        # 3. Process data via the method to populate internal attributes
        found_cluster_sizes = self.decoder.ldpc_decode(binary_syndrome, erasures)
        
        # 4. Cache cluster stats metadata for the DecoderSwitchingWrapper pass
        self.last_cluster_sizes = found_cluster_sizes
        
        # 5. Extract the actual correction pattern produced by the decode logic
        correction = self.decoder.correction
        
        return correction

class DecoderSwitchingBPLSD:
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
    
class DecoderSwitchingWrapper:
    def __init__(self, primary_decoder_class, secondary_decoder_class, primary_params, secondary_params, cluster_metric, cutoff, count_container=None, norm_order=2):
        self.primary_decoder = primary_decoder_class(**primary_params)
        self.secondary_decoder = secondary_decoder_class(**secondary_params)
        self.metric_key = cluster_metric
        self.cutoff = cutoff
        self.norm_order = norm_order
        self.count_container = count_container # should just pass in a list with [0]

    def decode(self, syndrome):
        # TODO : write our own cluster norm fraction function independent of primary decoder class and add it here
        corr_primary, _, _, soft_info = self.primary_decoder.decode(syndrome)
        cluster_gap = compute_cluster_norm_fraction(soft_info[self.metric_key], self.norm_order)

        if cluster_gap > self.cutoff:
            if self.count_container is not None:
                self.count_container[0] += 1

            return self.secondary_decoder.decode(syndrome)
        return corr_primary


# the wrapper that I am writing to take in general decoder1
class DecoderSwitchingWrapperDraftGeneral:
    def __init__(self, primary_decoder_class, secondary_decoder_class, primary_params, secondary_params, cluster_metric, cutoff, count_container=None, norm_order=2):
        self.primary_decoder = primary_decoder_class(**primary_params)
        self.secondary_decoder = secondary_decoder_class(**secondary_params)
        self.metric_key = cluster_metric
        self.cutoff = cutoff
        self.norm_order = norm_order
        self.count_container = count_container # expected to be a mutable list like [0]

    def decode(self, syndrome):
        # 1. Execute the primary decoder pass
        primary_output = self.primary_decoder.decode(syndrome)
        
        # 2. Extract correction and soft info dynamically depending on the primary decoder class
        if isinstance(primary_output, tuple):
            # Handles BP-LSD style outputs: (correction, soft_outputs_dict, ...)
            corr_primary = primary_output[0]
            
            # Find the soft outputs dictionary (usually the second or last element)
            soft_info = None
            for item in primary_output[1:]:
                if isinstance(item, dict):
                    soft_info = item
                    break
            
            # Extract the user-defined metric key from the dictionary
            if soft_info is not None and self.metric_key in soft_info:
                cluster_data = soft_info[self.metric_key]
            else:
                # Fallback if dictionary or specific key is missing
                cluster_data = primary_output
        else:
            # Handles Custom Union-Find wrapper style outputs where the direct 
            # return value might be the raw array of cluster sizes or direct correction
            corr_primary = primary_output
            cluster_data = primary_output

        # 3. Compute cluster norm fraction with your universal metric module
        # Pass cluster_data, the raw syndrome, and order settings to maximize flexibility
        cluster_gap = universal_cluster_norm_fraction(
            cluster_data=cluster_data, 
            syndrome=syndrome, 
            metric_key=self.metric_key, 
            norm_order=self.norm_order
        )

        # 4. Evaluate switching condition
        if cluster_gap > self.cutoff:
            if self.count_container is not None:
                self.count_container[0] += 1
                
            # Switch to strong fallback decoder (e.g., RelayBpWrapper)
            secondary_output = self.secondary_decoder.decode(syndrome)
            
            # If secondary decoder returns a tuple, isolate the correction slice
            if isinstance(secondary_output, tuple):
                return secondary_output[0]
            return secondary_output
            
        return corr_primary