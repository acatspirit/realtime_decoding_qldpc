import numpy as np
import pandas as pd
import os
from filelock import FileLock
import relay_bp
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from ldpc_post_selection.decoder import SoftOutputsBpLsdDecoder
from ldpc_post_selection.cluster_tools import compute_cluster_norm_fraction
from quits.decoder import sliding_window_circuit_mem
import stim
from quits.qldpc_code import BbCode, BpcCode, HgpCode
from quits.decoder import sliding_window_bposd_circuit_mem,detector_error_model_to_matrix
from quits.simulation import get_stim_mem_result
from quits import ErrorModel
import numpy as np
from tqdm import tqdm



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


###########
# code construction helpers
###########
def get_rsc_circuits(p,d_list,basis, rds=None):
    """
    Use STIM to get rotated surface code circuits with circuit-level errors

    :param d_list: the list of rsc d's to run 
    :param basis: the memory experiment basis
    :param p: the probability of physical error on a qubit. Same p used for spam, idle, gate errors.

    :return: the list of circuits generated by STIM with these inputs
    """
    circuits = []
    code_info = []
    for d in d_list:
        if not rds:
            rds = d
        circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{basis}",rounds=rds, distance=d,
                                    after_clifford_depolarization=p,
                                    before_round_data_depolarization=p,
                                    before_measure_flip_probability=p,
                                    after_reset_flip_probability=p,
                                    )
        circuits += [circuit]
        code_info += [get_parity_and_logs_rsc(d,basis)]
    return circuits, code_info

def get_parity_and_logs_rsc(d, basis):
    """
    Use stim circuit to get the parity check matrix 
    """
    circuit = stim.Circuit.generated(
    f"surface_code:rotated_memory_{basis}",
    rounds=1,
    distance=d,
    after_clifford_depolarization=0,
    before_measure_flip_probability=0,
    after_reset_flip_probability=0,
    # Use data depolarization to define the H matrix columns (the qubits)
    before_round_data_depolarization=0.001 
    )

    # 2. Use the package's decoder to extract the matrices
    # The decoder internally converts the DEM to check/logical matrices
    bplsd = SoftOutputsBpLsdDecoder(
        circuit=circuit,
        detector_time_coords=0 # Only one round, so time coord isn't strictly needed here
    )

    h = bplsd.H # This is your code-level H matrix
    l = bplsd.obs_matrix   # This is your logical operator matrix
    return h, l

def get_BB_circuit(d, basis, p):
    d_dict = {6:{'l':6, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},
              10: {'l':15, 'm':3, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [2,7], 'B_y_pows':[0]},
              12:{'l':12, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},}
    code_params = d_dict[d]

    error_model = ErrorModel(
        idle_error=p,
        sqgate_error=p,
        tqgate_error=p,
        spam_error=p,
        )

    bb = BbCode(
        l=code_params['l'],
        m=code_params['m'],
        A_x_pows=code_params['A_x_pows'],
        A_y_pows=code_params['A_y_pows'],
        B_x_pows=code_params['B_x_pows'],
        B_y_pows=code_params['B_y_pows'],
    )

    custom_circuit = bb.build_circuit(strategy="custom", num_rounds=d, basis=basis, error_model=error_model) # num_rounds fixed to d for us
    labeled_circuit = fix_bb_circuit_for_sliding_window(custom_circuit, d)
    return labeled_circuit, bb

def fix_bb_circuit_for_sliding_window(original_circuit, num_rounds):
    """
    Flattens a BB circuit and injects (t,) coordinates for sliding window decoding.
    """
    # 1. Flatten to ensure each round's detectors can have unique time coordinates
    flattened = original_circuit.flattened()
    detectors_per_round = original_circuit.num_detectors//(num_rounds+2) # num_rounds should be d
    new_circuit = stim.Circuit()
    detector_count = 0

    for instr in flattened:
        if instr.name == "DETECTOR":
            time_step = detector_count // detectors_per_round
            
            # Append detector with the new coordinate [time_step]
            new_circuit.append(
                "DETECTOR", 
                instr.targets_copy(), 
                [time_step] # This becomes index 0 for the decoder
            )
            detector_count += 1
        else:
            new_circuit.append(instr)
            
    return new_circuit

# --- 3. The Unified Worker Function ---

def run_single_trial(p, d, cutoff, code_type, num_shots, W, F, basis, csv_filename):
    # 1. Circuit Generation Logic
    if code_type == "RSC":
        rsc_circuits, rsc_codes = get_rsc_circuits(p,[d],basis)
        circuit, code_params = rsc_circuits[0], rsc_codes[0]
        det_time_index = 2 # (x, y, t)
    else:
        bb_circuit, bb_code = get_BB_circuit(d, basis, p)
        circuit=bb_circuit
        if basis == 'x':
            code_params = bb_code.hx, bb_code.lx
        else:
            code_params = bb_code.hz, bb_code.lz
        det_time_index = 0 # (t,)

    sampler = circuit.compile_detector_sampler()
    det_events, obs_flips = sampler.sample(shots=num_shots, separate_observables=True)

    trial_switch_counter = [0]
    dict_SWITCH = {
        # --- BPLSD Params ---
        'bp_method': 'minimum_sum',
        'lsd_method': "LSD_0",
        'lsd_order': 0,
        'detector_time_coords': det_time_index,
        'max_iter': 10,
        
        # --- Switching Params ---
        'switching_cutoff': cutoff, 
        'metric_key': 'cluster_llrs', # or cluster_sizes
        'switch_count_container': trial_switch_counter,
        
        # --- Relay-BP Fallback Config ---
        'strong_decoder_class': RelayBpWrapper,
        'strong_decoder_params': {
            'num_sets': 300, # the number of relay ensemble elements R= 601 in paper :0... may be really big ... start with 300
            'gamma0': 0.125, # the initial memory strength , 0.35 RSC, 0.125 Gross code [[144,12,12]]
            'gamma_dist_interval': (-0.175, 0.575), # uniform distribution for range of memory weight selection, [center - w/2, center + w/2], gross w = 0.75, c = 0.2, RSC w=0.8, c=0.3
            'set_max_iter': 30, # max BP iterations per relay ensable, tested with 60
            'pre_iter': 80, # number max bp iter for first ensemble, init 80
            'stop_nconv': 1 # number of relay solutions to find before stopping (choose best, run up to num_sets when picking parameters)

        }
    }

    # 2. Decoding
    # Note: Using quits sliding window function
    logical_pred = sliding_window_circuit_mem(
        det_events, circuit, code_params[0], code_params[1],
        W, F, DecoderSwitchingWrapper, DecoderSwitchingWrapper,
        dict_SWITCH, dict_SWITCH, 'priors', 'priors', 'decode', 'decode'
    )    
    pL = np.mean((obs_flips - logical_pred).any(axis=1))

    # 3. Log Preparation
    row = {
        'LER': float(pL), 'cutoff': cutoff, 'p': p, 'd': d, 'basis':basis, 'cluster_metric':dict_SWITCH['metric_key'],
        'num_shots': num_shots, 'code_type': code_type, 'num_switches': trial_switch_counter[0]
    }
    for k, v in dict_SWITCH.items():
        if k not in ['strong_decoder_class', 'strong_decoder_params', 'switch_count_container', 'metric_key', 'priors', 'detector_time_coords']:
            row[f"bplsd_{k}"] = v
    for k, v in dict_SWITCH['strong_decoder_params'].items():
        row[f"strong_{k}"] = v
    
    # 2. Use a Lock to write to the CSV safely
    # This creates a '.lock' file. Processes will wait their turn to write.
    lock_path = csv_filename + ".lock"
    lock = FileLock(lock_path)
    
    with lock:
        df_row = pd.DataFrame([row])
        header_needed = not os.path.exists(csv_filename)
        # We open, write, and close the file immediately
        df_row.to_csv(csv_filename, mode='a', index=False, header=header_needed)
    
    return row

def run_cluster_task():
    # 1. Configuration
    results_dir = "decoder_switching_results"
    os.makedirs(results_dir, exist_ok=True)

    # Updated p_list (5 values)
    p_list = np.logspace(-4, -2, 5) 
    d_list = [6, 10, 12]
    W,F = 5,3
    # cutoff_list = [0.005, 0.007, 0.01, 0.05, 0.1]
    cutoff_list = [0] # for when just running relay or bplsd
    code_types = ["BB"]
    basis = 'x'
    
    # Updated to match length of p_list (5 values)
    total_target_shots = [10**6, 10**6, 10**5, 10**5, 10**4]
    p_to_total_shots = dict(zip(p_list, total_target_shots))
    shots_per_job = 10000 

    # 2. Map Slurm Array ID to Parameters (Breadth-First)
    tasks = []
    
    # Calculate max batches needed by any single p
    max_batches_needed = max(int(target // shots_per_job) for target in total_target_shots)
    if max_batches_needed == 0: max_batches_needed = 1

    # Breadth-First loop: Iterate through batches first 
    # to ensure all p/d/cutoff combos start getting data immediately.
    for batch_idx in range(max_batches_needed):
        for p in p_list:
            num_jobs_for_this_p = int(p_to_total_shots[p] // shots_per_job)
            if num_jobs_for_this_p == 0: num_jobs_for_this_p = 1
            
            # Only add if this specific p requires more batches
            if batch_idx < num_jobs_for_this_p:
                for d in d_list:
                    for cutoff in cutoff_list:
                        for ct in code_types:
                            tasks.append({
                                'p': p, 'd': d, 'cutoff': cutoff, 
                                'code_type': ct, 'batch_idx': batch_idx
                            })
   
    # print(len(tasks) - 1)
    # 3. Execution
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if task_id >= len(tasks): 
        return
        
    task = tasks[task_id]
    p, d, cutoff, code_type = task['p'], task['d'], task['cutoff'], task['code_type']

    # --- Unique Output Filename (No locking needed) ---
    out_file = f"{results_dir}/res_p{p:.5f}_d{d}_c{cutoff}_batch{task['batch_idx']}_id{task_id}.csv"
    if os.path.exists(out_file): return # Skip if this specific batch file already exists


    # Generate and Sample
    if code_type == "RSC":
        rsc_circuits, rsc_codes = get_rsc_circuits(p, [d], basis)
        circuit, code_params = rsc_circuits[0], rsc_codes[0]
        det_time_index = 2
    else:
        circuit, bb_code = get_BB_circuit(d, basis, p)
        code_params = (bb_code.hx, bb_code.lx) if basis == 'x' else (bb_code.hz, bb_code.lz)
        det_time_index = 0

    sampler = circuit.compile_detector_sampler()
    det_events, obs_flips = sampler.sample(shots=shots_per_job, separate_observables=True)

    trial_switches = [0]
    # dict_SWITCH = {
    # 'max_iter': 10, 'detector_time_coords': det_time_index,
    # 'switching_cutoff': cutoff, 'switch_count_container': trial_switches,
    # 'bp_method': 'minimum_sum',
    # 'lsd_method': 'LSD_0',
    # 'lsd_order': 0,
    # 'strong_decoder_class': RelayBpWrapper,
    # 'strong_decoder_params': {
    #     'num_sets': 300, # the number of relay ensemble elements R= 601 in paper :0... may be really big ... start with 300
    #     'gamma0': 0.125, # the initial memory strength , 0.35 RSC, 0.125 Gross code [[144,12,12]]
    #     'gamma_dist_interval': (-0.175, 0.575), # uniform distribution for range of memory weight selection, [center - w/2, center + w/2], gross w = 0.75, c = 0.2, RSC w=0.8, c=0.3
    #     'set_max_iter': 30, # max BP iterations per relay ensable, tested with 60
    #     'pre_iter': 80, # number max bp iter for first ensemble, init 80
    #     'stop_nconv': 1 # number of relay solutions to find before stopping (choose best, run up to num_sets when picking parameters)
    #     }
    # }

    dict_RELAY = {
        'num_sets': 300, # the number of relay ensemble elements R= 601 in paper :0... may be really big ... start with 300
        'gamma0': 0.125, # the initial memory strength , 0.35 RSC, 0.125 Gross code [[144,12,12]]
        'gamma_dist_interval': (-0.175, 0.575), # uniform distribution for range of memory weight selection, [center - w/2, center + w/2], gross w = 0.75, c = 0.2, RSC w=0.8, c=0.3
        'set_max_iter': 30, # max BP iterations per relay ensable, tested with 60
        'pre_iter': 80, # number max bp iter for first ensemble, init 80
        'stop_nconv': 1 # number of relay solutions to find before stopping (choose best, run up to num_sets when picking parameters)
        }

    # Decode
    # logical_pred = sliding_window_circuit_mem(
    #     det_events, circuit, code_params[0], code_params[1], W, F, 
    #     DecoderSwitchingWrapper, DecoderSwitchingWrapper,
    #     dict_SWITCH, dict_SWITCH, 'priors', 'priors', 'decode', 'decode'
    # )

    logical_pred = sliding_window_circuit_mem(
        det_events, circuit, code_params[0], code_params[1], W, F, 
        RelayBpWrapper, RelayBpWrapper,
        dict_RELAY, dict_RELAY, 'priors', 'priors', 'decode', 'decode'
    )


    pL = np.mean((obs_flips - logical_pred).any(axis=1))
    
    # Save Single Batch Result
    row = {
        'LER': float(pL), 
        'cutoff': cutoff, 
        'p': p, 
        'd': d, 
        'code_type': code_type,
        'num_shots': shots_per_job, 
        'num_switches': trial_switches[0], 
        'basis': basis,
        
        # Metadata: Use .get() for everything to avoid KeyErrors
        'cluster_metric': 'llr',
        # 'bplsd_bp_method': dict_SWITCH.get('bp_method', 'minimum_sum'),
        # 'bplsd_lsd_method': dict_SWITCH.get('lsd_method', 'LSD_0'),
        # 'bplsd_lsd_order': dict_SWITCH.get('lsd_order', 0), # Changed from dict_SWITCH['lsd_order']
        # 'bplsd_max_iter': dict_SWITCH.get('max_iter', 10),
        # 'bplsd_switching_cutoff': cutoff,
        
        # 'strong_num_sets': dict_SWITCH['strong_decoder_params'].get('num_sets'),
        # 'strong_gamma0': dict_SWITCH['strong_decoder_params'].get('gamma0'),
        # 'strong_gamma_dist_interval': dict_SWITCH['strong_decoder_params'].get('gamma_dist_interval'),
        # 'strong_relay_max_iter': dict_SWITCH['strong_decoder_params'].get('relay_max_iter', 30)

        'strong_num_sets': dict_RELAY.get('num_sets'),
        'strong_gamma0': dict_RELAY.get('gamma0'),
        'strong_gamma_dist_interval': dict_RELAY.get('gamma_dist_interval'),
        'strong_relay_max_iter': dict_RELAY.get('relay_max_iter', 30)
    }

    pd.DataFrame([row]).to_csv(out_file, index=False)

if __name__ == "__main__":
    run_cluster_task()

