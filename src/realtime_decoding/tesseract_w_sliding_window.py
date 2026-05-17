from tesseract_decoder import tesseract
from quits.decoder import spacetime 
import stim
import numpy as np
import warnings
import tqdm


def chk_obs_priors_to_dem(chk,obs,priors):
    '''
    Convert detector error matrix to dem.

    Inputs:
        chk: 0/1 detector error matrix (# of detectors x # of circuit faults)
        obs: 0/1 observable matrix indicating whether the k-th fault flips each logical (# of logical qubits x # of circuit faults)
        priors: the error probabilities of each error mechanism in the circuit

    Outputs:
        DEM: stim' detector error model (only w/ error instructions i.e., no detector annotations, ticks etc)
    '''

    num_faults = np.shape(chk)[1]
    DEM        = stim.DetectorErrorModel()
    
    for k in range(num_faults):
        
        dets_flipped = chk[:,k]
        error_prob   = priors[k]

        dets_flipped = np.nonzero(dets_flipped)[0]

        targets = [stim.target_relative_detector_id(t) for t in dets_flipped]

        #check if observables are flipped by the error mechanism
        try:
            obs_flipped = np.nonzero(obs[:,k])[0]
        except IndexError: #if out of bounds, observable is not flipped for all subsequent error mechanisms (see spacetime function in quits)
            DEM.append("error",error_prob,targets)
            continue

        for l in obs_flipped:
            targets.append(stim.target_logical_observable_id(l))

        DEM.append("error",error_prob,targets)

    return DEM

def get_dems_per_window(window_check_set, window_observable_set, window_priors_set):
    '''
    Reconstruct the detector error models given check,obs,and error probabilities per window.
    The window parameters are obtained from example according to the spacetime function in quits.

    Inputs:
        window_check_set: list of detector error matrices for the various windows. (each is of size # of detectors x # of circuit faults)
        window_observable_set: list of 0/1 arrays dictating whether logical observables are flipped by circuit fault  (each is of size # of logical qubits x # of circuit faults)
        window_priors_set: the error probabilities for each circuit fault
    Outputs:
        windowed_dems_set: the reconstructed dem per window (list of dems)

    '''

    N = len(window_check_set)
    dems_per_window = []

    for k in range(N):

        chk    = window_check_set[k]
        obs    = window_observable_set[k].copy()
        priors = window_priors_set[k]

        # old method to pad 0s so that column size is the same for obs and chk
        # num_faults_in_check = np.shape(chk)[1]
        # num_faults_in_obs   = np.shape(obs)[1]

        # if num_faults_in_obs<num_faults_in_check: #pad with zeros since missing faults (which don't flip the observable) are in the end (see spacetime function)
            
        #     n_add = num_faults_in_check - num_faults_in_obs
        #     obs.resize((obs.shape[0], obs.shape[1] + n_add))
        #     new_indptr = np.pad(obs.indptr, (0, n_add), mode='edge')
        #     obs.indptr = new_indptr            

        dem = chk_obs_priors_to_dem(chk,obs,priors)
        dems_per_window.append(dem)

    return dems_per_window

def sliding_window_circuit_mem_tesseract(zcheck_samples, circuit: stim.Circuit, hz, lz, W: int, F: int, det_beam: int, tqdm_on=False):
    '''
    Implementation of sliding window decoder for tesseract, under circuit-level noise model.

    Inputs:
        zcheck_samples: detection events from stim circuit (restricted only on Z/X detectors) (size # of shots x # of detectors)
        circuit: the stim circuit
        hz: parity check matrix in code-capcity representing the Z/X stabilizers of the code
        lz: logical codeword matrix of the qec code (size # of logical qubits x # of data qubits)
        W: width of sliding window
        F: width of overlap between consecutive sliding windows
        det_beam: the beam search option for tesseract decoder
        tqdm_on: True/False: evaluating the iteration runtime

    Outputs:
        logical_z_pred: Decoder's prediction of whether the logical Z codewords flipped. Shape (# trials, # logical quits)

    '''

    num_trials = zcheck_samples.shape[0]
    num_rounds = zcheck_samples.shape[1] // hz.shape[0] - 2

    # update the total number of windows for decoding, the size of the last window
    if 2 + num_rounds - W >= 0:
        num_cor_rounds = (2 + num_rounds - W) // F  # num_cor_rounds=num of windows before the last window
        if (2 + num_rounds - W) % F != 0:  # we can slide one more window if the remaining rounds>W
            num_cor_rounds += 1
    else:
        num_cor_rounds = 0
        warnings.warn("Window size larger than the syndrome extraction rounds: Doing whole history correction")
    W_last = num_rounds + 2 - F * num_cor_rounds
    # update the window matrix and the decoder
    # spacetime detector error matrix
    window_check_set, window_observable_set, window_priors_set, window_update = spacetime(circuit, hz, W, F, num_cor_rounds)
    
    dems_per_window = get_dems_per_window(window_check_set, window_observable_set, window_priors_set)
    
    # start decoding
    if tqdm_on:
        iterator = tqdm(range(num_trials))
    else:
        iterator = range(num_trials)
    
    logical_z_pred = np.zeros((num_trials, lz.shape[0]), dtype=int)

    
    #Compile the decoder only per window, it's fixed per shot
    compiled_decoders = []
    for k in range(num_cor_rounds+1):
        config  = tesseract.TesseractConfig(dem=dems_per_window[k], det_beam=det_beam)
        decoder = config.compile_decoder()
        compiled_decoders.append(decoder)

    for i in iterator:  # each sample decoding
        accumulated_correction = np.zeros(window_observable_set[0].shape[0], dtype=int)
        syn_update = np.zeros(hz.shape[0], dtype=int)

        for k in range(num_cor_rounds):
            # syndrome of the window
            diff_syndrome = (zcheck_samples[i, F * k * hz.shape[0]:(F * k + W) * hz.shape[0]].copy()) % 2
            diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2  # update the syndrome based on the previous window decoding
            
            
            decoded_error_inds = compiled_decoders[k].decode_to_errors(diff_syndrome) #decode_to_errors of tesseract gives indices,so convert to T/F with total size # of circuit faults
            num_events_in_DEM  = dems_per_window[k].num_errors
            decoded_errors = np.zeros(num_events_in_DEM, dtype=bool)
            decoded_errors[decoded_error_inds] = True            

            correction = window_observable_set[k] @ decoded_errors[:window_observable_set[k].shape[1]] % 2  # interpret the correction operation as final observable flips

            syn_update = window_update[k] @ decoded_errors[:window_observable_set[k].shape[1]] % 2
            accumulated_correction = (accumulated_correction + correction) % 2

        # In the last round we just correct the whole window
        # syndrome of last round
        diff_syndrome = (zcheck_samples[i, (F * num_cor_rounds) * hz.shape[0]:].copy()) % 2
        diff_syndrome[:hz.shape[0]] = (diff_syndrome[:hz.shape[0]] + syn_update) % 2
        # Observable flips based on correction
        

        correction = compiled_decoders[-1].decode(diff_syndrome) #decode directly to observable flips since we dont have to do another synd_update
        accumulated_correction = (accumulated_correction + correction) % 2
        # Predicted observable flips
        logical_z_pred[i, :] = accumulated_correction

    return logical_z_pred
