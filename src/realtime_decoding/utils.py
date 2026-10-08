import stim
import numpy as np
from quits.decoder import detector_error_model_to_matrix

def spacetime_w_window_ranges(circuit, hz, W, F, num_cor_rounds):
    '''
    Obtain the spacetime slices of detector error matrix for sliding window decoder

    :param circuit: stim circuit
    :param hz: X or Z error parity check matrix of the code.
    :param W: Width of sliding window
    :param F: Width of overlap between consecutive sliding windows
    :param num_cor_rounds: number of windows before the last window

    :return window_check_set: a set of sliced spacetime detector error matrix for each window in sliding window decoder
    :return window_observable_set: a set of sliced observable matrix that marks the observable flips of each fault in each window
    :return window_priors_set: a set of probability for faults in each window
    :return window_update: the detector information update for next window of each fault mechanism in each window
    :return window_ranges: the indices to slice the priors
    '''
    if F == 0:
        raise ValueError("Input parameter F cannot be zero.")
    model = circuit.detector_error_model(decompose_errors=False)  # detector error model of the circuit
    check_matrix, observable_matrix, priors = detector_error_model_to_matrix(model)
    window_check_set = []
    window_observable_set = []
    window_priors_set = []
    window_update = []
    col_min = 0

    window_ranges = [] 
    '''Check_matrix for each window'''
    for k in range(num_cor_rounds):
        window_check_matrix = check_matrix[k * F * hz.shape[0]:(k * F + W) * hz.shape[0], col_min:]
        if len(window_check_matrix.indptr) == 1:
            raise ValueError("There is no noise in one of the decoding window. This means there are redundant detectors that do not check for any error.")
        col_max = np.max(np.where(np.diff(window_check_matrix.indptr) > 0)[0])  # all the columns that affect the window
        window_check_matrix = window_check_matrix[:, :col_max + 1]
        window_check_set.append(window_check_matrix)

        '''corresponding flips of observables: only care about the part we fix'''
        F_correction = window_check_matrix[:F * hz.shape[0], :]
        cor_max = np.max(np.where(np.diff(F_correction.indptr) > 0)[0])
        window_observable_matrix = observable_matrix[:, col_min:cor_max + 1 + col_min]
        window_observable_set.append(window_observable_matrix)

        window_ranges.append((col_min, col_max + 1 + col_min))

        '''probability of each fault'''
        window_priors = priors[col_min:col_max + 1 + col_min]
        window_priors_set.append(window_priors)
        '''updating the detector flips for the next window'''
        updated_info = check_matrix[(k + 1) * F * hz.shape[0]:((k + 1) * F + 1) * hz.shape[0], col_min:cor_max + 1 + col_min]
        col_min = (cor_max + 1) + col_min
        window_update.append(updated_info)
    
    '''last window check matrix'''
    last_window_check_matrix = check_matrix[F * num_cor_rounds * hz.shape[0]:, col_min:]
    window_check_set.append(last_window_check_matrix)
    '''last window observable flip'''
    last_window_observable_matrix = observable_matrix[:, col_min:]
    window_observable_set.append(last_window_observable_matrix)
    '''last window prior'''
    last_window_priors = priors[col_min:]
    window_priors_set.append(last_window_priors)

    window_ranges.append((col_min,len(priors)))

    return window_check_set, window_observable_set, window_priors_set, window_update,window_ranges


def chk_obs_priors_to_dem(chk,obs,priors):
    '''Get the DEM given the parity check matrix, observables matrix and priors.
    
    Inputs:
    chk: detector error matrix (or parity check matrix) (size # of dets x of faults)
    obs: 0/1 array of logical observables (whether the logical observable is flipped by the k-th fault) (size # of observables x # of faults)
    priors: probs per fault (for each column of chk, the detectors in the k-th fault column and the observables in the k-th fault column which are 1, are flipped with some prob)

    Outputs:
    DEM: detector error model
    '''

    num_faults   = np.shape(chk)[1]
    DEM          = stim.DetectorErrorModel()
    
    for k in range(num_faults):
        
        dets_flipped = chk[:,k]
        error_prob   = priors[k]

        dets_flipped = np.nonzero(dets_flipped)[0]

        targets = [stim.target_relative_detector_id(t) for t in dets_flipped]

        #check the nnz obs
        try:
            obs_flipped = np.nonzero(obs[:,k])[0]
        except IndexError: #if out of bounds, observable is not flipped for all subsequent error mechanisms (see spacetime function in quits)
            DEM.append("error",error_prob,targets)
            continue

        for l in obs_flipped:
            targets.append(stim.target_logical_observable_id(l))
        DEM.append("error",error_prob,targets)

    return DEM


def get_window_dems(window_check_set, window_observable_set, window_priors_set):
    '''Get the DEMs per window, given check matrix, observable matrix and priors per window.
    
    Input:
    window_check_set: list of check matrices per window
    window_observable_set: list of observable matrices per window
    window_priors_set: list of priors per window

    Outputs:
    window_dems_set: list of dems per window
    '''

    N = len(window_check_set)
    
    window_dems_set = []

    for k in range(N):

        chk = window_check_set[k]
        obs = window_observable_set[k].copy()
        priors = window_priors_set[k]
        dem = chk_obs_priors_to_dem(chk,obs,priors)
        window_dems_set.append(dem)

    return window_dems_set


def get_erased_mechanisms_from_DEM(dem:stim.DetectorErrorModel, p_cutoff = 0.03):
    """ Given a DEM, get the error mechanisms where erasures occurred. This is set by a cutoff, since erasures
        produce an error mechanism with p=0.5. General rule - cutoff should be p_max error?
    """ 

    erased_errors = np.zeros(shape=(dem.num_errors,)).flatten()
    i = 0

    for inst in dem:
        if inst.type == "error":
            if inst.args_copy()[0] > p_cutoff:
                erased_errors[i] = 1

            i+= 1

    return erased_errors

def get_erased_mechanisms_from_priors(window_prior_set,cutoff=0.2):
    #len of priors is num_faults

    # erased_errors_set = []

    # for k in range(len(window_prior_set)):

    #     priors = window_prior_set[k]
    #     num_faults    = np.shape(priors)[0]
    #     erased_errors = np.zeros(shape=(num_faults,)).flatten()

    #     locs = priors>cutoff 
    #     erased_errors[locs] = 1
    #     erased_errors_set.append(erased_errors)

    erased_errors_set = [
        (priors > cutoff).astype(np.uint8)
        for priors in window_prior_set
    ]
        

    return erased_errors_set


def get_erasure_set(window_check_set, window_observable_set, window_prior_set):
    
    
    # window_dems_set = get_window_dems(window_check_set, window_observable_set, window_prior_set)
    
    # erased_errors_set = []
    # for dem in window_dems_set:
    #     erased_errors = get_erased_mechanisms_from_DEM(dem)
    #     erased_errors_set.append(erased_errors)

    erased_errors_set_alt = get_erased_mechanisms_from_priors(window_prior_set,cutoff=0.2)

    # for m in range(len(erased_errors_set)):
    #     one = erased_errors_set[m]
    #     two = erased_errors_set_alt[m]

    #     if (one != two).any():
    #         raise Exception("ERROR.")

    return erased_errors_set_alt #erased_errors_set


def slice_erasures_to_windows(window_ranges, erasures_total, cutoff = 0.2):
    '''
    Inputs:
        window_ranges: a list of tuples (r[0],r[1]) indicating how we should slice windows -- extracted from spacetime function 
        erasures_total: a vector of 0/1 of size equal to the total num of errors in the entire dem/pcm
        cutoff: cutoff prob (0.5 should correspond to erasures)

    Output:
        erased_error_set: erasures split per window.
    '''

    erased_errors_set = [
        (erasures_total[r[0]:r[1]] > cutoff).astype(np.uint8)
        for r in window_ranges
    ]

    return erased_errors_set