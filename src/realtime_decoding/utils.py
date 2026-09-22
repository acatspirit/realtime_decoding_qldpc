import stim
import numpy as np

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
            else:
                erased_errors[i] = 0
            i+= 1

    return erased_errors


def get_erasure_set(window_check_set, window_observable_set, window_prior_set):
    window_dems_set = get_window_dems(window_check_set, window_observable_set, window_prior_set)
    erased_errors_set = []
    for dem in window_dems_set:
        erased_errors = get_erased_mechanisms_from_DEM(dem)
        erased_errors_set.append(erased_errors)
    return erased_errors_set
