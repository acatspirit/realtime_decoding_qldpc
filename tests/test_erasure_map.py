import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move one level back out of tests folder

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

import numpy as np
from src.realtime_decoding.decoder_switching_class import decoder_switching_class
from joblib import Parallel, delayed
from tesseract_decoder import tesseract
from src.realtime_decoding.circuits import create_bb_codes_circuit_ionic_model, add_erasures
import warnings

from quits.decoder import detector_error_model_to_matrix


def spacetime(circuit, hz, W, F, num_cor_rounds):
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

    ranges = []
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

        ranges.append((col_min, col_max + 1 + col_min))

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

    ranges.append((col_min,len(priors)))

    return window_check_set, window_observable_set, window_priors_set, window_update,ranges


basis      = 'Z'
code_name  = "[[144,12,12]]"
n, k, d    = map(int, code_name.strip("[]").split(","))


p = 1e-3
Re        = 0.95
p_pauli   = (1-Re)*p
p_erasure = Re * p 

num_rounds = 25
nbuffer = d            #Buffer region
F       = d//2         #Commit region
W       = nbuffer + F  #Entire window


circuit,bb         = create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=p_pauli,idling_erasures=False)
h = bb.hz


DEM_wo_erasures = circuit.detector_error_model(flatten_loops=True)    
circuit_w_erasure = add_erasures(circuit, p_erasure)

DEM_w_erasures = circuit_w_erasure.detector_error_model(flatten_loops=True)

print("num_errors1,2:",DEM_w_erasures.num_errors,DEM_wo_erasures.num_errors)


#============= Order stays the same ===============


priors_w_erasures = []
for k in range(DEM_wo_erasures.num_errors):
    inst1 = DEM_wo_erasures[k]
    inst2 = DEM_w_erasures[k]
    if inst1.type=="error" and inst2.type=="error":
        t1 = inst1.targets_copy()
        t2 = inst2.targets_copy()
        
        priors_w_erasures.append(inst2.args_copy()[0])
        if t1!=t2:
            raise Exception("Error")

#============= Check if quits preserves the order (ALL GOOD) =====================


chk,obs,priors_alt=detector_error_model_to_matrix(DEM_w_erasures)

print(np.sum(priors_w_erasures-priors_alt))

#============ Now, let's try to create a map and see if we get the erasure locations correct ==================


if 2 + num_rounds - W >= 0:
    num_cor_rounds = (2 + num_rounds - W) // F  # num_cor_rounds=num of windows before the last window
    if (2 + num_rounds - W) % F != 0:  # we can slide one more window if the remaining rounds>W
        num_cor_rounds += 1
else:
    num_cor_rounds = 0
    warnings.warn("Window size larger than the syndrome extraction rounds: Doing whole history correction")


chk,obs,priors = detector_error_model_to_matrix(DEM_w_erasures)


window_check_set, window_observable_set, window_priors_set, window_update,ranges = spacetime(circuit, h, W, F, num_cor_rounds)

print("erasures in non-erased DEM (1st window):",np.sum(window_priors_set[0]>0.2))

window_check_set, window_observable_set, window_priors_set, window_update,ranges = spacetime(circuit_w_erasure, h, W, F, num_cor_rounds)


for k in range(len(window_priors_set)):
    
    print("size:",np.shape(window_priors_set[k]))

window_priors_set_alt = []

for r in ranges:

    window_priors_set_alt.append(  priors[r[0]:r[1]]  )

#======= Priors split correctly ===========================
for k in range(len(window_priors_set)):

    if (window_priors_set_alt[k]!=window_priors_set[k]).any():
        raise Exception("Err")
    # else:
    #     print(window_priors_set_alt[k]-window_priors_set[k])
    #     print(np.sum(window_priors_set_alt[k]-window_priors_set[k]))


#Passes the test above. Now, let's say we create simple circuit, we collect erasure locations in total map, and then slice priors
#Does it produce correct things?

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

from src.realtime_decoding.utils import get_erasure_set

priors = np.array([inst.args_copy()[0] for inst in DEM_w_erasures if inst.type == "error"])
erasures_total = (priors > 0.2).astype(np.uint8)
erasures_total = np.asarray(erasures_total)
print("total # of erasures:",np.sum(erasures_total))

erasures_per_window_vol1 = slice_erasures_to_windows(ranges,erasures_total)
erasures_per_window_vol2 = get_erasure_set(window_check_set,window_observable_set,window_priors_set)



for k in range(len(erasures_per_window_vol1)):

    er1 = erasures_per_window_vol1[k]
    er2 = erasures_per_window_vol2[k]

    if (er1!=er2).any():
        raise Exception("Error not split correctly.")




