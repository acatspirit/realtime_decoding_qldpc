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

def test_tesseract_w_erasures_regular_decoding_DEM_w_erasures(num_shots=100,p=1e-3):
    '''
    Let's compare the following:

    -Circuit where all errors are Pauli (100%) (DEM from Pauli only)
    -Circuit where some of the Pauli have been converted to erasures (DEM from Pauli + erasure)

    Q: Does tesseract perform better on the latter?
    A: Seems like it performs worse w/ erasures. Probably because tesseract does not know how to handle the erasures.

    '''

    basis      = 'Z'
    code_name  = "[[72,12,6]]"
    n, k, d    = map(int, code_name.strip("[]").split(","))
    num_rounds = d 

    

    #---Run the first type of circuits (100% all Pauli noise)

    circuit,_=create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=None,idling_erasures=False)
    
    DEM = circuit.detector_error_model()

    sampler = circuit.compile_detector_sampler()
    det_events,obs_flips = sampler.sample(shots=num_shots,separate_observables=True)

    config  = tesseract.TesseractConfig(dem=DEM, 
                                        det_beam=10,
                                        pqlimit=50_000,
                                        beam_climbing=True,
                                        no_revisit_dets=True)

    decoder = config.compile_decoder()

    logical_pred = decoder.decode_batch(det_events)
    num_errors = np.mean(obs_flips ^ logical_pred,axis=1) #how many errors averaged per logical qubit. Should have length = num_shots?
    pL_rPauli = np.mean(num_errors)  #pL @ d -- ok

    
    
    

    #================ Now, let's do the erasures ===========================
    
    Re        = 0.79
    p_pauli   = (1-Re)*p
    p_erasure = Re * p 


    #--- Repeat this many times to sample erasures anew----
    circuit,_         = create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=p_pauli,idling_erasures=False)

    logical_pred  = np.zeros(np.shape(logical_pred),dtype=np.uint8)
    all_obs_flips = np.zeros(np.shape(obs_flips),dtype=np.uint8)
    for k in range(num_shots):
        print("k=",k,"out of:",num_shots)
        
        circuit_w_erasure = add_erasures(circuit, p_erasure)
        DEM               = circuit_w_erasure.detector_error_model()


        sampler              = circuit_w_erasure.compile_detector_sampler()
        det_events,obs_flips = sampler.sample(shots=1,separate_observables=True)

        config               = tesseract.TesseractConfig(dem=DEM, 
                                            det_beam=10,
                                            pqlimit=50_000,
                                            beam_climbing=True,
                                            no_revisit_dets=True)

        decoder = config.compile_decoder()

        pred               = decoder.decode_batch(det_events)
        logical_pred[k,:]  = pred 
        all_obs_flips[k,:] = obs_flips


    num_errors = np.mean(all_obs_flips ^ logical_pred,axis=1)
    pL         = np.mean(num_errors)  #pL @ d -- ok

    print("pL @ d (regular Pauli):",pL_rPauli)
    print("pL @ d (less Pauli, more erase):",pL)


    return 



def test_tesseract_w_erasures_regular_decoding_DEM_wo_erasures(num_shots=100,p=1e-3):
    '''
    Let's compare the following:

    -Circuit where all errors are Pauli (100%)                       (DEM from Pauli only)
    -Circuit where some of the Pauli have been converted to erasures (DEM from Pauli only but with lower known error rate)

    Q: Does tesseract perform better on the latter???
    A: 

    '''

    basis      = 'Z'
    code_name  = "[[72,12,6]]"
    n, k, d    = map(int, code_name.strip("[]").split(","))
    num_rounds = d 

    

    #---Run the first type of circuits (100% all Pauli noise)

    circuit,_=create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=None,idling_erasures=False)
    
    DEM = circuit.detector_error_model()

    sampler = circuit.compile_detector_sampler()
    det_events,obs_flips = sampler.sample(shots=num_shots,separate_observables=True)

    config  = tesseract.TesseractConfig(dem=DEM, 
                                        det_beam=10,
                                        pqlimit=50_000,
                                        beam_climbing=True,
                                        no_revisit_dets=True)

    decoder = config.compile_decoder()

    logical_pred = decoder.decode_batch(det_events)
    num_errors = np.mean(obs_flips ^ logical_pred,axis=1) #how many errors averaged per logical qubit. Should have length = num_shots?
    pL_rPauli = np.mean(num_errors)  #pL @ d -- ok

    #================ Now, let's do the erasures ===========================
    
    Re        = 0.79
    p_pauli   = (1-Re)*p
    p_erasure = Re * p 


    #--- Repeat this many times to sample erasures anew----

    det_events = np.zeros(np.shape(det_events),dtype=bool)
    obs_flips  = np.zeros(np.shape(obs_flips),dtype=bool)
    
    circuit,_         = create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=p_pauli,idling_erasures=False)
    for k in range(num_shots):
        print("k=",k,"out of:",num_shots)
        
        circuit_w_erasure = add_erasures(circuit, p_erasure)
        
        sampler                        = circuit_w_erasure.compile_detector_sampler()
        det_events[k,:],obs_flips[k,:] = sampler.sample(shots=1,separate_observables=True)


    circuit,_  = create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=p_pauli,idling_erasures=False)
    DEM        = circuit.detector_error_model() #Construct the DEM from the circuit that has no erasures.

    config  = tesseract.TesseractConfig(dem=DEM, 
                                        det_beam=10,
                                        pqlimit=50_000,
                                        beam_climbing=True,
                                        no_revisit_dets=True)

    decoder = config.compile_decoder()

   
    logical_pred = decoder.decode_batch(det_events)
    num_errors = np.mean(obs_flips ^ logical_pred,axis=1) #how many errors averaged per logical qubit. Should have length = num_shots?
    pL = np.mean(num_errors)  #pL @ d -- ok



    print("pL @ d (regular Pauli):",pL_rPauli)
    print("pL @ d (less Pauli, more erase):",pL)


    return 

def test_tesseract_w_erasures_regular_decoding_DEM_wo_erasures_fix_same_DEM(num_shots=100,p=1e-3):
    '''
    Let's compare the following:

    -Circuit where all errors are Pauli (100%)                       (DEM from Pauli only)
    -Circuit where some of the Pauli have been converted to erasures (DEM from Pauli only, same as above)

    Q: Does tesseract perform better on the latter???
    A: 

    '''

    basis      = 'Z'
    code_name  = "[[54,8,6]]"
    n, k, d    = map(int, code_name.strip("[]").split(","))
    num_rounds = d 


    #---Run the first type of circuits (100% all Pauli noise)

    circuit,_=create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=None,idling_erasures=False)
    
    DEM = circuit.detector_error_model()

    sampler = circuit.compile_detector_sampler()
    det_events,obs_flips = sampler.sample(shots=num_shots,separate_observables=True)

    config  = tesseract.TesseractConfig(dem=DEM, 
                                        det_beam=10,
                                        pqlimit=50_000,
                                        beam_climbing=True,
                                        no_revisit_dets=True)

    decoder = config.compile_decoder()

    logical_pred = decoder.decode_batch(det_events)
    num_errors = np.mean(obs_flips ^ logical_pred,axis=1) #how many errors averaged per logical qubit. Should have length = num_shots?
    pL_rPauli = np.mean(num_errors)  #pL @ d -- ok

    #================ Now, let's do the erasures ===========================
    
    Re        = 0.79
    p_pauli   = (1-Re)*p
    p_erasure = Re * p 

    #--- Repeat this many times to sample erasures anew----

    det_events = np.zeros(np.shape(det_events),dtype=bool)
    obs_flips  = np.zeros(np.shape(obs_flips),dtype=bool)
    
    circuit,_ = create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=p_pauli,idling_erasures=False)

    for k in range(num_shots):
        print("k=",k,"out of:",num_shots)
        
        circuit_w_erasure = add_erasures(circuit, p_erasure)
        
        sampler                        = circuit_w_erasure.compile_detector_sampler()
        det_events[k,:],obs_flips[k,:] = sampler.sample(shots=1,separate_observables=True)


    # circuit,_  = create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=p_pauli,idling_erasures=False)
    # DEM        = circuit.detector_error_model() #Construct the DEM from the circuit that has no erasures.

    # config  = tesseract.TesseractConfig(dem=DEM, 
    #                                     det_beam=10,
    #                                     pqlimit=50_000,
    #                                     beam_climbing=True,
    #                                     no_revisit_dets=True)

    # decoder = config.compile_decoder()

   
    logical_pred = decoder.decode_batch(det_events)
    num_errors = np.mean(obs_flips ^ logical_pred,axis=1) #how many errors averaged per logical qubit. Should have length = num_shots?
    pL = np.mean(num_errors)  #pL @ d -- ok

    print("pL @ d (regular Pauli):",pL_rPauli)
    print("pL @ d (less Pauli, more erase):",pL)

    return 


# test_tesseract_w_erasures_regular_decoding_DEM_w_erasures(num_shots=50,p=1e-2)
# test_tesseract_w_erasures_regular_decoding_DEM_wo_erasures(p=5e-3,num_shots=100)
# test_tesseract_w_erasures_regular_decoding_DEM_wo_erasures_fix_same_DEM(p=5e-3,num_shots=100)



basis      = 'Z'
code_name  = "[[144,12,12]]"
n, k, d    = map(int, code_name.strip("[]").split(","))
num_rounds = 20 

# p=1e-3
#---Run the first type of circuits (100% all Pauli noise)

# circuit,_=create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=None,idling_erasures=False)

# DEM = circuit.detector_error_model()
# print(circuit)

p         = 1e-4
Re        = 0.95
p_pauli   = (1-Re)*p
p_erasure = Re * p 
num_rounds = 5

circuit,_         = create_bb_codes_circuit_ionic_model(code_name,p,num_rounds,basis,p_pauli=p_pauli,idling_erasures=False)

DEM_wo_erasures = circuit.detector_error_model(flatten_loops=True)    
circuit_w_erasure = add_erasures(circuit, p_erasure)

DEM_w_erasures = circuit_w_erasure.detector_error_model(flatten_loops=True)

print("num_errors1,2:",DEM_w_erasures.num_errors,DEM_wo_erasures.num_errors) #Same # of errors


#============= Order stays the same ===============
print(DEM_wo_erasures[0])
print(DEM_w_erasures[0])

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

from quits import detector_error_model_to_matrix

chk,obs,priors_alt=detector_error_model_to_matrix(DEM_w_erasures)

print(np.sum(priors_w_erasures-priors_alt))

#============ Below checks for erased locations ===============



# for inst1 in DEM_wo_erasures:
#     if inst1.type=="error":
#         t1 = inst1.targets_copy()
#         p1 = inst1.args_copy()[0]
#         for inst2 in DEM_w_erasures:
#             if inst2.type=="error":
#                 t2 = inst2.targets_copy()
#                 p2 = inst2.args_copy()[0]
#                 if t1==t2:
#                     if abs(p1-p2)>1e-3:
#                         print("abs(p1-p2)",abs(p1-p2))

# from src.realtime_decoding.utils import *

# cnt=0
# locs = []
# cutoff = 0.2
# for inst in DEM_w_erasures:
#     if inst.type=="error":

#         p_val = inst.args_copy()[0]
#         if p_val>cutoff:
#             locs.append(cnt)

#             print("p:",p_val)

#         cnt+=1


# erased = get_erased_mechanisms_from_DEM(DEM_w_erasures)
# print(np.sum(erased))
# print(erased[locs])




