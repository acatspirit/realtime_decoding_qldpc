from quits.qldpc_code import BbCode
from quits import ErrorModel
import stim
import deltakit_stim

def fix_bb_circuit_for_sliding_window(original_circuit, num_rounds):
    """
    Flattens a BB circuit and injects (t,) coordinates for sliding window decoding.
    """
    # 1. Flatten to ensure each round's detectors can have unique time coordinates
    flattened = original_circuit.flattened()
    detectors_per_round = original_circuit.num_detectors//(num_rounds+2) # num_rounds should be d ?
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


def create_bb_codes_circuit(code_name: str, p: float, num_rounds: int, basis: str):
    
    '''Maybe we should compare codes that have the same # of logical qubits??
    This is because we call logical failure even if one out of k qubits has a logical error.
    So we should have a fixed k as well.
    
    So:
     [[72,12,6]], [[144,12,12]], [[288,12,18]]
     [[90,8,10]], [[162,8,14]], [[180,8,16]]

    '''

    bb_codes_dict = { "[[98,6,12]]": {'l': 7, 'm': 7, 'A_x_pows': [3], 'A_y_pows': [5,6], 'B_x_pows': [3,5], 'B_y_pows': [2]}, 
                
                # https://arxiv.org/pdf/2408.10001 some codes taken from this paper (all CSS).
                 "[[54,8,6]]":    {'l': 3, 'm': 9, 'A_x_pows': [0], 'A_y_pows': [2,4], 'B_x_pows': [1,2], 'B_y_pows': [3]},      
                 "[[90,8,10]]]": {'l':15, 'm':3, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [2,7], 'B_y_pows':[0]},      
                 "[[126,8,10]]":  {'l': 3, 'm': 21, 'A_x_pows': [0], 'A_y_pows': [2,10], 'B_x_pows': [1,2], 'B_y_pows': [3]}, 
                 "[[162,8,14]]":  {'l': 3, 'm': 27, 'A_x_pows': [0], 'A_y_pows': [10,14], 'B_x_pows': [1,2], 'B_y_pows': [12]}, 
                 "[[180,8,16]]": {'l': 6, 'm': 15, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [4,5], 'B_y_pows': [6]},    
                 
                 "[[150,16,8]]": {'l': 5, 'm': 15, 'A_x_pows': [0], 'A_y_pows': [6,8], 'B_x_pows': [1,4], 'B_y_pows': [5]}, 

                 "[[72,12,6]]":   {'l':6, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},    
                 "[[144,12,12]]": {'l':12, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},  
                 "[[288,12,18]]": {'l':12, 'm':12, 'A_x_pows': [3], 'A_y_pows': [2,7], 'B_x_pows': [1,2], 'B_y_pows':[3]},  
                 "[[360,12,24]]": {'l':30, 'm':6, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [25,26], 'B_y_pows':[3]},  

                 "[[288,24,12]]": {'l':12, 'm':12, 'A_x_pows': [6], 'A_y_pows': [1,2], 'B_x_pows': [2,4], 'B_y_pows':[3]},
                 "[[784,24,24]]": {'l':28, 'm':14, 'A_x_pows': [26], 'A_y_pows': [6,8], 'B_x_pows': [9,20], 'B_y_pows':[7]} 
                }


    # d_dict = { 6:{'l':6, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},    #this is [[72,12,6]]
    #             10: {'l':15, 'm':3, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [2,7], 'B_y_pows':[0]}, #this is [[90,8,10]]
    #              12:{'l':12, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},  #this is [[144,12,12]]
                
    #              14:{'l':3, 'm':27, 'A_x_pows': [0], 'A_y_pows': [10,14], 'B_x_pows': [1,2], 'B_y_pows':[12]},  #this is [[162,8,14]]
    #             16:{'l':6, 'm':15, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [4,5], 'B_y_pows':[6]},  #this is [[180,8,16]]
                
    #             18:{'l':12, 'm':12, 'A_x_pows': [3], 'A_y_pows': [2,7], 'B_x_pows': [1,2], 'B_y_pows':[3]},  #this is [[288,12,18]]
    #             24:{'l':28, 'm':14, 'A_x_pows': [26], 'A_y_pows': [6,8], 'B_x_pows': [9,20], 'B_y_pows':[7]}} #this is [[784,24,24]]
    


    # https://arxiv.org/pdf/2408.10001 some codes taken from this paper (k=8 codes taken from there)
    # https://github.com/qiskit-community/qcode-discovery (codes taken from here too)
    # Codes also taken from Bravyi et al. 2024 (arXiv:2308.07915): [[72,12,6]], [[90,8,10]], [[144,12,12]], [[288,12,18]], [[360,12,<=24]]

    # The code is defined by a pair of polynomials
    # A and B that depends on two variables x and y such that
    # x^l = 1
    # y^m = 1
    # A = x^{a_1} + y^{a_2} + y^{a_3} 
    # B = y^{b_1} + x^{b_2} + x^{b_3}    

    # https://arxiv.org/pdf/2407.15988
    # https://github.com/nbi-hyq/uf_decoder/blob/main/py_wrapper/example_bb_codes.py
    # l_dims = [{x: 6, y: 6}, {x: 15, y: 3}, {x: 9, y: 6}, {x: 12, y: 6}, {x: 12, y: 12}]
    # l_terms_a = [x**3+y+y**2, x**9+y+y**2, x**3+y+y**2, x**3+y+y**2, x**3+y**2+y**7]
    # l_terms_b = [y**3+x+x**2, 1+x**2+x**7, y**3+x+x**2, y**3+x+x**2, y**3+x+x**2]
    # n_list = [72, 90, 108, 144, 288]
    # k_list = [12, 8, 8, 12, 12]

    # [[784,24,24]]
    #ell,m = 28,14
    #a1,a2,a3=26,6,8
    #b1,b2,b3=7,9,20    

    # [[144,12,12]]
    # ell,m = 12,6
    # a1,a2,a3 = 3,1,2
    # b1,b2,b3 = 3,1,2    

    # https://github.com/sbravyi/BivariateBicycleCodes/blob/main/decoder_setup.py    

    # code_params = d_dict[d]
    code_params = bb_codes_dict[code_name]

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

    custom_circuit = bb.build_circuit(strategy="custom", num_rounds=num_rounds, basis=basis, error_model=error_model) 
    labeled_circuit = fix_bb_circuit_for_sliding_window(custom_circuit, num_rounds)

    return labeled_circuit, bb



def create_bb_codes_circuit_ionic_model(code_name: str, p: float, num_rounds: int, basis: str):
    
    '''Maybe we should compare codes that have the same # of logical qubits??
    This is because we call logical failure even if one out of k qubits has a logical error.
    So we should have a fixed k as well.
    
    So:
     [[72,12,6]], [[144,12,12]], [[288,12,18]]
     [[90,8,10]], [[162,8,14]], [[180,8,16]]

    '''

    bb_codes_dict = { "[[98,6,12]]": {'l': 7, 'm': 7, 'A_x_pows': [3], 'A_y_pows': [5,6], 'B_x_pows': [3,5], 'B_y_pows': [2]}, 
                
                # https://arxiv.org/pdf/2408.10001 some codes taken from this paper (all CSS).
                 "[[54,8,6]]":    {'l': 3, 'm': 9, 'A_x_pows': [0], 'A_y_pows': [2,4], 'B_x_pows': [1,2], 'B_y_pows': [3]},      
                 "[[90,8,10]]]": {'l':15, 'm':3, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [2,7], 'B_y_pows':[0]},      
                 "[[126,8,10]]":  {'l': 3, 'm': 21, 'A_x_pows': [0], 'A_y_pows': [2,10], 'B_x_pows': [1,2], 'B_y_pows': [3]}, 
                 "[[162,8,14]]":  {'l': 3, 'm': 27, 'A_x_pows': [0], 'A_y_pows': [10,14], 'B_x_pows': [1,2], 'B_y_pows': [12]}, 
                 "[[180,8,16]]": {'l': 6, 'm': 15, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [4,5], 'B_y_pows': [6]},    
                 
                 "[[150,16,8]]": {'l': 5, 'm': 15, 'A_x_pows': [0], 'A_y_pows': [6,8], 'B_x_pows': [1,4], 'B_y_pows': [5]}, 

                 "[[72,12,6]]":   {'l':6, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},    
                 "[[144,12,12]]": {'l':12, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},  
                 "[[288,12,18]]": {'l':12, 'm':12, 'A_x_pows': [3], 'A_y_pows': [2,7], 'B_x_pows': [1,2], 'B_y_pows':[3]},  
                 "[[360,12,24]]": {'l':30, 'm':6, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [25,26], 'B_y_pows':[3]},  

                 "[[288,24,12]]": {'l':12, 'm':12, 'A_x_pows': [6], 'A_y_pows': [1,2], 'B_x_pows': [2,4], 'B_y_pows':[3]},
                 "[[784,24,24]]": {'l':28, 'm':14, 'A_x_pows': [26], 'A_y_pows': [6,8], 'B_x_pows': [9,20], 'B_y_pows':[7]} 
                }


    # d_dict = { 6:{'l':6, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},    #this is [[72,12,6]]
    #             10: {'l':15, 'm':3, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [2,7], 'B_y_pows':[0]}, #this is [[90,8,10]]
    #              12:{'l':12, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},  #this is [[144,12,12]]
                
    #              14:{'l':3, 'm':27, 'A_x_pows': [0], 'A_y_pows': [10,14], 'B_x_pows': [1,2], 'B_y_pows':[12]},  #this is [[162,8,14]]
    #             16:{'l':6, 'm':15, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [4,5], 'B_y_pows':[6]},  #this is [[180,8,16]]
                
    #             18:{'l':12, 'm':12, 'A_x_pows': [3], 'A_y_pows': [2,7], 'B_x_pows': [1,2], 'B_y_pows':[3]},  #this is [[288,12,18]]
    #             24:{'l':28, 'm':14, 'A_x_pows': [26], 'A_y_pows': [6,8], 'B_x_pows': [9,20], 'B_y_pows':[7]}} #this is [[784,24,24]]
    


    # https://arxiv.org/pdf/2408.10001 some codes taken from this paper (k=8 codes taken from there)
    # https://github.com/qiskit-community/qcode-discovery (codes taken from here too)
    # Codes also taken from Bravyi et al. 2024 (arXiv:2308.07915): [[72,12,6]], [[90,8,10]], [[144,12,12]], [[288,12,18]], [[360,12,<=24]]

    # The code is defined by a pair of polynomials
    # A and B that depends on two variables x and y such that
    # x^l = 1
    # y^m = 1
    # A = x^{a_1} + y^{a_2} + y^{a_3} 
    # B = y^{b_1} + x^{b_2} + x^{b_3}    

    # https://arxiv.org/pdf/2407.15988
    # https://github.com/nbi-hyq/uf_decoder/blob/main/py_wrapper/example_bb_codes.py
    # l_dims = [{x: 6, y: 6}, {x: 15, y: 3}, {x: 9, y: 6}, {x: 12, y: 6}, {x: 12, y: 12}]
    # l_terms_a = [x**3+y+y**2, x**9+y+y**2, x**3+y+y**2, x**3+y+y**2, x**3+y**2+y**7]
    # l_terms_b = [y**3+x+x**2, 1+x**2+x**7, y**3+x+x**2, y**3+x+x**2, y**3+x+x**2]
    # n_list = [72, 90, 108, 144, 288]
    # k_list = [12, 8, 8, 12, 12]

    # [[784,24,24]]
    #ell,m = 28,14
    #a1,a2,a3=26,6,8
    #b1,b2,b3=7,9,20    

    # [[144,12,12]]
    # ell,m = 12,6
    # a1,a2,a3 = 3,1,2
    # b1,b2,b3 = 3,1,2    

    # https://github.com/sbravyi/BivariateBicycleCodes/blob/main/decoder_setup.py    

    # code_params = d_dict[d]
    code_params = bb_codes_dict[code_name]

    #tailored to ions
    error_model = ErrorModel(
        idle_error=p/100,
        sqgate_error=p/10,
        tqgate_error=p,
        spam_error=p/10,
        )

    bb = BbCode(
        l=code_params['l'],
        m=code_params['m'],
        A_x_pows=code_params['A_x_pows'],
        A_y_pows=code_params['A_y_pows'],
        B_x_pows=code_params['B_x_pows'],
        B_y_pows=code_params['B_y_pows'],
    )



    custom_circuit = bb.build_circuit(strategy="custom", num_rounds=num_rounds, basis=basis, error_model=error_model) 
    labeled_circuit = fix_bb_circuit_for_sliding_window(custom_circuit, num_rounds)

    return labeled_circuit, bb


def add_independent_leakage_errors_per_round(circuit: stim.Circuit, p_leak = 1e-5):
    '''Add independent leakage errors on the qubits, assuming a fixed leakage error rate per round of p_leak.
       Note, leakage errors are put at the beginning of each round: 
       -after instruction "R" for first round,
       -after "MR" in subsequent rounds where we apply p_leak on each qubit independently.
       -We also add heralded_leakage_event before each "MR" or "M" qubit (so we assume we can herald it)

    Inputs:
        circuit: the stim circuit which could have Pauli errors
        p_leak:  the leakage rate (default is chosen for trapped ions based on https://arxiv.org/pdf/2604.19481)
    Outputs:
        new_circuit: a deltakit_stim circuit which includes also leakage errors (leakage errors are also depicted in rates of dem)


    NOTE: IF leakage is combined with loss, we need to be extra careful, cause we might not be able to sample independently erasures and loss.
          For example, if a leaked qubit interacts with lost qubit, the loss propagates to the leaked qubit.

    
    TODO: Think more how we can combine loss + leakage + Pauli errors. Or we will study loss and leakage as independent mechanisms.


    '''

    new_circuit = deltakit_stim.Circuit()

    targets = [stim.GateTarget(k) for k in range(circuit.num_qubits)]  #These are all the qubits in the circuit

    for inst in circuit:

        if inst.name == "MR": #intermediate measurements of ancilla

            new_circuit.append(name="HERALD_LEAKAGE_EVENT",targets=inst.targets_copy())              #Can only have herald on ancillas that will be measured
            new_circuit.append(name=inst.name,targets=inst.targets_copy(),arg=inst.gate_args_copy()) #Apply MR
            new_circuit.append(name="LEAKAGE",targets=targets,arg=p_leak)                            #Leak again ALL QUBITS in the beginning of next round

        elif inst.name=="R": #Reset of all qubits @ beginning of experiment
            
            new_circuit.append(name=inst.name,targets=inst.targets_copy(),arg=inst.gate_args_copy()) #Apply R
            new_circuit.append(name="LEAKAGE",targets=inst.targets_copy(),arg=p_leak)                #Leak qubits 
            

        elif inst.name=="M": #final data qubit measurements 
            new_circuit.append(name="HERALD_LEAKAGE_EVENT",targets=inst.targets_copy())               #Herald the leakage on data qubit measurements
            new_circuit.append(name=inst.name,targets=inst.targets_copy(),arg=inst.gate_args_copy())  #Measure the data qubits
        
        else: #any other instruction
            new_circuit.append(name=inst.name,targets=inst.targets_copy(),arg=inst.gate_args_copy())  #All other instructions remain invariant

    return new_circuit
