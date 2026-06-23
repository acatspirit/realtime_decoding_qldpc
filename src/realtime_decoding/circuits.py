from quits.qldpc_code import BbCode
from quits import ErrorModel
import stim


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


def create_bb_codes_circuit(d: int, p: float, num_rounds: int, basis: str):
    
    d_dict = {6:{'l':6, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},    #this is [[72,12,6]]
                10: {'l':15, 'm':3, 'A_x_pows': [9], 'A_y_pows': [1,2], 'B_x_pows': [2,7], 'B_y_pows':[0]}, #this is [[90,8,10]]
                12:{'l':12, 'm':6, 'A_x_pows': [3], 'A_y_pows': [1,2], 'B_x_pows': [1,2], 'B_y_pows':[3]},  #this is [[144,12,12]]
                18:{'l':12, 'm':12, 'A_x_pows': [3], 'A_y_pows': [2,7], 'B_x_pows': [1,2], 'B_y_pows':[3]},  #this is [[288,12,18]]
                24:{'l':28, 'm':14, 'A_x_pows': [26], 'A_y_pows': [6,8], 'B_x_pows': [9,20], 'B_y_pows':[7]}} #this is [[784,24,24]]
    
    # The code is defined by a pair of polynomials
    # A and B that depends on two variables x and y such that
    # x^ell = 1
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

    custom_circuit = bb.build_circuit(strategy="custom", num_rounds=num_rounds, basis=basis, error_model=error_model) 
    labeled_circuit = fix_bb_circuit_for_sliding_window(custom_circuit, num_rounds)

    return labeled_circuit, bb
