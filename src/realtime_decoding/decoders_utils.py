import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) #move one level back out of the src file (or i guess for arianna's package 2 levels)

# ROOT = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", ".." )
# )

# sys.path.append(ROOT)
# sys.path.append(os.path.join(ROOT, "BeamSearchDecoder"))


from src.realtime_decoding.utils import get_window_dems
from ldpc.bplsd_decoder import BpLsdDecoder
from typing import Optional
from tesseract_decoder import tesseract
import relay_bp
import numpy as np
from scipy.sparse import csr_matrix
# import py_wrapper.py_decoder as uf
from uf_decoder.py_wrapper import py_decoder as uf
from BeamSearchDecoder.beamsearch import BeamSearchDecoder


"""
Helper functions for configuring decoders. Set parameters in this file for decoders
"""

def collect_default_decoder_params(decoder):
    '''
    Set the default parameters for each decoder. 

    Input:
        decoder: 'tesseract', 'relay_bp', 'bplsd', 'uf', 'beamsearch8_230iters' or 'beamsearch32_340iters'
    
    Output:
        decoder_params: dictionary with the key-value pairs the inputs and values for configuring the various decoders.

    '''
    if decoder=='tesseract': # make params a little worse

        decoder_params = {'det_beam': 10, 
                          'pqlimit': 50_000, 
                          'beam_climbing': True,
                          'no_revisit_dets': True} 

    elif decoder=='relay_bp':
        decoder_params =  {
        'gamma_dist_interval': (-0.175, 0.575),
        'gamma0': 0.125,
        'pre_iter': 80,
        'num_sets': 300,
        'set_max_iter': 60,
        'stop_nconv' : 1 }

    elif decoder=='bplsd':
        decoder_params = {
            'bp_method': 'product_sum',
            'max_bp_iters': 30,
            'schedule': 'serial',
            'lsd_method': 'lsd_cs',
            'lsd_order': 0,
        }    

    elif decoder == 'uf':
        decoder_params = {'num_qubits':0}    # there are no params to pass to UF

    elif decoder == 'beamsearch8_230iters':

        decoder_params = {'max_bp_iters': 230,
                          'beam_width': 8,
                          'initial_iters': 30,
                          'iters_per_round': 20,
                          'max_rounds': 10,
                          'num_results': 1}
        
    elif decoder == 'beamsearch32_340iters':
     
        decoder_params = {'max_bp_iters': 340,
                          'beam_width': 32,
                          'initial_iters': 40,
                          'iters_per_round': 30,
                          'max_rounds': 10,
                          'num_results': 1}


    return decoder_params


def configure_tesseract_per_sliding_window(window_check_set, window_observable_set, window_priors_set, decoder_params: Optional[dict] = None):
    '''Compile tesseract decoder for all windows.

    Inputs:
        window_check_set: list of check matrices per window
        window_observable_set: list of observables matrices per window
        window_priors_set: list of priors per window:
        decoder_params: dictionary of decoder parameters with key 'det_beam' (default is 3)

    Outputs:
        window_dems: dem per window
        compiled_decoders: list of tesseract decoders per window

    '''
    
    defaults = collect_default_decoder_params('tesseract')

    if decoder_params is None:
        decoder_params = {}

    for key, value in defaults.items():
        decoder_params.setdefault(key, value)

    window_dems = get_window_dems(window_check_set, window_observable_set, window_priors_set)

    compiled_decoders = []
    for k in range(len(window_check_set)):
        config  = tesseract.TesseractConfig(dem=window_dems[k], 
                                            det_beam=decoder_params['det_beam'],
                                            pqlimit=decoder_params['pqlimit'],
                                            beam_climbing=decoder_params['beam_climbing'],
                                            no_revisit_dets=decoder_params['no_revisit_dets'])

        decoder = config.compile_decoder()
        compiled_decoders.append(decoder)

    return window_dems,compiled_decoders

def configure_beamsearch_per_sliding_window(window_check_set, window_observable_set, window_priors_set, decoder_params: Optional[dict] = None, beamsearch_option = 1):
    '''
    Configure beamsearch decoder for all windows.

    Inputs:
        window_check_set: list of check matrices per window
        window_observable_set: list of observables matrices per window
        window_priors_set: list of priors per window:    
        decoder_params: dictionary of decoder parameters with keys: 'bp_method', 'max_bp_iters', 'schedule', 'lsd_method', 'lsd_order'
        (if some key(s) is/are missing defaults to 'product_sum', 30, 'serial', 'lsd_cs', 0 )
        beamsearch_option: 1 for beamsearch8_230iters or sth else for beamsearch32_340iters
    
    Outputs:
        compiled_decoders: list of beamsearch decoders per window
    '''
    
    if beamsearch_option==1:
        defaults = collect_default_decoder_params('beamsearch8_230iters')
    else:
        defaults = collect_default_decoder_params('beamsearch32_340iters')

    if decoder_params is None:
        decoder_params = {}

    for key, value in defaults.items():
        decoder_params.setdefault(key, value)

    # window_dems = get_window_dems(window_check_set, window_observable_set, window_priors_set)

    beamsearch_decoders = []
    
    for k in range(len(window_check_set)):

        decoder = BeamSearchDecoder(pcm = window_check_set[k],         
                                    error_channel= window_priors_set[k],          
                                    max_rounds=decoder_params['max_rounds'],
                                    beam_width=decoder_params['beam_width'],
                                    num_results=decoder_params['num_results'],
                                    iters_per_round=decoder_params['iters_per_round'],
                                    initial_iters=decoder_params['initial_iters'],
                                    )

        beamsearch_decoders.append(decoder)

    return beamsearch_decoders


def configure_relay_bp_per_sliding_window(window_check_set, window_priors_set, decoder_params: Optional[dict] = None):
    '''
    Configure relay-bp for all windows.

    Inputs:
        window_check_set: list of check matrices per window
        window_observable_set: list of observables matrices per window
        window_priors_set: list of priors per window:    
        decoder_params: dictionary of decoder parameters with keys: 'bp_method', 'max_bp_iters', 'schedule', 'lsd_method', 'lsd_order'
        (if some key(s) is/are missing defaults to 'product_sum', 30, 'serial', 'lsd_cs', 0 )
    
    Outputs:
        compiled_decoders: list of relay-bp decoders per window
    '''
    defaults = collect_default_decoder_params('relay_bp')

    if decoder_params is None:
        decoder_params = {}

    for key, value in defaults.items():
        decoder_params.setdefault(key, value)

    compiled_decoders = []
    for k in range(len(window_check_set)):
        relay_decoder = relay_bp.RelayDecoderF32(window_check_set[k].tocsr(),                                   
                                 error_priors=np.array(window_priors_set[k],dtype=np.float64),
                                 gamma0=decoder_params['gamma0'],
                                 pre_iter=decoder_params['pre_iter'],
                                 num_sets=decoder_params['num_sets'],
                                 set_max_iter=decoder_params['set_max_iter'],
                                 gamma_dist_interval=decoder_params['gamma_dist_interval'],
                                 stop_nconv=decoder_params['stop_nconv'])
        
    
        compiled_decoders.append(relay_decoder)


    return compiled_decoders

def configure_bplsd_decoder_per_sliding_window(window_check_set,window_priors_set, decoder_params: Optional[dict] = None):
    '''
    Configure bplsd for all windows.

    Inputs:
        window_check_set: list of check matrices per window
        window_observable_set: list of observables matrices per window
        window_priors_set: list of priors per window:    
        decoder_params: dictionary of decoder parameters with keys: 'bp_method', 'max_bp_iters', 'schedule', 'lsd_method', 'lsd_order'
        (if some key(s) is/are missing defaults to 'product_sum', 30, 'serial', 'lsd_cs', 0 )
    
    Outputs:
        bplsd_decoders: list of bplsd decoders per window
    '''

    defaults = collect_default_decoder_params('bplsd')


    if decoder_params is None:
        decoder_params = {}

    for key, value in defaults.items():
        decoder_params.setdefault(key, value)

    bplsd_decoders = []
    for k in range(len(window_check_set)):

        decoder = BpLsdDecoder(window_check_set[k],
                            error_channel=window_priors_set[k],
                            bp_method=decoder_params['bp_method'],
                            max_iter=decoder_params['max_bp_iters'],
                            schedule=decoder_params['schedule'],
                            lsd_method=decoder_params['lsd_method'],
                            lsd_order=decoder_params['lsd_order'])
        
    
        decoder.set_do_stats(True) #Enable this to collect the statistics of clusters
        bplsd_decoders.append(decoder)


    return bplsd_decoders

def configure_uf_decoder_per_sliding_window(window_check_set, erased_errors_set=None,):
    '''
    Configure UF for all windows.

    Inputs:
        window_check_set: list of check matrices per window
        erased_errors_set: list of erasures per window

    Outputs:
        uf_decoders: list of UF decoders per window
        erasure_array: list of erasures per window 
    '''

    uf_decoders = []
    
    if erased_errors_set is None:
        erasure_array = [np.zeros(window_check_set[k].shape[1], dtype=np.uint8) for k in range(len(window_check_set))]
    else:
        erasure_array = erased_errors_set

    for k in range(len(window_check_set)):
        decoder = uf.UFDecoder(csr_matrix(window_check_set[k]))
        uf_decoders.append(decoder)

    return uf_decoders, erasure_array