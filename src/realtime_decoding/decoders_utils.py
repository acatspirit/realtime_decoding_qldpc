from src.utils import get_window_dems
from ldpc.bplsd_decoder import BpLsdDecoder
from typing import Optional
from tesseract_decoder import tesseract
import relay_bp
import numpy as np

def collect_default_decoder_params(decoder):

    if decoder=='tesseract':
        #We could change some of these parameters
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


def configure_relay_bp_per_sliding_window(window_check_set, window_priors_set, decoder_params: Optional[dict] = None):

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