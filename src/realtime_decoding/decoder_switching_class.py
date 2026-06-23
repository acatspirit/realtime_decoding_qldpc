import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move one level back out of the src file (or i guess for arianna's package 2 levels)


from quits import detector_error_model_to_matrix
from tesseract_decoder import tesseract
from ldpc.bplsd_decoder import BpLsdDecoder
import warnings

from quits.decoder import spacetime
import numpy as np
from ldpc_post_selection.src.ldpc_post_selection.cluster_tools import compute_lp_norm
from helper_cluster_tools import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

from src.circuits import create_bb_codes_circuit
from joblib import Parallel, delayed
from src.decoders_utils import configure_tesseract_per_sliding_window, configure_bplsd_decoder_per_sliding_window, configure_relay_bp_per_sliding_window, collect_default_decoder_params
from typing import Optional
import relay_bp

#Structure of the class:
#Creates a stim circuit upon initialization, samples from it
#Configures strong and weak decoders per window
#Needs a choise for strong decoder & weak decoder

#Performs
#i)   batch decoding 
#ii)  sliding window decoding with one decoder
#iii) sliding window decoding where it switches from weak to strong decoder, based on cluster norm cutoff


class tesseract_wrapper:
    '''
    Helper tesseract wrapper for decoding. It converts decoded_errors which are returned as indices into 0/1 array for error mechanisms in the decoding region.
    '''
    def __init__(self,decoder,num_events):

        self.decoder=decoder
        self.num_events =num_events 

    def __call__(self, syndrome):

        decoded_error_inds = self.decoder.decode_to_errors(syndrome) #this will give indices so convert to 0/1 array
        
        decoded_errors = np.zeros(self.num_events, dtype=np.uint8)
        decoded_errors[decoded_error_inds] = 1        

        return decoded_errors
    

class relaybp_wrapper:
    '''
    Helper relaybp wrapper for decoding. It converts decoded_errors which are returned as indices into 0/1 array for error mechanisms in the decoding region.
    '''
    def __init__(self,decoder):

        self.decoder=decoder

    def __call__(self,syndrome):

        decoded_errors = self.decoder.decode_batch(np.expand_dims(np.array(syndrome,dtype=np.uint8),axis=0)) 
        decoded_errors = np.squeeze(decoded_errors)

        return decoded_errors



#TODO: need some cluster_norm_wrapper which depending on bplsd or union find, or other cluster-based weak decoder collects the soft-output information 
#      (currently limited in decoder.statistics see weak decoding functions)

#TODO: To make the decoder_switching class more modular we could make it accept the errormodel instead of a value of p 
#TODO: need uf wrapper because we might also be inputing erasures

class decoder_switching_class:

    def __init__(self, d: int, num_rounds: int, p: float, basis: str, num_shots: int, W: int, F: int, 
                 strong_decoder_option: str, 
                 weak_decoder_option: str, 
                 strong_decoder_params: Optional[dict] = None,
                 weak_decoder_params: Optional[dict] = None):
        
        '''
        Inputs:
        d: distance of bb code
        num_rounds: # of syndrome extraction rounds
        p: error rate
        basis: 'Z' or 'X'
        num_shots: number of shots for memory experiment
        W: size of total window
        F: size of commit region
        strong_decoder_option: option to choose strong decoder from 'tesseract' or 'relay_bp'
        weak_decoder_option: option to choose weak decoder from 'uf' or 'bplsd'
        strong_decoder_params: dictionary with strong decoder parameters 
        weak_decoder_params: dictionary with weak decoder parameters 
        '''

     
        circuit,bb = create_bb_codes_circuit(d, p, num_rounds, basis)
        sampler    = circuit.compile_detector_sampler()

        detection_events,obs_flips = sampler.sample(shots=num_shots,separate_observables=True)

        self.detection_events = np.array(detection_events,dtype=np.uint8)
        self.obs_flips        = obs_flips

        self.circuit = circuit 
        self.bb      = bb 
        self.num_shots = num_shots
        
        if basis=='Z':
            h = bb.hz 
            self.logical = bb.lz
            
        elif basis=='X':
            h = bb.hx 
            self.logical = bb.lx

        self.h = h 
        self.W = W
        self.F = F

        # update the total number of windows for decoding, the size of the last window
        if 2 + num_rounds - W >= 0:
            num_cor_rounds = (2 + num_rounds - W) // F  # num_cor_rounds=num of windows before the last window
            if (2 + num_rounds - W) % F != 0:  # we can slide one more window if the remaining rounds>W
                num_cor_rounds += 1
        else:
            num_cor_rounds = 0
            warnings.warn("Window size larger than the syndrome extraction rounds: Doing whole history correction")
        

        self.num_cor_rounds                                                                           = num_cor_rounds
        self.window_check_set, self.window_observable_set, self.window_priors_set, self.window_update = self._prepare_windows()

        #------ Collect strong/weak decoders only once per window -----------
        if strong_decoder_option=='tesseract':
            
            self.window_dems,self.strong_decoder = configure_tesseract_per_sliding_window(self.window_check_set,self.window_observable_set,self.window_priors_set,strong_decoder_params)

            self.strong_decode_function = [tesseract_wrapper(decoder, dem.num_errors)
                                           for decoder, dem in zip(self.strong_decoder, self.window_dems)]            

        elif strong_decoder_option=='relay_bp':
            self.strong_decoder         = configure_relay_bp_per_sliding_window(self.window_check_set, self.window_priors_set,strong_decoder_params)
            self.strong_decode_function = [relaybp_wrapper(decoder)
                                         for decoder in self.strong_decoder] 


        else:
            raise NotImplementedError("No other strong decoder besides tesseract is implemented for now")
        

        if weak_decoder_option=='bplsd':

            self.weak_decoder         = configure_bplsd_decoder_per_sliding_window(self.window_check_set, self.window_priors_set,weak_decoder_params)
            self.weak_decode_function = [getattr(decoder,"decode",None)
                                         for decoder in self.weak_decoder] 

        else:
            raise NotImplementedError("No other weak decoder besides bplsd is implemented for now")
        

        
        return 

    def _prepare_windows(self):
        '''
        Prepare the windows for sliding window decoding.

        Outputs:
        window_check_set: list of parity check matrices per window
        window_observable_set: list of observable matrices per window
        window_priors_set: list of priors per window
        window_update: list of updates per window (?)
        '''

        window_check_set, window_observable_set, window_priors_set, window_update = spacetime(self.circuit, self.h, self.W, self.F, self.num_cor_rounds)


        return window_check_set, window_observable_set, window_priors_set, window_update 

    def decode_last_window_w_weak_decoder(self, F: int, num_checks: int, shot_index: int, syn_update, accumulated_correction, num_cor_rounds: int, norm_order=2):
        '''
        Decode the last window w/ the weak decoder.
        '''
        k          = -1

        decoder         = self.weak_decoder[k]
        num_faults_in_F = self.window_observable_set[k].shape[1] #number of faults in commit region (num_faults_in_F=num_faults_in_W for last iteration)
        num_faults_in_W = np.shape(self.window_check_set[k])[1]  

        diff_syndrome              = self.detection_events[shot_index, (F * num_cor_rounds) * num_checks:].copy()
        # diff_syndrome[:num_checks] = (diff_syndrome[:num_checks] + syn_update) % 2
        diff_syndrome[:num_checks] ^= syn_update
        
        decoded_errors = self.weak_decode_function[k](diff_syndrome)
        correction     = self.window_observable_set[num_cor_rounds] @ decoded_errors % 2

        #TODO: This needs to be handled externally -- need a general wrapper applicable for UF & BPLSD -- ideally configured upon initialization of the object
        stats           = decoder.statistics
        cluster_norm    = collect_cluster_norm(stats, num_faults_in_W,num_faults_in_F, norm_order)      

        # accumulated_correction = (accumulated_correction + correction) % 2
        accumulated_correction ^= (correction) 

        return accumulated_correction,cluster_norm

    def decode_main_window_w_weak_decoder(self, W: int, F: int, num_checks: int, current_window_index: int, shot_index: int, syn_update, accumulated_correction, norm_order=2):
        '''
        Decode any window besides last w/ a weak cluster based decoder and keep truck of the cluster norm information.
        '''

        k          = current_window_index
    
        decoder         = self.weak_decoder[k]
        num_faults_in_F = self.window_observable_set[k].shape[1] #number of faults in commit region F
        num_faults_in_W = np.shape(self.window_check_set[k])[1]  #number of faults in the entire window W

        diff_syndrome              = self.detection_events[shot_index, F * k * num_checks:(F * k + W) * num_checks].copy()
        # diff_syndrome[:num_checks] = (diff_syndrome[:num_checks] + syn_update) % 2  #update syndrome based on previous window decoding
        diff_syndrome[:num_checks] ^= syn_update   #update syndrome based on previous window decoding

        
        decoded_errors      = self.weak_decode_function[k](diff_syndrome) #Correction in entire window W, (only part F is committed)
        decoded_errors_in_F = decoded_errors[:self.window_observable_set[k].shape[1]]
        
        correction = self.window_observable_set[k] @ decoded_errors_in_F % 2  #the window_observable_set is set of observables only in region F. # of observables x # of faults in region F

        syn_update = self.window_update[k] @ decoded_errors_in_F % 2
    
        stats           = decoder.statistics
        cluster_norm    = collect_cluster_norm(stats, num_faults_in_W,num_faults_in_F, norm_order)      

        # accumulated_correction = (accumulated_correction + correction) % 2
        accumulated_correction ^= (correction) 


        return syn_update,accumulated_correction,cluster_norm
    
    def decode_main_window_w_strong_decoder(self, W: int, F: int, num_checks: int, current_window_index: int, shot_index: int, syn_update, accumulated_correction):
        '''
        Decode any window besides last w/ a strong decoder
        '''

        k          = current_window_index

        diff_syndrome              = self.detection_events[shot_index, F * k * num_checks:(F * k + W) * num_checks].copy()
        # diff_syndrome[:num_checks] = (diff_syndrome[:num_checks] + syn_update) % 2  #update syndrome based on previous window decoding
        diff_syndrome[:num_checks] ^= syn_update
        
        decoded_errors = self.strong_decode_function[k](diff_syndrome) #Correction in entire window W, (only part F is committed)
        decoded_errors_in_F = decoded_errors[:self.window_observable_set[k].shape[1]]
        
        correction = self.window_observable_set[k] @ decoded_errors_in_F % 2  #the window_observable_set is set of observables only in region F. # of observables x # of faults in region F

        syn_update = self.window_update[k] @ decoded_errors_in_F % 2

        # accumulated_correction = (accumulated_correction + correction) % 2
        accumulated_correction ^= (correction) 

        return syn_update,accumulated_correction
    
    def decode_last_window_w_strong_decoder(self, F: int, num_checks: int, shot_index: int, syn_update, num_cor_rounds: int, accumulated_correction):
        '''
        Decode the last window w/ the strong decoder.
        '''
        k          = -1

        diff_syndrome              = self.detection_events[shot_index, (F * num_cor_rounds) * num_checks:].copy()
        # diff_syndrome[:num_checks] = (diff_syndrome[:num_checks] + syn_update) % 2
        diff_syndrome[:num_checks] ^= syn_update
        
        decoded_errors = self.strong_decode_function[k](diff_syndrome)
        correction     = self.window_observable_set[num_cor_rounds] @ decoded_errors % 2

        # accumulated_correction = (accumulated_correction + correction) % 2
        accumulated_correction ^= (correction) 

        return accumulated_correction

    def decode_with_sliding_window_and_decoder_switching(self, cluster_norm_cutoff: float, norm_order=2):

        num_checks       = self.h.shape[0]
        logical_pred = np.zeros((self.num_shots, self.logical.shape[0]), dtype=np.uint8)

        cluster_norms_per_shot = []
        switch_times_per_shot  = []
        
        W = self.W 
        F = self.F
        num_cor_rounds = self.num_cor_rounds

        for shot_index in range(self.num_shots):

            accumulated_correction = np.zeros(self.window_observable_set[0].shape[0], dtype=np.uint8)
            syn_update = np.zeros(num_checks, dtype=np.uint8)            

            cluster_norm_per_window  = []
            switch_times             = 0

            for current_window_index in range(num_cor_rounds): #all windows besides last

                syn_update_weak,accumulated_correction_weak,cluster_norm = self.decode_main_window_w_weak_decoder(W,F, num_checks, current_window_index, shot_index, syn_update, accumulated_correction, norm_order=norm_order)
                cluster_norm_per_window.append(cluster_norm)

                if cluster_norm>cluster_norm_cutoff:
                    switch_times+=1
                    syn_update, accumulated_correction = self.decode_main_window_w_strong_decoder(W,F, num_checks, current_window_index, shot_index, syn_update, accumulated_correction)
                
                else: #keep syndrome update and accumulated correction from weak decoder
                    syn_update             = syn_update_weak
                    accumulated_correction = accumulated_correction_weak

            #decode the last window
            accumulated_correction_weak,cluster_norm = self.decode_last_window_w_weak_decoder(F, num_checks, shot_index, syn_update, accumulated_correction, num_cor_rounds,norm_order=norm_order)
            cluster_norm_per_window.append(cluster_norm)

            if cluster_norm>cluster_norm_cutoff:
                switch_times+=1
                accumulated_correction = self.decode_last_window_w_strong_decoder(F, num_checks, shot_index, syn_update, num_cor_rounds, accumulated_correction)
            else: #keep accumulated correction from weak decoder
                accumulated_correction = accumulated_correction_weak


            logical_pred[shot_index, :] = accumulated_correction
            cluster_norms_per_shot.append(cluster_norm_per_window)
            switch_times_per_shot.append(switch_times)

        
        logical_errors_per_shot = {"switching": np.any(self.obs_flips ^ logical_pred,axis=1)}

        return cluster_norms_per_shot,switch_times_per_shot,logical_errors_per_shot

    def decode_with_sliding_window(self,decoder_option: str, norm_order: int):
        '''
        Decode w/ sliding window and no decoder switching. Choose weak or strong decoder option.

        Input:
            decoder_option: 'weak' or 'strong' and uses the weak or strong set by the class upon initialization
            norm_order: integer defining the cluster norm order, in case we use the weak decoder

        Outputs:
            cluster_norms_per_shot: if weak decoder was chosen output a list of cluster norms lists per shot
            logical_errors_per_shot: returned both for weak/strong decoder and gives a 0/1 array of whether or not we made a logical error per shot
        '''

        num_checks   = self.h.shape[0]
        num_logicals = self.window_observable_set[0].shape[0]
        logical_pred = np.zeros((self.num_shots, self.logical.shape[0]), dtype=np.uint8)

        W = self.W 
        F = self.F
        num_cor_rounds = self.num_cor_rounds
        
        if decoder_option=='weak':

            cluster_norms_per_shot = []

            for shot_index in range(self.num_shots):

                accumulated_correction = np.zeros(num_logicals, dtype=np.uint8)
                syn_update = np.zeros(num_checks, dtype=np.uint8)            

                cluster_norm_per_window  = []
                for current_window_index in range(self.num_cor_rounds): #all windows besides last

                    syn_update,accumulated_correction,cluster_norm = self.decode_main_window_w_weak_decoder(W,F,num_checks, current_window_index, shot_index, syn_update, accumulated_correction, norm_order=norm_order)
                    cluster_norm_per_window.append(cluster_norm)

                #decode the last window
                accumulated_correction,cluster_norm = self.decode_last_window_w_weak_decoder(F, num_checks, shot_index, syn_update, accumulated_correction, num_cor_rounds, norm_order=norm_order)
                cluster_norm_per_window.append(cluster_norm)


                logical_pred[shot_index, :] = accumulated_correction
                cluster_norms_per_shot.append(cluster_norm_per_window)

            
            return cluster_norms_per_shot, np.any(self.obs_flips ^ logical_pred,axis=1)

        elif decoder_option=='strong':

            for shot_index in range(self.num_shots):

                accumulated_correction = np.zeros(num_logicals, dtype=np.uint8)
                syn_update = np.zeros(num_checks, dtype=np.uint8)            
                
                for current_window_index in range(self.num_cor_rounds): #all windows besides last

                    syn_update,accumulated_correction = self.decode_main_window_w_strong_decoder(W,F,num_checks,current_window_index, shot_index, syn_update, accumulated_correction)
                    
                #decode the last window
                accumulated_correction = self.decode_last_window_w_strong_decoder(F, num_checks, shot_index, syn_update, num_cor_rounds, accumulated_correction)
                
                logical_pred[shot_index, :] = accumulated_correction

            return np.any(self.obs_flips ^ logical_pred,axis=1)

        return 

    #TODO: Need wrapper for uf
    def decode_full_syndrome_history(self,decoder: str):
        '''
        Do regular decoding of the entire syndrome history w/o sliding window. Note this uses 'decode' function of all decoders (not decoding batch).
        
        Input:
            decoder: 'tesseract', 'bplsd', 'relay_bp'
        Output:
            logical_errors_per_shot: 0/1 array whether or not we made a logical error for each shot
        '''

        defaults       = collect_default_decoder_params(decoder)
        dem            = self.circuit.detector_error_model()
        chk,obs,priors = detector_error_model_to_matrix(dem)
        det_events     = self.detection_events

        logical_pred = np.zeros((self.num_shots, self.logical.shape[0]), dtype=np.uint8)

        if decoder=='tesseract':
            config  = tesseract.TesseractConfig(dem=dem, det_beam=defaults['det_beam'])
            tesseract_decoder = config.compile_decoder()

            for i in range(self.num_shots):
            
                decoded_errors    = tesseract_wrapper(tesseract_decoder,dem.num_errors)(det_events[i,:])
                logical_pred[i,:] = obs @ decoded_errors % 2 
            
        elif decoder=='relay_bp':
            
            relay_decoder = relay_bp.RelayDecoderF32(chk.tocsr(),                                   
                                    error_priors=np.array(priors,dtype=np.float64),
                                    gamma0=defaults['gamma0'],
                                    pre_iter=defaults['pre_iter'],
                                    num_sets=defaults['num_sets'],
                                    set_max_iter=defaults['set_max_iter'],
                                    gamma_dist_interval=defaults['gamma_dist_interval'],
                                    stop_nconv=defaults['stop_nconv'])

            for i in range(self.num_shots):
            
                decoded_errors    = relaybp_wrapper(relay_decoder)(det_events[i,:])
                logical_pred[i,:] = obs @ decoded_errors % 2 

        elif decoder=='bplsd':

            bplsd_decoder = BpLsdDecoder(chk,
                                error_channel=priors,
                                bp_method=defaults['bp_method'],
                                max_iter=defaults['max_bp_iters'],
                                schedule=defaults['schedule'],
                                lsd_method=defaults['lsd_method'],
                                lsd_order=defaults['lsd_order'])
            
            for i in range(self.num_shots):
            
                decoded_errors    = bplsd_decoder.decode(det_events[i,:])
                logical_pred[i,:] = obs @ decoded_errors % 2 

            

        else:
            raise NotImplemented("No other decoder choice besides tesseract, relay_bp and bplsd are available.")
        



        return np.any(self.obs_flips ^ logical_pred,axis=1)


    def calculate_ler(self,logical_errors_per_shot):

        return np.sum(logical_errors_per_shot["switching"])/self.num_shots




