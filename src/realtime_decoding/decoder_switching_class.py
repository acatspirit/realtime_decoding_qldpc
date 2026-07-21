import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) #move to level before src file


from quits import detector_error_model_to_matrix
from tesseract_decoder import tesseract
from ldpc.bplsd_decoder import BpLsdDecoder
import warnings

from quits.decoder import spacetime
import numpy as np
from src.realtime_decoding.helper_cluster_tools import * # figure out the improt

# fix the imports
from src.realtime_decoding.circuits import create_bb_codes_circuit, create_bb_codes_circuit_ionic_model, add_independent_leakage_errors_per_round
from src.realtime_decoding.decoders_utils import configure_tesseract_per_sliding_window, configure_bplsd_decoder_per_sliding_window, configure_relay_bp_per_sliding_window, configure_uf_decoder_per_sliding_window, collect_default_decoder_params
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
class uf_wrapper:
    def __init__(self, decoder, erasures, **params):
        """
        A sliding-window compatible wrapper for the custom nbi-hyq Union Find decoder.
        
        Parameters
        ----------
        check_matrix : array_like or csr_matrix
            The parity-check matrix (H) associated with the current decoding window.
        """
        # if not isinstance(check_matrix, csr_matrix):
        #     check_matrix = csr_matrix(check_matrix)
            
        # self.check_matrix = check_matrix
        
        # Initialize the underlying C-wrapped Union Find decoder instance
        self.decoder = decoder
        self.erasures = erasures
        
        # Public instance attribute for DecoderSwitchingWrapper to read
        self.cluster_sizes = None
        self.cluster_dict = None
        self.commit_region = None
        # self.committed_clusters = None
        
    def __call__(self, syndrome):
        """
        Decodes a window syndrome vector and returns a correction vector.
        
        Parameters
        ----------
        syndrome : np.ndarray
            The syndrome vector generated inside the current sliding window.
            
        Returns
        -------
        correction : np.ndarray
            The predicted correction vector extracted from the decoder instance.
        """
        # 1. Enforce strict contiguous data alignment types for the underlying C structures
        binary_syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        
        # 2. Process data via the method to populate internal attributes
        found_cluster_sizes, cluster_map = self.decoder.ldpc_decode(binary_syndrome, self.erasures) # sizes of cluster, list with index = fault id and value = cluster id
        
        # 3. Cache cluster stats metadata 
        # clusters in whole window
        self.cluster_sizes = found_cluster_sizes
        self.cluster_map = cluster_map
        # clusters in commit region
        self.committed_clusters = np.append(self.cluster_map[:self.commit_region], np.zeros(len(self.cluster_map) - self.commit_region, dtype=np.uint8))
        _, self.committed_cluster_sizes = np.unique(self.committed_clusters, return_counts=True)

        # 4. Extract the actual correction pattern produced by the decode logic
        correction = self.decoder.correction
        
        return correction
    
    def set_commit_region(self, commit_region):
        self.commit_region = commit_region

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

class decoder_switching_class:

    def __init__(self, code_name: str, num_rounds: int, p: float, basis: str, num_shots: int, W: int, F: int, 
                 strong_decoder_option: str, 
                 weak_decoder_option: str, 
                 strong_decoder_params: Optional[dict] = None,
                 weak_decoder_params: Optional[dict] = None,
                 p_leak = 0,
                 noise_model = "ionic"):
        
        '''
        Inputs:
        code_name: code_name of bb code of the form "[[n,k,d]]" (see circuits.py)
        num_rounds: # of syndrome extraction rounds
        p: physical error rate
        basis: 'Z' or 'X'
        num_shots: number of shots for memory experiment
        W: size of total window
        F: size of commit region
        strong_decoder_option: option to choose strong decoder from 'tesseract' or 'relay_bp'
        weak_decoder_option: option to choose weak decoder from 'uf' or 'bplsd'
        strong_decoder_params: dictionary with strong decoder parameters 
        weak_decoder_params: dictionary with weak decoder parameters 
        p_leak: leakage error rate per qubit, for each round (default set to 0)
        noise_model: "ionic" or "standard"

        ---decoder_params are optional. default parameters can be found in decoders_utils.py---
        '''

        if noise_model == "standard":
            circuit,bb = create_bb_codes_circuit(code_name, p, num_rounds, basis)
        elif noise_model == "ionic":
            circuit, bb = create_bb_codes_circuit_ionic_model(code_name, p, num_rounds, basis)
        else:
            NotImplementedError("No other noise models have been implemented")


        #Add leakage errors here (In either case we use dem that has only regular dets -- no leakage-aware decoding implement for now):

        if p_leak>0: #Sample from circuit that has leakage 
            n, _, _ = map(int, code_name.strip("[]").split(","))

            circuit_w_leakage, det_types = add_independent_leakage_errors_per_round(circuit,n,p_leak=p_leak)
            sampler    = circuit_w_leakage.compile_detector_sampler()

            detection_events_init,obs_flips = sampler.sample(shots=num_shots,separate_observables=True)
            detection_events_init = np.array(detection_events_init,dtype=np.uint8)

            self.leakage_detection_events = detection_events_init[:,det_types['leakage_dets']] #store other det events, in case we want to postselect on the no-leakage events
            
            detection_events = detection_events_init[:,det_types['regular_dets']] #restrict det events only to regular detectors (exclude dets used for leakage tracking)


        else: #Sample from regular circuit 

            sampler    = circuit.compile_detector_sampler()
            detection_events,obs_flips = sampler.sample(shots=num_shots,separate_observables=True)
            detection_events = np.array(detection_events,dtype=np.uint8)


        self.detection_events = detection_events
        self.obs_flips        = obs_flips

        self.circuit = circuit 
        self.bb      = bb 
        self.num_shots = num_shots
        
        if basis=='Z' or basis=='z':
            h = bb.hz 
            self.logical = bb.lz
            
        elif basis=='X' or basis=='x':
            h = bb.hx 
            self.logical = bb.lx

        self.h = h 
        self.W = W
        self.F = F
        self.weak_decoder_option = weak_decoder_option
        self.strong_decoder_option = strong_decoder_option
        # self.committed_clusters = []

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
            raise NotImplementedError("No other strong decoder besides tesseract or relay-bp is implemented for now. Choose from tesseract or relay_bp.")
        

        if weak_decoder_option=='bplsd':

            self.weak_decoder         = configure_bplsd_decoder_per_sliding_window(self.window_check_set, self.window_priors_set,weak_decoder_params)
            self.weak_decode_function = [getattr(decoder,"decode",None)
                                         for decoder in self.weak_decoder] 
        elif weak_decoder_option == 'uf':
            self.weak_decoder, erasures = configure_uf_decoder_per_sliding_window(self.window_check_set, self.window_priors_set,erasures=None, decoder_params=weak_decoder_params)
            self.weak_decode_function = [uf_wrapper(decoder, erasure_array) for decoder, erasure_array in zip(self.weak_decoder, erasures)]
        else:
            raise NotImplementedError("No other weak decoder besides bplsd and uf are implemented for now.")
        

        
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

        Inputs:
            F: commit region
            num_checks: # of checks (row size of parity check matrix)
            shot_index: index of i-th shot we decode
            syn_update: the syndrome update from previous window decoding
            accumulated_correction: the inferred logical flip so far
            num_cor_rounds: num of windows before last window (see __init__)
            norm_order: the order to evaluate cluster norm

        Outputs:
            accumulated_correction: updated predicted logical flip (length of # of logical qubits)
            cluster_norm: the calculated cluster norm for this window, calculated and normalized only over faults in commit region F
        '''
        k          = -1

        decoder         = self.weak_decoder[k]
        num_faults_in_F = self.window_observable_set[k].shape[1] #number of faults in commit region (num_faults_in_F=num_faults_in_W for last iteration)
        num_faults_in_W = np.shape(self.window_check_set[k])[1]  
        if self.weak_decoder_option == 'uf':
            self.weak_decode_function[k].set_commit_region(num_faults_in_F)
        diff_syndrome              = self.detection_events[shot_index, (F * num_cor_rounds) * num_checks:].copy()
        diff_syndrome[:num_checks] ^= syn_update
        
        decoded_errors = self.weak_decode_function[k](diff_syndrome)
        correction     = self.window_observable_set[num_cor_rounds] @ decoded_errors % 2


        #TODO: This needs to be handled externally -- need a general wrapper applicable for UF & BPLSD -- ideally configured upon initialization of the object
        if self.weak_decoder_option == 'bplsd':
            stats           = decoder.statistics
        elif self.weak_decoder_option == 'uf':
            # Handle UF decoder specific logic here
            # decoder.set_commit_region(num_faults_in_F)
            stats = np.array(self.weak_decode_function[k].cluster_map) # the map of committed clusters in the region set by F
        else:
            raise ValueError(f"Unsupported decoder type: {self.weak_decoder_option}")
        cluster_norm    = collect_cluster_norm(stats, num_faults_in_W,num_faults_in_F, norm_order, self.weak_decoder_option)      # add option for UF / BPLSD

        # accumulated_correction ^= (correction) 

        # TODO update this
        # self.committed_clusters.append(stats) # check this

        return accumulated_correction ^ correction,cluster_norm

    def decode_main_window_w_weak_decoder(self, W: int, F: int, num_checks: int, current_window_index: int, shot_index: int, syn_update, accumulated_correction, norm_order=2):
        '''
        Decode any window besides last window w/ the weak decoder.

        Inputs:
            W: total size of window
            F: commit region
            num_checks: # of checks (row size of parity check matrix)
            current_window_index: integer k>=0 which tracks the current window that we will process
            shot_index: index of i-th shot we decode
            syn_update: the syndrome update from previous window decoding
            accumulated_correction: the inferred logical flip so far
            norm_order: the order to evaluate cluster norm


        Outputs:
            syn_update: the syndrome update to apply to syndromes in next window
            accumulated_correction: updated predicted logical flip so far (length of # of logical qubits)
            cluster_norm: the calculated cluster norm for this window, calculated and normalized only over faults in commit region F
            
        '''

        k          = current_window_index
        decoder         = self.weak_decoder[k]
        num_faults_in_F = self.window_observable_set[k].shape[1] #number of faults in commit region F
        num_faults_in_W = np.shape(self.window_check_set[k])[1]  #number of faults in the entire window W
        if self.weak_decoder_option == 'uf':
            self.weak_decode_function[k].set_commit_region(num_faults_in_F)

        diff_syndrome              = self.detection_events[shot_index, F * k * num_checks:(F * k + W) * num_checks].copy()
        diff_syndrome[:num_checks] ^= syn_update   #update syndrome based on previous window decoding

        
        decoded_errors      = self.weak_decode_function[k](diff_syndrome) #Correction in entire window W, (only part F is committed)
        decoded_errors_in_F = decoded_errors[:self.window_observable_set[k].shape[1]]
        
        correction = self.window_observable_set[k] @ decoded_errors_in_F % 2  #the window_observable_set is set of observables only in region F. # of observables x # of faults in region F

        # syn_update = self.window_update[k] @ decoded_errors_in_F % 2

        if self.weak_decoder_option == 'bplsd':
            stats           = decoder.statistics
        elif self.weak_decoder_option == 'uf':
            stats           = np.array(self.weak_decode_function[k].cluster_map)
        else:
            raise ValueError(f"Unsupported decoder type: {self.weak_decoder_option}")
        
        
        cluster_norm    = collect_cluster_norm(stats, num_faults_in_W,num_faults_in_F, norm_order, self.weak_decoder_option)      

        # accumulated_correction ^= (correction) 

        # TODO update this
        # self.committed_clusters.append(stats) # check this

        return self.window_update[k] @ decoded_errors_in_F % 2, accumulated_correction ^ correction, cluster_norm
    
    def decode_main_window_w_strong_decoder(self, W: int, F: int, num_checks: int, current_window_index: int, shot_index: int, syn_update, accumulated_correction):
        '''
        Decode any window besides last window w/ the strong decoder.

        Inputs:
            W: total size of window
            F: commit region
            num_checks: # of checks (row size of parity check matrix)
            current_window_index: integer k>=0 which tracks the current window that we will process
            shot_index: index of i-th shot we decode
            syn_update: the syndrome update from previous window decoding
            accumulated_correction: the inferred logical flip so far


        Outputs:
            syn_update: the syndrome update to apply to syndromes in next window
            accumulated_correction: updated predicted logical flip so far (length of # of logical qubits)
            
        '''

        k          = current_window_index

        diff_syndrome              = self.detection_events[shot_index, F * k * num_checks:(F * k + W) * num_checks].copy()
        diff_syndrome[:num_checks] ^= syn_update
        
        decoded_errors      = self.strong_decode_function[k](diff_syndrome) #Correction in entire window W, (only part F is committed)
        decoded_errors_in_F = decoded_errors[:self.window_observable_set[k].shape[1]]
        
        correction = self.window_observable_set[k] @ decoded_errors_in_F % 2  #the window_observable_set is set of observables only in region F. # of observables x # of faults in region F

        syn_update = self.window_update[k] @ decoded_errors_in_F % 2

        accumulated_correction ^= (correction) 

        return syn_update,accumulated_correction
    
    def decode_last_window_w_strong_decoder(self, F: int, num_checks: int, shot_index: int, syn_update, num_cor_rounds: int, accumulated_correction):
        '''
        Decode the last window w/ the strong decoder.

        Inputs:
            F: commit region
            num_checks: # of checks (row size of parity check matrix)
            shot_index: index of i-th shot we decode
            syn_update: the syndrome update from previous window decoding
            accumulated_correction: the inferred logical flip so far


        Outputs:
            accumulated_correction: updated predicted logical flip so far (length of # of logical qubits)
            
        '''
        k          = -1

        diff_syndrome              = self.detection_events[shot_index, (F * num_cor_rounds) * num_checks:].copy()
        diff_syndrome[:num_checks] ^= syn_update
        
        decoded_errors = self.strong_decode_function[k](diff_syndrome)
        correction     = self.window_observable_set[num_cor_rounds] @ decoded_errors % 2

        accumulated_correction ^= (correction) 

        return accumulated_correction

    def decode_with_sliding_window_and_decoder_switching(self, cluster_norm_cutoff: float, norm_order=2, rel_error_tol = 0.2):
        '''
        Decode w/ sliding window and decoder switching from weak to strong decoder, if we exceed the specified cluster_norm_cutoff for any window, for a given shot.

        Input:
            cluster_norm_cuoff: max cluster size accepted to accept weak decoder's correction
            norm_order: integer defining the cluster norm order for the weak decoder
            reL_error_tol: relative error tolerance sigma_{p_L}/p_L where sigma_{p_L} = \sqrt{p_L*(1-p_L)/N}. If we reach the rel_error_tol, then we can exit early the computation.

        Outputs:
            N: new number of shots which can be different than self.num_shots, if we reached the rel_error accuracy faster than the total number of shots specified.
               N<=self.num_shots. If relative accuracy was not reached, output self.num_shots.
            cluster_norms_per_shot: a list of cluster norms lists per shot (inner list is cluster norms per window)
            switch_times_per_shot: a list of how many times in total we switched to the strong decoder per shot
            logical_errors_per_shot: returned both for weak/strong decoder and gives a 0/1 array of whether or not we made a logical error per shot
        '''

        num_checks       = self.h.shape[0]
        logical_pred = np.zeros((self.num_shots, self.logical.shape[0]), dtype=np.uint8)

        cluster_norms_per_shot = []
        switch_times_per_shot  = []
        
        W = self.W 
        F = self.F
        num_cor_rounds = self.num_cor_rounds

        failures_cnt   = 0               #count decoded logical failures
        epsilon        = rel_error_tol   #default is 20% relative error -- should be chosen based on how we simulate this externally (e.g., if we break into tasks of shots via multiprocessing we don't need a very small epsilon)
        shots_to_check = 20              #how often to check the precision in LER


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

            failures_cnt += np.mean(self.obs_flips[shot_index,:] ^ logical_pred[shot_index,:])

            if (shot_index + 1) % shots_to_check == 0 and failures_cnt > 0:
                N = shot_index + 1
                p = failures_cnt / N
                sigma = np.sqrt(p * (1 - p) / N)
                rel_err = sigma / p

                if rel_err<epsilon:

                    print("-------- Early exit. total # of shots vs shots run:", (self.num_shots,N))

                    return N, cluster_norms_per_shot, switch_times_per_shot, np.mean(self.obs_flips[:N,:] ^ logical_pred[:N,:],axis=1)             

        
        return self.num_shots, cluster_norms_per_shot,switch_times_per_shot,np.mean(self.obs_flips ^ logical_pred,axis=1)

    def decode_with_sliding_window(self, decoder_option: str, norm_order: int, rel_error_tol = 0.2):
        '''
        Decode w/ sliding window and no decoder switching. Choose weak or strong decoder option.

        Input:
            decoder_option: 'weak' or 'strong' and uses the weak or strong set upon initialization
            norm_order: integer defining the cluster norm order, in case we use the weak decoder
            reL_error_tol: relative error tolerance sigma_{p_L}/p_L where sigma_{p_L} = \sqrt{p_L*(1-p_L)/N}. If we reach the rel_error_tol, then we can exit early the computation.

        Outputs:
            N: new number of shots which can be different than self.num_shots, if we reached the rel_error accuracy faster than the total number of shots specified.
               N<=self.num_shots. If relative accuracy was not reached, output self.num_shots.
            cluster_norms_per_shot: if weak decoder was chosen, output a list of cluster norms lists per shot (inner list is cluster norms per window)
            logical_errors_per_shot: returned both for weak/strong decoder and gives a 0/1 array of whether or not we made a logical error per shot
        '''

        num_checks   = self.h.shape[0]
        num_logicals = self.window_observable_set[0].shape[0]
        logical_pred = np.zeros((self.num_shots, self.logical.shape[0]), dtype=np.uint8)

        W = self.W 
        F = self.F
        num_cor_rounds = self.num_cor_rounds
        
        failures_cnt   = 0               #count decoded logical failures
        epsilon        = rel_error_tol   #default is 20% relative error -- should be chosen based on how we simulate this externally (e.g., if we break into tasks of shots via multiprocessing we don't need a very small epsilon)
        shots_to_check = 20              #how often to check the precision in LER

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

                failures_cnt += np.mean(self.obs_flips[shot_index,:] ^ logical_pred[shot_index,:])

                if (shot_index + 1) % shots_to_check == 0 and failures_cnt > 0:
                    N = shot_index + 1
                    p = failures_cnt / N
                    sigma = np.sqrt(p * (1 - p) / N)
                    rel_err = sigma / p

                    if rel_err<epsilon:

                        print("-------- Early exit. total # of shots vs shots run:", (self.num_shots,N))

                        return N, cluster_norms_per_shot, np.mean(self.obs_flips[:N,:] ^ logical_pred[:N,:],axis=1) 

            
            return self.num_shots,cluster_norms_per_shot, np.mean(self.obs_flips ^ logical_pred,axis=1)

        elif decoder_option=='strong':

            
            for shot_index in range(self.num_shots):

                accumulated_correction = np.zeros(num_logicals, dtype=np.uint8)
                syn_update = np.zeros(num_checks, dtype=np.uint8)            
                
                for current_window_index in range(self.num_cor_rounds): #all windows besides last

                    syn_update,accumulated_correction = self.decode_main_window_w_strong_decoder(W,F,num_checks,current_window_index, shot_index, syn_update, accumulated_correction)
                    
                #decode the last window
                accumulated_correction = self.decode_last_window_w_strong_decoder(F, num_checks, shot_index, syn_update, num_cor_rounds, accumulated_correction)
                
                logical_pred[shot_index, :] = accumulated_correction

                failures_cnt += np.mean(self.obs_flips[shot_index,:] ^ logical_pred[shot_index,:])

                
                if (shot_index + 1) % shots_to_check == 0 and failures_cnt > 0:

                    N = shot_index + 1
                    p = failures_cnt / N
                    sigma = np.sqrt(p * (1 - p) / N)
                    rel_err = sigma / p

                    if rel_err<epsilon:

                        print("-------- Early exit. total # of shots vs shots run:", (self.num_shots,N))

                        return N, np.mean(self.obs_flips[:N,:] ^ logical_pred[:N,:],axis=1) #output updated shots


            return self.num_shots,np.mean(self.obs_flips ^ logical_pred,axis=1)

        return 

    def decode_full_syndrome_history(self,decoder: str):
        '''
        Do regular decoding of the entire syndrome history w/o sliding window. 
        
        Input:
            decoder: option for decoder which can be 'tesseract', 'bplsd', or 'relay_bp'
        Output:
            logical_errors_per_shot: 0/1 array whether or not we made a logical error for each shot
        '''

        defaults       = collect_default_decoder_params(decoder)  #import default parameters from decoders_utils.py
        dem            = self.circuit.detector_error_model()
        chk,obs,priors = detector_error_model_to_matrix(dem) 
        det_events     = self.detection_events

        logical_pred = np.zeros((self.num_shots, self.logical.shape[0]), dtype=np.uint8)

        if decoder=='tesseract':
              
            config  = tesseract.TesseractConfig(dem=dem, 
                                                det_beam=5, #defaults['det_beam']
                                                pqlimit=5_000, #defaults['pqlimit']
                                                beam_climbing=defaults['beam_climbing'],
                                                no_revisit_dets=defaults['no_revisit_dets'])#
            
            tesseract_decoder = config.compile_decoder()

            pred_flips   = tesseract_decoder.decode_batch(det_events) #decode_batch outputs directly the logical predictions
            logical_pred = np.mean(self.obs_flips ^ pred_flips, axis=1)

            return self.num_shots,logical_pred

            
        elif decoder=='relay_bp':
            
            relay_decoder = relay_bp.RelayDecoderF32(chk.tocsr(),                                   
                                                     error_priors=np.array(priors,dtype=np.float64),
                                                     gamma0=defaults['gamma0'],
                                                     pre_iter=defaults['pre_iter'],
                                                     num_sets=defaults['num_sets'],
                                                     set_max_iter=defaults['set_max_iter'],
                                                     gamma_dist_interval=defaults['gamma_dist_interval'],
                                                     stop_nconv=defaults['stop_nconv'])

            shots_to_check = 20
            epsilon        = 0.2
            failures_cnt   = 0

            for i in range(self.num_shots):
            
                decoded_errors    = relaybp_wrapper(relay_decoder)(det_events[i,:])
                logical_pred[i,:] = obs @ decoded_errors % 2 

                failures_cnt += np.mean(self.obs_flips[i,:] ^ logical_pred[i,:])

                if (i + 1) % shots_to_check == 0 and failures_cnt > 0:
                    N = i + 1
                    p = failures_cnt / N
                    sigma = np.sqrt(p * (1 - p) / N)
                    rel_err = sigma / p

                    if rel_err<epsilon:

                        print("-------- Early exit. total # of shots vs shots run:", (self.num_shots,N))
                        return N, np.mean(self.obs_flips[:N,:] ^ logical_pred[:N,:],axis=1)                

        elif decoder=='bplsd':

            bplsd_decoder = BpLsdDecoder(chk, 
                                        error_channel=priors,
                                        bp_method=defaults['bp_method'],
                                        max_iter=defaults['max_bp_iters'],
                                        schedule=defaults['schedule'],
                                        lsd_method=defaults['lsd_method'],
                                        lsd_order=defaults['lsd_order'])
            
            shots_to_check = 20
            epsilon        = 0.2
            failures_cnt   = 0
            for i in range(self.num_shots):
            
                decoded_errors    = bplsd_decoder.decode(det_events[i,:])
                logical_pred[i,:] = obs @ decoded_errors % 2 

                failures_cnt += np.mean(self.obs_flips[i,:] ^ logical_pred[i,:])

                if (i + 1) % shots_to_check == 0 and failures_cnt > 0:
                    N = i + 1
                    p = failures_cnt / N
                    sigma = np.sqrt(p * (1 - p) / N)
                    rel_err = sigma / p

                    if rel_err<epsilon:

                        print("-------- Early exit. total # of shots vs shots run:", (self.num_shots,N))
                        return N, np.mean(self.obs_flips[:N,:] ^ logical_pred[:N,:],axis=1)
            

        else:
            raise NotImplemented("No other decoder choice besides tesseract, relay_bp and bplsd are available for now.")

        return self.num_shots,np.mean(self.obs_flips ^ logical_pred,axis=1)



