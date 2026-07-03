import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move one level back out of tests folder

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

import numpy as np
from src.realtime_decoding import decoder_switching_class
from joblib import Parallel, delayed


'''Test if the ler we are getting for no leakage is the same as the ler when we post-select on the no leakage case (i.e., all leakage detectors recorder 0 -- no leakage)
'''

#NOTE: as leakage rate increases, the # of non-zero syndrome increases, and the simulation tends to slow down because
#the decoder tries to decode a higher density of syndromes

def unpack_results(results,ps,code_names):
    '''unpacking joblib results for the function below'''
    total_errors  = {}
    total_shots   = {}
    
    for code_name,p,shot,result in results:
        total_errors[(code_name, p,)] = 0
        total_shots[(code_name,  p,)] = 0    
    
    for code_name,p,shot,result in results:

        total_errors[(code_name,p)] += result["logical_errors"]
        total_shots[(code_name,p)]  += shot
        
        
    ler_results = {(code_name,p): total_errors[(code_name,p)] / total_shots[(code_name,p)]
                        for code_name in code_names
                        for p in ps
                        }
    
    yerr_results = {(code_name,p ): np.sqrt(ler_results[(code_name,p)]*(1-ler_results[(code_name,p)])/total_shots[(code_name,p)])
                        for code_name in code_names
                        for p in ps
                        
                        }      

    return ler_results,yerr_results

def test_postselection_on_no_leakage(num_shots=80_000):
    '''
    Pick the strong decoder (relay_bp) as an example.
    '''

    strong_decoder = 'relay_bp'
    weak_decoder   = 'bplsd'

    code_names = ["[[72,12,6]]",  ] #"[[90,8,10]]]", "[[144,12,12]]"

    ps         = [1.5e-3, 2e-3,  ] #Just pick these 2 values for the test
    basis      = 'Z'
    num_rounds = 20 #Fix number of rounds for the test

    #--- Main sim script ---------------------

    def process_one_round_value(code_name,p,num_shots,p_leak):
        
        n, k, d = map(int, code_name.strip("[]").split(","))

        print("code,rds,p,shots:",(code_name,num_rounds,p,num_shots))

        nbuffer = d            #W-F = buffer -> W = buffer +F
        F       = d//2         #just pick this commit region so that there are at least some windows to decode for rds>20 (e.g. for d=12, we have 20 rounds, and W = 12 +6 = 18)
        W       = nbuffer + F  #entire window

        test  = decoder_switching_class(code_name=code_name,
                                            num_rounds=num_rounds,
                                            p=p,
                                            basis=basis,
                                            num_shots=num_shots,
                                            W=W,
                                            F=F,
                                            strong_decoder_option=strong_decoder,
                                            weak_decoder_option=weak_decoder,
                                            p_leak=p_leak)    

        #Post-select on the no-leakage
        if p_leak>0:
            det_events_of_leakage_dets = test.leakage_det_events

            keep_shots_inds = []
            for k in range(num_shots):
                if sum(det_events_of_leakage_dets[k,:])==0: #no leakage detected
                    keep_shots_inds.append(k)

            if keep_shots_inds==[]:
                raise Exception("No surviving shot. Either reduce p_leak or increase the total # of shots.") 
            
            #Overwrite object's fields based on post-selected events
            test.detection_events = test.detection_events[keep_shots_inds,:]
            test.obs_flips        = test.obs_flips[keep_shots_inds,:]
            test.num_shots        = len(keep_shots_inds)

        new_num_shots,logical_errors_weak = test.decode_with_sliding_window(decoder_option='strong',norm_order=2)
        #new_num_shots could be lower if we satisfy rel_error in ler sooner than specified num_shots

        if p_leak>0:
            rejected_shots = num_shots-len(keep_shots_inds)

            print("rejected:",rejected_shots) 
        elif p_leak==0:
            rejected_shots = 0

        result = {"logical_errors": np.sum(logical_errors_weak)}

        print("Sim done.")

        return code_name,p,new_num_shots,result

    #-----------------------------------------


    #---- Use joblib for the simulation --------------
    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(500, num_shots // (100 * n_jobs)) #200

    for code_name in code_names:

            for p in ps:

                if p >= 4e-3: 
                    tasks.append((code_name, p, 500))

                else:

                    tasks.extend(
                        (code_name,p,chunk_size)
                        for _ in range(num_shots // chunk_size) )       

    
    #case for p_leak=0
    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots,p_leak=0) for code_name, p,shots in tasks)      

    ler_results_no_leak,yerr_results_no_leak = unpack_results(results,ps,code_names)

    #case for p_leak=1e-5
    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots,p_leak=1e-5) for code_name, p,shots in tasks)      
    ler_results_w_leak,yerr_results_w_leak = unpack_results(results,ps,code_names)

    
    #------------------- Plot the results ----------------------------------------

    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    cnt=0
    for code_name in code_names:

        y = {p: ler_results_no_leak[(code_name,p)] for p in ps}
        yerr =  {p: yerr_results_no_leak[(code_name,p)] for p in ps}

        plt.errorbar(ps, y.values(),yerr=yerr.values(), label= f"{code_name}, w/ leakage",color=colors[cnt],marker='o',linewidth=2,
                     markeredgecolor='k',markersize=15)
        
        y = {p: ler_results_w_leak[(code_name,p)] for p in ps}
        yerr =  {p: yerr_results_w_leak[(code_name,p)] for p in ps}
        
        plt.errorbar(ps, y.values(),yerr=yerr.values(), label= f"{code_name}, post-selected no leakage",color=colors[cnt],marker='s',linestyle='--')

        cnt+=1


    plt.xlabel("$p$")
    plt.ylabel("$P_L$")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=12,loc='best')

    plt.tight_layout()
    plt.show()    

    return 
