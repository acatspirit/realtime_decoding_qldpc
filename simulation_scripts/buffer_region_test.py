import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move one level back out of simulation_scripts

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

import numpy as np
from src.realtime_decoding import decoder_switching_class
from joblib import Parallel, delayed


def test_buffer_region(p = 1e-3, num_shots = 100_000, basis='Z'):
    '''
    Test buffer region size to see how different LER is compared to decode_batching for d rounds.
    Buffer region of O(d) should keep the LER close to the LER obtained by decoding the full d rounds.
    
    Choice below runs relay_bp as strong decoder for bb codes with d\in[6,10,12] and various # of syndrome extraction rounds.

    Input:
        p: physical error strength for circuit-level noise
        num_shots: total # of shots to run
        basis: 'Z' or 'X'
    '''
    
    ds                    = [6,10,12]
    all_num_rounds        = [20, 30, 40, 50]
    strong_decoder        = 'relay_bp' #or tesseract
    weak_decoder          = 'bplsd'

    def process_one_round_value(d,num_rounds,num_shots):
        
        print("d,rds,shots:",(d,num_rounds,num_shots))

        nbuffer = d            #W-F = buffer -> W = buffer +F
        F       = d//2         #just pick this commit region so that there are at least some windows to decode for rds>20 (e.g. for d=12, we have 20 rounds, and W = 12 +6 = 18)
        W       = nbuffer + F  #entire window

        test  = decoder_switching_class(d=d,
                                        num_rounds=num_rounds,
                                        p=p,
                                        basis=basis,
                                        num_shots=num_shots,
                                        W=W,
                                        F=F,
                                        strong_decoder_option=strong_decoder,
                                        weak_decoder_option=weak_decoder,)    
        

        logical_errors_batch                 = test.decode_full_syndrome_history(strong_decoder)#decoding of d-rounds
        shots_sliding,logical_errors_sliding = test.decode_with_sliding_window('strong', norm_order=2)  #sliding window decoding with buffer size=d, commit = d//2

        #uncomment for weak decoder
        # logical_errors_batch                               = test.decode_full_syndrome_history(weak_decoder)#decoding of d-rounds
        # shots_sliding,cluster_norms,logical_errors_sliding = test.decode_with_sliding_window('weak',norm_order=2)  #sliding window decoding with buffer size=d, commit = d//2
        

        result = {"err_batch": np.sum(logical_errors_batch),
                  "err_sliding": np.sum(logical_errors_sliding),
                  "shots_sliding": shots_sliding,
                  "shots_batch": num_shots}

        return d,num_rounds,num_shots,result

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(200, num_shots // (100 * n_jobs)) #200

    for d in ds:
        for rd in all_num_rounds:

            tasks.extend(
                (d,rd, chunk_size)
                for _ in range(num_shots // chunk_size)
            )       

    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(d,rd,shots) for d,rd, shots in tasks)      
    print("Done.")
    
    total_errors = {}
    total_shots  = {}

    for d,rd,shot,result in results:
        total_errors[(d, rd,"batch")]   = 0
        total_errors[(d, rd,"sliding")] = 0
        total_shots[(d, rd,"batch")]    = 0    
        total_shots[(d, rd,"sliding")]  = 0    

    for d,rd,shot,result in results:

        print("rd,shot:",(rd,shot))
        total_errors[(d,rd,"batch")]   += result["err_batch"]
        total_errors[(d,rd,"sliding")] += result["err_sliding"]
        
        total_shots[(d,rd,"batch")]   += result["shots_batch"]
        total_shots[(d,rd,"sliding")] += result["shots_sliding"]


    ler_results_batch = {(d, rd): total_errors[(d, rd,"batch")] / total_shots[(d, rd,"batch")]
                        for d in ds 
                        for rd in all_num_rounds}

    ler_results_sliding = {(d, rd): total_errors[(d, rd,"sliding")] / total_shots[(d, rd, "sliding")]
                        for d in ds
                        for rd in all_num_rounds}
    
    err_batch = { (d, rd): np.sqrt(ler_results_batch[(d,rd,)]*(1-ler_results_batch[(d,rd)])/total_shots[(d, rd,"batch")]) 
                 for d in ds
                 for rd in all_num_rounds}
    
    err_sliding = { (d, rd): np.sqrt(ler_results_sliding[(d,rd)]*(1-ler_results_sliding[(d,rd)])/total_shots[(d, rd,"sliding")]) 
                   for d in ds
                   for rd in all_num_rounds}

    
    for d in ds:

        y_batch = {rd: ler_results_batch[(d,rd)] for rd in all_num_rounds}
        yerr_batch = {rd: err_batch[(d,rd)] for rd in all_num_rounds}

        #plot batch decoding
        plt.errorbar(all_num_rounds,y_batch.values(),yerr=yerr_batch.values(),label=f'batch, d={d}',marker='o')
        #plot sliding decoding

        y_sliding = {rd: ler_results_sliding[(d,rd)] for rd in all_num_rounds}
        yerr_sliding = {rd: err_sliding[(d,rd)] for rd in all_num_rounds}

        plt.errorbar(all_num_rounds,y_sliding.values(),yerr=yerr_sliding.values(),label=f'sliding, d={d}',linestyle='--',marker='o')

    plt.title(f"$n_{{buffer}}=d$, shots={num_shots}, p={p}")
    plt.yscale('log')
    plt.ylabel('LER')
    plt.xlabel('rounds')
    plt.legend(fontsize=12)
    plt.xticks(range(min(all_num_rounds),max(all_num_rounds),3))

    plt.show()

    return 
