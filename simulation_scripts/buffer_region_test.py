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
    
    # ds                    = [6,10,12]
    code_names = ["[[72,12,6]]", "[[90,8,10]]]", "[[144,12,12]]"]

    all_num_rounds        = [20, 30, 40, 50]
    strong_decoder        = 'relay_bp' #or tesseract
    weak_decoder          = 'bplsd'

    def process_one_round_value(code_name,num_rounds,num_shots):
        
        print("code_name,rds,shots:",(code_name,num_rounds,num_shots))

        n, k, d = map(int, code_name.strip("[]").split(","))


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
                                        weak_decoder_option=weak_decoder,)    
        

        shots_batch,logical_errors_batch                 = test.decode_full_syndrome_history(strong_decoder)#decoding of d-rounds
        shots_sliding,logical_errors_sliding = test.decode_with_sliding_window('strong', norm_order=2)  #sliding window decoding with buffer size=d, commit = d//2

        #uncomment for weak decoder
        # logical_errors_batch                               = test.decode_full_syndrome_history(weak_decoder)#decoding of d-rounds
        # shots_sliding,cluster_norms,logical_errors_sliding = test.decode_with_sliding_window('weak',norm_order=2)  #sliding window decoding with buffer size=d, commit = d//2
        

        result = {"err_batch": np.sum(logical_errors_batch),
                  "err_sliding": np.sum(logical_errors_sliding),
                  "shots_sliding": shots_sliding,
                  "shots_batch": shots_batch}

        return code_name,num_rounds,result

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(200, num_shots // (100 * n_jobs)) #200

    for code_name in code_names:
        for rd in all_num_rounds:

            tasks.extend(
                (code_name,rd, chunk_size)
                for _ in range(num_shots // chunk_size)
            )       

    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,rd,shots) for code_name,rd, shots in tasks)      
    print("Done.")
    
    total_errors = {}
    total_shots  = {}

    for code_name,rd,result in results:
        total_errors[(code_name, rd,"batch")]   = 0
        total_errors[(code_name, rd,"sliding")] = 0
        total_shots[(code_name, rd,"batch")]    = 0    
        total_shots[(code_name, rd,"sliding")]  = 0    

    for code_name,rd,result in results:

        
        total_errors[(code_name,rd,"batch")]   += result["err_batch"]
        total_errors[(code_name,rd,"sliding")] += result["err_sliding"]
        
        total_shots[(code_name,rd,"batch")]   += result["shots_batch"]
        total_shots[(code_name,rd,"sliding")] += result["shots_sliding"]


    ler_results_batch = {(code_name, rd): total_errors[(code_name, rd,"batch")] / total_shots[(code_name, rd,"batch")]
                        for code_name in code_names
                        for rd in all_num_rounds}

    ler_results_sliding = {(code_name, rd): total_errors[(code_name, rd,"sliding")] / total_shots[(code_name, rd, "sliding")]
                        for code_name in code_names
                        for rd in all_num_rounds}
    
    err_batch = { (code_name, rd): np.sqrt(ler_results_batch[(code_name,rd,)]*(1-ler_results_batch[(code_name,rd)])/total_shots[(code_name, rd,"batch")]) 
                 for code_name in code_names
                 for rd in all_num_rounds}
    
    err_sliding = { (code_name, rd): np.sqrt(ler_results_sliding[(code_name,rd)]*(1-ler_results_sliding[(code_name,rd)])/total_shots[(code_name, rd,"sliding")]) 
                   for code_name in code_names
                   for rd in all_num_rounds}

    
    for code_name in code_names:

        y_batch = {rd: ler_results_batch[(code_name,rd)] for rd in all_num_rounds}
        yerr_batch = {rd: err_batch[(code_name,rd)] for rd in all_num_rounds}

        #plot batch decoding
        plt.errorbar(all_num_rounds,y_batch.values(),yerr=yerr_batch.values(),label=f'batch, {code_name}',marker='o')
        #plot sliding decoding

        y_sliding = {rd: ler_results_sliding[(code_name,rd)] for rd in all_num_rounds}
        yerr_sliding = {rd: err_sliding[(code_name,rd)] for rd in all_num_rounds}

        plt.errorbar(all_num_rounds,y_sliding.values(),yerr=yerr_sliding.values(),label=f'sliding, {code_name}',linestyle='--',marker='o')

    plt.title(f"$n_{{buffer}}=d$, p={p}")
    plt.yscale('log')
    plt.ylabel('LER')
    plt.xlabel('rounds')
    plt.legend(fontsize=12)
    plt.xticks(range(min(all_num_rounds),max(all_num_rounds),3))

    plt.show()

    return 
