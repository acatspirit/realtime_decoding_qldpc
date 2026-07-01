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


def decode_regular_d_rounds(num_shots=60_000, decoder= 'relay_bp'):
    '''
    Decode d-rounds of syndrome extraction w/o sliding window, to identify the thresholds.
    
    -For p=1e-3, 2e-3 we run num_shots shots in total, unless relative_error
    is reached before completing num_shots in total.

    -For p>=3e-3 we set a max limit of 30_000 shots, but could run fewer if relative_error
    is reached before completing the 30_000 shots.

    decoder: 'relay_bp', 'bplsd', or 'tesseract'.
    
    '''

    #Codes with same k:
    # code_names = ["[[54,8,6]]", "[[90,8,10]]]", "[[126,8,10]]", "[[162,8,14]]" ,  "[[180,8,16]]"] 

    #Codes w/ various k:
    code_names = ["[[72,12,6]]", "[[90,8,10]]]", "[[144,12,12]]"]
    

    ps                    = [1e-3,  2e-3, 3e-3,  4e-3, 4.5e-3, 5e-3] 
    basis                 = 'Z'

    
    def process_one_round_value(code_name,p,num_shots):

        n, k, d = map(int, code_name.strip("[]").split(","))
 
        num_rounds = d

        print("code,rds,p,shots:",(code_name,num_rounds,p,num_shots))

        #buffer, W, F, strong/weak decoders do not matter we are doing full decoding
        strong_decoder = 'relay_bp'
        weak_decoder   = 'bplsd'

        nbuffer = d            #W-F = buffer -> W = buffer +F
        F       = d//2         #just pick this commit region so that there are at least some windows to decode for rds>20 (e.g. for d=12, we have 20 rounds, and W = 12 + 6 = 18)
        W       = nbuffer + F  #entire window

        test  = decoder_switching_class(code_name=code_name,
                                            num_rounds=num_rounds,
                                            p=p,
                                            basis=basis,
                                            num_shots=num_shots,
                                            W=W,
                                            F=F,
                                            strong_decoder_option=strong_decoder,
                                            weak_decoder_option=weak_decoder)    

        #Choose decoder here 
        shots,logical_errors_weak = test.decode_full_syndrome_history(decoder)

        result = {"logical_errors": np.sum(logical_errors_weak)}

        print("Sim done.")

        return code_name,p,shots,result #returns updated shots


    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(2000, num_shots // (100 * n_jobs)) #200

    for code_name in code_names:

        for p in ps:
            
            if p>=3e-3: #30000 shots
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))

                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))

                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))   

                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))
                tasks.append((code_name, p,  1000))               




            else:

                tasks.extend(
                    (code_name,p,chunk_size)
                    for _ in range(num_shots // chunk_size) )       

    
    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots) for code_name, p,shots in tasks)      


    #---- Collect total logical errors and total # of shots ----------------------

    total_errors  = {}
    total_shots   = {}
    
    
    for code_name,p,shot,result in results:
        total_errors[(code_name, p)] = 0
        total_shots[(code_name,  p)] = 0    
        

    for code_name,p,shot,result in results:

        total_errors[(code_name,p)] += result["logical_errors"]
        total_shots[(code_name,p)]  += shot
        

    #logical error rates
    ler_results = {(code_name,p): total_errors[(code_name,p)] / total_shots[(code_name,p)]
                        for code_name in code_names
                        for p in ps
                        }
    
    #statistical uncertainty
    yerr_results = {(code_name,p): np.sqrt(ler_results[(code_name,p)]*(1-ler_results[(code_name,p)])/total_shots[(code_name,p)])
                        for code_name in code_names
                        for p in ps
                        
                        }      
    
    #---- Plot the results ----------------------    

    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig, ax = plt.subplots()
    cnt=0
    for code_name in code_names:

        y    = {p: ler_results[(code_name,p)] for p in ps}
        yerr = {p: yerr_results[(code_name,p)] for p in ps}

        ax.errorbar(ps,y.values(),yerr=yerr.values(),label=f"{code_name}, relay-bp",color=colors[cnt],marker='o')
        cnt+=1


    ax.set_xlabel("$p$")
    ax.set_ylabel("$P_L$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12,loc='best')

    

    plt.tight_layout()
    plt.show()        


    return 
