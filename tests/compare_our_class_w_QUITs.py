#Test that our class simulator produces the same as QUITs

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move one level back out of the tests folder


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"
from joblib import Parallel, delayed
from src.decoder_switching_class import decoder_switching_class


from quits import sliding_window_bplsd_circuit_mem


def run_specific_round_sliding_window_weak(num_shots=10_000):


    strong_decoder = 'relay_bp'
    weak_decoder   = 'bplsd'

    code_names = ["[[72,12,6]]", "[[90,8,10]]]", "[[126,8,10]]", ]   
    
    ps = [3e-3,4e-3,5e-3]

    basis                 = 'Z'
    
    num_rounds        = 22 

    def process_one_round_value(code_name,p,num_shots):
        
        print("Code_name,rds,p,shots:",(code_name,num_rounds,p,num_shots))

        n, k, d = map(int, code_name.strip("[]").split(","))
        

        nbuffer = d            #W-F = buffer -> W = buffer +F
        F       = d//2         #Commit region
        W       = nbuffer + F  #Entire window

        test  = decoder_switching_class(code_name=code_name,
                                            num_rounds=num_rounds,
                                            p=p,
                                            basis=basis,
                                            num_shots=num_shots,
                                            W=W,
                                            F=F,
                                            strong_decoder_option=strong_decoder,
                                            weak_decoder_option=weak_decoder)    

        zcheck_samples = test.detection_events
        
        new_shots,_,logical_errors = test.decode_with_sliding_window(decoder_option='weak',norm_order=2)

        #---Redo calculation w/ QUITS ---- should get the same -----
        
        logical_pred_quits = sliding_window_bplsd_circuit_mem(zcheck_samples,test.circuit,
                                         test.h,test.logical,W=W,F=F,max_iter=30,lsd_order=0,bp_method='product_sum',
                                         schedule='serial',lsd_method='lsd_cs')
        

        result = {"logical_errors": np.sum(logical_errors),"logical_errors_quits":np.sum(np.mean(test.obs_flips ^ logical_pred_quits,axis=1)),
                  "shots_quits":num_shots}

        print("Sim done.")

        return code_name,p,new_shots,result

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(200, num_shots // (100 * n_jobs)) #200
    for code_name in code_names:

        for p in ps:


            tasks.extend(
                (code_name,p,chunk_size)
                for _ in range(num_shots // chunk_size) )       


    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots) for code_name, p,shots in tasks)      



    total_errors  = {}
    total_shots   = {}

    total_shots_Q={}
    total_errors_Q={}
    

    for code_name,p,shot,result in results:
        total_errors[(code_name, p, )] = 0
        total_shots[(code_name,  p, )] = 0    
        
        #QUITS
        total_shots_Q[(code_name,p)] = 0
        total_errors_Q[(code_name, p, )] = 0
        

    for code_name,p,shot,result in results:

        total_errors[(code_name,p)] += result["logical_errors"]
        total_shots[(code_name,p)] += shot

        total_errors_Q[(code_name,p)] += result["logical_errors_quits"]
        total_shots_Q[(code_name,p)] += result["shots_quits"]
        
        
    ler_results = {(code_name,p,): total_errors[(code_name,p)] / total_shots[(code_name,p)]
                        for code_name in code_names
                        for p in ps
                        
                        }
    
    ler_results_Q = {(code_name,p,): total_errors_Q[(code_name,p)] / total_shots_Q[(code_name,p)]
                        for code_name in code_names
                        for p in ps
                        
                        }
    
    
    yerr_results = {(code_name,p, ): np.sqrt(ler_results[(code_name,p,)]*(1-ler_results[(code_name,p,)])/total_shots[(code_name,p,)])
                        for code_name in code_names
                        for p in ps
                        
                        }      


    yerr_results_Q = {(code_name,p, ): np.sqrt(ler_results_Q[(code_name,p,)]*(1-ler_results_Q[(code_name,p,)])/total_shots_Q[(code_name,p,)])
                        for code_name in code_names
                        for p in ps
                        
                        }      

    fig, ax = plt.subplots()

    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    cnt=0

    for code_name in code_names:

        n, k, d = map(int, code_name.strip("[]").split(","))

        pL_vals = {p: ler_results[(code_name,p,)] for p in ps}
        pL_errs = {p: yerr_results[(code_name,p,)] for p in ps}

        eps = {p: 1-(1-pL_vals[p])**(1/num_rounds) for p in ps}

        eps_errs = {
            p: (pL_errs[p] / num_rounds)
            * (1 - pL_vals[p])**(1 / num_rounds - 1)
            for p in ps
        }        

        ax.errorbar(ps,eps.values(),yerr=eps_errs.values(),label=f"{code_name}, {weak_decoder}",color=colors[cnt],marker='o',linestyle='--',markeredgecolor='k')

        #quits:
        pL_vals = {p: ler_results_Q[(code_name,p,)] for p in ps}
        pL_errs = {p: yerr_results_Q[(code_name,p,)] for p in ps}

        eps = {p: 1-(1-pL_vals[p])**(1/num_rounds) for p in ps}
        eps_errs = {
            p: (pL_errs[p] / num_rounds)
            * (1 - pL_vals[p])**(1 / num_rounds - 1)
            for p in ps
        }        

        ax.errorbar(ps,eps.values(),yerr=eps_errs.values(),label=f"{code_name}, {weak_decoder}, w/ QUITS",color=colors[cnt],marker='s',alpha=0.5,linestyle='-.',markeredgecolor='k')


        cnt+=1

    ax.set_xlabel("$p$")
    ax.set_ylabel("LER per round")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()        

    return 


run_specific_round_sliding_window_weak(num_shots=400)




