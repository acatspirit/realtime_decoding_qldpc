import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move to level before sims file


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"
from joblib import Parallel, delayed
from src.realtime_decoding.decoder_switching_class import decoder_switching_class

'''Plot the distribution of cluster norms for all windows, and
   the switching rates.'''


def process_one_round_value(code_name,p,num_shots,norm_order, num_rounds=25, basis='Z', strong_decoder='relay_bp', decoder_option='weak', weak_decoder='bplsd'):
        
        print("Code_name,rds,p,shots:",(code_name,num_rounds,p,num_shots))

        n, k, d = map(int, code_name.strip("[]").split(","))
        

        nbuffer = d            #Buffer region
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
        
        new_shots,cluster_norms,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order) 

        result = {"logical_errors": np.sum(logical_errors), "cluster_norms": cluster_norms}

        print("Sim done.")

        return code_name,p,new_shots,result,logical_errors




def get_cluster_norm_distributions_and_switch_probs(weak_decoder='bplsd',num_shots=10_000,norm_order=2, p=1e-4):

    basis      = 'Z' #basis determining the memory experiment for the BB codes
    code_names = ["[[72,12,6]]", "[[90,8,10]]" ,"[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]   

    decoder_option = 'weak'
    strong_decoder = 'relay_bp' #doesnt matter
    num_rounds = 25

    # if weak_decoder == 'bplsd':
    #     p = 2e-3
    # elif weak_decoder=='uf':
    #     p = 1e-4     
        
    num_rounds = 25


    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(1_000, num_shots // (100 * n_jobs)) 

    for code_name in code_names:

        tasks.extend( (code_name,p,chunk_size)
                       for _ in range(num_shots // chunk_size) )       

    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots,norm_order) for code_name, p,shots in tasks)      



    total_errors  = {}
    total_shots   = {}
    cluster_norms = {}
    error_per_case = {}

    for code_name,p,shot,result,temp in results:
        total_errors[(code_name, p)] = 0
        total_shots[(code_name,  p)] = 0    
        error_per_case[(code_name,p)] = []

        if (code_name,p) not in cluster_norms:
                cluster_norms[(code_name,p)] = []

        cluster_norms[(code_name,p)].append(result["cluster_norms"])        
        

    for key in cluster_norms:
        cluster_norms[key] = np.concatenate(cluster_norms[key], axis=0)

    for code_name,p,shot,result,temp in results:

        total_errors[(code_name,p)] += result["logical_errors"]
        total_shots[(code_name,p)]  += shot
        error_per_case[(code_name,p)] = np.concatenate((error_per_case[(code_name,p)],temp),axis=0)
            
        
        
    ler_results = {(code_name,p,): total_errors[(code_name,p,)] / total_shots[(code_name,p,)]
                        for code_name in code_names
                        
                        }
    
    
    yerr_results = {(code_name,p, ): np.sqrt(ler_results[(code_name,p,)]*(1-ler_results[(code_name,p,)])/total_shots[(code_name,p,)])
                        for code_name in code_names
                        
                        }      
    


    fig, ax = plt.subplots(2,1)

    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    cnt=0

    
    for code_name in code_names:
        
        data     = cluster_norms[(code_name,p)].flatten()
        log_data = np.log10(data[data>0])

        ax[0].hist(
            log_data,
            bins=20,
            label=f"{code_name}",
            color=colors[cnt],
            weights=np.ones_like(log_data) / len(log_data),
            alpha=0.7,
        )     

        ax[0].axvline(np.median(log_data), linestyle='--', color=colors[cnt]) #label='median',
        # ax[0].axvline(np.quantile(log_data, 0.9), linestyle='-.', label='90%',color=colors[cnt])
        cnt+=1
        
    
    ax[0].set_xlabel(r'$\log_{10}(\mathrm{cluster\ norm})$')
    ax[0].set_ylabel("Norm. counts")
    ax[0].set_title(f"$p=${p}, $N=${num_shots}, $r={num_rounds}$")
    ax[0].legend(fontsize=13)
    
    
    #Now, let's also plot the switch rate as gamma_switch = \int_0^{g_th} dg p(g) given a threshold
    for code_name in code_names:
        
        data = cluster_norms[(code_name,p)].flatten()

        sorted_data = np.sort(data)
        cdf = np.arange(1, len(data)+1) / len(data) #gamma_siwtch = 1-CDF(g_th)

        ax[1].plot(sorted_data, 1-cdf, label=f"{code_name}",marker='o')

    ax[1].set_ylabel("$p_{switch}$")
    ax[1].set_xlabel("$g_{th}$")
    ax[1].legend(fontsize=13)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    plt.tight_layout()
    plt.show()        

 
    return 


weak_decoder = 'uf'
num_shots    = 20_000
norm_order   = 2


get_cluster_norm_distributions_and_switch_probs(weak_decoder=weak_decoder,
                                                num_shots=num_shots,
                                                norm_order=norm_order)
