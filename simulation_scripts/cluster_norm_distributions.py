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
import pickle 

'''For a specific code, check how the switch rate changes as a function of physical error rates

-Adjust the path to save the data
-Adjust the physical error rate range for uf
-Run the simulation once per code_name (see at the bottom)
-Cluster norm data can be loaded using the function "get_cutoffs_for_input_switch_rate" in ler_for_decoder_switching.py
to determine the target cutoff.
'''

def switch_rate_vs_p(code_name = "[[72,12,6]]", weak_decoder='bplsd',num_shots=10_000, shots_per_job=5_000,norm_order=2):

    basis      = 'Z' #basis determining the memory experiment for the BB codes
    
    decoder_option = 'weak'
    strong_decoder = 'relay_bp' #doesnt matter
    num_rounds = 25
    rel_error_tol = 0.1 #10%

    if weak_decoder == 'bplsd':
        ps = [2e-3,3e-3,4e-3,5e-3,6e-3,7e-3] #
    elif weak_decoder=='uf':
        ps = [1e-4,2e-4,3e-4,4e-4,5e-4]

    def process_one_round_value(code_name,p,num_shots,norm_order):
        
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
        
        new_shots,cluster_norms,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order,
                                                                                 rel_error_tol=rel_error_tol) 

        result = {"logical_errors": np.sum(logical_errors), "cluster_norms": cluster_norms}

        print("Sim done.")

        return code_name,p,new_shots,result,logical_errors

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(shots_per_job, num_shots // (100 * n_jobs)) 

    for p in ps:

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
            
    
    fig, ax = plt.subplots(2,1)

    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    cnt=0

    for p in ps:
        
        data     = cluster_norms[(code_name,p)].flatten()
        log_data = np.log10(data[data>0])

        ax[0].hist(
            log_data,
            bins=20,
            label=f"{code_name}, p={p}",
            color=colors[cnt],
            weights=np.ones_like(log_data) / len(log_data),
            alpha=0.7,
        )     

        ax[0].axvline(np.median(log_data), linestyle='--', color=colors[cnt]) #label='median',
        cnt+=1
        
    
    ax[0].set_xlabel(r'$\log_{10}(\mathrm{cluster\ norm})$')
    ax[0].set_ylabel("Norm. counts")
    ax[0].set_title(f"$N=${num_shots}, $r={num_rounds}$")
    ax[0].legend(fontsize=13)
    
    


    # Choose a common cutoff grid spanning all p values
    all_data = np.concatenate(  [cluster_norms[(code_name, p)].flatten() for p in ps] )

    gmin = np.min(all_data[all_data > 0])
    gmax = np.max(all_data)

    cutoffs = np.logspace(np.log10(gmin), np.log10(gmax), 150)

    switch_rates = np.zeros((len(ps), len(cutoffs)))

    for i, p in enumerate(ps):
        data = cluster_norms[(code_name, p)].flatten()

        for j, g_th in enumerate(cutoffs):
            switch_rates[i, j] = np.mean(data > g_th)

    switch_rates = np.ma.masked_equal(switch_rates, 0)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")   # masked values -> white


    from matplotlib.colors import LogNorm
    pcm = ax[1].pcolormesh(
        cutoffs,
        ps,
        switch_rates,
        shading="auto",
        cmap="viridis",
        norm=LogNorm(vmin=switch_rates.min(), vmax=switch_rates.max())
    )

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    ax[1].set_xlabel(r"$g_{\rm th}$")
    ax[1].set_ylabel(r"$p$")
    cbar = plt.colorbar(pcm, ax=ax[1])
    cbar.set_label(r"$p_{\rm switch}$")

    plt.tight_layout()
    plt.show()            

    dict_to_save = {'code_name': code_name,
                    'ps': ps,
                    'decoder': weak_decoder,
                    'norm_order': norm_order,
                    'cluster_norms': cluster_norms,
                    'switch_rates': switch_rates,
                    'cutoffs': cutoffs,
                    'all_cluster_norms_per_p': all_data

    }

     

    txt_to_save = sys.path[-1] + f'/saved_data/cluster_norm_statistics/cluster_norm_distributions_code_{code_name}_{weak_decoder}_max_shots_{num_shots}.txt'

    with open(txt_to_save, "wb") as file:
        pickle.dump(dict_to_save, file)

    #to load do:
    # with open(txt_to_load, "rb") as file:
    #     data = pickle.load(file)

    return 


code_name = "[[72,12,6]]" 
# code_name = "[[90,8,10]]" 
# code_name = "[[126,8,10]]"
# code_name = "[[144,12,12]]"
# code_name = "[[162,8,14]]"
num_shots = 100_000
shots_per_job = 10_000

switch_rate_vs_p(code_name = code_name, weak_decoder='uf',num_shots=num_shots,shots_per_job = shots_per_job,norm_order=2)
