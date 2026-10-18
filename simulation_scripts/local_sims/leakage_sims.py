import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #move to level before sims file
import numpy as np


from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

from joblib import Parallel, delayed
from src.realtime_decoding.decoder_switching_class import decoder_switching_class


import pickle 

def get_cutoffs_for_input_switch_rate(target_switch_rate,plot=False):
    '''
    Load the cluster norm distributions extracted from bplsd, and estimate the cutoffs required to achieve a target switch rate.
    Note the cluster norms were obtained under Pauli only noise (no leakage or erasures).

    Input:
        target_switch_rate: the target switch rate, p_s, to achieve for each code and "p"
        plot: whether or not to plot the distributions

    Output:
        cutoffs_to_set: dictionary of with keys (code_name,p) and values the cluster norm cutoffs
        data_per_code: list with cluster norms per code
    '''

    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"] 

    weak_decoder = 'bplsd'
    num_shots = 100_000

    cutoffs_to_set = {}

    if plot:
        fig,ax = plt.subplots(1,5,figsize=(20,5))

    data_per_code = []
    cnt=0
    for code_name in code_names:
    
        txt_to_load = sys.path[-1] + f'/saved_data/cluster_norm_statistics_final/bplsd/cluster_norm_distributions_code_{code_name}_{weak_decoder}_max_shots_{num_shots}.txt'

        with open(txt_to_load, "rb") as file:
            data = pickle.load(file)

        data_per_code.append(data)

        ps = data['ps']

        switch_rates   = data['switch_rates']
        cutoffs = data['cutoffs']
        
        
        for k in range(len(ps)):
            
            key = (code_name,ps[k])
            diff = np.abs(switch_rates[k] - target_switch_rate) 
            locs = np.argmin(diff)                   #Find location for which switch rate is closest to our target switch rate 
            cutoffs_to_set[key] = cutoffs[locs]      #Collect the cutoff value

            if plot:

                ax[cnt].semilogx(cutoffs, switch_rates[k], marker='.',label=f' p={ps[k]}')
                ax[cnt].axhline(target_switch_rate)
                
                ax[cnt].set_xlabel("cutoff")
                if cnt==0:
                    ax[cnt].set_ylabel("switch rate")
                ax[cnt].grid()
                ax[cnt].set_yscale('log')
                # ax[cnt].set_xscale('log')
                ax[cnt].legend(fontsize=10)
                ax[cnt].set_title(code_name)
        cnt+=1
                
    if plot:
        print(cutoffs_to_set)
        plt.tight_layout()
        plt.show()

    

    return cutoffs_to_set,data_per_code

def bplsd_to_strong_w_target_switch_rate(p_leak=1e-5,target_switch_rate=1e-2,strong_decoder='tesseract',num_shots=100_000,shots_per_job=10_000):
    '''
    Run decoder switching w/ leakage errors, for bplsd to tesseract or relay_bp.

    Input:
        p_leak: leakage error rate per cycle
        target_switch_rate: switch rate probability to achieve per code (estimated from Pauli only noise)
        strong_decoder: "tesseract" or "relay_bp"
        num_shots: maximum # of shots for each (code_name,p)
        shots_per_job: # of shots for each worker

    '''

    weak_decoder='bplsd'
    cutoffs_to_set,_ = get_cutoffs_for_input_switch_rate(target_switch_rate=target_switch_rate)

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]


    basis = 'Z'
    num_rounds = 25
    code_names = ["[[72,12,6]]","[[90,8,10]]","[[126,8,10]]","[[144,12,12]]","[[162,8,14]]"]
    
    ps         = [3e-3,] 
    

    def process_one_round_value(code_name,p,num_shots,cutoff):
        
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
                                            weak_decoder_option=weak_decoder,
                                            p_leak=p_leak)    
        
        new_shots,cluster_norms,switch_times,logical_errors = test.decode_with_sliding_window_and_decoder_switching(cluster_norm_cutoff=cutoff)


        num_windows =len(test.weak_decoder)

        # print("-----shape of switch times:",np.shape(switch_times))
        # print("-----shape of logical errors:",np.shape(logical_errors))
        # print("-----num_windows:",num_windows)

        switch_prob = np.sum(switch_times)/(new_shots*num_windows)
        

        result = {"logical_errors": np.sum(logical_errors),"cluster_norms": cluster_norms,
                  "switch_times": np.sum(switch_times), "num_windows": num_windows}
        

        print("Sim done.")

        return code_name,p,new_shots,result
    
    

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(shots_per_job, num_shots // (100 * n_jobs)) 

    for code_name in code_names:

        for p in ps:

            tasks.extend(
                (code_name,p,chunk_size)
                for _ in range(num_shots // chunk_size) )       


    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots,cutoffs_to_set[(code_name,p)]) for code_name, p,shots in tasks)      
    
    total_errors  = {}
    total_shots   = {}
    total_switch_times = {}
    total_windows = {}

    for code_name,p,shot,result in results:
        key = (code_name,p)
        total_errors[key] = 0
        total_shots[key] = 0    
        total_switch_times[key] = 0    
        total_windows[key] = 0
        

    for code_name,p,shot,result in results:
        key = (code_name,p)
        total_errors[key] += result["logical_errors"]
        total_shots[key] += shot
        total_switch_times[key] += result["switch_times"]

        total_windows[key] += result['num_windows'] * shot
        
        
    ler_results = {(code_name,p,): total_errors[(code_name,p,)] / total_shots[(code_name,p,)]
                        for code_name in code_names
                        for p in ps
                        }
    
    
    yerr_results = {(code_name,p, ): np.sqrt(ler_results[(code_name,p,)]*(1-ler_results[(code_name,p,)])/total_shots[(code_name,p,)])
                        for code_name in code_names
                        for p in ps
                        }      
    

    fig, ax = plt.subplots(1,2)

    eps_to_save = {}
    errs_in_eps_to_save ={}
    #Plot decoder switching results:
    cnt=0
    for code_name in code_names:

        n, k, d = map(int, code_name.strip("[]").split(","))

        pL_vals = {p: ler_results[(code_name,p,)] for p in ps}
        pL_errs = {p: yerr_results[(code_name,p,)] for p in ps}

        eps = {p: 1-(1-pL_vals[p])**(1/num_rounds) for p in ps}

        eps_errs = { p: (pL_errs[p] / num_rounds) * (1 - pL_vals[p])**(1 / num_rounds - 1)
                    for p in ps }        

        ax[0].errorbar(ps,eps.values(),yerr=eps_errs.values(),label=f"{code_name}, dec. sw.",color=colors[cnt],marker='o',markeredgecolor='k')
        cnt+=1

        eps_to_save[code_name] = eps
        errs_in_eps_to_save[code_name] = eps_errs 



    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].legend(fontsize=9,)
    ax[0].set_xlabel("physical error rate")
    ax[0].set_ylabel("LER per SEC")

    

    switch_rates = {
        key: total_switch_times[key] / total_windows[key]
        for key in total_switch_times
    }    

    switch_yerr = {
        key: np.sqrt(
            switch_rates[key] * (1 - switch_rates[key]) / total_windows[key]
        )
        for key in switch_rates
    }    

    cnt=0

    for code_name in code_names:
        # all_switch_rates = {p: total_switch_times[(code_name,p)] for p in ps}
        y = np.array([ switch_rates[(code_name, p)] for p in ps ])

        yerr = np.array([ switch_yerr[(code_name, p)] for p in ps ])

        
        ax[1].errorbar(ps,y,yerr=yerr, marker='o',color=colors[cnt],label=f"{code_name}")   
        cnt+=1 

    

    ax[1].set_ylabel("$p_{switch}$")
    ax[1].set_xlabel("physical error rate")
    ax[1].set_title(f"for $p_s$: {target_switch_rate}")
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].legend(fontsize=9)


  

    plt.tight_layout()
    plt.show()

    dict_to_save = {"basis":basis,
                    "weak_decoder": weak_decoder,
                    "strong_decoder": strong_decoder,
                    "target_switch_rate": target_switch_rate,
                    "codes": code_names,
                    "ps": ps,
                    "r":num_rounds,
                    "total_errors":total_errors,
                    "shots":total_shots,                                   #these are the actual shots that were run for any code and p
                    "pL@r":ler_results,
                    "std_pL@r":yerr_results,
                    "epsilons":eps_to_save,
                    "std_epsilons":errs_in_eps_to_save,
                    "switch_rates": switch_rates,
                    "switch_rate_err": switch_yerr,
                    "total_switch_times": total_switch_times,
                    "total_windows": total_windows,
                    "p_leak": p_leak}         

    txt_to_save = sys.path[-1] + f'/saved_data/decoder_switching_data/leakage_bplsd_to_{strong_decoder}/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3_p_leak_{p_leak}.txt'

    # with open(txt_to_save, 'w') as file:
    #     file.write(str(dict_to_save))      



    return 

#Run this for decoder switching
# bplsd_to_strong_w_target_switch_rate(p_leak=1e-5,target_switch_rate=1e-2,strong_decoder='relay_bp')


def get_ler_per_SEC(p_leak=1e-5,num_shots=100_000,shots_per_job=10_000,weak_decoder='bplsd',strong_decoder='relay_bp',decoder_option= 'weak',norm_order=2):
    '''
    Get the ler per syndrome extraction cycle (\epsilon). This quantity is calculated by simulating some fixed r
    and then extracting epsilon = 1-(1-p_L)^{1/r}.
    The buffer region is fixed to O(d) for each code & the commit region to d//2.

    Inputs:
        p_leak: leakage_error rate per cycle
        num_shots: max number of shots 
        shots_per_job: # of shots per worker
        weak_decoder: 'bplsd' or 'uf'
        strong_decoder: 'relay_bp' or 'tesseract'
        decoder_option: 'weak' or 'strong' to pick the weak/strong decoder for sliding window
        norm_order: order for calculating the cluster norm
        
    '''

    basis      = 'Z' 
    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]   
    ps = [3e-3,] 
    num_rounds = 25
    

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
                                            weak_decoder_option=weak_decoder,
                                            p_leak=p_leak)    
        if decoder_option=='strong':
            new_shots,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order)
        else:
            new_shots,_,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order) #suppress cluster norms output

        result = {"logical_errors": np.sum(logical_errors)}

        print("Sim done.")

        return code_name,p,new_shots,result

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(shots_per_job, num_shots // (100 * n_jobs)) 

    for code_name in code_names:

        for p in ps:

                tasks.extend(
                    (code_name,p,chunk_size)
                    for _ in range(num_shots // chunk_size) )       

    

    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots,norm_order) for code_name, p,shots in tasks)      


    total_errors  = {}
    total_shots   = {}
    

    for code_name,p,shot,result in results:
        total_errors[(code_name, p, )] = 0
        total_shots[(code_name,  p, )] = 0    
        

    for code_name,p,shot,result in results:

        total_errors[(code_name,p)] += result["logical_errors"]
        total_shots[(code_name,p,)] += shot
        
        
    ler_results = {(code_name,p,): total_errors[(code_name,p,)] / total_shots[(code_name,p,)]
                        for code_name in code_names
                        for p in ps
                        }
    
    
    yerr_results = {(code_name,p, ): np.sqrt(ler_results[(code_name,p,)]*(1-ler_results[(code_name,p,)])/total_shots[(code_name,p,)])
                        for code_name in code_names
                        for p in ps
                        }      
    

    fig, ax = plt.subplots()

    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    cnt=0


    eps_to_save = {}
    errs_in_eps_to_save = {}

    if decoder_option =='weak':
        decoder_label = weak_decoder 
    else:
        decoder_label = strong_decoder


    for code_name in code_names:

        n, k, d = map(int, code_name.strip("[]").split(","))

        pL_vals = {p: ler_results[(code_name,p,)] for p in ps}
        pL_errs = {p: yerr_results[(code_name,p,)] for p in ps}

        eps = {p: 1-(1-pL_vals[p])**(1/num_rounds) for p in ps}

        eps_errs = { p: (pL_errs[p] / num_rounds) * (1 - pL_vals[p])**(1 / num_rounds - 1)
                    for p in ps }        

        ax.errorbar(ps,eps.values(),yerr=eps_errs.values(),label=f"{code_name}, {decoder_label}",color=colors[cnt],marker='o',markeredgecolor='k')
        cnt+=1

        eps_to_save[code_name] = eps
        errs_in_eps_to_save[code_name] = eps_errs

    
    dict_to_save = {"basis":basis,
                    "codes": code_names,
                    "ps": ps,
                    "r":num_rounds,
                    "total_errors":total_errors,
                    "shots":total_shots,                                   #these are the actual shots that were run for any code and p
                    "pL@r":ler_results,
                    "std_pL@r":yerr_results,
                    "epsilons":eps_to_save,
                    "std_epsilons":errs_in_eps_to_save}
    

    if decoder_option=='weak':
        decoder_label = weak_decoder 
    else:
        decoder_label = strong_decoder
    
    txt_to_save = sys.path[-1] + f'/saved_data/single_decoder_sliding/{decoder_label}/single_sliding_window_{decoder_label}_max_shots_{num_shots}_p_3e_minus_3_p_leak_{p_leak}.txt' #single_sliding_window_{decoder_label}_max_shots_{num_shots}_p_3e_minus_3_p_leak_{p_leak}.txt

    # with open(txt_to_save, 'w') as file:
    #     file.write(str(dict_to_save))      

    # To load the data simply do:
    # with open(txt_to_load,"r") as f:
    #     data = eval(f.read())
    

    ax.set_xlabel("$p$")
    ax.set_ylabel("LER per SEC")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12)

    plt.tight_layout()

    
    # figure_plot = sys.path[-1] + f'/saved_data/single_decoder_sliding/{decoder_label}/single_sliding_window_{decoder_label}_max_shots_{num_shots}_p_3e_minus_3_p_leak_{p_leak}.pdf'
    

    # fig.savefig(figure_plot,bbox_inches='tight')

    plt.show()        

    return 



#Run this
get_ler_per_SEC(p_leak=1e-5,num_shots=5000,shots_per_job=500,weak_decoder='bplsd',strong_decoder='relay_bp',decoder_option= 'strong',norm_order=2)