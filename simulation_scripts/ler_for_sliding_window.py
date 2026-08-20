import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move to level before src file

import json
import shutil
import subprocess

''' Get the ler performance for a single decoder w/ sliding window.'''


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"
from joblib import Parallel, delayed
from src.realtime_decoding.decoder_switching_class import decoder_switching_class
from scipy.optimize import curve_fit


'''Run this to get LER per syndrome extraction cycle by fitting epsilon'''
def get_ler_per_SEC_fitted_eps_from_many_rounds(num_shots=10_000,weak_decoder='bplsd',strong_decoder='relay_bp',decoder_option= 'weak',norm_order=2):
    '''
    Get the ler per syndrome extraction cycle (\epsilon). This quantity is fitted by calculating p_L(r) for different "r"
    and then fitting p_L(r)=1-(1-\epsilon)^r to get \epsilon. 
    The buffer region is fixed to O(d) for each code & the commit region to d//2.

    Inputs:
        num_shots: max number of shots to run the simulation (for p<8e-3, for p>=8e-3 we run fewer shots)
        weak_decoder: 'bplsd' or 'uf'
        strong_decoder: 'relay_bp' or 'tesseract'
        decoder_option: 'weak' or 'strong' to pick the weak/strong decoder for sliding window
        norm_order: order for calculating the cluster norm
    '''

    basis      = 'Z' #basis determining the memory experiment for the BB codes
    code_names = ["[[72,12,6]]", "[[90,8,10]]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]   

    ps             = [ 6e-3,  7e-3,  8e-3, 9e-3, 1e-2]  #physical error rates
    all_num_rounds = [ 25, 30, 35, 40] 
    
    
    def eqn_for_fit(n, eps):
        return (1 - (1 - eps)**n)
    
    def extract_fitted_epsilon(cycles,pL_vals,pL_errs):
        
        pL_errs = np.asarray(pL_errs)
        pL_errs = np.where(pL_errs == 0, 1e-6, pL_errs)        


        eps0 = np.mean(1 - (1 - np.asarray(pL_vals))**(1 / np.asarray(cycles)))
        eps0 = np.clip(eps0, 0, 1)

        popt, pcov = curve_fit(
            eqn_for_fit,
            cycles,
            pL_vals,
            sigma=pL_errs,
            absolute_sigma=True,
            p0=[eps0],
            bounds=(0, 1)
            )        
        eps_fit = popt[0]    
        eps_err = np.sqrt(pcov[0,0])

        return eps_fit, eps_err


    def process_one_round_value(code_name,p,num_rounds,num_shots,norm_order):
        
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
        
        if decoder_option=='strong':
            new_shots,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order)
        else:
            new_shots,_,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order) #suppress the cluster norms output

        result = {"logical_errors": np.sum(logical_errors)}

        print("Sim done.")

        return code_name,p,num_rounds,new_shots,result

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(200, num_shots // (100 * n_jobs)) #200
    for code_name in code_names:

        for rd in all_num_rounds:

            for p in ps:

                if p >= 8e-3:

                    tasks.append((code_name, p, rd, 50))
                    tasks.append((code_name, p, rd, 50))
                    tasks.append((code_name, p, rd, 50))
                    tasks.append((code_name, p, rd, 50))

                else:

                    tasks.extend(
                        (code_name,p,rd,chunk_size)
                        for _ in range(num_shots // chunk_size) )       


    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,rd,shots,norm_order) for code_name, p,rd,shots in tasks)      



    total_errors  = {}
    total_shots   = {}
    

    for code_name,p,rd,shot,result in results:
        total_errors[(code_name, p, rd)] = 0
        total_shots[(code_name,  p, rd)] = 0    
        

    for code_name,p,rd,shot,result in results:

        total_errors[(code_name,p,rd)] += result["logical_errors"]
        total_shots[(code_name,p,rd)] += shot
        
        
    ler_results = {(code_name,p,rd): total_errors[(code_name,p,rd)] / total_shots[(code_name,p,rd)]
                        for code_name in code_names
                        for p in ps
                        for rd in all_num_rounds
                        }
    

    
    yerr_results = {(code_name,p, rd): np.sqrt(ler_results[(code_name,p,rd)]*(1-ler_results[(code_name,p,rd)])/total_shots[(code_name,p,rd)])
                        for code_name in code_names
                        for p in ps
                        for rd in all_num_rounds
                        }      
    
    #Now for each ler per d, rd, p get the epsilon parameters.

    
    epsilon_fitted     = {(code_name,p): 0 for code_name in code_names for p in ps}
    epsilon_err_fitted = {(code_name,p): 0 for code_name in code_names for p in ps}

    pL_d     = {(code_name,p): 0 for code_name in code_names for p in ps}
    pL_d_err = {(code_name,p): 0 for code_name in code_names for p in ps}


    for code_name in code_names:

        n, k, d = map(int, code_name.strip("[]").split(","))

        for p in ps:
            pL_vals = {rd: ler_results[(code_name,p,rd)] for rd in all_num_rounds}
            pL_errs = {rd: yerr_results[(code_name,p,rd)] for rd in all_num_rounds}
            n_vals  = all_num_rounds



            
            eps_fit, eps_err = extract_fitted_epsilon(np.asarray(n_vals),np.asarray(list(pL_vals.values())),np.asarray(list(pL_errs.values())))

            epsilon_fitted[(code_name,p)] = eps_fit
            epsilon_err_fitted[(code_name,p)] = eps_err 



            #Using the fitted now get the final lers and their error
            pL_d[(code_name,p)]     = (1-(1-eps_fit)**d)
            pL_d_err[(code_name,p)] = abs( d * (1 - eps_fit)**(d - 1) ) * eps_err


    #Now plot pL_d and its error
    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig, ax = plt.subplots()
    cnt=0
    for code_name in code_names:
        y = {p: epsilon_fitted[(code_name,p)] for p in ps}
        yerr = {p: epsilon_err_fitted[(code_name,p)] for p in ps}

        ax.errorbar(ps,y.values(),yerr=yerr.values(),label=f"{code_name}, {decoder_label}",color=colors[cnt],marker='o',
                    markeredgecolor='k')
        cnt+=1

    ax.set_xlabel("$p$")
    ax.set_ylabel("LER per SEC")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12)


    # cnt=0
    # for code_name in code_names:
    #     y = {p: pL_d[(code_name,p)] for p in ps}
    #     yerr = {p: pL_d_err[(code_name,p)] for p in ps}

    #     ax[1].errorbar(ps,y.values(),yerr=yerr.values(),label=f"{code_name}, {decoder_label}",color=colors[cnt],marker='o')


    #     cnt+=1
        
    # ax[1].set_xlabel("$p$")
    # ax[1].set_ylabel("$p_L(d)$")
    # ax[1].set_yscale('log')
    # ax[1].set_xscale('log')
    # ax[1].legend(fontsize=12)
    

    plt.tight_layout()
    plt.show()        



    return 





# get_ler_per_SEC_fitted_eps_from_many_rounds(num_shots=1_000)


def get_ler_per_SEC_eps_extracted_from_one_round(num_shots=10_000,weak_decoder='uf',strong_decoder='relay_bp',decoder_option= 'weak',norm_order=2):
    '''
    Get the ler per syndrome extraction cycle (\epsilon). This quantity is calculated by simulating some fixed r
    and then extracting epsilon = 1-(1-p_L)^{1/r}.
    The buffer region is fixed to O(d) for each code & the commit region to d//2.

    Inputs:
        num_shots: max number of shots to run the simulation (for p<8e-3, for p>=8e-3 we run fewer shots)
        weak_decoder: 'bplsd' or 'uf'
        strong_decoder: 'relay_bp' or 'tesseract'
        decoder_option: 'weak' or 'strong' to pick the weak/strong decoder for sliding window
        norm_order: order for calculating the cluster norm
        
    '''

    basis      = 'Z' #basis determining the memory experiment for the BB codes
    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]   
    # code_names = ["[[72,12,6]]"]   

    # ps         = [6e-3,  7e-3,  8e-3, 9e-3, 1e-2]    
    # union find has a way lower threshold
    ps = np.logspace(-4, -2.5, num=10)  #physical error rates 
    # ps = [1e-4, 5e-4]
    num_rounds = 25
    max_shots_above_8e_minus3 = 1000 #this can be adjusted
    

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
        if decoder_option=='strong':
            new_shots,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order, rel_error_tol=0.05)
        else:
            new_shots,_,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order, rel_error_tol=0.05) #suppress cluster norms output

        result = {"logical_errors": np.sum(logical_errors)}

        print("Sim done.")

        return code_name,p,new_shots,result

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(200, num_shots // (100 * n_jobs)) 

    
    for code_name in code_names:

        for p in ps:

            if p >= 8e-3:
                
                for _ in range(max_shots_above_8e_minus3//50): #break into batches of 50
                    tasks.append((code_name, p, 50))

            else:

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
                    "max_shots_above_8e_minus3":max_shots_above_8e_minus3, #this is just the max we set for p>=8e-3
                    "total_errors":total_errors,
                    "shots":total_shots,                                   #these are the actual shots that were run for any code and p
                    "pL@r":ler_results,
                    "std_pL@r":yerr_results,
                    "epsilons":eps_to_save,
                    "std_epsilons":errs_in_eps_to_save}
    
    
    
    txt_to_save = sys.path[-1] + f'/data/raw/single_sliding_window_{decoder_label}_max_shots_{num_shots}.txt'

    with open(txt_to_save, 'w') as file:
        file.write(str(dict_to_save))      

    #To load the data simply do:
    # with open(txt_to_load,"r") as f:
    #     data = eval(f.read())

    

    ax.set_xlabel("$p$")
    ax.set_ylabel("LER per SEC")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12)

    plt.tight_layout()

    
    figure_plot = sys.path[-1] + f'/data/plots/single_sliding_window_{decoder_label}_max_shots_{num_shots}.pdf'
    

    fig.savefig(figure_plot,bbox_inches='tight')

    plt.show()        

    return 


# finish updating this before I get back
def get_ler_per_SEC_eps_extracted_from_one_round_switching(num_shots=10_000,weak_decoder='uf',strong_decoder='relay_bp',decoder_option= 'weak',cutoff=0.8,norm_order=2):
    '''
    Get the ler per syndrome extraction cycle (\epsilon). This quantity is calculated by simulating some fixed r
    and then extracting epsilon = 1-(1-p_L)^{1/r}.
    The buffer region is fixed to O(d) for each code & the commit region to d//2.

    Inputs:
        num_shots: max number of shots to run the simulation (for p<8e-3, for p>=8e-3 we run fewer shots)
        weak_decoder: 'bplsd' or 'uf'
        strong_decoder: 'relay_bp' or 'tesseract'
        decoder_option: 'weak' or 'strong' to pick the weak/strong decoder for sliding window
        norm_order: order for calculating the cluster norm
        
    '''

    basis      = 'Z' #basis determining the memory experiment for the BB codes
    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]   
    # code_names = ["[[72,12,6]]"]   

    # ps         = [6e-3,  7e-3,  8e-3, 9e-3, 1e-2]    
    # union find has a way lower threshold
    ps = np.logspace(-4, -2.5, num=10)  #physical error rates 
    # ps = [1e-4, 5e-4]
    num_rounds = 25
    max_shots_above_8e_minus3 = 1000 #this can be adjusted
    

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
        new_shots,_,switch_times_per_shot, logical_errors = test.decode_with_sliding_window_and_decoder_switching(cluster_norm_cutoff=cutoff,norm_order=norm_order, rel_error_tol=0.05) #suppress cluster norms output

        result = {"logical_errors": np.sum(logical_errors)}
        switches= {"num_switches": np.sum(switch_times_per_shot)}

        print("Sim done.")

        return code_name,p,new_shots,result, switches

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(200, num_shots // (100 * n_jobs)) 

    
    for code_name in code_names:

        for p in ps:

            if p >= 8e-3:
                
                for _ in range(max_shots_above_8e_minus3//50): #break into batches of 50
                    tasks.append((code_name, p, 50))

            else:

                tasks.extend(
                    (code_name,p,chunk_size)
                    for _ in range(num_shots // chunk_size) )       


    results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,shots,norm_order) for code_name, p,shots in tasks)      


    total_errors  = {}
    total_shots   = {}
    

    for code_name,p,shot,result,switch_times in results:
        total_errors[(code_name, p, )] = 0
        total_shots[(code_name,  p, )] = 0    
        

    for code_name,p,shot,result,switch_times in results:

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
                    "max_shots_above_8e_minus3":max_shots_above_8e_minus3, #this is just the max we set for p>=8e-3
                    "total_errors":total_errors,
                    "shots":total_shots,                                   #these are the actual shots that were run for any code and p
                    "pL@r":ler_results,
                    "std_pL@r":yerr_results,
                    "epsilons":eps_to_save,
                    "std_epsilons":errs_in_eps_to_save}
    
    
    
    txt_to_save = sys.path[-1] + f'/data/raw/single_sliding_window_switching_{decoder_label}_max_shots_{num_shots}_cutoff0.8.txt'

    with open(txt_to_save, 'w') as file:
        file.write(str(dict_to_save))      

    #To load the data simply do:
    # with open(txt_to_load,"r") as f:
    #     data = eval(f.read())

    

    ax.set_xlabel("$p$")
    ax.set_ylabel("LER per SEC")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12)

    plt.tight_layout()

    
    figure_plot = sys.path[-1] + f'/data/plots/single_sliding_window_switching_{decoder_label}_max_shots_{num_shots}_cutoff0.8.pdf'
    

    fig.savefig(figure_plot,bbox_inches='tight')

    plt.show()        

    return 

def get_ler_for_sliding_window_dcc(decoder_name, decoder_option='weak', num_shots=100_000, shots_per_job=10_000, norm_order=2, rel_error_tol=0.2):
    '''
    Inputs:
    decoder_name: the name of the decoder to use (e.g., 'uf', 'bplsd', 'relay_bp', 'tesseract')
    decoder_option: 'weak' or 'strong' - dictates which internal loop decode_with_sliding_window executes
    num_shots: max # of shots per (p,code)
    shots_per_job: how many shots to consider per job of joblib
    '''

    # Handle local testing fallback natively
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    basis      = 'Z'
    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]    
    ps = np.logspace(-4, -3.5, 6)
    num_rounds = 25
    
    tasks = []

    for code_name in code_names:
        for p in ps:
            for _ in range(num_shots // shots_per_job):
                tasks.append((code_name, p, shots_per_job))
                
    if task_id >= len(tasks):
        print(f"Task ID {task_id} is out of bounds for {len(tasks)} tasks. Exiting cleanly.")
        return

    code_name, p, shots = tasks[task_id]

    print(f"--- RUNNING ARRAY TASK {task_id} ---")
    print(f"Code: {code_name}, p: {p}, shots: {shots}, Decoder: {decoder_name} ({decoder_option})")

    n, k, d = map(int, code_name.strip("[]").split(","))
            
    nbuffer = d            # Buffer region
    F       = d//2         # Commit region
    W       = nbuffer + F  # Entire window

    # Route the decoder name to the proper initialization slot for the class
    weak_dec = decoder_name if decoder_option == 'weak' else 'uf' # uf is a dummy placeholder if strong is chosen
    strong_dec = decoder_name if decoder_option == 'strong' else 'relay_bp'

    test = decoder_switching_class(
        code_name=code_name,
        num_rounds=num_rounds,
        p=p,
        basis=basis,
        num_shots=shots,
        W=W,
        F=F,
        strong_decoder_option=strong_dec,
        weak_decoder_option=weak_dec
    )    
    
    # Run the sliding window function and unpack based on option
    if decoder_option == 'weak':
        new_shots, cluster_norms, logical_errors = test.decode_with_sliding_window(
            decoder_option=decoder_option, 
            norm_order=norm_order, 
            rel_error_tol=rel_error_tol
        )
    else:
        new_shots, logical_errors = test.decode_with_sliding_window(
            decoder_option=decoder_option, 
            norm_order=norm_order, 
            rel_error_tol=rel_error_tol
        )

    # Setup directories
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "data" / "sliding_window_data" / f"raw_batches_{decoder_name}_{decoder_option}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = output_dir / f"task_{task_id}_{code_name}_p{p:.6f}.json"
    
    # Get total block failures (logical_errors > 0 ensures any observable flip counts as a block failure)
    block_errors = int(np.sum(logical_errors > 0))

    dict_to_save = {
        "task_id": task_id,
        "basis": basis,
        "decoder_name": decoder_name,
        "decoder_option": decoder_option,
        "code_name": code_name,
        "p": p,
        "r": num_rounds,
        "shots_run": new_shots,
        "logical_errors": block_errors
    }

    with open(file_name, 'w') as file:
        json.dump(dict_to_save, file)
        
    print(f"Task {task_id} finished successfully. Saved to {file_name}")
    return

def download_from_dcc(remote_path, local_dir, username="am1155", host="dcc-login.oit.duke.edu"):
    """
    Downloads a file or directory from the Duke Compute Cluster (DCC) using scp.
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    scp_command = [
        "scp",
        "-r", 
        f"{username}@{host}:{remote_path}",
        str(local_path)
    ]
    
    print(f"Executing: {' '.join(scp_command)}")
    
    try:
        subprocess.run(scp_command, check=True)
        print(f"✅ Success! Data downloaded to: {local_path.absolute()}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred during download. Return code: {e.returncode}")
        print("Check if the remote path is correct and that you are connected to the Duke VPN.")

def merge_dcc_results_sliding_window(decoder_name, decoder_option, num_shots_max, dcc_data_dir="/hpc/group/brownlab/am1155/realtime_decoding_qldpc/simulation_scripts/data/sliding_window_data"):
    """
    After running on the DCC, converts data that belongs to one task into a full statistics dictionary.
    """

    # Setup paths using pathlib
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "sliding_window_data_temp" / f"raw_batches_{decoder_name}_{decoder_option}"
    
    out_dir = script_dir.parent / "data" / "sliding_window_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_to_save = out_dir / f'sliding_window_{decoder_name}_{decoder_option}_max_shots_{num_shots_max}.txt'

    download_from_dcc(
            remote_path=dcc_data_dir + f"/raw_batches_{decoder_name}_{decoder_option}/*.json",
            local_dir=input_dir
        )
    
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return
        
    # Initialize aggregators
    total_errors = {}
    total_shots = {}
    
    # Track static parameters to rebuild the final dictionary
    basis = None
    num_rounds = None
    code_names = set()
    ps = set()
    
    # --- STEP 0: Read existing file to prepopulate the dictionaries ---
    if txt_to_save.exists():
        try:
            with open(txt_to_save, 'r') as file:
                existing_dict = eval(file.read())
                
            print("Found existing file. Loading previous shots and errors...")
            
            basis = existing_dict.get("basis")
            num_rounds = existing_dict.get("r")
            
            old_errors = existing_dict.get("total_errors", {})
            old_shots = existing_dict.get("shots", {})
            
            for key in old_shots:
                total_errors[key] = old_errors.get(key, 0)
                total_shots[key] = old_shots.get(key, 0)
                code_names.add(key[0]) 
                ps.add(key[1])
                
        except Exception as e:
            print(f"⚠️ Could not load existing file. Starting fresh. Error: {e}")
            
    # 1. Find and aggregate all JSON files
    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} result files. Merging...")
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        code = data["code_name"]
        p = data["p"]
        key = (code, p)
        
        code_names.add(code)
        ps.add(p)
        
        # Grab static values from the first file we open
        if basis is None:
            basis = data.get("basis", "Z")
            num_rounds = data.get("r", 25)
            
        if key not in total_errors:
            total_errors[key] = 0
            total_shots[key] = 0
            
        # Add up the raw counts
        total_errors[key] += data["logical_errors"]
        total_shots[key] += data["shots_run"]

    def sort_by_d_then_k(code_str):
        try:
            parts = code_str.strip("[]").split(",")
            return (int(parts[2]), int(parts[1]))
        except (IndexError, ValueError):
            return (0, 0)
        
    code_names = sorted(list(code_names), key=sort_by_d_then_k)
    ps = sorted(list(ps))

    # 2. Calculate LER & Standard Errors
    ler_results = {}
    yerr_results = {}
    
    for code in code_names:
        for p in ps:
            key = (code, p)
            if key in total_shots and total_shots[key] > 0:
                ler = total_errors[key] / total_shots[key]
                ler_results[key] = ler
                yerr_results[key] = np.sqrt(ler * (1 - ler) / total_shots[key])
            else:
                ler_results[key] = np.nan
                yerr_results[key] = np.nan

    # 3. Calculate Epsilons
    eps_to_save = {}
    errs_in_eps_to_save = {}
    
    for code in code_names:
        pL_vals = {p: ler_results.get((code, p), np.nan) for p in ps}
        pL_errs = {p: yerr_results.get((code, p), np.nan) for p in ps}
        
        eps = {}
        eps_errs = {}
        for p in ps:
            ler = pL_vals[p]
            err = pL_errs[p]
            if not np.isnan(ler) and ler < 1.0: 
                eps[p] = 1 - (1 - ler)**(1 / num_rounds)
                eps_errs[p] = (err / num_rounds) * (1 - ler)**((1 / num_rounds) - 1)
            else:
                eps[p] = np.nan
                eps_errs[p] = np.nan
                
        eps_to_save[code] = eps
        errs_in_eps_to_save[code] = eps_errs

    # 4. Build Final Target Dictionary
    dict_to_save = {
        "basis": basis,
        "decoder_name": decoder_name,
        "decoder_option": decoder_option,
        "codes": code_names,
        "ps": ps,
        "r": num_rounds,
        "total_errors": total_errors,
        "shots": total_shots,
        "pL@r": ler_results,
        "std_pL@r": yerr_results,
        "epsilons": eps_to_save,
        "std_epsilons": errs_in_eps_to_save
    }
    
    # 5. Save and Cleanup
    with open(txt_to_save, 'w') as file:
        file.write(str(dict_to_save))
        
    print(f"Merge complete! Saved master dictionary to:\n{txt_to_save}")
    
    try:
        shutil.rmtree(input_dir.parent)
        print(f"🧹 Cleanup successful: Deleted raw data directory {input_dir.parent}")
    except Exception as e:
        print(f"⚠️ Could not delete directory {input_dir.parent}. Error: {e}")
        
    return dict_to_save

if __name__ == "__main__":
    num_shots      = 100_000
    weak_decoder   = 'uf'
    strong_decoder = 'tesseract'
    decoder_option = 'strong'
    cutoff=0.8

    get_ler_for_sliding_window_dcc(decoder_name=strong_decoder, decoder_option=decoder_option, num_shots=num_shots, shots_per_job=50_000, norm_order=2, rel_error_tol=0.05)
    # get_ler_per_SEC_eps_extracted_from_one_round_switching(num_shots=num_shots,
    #                                             weak_decoder=weak_decoder,
    #                                             strong_decoder=strong_decoder,
    #                                             decoder_option= decoder_option,
    #                                             cutoff=cutoff,
    #                                             norm_order=2)


    # txt_to_load = sys.path[-1] + f'/saved_data/single_sliding_window_{strong_decoder}_max_shots_{num_shots}.txt'
    # with open(txt_to_load,"r") as f:
    #     data = eval(f.read())




  



