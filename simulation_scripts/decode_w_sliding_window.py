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

from scipy.optimize import curve_fit

def eqn_for_fit(n, eps):
    '''
    Function p_L(n) = 1-(1-epsilon)^n of logical error rate @ n cycles, to be used for fitting an epsilon value.

    Input:
        n: number of cycles
        eps: epsilon value

    Output:
        1-(1-epsilon)^n
    '''

    return (1 - (1 - eps)**n)


def extract_fitted_epsilon(cycles,pL_vals,pL_errs):
    '''
    Perform the fit to find the epsilon, which is the number of logical errors per cycle.

    Input:
        cycles: array of cycle numbers (list of ints)
        pL_vals: extracted logical error rate from decoding n cycles (list)
        pL_errs: standard deviations \sigma for p_L for each cycle   (list)
    
    Output:
        eps_fit: fitted epsilon value
        eps_err: the uncertainty associated with epsilon
    '''

    pL_errs = np.asarray(pL_errs)
    pL_errs = np.where(pL_errs == 0, 1e-6, pL_errs)    #if some ler is 0, then we should set some nnz value

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


def process_one_round_value_strong(code_name: str, strong_decoder: str, num_rounds: int, p: float, num_shots: int, basis='Z'):
    '''
    Run sliding window decoding sim for strong decoder, for a particular value of cycle number and physical error rate.

    Inputs:
        code_name: "[[n,k,d]]" format (see choices in circuits.py)
        strong_decoder: choose from 'relay_bp' or 'tesseract'
        num_rounds: number of cycles to run sliding window decoding
        p: physical error rate
        num_shots: # of decoding shots
        basis: 'Z' or 'X' to run Z or X memory

    Outputs:
        code_name: "[[n,k,d]]" format
        p: physical error rate
        num_rounds: num_rounds
        result: dictionary with keys "logical_error" and "shots" 
    '''
    n, k, d = map(int, code_name.strip("[]").split(","))

    print("code,rds,p,shots:",(code_name,num_rounds,p,num_shots))


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
                                        weak_decoder_option='bplsd')     #weak doesn't matter here

    
    shots,logical_errors = test.decode_with_sliding_window(decoder_option='strong',
                                                                norm_order=2,) 

    result = {"logical_errors": np.sum(logical_errors), "shots" : shots}

    print("Sim done.")

    return code_name,p,num_rounds,result 


def process_one_round_value_weak(code_name: str, weak_decoder: str, num_rounds: int, p: float, num_shots: int, basis='Z'):
    '''
    Run sliding window decoding sim for weak decoder, for a particular value of cycle number and physical error rate.

    Inputs:
        code_name: "[[n,k,d]]" format (see choices in circuits.py)
        weak_decoder: choose from 'bplsd' or 'uf' (NOTE: UF is not implemented yet)
        num_rounds: number of cycles to run sliding window decoding
        p: physical error rate
        num_shots: # of decoding shots
        basis: 'Z' or 'X' to run Z or X memory

    Outputs:
        code_name: "[[n,k,d]]" format
        p: physical error rate
        num_rounds: num_rounds
        result: dictionary with keys "logical_error", "shots" , and "cluster_norms"
    '''

    n, k, d = map(int, code_name.strip("[]").split(","))

    print("code,rds,p,shots:",(code_name,num_rounds,p,num_shots))


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
                                        strong_decoder_option='relay_bp', #strong decoder doesn't matter
                                        weak_decoder_option=weak_decoder)    

    
    shots,cluster_norms_per_shot,logical_errors = test.decode_with_sliding_window(decoder_option='weak',
                                                                norm_order=2,) 

    result = {"logical_errors": np.sum(logical_errors), "shots" : shots, "cluster_norms": cluster_norms_per_shot}

    print("Sim done.")

    return code_name,p,num_rounds,result 


def decode_single_decoder_sliding_window(num_shots=60_000, strong_decoder = 'relay_bp', weak_decoder= 'bplsd', decoder_option='weak'):
    '''
    Decode w/ sliding window using 1 decoder. Fit epsilon (ler per round), and then plot pL @d rounds.
        
    Inputs:
        num_shots: number of decoding shots
        strong_decoder:  'relay_bp' or 'tesseract'
        weak_decoder: 'bplsd' or 'uf' (NOTE: uf is not implemented yet)
        decoder_option: 'weak' or 'strong' to select the weak or strong decoder to do sliding window.


    -For p=1e-3, 2e-3 we run num_shots shots in total, unless relative_error
    is reached before completing num_shots in total.

    -For p>=3e-3 we set a max limit of 30_000 shots, but could run fewer if relative_error
    is reached before completing the 30_000 shots.

    -By default we run Z-basis.

    '''

    #Codes w/ various k:
    code_names = ["[[72,12,6]]", "[[90,8,10]]]", "[[144,12,12]]"]

    ps                    = [1e-3,  2e-3, 3e-3,  4e-3, 4.5e-3, 5e-3] 
    all_num_rounds        = [20, 25, 30, 35]

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(2000, num_shots // (100 * n_jobs)) #200

    for code_name in code_names:

        for rd in all_num_rounds:

            for p in ps:
                
                if p>=3e-3: #30000 shots
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))

                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))

                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))
                    tasks.append((code_name, p, rd,  1000))

                else:

                    tasks.extend(
                        (code_name,p,rd,chunk_size)
                        for _ in range(num_shots // chunk_size) )       


    if decoder_option=='strong':
        results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value_strong)(code_name,strong_decoder,rd,p,shots) 
                                                  for code_name, p, rd,shots in tasks)      
    elif decoder_option=='weak':
        results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value_weak)(code_name,weak_decoder,rd,p,shots) 
                                                  for code_name, p, rd,shots in tasks)      
    


    #---- Collect total logical errors and total # of shots ----------------------

    total_errors  = {}
    total_shots   = {}
    
    
    for code_name,p,rd,result in results:
        total_errors[(code_name, p, rd)] = 0
        total_shots[(code_name,  p, rd)] = 0    
        

    for code_name,p,rd,result in results:

        total_errors[(code_name,p,rd)] += result["logical_errors"]
        total_shots[(code_name,p,rd)]  += results["shots"]
        

    #logical error rates
    ler_results = {(code_name,p,rd): total_errors[(code_name,p,rd)] / total_shots[(code_name,p,rd)]
                        for code_name in code_names
                        for p in ps
                        for rd in all_num_rounds
                        }
    
    #statistical uncertainty
    yerr_results = {(code_name,p,rd): np.sqrt(ler_results[(code_name,p,rd)]*(1-ler_results[(code_name,p,rd)])/total_shots[(code_name,p,rd)])
                        for code_name in code_names
                        for p in ps
                        for rd in all_num_rounds
                        
                        }      
    

    #---------- Now for each ler per code, rd, p get the epsilon parameters --------------------------------
    epsilon_fitted     = {(code_name,p): 0 for code_name in code_names for p in ps}
    epsilon_err_fitted = {(code_name,p): 0 for code_name in code_names for p in ps}

    pL_d     = {(code_name,p): 0 for code_name in code_names for p in ps}
    pL_d_err = {(code_name,p): 0 for code_name in code_names for p in ps}

    for code_name in code_names:

        _, _, d = map(int, code_name.strip("[]").split(","))

        for p in ps:

            pL_vals = {rd: ler_results[(code_name,p,rd)] for rd in all_num_rounds}
            pL_errs = {rd: yerr_results[(code_name,p,rd)] for rd in all_num_rounds}
            n_vals  = all_num_rounds

            eps_fit, eps_err = extract_fitted_epsilon(np.asarray(n_vals),np.asarray(list(pL_vals.values())),np.asarray(list(pL_errs.values())))

            epsilon_fitted[(code_name,p)] = eps_fit
            epsilon_err_fitted[(code_name,p)] = eps_err 


            #Using the fitted now get the final lers and their error
            pL_d[(code_name,p)]     =  (1-(1-eps_fit)**d)
            pL_d_err[(code_name,p)] = abs( d * (1 - eps_fit)**(d - 1) ) * eps_err


    
    #------------------------- Plot the results (eps_fitted, and pL@d) ----------------------    

    if decoder_option=='weak':
        decoder = weak_decoder 
    else:
        decoder = strong_decoder
    
    colors=["tab:blue","tab:orange","tab:green","tab:red", "tab:purple"]

    fig, ax = plt.subplots(2,1)
    cnt=0
    for code_name in code_names:

        y = {p: epsilon_fitted[(code_name,p)] for p in ps}
        yerr = {p: epsilon_err_fitted[(code_name,p)] for p in ps}

        ax[0].errorbar(ps,y.values(),yerr=yerr.values(),label=f"{code_name}, {decoder}",color=colors[cnt],marker='o')
        cnt+=1

    ax[0].set_xlabel("$p$")
    ax[0].set_ylabel("$\epsilon$")
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].legend(fontsize=12)


    cnt=0
    for code_name in code_names:
        y = {p: pL_d[(code_name,p)] for p in ps}
        yerr = {p: pL_d_err[(code_name,p)] for p in ps}

        ax[1].errorbar(ps,y.values(),yerr=yerr.values(),label=f"{code_name}, {decoder}",color=colors[cnt],marker='o')


        cnt+=1
        
    ax[1].set_xlabel("$p$")
    ax[1].set_ylabel("$p_L(d)$")
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].legend(fontsize=12)
    

    plt.tight_layout()
    plt.show()        


    return 
