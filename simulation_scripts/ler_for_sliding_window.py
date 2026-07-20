import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move to level before src file


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

        print("Sim done.")

        return code_name,p,new_shots,result, np.sum(switch_times_per_shot)

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



num_shots      = 100_000
weak_decoder   = 'uf'
strong_decoder = 'tesseract'
decoder_option = 'strong'
cutoff=0.8
get_ler_per_SEC_eps_extracted_from_one_round_switching(num_shots=num_shots,
                                             weak_decoder=weak_decoder,
                                             strong_decoder=strong_decoder,
                                             decoder_option= decoder_option,
                                             cutoff=cutoff,
                                             norm_order=2)


# txt_to_load = sys.path[-1] + f'/saved_data/single_sliding_window_{strong_decoder}_max_shots_{num_shots}.txt'
# with open(txt_to_load,"r") as f:
#     data = eval(f.read())




  



