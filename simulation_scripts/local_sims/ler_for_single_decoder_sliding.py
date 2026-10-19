import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) #move to level before src file


''' Get the ler performance for a single decoder w/ sliding window.'''


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"
from joblib import Parallel, delayed
from src.realtime_decoding.decoder_switching_class import decoder_switching_class





# '''no fit, just invert the expression to see if it stays constant'''
# def plot_ler_per_SEC_for_different_rds(num_shots=10_000,weak_decoder='bplsd',strong_decoder='relay_bp',decoder_option= 'weak',norm_order=2):
#     '''

#     Inputs:
#         num_shots: max number of shots to run the simulation (for p<8e-3, for p>=8e-3 we run fewer shots)
#         weak_decoder: 'bplsd' or 'uf'
#         strong_decoder: 'relay_bp' or 'tesseract'
#         decoder_option: 'weak' or 'strong' to pick the weak/strong decoder for sliding window
#         norm_order: order for calculating the cluster norm
        
#     '''

#     basis      = 'Z' #basis determining the memory experiment for the BB codes
#     code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]    #"[[144,12,12]]", "[[162,8,14]]"

#     ps         = [8e-3,  ]     

#     # all_num_rounds = [25,30,35,40]
#     # all_num_rounds = [20,30,40,50,60]
#     all_num_rounds = [30,40,50,]
    
#     max_shots_above_8e_minus3 = num_shots #this can be adjusted
    

#     def process_one_round_value(code_name,p,num_rounds,num_shots,norm_order):
        
#         print("Code_name,rds,p,shots:",(code_name,num_rounds,p,num_shots))

#         n, k, d = map(int, code_name.strip("[]").split(","))
        

#         nbuffer = d            #Buffer region
#         F       = d            #Commit region
#         W       = nbuffer + F  #Entire window

#         test  = decoder_switching_class(code_name=code_name,
#                                             num_rounds=num_rounds,
#                                             p=p,
#                                             basis=basis,
#                                             num_shots=num_shots,
#                                             W=W,
#                                             F=F,
#                                             strong_decoder_option=strong_decoder,
#                                             weak_decoder_option=weak_decoder)    
        
#         det_events = test.detection_events
#         obs_flips  = test.obs_flips
#         if decoder_option=='strong':
#             new_shots,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order)
#         else:
#             new_shots,_,logical_errors = test.decode_with_sliding_window(decoder_option=decoder_option,norm_order=norm_order) #suppress cluster norms output

#         result = {"logical_errors": np.sum(logical_errors)}

#         print("Sim done.")

#         return code_name,p,num_rounds,new_shots,result,det_events,obs_flips

#     tasks = []
#     import multiprocessing as mp
#     n_jobs = mp.cpu_count()    
#     chunk_size = max(500, num_shots // (100 * n_jobs)) 

    
#     for code_name in code_names:

#         for rd in all_num_rounds:
#             for p in ps:

#                 if p >= 8e-3:
                    
#                     for _ in range(max_shots_above_8e_minus3//500): #break into batches of 200
#                         tasks.append((code_name, p,rd, 500))

#                 else:

#                     tasks.extend(
#                         (code_name,p,rd,chunk_size)
#                         for _ in range(num_shots // chunk_size) )       


#     results = Parallel(n_jobs=-1,verbose=10,)(delayed(process_one_round_value)(code_name,p,rd,shots,norm_order) for code_name, p,rd,shots in tasks)      


#     total_errors   = {}
#     total_shots    = {}
#     all_det_events = {}
#     all_obs_flips  = {}
    

#     for code_name,p,rd,shot,result,det_events,obs_flips in results:
#         total_errors[(code_name, p, rd)] = 0
#         total_shots[(code_name,  p, rd)] = 0    
#         all_det_events[(code_name,p,rd)] = []
#         all_obs_flips[(code_name,p,rd)] = []
        

#     for code_name,p,rd,shot,result,det_events,obs_flips in results:
#         total_errors[(code_name,p,rd)] += result["logical_errors"]
#         total_shots[(code_name,p,rd)]  += shot
#         all_det_events[(code_name,p,rd)] = [all_det_events[(code_name,p,rd)],det_events]
#         all_obs_flips[(code_name,p,rd)]  = [all_obs_flips[(code_name,p,rd)],obs_flips]

        
        
#     ler_results = {(code_name,p,rd): total_errors[(code_name,p,rd)] / total_shots[(code_name,p,rd)]
#                         for code_name in code_names
#                         for p in ps
#                         for rd in all_num_rounds
#                         }
    
    
#     yerr_results = {(code_name,p, rd): np.sqrt(ler_results[(code_name,p,rd)]*(1-ler_results[(code_name,p,rd)])/total_shots[(code_name,p,rd)])
#                         for code_name in code_names
#                         for p in ps
#                         for rd in all_num_rounds
#                         }      
    

    

#     colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    

#     eps_to_save = {}
#     errs_in_eps_to_save = {}

#     if decoder_option =='weak':
#         decoder_label = weak_decoder 
#     else:
#         decoder_label = strong_decoder

#     # figures = []
#     # for code_name in code_names:

#     #     fig, ax = plt.subplots()
#     #     n, k, d = map(int, code_name.strip("[]").split(","))

#     #     print("code_name:",code_name)
#     #     cnt=0
#     #     for p in ps:
#     #         #for one p-value , get the epsilon for each rd and plot it
#     #         pL_vals = {rd: ler_results[(code_name,p,rd)] for rd in all_num_rounds}
#     #         pL_errs = {rd: yerr_results[(code_name,p,rd)] for rd in all_num_rounds}

#     #         eps = {rd: 1-(1-pL_vals[rd])**(1/rd) for rd in all_num_rounds}

#     #         eps_errs = { rd: (pL_errs[rd] / rd) * (1 - pL_vals[rd])**(1 / rd - 1)
#     #                     for rd in all_num_rounds }        
            

#     #         ax.errorbar(all_num_rounds,
#     #                     eps.values(),
#     #                     yerr=eps_errs.values(),
#     #                     label=f"{code_name}, {decoder_label}, p={p}",color=colors[cnt],marker='o',markeredgecolor='k')
            
#     #         print("p=",p)
#     #         print("eps:",eps)
#     #         print("eps_err:",eps_errs)
#     #         print("--------")            

#     #         cnt+=1


#     #     ax.set_xlabel("$r$")
#     #     ax.set_ylabel("LER per SEC")
#     #     ax.set_yscale('log')
#     #     # ax.set_xscale('log')
#     #     ax.legend(fontsize=12)

#     #     fig.tight_layout()


#     #     # plt.show() 
#     #     figures.append(fig)       

#     # for fig in figures:
#     #     plt.figure(fig.number)
#     #     plt.show()

#     fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
#     axes = axes.ravel()

#     for idx, code_name in enumerate(code_names):

#         ax = axes[idx]

#         n, k, d = map(int, code_name.strip("[]").split(","))

#         print("code_name:", code_name)

#         cnt = 0
#         for p in ps:

#             pL_vals = {rd: ler_results[(code_name, p, rd)] for rd in all_num_rounds}
#             pL_errs = {rd: yerr_results[(code_name, p, rd)] for rd in all_num_rounds}

#             eps = [1 - (1 - pL_vals[rd])**(1 / rd) for rd in all_num_rounds]

#             eps_errs = [
#                 (pL_errs[rd] / rd) * (1 - pL_vals[rd])**(1 / rd - 1)
#                 for rd in all_num_rounds
#             ]

#             ax.errorbar(
#                 all_num_rounds,
#                 eps,
#                 yerr=eps_errs,
#                 label=f"$p={p}$",
#                 color=colors[cnt],
#                 marker='o',
#                 markeredgecolor='k'
#             )

#             cnt += 1

#         ax.set_title(code_name)
#         ax.set_xlabel("$r$")
#         ax.set_ylabel("LER per SEC")
#         ax.set_yscale("log")
#         # ax.set_xscale("log")
#         ax.legend(fontsize=10)

#     # Hide unused subplot(s)
#     for ax in axes[len(code_names):]:
#         ax.set_visible(False)

#     fig.suptitle(f"{decoder_label} decoder", fontsize=16)
#     fig.tight_layout()

#     plt.show()    

#     return 

# plot_ler_per_SEC_for_different_rds(num_shots=num_shots,weak_decoder=weak_decoder,strong_decoder=strong_decoder,decoder_option= decoder_option)

def get_ler_per_SEC_eps_extracted_from_one_round(ps,num_shots=100_000,shots_per_job=10_000,p_leak=0,weak_decoder='bplsd',strong_decoder='relay_bp',decoder_option= 'weak',norm_order=2):
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

    num_rounds = 25
    max_shots_above_8e_minus3 = 15_000 #this can be adjusted

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

            if p >= 8e-3: 

                for _ in range(max_shots_above_8e_minus3//1_000): 
                    tasks.append((code_name, p, 1_000))

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
    
    

    
    # txt_to_save = sys.path[-1] + f'/saved_data/single_decoder_sliding/{decoder_label}/single_sliding_window_{decoder_label}_max_shots_{num_shots}_logspace.txt'
    txt_to_save = sys.path[-1] + f'/saved_data/single_decoder_sliding/{decoder_label}/single_sliding_window_{decoder_label}_max_shots_{num_shots}.txt'

    with open(txt_to_save, 'w') as file:
        file.write(str(dict_to_save))      

    # To load the data simply do:
    # with open(txt_to_load,"r") as f:
    #     data = eval(f.read())

    ax.set_xlabel("$p$")
    ax.set_ylabel("LER per SEC")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=12)

    plt.tight_layout()

    
    # figure_plot = sys.path[-1] + f'/saved_data/single_sliding_window_{decoder_label}_max_shots_{num_shots}.pdf'
    

    # fig.savefig(figure_plot,bbox_inches='tight')

    plt.show()        

    return 


num_shots     = 500_000
shots_per_job = 50_000
# ps            = np.logspace(-4,-3.5,6) #logspace for uf
ps            = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3]

get_ler_per_SEC_eps_extracted_from_one_round(ps,num_shots=num_shots,shots_per_job=shots_per_job,p_leak=0,
                                             weak_decoder='uf',
                                             strong_decoder='relay_bp',
                                             decoder_option= 'weak')




