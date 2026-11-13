import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) #move to level before sims file
import numpy as np


from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

def merge_results_to_existing_dict(data1,data2):
    '''
    If we have 2 dictionaries where 1 has same dictionaries of physical error rates, we combine ler results into existing dictionary (data1)
    '''
    #update the total errors, shots, pL@r and std_pL@r
    error_dict=data2['total_errors']

    for key,val in error_dict.items():

        for key2 in data1['total_errors'].keys():
            if key==key2:
                data1['total_errors'][key2] += val 
                
    #update the # of shots
    shot_dict = data2['shots']
    for key,val in shot_dict.items():
        
        for key2 in data1['shots'].keys():
            if key==key2:        
                data1['shots'][key2]+=val 
                

    r2 = data2['r'] #pL @ r = num_errors/num_shots
    r1 = data1['r']
    if r1!=r2:
        raise ValueError("Cannot combine.")

    #update the pL@r  = num_errors/num_shots 
    for key,val in data1['pL@r'].items():

        data1['pL@r'][key] = data1['total_errors'][key]/data1['shots'][key]
    
    #update the std for pL@r
    for key in data1['std_pL@r'].keys():

        y = data1['pL@r'][key]
        data1['std_pL@r'][key] = np.sqrt(y*(1-y)/data1['shots'][key])

    
    #update the epsilons
    for key in data1['epsilons'].keys(): #code_name is key

        temp_dict_epsilons = data1['epsilons'][key].copy() #key is p, value is epsilon 
        temp_dict_std_epsilons = data1['std_epsilons'][key].copy()

        for key2,val in data1['pL@r'].items(): #key is (code_name,p)

            if key!=key2[0]:
                continue

            phys_rate   = key2[1]
            pL          = val 
            pL_err      = data1['std_pL@r'][key2]

            epsilon     = 1-(1-pL)**(1/r1)
            epsilon_err = ( pL_err / r1 ) * (1-pL)**(1/r1 -1)

            temp_dict_epsilons[phys_rate] = epsilon 
            temp_dict_std_epsilons[phys_rate] = epsilon_err 

        data1['epsilons'][key] = temp_dict_epsilons 
        data1['std_epsilons'][key] = temp_dict_std_epsilons


    return data1

def merge_low_and_high_ps(data_low_ps,data_high_ps):
    '''Merge 2 dictionaries for which we have different physical error rates where we evaluate epsilons and pL@d.
    '''

    #Merge the epsilons and std_epsilons. Using those we can then evaluate pL@d if we want
    
    data = {
        "ps": sorted(set(data_low_ps["ps"]) | set(data_high_ps["ps"])),
        "epsilons": {},
        "std_epsilons": {},
    }

    for code in data_low_ps["epsilons"]:
        data["epsilons"][code] = {
            **data_low_ps["epsilons"][code],
            **data_high_ps["epsilons"][code],   # values from data2 overwrite duplicates
        }

        data["std_epsilons"][code] = {
            **data_low_ps["std_epsilons"][code],
            **data_high_ps["std_epsilons"][code],
        }    


    return data


#===== To load relay_bp results ==============
def get_relay_bp_data_lower_ps():
    '''Use this for relay_bp to add the p=2e-3 case'''
    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/relay_bp/single_sliding_window_relay_bp_max_shots_180000_lower_ps.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/relay_bp/single_sliding_window_relay_bp_max_shots_300000_p_2e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)



    return data1

def get_relay_bp_data_higher_ps():

    '''load the results of relay_bp single-sliding window for ps \in [6e-3,1e-2]'''

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/relay_bp/single_sliding_window_relay_bp_max_shots_20000.txt'
    with open(file,"r") as f:
        data = eval(f.read())    


    return data

#===== To load tesseract results ===============
def get_tesseract_data_lower_ps():

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_100000_lower_ps.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_300000_p_3e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_300000_p_2e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_500000_p_3e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_500000_p_2e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_2000000_p_2e_minus_3_Gross_only.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)    

    return data1

def get_tesseract_data_higher_ps():

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_30000.txt'
    with open(file,"r") as f:
        data = eval(f.read())    

    return data

#===== To load bplsd results ===============
def get_bplsd_data_lower_ps():

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/bplsd/single_sliding_window_bplsd_max_shots_120000_lower_ps.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/bplsd/single_sliding_window_bplsd_max_shots_200000_p_2e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    
    
    data1 = merge_results_to_existing_dict(data1,data2)        

    return data1

def get_bplsd_data_higher_ps():

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/bplsd/single_sliding_window_bplsd_max_shots_30000.txt'
    with open(file,"r") as f:
        data = eval(f.read())    

    return data

#===== To load union-find results ===============
def get_uf_data():

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/uf/single_sliding_window_uf_max_shots_500000.txt'
    with open(file,"r") as f:
        data = eval(f.read())    


    return data


#========== Plot the sliding window performance ===========
def plot_single_sliding_benchmarks(plotted_quantity):

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig,ax = plt.subplots(1,4,
                          figsize=(20, 4))

    #================== Plot relay_bp  ======================================
    
    data_relay_bp_lower_ps  = get_relay_bp_data_lower_ps()
    data_relay_bp_higher_ps = get_relay_bp_data_higher_ps()
    data_relay_bp           = merge_low_and_high_ps(data_relay_bp_lower_ps,data_relay_bp_higher_ps)
    data                    = data_relay_bp
    ps                      = data['ps']

    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))

            ax[0].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
               
                color=colors[cnt],
                markeredgecolor='k'
            )
            cnt+=1



    elif plotted_quantity=='pL@d':
        for cnt, (code_name, val) in enumerate(data["epsilons"].items()):

            # sort by p 
            ps   = np.array(sorted(val.keys()))
            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))

            yerr = abs(d * (1 - y)**(d - 1)) * yerr #std of pL @ d
            y    = 1 - (1 - y)**d                   #pL @ d

            ax[0].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
                color=colors[cnt]
            )

    #================ Plot tesseract =====================================================

    data_tesseract_lower_ps  = get_tesseract_data_lower_ps()
    data_tesseract_higher_ps = get_tesseract_data_higher_ps()
    data_tesseract          = merge_low_and_high_ps(data_tesseract_lower_ps,data_tesseract_higher_ps)
    data                    = data_tesseract
    ps                      = data['ps']

    
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))



            ax[1].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',       
                color=colors[cnt],
                markeredgecolor='k'
            )
            cnt+=1

   



    elif plotted_quantity=='pL@d':
        for cnt, (code_name, val) in enumerate(data["epsilons"].items()):

            # sort by p 
            ps   = np.array(sorted(val.keys()))
            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))

            yerr = abs(d * (1 - y)**(d - 1)) * yerr #std of pL @ d
            y    = 1 - (1 - y)**d                   #pL @ d

            ax[1].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
                color=colors[cnt]
            )

    #================= Plot bplsd =======================================================

    data_bplsd_lower_ps  = get_bplsd_data_lower_ps()
    data_bplsd_higher_ps = get_bplsd_data_higher_ps()
    data_bplsd           = merge_low_and_high_ps(data_bplsd_lower_ps,data_bplsd_higher_ps)
    data                 = data_bplsd
    ps                   = data['ps']
    
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))

            ax[2].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
                color=colors[cnt],
                markeredgecolor='k'
            )
            cnt+=1


    elif plotted_quantity=='pL@d':
        for cnt, (code_name, val) in enumerate(data["epsilons"].items()):

            # sort by p 
            ps   = np.array(sorted(val.keys()))
            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))

            yerr = abs(d * (1 - y)**(d - 1)) * yerr #std of pL @ d
            y    = 1 - (1 - y)**d                   #pL @ d

            ax[2].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
                color=colors[cnt]
            )

    #======================= Plot uf ================================================================

    data_uf  = get_uf_data()
    data     = data_uf
    ps       = data['ps']
    
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))



            ax[3].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
                color=colors[cnt],
                markeredgecolor='k'
            )
            cnt+=1


    elif plotted_quantity=='pL@d':
        for cnt, (code_name, val) in enumerate(data["epsilons"].items()):

            # sort by p 
            ps   = np.array(sorted(val.keys()))
            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))

            yerr = abs(d * (1 - y)**(d - 1)) * yerr #std of pL @ d
            y    = 1 - (1 - y)**d                   #pL @ d

            ax[3].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
                color=colors[cnt]
            )



    decoders = ['Relay-BP','Tesseract','BP-LSD','UF']
    for k in range(4):


        ax[k].set_yscale('log')
        ax[k].set_xscale('log')
        ax[k].legend(fontsize=10)
        # ax[k].set_ylim([1e-9,5e-2])
        ax[k].set_ylim([1e-6,1e-3])
        
        if k==0:
            ax[k].set_ylabel("LER per SEC")

        ax[k].set_title(f"{decoders[k]}",fontsize=15)
        
        if k!=3:
            ax[k].set_xticks([2e-3, 4e-3, 6e-3, 8e-3, 1e-2])
            ax[k].set_xticklabels([
            r"$2$",
            r"$4$",
            r"$6$",
            r"8",
            r"10",
            ])
            # ax[k].minorticks_off() 

            ax[k].set_xlabel(r"physical error rate ($\times 10^{-3}$)")
        else:
            ax[k].set_xticks([1e-4, 3e-4, 5e-4, 7e-4, 9e-4, 2e-3, 3e-3])
            ax[k].set_xticklabels([
            r"$1$",
            r"$3$",
            r"$5$",
            r"$7$",
            r"$9$",
            r"$20$",
            r"$30$",
            ])            
            ax[k].set_xlabel(r"physical error rate ($\times 10^{-4}$)")
            ax[k].minorticks_off() 

    plt.tight_layout()
    plt.show()

    # fig.savefig("/Users/evatakou/Desktop/Figs_realtime/from_python/single_sliding_benchmarks.pdf",bbox_inches='tight')

    return 



plot_single_sliding_benchmarks('epsilons')