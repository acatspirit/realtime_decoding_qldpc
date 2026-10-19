import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #move to level before sims file
import numpy as np


from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Microsoft Sans Serif"


def get_relay_bp_data_w_leakage():
    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/relay_bp/single_sliding_window_relay_bp_max_shots_20000_p_3e_minus_3_p_leak_1e-05.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    


    return data1

def get_bplsd_data_w_leakage():
    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/bplsd/single_sliding_window_bplsd_max_shots_20000_p_3e_minus_3_p_leak_1e-05.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    


    return data1


def merge_results_to_existing_dict(data1,data2, update_switch_times=False):
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

    if update_switch_times:

        for key in data2['total_switch_times'].keys():

            data1['total_switch_times'][key] += data2['total_switch_times'][key]
            data1['total_windows'][key]      += data2['total_windows'][key]

        #now update the switch_rates:
        for key in data2['switch_rates'].keys():
            #(code_name,p) is the key
            data1['switch_rates'][key] = data1['total_switch_times'][key] / data1['total_windows'][key]

        for key in data2['switch_rate_err'].keys():

            data1['switch_rate_err'][key] = np.sqrt(

                data1['switch_rates'][key] * (1-data1['switch_rates'][key])/data1['total_windows'][key]

            )        


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



def get_bplsd_data_lower_ps():

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/bplsd/single_sliding_window_bplsd_max_shots_120000_lower_ps.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/bplsd/single_sliding_window_bplsd_max_shots_200000_p_2e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    
    
    data1 = merge_results_to_existing_dict(data1,data2)        

    return data1


'''
darker color -> strong
hatch -> no leakage
'''
def plot_relay_bp_and_bplsd_w_and_wo_leakage():

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig,ax = plt.subplots(figsize=(5,4))


    #========= Plot strong decoder ===============================================
    ps = [3e-3]


    data_relay_bp  = get_relay_bp_data_w_leakage()
    data           = data_relay_bp


    cnt=0

    for code_name,val in data['epsilons'].items():

        y    = np.array([val[p] for p in ps]) #epsilons
        yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

        n, k, d = map(int, code_name.strip("[]").split(","))


        ax.bar(
            cnt, y, yerr=yerr,
            label=f"relay-bp (w/ leak.)",
            color=colors[cnt],
            edgecolor='k',
            # markeredgecolor='k',
            # marker='s',
            # markersize=7,
            # linestyle='-.',
            linestyle='-',
            width=0.4,
            ecolor='k',
            capsize=4,
                  
            # alpha=0.5
        )
        cnt+=1


    #================= Plot bplsd =======================================================

    data_bplsd  = get_bplsd_data_w_leakage()
    data        = data_bplsd
    
    
    cnt=0

    for code_name,val in data['epsilons'].items():

        y    = np.array([val[p] for p in ps]) #epsilons
        yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

        n, k, d = map(int, code_name.strip("[]").split(","))

        ax.bar(
            cnt+0.4, y, yerr=yerr,
            label=f"bplsd (w/ leak.)",
            # marker='d',
            color=colors[cnt],
            edgecolor='k',
            # markeredgecolor='k',
            # markersize=7,
            linestyle='-',
            alpha=0.7,
            ecolor='k',
            width=0.4,
            capsize=4,
            # capsize=2
        )
        cnt+=1


    #-------- Now plot/wo leakage too for bplsd --------------


    data_bplsd  = get_bplsd_data_lower_ps()
    data        = data_bplsd
    
    cnt=0

    for code_name,val in data['epsilons'].items():

        y    = np.array([val[p] for p in ps]) #epsilons
        yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

        n, k, d = map(int, code_name.strip("[]").split(","))

        ax.bar(
            cnt+0.4, y, yerr=yerr,
            label=f"bplsd (no leak.)",
            # marker='d',
            color=colors[cnt],
            edgecolor='k',
            # markeredgecolor='k',
            # markersize=7,
            linestyle='-',
            hatch='/',
            alpha=0.7,
            ecolor='k',
            width=0.2,
            capsize=4,

            # capsize=2
        )
        cnt+=1


    #-------- Now plot/wo leakage too for relay --------------

    data_bplsd  = get_relay_bp_data_lower_ps()
    data        = data_bplsd
    
    cnt=0

    for code_name,val in data['epsilons'].items():

        y    = np.array([val[p] for p in ps]) #epsilons
        yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

        n, k, d = map(int, code_name.strip("[]").split(","))

        ax.bar(
            cnt, y, yerr=yerr,
            label=f"relay (no leak.)",
            # marker='d',
            color=colors[cnt],
            edgecolor='k',
            # markeredgecolor='k',
            # markersize=7,
            linestyle='-',
            hatch='/',
            # alpha=0.7,
            ecolor='k',
            width=0.25,
            capsize=4,

            # capsize=2
        )
        cnt+=1


    code_names = [
        '[[72,12,6]]',
        '[[90,8,10]]',
        '[[126,8,10]]',
        '[[144,12,12]]',
        '[[162,8,14]]'
    ]
    ax.set_xticks(np.arange(5), code_names,rotation=45,fontsize=13)

    

    ax.set_yscale('log')
    # ax.legend(fontsize=10)
    ax.set_ylabel("LER per SEC")
    # ax[0].set_xlabel(r"BB code")
    ax.set_title(f"$p$={ps[0]}",fontsize=15)



    plt.tight_layout()
    plt.show()

    # fig.savefig(f"/Users/evatakou/Desktop/Figs_realtime/from_python/w_leakage_bplsd_and_relay_bp.pdf",bbox_inches='tight')

    return 


# plot_relay_bp_and_bplsd_w_and_wo_leakage()

#======== Plot decoder switching results =



def merge_results_to_existing_dict(data1,data2, update_switch_times=False):
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

    if update_switch_times:

        for key in data2['total_switch_times'].keys():

            data1['total_switch_times'][key] += data2['total_switch_times'][key]
            data1['total_windows'][key]      += data2['total_windows'][key]

        #now update the switch_rates:
        for key in data2['switch_rates'].keys():
            #(code_name,p) is the key
            data1['switch_rates'][key] = data1['total_switch_times'][key] / data1['total_windows'][key]

        for key in data2['switch_rate_err'].keys():

            data1['switch_rate_err'][key] = np.sqrt(

                data1['switch_rates'][key] * (1-data1['switch_rates'][key])/data1['total_windows'][key]

            )        


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

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/tesseract/single_sliding_window_tesseract_max_shots_2000000_p_2e_minus_3_Gross_only.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)    

    return data1



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


def get_decoder_switching_data_bplsd_to_relaybp(target_switch_rate=5e-3):
    strong_decoder = 'relay_bp'
    weak_decoder   = 'bplsd'
    if target_switch_rate==5e-3:
    

        num_shots = 200_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())


        num_shots = 300_000
        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_2e_minus_3.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())


        data1=merge_results_to_existing_dict(data1,data2) #combine 2e-3 results into existing data

        num_shots = 400_000
        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_2e_minus_3_Gross_only.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2)



    elif target_switch_rate==1e-3:

        num_shots = 200_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())


        num_shots = 300_000
        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_2e_minus_3.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())


        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)

    elif target_switch_rate==1e-2:

        num_shots = 200_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())

        num_shots = 400_000
        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())



    return data1

def get_decoder_switching_data_bplsd_to_relaybp_w_leakage(target_switch_rate=5e-3):
    strong_decoder = 'relay_bp'
    weak_decoder   = 'bplsd'

    if target_switch_rate==1e-2:
    

        num_shots = 100_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/leakage_bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3_p_leak_1e-05.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())




    return data1


def get_decoder_switching_data_bplsd_to_tesseract(target_switch_rate=5e-3):

    strong_decoder = 'tesseract'
    weak_decoder   = 'bplsd'

    if target_switch_rate==5e-3:
    
        
        num_shots = 200_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())

        num_shots = 500_000

        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)

        num_shots = 2_000_000

        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_2e_minus_3_Gross_only.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)        


    elif target_switch_rate==1e-3:

        num_shots = 200_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())


        num_shots = 500_000


        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)

        num_shots = 800_000


        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_2e_minus_3.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)

        num_shots = 2_000_000


        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_2e_minus_3.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)        

    elif target_switch_rate==1e-2:

        num_shots = 200_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())

        num_shots = 300_000


        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3_162_8_14.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)        


        num_shots = 500_000

        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3_Gross_only.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)        


    return data1

def get_decoder_switching_data_bplsd_to_tesseract_w_leakage(target_switch_rate=1e-2):

    strong_decoder = 'tesseract'
    weak_decoder   = 'bplsd'

    if target_switch_rate==1e-2:
    
        
        num_shots = 100_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/leakage_bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3_p_leak_1e-05.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())


    return data1


def plot_decoder_switching_benchmarks(strong_decoder,plotted_quantity,target_switch_rate=1e-2):

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig,ax = plt.subplots(1,2,figsize=(7.5,4))


    #========= Plot strong decoder ===============================================
    ps = [3e-3]

    if strong_decoder=='relay_bp':

        data_relay_bp  = get_relay_bp_data_lower_ps()
        data           = data_relay_bp
        # ps             = data['ps']

    elif strong_decoder=='tesseract':
        data_tesseract  = get_tesseract_data_lower_ps()
        data            = data_tesseract
        
        # ps              = data['ps']


    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))


            ax[0].bar(
                cnt, y, yerr=yerr,
                # label=f"tess. ($p_{{leak}}=0$)",
                color=colors[cnt],
                edgecolor='k',
                # markeredgecolor='k',
                # marker='s',
                # markersize=7,
                # linestyle='-.',
                linestyle='-',
                width=0.5,
                ecolor='k',
                capsize=4                
                # alpha=0.5
            )
            cnt+=1


    


    #================= Plot bplsd =======================================================

    data_bplsd  = get_bplsd_data_lower_ps()
    data        = data_bplsd
    # ps          = data['ps']
    
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))


            ax[0].bar(
                cnt, y, yerr=yerr,
                # label=f"{code_name}, bplsd",
                # marker='d',
                color=colors[cnt],
                edgecolor='k',
                # markeredgecolor='k',
                # markersize=7,
                linestyle='-',
                alpha=0.7,
                ecolor='k',
                # capsize=2
            )
            cnt+=1

    # ax[0].set_yscale('log')
    # plt.show()
    # return


    #========== Plot decoder switching (no leakage) =================================

    if strong_decoder=='relay_bp':
        data = get_decoder_switching_data_bplsd_to_relaybp(target_switch_rate)
    elif strong_decoder=='tesseract':
        data = get_decoder_switching_data_bplsd_to_tesseract(target_switch_rate)

    print("ps:",ps)
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))


            ax[0].errorbar(
                cnt+0.25, y, yerr=yerr,
                # label=f"$p_{{leak}}$",
                marker='o',
                markersize=7,
                color=colors[cnt],
                markeredgecolor='k',
                elinewidth=1.5,
                capsize=5,
                capthick=2,
                ecolor='black',
                
            )
            cnt+=1


    switch_rates = {
        key: data['total_switch_times'][key] / data['total_windows'][key]
        for key in data['total_switch_times'].keys()
    }    

    switch_yerr = {
        key: np.sqrt(
            switch_rates[key] * (1 - switch_rates[key]) / data['total_windows'][key]
        )
        for key in switch_rates
    }    

    cnt=0
    for code_name in data['codes']:
        
        y = np.array([ switch_rates[(code_name, p)] for p in data['ps'] ])

        yerr = np.array([ switch_yerr[(code_name, p)] for p in data['ps'] ])

        
        ax[1].bar(cnt,y,
                  yerr=yerr, 
                #   marker='d',
                  color=colors[cnt],
                #   label=f"{code_name}, $p_{{leak}}=0$",
                  width=0.5,
                  edgecolor='k')   
        cnt+=1 


    #========== Plot decoder switching (w/ leakage) =================================

    if strong_decoder=='relay_bp':
        data = get_decoder_switching_data_bplsd_to_relaybp_w_leakage(target_switch_rate)
    elif strong_decoder=='tesseract':
        data = get_decoder_switching_data_bplsd_to_tesseract_w_leakage(target_switch_rate)
    
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array([val[p] for p in ps]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in ps]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))


            ax[0].errorbar(
                cnt+0.25, y, yerr=yerr,
                # label=f"${code_name}, p_{{leak}}=10^{-5}$",
                marker='d',
                markersize=7,
                color=colors[cnt],
                markeredgecolor='k',
                elinewidth=1.5,
                capsize=5,
                capthick=2,
                ecolor='black',
                
            )
            cnt+=1


    code_names = [
        '[[72,12,6]]',
        '[[90,8,10]]',
        '[[126,8,10]]',
        '[[144,12,12]]',
        '[[162,8,14]]'
    ]
    ax[0].set_xticks(np.arange(5), code_names,rotation=45,fontsize=13)
    ax[1].set_xticks(np.arange(5), code_names,rotation=45,fontsize=13)

    ax[0].set_yscale('log')
    # ax[0].legend(fontsize=10)
    ax[0].set_ylabel("LER per SEC")
    # ax[0].set_xlabel(r"BB code")
    ax[0].set_title(f"$p$={ps[0]}",fontsize=15)

    #========= Plot the switch frequency also ================

    print(data['total_windows'].keys()) #total_switch_times, total_windows



    switch_rates = {
        key: data['total_switch_times'][key] / data['total_windows'][key]
        for key in data['total_switch_times'].keys()
    }    


    switch_yerr = {
        key: np.sqrt(
            switch_rates[key] * (1 - switch_rates[key]) / data['total_windows'][key]
        )
        for key in switch_rates
    }    

    cnt=0
    for code_name in data['codes']:
        
        y = np.array([ switch_rates[(code_name, p)] for p in data['ps'] ])

        yerr = np.array([ switch_yerr[(code_name, p)] for p in data['ps'] ])

        
        ax[1].bar(cnt,y,
                  yerr=yerr, 
                #   marker='o',
                  color=colors[cnt],
                #   label=rf"{code_name}, $p_{{leak}}=10^{{-5}}$",
                  alpha=0.7,
                  edgecolor='k')   
        cnt+=1 


    ax[1].set_ylabel("$p_s$")
    # ax[1].set_xscale('log')
    # ax[1].set_xticks([2e-3, 3e-3, 4e-3, 5e-3])
    # ax[1].set_xticklabels([
    # r"$2$",
    # r"$3$",
    # r"$4$",
    # r"$5$",
    # ])
    # ax[1].minorticks_off() 

    # ax[1].set_xlabel(r"physical error rate ($\times 10^{-3}$)")

    ax[1].set_title(f"target $p_s=$ {target_switch_rate}",fontsize=15)
    ax[1].set_yscale('log')
    
    # ax[1].legend(fontsize=9,ncols=2)    




    plt.tight_layout()
    plt.show()

    # fig.savefig(f"/Users/evatakou/Desktop/Figs_realtime/from_python/decoder_switching_benchmarks_ps_{target_switch_rate}_strong_{strong_decoder}_leakage.pdf",bbox_inches='tight')

    return 



plot_decoder_switching_benchmarks(strong_decoder='relay_bp',plotted_quantity='epsilons',target_switch_rate=1e-2)