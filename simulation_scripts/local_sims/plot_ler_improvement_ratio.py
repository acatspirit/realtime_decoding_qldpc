'''
Plot LER improvement as a function of the switch rate for each code
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) #move to level before sims file
import numpy as np
from numpy import array

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

#====================== To load results for switching from bplsd to strong decoders =================================

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
    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/relay_bp/single_sliding_window_relay_bp_max_shots_180000_lower_ps.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/relay_bp/single_sliding_window_relay_bp_max_shots_300000_p_2e_minus_3.txt'
    with open(file,"r") as f:
        data2 = eval(f.read())    

    data1 = merge_results_to_existing_dict(data1,data2)

    return data1


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

        num_shots = 400_000
        file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3.txt'
        
        with open(file1,"r") as f:
            data1 = eval(f.read())

        num_shots = 200_000
        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3.txt'


        with open(file2,"r") as f:
            data2 = eval(f.read())


        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)


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

        num_shots = 500_000
        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3_Gross_only.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)      


        num_shots = 300_000
        file2 = sys.path[-1] + f'/saved_data/decoder_switching_data/bplsd_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}_p_3e_minus_3_162_8_14.txt'
        
        with open(file2,"r") as f:
            data2 = eval(f.read())

        data1=merge_results_to_existing_dict(data1,data2,update_switch_times=True)      


    return data1


'''Use this to plot the LER improvement for bplsd'''
def plot_LER_improvement_bplsd(strong_decoder):

    target_rates = [1e-3, 5e-3,1e-2] #fill here other switch rates

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig,ax = plt.subplots(figsize=(5, 4))

    p = 3e-3 #pick a fixed physical error rate
    code_names = ["[[72,12,6]]","[[90,8,10]]","[[126,8,10]]","[[144,12,12]]","[[162,8,14]]"] #,"[[144,12,12]]"


    #================= Plot bplsd =======================================================

    data_bplsd  = get_bplsd_data_lower_ps()

    cnt=0

    for code_name in code_names:

        ler_improvement = []
        std_ler_improvement = []
        actual_switch_rates = []
        std_switch_rate     = []

        for switch_rate in target_rates:

            if strong_decoder=='relay_bp':
                data_switching = get_decoder_switching_data_bplsd_to_relaybp(switch_rate)
            elif strong_decoder=='tesseract':
                data_switching = get_decoder_switching_data_bplsd_to_tesseract(switch_rate)
            
        
            epsilon_weak     = data_bplsd['epsilons'][code_name][p]
            epsilon_err_weak = data_bplsd['std_epsilons'][code_name][p]

            #Get the same key of data_switching

            epsilon_switch     = data_switching['epsilons'][code_name][p]
            epsilon_err_switch = data_switching['std_epsilons'][code_name][p]

            delta_epsilon = epsilon_weak/epsilon_switch

            var_weak = epsilon_err_weak **2
            var_switch = epsilon_err_switch **2


            #This is for \epsilon_weak / \epsilon_switch
            std_dev = np.sqrt(
                var_weak / epsilon_switch**2
                +
                (epsilon_weak**2 / epsilon_switch**4) * var_switch
            )            
                      

            ler_improvement.append(delta_epsilon)
            std_ler_improvement.append(std_dev)


            switch_rate = data_switching['total_switch_times'][(code_name,p)] / data_switching['total_windows'][(code_name,p)]
            switch_err  = np.sqrt(switch_rate * (1 - switch_rate)/ data_switching['total_windows'][(code_name,p)])


            actual_switch_rates.append(switch_rate)
            std_switch_rate.append(switch_err)


        ax.errorbar(
            x=actual_switch_rates, y=ler_improvement, yerr=std_ler_improvement, xerr = std_switch_rate,
            label=f"{code_name}",
            marker='o',
            color=colors[cnt],
            markeredgecolor='k',
            markersize=7,
            linestyle='-.',
            
            alpha=0.9
        )
        cnt+=1


    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.set_ylabel(r"$\epsilon_{weak}/\epsilon_{switch}$")

    ax.set_xlabel("$p_s$")

    ax.set_yscale('log')
    ax.set_title(f"$p={p}$",fontsize=14)
    

    #========= Plot the switch frequency also ================

    # ax.set_ylim([1e-2,1e2])
    plt.tight_layout()
    plt.show()

    # fig.savefig(f"/Users/evatakou/Desktop/Figs_realtime/from_python/ler_improv_weak_bplsd_strong_{strong_decoder}.pdf",bbox_inches='tight')

    return 


# plot_LER_improvement_bplsd('relay_bp')


#=============  LER improvement for uf weak ======================

def get_uf_data():

    file = sys.path[-1] + f'/saved_data/single_decoder_sliding/uf/single_sliding_window_uf_max_shots_1000000_logspace.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    return data1

def get_decoder_switching_data_uf_to_strong(target_switch_rate=5e-3,strong_decoder='relay_bp'):

    if target_switch_rate==1e-2:
        num_shots      = 5_000_000
    elif target_switch_rate==1e-3:
        num_shots      = 2_000_000
    elif target_switch_rate==5e-3:
        num_shots      = 2_000_000

    weak_decoder   = 'uf'

    file1 = sys.path[-1] + f'/saved_data/decoder_switching_data/{weak_decoder}_to_{strong_decoder}/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}.txt'
        
    with open(file1,"r") as f:
        data1 = eval(f.read())

    return data1

def plot_LER_improvement_uf(strong_decoder):

    target_rates = [1e-3, 5e-3,1e-2] #fill here other switch rates

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig,ax = plt.subplots(figsize=(5, 4))

    
    code_names = ["[[72,12,6]]","[[90,8,10]]","[[126,8,10]]","[[144,12,12]]","[[162,8,14]]"] #,"[[144,12,12]]"

   
    data_uf  = get_uf_data()
    

    data_switching = get_decoder_switching_data_uf_to_strong(1e-3,strong_decoder)
    
    p              = list(data_uf['epsilons'][code_names[0]].keys())[1] #Get the error rate = 1.26 x 1e-4 

    cnt=0
    for code_name in code_names:

        ler_improvement = []
        std_ler_improvement = []
        actual_switch_rates = []
        std_switch_rate     = []

        for switch_rate in target_rates:

            data_switching = get_decoder_switching_data_uf_to_strong(switch_rate,strong_decoder)
            
        
            epsilon_weak     = data_uf['epsilons'][code_name][p]
            epsilon_err_weak = data_uf['std_epsilons'][code_name][p]

            #Get the same key of data_switching

            epsilon_switch     = data_switching['epsilons'][code_name][p]
            epsilon_err_switch = data_switching['std_epsilons'][code_name][p]

            delta_epsilon = epsilon_weak/epsilon_switch


            var_weak = epsilon_err_weak **2
            var_switch = epsilon_err_switch **2


            #This is for \epsilon_weak / \epsilon_switch
            std_dev = np.sqrt(
                var_weak / epsilon_switch**2
                +
                (epsilon_weak**2 / epsilon_switch**4) * var_switch
            )            

            ler_improvement.append(delta_epsilon)
            std_ler_improvement.append(std_dev)


            switch_rate = data_switching['total_switch_times'][(code_name,p)] / data_switching['total_windows'][(code_name,p)]
            switch_err  = np.sqrt(switch_rate * (1 - switch_rate)/ data_switching['total_windows'][(code_name,p)])


            actual_switch_rates.append(switch_rate)
            std_switch_rate.append(switch_err)


        ax.errorbar(
            x=actual_switch_rates, y=ler_improvement, yerr=std_ler_improvement, xerr = std_switch_rate,
            label=f"{code_name}",
            marker='o',
            color=colors[cnt],
            markeredgecolor='k',
            markersize=7,
            linestyle='-.',
            
            alpha=0.9
        )
        cnt+=1


    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.set_ylabel(r"$\epsilon_{weak}/\epsilon_{switch}$")

    ax.set_xlabel("$p_s$")

    ax.set_yscale('log')
    mantissa, exponent = f"{p:.2e}".split("e")
    ax.set_title(
        rf"$p={mantissa}\times 10^{{{int(exponent)}}}$",
        fontsize=14
    )    


    plt.tight_layout()
    plt.show()

    # fig.savefig(f"/Users/evatakou/Desktop/Figs_realtime/from_python/ler_improv_weak_bplsd_strong_{strong_decoder}.pdf",bbox_inches='tight')

    return 


# plot_LER_improvement_uf("relay_bp")


#========== LER improvement of switching over beamsearch ========


def get_beam_search_data_lower_ps(option):

    if option==1:
        file = sys.path[-1] + f'/saved_data/single_decoder_sliding/beamsearch/single_sliding_window_beam8_230iters_max_shots_50000_lower_ps.txt'
        with open(file,"r") as f:
            data1 = eval(f.read())    
    elif option==2:
        file = sys.path[-1] + f'/saved_data/single_decoder_sliding/beamsearch/single_sliding_window_beam32_340iters_max_shots_100000_lower_ps.txt'
        with open(file,"r") as f:
            data1 = eval(f.read())    

    return data1


def plot_LER_improvement_over_BeamSearch(strong_decoder,option):

    target_rates = [1e-3, 5e-3,1e-2] #fill here other switch rates

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig,ax = plt.subplots(figsize=(5, 4))

    p = 3e-3 #pick a fixed physical error rate
    code_names = ["[[72,12,6]]","[[90,8,10]]","[[126,8,10]]","[[144,12,12]]","[[162,8,14]]"] #,"[[144,12,12]]"


    #================= Plot bplsd =======================================================

    data_beamsearch  = get_beam_search_data_lower_ps(option=option)

    cnt=0

    for code_name in code_names:

        ler_improvement = []
        std_ler_improvement = []
        actual_switch_rates = []
        std_switch_rate     = []

        for switch_rate in target_rates:

            if strong_decoder=='relay_bp':
                data_switching = get_decoder_switching_data_bplsd_to_relaybp(switch_rate)
            elif strong_decoder=='tesseract':
                data_switching = get_decoder_switching_data_bplsd_to_tesseract(switch_rate)
            
        
            epsilon_weak     = data_beamsearch['epsilons'][code_name][p]
            epsilon_err_weak = data_beamsearch['std_epsilons'][code_name][p]

            
            #Get the same key of data_switching

            epsilon_switch     = data_switching['epsilons'][code_name][p]
            epsilon_err_switch = data_switching['std_epsilons'][code_name][p]


            delta_epsilon = epsilon_weak/epsilon_switch


            var_weak = epsilon_err_weak **2
            var_switch = epsilon_err_switch **2

            #This is for \epsilon_weak / \epsilon_switch
            std_dev = np.sqrt(
                var_weak / epsilon_switch**2
                +
                (epsilon_weak**2 / epsilon_switch**4) * var_switch
            )            
                      

            ler_improvement.append(delta_epsilon)
            std_ler_improvement.append(std_dev)


            switch_rate = data_switching['total_switch_times'][(code_name,p)] / data_switching['total_windows'][(code_name,p)]
            switch_err  = np.sqrt(switch_rate * (1 - switch_rate)/ data_switching['total_windows'][(code_name,p)])


            actual_switch_rates.append(switch_rate)
            std_switch_rate.append(switch_err)


        ax.errorbar(
            x=actual_switch_rates, y=ler_improvement, yerr=std_ler_improvement, xerr = std_switch_rate,
            label=f"{code_name}",
            marker='o',
            color=colors[cnt],
            markeredgecolor='k',
            markersize=7,
            linestyle='--',
            
            alpha=0.9
        )
        cnt+=1



    if option == 1:
        label = r'\mathrm{beam8\_230iters}'
    elif option == 2:
        label = r'\mathrm{beam32\_340iters}'

    ax.set_ylabel(rf"$\epsilon_{{{label}}}/\epsilon_{{\mathrm{{switch}}}}$")

    ax.set_xscale('log')
    ax.legend(fontsize=10)


    ax.set_xlabel("$p_s$")

    ax.set_yscale('log')
    ax.set_title(f"$p={p}$",fontsize=14)
    

    #========= Plot the switch frequency also ================

    # ax.set_ylim([1e-2,1e2])
    plt.tight_layout()
    plt.show()

    # fig.savefig(f"/Users/evatakou/Desktop/Figs_realtime/from_python/decoder_switch_vs_beam_option_{option}_strong_decoder_{strong_decoder}.pdf",bbox_inches='tight')

    return 


plot_LER_improvement_over_BeamSearch('tesseract',option=1)