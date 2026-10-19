'''
Plot uf->tesseract or relay-bp
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) #move to level before sims file
import numpy as np


from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

from numpy import array


#===== To load bplsd results ===============
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


def plot_decoder_switching_benchmarks(strong_decoder,plotted_quantity,target_switch_rate=1e-3):

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    fig,ax = plt.subplots(1,2,figsize=(7.5,5))

    #================= Plot uf =======================================================

    data_uf  = get_uf_data()
    data        = data_uf
    ps          = data['ps']
    
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():
            
            y    = np.array(list(val.values()))
            yerr = np.array(list(data["std_epsilons"][code_name].values())) #std_epsilons
            
    
            n, k, d = map(int, code_name.strip("[]").split(","))

            ax[0].errorbar(
                ps, y, yerr=yerr,
                marker='d',
                color=colors[cnt],
                markeredgecolor='k',
                markersize=7,
                linestyle='None',
                alpha=0.7
            )
            cnt+=1


    elif plotted_quantity=='pL@d':
        for cnt, (code_name, val) in enumerate(data["epsilons"].items()):

            y    = np.array(list(val.values()))
            yerr = np.array(list(data["std_epsilons"][code_name].values())) #std_epsilons


            n, k, d = map(int, code_name.strip("[]").split(","))

            yerr = abs(d * (1 - y)**(d - 1)) * yerr #std of pL @ d
            y    = 1 - (1 - y)**d                   #pL @ d

            ax[0].errorbar(
                ps, y, yerr=yerr,
                # label=f"{code_name}",
                marker='d',
                color=colors[cnt],
                
                alpha=0.5
            )


    #========== Plot decoder switching =================================
    data = get_decoder_switching_data_uf_to_strong(target_switch_rate,strong_decoder)


    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():

            y    = np.array(list(val.values()))
            yerr = np.array(list(data["std_epsilons"][code_name].values())) 


            n, k, d = map(int, code_name.strip("[]").split(","))

            ax[0].errorbar(
                ps, y, yerr=yerr,
                label=f"{code_name}",
                marker='o',
                color=colors[cnt],
                markeredgecolor='k',
                
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
                color=colors[cnt],
                
            )
    


    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].legend(fontsize=10)
    ax[0].set_ylabel("LER per SEC")

    ax[0].set_xlabel(r"physical error rate")
    ax[0].set_ylim([3e-8,1e-3])
    ax[0].set_title(f"target $p_s=$ {target_switch_rate}",fontsize=15)
    

    #========= Plot the switch frequency also ================


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

        ps_code = [
            p for (code, p) in switch_rates.keys()
            if code == code_name
        ]            
        y = np.array([ switch_rates[(code_name, p)] for p in ps_code ])
        yerr = np.array([ switch_yerr[(code_name, p)] for p in ps_code ])

        
        ax[1].errorbar(ps,y,yerr=yerr, marker='o',color=colors[cnt],label=f"{code_name}")   
        cnt+=1 


    ax[1].set_ylabel("$p_s$")
    ax[1].set_xscale('log')


    ax[1].set_xlabel(r"physical error rate")

    ax[1].set_title(f"target $p_s=$ {target_switch_rate}",fontsize=15)
    ax[1].set_yscale('log')
    
    ax[1].legend(fontsize=9)    


    plt.tight_layout()
    plt.show()

    # fig.savefig(f"/Users/evatakou/Desktop/Figs_realtime/from_python/decoder_switching_benchmarks_ps_{target_switch_rate}_strong_{strong_decoder}.pdf",bbox_inches='tight')

    return 


plot_decoder_switching_benchmarks(strong_decoder='tesseract',plotted_quantity='epsilons',target_switch_rate=1e-2)






