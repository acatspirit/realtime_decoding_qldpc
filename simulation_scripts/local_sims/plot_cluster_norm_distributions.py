import sys
import os
import gzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #move to level before sims file


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Microsoft Sans Serif"
from joblib import Parallel, delayed
from src.realtime_decoding.decoder_switching_class import decoder_switching_class
import pickle 
import zipfile

def plot_switch_rates_vs_cutoff(target_switch_rate, weak_decoder,path, noise_type='Pauli'):
    '''
    Plot the switch rate distributions as a function of cluster norm cutoffs.

    Input:
        target_switch_rate: value of switch rate to draw a horizontal line on the plot
        weak_decoder: 'bplsd' or 'uf'
        path: path to save the plot
        cluster_type: 'Pauli', 'erasures' to select Pauli only, or Pauli + erasures data (applicable only for UF)

    '''

    code_names   = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"] 

    if weak_decoder=='bplsd':
        num_shots    = 100_000
        if noise_type=='erasures':
            raise Exception("For BPLD we have only cluster norms sampled under Pauli noise. Select uf instead.")
    elif weak_decoder=='uf':
        if noise_type=='Pauli':
            num_shots  = 500_000
        else:
            num_shots = 50_000



    cutoffs_to_set = {}

    
    fig,ax = plt.subplots(1,len(code_names),sharey=True,figsize=(12,4))

    data_per_code = []
    cnt=0
    for code_name in code_names:

        if noise_type=='Pauli':
            txt_to_load = sys.path[-1] + f'/saved_data/cluster_norm_statistics_final/{weak_decoder}/cluster_norm_distributions_code_{code_name}_{weak_decoder}_max_shots_{num_shots}.txt'
            with open(txt_to_load, "rb") as file:
                data = pickle.load(file)
        else:
            txt_to_load = sys.path[-1] + f'/saved_data/cluster_norm_statistics_final/uf_compressed/cluster_norm_distributions_code_{code_name}_{weak_decoder}_max_shots_{num_shots}_w_erasures.txt.zip'
            with zipfile.ZipFile(txt_to_load, "r") as z:
                
                with z.open(z.namelist()[0], "r") as file:
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

            
            p_str = f'{ps[k]:.1e}'
            mantissa, exponent = p_str.split('e')
            exponent = int(exponent)

            ax[cnt].semilogx(cutoffs, switch_rates[k], marker='.',label=rf'$p={mantissa}\times 10^{{{exponent}}}$')
            ax[cnt].axhline(target_switch_rate,color='k',linestyle='--')
            
            # ax[cnt].set_xlabel("cutoff")
            if cnt==0:
                ax[cnt].set_ylabel("switch rate")
            ax[cnt].grid()
            ax[cnt].set_yscale('log')
            ax[cnt].set_xscale('log')
            ax[cnt].set_title(code_name)
            
            

        cnt+=1


    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,fontsize=10,bbox_to_anchor=(0.5, 1.07),)

    fig.supxlabel("cutoff", fontsize=10)
    
    plt.tight_layout()
    plt.show()

    fig.savefig(f"{path}/switch_rates_vs_cutoff_{weak_decoder}.pdf",bbox_inches='tight')


    return 

path = '/Users/evatakou/Desktop'
# plot_switch_rates_vs_cutoff(1e-2,weak_decoder='uf',path=path,noise_type='erasures')


def plot_cluster_norm_dist_and_switch_rate_vs_cutoff_and_p(path,code_name = "[[162,8,14]]", weak_decoder='bplsd',noise_type='Pauli'):
    '''
    Plot the cluster norm distributions for a particular code.

    Input:
        path: the path to save the plot
        code_name: one of "[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"
        weak_decoder: 'bplsd' or 'uf'
    
    '''
    num_rounds = 25

    if weak_decoder=='bplsd':
        num_shots    = 100_000
        if noise_type=='erasures':
            raise Exception("For BPLD we have only cluster norms sampled under Pauli noise. Select uf instead.")

        decoder_label = 'BP-LSD'
    elif weak_decoder=='uf':
        decoder_label = 'UF'
        if noise_type=='Pauli':
            num_shots  = 500_000
        else:
            num_shots = 50_000

    if noise_type=='Pauli':

        txt_to_load = sys.path[-1] + f'/saved_data/cluster_norm_statistics_final/{weak_decoder}/cluster_norm_distributions_code_{code_name}_{weak_decoder}_max_shots_{num_shots}.txt'

        with open(txt_to_load, "rb") as file:
            data_all = pickle.load(file)

    else:
        txt_to_load = sys.path[-1] + f'/saved_data/cluster_norm_statistics_final/uf_compressed/cluster_norm_distributions_code_{code_name}_{weak_decoder}_max_shots_{num_shots}_w_erasures.txt.zip'
        with zipfile.ZipFile(txt_to_load, "r") as z:
            
            with z.open(z.namelist()[0], "r") as file:
                data_all = pickle.load(file)        



    ps = data_all['ps']

    switch_rates   = data_all['switch_rates']
    cutoffs = data_all['cutoffs']
    
    fig, ax = plt.subplots(2,1)

    colors=["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    cnt=0

    for p in ps:
        
        data     = data_all['cluster_norms'][(code_name,p)].flatten()
        log_data = np.log10(data[data>0])

        p_str = f'{p:.1e}'
        mantissa, exponent = p_str.split('e')
        exponent = int(exponent)

        ax[0].hist(
            log_data,
            bins=20,
            label=rf"$p={mantissa}\times 10^{{{exponent}}}$",
            color=colors[cnt],
            weights=np.ones_like(log_data) / len(log_data),
            alpha=0.7,
        )     

        ax[0].axvline(np.median(log_data), linestyle='--', color=colors[cnt]) 
        cnt+=1
        
    
    ax[0].set_xlabel(r'$\log_{10}(\mathrm{cluster\ norm})$')
    ax[0].set_ylabel("Norm. counts")
    ax[0].set_title(f"$N=${num_shots}, $r={num_rounds}$ with {decoder_label}, {code_name}")
    ax[0].legend(fontsize=9)
    

    # Choose a common cutoff grid spanning all p values
    all_data = np.concatenate(  [data_all['cluster_norms'][(code_name, p)].flatten() for p in ps] )

    gmin = np.min(all_data[all_data > 0])
    gmax = np.max(all_data)

    cutoffs = np.logspace(np.log10(gmin), np.log10(gmax), 150)

    switch_rates = np.zeros((len(ps), len(cutoffs)))

    for i, p in enumerate(ps):
        data = data_all['cluster_norms'][(code_name, p)].flatten()

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

    fig.savefig(f"{path}/cluster_norm_distr_{code_name}_{weak_decoder}.pdf",bbox_inches='tight')

    return 




plot_cluster_norm_dist_and_switch_rate_vs_cutoff_and_p(path=path,weak_decoder='uf',noise_type='erasures')


