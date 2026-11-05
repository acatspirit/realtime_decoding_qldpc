import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #move to level before sims file
import numpy as np
from numpy import array


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors, pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Microsoft Sans Serif"


def get_uf_tesseract_data_w_erasure():
    
    file = sys.path[-1] + f'/data/decoder_switching_results/decoder_switching_target_ps_0.005_weak_uf_strong_tesseract_max_shots_5000000_erasures_0.7941.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    


    return data1

def get_uf_relay_bp_data_w_erasure():
    
    file = sys.path[-1] + f'/data/decoder_switching_results/decoder_switching_target_ps_0.005_weak_uf_strong_relay_bp_max_shots_5000000_erasures_0.7941.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    


    return data1

def get_uf_data_w_erasure():

    file = sys.path[-1] + f'/data/sliding_window_results/sliding_window_uf_weak_max_shots_100000_erasures_0.7941.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    


    return data1


def get_decoder_switching_data_uf_to_relaybp(target_switch_rate, num_shots= 2000000):
    
    file = sys.path[-1] + f'/saved_data/decoder_switching_data/uf_to_relay_bp/decoder_switching_target_ps_{target_switch_rate}_weak_uf_strong_relay_bp_max_shots_{num_shots}.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    return data1

def get_decoder_switching_data_uf_to_tesseract(target_switch_rate, num_shots= 2000000):
    
    file = sys.path[-1] + f'/saved_data/decoder_switching_data/uf_to_tesseract/decoder_switching_target_ps_{target_switch_rate}_weak_uf_strong_tesseract_max_shots_{num_shots}.txt'
    with open(file,"r") as f:
        data1 = eval(f.read())    

    return data1

def plot_decoder_switching_benchmarks(strong_decoder,plotted_quantity,target_switch_rate=5e-3):

    colors = ["tab:blue","tab:orange"]
    code_names = ['[[72,12,6]]', '[[90,8,10]]']

    fig,ax = plt.subplots(1,2,figsize=(10,7), tight_layout=True)


    #========= Plot strong decoder ===============================================
    ps = [2e-4]


    #================= Plot UF with erasures =======================================================

    data_uf  = get_uf_data_w_erasure()
    data        = data_uf
    # ps          = data['ps']
    
    if plotted_quantity=='epsilons':
        
        
        cnt=0

        for code_name,val in data['epsilons'].items():
            if (code_name not in code_names):
                continue

            y    = np.array([val[p] for p in val.keys() if np.isclose(p, ps[0], atol=1e-6)]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in val.keys() if np.isclose(p, ps[0], atol=1e-6)]) #std_epsilons

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

    def get_switch_stats(dataset, target_p):
            rates = {
                k: dataset['total_switch_times'][k] / dataset['total_windows'][k]
                for k in dataset['total_switch_times']
            }
            yerrs = {
                k: np.sqrt(rates[k] * (1 - rates[k]) / dataset['total_windows'][k])
                for k in rates
            }
            return rates, yerrs

    #========== Plot decoder switching (no erasures) =================================

    if strong_decoder=='relay_bp':
        data = get_decoder_switching_data_uf_to_relaybp(target_switch_rate)
    elif strong_decoder=='tesseract':
        data = get_decoder_switching_data_uf_to_tesseract(target_switch_rate)

    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():
            if (code_name not in code_names):
                continue
            
            y    = np.array([val[p] for p in val.keys() if np.isclose(p, ps[0], atol=1e-6)]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in val.keys() if np.isclose(p, ps[0], atol=1e-6)]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))


            ax[0].errorbar(
                cnt+0.25, y, yerr=yerr,
                label=f"no erasures",
                marker='o',
                markersize=12,
                color=colors[cnt],
                markeredgecolor='k',
                elinewidth=1.5,
                capsize=7,
                capthick=3,
                ecolor='black',
                
            )
            cnt+=1

    rates_no_era, yerr_no_era = get_switch_stats(data, ps[0])
    # switch_rates = {
    #     key: data['total_switch_times'][key] / data['total_windows'][key]
    #     for key in data['total_switch_times'].keys()
    # }    

    # switch_yerr = {
    #     key: np.sqrt(
    #         switch_rates[key] * (1 - switch_rates[key]) / data['total_windows'][key]
    #     )
    #     for key in switch_rates
    # }    

    # cnt=0
    # for code_name in data['codes']:
    #     if (code_name not in code_names):
    #         continue
    #     target_p = ps[0]
            
    #     # y = np.array([next( rate for (c, p_val), rate in switch_rates.items() 
    #     #     if c == code_name and np.isclose(p_val, p, rtol=1e-3)) for p in data['ps']])
    #     y_val = next(rate for (c, p), rate in switch_rates.items() if c == code_name and np.isclose(p, target_p, rtol=1e-3))
    #     yerr_val = next(err for (c, p), err in switch_yerr.items() if c == code_name and np.isclose(p, target_p, rtol=1e-3))

    #     # yerr = np.array([next(
    #     # err for (c, p_val), err in switch_yerr.items() 
    #     # if c == code_name and np.isclose(p_val, p, rtol=1e-3)) for p in data['ps']])

        
    #     ax[1].bar(cnt,y_val,
    #               yerr=yerr_val, 
    #             #   marker='d',
    #               color=colors[cnt],
    #               label=f"erasure aware",
    #               width=0.5,
    #               edgecolor='k')   
    #     cnt+=1 


    #========== Plot decoder switching (w/ erasure) =================================

    if strong_decoder=='relay_bp':
    
        data_relay_bp  = get_uf_relay_bp_data_w_erasure()
        data           = data_relay_bp
        # ps             = data['ps']
    
    elif strong_decoder=='tesseract':
        data_tesseract  = get_uf_tesseract_data_w_erasure()
        data            = data_tesseract
    
    if plotted_quantity=='epsilons':
        
        cnt=0

        for code_name,val in data['epsilons'].items():
            if (code_name not in code_names):
                continue

            y    = np.array([val[p] for p in val.keys() if np.isclose(p, ps[0], atol=1e-6)]) #epsilons
            yerr = np.array([data["std_epsilons"][code_name][p] for p in val.keys() if np.isclose(p, ps[0], atol=1e-6)]) #std_epsilons

            n, k, d = map(int, code_name.strip("[]").split(","))


            ax[0].errorbar(
                cnt+0.25, y, yerr=yerr,
                # label=f"${code_name}, p_{{leak}}=10^{-5}$",
                marker='d',
                markersize=12,
                color=colors[cnt],
                markeredgecolor='k',
                elinewidth=1.5,
                capsize=7,
                capthick=3,
                ecolor='black',
                
            )
            cnt+=1


    ax[0].set_xticks(np.arange(2), code_names,rotation=45,fontsize=15)
    ax[1].set_xticks(np.arange(2), code_names,rotation=45,fontsize=15)


    ax[0].set_yscale('log')
    # ax[0].legend(fontsize=10)
    ax[0].set_ylabel("LER per SEC", fontsize=20)
    # ax[0].set_xlabel(r"BB code")
    ax[0].set_title(f"$p$={ps[0]}",fontsize=20)
    ax[0].legend(fontsize=18,ncols=2)

    #========= Plot the switch frequency also ================

    print(data['total_windows'].keys()) #total_switch_times, total_windows



    # switch_rates = {
    #     key: data['total_switch_times'][key] / data['total_windows'][key]
    #     for key in data['total_switch_times'].keys()
    # }    


    # switch_yerr = {
    #     key: np.sqrt(
    #         switch_rates[key] * (1 - switch_rates[key]) / data['total_windows'][key]
    #     )
    #     for key in switch_rates
    # }    


    # Compute stats for both datasets
    
    rates_era, yerr_era       = get_switch_stats(data, ps[0])

    target_p = ps[0]
    bar_width = 0.5

    for cnt, code_name in enumerate(code_names):
        
        # --- 1. Find No Erasures Data ---
        y_no_era = None
        err_no_era = None
        for (c, p), rate in rates_no_era.items():
            if code_name in c and np.isclose(p, target_p, rtol=1e-2):
                y_no_era = rate
                err_no_era = yerr_no_era[(c, p)]
                break

        if y_no_era is not None:
            ax[1].bar(
                cnt, y_no_era, yerr=err_no_era,
                width=bar_width,
                color=colors[cnt],
                edgecolor='k',
                capsize=4,
                label='No erasures' if cnt == 0 else ""
            )

        # --- 2. Find With Erasures Data ---
        y_era = None
        err_era = None
        for (c, p), rate in rates_era.items():
            if code_name in c and np.isclose(p, target_p, rtol=1e-3):
                y_era = rate
                err_era = yerr_era[(c, p)]
                break

        if y_era is not None:
            ax[1].bar(
                cnt, y_era, yerr=err_era,
                width=bar_width - 0.05,
                color=colors[cnt],
                alpha=0.6,
                hatch='//',
                edgecolor='k',
                capsize=4,
                label='With erasures' if cnt == 0 else ""
            )

    # cnt=0
    # for code_name in data['codes']:
    #     if (code_name not in code_names):
    #         continue
        
    #     target_p = ps[0]
    #     print(f"code_name: {code_name}, target_p: {target_p}, switch_rates: {switch_rates}, switch_yerr: {switch_yerr}")

    #     # Single float lookups using np.isclose to avoid KeyError:
    #     y_val = next(rate for (c, p), rate in switch_rates.items() if c == code_name and np.isclose(p, target_p, rtol=1e-3))
    #     yerr_val = next(err for (c, p), err in switch_yerr.items() if c == code_name and np.isclose(p, target_p, rtol=1e-3))

    #     ax[1].bar(
    #         cnt, y_val, yerr=yerr_val,
    #         color=colors[cnt],
    #         width=0.4,
    #         edgecolor='k',
    #         hatch='//',
    #         capsize=4
    #     )

        # yerr = np.array([ switch_yerr[(code_name, p)] for p in data['ps'] ])

        
        # ax[1].bar(cnt,y,
        #           yerr=yerr, 
        #         #   marker='o',
        #           color=colors[cnt],
        #         #   label=rf"{code_name}, $R_{{e}}=0.7941$",
        #           alpha=0.7,
        #           edgecolor='k')   
        # cnt+=1 


    # ax[1].set_ylabel("$p_s$")
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

    # ax[1].set_title(f"target $p_s=$ {target_switch_rate}",fontsize=15)
    # ax[1].set_yscale('log')
    
    # ax[1].legend(fontsize=9,ncols=2)   

    style_legend_elements = [
    Patch(facecolor='gray', edgecolor='k', alpha=0.7, label='UF (w/ erasures)'),
    Line2D([0], [0], marker='o', color='k', label='Switching (no erasures)', 
           linestyle='None', markersize=7, markeredgecolor='k'),
    Line2D([0], [0], marker='d', color='k', label='Switching (w/ erasures)', 
           linestyle='None', markersize=7, markeredgecolor='k'),
    ]
    grayscale_handles_right = [
    Patch(facecolor='gray', edgecolor='k', label='No erasures'),
    Patch(facecolor='white', edgecolor='k', hatch='//', label='With erasures')
    ]

    # 2. Add the custom legend to ax[0]
    ax[0].legend(handles=style_legend_elements, bbox_to_anchor=(0.5, 1.15), 
    loc='lower left', fontsize=12, frameon=True) 
    ax[1].legend(handles=grayscale_handles_right, bbox_to_anchor=(0.5, 1.15), 
    loc='lower left', fontsize=12, frameon=True)
    ax[1].set_title(f"target $p_s=$ {target_switch_rate}",fontsize=15)



    # plt.legend()
    plt.tight_layout()
    plt.show()

    fig.savefig(f"/Users/ariannameinking/Documents/Brown_Research/realtime_decoding_qldpc/data/plots/decoder_switching_benchmarks_ps_{target_switch_rate}_strong_{strong_decoder}_erasure.pdf",bbox_inches='tight')
   
    return 



plot_decoder_switching_benchmarks(strong_decoder='tesseract',plotted_quantity='epsilons',target_switch_rate=5e-3)                                  