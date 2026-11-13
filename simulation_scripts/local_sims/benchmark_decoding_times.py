import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) #move to level before sims file


''' Get the ler performance for a single decoder w/ sliding window.'''


import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

from joblib import Parallel, delayed
from src.realtime_decoding.decoder_switching_class import decoder_switching_class
from tesseract_decoder import tesseract
from src.realtime_decoding.decoders_utils import collect_default_decoder_params
from quits import detector_error_model_to_matrix

import time
import relay_bp

from BeamSearchDecoder.decoder.beam_search_decoder import BeamSearchDecoder
from uf_decoder.py_wrapper import py_decoder as uf
from scipy.sparse import csr_matrix
from ldpc.bplsd_decoder import BpLsdDecoder


#=================== Test on code-capacity for bplsd and uf =======================================

def code_capacity_test_bplsd_vs_uf(p,num_shots):

    from src.realtime_decoding.circuits import create_bb_codes_circuit

    basis = 'Z'
    code_name = "[[72,12,6]]"
    
    num_rounds = 1
    
    circuit, bb = create_bb_codes_circuit(code_name,p,num_rounds,basis)

    H = bb.hz 

    decoder_uf = uf.UFDecoder(csr_matrix(H))

    erasures = np.zeros(shape=H.shape[1], dtype=np.uint8)

    all_elapsed_times = []
    for i in range(num_shots):

        error = np.random.binomial(n=1, p=p, size=H.shape[1]).astype(np.uint8)
        
        syndrome = (H @ error % 2).astype(np.uint8)

        start_time = time.perf_counter()
        decoder_uf.ldpc_decode(syndrome,erasures)
        end_time = time.perf_counter()
        
        elapsed_time = end_time-start_time
        all_elapsed_times.append(elapsed_time)

        
    mean_elapsed_time = np.mean(all_elapsed_times)
    print("mean time for uf decoding:",mean_elapsed_time)



    decoder_params = {'bp_method': 'product_sum',
                'max_bp_iters': 0,
                'schedule': 'serial',
                'lsd_method': 'lsd_cs',
                'lsd_order': 0}

    decoder = BpLsdDecoder(H,
                        error_rate = p,
                        bp_method=decoder_params['bp_method'],
                        max_iter=decoder_params['max_bp_iters'],
                        schedule=decoder_params['schedule'],
                        lsd_method=decoder_params['lsd_method'],
                        lsd_order=decoder_params['lsd_order'])
    


    all_elapsed_times = []
    for i in range(num_shots):

        error = np.random.binomial(n=1, p=p, size=H.shape[1]).astype(np.uint8)
        syndrome = (H @ error % 2).astype(np.uint8)

        start_time = time.perf_counter()
        decoder.decode(syndrome)
        end_time = time.perf_counter()
        
        elapsed_time = end_time-start_time
        all_elapsed_times.append(elapsed_time)

    mean_elapsed_time_bplsd = np.mean(all_elapsed_times)
    print("mean time for bplsd decoding:",mean_elapsed_time_bplsd)


    print("Ratio:",mean_elapsed_time_bplsd/mean_elapsed_time)

    return 


# code_capacity_test_bplsxd_vs_uf(p=1e-1,num_shots=100_000)


#==================== Test for circuit level noise for all decoders ==========================================


def configure_decoder_for_timing_test(code_name,p,num_shots,decoder_option,num_rounds=25,):
    '''
    

    decoder_option: 'tesseract', 'relay_bp', 'beamsearch8_230iters', 'beamsearch32_340iters', 'uf'

    '''
    from BeamSearchDecoder.decoder.beam_search_decoder import BeamSearchDecoder
    
    basis      = 'Z'

    n, k, d = map(int, code_name.strip("[]").split(","))

    nbuffer = d            #Buffer region
    F       = d//2         #Commit region
    W       = nbuffer + F  #Entire window

    strong_decoders = ['tesseract','relay_bp','beamsearch8_230iters','beamsearch32_340iters']
    weak_decoders   = ['uf','bplsd']


    if decoder_option in strong_decoders:
        strong_decoder_option = decoder_option 
        weak_decoder_option = 'uf' #just pick one, irrelevant
    else:
        strong_decoder_option = 'relay_bp' #just pick one, irrelevant
        weak_decoder_option = decoder_option

    
    if decoder_option == 'beamsearch8_230iters' or decoder_option=='beamsearch32_340iters':
        strong_decoder_option = 'beamsearch'

    beamsearch_option = 1
    if decoder_option == 'beamsearch32_340iters':
        beamsearch_option = 2

    
    test  = decoder_switching_class(code_name=code_name,
                                        num_rounds=num_rounds,
                                        p=p,
                                        basis=basis,
                                        num_shots=num_shots,
                                        W=W,
                                        F=F,
                                        strong_decoder_option=strong_decoder_option,
                                        weak_decoder_option=weak_decoder_option,
                                        beamsearch_option=beamsearch_option)


    defaults       = collect_default_decoder_params(decoder_option)  #import default parameters from decoders_utils.py
    circuit        = test.circuit 
    dem            = circuit.detector_error_model()
    chk,obs,priors = detector_error_model_to_matrix(dem) 
    det_events     = test.detection_events

    if decoder_option=='tesseract':
        config  = tesseract.TesseractConfig(dem=dem, 
                                            det_beam=10, #defaults['det_beam']
                                            pqlimit=50_000, #defaults['pqlimit']
                                            beam_climbing=defaults['beam_climbing'],
                                            no_revisit_dets=defaults['no_revisit_dets'])#

        decoder = config.compile_decoder()
    
    elif decoder_option=='relay_bp':

        decoder = relay_bp.RelayDecoderF32(chk.tocsr(),                                   
                                    error_priors=np.array(priors,dtype=np.float64),
                                    gamma0=defaults['gamma0'],
                                    pre_iter=defaults['pre_iter'],
                                    num_sets=defaults['num_sets'],
                                    set_max_iter=defaults['set_max_iter'],
                                    gamma_dist_interval=defaults['gamma_dist_interval'],
                                    stop_nconv=defaults['stop_nconv'])
    
    elif decoder_option=='beamsearch32_340iters' or decoder_option=='beamsearch8_230iters':

        decoder = BeamSearchDecoder(pcm = chk,         
                                    error_channel= priors,          
                                    max_rounds=defaults['max_rounds'],
                                    beam_width=defaults['beam_width'],
                                    num_results=defaults['num_results'],
                                    iters_per_round=defaults['iters_per_round'],
                                    initial_iters=defaults['initial_iters'],
                                    )
    elif decoder_option=='uf':
        decoder = uf.UFDecoder(csr_matrix(chk))

    elif decoder_option=='bplsd':
        
        decoder = BpLsdDecoder(chk,
                            error_channel=priors,
                            bp_method= defaults['bp_method'],
                            # bp_method = 'minimum_sum',
                            max_iter=defaults['max_bp_iters'],
                            schedule='parallel',#defaults['schedule'],
                            lsd_method=defaults['lsd_method'],
                            lsd_order=defaults['lsd_order'])

    return decoder,det_events,chk

def run_timing_test(num_rounds=25,num_shots=100,decoder_option='relay_bp',):

    #configure decoders first for each code_name,p

    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"] 
    colors     = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']

    if decoder_option=='uf':

        ps = [1e-4, 2e-4, 5e-4, 1e-3, 3e-3]
    else:
        ps = [1e-4, 5e-4, 1e-3, 3e-3] 

    tasks = [
        (code_name, p)
        for code_name in code_names
        for p in ps
    ]


    def decoding_time_one_task_strong(task,num_shots):

        decoder,det_events,_ = configure_decoder_for_timing_test(task[0],task[1],num_shots=num_shots,
                                                               decoder_option=decoder_option,
                                                               )
        

        all_elapsed_times = []

       

        for i in range(num_shots):

            restricted_events = np.array(det_events[i,:],dtype=np.uint8)
            
            start_time = time.perf_counter()
            decoder.decode(restricted_events)
            end_time = time.perf_counter()
            elapsed_time = end_time-start_time
            all_elapsed_times.append(elapsed_time/num_rounds)            

        mean_elapsed_time = np.mean(all_elapsed_times)
        std_elapsed_time  = np.std(all_elapsed_times)
        
        

        return mean_elapsed_time,std_elapsed_time,np.percentile(all_elapsed_times,99.9)

    def decoding_time_one_task_uf(task,num_shots):

        decoder,det_events,chk = configure_decoder_for_timing_test(task[0],task[1],num_shots=num_shots,
                                                               decoder_option=decoder_option,
                                                                   )
        if task[1]==3e-3:
            num_shots=100

        all_elapsed_times = []

        
        erasure_array = np.zeros(chk.shape[1], dtype=np.uint8)

        for i in range(num_shots):

            restricted_events = np.array(det_events[i,:],dtype=np.uint8)
            
            start_time = time.perf_counter()
            decoder.ldpc_decode(restricted_events,erasure_array)
            end_time = time.perf_counter()
            elapsed_time = end_time-start_time
            all_elapsed_times.append(elapsed_time/num_rounds)

          

        mean_elapsed_time = np.mean(all_elapsed_times)
        std_elapsed_time  = np.std(all_elapsed_times)
        
        

        return mean_elapsed_time,std_elapsed_time,np.percentile(all_elapsed_times,99.9)


    print("Running timing test...")

    if decoder_option=='uf':
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(decoding_time_one_task_uf)(
                task, num_shots,
            )
            for task in tasks
        )

    else:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(decoding_time_one_task_strong)(
                task, num_shots,
            )
            for task in tasks
        )        

    print("Done.")

    mean_times_to_save = {}
    mean_stdev_to_save = {}
    p99_9_to_save = {}
    cnt=0
    for code_name in code_names:

        all_mean = []
        all_std = []
        all_p99 = []

        for p in ps:

            # Find result corresponding to this (code_name, p)
            idx = tasks.index((code_name, p,))
            mean, std, p99 = results[idx]

            all_mean.append(mean)
            all_std.append(std)
            all_p99.append(p99)

        # Plot
        plt.errorbar(
            ps,
            all_mean,
            yerr=all_std,
            label=f"{code_name} mean",
            marker='o',
            color=colors[cnt],
            markeredgecolor='k'
        )

        plt.errorbar(
            ps,
            all_p99,
            marker='s',
            color=colors[cnt],
            linestyle='None',
            markeredgecolor='k'
        )

        mean_times_to_save[code_name] = all_mean
        mean_stdev_to_save[code_name] = all_std
        p99_9_to_save[code_name] = all_p99

        cnt+=1

    

    dict_to_save = {"ps":ps,
                    "mean_decoding_times": mean_times_to_save,
                    "std_dev": mean_stdev_to_save,
                    "p99.9":p99_9_to_save,
                    "codes":code_names}

    

    
    txt_to_save = sys.path[-1] + f'/saved_data/timings/decoding_times_{decoder_option}_shots_{num_shots}_bp_schedule_parallel.txt'

    with open(txt_to_save, 'w') as file:
        file.write(str(dict_to_save))   


    plt.legend(fontsize=10)
    plt.xlabel("physical error rate")
    plt.ylabel("Decoding time (s)")
    # plt.title("decoder_option")
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    return 

# run_timing_test(num_shots=50_000,decoder_option='bplsd',)


def plot_timings():

    fig,ax = plt.subplots(1,6,figsize=(13,4))
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    num_shots = 50_000
    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]

    all_files = [
        sys.path[-1] + f'/saved_data/timings/decoding_times_beamsearch_option_1_shots_{num_shots}.txt', #beam8_230iters 
        sys.path[-1] + f'/saved_data/timings/decoding_times_beamsearch_option_2_shots_{num_shots}.txt', #beam32_340iters
        sys.path[-1] + f'/saved_data/timings/decoding_times_tesseract_shots_{num_shots}.txt', #tesseract 
        sys.path[-1] + f'/saved_data/timings//decoding_times_relay_bp_shots_{num_shots}.txt', #relay-bp 
        sys.path[-1] + f'/saved_data/timings//decoding_times_bplsd_shots_{num_shots}_bp_schedule_parallel.txt', #bp-lsd 
        sys.path[-1] + f'/saved_data/timings//decoding_times_uf_shots_{num_shots}.txt' #uf    
    ]

    decoder_cnt = 0
    for file in all_files: 
        with open(file,"r") as f:
            data = eval(f.read())    

        cnt=0
        for code in code_names:

            times = data['mean_decoding_times'][code]
            std_dev = data['std_dev'][code]
            ps = data['ps']

            p99_9 = data['p99.9'][code]

            # ax[decoder_cnt].errorbar(ps,y=times,yerr=std_dev,marker='o',label=code,color=colors[cnt],markeredgecolor='k')
            ax[decoder_cnt].plot(ps,times,marker='o',label=f"mean, {code}",color=colors[cnt],markeredgecolor='k')
            ax[decoder_cnt].plot(ps,p99_9,marker='s',label=f"p99.9, {code}",color=colors[cnt],linestyle='None',markeredgecolor='k')

            cnt+=1

        decoder_cnt+=1

    names = ["Beam8_230iters", "Beam32_340iters", "Tesseract", "Relay-BP", "BP-LSD", "UF"]
    cnt=0
    for name in names:
        ax[cnt].set_title(name,fontsize=13)
        cnt+=1

    for k in range(len(names)):
        
        ax[k].set_xlabel("physical error rate")
        # if k==2:
        #     ax[k].legend(fontsize=8,ncols=1)

        ax[k].axhline(y=1e-3, linestyle='--',color='k',linewidth=0.8)
        ax[k].axhline(y=1e-4, linestyle='--',color='k',linewidth=0.8)
        if k==0:
            ax[k].set_ylabel("$T_{dec.}$ (s)")
        ax[k].set_yscale('log')
        ax[k].set_xscale('log')
        ax[k].set_ylim([2e-6,3e-1])


    handles, labels = ax[0].get_legend_handles_labels()

    # Add common legend at the top
    fig.legend(handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=5, fontsize=13)


    plt.tight_layout()
    plt.show()

    # fig.savefig(f"/Users/evatakou/Desktop/Figs_realtime/from_python/Timings_all_decoders.pdf",bbox_inches='tight')

    return 


plot_timings()

