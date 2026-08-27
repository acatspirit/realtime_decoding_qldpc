'''
Use this to run decoder switching simulations, given a fixed target switch rate
'''

import subprocess
import shutil
import sys
import os
import gzip
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #move to level before sims file
import numpy as np


from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
plt.rcParams["font.family"] = "Microsoft Sans Serif"

from joblib import Parallel, delayed
import json
from pathlib import Path
from src.realtime_decoding.decoder_switching_class import decoder_switching_class
from numpy import array
import pickle 

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

'''This is my function for how to load bplsd data of cluster norm distributions
Adjust for UF
'''

def get_cutoffs_for_input_switch_rate(target_switch_rate,plot=False, weak_decoder='uf', num_shots=500_000, folder_name = sys.path[-1] + f'/data/cluster_norm_statistics/'):


    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"] 
    cutoffs_to_set = {}

    if plot:
        fig,ax = plt.subplots(1,5,figsize=(20,5))

    data_per_code = []
    cnt=0
    for code_name in code_names:
    
        txt_to_load = folder_name + f'cluster_norm_distributions_code_{code_name}_{weak_decoder}_max_shots_{num_shots}.pkl.gz'

        if Path(txt_to_load).name.endswith('.pkl.gz'):
            with gzip.open(txt_to_load, "rb") as file:
                data = pickle.load(file)
        else:
            with open(txt_to_load, "rb") as file:
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

            if plot:

                

                ax[cnt].semilogx(cutoffs, switch_rates[k], marker='.',label=f' p={ps[k]}')
                ax[cnt].axhline(target_switch_rate)
                
                ax[cnt].set_xlabel("cutoff")
                if cnt==0:
                    ax[cnt].set_ylabel("switch rate")
                ax[cnt].grid()
                ax[cnt].set_yscale('log')
                # ax[cnt].set_xscale('log')
                ax[cnt].legend(fontsize=10)
                ax[cnt].set_title(code_name)
        cnt+=1
                
    if plot:
        print(cutoffs_to_set)
        plt.tight_layout()
        plt.show()

    

    return cutoffs_to_set,data_per_code


# target_switch_rate = 1e-3
# get_cutoffs_for_input_switch_rate(target_switch_rate,plot=True)

'''Main function for decoder switching.

-Adjust cutoffs (load for uf)
-Adjust the path to save the results
-Adjust the physical error rate range for uf

'''
def sort_by_d_then_k(code_str):
        try:
            # Strip the brackets and split the string into a list of numbers
            parts = code_str.strip("[]").split(",")
            
            # parts[0] is n, parts[1] is k, parts[2] is d
            k = int(parts[1])
            d = int(parts[2])
            n = int(parts[0])
            
            # Return tuple (d, k) so it sorts by d first, then breaks ties with k
            return (d, n, k)
        except (IndexError, ValueError):
            return (0, 0, 0) # Fallback


def get_ler_for_decoder_switching(num_shots,shots_per_job,target_switch_rate=5e-3,weak_decoder='bplsd',strong_decoder='relay_bp'):
    '''
    Inputs:
    num_shots: max # of shots per (p,code)
    shots_per_job: how many shots to consider per job of joblib
    target_switch_rate: the switch probability (fixed across p's and codes)
    weak_decoder: 'bplsd' or 'uf'
    strong_decoder: 'relay_bp' or 'tesseract'
    '''


    #ADJUST THIS FOR UF
    cutoffs_to_set,_ = get_cutoffs_for_input_switch_rate(target_switch_rate=target_switch_rate) 

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    basis      = 'Z'
    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]    
    ps         = [2e-3,3e-3,4e-3,5e-3] #I RUN THESE RATES ONLY FOR BPLSD
    num_rounds = 25
    

    def process_one_round_value(code_name,p,num_shots,cutoff):
        
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
        
        new_shots,cluster_norms,switch_times,logical_errors = test.decode_with_sliding_window_and_decoder_switching(cluster_norm_cutoff=cutoff)


        num_windows =len(test.weak_decoder) #total # of windows -- needed for getting switch rates

        
        result = {"logical_errors": np.sum(logical_errors),"cluster_norms": cluster_norms,
                  "switch_times": np.sum(switch_times), "num_windows": num_windows}
        

        print("Sim done.")

        return code_name,p,new_shots,result
    
    

    tasks = []
    import multiprocessing as mp
    n_jobs = mp.cpu_count()    
    chunk_size = max(shots_per_job, num_shots // (100 * n_jobs)) 

    for code_name in code_names:

        for p in ps:

            tasks.extend(
                (code_name,p,chunk_size)
                for _ in range(num_shots // chunk_size) )       


    results = Parallel(n_jobs=-1,verbose=50,)(delayed(process_one_round_value)(code_name,p,shots,cutoffs_to_set[(code_name,p)]) for code_name, p,shots in tasks)      
    
    total_errors  = {}
    total_shots   = {}
    total_switch_times = {}
    total_windows = {}

    for code_name,p,shot,result in results:
        key = (code_name,p)
        total_errors[key] = 0
        total_shots[key] = 0    
        total_switch_times[key] = 0    
        total_windows[key] = 0
        

    for code_name,p,shot,result in results:
        key = (code_name,p)
        total_errors[key] += result["logical_errors"]
        total_shots[key] += shot
        total_switch_times[key] += result["switch_times"]

        total_windows[key] += result['num_windows'] * shot
        
        
    ler_results = {(code_name,p,): total_errors[(code_name,p,)] / total_shots[(code_name,p,)]
                        for code_name in code_names
                        for p in ps
                        }
    
    
    yerr_results = {(code_name,p, ): np.sqrt(ler_results[(code_name,p,)]*(1-ler_results[(code_name,p,)])/total_shots[(code_name,p,)])
                        for code_name in code_names
                        for p in ps
                        }      
    

    fig, ax = plt.subplots(1,2)

    eps_to_save = {}
    errs_in_eps_to_save ={}


    #================= Plot ler from decoder switching ===============================================
    
    cnt=0
    for code_name in code_names:

        n, k, d = map(int, code_name.strip("[]").split(","))

        pL_vals = {p: ler_results[(code_name,p,)] for p in ps}
        pL_errs = {p: yerr_results[(code_name,p,)] for p in ps}

        eps = {p: 1-(1-pL_vals[p])**(1/num_rounds) for p in ps}

        eps_errs = { p: (pL_errs[p] / num_rounds) * (1 - pL_vals[p])**(1 / num_rounds - 1)
                    for p in ps }        

        ax[0].errorbar(ps,eps.values(),yerr=eps_errs.values(),label=f"{code_name}, {weak_decoder} to {strong_decoder}",color=colors[cnt],marker='o',markeredgecolor='k')
        cnt+=1

        eps_to_save[code_name] = eps
        errs_in_eps_to_save[code_name] = eps_errs 

     

    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].legend(fontsize=9)
    ax[0].set_xlabel("physical error rate")
    ax[0].set_ylabel("LER per SEC")

    

    switch_rates = {
        key: total_switch_times[key] / total_windows[key]
        for key in total_switch_times
    }    

    switch_yerr = {
        key: np.sqrt(
            switch_rates[key] * (1 - switch_rates[key]) / total_windows[key]
        )
        for key in switch_rates
    }    

    cnt=0

    #================= Plot the switch rate ===============================================

    for code_name in code_names:
        # all_switch_rates = {p: total_switch_times[(code_name,p)] for p in ps}
        y = np.array([ switch_rates[(code_name, p)] for p in ps ])

        yerr = np.array([ switch_yerr[(code_name, p)] for p in ps ])

        
        ax[1].errorbar(ps,y,yerr=yerr, marker='o',color=colors[cnt],label=f"{code_name}")   
        cnt+=1 

    

    ax[1].set_ylabel("$p_{switch}$")
    ax[1].set_xlabel("physical error rate")
    ax[1].set_title(f"for $p_s$: {target_switch_rate}")
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].legend(fontsize=9)


  

    plt.tight_layout()
    plt.show()

    dict_to_save = {"basis":basis,
                    "weak_decoder": weak_decoder,
                    "strong_decoder": strong_decoder,
                    "target_switch_rate": target_switch_rate,
                    "codes": code_names,
                    "ps": ps,
                    "r":num_rounds,
                    "total_errors":total_errors,
                    "shots":total_shots,                                   #these are the actual shots that were run for any code and p
                    "pL@r":ler_results,
                    "std_pL@r":yerr_results,
                    "epsilons":eps_to_save,
                    "std_epsilons":errs_in_eps_to_save,
                    "switch_rates": switch_rates,
                    "switch_rate_err": switch_yerr,
                    "total_switch_times": total_switch_times,
                    "total_windows": total_windows}         
    

    txt_to_save = sys.path[-1] + f'/saved_data/decoder_switching_data/decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots}.txt' #p_2e_minus_3_Gross_only

    with open(txt_to_save, 'w') as file:
        file.write(str(dict_to_save))      


    return 


def get_ler_for_decoder_switching_dcc(target_switch_rate=2.5e-1, num_shots=100_000, shots_per_job=10_000, weak_decoder='uf', strong_decoder='tesseract'):
    '''
    Inputs:
    num_shots: max # of shots per (p,code)
    shots_per_job: how many shots to consider per job of joblib
    target_switch_rate: the switch probability (fixed across p's and codes)
    weak_decoder: 'bplsd' or 'uf'
    strong_decoder: 'relay_bp' or 'tesseract'
    '''


    cutoffs_to_set,_ = get_cutoffs_for_input_switch_rate(target_switch_rate=target_switch_rate, weak_decoder=weak_decoder) # removed num_shots since we don't really care

    # colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"),0)

    basis      = 'Z'
    code_names = ["[[72,12,6]]", "[[90,8,10]]", "[[126,8,10]]", "[[144,12,12]]", "[[162,8,14]]"]    
    # ps         = [2e-3,3e-3,4e-3,5e-3] #I RUN THESE RATES ONLY FOR BPLSD
    ps = np.logspace(-4,-3.5,6)
    num_rounds = 25
    
    tasks = []

    for code_name in code_names:
        for p in ps:
            for _ in range(num_shots // shots_per_job):
                tasks.append((code_name,p,shots_per_job))
                
    if task_id >= len(tasks):
        print(f"Task ID {task_id} is out of bounds for {len(tasks)} tasks. Exiting cleanly.")
        return

    code_name, p, shots = tasks[task_id]
    cutoff = cutoffs_to_set[(code_name, p)]

    print(f"--- RUNNING ARRAY TASK {task_id} ---")
    print(f"Code: {code_name}, p: {p}, shots: {shots}")

    n, k, d = map(int, code_name.strip("[]").split(","))
            
    
    nbuffer = d            #Buffer region
    F       = d//2         #Commit region
    W       = nbuffer + F  #Entire window

    test  = decoder_switching_class(code_name=code_name,
                                        num_rounds=num_rounds,
                                        p=p,
                                        basis=basis,
                                        num_shots=shots,
                                        W=W,
                                        F=F,
                                        strong_decoder_option=strong_decoder,
                                        weak_decoder_option=weak_decoder)    
    
    new_shots,cluster_norms,switch_times,logical_errors = test.decode_with_sliding_window_and_decoder_switching(cluster_norm_cutoff=cutoff, rel_error_tol=0.01)


    num_windows =len(test.weak_decoder) #total # of windows -- needed for getting switch rates

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "data" / "decoder_switching_data" / f"raw_batches_target_{target_switch_rate}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = output_dir / f"task_{task_id}_{code_name}_p{p:.6f}.json"
    
    dict_to_save = {
        "task_id": task_id,
        "basis": basis,
        "weak_decoder": weak_decoder,
        "strong_decoder": strong_decoder,
        "target_switch_rate": target_switch_rate,
        "code_name": code_name,
        "p": p,
        "r": num_rounds,
        "shots_run": new_shots,
        "logical_errors": int(np.sum(logical_errors)),
        "switch_times": int(np.sum(switch_times)),
        "num_windows": num_windows
    }

    with open(file_name, 'w') as file:
        json.dump(dict_to_save, file)
        
    print(f"Task {task_id} finished successfully. Saved to {file_name}")
    return

def download_from_dcc(remote_path, local_dir, username="am1155", host="dcc-login.oit.duke.edu"):
    """
    Downloads a file or directory from the Duke Compute Cluster (DCC) using scp.
    
    Parameters:
    - username (str): Your Duke NetID (e.g., 'am1155').
    - remote_path (str): The absolute path to the data on the cluster.
    - local_dir (str/Path): The local directory where you want to save the data.
    - host (str): The DCC login node address.
    """
    
    # Ensure the local directory exists
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Build the scp command
    # We include '-r' (recursive) by default so it works for both single files and entire folders of CSVs
    scp_command = [
        "scp",
        "-r", 
        f"{username}@{host}:{remote_path}",
        str(local_path)
    ]
    
    print(f"Executing: {' '.join(scp_command)}")
    
    try:
        # Execute the command
        # Note: We don't use capture_output=True here so that if Duo/password 
        # prompts occur, they still show up in your terminal for you to interact with.
        subprocess.run(scp_command, check=True)
        print(f"✅ Success! Data downloaded to: {local_path.absolute()}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred during download. Return code: {e.returncode}")
        print("Check if the remote path is correct and that you are connected to the Duke VPN.")

def merge_dcc_results(target_switch_rate, weak_decoder, strong_decoder, num_shots_max, dcc_data_dir="/hpc/group/brownlab/am1155/realtime_decoding_qldpc/simulation_scripts/data/decoder_switching_data"):
    """
    after running on the DCC, converts data that belongs to one task into full statistics version / calculating LERs 
    """

    # Setup paths using pathlib
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "decoder_switching_data_temp" / f"raw_batches_target_{target_switch_rate}"
    
    out_dir = script_dir.parent / "data" / "decoder_switching_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_to_save = out_dir / f'decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots_max}.txt'

    download_from_dcc(
            remote_path=dcc_data_dir + f"/raw_batches_target_{target_switch_rate}/*.json",
            local_dir=input_dir
        )
    
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return
        
    # Initialize aggregators
    total_errors = {}
    total_shots = {}
    total_switch_times = {}
    total_windows = {}
    
    # Track static parameters to rebuild the final dictionary
    basis = None
    num_rounds = None
    code_names = set()
    ps = set()
    
    # --- NEW: STEP 0. Read existing file to prepopulate the dictionaries ---
    if txt_to_save.exists():
        try:
            with open(txt_to_save, 'r') as file:
                existing_dict = eval(file.read())
                
            print("Found existing file. Loading previous shots and errors...")
            
            # Carry over static parameters
            basis = existing_dict.get("basis")
            num_rounds = existing_dict.get("r")
            
            # Prepopulate the aggregators with the old data
            old_errors = existing_dict.get("total_errors", {})
            old_shots = existing_dict.get("shots", {})
            old_switch_times = existing_dict.get("total_switch_times", {})
            old_windows = existing_dict.get("total_windows", {})
            
            for key in old_shots:
                total_errors[key] = old_errors.get(key, 0)
                total_shots[key] = old_shots.get(key, 0)
                total_switch_times[key] = old_switch_times.get(key, 0)
                total_windows[key] = old_windows.get(key, 0)
                
                # Add the old codes and p's to the sets so they are preserved
                code_names.add(key[0]) 
                ps.add(key[1])
                
        except Exception as e:
            print(f"⚠️ Could not load existing file. Starting fresh. Error: {e}")
            
            
    # 1. Find and aggregate all JSON files (This now adds to the old data)
    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} result files. Merging...")
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        code = data["code_name"]
        p = data["p"]
        key = (code, p)
        
        code_names.add(code)
        ps.add(p)
        
        # Grab static values from the first file we open (if not already set by the old dict)
        if basis is None:
            basis = data["basis"]
            num_rounds = data["r"]
            
        # Initialize keys if they don't exist yet
        if key not in total_errors:
            total_errors[key] = 0
            total_shots[key] = 0
            total_switch_times[key] = 0
            total_windows[key] = 0
            
        # Add up the raw counts (Old data + New data)
        total_errors[key] += data["logical_errors"]
        total_shots[key] += data["shots_run"]
        total_switch_times[key] += data["switch_times"]
        total_windows[key] += data["num_windows"] * data["shots_run"]

        
    code_names = sorted(list(code_names), key=sort_by_d_then_k)
    ps = sorted(list(ps))

    # 2. Calculate LER & Standard Errors (This now computes on the COMBINED shots and errors)
    ler_results = {}
    yerr_results = {}
    
    for code in code_names:
        for p in ps:
            key = (code, p)
            if key in total_shots and total_shots[key] > 0:
                ler = total_errors[key] / total_shots[key]
                ler_results[key] = ler
                yerr_results[key] = np.sqrt(ler * (1 - ler) / total_shots[key])
            else:
                ler_results[key] = np.nan
                yerr_results[key] = np.nan

    # 3. Calculate Epsilons (Computed on COMBINED LER)
    eps_to_save = {}
    errs_in_eps_to_save = {}
    
    for code in code_names:
        pL_vals = {p: ler_results.get((code, p), np.nan) for p in ps}
        pL_errs = {p: yerr_results.get((code, p), np.nan) for p in ps}
        
        eps = {}
        eps_errs = {}
        for p in ps:
            ler = pL_vals[p]
            err = pL_errs[p]
            if not np.isnan(ler) and ler < 1.0: # safety check against 0 division/complex numbers
                eps[p] = 1 - (1 - ler)**(1 / num_rounds)
                eps_errs[p] = (err / num_rounds) * (1 - ler)**((1 / num_rounds) - 1)
            else:
                eps[p] = np.nan
                eps_errs[p] = np.nan
                
        eps_to_save[code] = eps
        errs_in_eps_to_save[code] = eps_errs

    # 4. Calculate Switch Rates (Computed on COMBINED windows)
    switch_rates = {
        key: total_switch_times[key] / total_windows[key] if total_windows[key] > 0 else np.nan
        for key in total_switch_times
    }    

    switch_yerr = {
        key: np.sqrt(switch_rates[key] * (1 - switch_rates[key]) / total_windows[key]) if total_windows[key] > 0 else np.nan
        for key in switch_rates
    }
    
    # 5. Build Final Target Dictionary
    dict_to_save = {
        "basis": basis,
        "weak_decoder": weak_decoder,
        "strong_decoder": strong_decoder,
        "target_switch_rate": target_switch_rate,
        "codes": code_names,
        "ps": ps,
        "r": num_rounds,
        "total_errors": total_errors,
        "shots": total_shots,
        "pL@r": ler_results,
        "std_pL@r": yerr_results,
        "epsilons": eps_to_save,
        "std_epsilons": errs_in_eps_to_save,
        "switch_rates": switch_rates,
        "switch_rate_err": switch_yerr,
        "total_switch_times": total_switch_times,
        "total_windows": total_windows
    }
    
    # 6. Save back to original .txt format, replacing the old file with the newly updated one
    with open(txt_to_save, 'w') as file:
        file.write(str(dict_to_save))
        
    print(f"Merge complete! Saved master dictionary to:\n{txt_to_save}")
    
    try:
        shutil.rmtree(input_dir.parent)
        print(f"🧹 Cleanup successful: Deleted raw data directory {input_dir.parent}")
    except Exception as e:
        print(f"⚠️ Could not delete directory {input_dir.parent}. Error: {e}")
        
    return dict_to_save

def plot_decoder_switching_results(target_switch_rate, weak_decoder, strong_decoder, num_shots_max, data_dict=None, include_strong=True, include_weak=True, p_range=None):
    """
    Plots the results from the merged decoder switching data.
    """
    script_dir = Path(__file__).resolve().parent
    if data_dict is None:
        results_file = script_dir.parent / "data" / "decoder_switching_results" / f'decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots_max}.txt'
        
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            return
        
        with open(results_file, 'r') as file:
            # eval handles tuple keys correctly for the dictionary
            data_dict = eval(file.read())


    if include_weak: # right now we just want to plot the weak / switching comparison
        if weak_decoder == 'uf':
            weak_results_file = script_dir.parent / "data" / "raw" / "single_sliding_window_uf_max_shots_15000.txt"
        elif weak_decoder == 'bplsd':
            weak_results_file = script_dir.parent / "saved_data" / "single_sliding_window_bplsd_max_shots_30000.txt"

        if weak_results_file and weak_results_file.exists():
                    with open(weak_results_file, 'r') as file:
                        weak_data_dict = eval(file.read())
        else:
            print(f"Weak decoder results file not found: {weak_results_file}")
            weak_data_dict = None


    if include_strong:
        if strong_decoder == 'relay_bp':
            strong_results_file = script_dir.parent / "saved_data" / "single_sliding_window_relay_bp_max_shots_20000.txt"
        elif strong_decoder == 'tesseract':
            strong_results_file = script_dir.parent / "data" / "raw" / "single_sliding_window_tesseract_max_shots_100000.txt"

        if strong_results_file and strong_results_file.exists():
            with open(strong_results_file, 'r') as file:
                strong_data_dict = eval(file.read())
        else:
            print(f"Strong decoder results file not found: {strong_results_file}")
            strong_data_dict = None


    code_names = sorted(list(data_dict["codes"]), key=sort_by_d_then_k)
    ps = data_dict["ps"]
    if p_range is not None:
        ps = [p for p in ps if p_range[0] <= p <= p_range[1]]
    eps_to_save = data_dict["epsilons"]
    errs_in_eps_to_save = data_dict["std_epsilons"]

    
    fig, ax = plt.subplots(1, 1, figsize=(5, 7)) 
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    
    #================= Plot ler from decoder switching ===============================================
    cnt = 0
    for code_name in code_names:
        eps_vals = [eps_to_save[code_name][p] for p in ps  if p_range[0] <= p <= p_range[1]]
        eps_errs = [errs_in_eps_to_save[code_name][p] for p in ps  if p_range[0] <= p <= p_range[1]]
        
        ax.errorbar(
            ps, 
            eps_vals, 
            yerr=eps_errs, 
            label=f"{code_name}", 
            color=colors[cnt % len(colors)], 
            marker='o', 
            markeredgecolor='k'
        )
        if include_weak:
            if weak_data_dict and code_name in weak_data_dict.get("epsilons", {}):
                w_ps = sorted([p for p in weak_data_dict["epsilons"][code_name].keys() if p_range[0] <= p <= p_range[1]])
                w_eps_vals = [weak_data_dict["epsilons"][code_name][p] for p in w_ps if p_range[0] <= p <= p_range[1]]
                w_eps_errs = [weak_data_dict["std_epsilons"][code_name].get(p, 0) for p in w_ps if p_range[0] <= p <= p_range[1]]
                
                weak_line, weak_caps, weak_bars = ax.errorbar(
                    w_ps, w_eps_vals, yerr=w_eps_errs,
                    color=colors[cnt % len(colors)], marker='s', markeredgecolor='k',linestyle='None'
                )
                weak_line.set_alpha(0.5)  # Set transparency for weak decoder line 

        if include_strong:
            # 3. Plot the Strong Decoder Baseline (Dotted Line, Triangle Marker)
            if strong_data_dict and code_name in strong_data_dict.get("epsilons", {}):
                s_ps = sorted([p for p in strong_data_dict["epsilons"][code_name].keys() if p_range[0] <= p <= p_range[1]])
                s_eps_vals = [strong_data_dict["epsilons"][code_name][p] for p in s_ps if p_range[0] <= p <= p_range[1]]
                s_eps_errs = [strong_data_dict["std_epsilons"][code_name].get(p, 0) for p in s_ps if p_range[0] <= p <= p_range[1]]

                strong_line, strong_caps, strong_bars = ax.errorbar(
                    s_ps, s_eps_vals, yerr=s_eps_errs, 
                    color=colors[cnt % len(colors)], marker='^', markeredgecolor='k', linestyle='None'
                )
                strong_line.set_alpha(0.5)  # Set transparency for strong decoder line

        cnt += 1
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True) 
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"target $p_s =$ {target_switch_rate}")
    ax.set_xlabel(rf"physical error rate")
    ax.set_ylabel("LER per SEC")
    ax.set_ylim(bottom=1e-8)
    
    plt.tight_layout()
    plt.show()



def plot_decoder_switching_results_switch_rate(target_switch_rate, weak_decoder, strong_decoder, num_shots_max, data_dict=None, include_strong_and_weak=True):
    """
    Plots the results from the merged decoder switching data.
    """
    if data_dict is None:
        script_dir = Path(__file__).resolve().parent
        results_file = script_dir.parent / "data" / "decoder_switching_results" / f'decoder_switching_target_ps_{target_switch_rate}_weak_{weak_decoder}_strong_{strong_decoder}_max_shots_{num_shots_max}.txt'
        
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            return
        
        with open(results_file, 'r') as file:
            # eval handles tuple keys correctly for the dictionary
            data_dict = eval(file.read())

    if include_strong_and_weak: # right now we just want to plot the weak / switching comparison
        script_dir = Path(__file__).resolve().parent
        if weak_decoder == 'uf':
            weak_results_file = script_dir.parent / "data" / "raw" / "single_sliding_window_uf_max_shots_15000.txt"
        elif weak_decoder == 'bplsd':
            weak_results_file = script_dir.parent / "saved_data" / "single_sliding_window_bplsd_max_shots_30000.txt"

        if strong_decoder == 'relay_bp':
            strong_results_file = script_dir.parent / "saved_data" / "single_sliding_window_relay_bp_max_shots_20000.txt"
        elif strong_decoder == 'tesseract':
            strong_results_file = script_dir.parent / "data" / "raw" / "single_sliding_window_tesseract_max_shots_100000.txt"

        if weak_results_file and weak_results_file.exists():
            with open(weak_results_file, 'r') as file:
                weak_data_dict = eval(file.read())
        else:
            print(f"Weak decoder results file not found: {weak_results_file}")
            weak_data_dict = None

        if strong_results_file and strong_results_file.exists():
            with open(strong_results_file, 'r') as file:
                strong_data_dict = eval(file.read())
        else:
            print(f"Strong decoder results file not found: {strong_results_file}")
            strong_data_dict = None


    code_names = data_dict["codes"]
    ps = data_dict["ps"]
    eps_to_save = data_dict["epsilons"]
    errs_in_eps_to_save = data_dict["std_epsilons"]
    switch_rates = data_dict["switch_rates"]
    switch_yerr = data_dict["switch_rate_err"]
    
    # Grab parameters from dict if available, otherwise use function arguments
    weak_dec = data_dict.get("weak_decoder", weak_decoder)
    strong_dec = data_dict.get("strong_decoder", strong_decoder)
    ps_target = data_dict.get("target_switch_rate", target_switch_rate)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    
    #================= Plot ler from decoder switching ===============================================
    cnt = 0
    for code_name in code_names:
        eps_vals = [eps_to_save[code_name][p] for p in ps]
        eps_errs = [errs_in_eps_to_save[code_name][p] for p in ps]
        
        ax[0].errorbar(
            ps, 
            eps_vals, 
            yerr=eps_errs, 
            label=f"{code_name}, {weak_dec} to {strong_dec}", 
            color=colors[cnt % len(colors)], 
            marker='o', 
            markeredgecolor='k'
        )
        if include_strong_and_weak:
            if weak_data_dict and code_name in weak_data_dict.get("epsilons", {}):
                w_ps = sorted(list(weak_data_dict["epsilons"][code_name].keys()))
                w_eps_vals = [weak_data_dict["epsilons"][code_name][p] for p in w_ps]
                w_eps_errs = [weak_data_dict["std_epsilons"][code_name].get(p, 0) for p in w_ps]
                
                ax[0].errorbar(
                    w_ps, w_eps_vals, yerr=w_eps_errs, 
                    label=f"{code_name}, {weak_dec}", 
                    color=colors[cnt % len(colors)], marker='s', markeredgecolor='k', linestyle='--'
                )

            # 3. Plot the Strong Decoder Baseline (Dotted Line, Triangle Marker)
            if strong_data_dict and code_name in strong_data_dict.get("epsilons", {}):
                s_ps = sorted(list(strong_data_dict["epsilons"][code_name].keys()))
                s_eps_vals = [strong_data_dict["epsilons"][code_name][p] for p in s_ps]
                s_eps_errs = [strong_data_dict["std_epsilons"][code_name].get(p, 0) for p in s_ps]
                
                ax[0].errorbar(
                    s_ps, s_eps_vals, yerr=s_eps_errs, 
                    label=f"{code_name}, {strong_dec}", 
                    color=colors[cnt % len(colors)], marker='^', markeredgecolor='k', linestyle=':'
                )

        cnt += 1
        
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].legend(fontsize=9)
    ax[0].set_xlabel("physical error rate")
    ax[0].set_ylabel("LER per SEC")
    
    #================= Plot the switch rate ===============================================
    cnt = 0
    for code_name in code_names:
        # Tuple keys are preserved, so we can access them natively
        y = np.array([switch_rates[(code_name, p)] for p in ps])
        yerr = np.array([switch_yerr[(code_name, p)] for p in ps])
        
        ax[1].errorbar(
            ps, 
            y, 
            yerr=yerr, 
            marker='o', 
            color=colors[cnt % len(colors)], 
            label=f"{code_name}"
        )   
        cnt += 1
        
    ax[1].set_ylabel("$p_{switch}$")
    ax[1].set_xlabel("physical error rate")
    ax[1].set_title(f"for $p_s$: {ps_target}")
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_shots = 10_000_000
    shots_per_job = 500_000
    target_switch_rate = 1e-2
    weak_decoder = 'uf'
    strong_decoder = 'tesseract'

    # to run on the cluster / get data on cluster
    # get_ler_for_decoder_switching_dcc(num_shots=num_shots, shots_per_job=shots_per_job, target_switch_rate=target_switch_rate, weak_decoder=weak_decoder, strong_decoder=strong_decoder)

    # run this once you have stuff from the cluster, download by uncommenting below, comment the get_ler_for_decoder_switching_dcc line above, and run this script again
    merge_dcc_results(
        target_switch_rate=target_switch_rate, # Update with the switch rate you ran
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        num_shots_max=num_shots     # Update to your actual num_shots
    )

    # run this to plot the results from decoder switching
    plot_decoder_switching_results(
        target_switch_rate=target_switch_rate, # Update with the switch rate you ran
        weak_decoder=weak_decoder,
        strong_decoder=strong_decoder,
        num_shots_max=num_shots,     # Update to your actual num_shots
        include_strong=False,
        include_weak=True,
        p_range=(10**(-4), 10**(-3.5))  # Optional: specify a range of p values to plot
    )
    