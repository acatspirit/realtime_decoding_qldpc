import pandas as pd
import glob
import os

def merge_results(results_folder, master_file):
    # 1. Configuration: Define column types
    keys = ['p', 'd', 'cutoff', 'code_type', 'basis']
    
    # Columns we need to sum up to do weighted averages
    val_cols = ['num_shots', 'num_switches']
    
    # Metadata columns to preserve (we'll just take the 'first' one found)
    meta_cols = [
        'cluster_metric', 'bplsd_bp_method', 'bplsd_lsd_method', 
        'bplsd_lsd_order', 'bplsd_max_iter', 'bplsd_switching_cutoff', 
        'strong_num_sets', 'strong_gamma0', 'strong_gamma_dist_interval', 
        'strong_relay_max_iter'
    ]

    # 2. Load all data
    data_frames = []
    
    # Load existing master data if it exists
    if os.path.exists(master_file) and os.path.getsize(master_file) > 0:
        data_frames.append(pd.read_csv(master_file))
        print(f"Loading existing data from {master_file}")

    # Load all new cluster files
    cluster_files = glob.glob(f"{results_folder}/*.csv")
    if not cluster_files and not data_frames:
        print("No data found to merge.")
        return
    
    for f in cluster_files:
        data_frames.append(pd.read_csv(f))
    
    df_all = pd.concat(data_frames, ignore_index=True)

    # 3. Calculate Total Errors for weighted LER
    # We do this before grouping so we can sum the absolute number of failures
    df_all['total_errors'] = df_all['LER'] * df_all['num_shots']

    # 4. Group and Aggregate
    # This is the "Reduce" step. 
    # - We group by the physical parameters (the keys)
    # - we SUM the shots and errors
    # - we take the FIRST value for the decoder settings (since they are constant)
    agg_logic = {col: 'sum' for col in val_cols + ['total_errors']}
    for col in meta_cols:
        if col in df_all.columns:
            agg_logic[col] = 'first'

    grouped = df_all.groupby(keys).agg(agg_logic).reset_index()

    # 5. Final Calculations
    grouped['LER'] = grouped['total_errors'] / grouped['num_shots']
    
    # Reorder columns to match your desired format
    final_cols = ['LER', 'cutoff'] + [c for col in [keys, val_cols, meta_cols] for c in col if c not in ['cutoff']]
    # Filter only columns that actually exist to avoid errors
    existing_final_cols = [c for c in final_cols if c in grouped.columns]
    
    updated_master = grouped[existing_final_cols]

    # 6. Save (Overwrite the file with the new combined total)
    # IMPORTANT: Do NOT use mode='a' here because updated_master already contains 
    # the old data plus the new data.
    updated_master.to_csv(master_file, index=False)
    
    print(f"Merge Complete!")
    print(f"Total files processed: {len(cluster_files)}")
    print(f"Total rows in master: {len(updated_master)}")

if __name__ == "__main__":
    merge_results("/Users/ariannameinking/Documents/Brown_Research/realtime_decoding_qldpc/simulation_scripts/decoder_switching_results", "bplsd_relaybp.csv")