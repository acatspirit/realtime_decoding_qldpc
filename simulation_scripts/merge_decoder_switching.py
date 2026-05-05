import pandas as pd
import glob
import os

def merge_results(results_folder, master_file):
    # 1. Identifying Keys
    # These are the columns that must match for rows to be combined.
    keys = ['p', 'd', 'cutoff', 'code_type', 'basis']
    
    # Columns to sum up (to calculate weighted averages later)
    val_cols = ['num_shots', 'num_switches']
    
    # Metadata columns to preserve (taking the 'first' one found for each group)
    meta_cols = [
        'cluster_metric', 'bplsd_bp_method', 'bplsd_lsd_method', 
        'bplsd_lsd_order', 'bplsd_max_iter', 'bplsd_switching_cutoff', 
        'strong_num_sets', 'strong_gamma0', 'strong_gamma_dist_interval', 
        'strong_relay_max_iter'
    ]

    data_frames = []
    
    # 2. Load existing master data if it exists
    if os.path.exists(master_file) and os.path.getsize(master_file) > 0:
        data_frames.append(pd.read_csv(master_file))
        print(f"Loading existing master data: {master_file}")

    # Load all new batch files from the cluster
    cluster_files = glob.glob(f"{results_folder}/*.csv")
    if not cluster_files:
        if not data_frames:
            print("No data found to process.")
            return
        print("No new batch files found. Master file is already up to date.")
        return
    
    for f in cluster_files:
        data_frames.append(pd.read_csv(f))
    
    # Combine everything into one big pool
    df_all = pd.concat(data_frames, ignore_index=True)

    # 3. Prepare for Weighted Average
    # We calculate the absolute number of failures (Total Errors = LER * Shots)
    # This allows us to sum errors across batches of different sizes.
    df_all['total_errors'] = df_all['LER'] * df_all['num_shots']

    # 4. Group and Aggregate (The "Weighted Average" Step)
    # This collapses all rows where the 'keys' match.
    agg_logic = {col: 'sum' for col in val_cols + ['total_errors']}
    for col in meta_cols:
        if col in df_all.columns:
            agg_logic[col] = 'first'

    grouped = df_all.groupby(keys).agg(agg_logic).reset_index()

    # 5. Final Weighted LER Calculation
    # LER = (Sum of all errors) / (Sum of all shots)
    grouped['LER'] = grouped['total_errors'] / grouped['num_shots']
    
    # Organize columns for readability
    final_cols = ['LER', 'cutoff'] + [c for col in [keys, val_cols, meta_cols] for c in col if c not in ['cutoff']]
    existing_final_cols = [c for c in final_cols if c in grouped.columns]
    
    updated_master = grouped[existing_final_cols]

    # 6. Save the Updated Master (Overwriting with the new aggregated totals)
    updated_master.to_csv(master_file, index=False)
    
    # 7. Cleanup (The Safety Step)
    # Instead of moving to an archive, we delete the files. 
    # This ensures that the next time you run this script, you don't 
    # add these same shots to the master file again.
    for f in cluster_files:
        os.remove(f)
    
    print(f"Merge Complete!")
    print(f"Processed and deleted {len(cluster_files)} new batch files.")
    print(f"Total shots now in master for first row: {updated_master.iloc[0]['num_shots']}")

if __name__ == "__main__":
    # Update the path to your results directory
    outputs_path = "simulation_scripts/decoder_switching_results."
    combined_results = "data/raw/bplsd_relaybp.csv"

    merge_results(outputs_path, combined_results)