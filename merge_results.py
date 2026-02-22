import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

def merge_experiment_results(base_path):
    base_dir = Path(base_path)
    experiment_results_dir = base_dir / "experiment_results"
    
    if not experiment_results_dir.exists():
        print(f"Error: {experiment_results_dir} does not exist.")
        return

    # Group subfolders by name across all study folders
    config_groups = defaultdict(list)
    
    study_folders = [f for f in experiment_results_dir.iterdir() if f.is_dir() and f.name.startswith("study_")]
    
    for study_folder in study_folders:
        subfolders = [f for f in study_folder.iterdir() if f.is_dir()]
        for subfolder in subfolders:
            config_groups[subfolder.name].append(subfolder)

    all_results = []

    for config_name, folders in config_groups.items():
        print(f"Processing configuration: {config_name}")
        
        prompts_dfs = []
        results_dfs = []
        
        for folder in folders:
            prompts_file = folder / "prompts.csv"
            results_file = folder / "results_evaluated.csv"
            
            if prompts_file.exists():
                df = pd.read_csv(prompts_file)
                # Rename language column to 'prompt'
                # Language column is the one that is not 'csrt', 'id', or 'translation_iter'
                lang_cols = [col for col in df.columns if col not in ['csrt', 'id', 'translation_iter']]
                if lang_cols:
                    df = df.rename(columns={lang_cols[0]: 'prompt'})
                prompts_dfs.append(df)
                
            if results_file.exists():
                df = pd.read_csv(results_file)
                # Rename response column
                response_cols = [col for col in df.columns if col.startswith("response_")]
                if response_cols:
                    df = df.rename(columns={response_cols[0]: 'response'})
                results_dfs.append(df)
        
        if not prompts_dfs or not results_dfs:
            print(f"Skipping {config_name} due to missing files.")
            continue
            
        # Merge prompts
        merged_prompts = pd.concat(prompts_dfs, ignore_index=True)
        # Drop id and translation_iter
        cols_to_drop = [col for col in ["id", "translation_iter"] if col in merged_prompts.columns]
        merged_prompts = merged_prompts.drop(columns=cols_to_drop)
        
        # Drop duplicates
        merged_prompts = merged_prompts.drop_duplicates()
        
        # Merge results_evaluated
        merged_results = pd.concat(results_dfs, ignore_index=True)
            
        # Join
        joined_df = pd.merge(merged_results, merged_prompts, on="csrt", how="left")
        
        # Add configuration column
        joined_df["configuration"] = config_name
        
        # Drop iteration column
        if "iteration" in joined_df.columns:
            joined_df = joined_df.drop(columns=["iteration"])
            
        all_results.append(joined_df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_file = experiment_results_dir / "merged_results_evaluated.csv"
        final_df.to_csv(output_file, index=False)
        print(f"Successfully saved merged results to {output_file}")
    else:
        print("No results found to merge.")

if __name__ == "__main__":
    base_path = r"c:\Users\moham\Documents\Master Data Science\data_ethics_seminar"
    merge_experiment_results(base_path)
