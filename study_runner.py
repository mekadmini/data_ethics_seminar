import os
import time
import itertools
from datetime import datetime
import pandas as pd
import json
from lib.custom_types import MatrixLanguage, EmbeddedLanguage
from main import generate_prompts, run_experiment
from experiment import aggregate_results

def find_best_configuration(summary_df):
    """
    Analyzes the summary dataframe to find the configuration with the highest ASR.
    """
    if summary_df.empty:
        print("No results to analyze.")
        return

    # Sort by ASR (Contextual) descending
    sorted_df = summary_df.sort_values(by="asr_both", ascending=False)
    
    print("\n🏆 --- Best Configurations --- 🏆")
    print(sorted_df[['scenario', 'asr_both', 'asr_response', 'at_least_one_success', 'consistent_success']].head(5))

    best = sorted_df.iloc[0]
    print(f"\n✅ Winning Configuration: {best['scenario']}")
    print(f"   ASR (Contextual): {best['asr_both']:.2f}%")
    print(f"   Consistent Success: {best['consistent_success']:.2f}%")


def run_study(use_google=False, max_workers=4, resume_from=None, target_model="llama3", judge_model="llama-guard3", prompts_only=False):
    # Base configuration ensuring we use a small split for testing
    base_config = {
        "matrix_language": MatrixLanguage.ITALIAN,
        "batch_size": 10,
        "content_swaps": True,
        "func_swaps": True,
        "dataset_split": "train[:3]", # Small split for testing
        "iterations": 10, # Attack iterations (N responses per prompt)
        "translation_iterations": 10, # Translation iterations (N variations per source prompt)
        "use_google_api": use_google,
        "max_workers": max_workers,
        "target_model": target_model,
        "judge_model": judge_model
    }

    # --- Define Parameter Grid ---
    #swap_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    swap_ratios = [0.9]
    
    # Define language dominance strategies
    strategies = {
        "Arabic_Dom": {
            EmbeddedLanguage.ARABIC: 0.8,
            EmbeddedLanguage.GREEK: 0.1,
            EmbeddedLanguage.SPANISH: 0.1
        },
        "Greek_Dom": {
            EmbeddedLanguage.ARABIC: 0.1,
            EmbeddedLanguage.GREEK: 0.8,
            EmbeddedLanguage.SPANISH: 0.1
        },
        "Spanish_Dom": {
            EmbeddedLanguage.ARABIC: 0.1,
            EmbeddedLanguage.GREEK: 0.1,
            EmbeddedLanguage.SPANISH: 0.8
        },
        "Balanced": {
            EmbeddedLanguage.ARABIC: 1/3,
            EmbeddedLanguage.GREEK: 1/3,
            EmbeddedLanguage.SPANISH: 1/3
        }
    }
    
    study_scenarios = []
    
    # Generate all combinations
    for ratio, (strat_name, weights) in itertools.product(swap_ratios, strategies.items()):
        scenario_name = f"Ratio_{ratio}_{strat_name}"
        study_scenarios.append({
            "name": scenario_name,
            "swap_ratio": ratio,
            "embedded_languages": [EmbeddedLanguage.ARABIC, EmbeddedLanguage.GREEK, EmbeddedLanguage.SPANISH],
            "language_weights": weights
        })

    # --- Determine Output Directory ---
    if resume_from:
        study_root = resume_from
        if not os.path.exists(study_root):
            raise FileNotFoundError(f"Resume directory not found: {study_root}")
        print(f"🔄 Resuming Study from {study_root}...")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_root = f"experiment_results/study_{timestamp}"
        os.makedirs(study_root, exist_ok=True)
        print(f"Starting New Study at {study_root}...")

    print(f"Total Scenarios to run: {len(study_scenarios)}")

    summary_results = []

    for i, scenario in enumerate(study_scenarios):
        print(f"\n--- Running Scenario {i+1}/{len(study_scenarios)}: {scenario['name']} ---")
        
        # Merge base config with scenario
        config = base_config.copy()
        config.update(scenario)
        
        # Create output directory for this specific run
        run_dir = os.path.join(study_root, scenario['name'])
        config['output_dir'] = run_dir
        config['experiment_name'] = scenario['name']
        
        # Ensure 'input_dir' is set for run_experiment
        config['input_dir'] = run_dir

        # 1. Generate Prompts OR Reuse Existing
        # Note: generate_prompts inside main.py creates the folder if it doesn't exist
        prompts_path = os.path.join(run_dir, "prompts.csv")
        config_path = os.path.join(run_dir, "config.json")
        
        if os.path.exists(config_path):
             print(f"🔄 Found existing config.json in {run_dir}. Loading...")
             try:
                 with open(config_path, 'r') as f:
                     loaded_config = json.load(f)
                 
                 # Logic Fix:
                 # 1. Start with 'config' (base + scenario)
                 # 2. Update with 'loaded_config' (so file settings like 'iterations' override defaults)
                 config.update(loaded_config)
                 
                 # 3. Enforce CLI overrides specifically
                 config['target_model'] = target_model
                 config['judge_model'] = judge_model
                 config['max_workers'] = max_workers
                 config['use_google_api'] = use_google
                 
             except Exception as e:
                 print(f"⚠️ Failed to load config.json: {e}. Using generated config.")

        if os.path.exists(prompts_path):
             print(f"🔄 Found existing prompts.csv in {run_dir}. Reusing to ensure consistency.")
             output_path = run_dir
        else:
             output_path = generate_prompts(config=config)
        
        if prompts_only:
            print(f"⏭️ Prompts generated/verified for {scenario['name']}. Skipping experiment (Prompts Only mode).")
            continue

        # 2. Run Experiment (Generation + Eval)
        # We pass 'config' as the args argument so run_experiment sees 'iterations'
        df_eval = run_experiment(args=config, input_dir=output_path)
        
        # 3. Aggregate Results
        stats = aggregate_results(df_eval, prompt_col="csrt", output_dir=run_dir)
        stats['scenario'] = scenario['name']
        summary_results.append(stats)
        
        print(f"Scenario {scenario['name']} completed.")

    if not summary_results:
        print("\n--- No experiments run (Prompts Only or no results). Exiting. ---")
        return

    print(f"\n--- Study Completed. Summary of ASR ---")
    summary_df = pd.DataFrame(summary_results)
    
    # Find and print best config
    find_best_configuration(summary_df)
    
    summary_path = os.path.join(study_root, "final_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Global Summary saved to {summary_path}")

    print(f"\nFull Study Completed. Results in {study_root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_google', action='store_true', help='Use Google Translate API')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel threads for LLM generation')
    parser.add_argument('--resume_from', type=str, default=None, help='Directory to resume the study from (e.g., experiment_results/study_2024...)')
    parser.add_argument('--target_model', type=str, default='llama3', help='Ollama model to attack (default: llama3)')
    parser.add_argument('--judge_model', type=str, default='llama-guard3', help='Ollama model for safety evaluation (default: llama-guard3)')
    parser.add_argument('--prompts_only', action='store_true', help='Only generate prompts and config, do not run the experiment')
    args = parser.parse_args()
    
    run_study(
        use_google=args.use_google, 
        max_workers=args.max_workers, 
        resume_from=args.resume_from,
        target_model=args.target_model,
        judge_model=args.judge_model,
        prompts_only=args.prompts_only
    )
