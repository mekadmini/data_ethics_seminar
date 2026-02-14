import os
import time
import itertools
from datetime import datetime
import pandas as pd
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


def run_study():
    # Base configuration ensuring we use a small split for testing
    base_config = {
        "matrix_language": MatrixLanguage.ITALIAN,
        "batch_size": 10,
        "content_swaps": True,
        "func_swaps": True,
        "dataset_split": "train[:3]", # Small split for testing
        "iterations": 1 # Generate N responses per prompt
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


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_root = f"experiment_results/study_{timestamp}"
    os.makedirs(study_root, exist_ok=True)

    print(f"Starting Study at {study_root}...")
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

        # 1. Generate Prompts
        # Note: generate_prompts inside main.py creates the folder if it doesn't exist
        output_path = generate_prompts(config=config)
        
        # 2. Run Experiment (Generation + Eval)
        # We pass 'config' as the args argument so run_experiment sees 'iterations'
        df_eval = run_experiment(args=config, input_dir=output_path)
        
        # 3. Aggregate Results
        stats = aggregate_results(df_eval, prompt_col="csrt", output_dir=run_dir)
        stats['scenario'] = scenario['name']
        summary_results.append(stats)
        
        print(f"Scenario {scenario['name']} completed.")

    print(f"\n--- Study Completed. Summary of ASR ---")
    summary_df = pd.DataFrame(summary_results)
    
    # Find and print best config
    find_best_configuration(summary_df)
    
    summary_path = os.path.join(study_root, "final_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Global Summary saved to {summary_path}")

    print(f"\nFull Study Completed. Results in {study_root}")

if __name__ == "__main__":
    run_study()
