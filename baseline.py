import json
import os
from datetime import datetime

import pandas as pd
from datasets import load_dataset

from main import run_experiment
from lib.custom_types import MatrixLanguage


def run_baseline(max_workers=4, target_model="llama3", judge_model="llama-guard3", n_repeat=1, dataset_split="train[:30]", resume_from=None):
    """
    Runs a clean baseline evaluation by extracting original prompts
    for all available Matrix Languages and directly passing them to the LLM 
    and the judge, completely bypassing the translation and POS tagging pipeline.
    """
    
    # 1. Setup paths
    if resume_from:
        study_root = resume_from
        if not os.path.exists(study_root):
            raise FileNotFoundError(f"Resume directory not found: {study_root}")
        print(f"🔄 Resuming Clean Baseline Study from {study_root}...")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_root = f"experiment_results/baseline_{timestamp}"
        os.makedirs(study_root, exist_ok=True)
        print(f"Starting Clean Baseline Study at {study_root}...")

    print(f"Loading MultiJail dataset ({dataset_split})...")
    dataset = load_dataset("DAMO-NLP-SG/MultiJail", split=dataset_split)
    multijail = dataset.to_pandas()

    # Get all matrix languages dynamically from the Enum-like class
    languages = [MatrixLanguage.ENGLISH, MatrixLanguage.ITALIAN]

    for lang in languages:
        if lang not in multijail.columns:
            print(f"⚠️ Skipping language '{lang}': Not found in MultiJail dataset columns.")
            continue

        print(f"\n{'='*50}")
        print(f"🚀 Running Baseline for Matrix Language: {lang.upper()}")
        print(f"{'='*50}")

        run_dir = os.path.join(study_root, f"baseline_{lang}")
        os.makedirs(run_dir, exist_ok=True)

        prompts_path = os.path.join(run_dir, "prompts.csv")
        config_path = os.path.join(run_dir, "config.json")

        # 2. Save pseudo-config for tracking
        config = {
            "experiment_name": f"baseline_{lang}",
            "dataset_split": dataset_split,
            "iterations": 5, # Standard number of attack attempts to get ASR
            "translation_iterations": 1,
            "max_workers": max_workers,
            "target_model": target_model,
            "judge_model": judge_model,
            "n_repeat": n_repeat,
            "matrix_language": lang,
            "swap_ratio": 0.0,
            "embedded_languages": [],
            "content_swaps": False,
            "func_swaps": False
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # 3. Generate Prompts (Direct Copy of 'matrix_language' from dataset)
        csrt = pd.DataFrame()
        csrt['id'] = multijail['id']
        csrt[lang] = multijail[lang]
        
        # The pipeline execution logic looks for the 'csrt' column as the prompt to feed the LLM
        csrt['csrt'] = multijail[lang]  
        
        # The pipeline expects an 'iteration' column to differentiate translation variants
        csrt['iteration'] = 1 

        if not os.path.exists(prompts_path):
            csrt.to_csv(prompts_path, index=False)
            print(f"✅ Baseline prompts for '{lang}' saved to: {prompts_path}")
        else:
            print(f"🔄 Found existing prompts.csv for '{lang}'. Reusing.")

        # 4. Run Experiment (Generation + Eval)
        run_experiment(
            input_dir=run_dir,
            iterations=config['iterations'],
            n_repeat=config['n_repeat'],
            max_workers=config['max_workers'],
            target_model=config['target_model'],
            judge_model=config['judge_model']
        )

    print(f"\n✅ Clean Baseline Study Completed! Results are inside: {study_root}")
    print("👉 Next, you can run `python consolidate_results.py` to add these baseline metrics.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a clean baseline test for all matrix languages without translations.")
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel threads for LLM generation')
    parser.add_argument('--target_model', type=str, default='llama3', help='Ollama model to attack (default: llama3)')
    parser.add_argument('--judge_model', type=str, default='llama-guard3', help='Ollama model for safety evaluation (default: llama-guard3)')
    parser.add_argument('--n_repeat', type=int, default=1, help='Number of times to repeat the prompt')
    parser.add_argument('--dataset_split', type=str, default='train[:30]', help='Dataset split to use (e.g. train[:315])')
    parser.add_argument('--resume_from', type=str, default=None, help='Directory to resume the study from (e.g., experiment_results/baseline_2024...)')
    args = parser.parse_args()

    run_baseline(
        max_workers=args.max_workers,
        target_model=args.target_model,
        judge_model=args.judge_model,
        n_repeat=args.n_repeat,
        dataset_split=args.dataset_split,
        resume_from=args.resume_from
    )
