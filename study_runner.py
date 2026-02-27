import itertools
import json
import os
from datetime import datetime

from lib.custom_types import MatrixLanguage, EmbeddedLanguage
from main import generate_prompts, run_experiment


def run_study(use_google=False, max_workers=4, resume_from=None, target_model="llama3", judge_model="llama-guard3",
              prompts_only=False, n_repeat=1):
    # Base configuration ensuring we use a small split for testing
    base_config = {
        "matrix_language": MatrixLanguage.ITALIAN,
        "batch_size": 10,
        "content_swaps": True,
        "func_swaps": True,
        "dataset_split": "train[:30]",  # Small split for testing
        "iterations": 5,  # Attack iterations (N responses per prompt)
        "translation_iterations": 5,  # Translation iterations (N variations per source prompt)
        "use_google_api": use_google,
        "max_workers": max_workers,
        "target_model": target_model,
        "judge_model": judge_model,
        "n_repeat": n_repeat
    }

    # --- Define Parameter Grid ---
    swap_ratios = [0.8, 0.6, 0.3, 0,1]

    # Matrix languages to test
    matrix_languages = [MatrixLanguage.ENGLISH, MatrixLanguage.ITALIAN]

    # Embedded languages pool
    embedded_pool = [EmbeddedLanguage.ARABIC, EmbeddedLanguage.GREEK, EmbeddedLanguage.SPANISH,
                     EmbeddedLanguage.JAPANESE]

    # Define language dominance strategies (Dynamic)
    strategies = {}

    # 1. Balanced: Equal weights (0.25 each)
    strategies["Balanced"] = {lang: 0.25 for lang in embedded_pool}

    # 2. Dominant: One language gets 0.7, others split the remaining 0.3 (0.1 each)
    for lang in embedded_pool:
        # Get dominance name (e.g. "Arabic_Dom")
        lang_name = lang.name.capitalize() if hasattr(lang, 'name') else str(lang).capitalize()
        if lang == EmbeddedLanguage.ARABIC:
            lang_name = "Arabic"  # Manual overrides if needed
        elif lang == EmbeddedLanguage.GREEK:
            lang_name = "Greek"
        elif lang == EmbeddedLanguage.SPANISH:
            lang_name = "Spanish"
        elif lang == EmbeddedLanguage.JAPANESE:
            lang_name = "Japanese"
        elif lang == EmbeddedLanguage.ITALIAN:
            lang_name = "Italian"

        strat_key = f"{lang_name}_Dom"

        weights = {}
        for l in embedded_pool:
            if l == lang:
                weights[l] = 0.7
            else:
                weights[l] = 0.1
        strategies[strat_key] = weights

    # Define filter configurations
    filter_configs = [
        ("Both", {"content_swaps": True, "func_swaps": True}),
        ("FuncOnly", {"content_swaps": False, "func_swaps": True}),
        ("ContentOnly", {"content_swaps": True, "func_swaps": False})
    ]

    study_scenarios = []

    # Generate all combinations
    # Loop order: Matrix -> Ratio -> Strategy -> Filter
    for matrix_lang in matrix_languages:
        mat_code = matrix_lang.value if hasattr(matrix_lang, 'value') else str(matrix_lang)
        mat_prefix = mat_code.capitalize()  # "It", "En"

        for ratio, (strat_name, weights), (filter_name, filter_settings) in itertools.product(swap_ratios,
                                                                                              strategies.items(),
                                                                                              filter_configs):
            # Scenario naming: It_Ratio_0.9_Arabic_Dom_Both
            scenario_name = f"{mat_prefix}_Ratio_{ratio}_{strat_name}_{filter_name}"

            study_scenarios.append({
                "name": scenario_name,
                "matrix_language": matrix_lang,
                "swap_ratio": ratio,
                "embedded_languages": embedded_pool,
                "language_weights": weights,
                **filter_settings
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

    for i, scenario in enumerate(study_scenarios):
        print(f"\n--- Running Scenario {i + 1}/{len(study_scenarios)}: {scenario['name']} ---")

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
                config['n_repeat'] = n_repeat

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
        run_experiment(
            input_dir=output_path,
            iterations=config.get('iterations', 1),
            n_repeat=config.get('n_repeat', 1),
            max_workers=config.get('max_workers', 4),
            target_model=config.get('target_model', 'llama3'),
            judge_model=config.get('judge_model', 'llama-guard3')
        )

        print(f"Scenario {scenario['name']} completed.")

    print(f"\n✅ Full Study Completed! Raw results are inside: {study_root}")
    print("👉 To consolidate results into one dataset, run: python consolidate_results.py")
    print("👉 To compute mathematical analysis metrics, run: python compute_metrics.py\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_google', action='store_true', help='Use Google Translate API')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel threads for LLM generation')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Directory to resume the study from (e.g., experiment_results/study_2024...)')
    parser.add_argument('--target_model', type=str, default='llama3', help='Ollama model to attack (default: llama3)')
    parser.add_argument('--judge_model', type=str, default='llama-guard3',
                        help='Ollama model for safety evaluation (default: llama-guard3)')
    parser.add_argument('--prompts_only', action='store_true',
                        help='Only generate prompts and config, do not run the experiment')
    parser.add_argument('--n_repeat', type=int, default=1, help='Number of times to repeat the prompt')
    args = parser.parse_args()

    run_study(
        use_google=args.use_google,
        max_workers=args.max_workers,
        resume_from=args.resume_from,
        target_model=args.target_model,
        judge_model=args.judge_model,
        prompts_only=args.prompts_only,
        n_repeat=args.n_repeat
    )
