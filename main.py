import argparse

import pandas as pd
from datasets import load_dataset

from experiment import generate_and_save_stream, evaluate_saved_results
from lib.custom_types import MatrixLanguage, EmbeddedLanguage
from prompt_generator import save_experiment, DatasetSwapper


def generate_prompts(config=None, use_google_api=None, dataset_split=None, translation_iterations=None, batch_size=None, output_dir=None):
    # --- 1. Configuration ---
    if config is None:
        # Default config if none provided (e.g. running from CLI without granular args)
        config = {
            # Ensure this matches your Enum definition or use string "en"
            "matrix_language": MatrixLanguage.ITALIAN,

            "embedded_languages": [
                EmbeddedLanguage.ARABIC,
                EmbeddedLanguage.GREEK,
                EmbeddedLanguage.JAPANESE
            ],

            # 40% of all valid words will be translated
            "swap_ratio": 0.9,

            # Of that 40%, how is the pie shared?
            # (Ensure keys match the Enums in 'embedded_languages')
            "language_weights": {
                EmbeddedLanguage.GREEK: 0.5,  # 50% chance if swapping
                EmbeddedLanguage.ARABIC: 0.3,  # 30% chance if swapping
                EmbeddedLanguage.SPANISH: 0.2  # 20% chance if swapping
            },

            "batch_size": 10,
            "content_swaps": True,
            "func_swaps": True,
            "dataset_split": "train[:20]",
            "experiment_name": "default_experiment",
            "use_google_api": False
        }

    # Override config with explicit kwargs if present
    if use_google_api is not None:
        config['use_google_api'] = use_google_api
    if dataset_split is not None:
        config['dataset_split'] = dataset_split
    if translation_iterations is not None:
        config['translation_iterations'] = translation_iterations
    if batch_size is not None:
        config['batch_size'] = batch_size
    if output_dir is not None:
        config['output_dir'] = output_dir

    # Determine unique output directory
    # If config allows specifying an output root, use it. Otherwise default.
    base_output_dir = config.get("output_dir", "experiment_results")
    experiment_name = config.get("experiment_name", "default")

    # We might want a subfolder per experiment if running multiple
    # output_dir = os.path.join(base_output_dir, experiment_name)
    # For now, let's just use the passed output_dir or default
    output_dir = base_output_dir

    # --- 2. Load Data ---
    print(f"Loading MultiJail dataset ({config['dataset_split']})...")
    dataset = load_dataset("DAMO-NLP-SG/MultiJail", split=config['dataset_split'])
    multijail = dataset.to_pandas()

    # --- 3. Prepare DataFrame ---
    # We will expand the dataframe to have multiple rows per prompt if iterations > 1
    iterations = config.get("translation_iterations", 1)
    print(f"Generating {iterations} variations per prompt...")

    all_results = []

    source_col = config["matrix_language"]
    processor = DatasetSwapper(source_col)

    # Pre-check logic
    use_google = config.get('use_google_api', False)
    if not use_google:
        try:
            import argostranslate.package
        except ImportError:
            print("Wrapper: argostranslate not found!")
    else:
        print("Using Google Translate API (Online)...")

    for i in range(iterations):
        print(f"--- Translation Iteration {i + 1}/{iterations} ---")

        # Create a temp copy for this iteration
        temp_df = pd.DataFrame()
        temp_df['id'] = multijail['id']
        temp_df[source_col] = multijail[source_col]

        # Apply Code Switching
        switched_texts = processor.process_dataframe(
            df=temp_df,
            text_column=source_col,
            embedded_langs=config["embedded_languages"],
            swap_ratio=config["swap_ratio"],
            language_weights=config["language_weights"],
            batch_size=config["batch_size"],
            content_swaps=config.get("content_swaps", True),
            func_swaps=config.get("func_swaps", True),
            use_google_api=use_google  # <--- Pass flag
        )

        temp_df['csrt'] = switched_texts
        temp_df['translation_iter'] = i + 1

        all_results.append(temp_df)

    # Combine all iterations
    csrt = pd.concat(all_results, ignore_index=True)

    # Sort by ID to keep variations together
    csrt = csrt.sort_values(by=['id', 'translation_iter'])

    # --- 5. Save Results & Parameters ---
    save_experiment(csrt, config, output_dir=output_dir)

    # Verify
    print("\nSample Output:")
    print(csrt[[source_col, 'csrt', 'translation_iter']].head(3))

    return output_dir


def run_experiment(
    input_dir="experiment_results",
    iterations=1,
    n_repeat=1,
    max_workers=4,
    target_model="llama3",
    judge_model="llama-guard3"
):
    INPUT_FILE = f"{input_dir}/prompts.csv"
    INTERMEDIATE_FILE = f"{input_dir}/results_raw.csv"
    FINAL_FILE = f"{input_dir}/results_evaluated.csv"

    PROMPT_COL = "csrt"

    print(f"Running experiment in: {input_dir}")
    print(f"Target: {target_model}, Judge: {judge_model}")

    # 1. Run Generation (Saves progressively)
    generate_and_save_stream(
        input_csv=INPUT_FILE,
        output_csv=INTERMEDIATE_FILE,
        prompt_col=PROMPT_COL,
        model_name=target_model,
        iterations=iterations,
        max_workers=max_workers,
        n_repeat=n_repeat
    )

    # 2. Run Evaluation (Reads the saved file)
    df_eval = evaluate_saved_results(
        input_csv=INTERMEDIATE_FILE,
        output_csv=FINAL_FILE,
        prompt_col=PROMPT_COL,
        target_model=target_model,
        judge_model=judge_model,
        max_workers=max_workers
    )

    return df_eval


def handle_generate(args):
    generate_prompts(
        use_google_api=args.use_google,
        dataset_split=args.dataset_split,
        translation_iterations=args.translation_iterations,
        batch_size=args.batch_size
    )

def handle_experiment(args):
    run_experiment(
        input_dir=args.input_dir,
        iterations=args.iterations,
        n_repeat=args.n_repeat,
        max_workers=args.max_workers,
        target_model=args.target_model,
        judge_model=args.judge_model
    )

def main():
    parser = argparse.ArgumentParser(description="LeakyGPT clt")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Command 1: generate-prompts ---
    parser_greet = subparsers.add_parser('generate', help='Generate code-switched prompts')
    parser_greet.add_argument('--use_google', action='store_true',
                              help='Use Google Translate API instead of local Argos')
    parser_greet.add_argument('--dataset_split', type=str, help='Dataset split (e.g. train[:10])')
    parser_greet.add_argument('--translation_iterations', type=int, help='Number of translation variations per prompt')
    parser_greet.add_argument('--batch_size', type=int, help='Batch size for processing')
    parser_greet.set_defaults(func=handle_generate)

    # --- Command 2: experiment ---
    parser_calc = subparsers.add_parser('experiment', help='Feed the code-switched prompts to the llm')
    parser_calc.add_argument('--input_dir', type=str, default='experiment_results',
                             help='Directory containing prompts.csv')
    parser_calc.add_argument('--iterations', type=int, default=1, help='Number of iterations per prompt')
    parser_calc.add_argument('--n_repeat', type=int, default=1, help='Number of times to repeat the prompt')
    parser_calc.add_argument('--max_workers', type=int, default=4, help='Number of parallel threads')
    parser_calc.add_argument('--target_model', type=str, default='llama3', help='Target Ollama model')
    parser_calc.add_argument('--judge_model', type=str, default='llama-guard3', help='Safety judge Ollama model')
    parser_calc.set_defaults(func=handle_experiment)

    # --- Command 3:  ---
    # parser_calc = subparsers.add_parser('eval', help='Evaluate the llm answers')
    # parser_calc.set_defaults(func=handle_calc)

    # 3. Parse arguments
    args = parser.parse_args()

    # 4. Execute the appropriate function
    # Because we used set_defaults(func=...), args.func holds the function to call
    args.func(args)


if __name__ == '__main__':
    main()
