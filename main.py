import pandas as pd
from datasets import load_dataset
import argparse
from experiment import generate_and_save_stream, evaluate_saved_results
from lib.custom_types import MatrixLanguage, EmbeddedLanguage
from prompt_generator import save_experiment, DatasetSwapper


def generate_prompts(args):
    # --- 1. Configuration ---
    config = {
        # Ensure this matches your Enum definition or use string "en"
        "matrix_language": MatrixLanguage.ITALIAN,

        "embedded_languages": [
            EmbeddedLanguage.ARABIC,
            EmbeddedLanguage.GREEK,
            EmbeddedLanguage.SPANISH
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
        "dataset_split": "train[:20]"
    }

    # --- 2. Load Data ---
    print(f"Loading MultiJail dataset ({config['dataset_split']})...")
    dataset = load_dataset("DAMO-NLP-SG/MultiJail", split=config['dataset_split'])
    multijail = dataset.to_pandas()

    # --- 3. Prepare DataFrame ---
    csrt = pd.DataFrame()
    csrt['id'] = multijail['id']
    source_col = config["matrix_language"]
    csrt[source_col] = multijail[source_col]

    # --- 4. Processing ---
    processor = DatasetSwapper(source_col)

    print(f"Starting Code-Switching for matrix language: {source_col}...")

    switched_texts = processor.process_dataframe(
        df=csrt,
        text_column=source_col,
        embedded_langs=config["embedded_languages"],
        swap_ratio=config["swap_ratio"],
        language_weights=config["language_weights"],  # Passing the weights
        batch_size=config["batch_size"]
    )

    csrt['csrt'] = switched_texts

    # --- 5. Save Results & Parameters ---
    save_experiment(csrt, config, output_dir="experiment_results")

    # Verify
    print("\nSample Output:")
    print(csrt[[source_col, 'csrt']].head(3))


def run_experiment(args):
    INPUT_FILE = "experiment_results/prompts.csv"
    INTERMEDIATE_FILE = "experiment_results/results_raw.csv"
    FINAL_FILE = "experiment_results/results_evaluated.csv"

    PROMPT_COL = "csrt"
    TARGET_MODEL = "llama3"
    JUDGE_MODEL = "llama-guard3"

    # 1. Run Generation (Saves progressively)
    generate_and_save_stream(
        input_csv=INPUT_FILE,
        output_csv=INTERMEDIATE_FILE,
        prompt_col=PROMPT_COL,
        model_name=TARGET_MODEL
    )

    # 2. Run Evaluation (Reads the saved file)
    evaluate_saved_results(
        input_csv=INTERMEDIATE_FILE,
        output_csv=FINAL_FILE,
        prompt_col=PROMPT_COL,
        target_model=TARGET_MODEL,
        judge_model=JUDGE_MODEL
    )


def main():
    parser = argparse.ArgumentParser(description="LeakyGPT clt")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Command 1: generate-prompts ---
    parser_greet = subparsers.add_parser('generate', help='Generate code-switched prompts')
    parser_greet.set_defaults(func=generate_prompts)

    # --- Command 2:  ---
    parser_calc = subparsers.add_parser('experiment', help='Feed the code-switched prompts to the llm')
    parser_calc.set_defaults(func=run_experiment)

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