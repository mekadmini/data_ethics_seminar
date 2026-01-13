import csv
import os

import ollama
import pandas as pd
from tqdm import tqdm


class OllamaHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def chat(self, messages: list) -> str:
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"[ERROR] {str(e)}"


def generate_and_save_stream(input_csv, output_csv, prompt_col, model_name):
    """
    Stage 1: Reads prompts, generates answers, and appends to CSV immediately.
    """
    print(f"🚀 STAGE 1: Generating with {model_name}...")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file {input_csv} not found.")

    df = pd.read_csv(input_csv)
    handler = OllamaHandler(model_name)

    # Check if output exists to determine if we need to write headers
    file_exists = os.path.exists(output_csv)

    # We open the file in 'append' mode ('a') with buffering=1 (line buffering)
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        # Define columns: Original Prompt + New Response
        fieldnames = [prompt_col, f"response_{model_name}"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        # If file existed, we might want to skip rows already done.
        # (Optional logic: read existing output_csv and calculate start_index)
        # For simplicity here, we assume a fresh start or you manually manage the input df.

        print(f"💾 Saving to: {output_csv}")

        for text in tqdm(df[prompt_col], desc="Generating"):
            text_str = str(text) if not pd.isna(text) else ""

            # Generate
            if text_str.strip():
                response = handler.chat([{'role': 'user', 'content': text_str}])
            else:
                response = ""

            # Write IMMEDIATELY to disk
            writer.writerow({
                prompt_col: text_str,
                f"response_{model_name}": response
            })

            # Force write to disk (ensures data is safe even if script is killed 1ms later)
            f.flush()
            os.fsync(f.fileno())

    print("✅ Stage 1 Complete.")


def evaluate_saved_results(input_csv, output_csv, prompt_col, target_model, judge_model):
    """
    Stage 2: Reads the raw results and adds safety classification.
    """
    print(f"\n🕵️ STAGE 2: Evaluating with {judge_model}...")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File {input_csv} not found. Run Stage 1 first.")

    df = pd.read_csv(input_csv)
    judge = OllamaHandler(judge_model)

    target_response_col = f"response_{target_model}"

    safety_statuses = []
    judge_explanations = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        prompt = str(row[prompt_col])
        response = str(row[target_response_col])

        if response == "[ERROR]" or not response.strip():
            safety_statuses.append("error")
            judge_explanations.append("No content")
            continue

        # Llama Guard Prompt Structure
        messages = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
        ]

        judge_raw = judge.chat(messages)

        # Parse Llama Guard (Usually outputs "safe" or "unsafe\nS1")
        clean_out = judge_raw.strip()
        is_safe = "safe" if clean_out.lower().startswith("safe") else "unsafe"

        safety_statuses.append(is_safe)
        judge_explanations.append(clean_out)

    # Add columns
    df['safety_status'] = safety_statuses
    df['judge_raw'] = judge_explanations

    # Save final
    df.to_csv(output_csv, index=False)
    print(f"✅ Stage 2 Complete. Final results at: {output_csv}")
    print("\n--- Summary ---")
    print(df['safety_status'].value_counts())


if __name__ == "__main__":
    # --- Configuration ---
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
