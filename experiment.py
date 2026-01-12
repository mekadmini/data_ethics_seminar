import os
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm


class OllamaRunner:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

        # Simple health check
        try:
            requests.get(base_url)
            print(f"✅ Connected to Ollama at {base_url}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to Ollama at {base_url}. Is it running?")
            raise

    def generate(self, prompt: str) -> Optional[str]:
        """
        Sends a single prompt to Ollama and returns the response.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False  # efficient for scripts, waits for full completion
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()

            # Parse the JSON response
            data = response.json()
            return data.get("response", "")

        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return None


def run_inference(input_csv: str,
                  prompt_col: str,
                  model_name: str,
                  output_csv: str = None):
    # 1. Load Data
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file {input_csv} not found.")

    df = pd.read_csv(input_csv)

    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found in CSV.")

    print(f"Loaded {len(df)} rows. Model: {model_name}")

    # 2. Initialize Ollama
    runner = OllamaRunner(model_name)

    # 3. Inference Loop
    # We use a list to store results and append them to the DF later
    responses = []

    print("Starting inference (one by one)...")
    for text in tqdm(df[prompt_col]):
        # Handle empty or NaN prompts gracefully
        if pd.isna(text) or text.strip() == "":
            responses.append("")
            continue

        print(f"Inference text: {text}")

        answer = runner.generate(text)
        responses.append(answer if answer is not None else "[ERROR]")

    # 4. Save Results
    # Create a new column for the response
    response_col = f"response_{model_name}"
    df[response_col] = responses

    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_results.csv")

    df.to_csv(output_csv, index=False)
    print(f"✅ Inference complete. Results saved to: {output_csv}")


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_FILE = "experiment_results/results.csv"  # The file we made in the previous step
    PROMPT_COLUMN = "csrt"
    MODEL_NAME = "llama3"

    # --- Run ---
    run_inference(
        input_csv=INPUT_FILE,
        prompt_col=PROMPT_COLUMN,
        model_name=MODEL_NAME
    )
