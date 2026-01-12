import json
import os
from typing import List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from code_switcher import code_switch
from lib.custom_types import MatrixLanguage, EmbeddedLanguage
from lib.tagger import POSMasker


class DatasetSwapper:
    def __init__(self, matrix_language: MatrixLanguage):
        """
        Initialize the heavy resources (POSMasker) once here.
        """
        print(f"Initializing POSMasker for {matrix_language}...")
        self.matrix_lang = matrix_language
        self.masker = POSMasker(matrix_language)

    def process_dataframe(self,
                          df: pd.DataFrame,
                          text_column: str,
                          embedded_langs: List[EmbeddedLanguage],
                          swap_ratio: float,
                          batch_size: int = 32) -> List[str]:
        """
        Applies code switching to a dataframe column in batches.
        Returns a list of switched strings.
        """
        inputs = df[text_column].tolist()
        results = []

        # Process in batches to manage memory and API limits
        total = len(inputs)
        print(f"Processing {total} rows with batch size {batch_size}...")

        for i in tqdm(range(0, total, batch_size)):
            batch = inputs[i: i + batch_size]

            # Call your modular function, passing the preloaded masker
            switched_batch = code_switch(
                input_sentences=batch,
                matrix_language=self.matrix_lang,
                embedded_languages=embedded_langs,
                swap_ratio=swap_ratio,
                masker=self.masker  # <--- Key for speed!
            )

            results.extend(switched_batch)

        return results


def save_experiment(df: pd.DataFrame, config: dict, output_dir: str = "output"):
    """
    Saves the dataframe to CSV and the config to JSON.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Save Config
    json_path = os.path.join(output_dir, "config.json")
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to: {json_path}")

    # 2. Save Data
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")


if __name__ == "__main__":
    # --- 1. Configuration ---
    config = {
        "matrix_language": MatrixLanguage.ENGLISH,
        "embedded_languages": [EmbeddedLanguage.ARABIC, EmbeddedLanguage.GREEK, EmbeddedLanguage.SPANISH],
        "swap_ratio": 0.4,
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

    # Dynamic column selection based on matrix language string
    source_col = config["matrix_language"]
    csrt[source_col] = multijail[source_col]

    # --- 4. Processing ---
    processor = DatasetSwapper(config["matrix_language"])

    print(f"Starting Code-Switching for matrix language: {source_col}...")

    switched_texts = processor.process_dataframe(
        df=csrt,
        text_column=source_col,
        embedded_langs=config["embedded_languages"],
        swap_ratio=config["swap_ratio"],
        batch_size=config["batch_size"]
    )

    csrt['csrt'] = switched_texts

    # --- 5. Save Results & Parameters ---
    save_experiment(csrt, config, output_dir="experiment_results")

    # Verify
    print("\nSample Output:")
    print(csrt[[source_col, 'csrt']].head(3))
