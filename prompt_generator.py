import json
import os
from typing import List, Dict, Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from lib.code_switcher import code_switch
from lib.custom_types import MatrixLanguage, EmbeddedLanguage
from lib.tagger import POSMasker


class DatasetSwapper:
    def __init__(self, matrix_language: MatrixLanguage):
        """
        Initialize the heavy resources (POSMasker) once here.
        """
        # Handle case where matrix_language might be passed as a string
        lang_str = matrix_language.value if hasattr(matrix_language, 'value') else str(matrix_language)

        print(f"Initializing POSMasker for {lang_str}...")
        self.matrix_lang = matrix_language
        self.masker = POSMasker(matrix_language)

    def process_dataframe(self,
                          df: pd.DataFrame,
                          text_column: str,
                          embedded_langs: List[EmbeddedLanguage],
                          swap_ratio: float,
                          language_weights: Optional[Dict[EmbeddedLanguage, float]] = None,
                          batch_size: int = 32,
                          use_google_api: bool = False) -> List[str]:
        """
        Applies code switching to a dataframe column in batches.
        Supports language_weights for weighted distribution of embedded languages.
        """
        inputs = df[text_column].tolist()
        results = []

        # Process in batches to manage memory and API limits
        total = len(inputs)
        print(f"Processing {total} rows with batch size {batch_size}...")

        for i in tqdm(range(0, total, batch_size)):
            batch = inputs[i: i + batch_size]

            switched_batch = code_switch(
                input_sentences=batch,
                matrix_language=self.matrix_lang,
                embedded_languages=embedded_langs,
                swap_ratio=swap_ratio,
                language_weights=language_weights,  # <--- Passing weights
                masker=self.masker,
                use_google_api=use_google_api
            )

            results.extend(switched_batch)

        return results


def save_experiment(df: pd.DataFrame, config: dict, output_dir: str = "output"):
    """
    Saves the dataframe to CSV and the config to JSON.
    Handles Enum serialization automatically.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Helper to convert Enums to strings for JSON
    def enum_serializer(o):
        if hasattr(o, 'value'):
            return o.value
        return str(o)

    # 1. Save Config
    json_path = os.path.join(output_dir, "config.json")

    # We copy the config to avoid modifying the original dictionary in memory
    # Ideally, we rely on json.dump's 'default' parameter to handle the Enums inside the dict
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4, default=enum_serializer)

    print(f"Config saved to: {json_path}")

    # 2. Save Data
    csv_path = os.path.join(output_dir, "prompts.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")


if __name__ == "__main__":
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
