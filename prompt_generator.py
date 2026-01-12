from typing import List

import pandas as pd
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


# --- Main Execution Script ---

if __name__ == "__main__":
    from datasets import load_dataset

    # 1. Configuration
    # We select the matrix language here. Since MatrixLanguage.ENGLISH is "en",
    # we can use this variable directly for column selection.
    matrix_lang = MatrixLanguage.ENGLISH

    # 2. Load Data
    print("Loading MultiJail dataset...")
    # Using a small slice for demonstration
    dataset = load_dataset("DAMO-NLP-SG/MultiJail", split='train[:20]')
    multijail = dataset.to_pandas()

    # 3. Prepare Output DataFrame Dynamically
    csrt = pd.DataFrame()
    csrt['id'] = multijail['id']

    # DYNAMIC SELECTION: Use the matrix_lang string ("en") to pick the column
    csrt[matrix_lang] = multijail[matrix_lang]

    # 4. Initialize the Wrapper
    processor = DatasetSwapper(matrix_lang)

    # 5. Run the modular logic
    print(f"Starting Code-Switching for matrix language: {matrix_lang}...")
    switched_texts = processor.process_dataframe(
        df=csrt,
        text_column=matrix_lang,
        embedded_langs=[EmbeddedLanguage.GERMAN, EmbeddedLanguage.FRENCH],
        swap_ratio=0.5,
        batch_size=10
    )

    # 6. Save results
    csrt['csrt'] = switched_texts

    # Verify
    print("\nSample Output:")
    # displaying the dynamic source column vs the switched column
    print(csrt[[matrix_lang, 'csrt']].head(3))
