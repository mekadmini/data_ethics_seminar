from typing import List, Iterable, Tuple

import spacy

from lib.custom_types import MatrixLanguage, PossibleSwaps


class POSMasker:
    def __init__(self, matrix_language: MatrixLanguage):
        model_map = {
            MatrixLanguage.ENGLISH: "en_core_web_sm",
            MatrixLanguage.SPANISH: "es_core_news_sm",
            MatrixLanguage.FRENCH: "fr_core_news_sm",
            MatrixLanguage.GERMAN: "de_core_news_sm",
            MatrixLanguage.GREEK: "el_core_news_sm",
            MatrixLanguage.ITALIAN: "it_core_news_sm"
        }
        try:
            model_name = model_map[matrix_language]
        except:
            raise NotImplementedError()

        # Load once, disable unused components
        self.nlp = spacy.load(model_name, disable=["ner", "parser", "lemmatizer"])

    def get_docs_and_masks(self,
                           sentences: List[str],
                           allowed_swaps: Iterable[PossibleSwaps]) -> List[Tuple[spacy.tokens.Doc, List[bool]]]:

        # 1. Determine Target POS Tags‘
        target_pos = set()
        for swap in allowed_swaps:
            if swap == PossibleSwaps.CONTENT_SWAP:
                target_pos.update({"NOUN", "VERB", "ADJ", "ADV", "PROPN"})
            elif swap == PossibleSwaps.FUNCTION_SWAP:
                target_pos.update({"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ"})

        results = []
        # 2. Pipe allows batch processing
        for doc in self.nlp.pipe(sentences):
            mask = [token.pos_ in target_pos for token in doc]
            results.append((doc, mask))

        return results
