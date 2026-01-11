import random
from typing import List, Dict, Set

from deep_translator import GoogleTranslator

from lib.custom_types import MatrixLanguage, EmbeddedLanguage, PossibleSwaps
from lib.tagger import POSMasker


def batch_translate(words_map: Dict[str, Set[str]]) -> Dict[str, Dict[str, str]]:
    """
    Simulates batch translation. 
    Input: {'dog': {'de', 'fr'}, 'cat': {'de'}}
    Output: {'dog': {'de': 'Hund', 'fr': 'chien'}, ...}
    """
    # TODO: replace this with a real batch API call to Google
    results = {}

    print(f" > Translating {len(words_map)} unique words...")

    for word, langs in words_map.items():
        results[word] = {}
        for lang in langs:
            try:
                # Using deep_translator for the demo
                trans = GoogleTranslator(source='auto', target=lang).translate(word)
                results[word][lang] = trans
            except:
                results[word][lang] = word
    return results


def code_switch(input_sentences: List[str],
                matrix_language: MatrixLanguage,
                embedded_languages: List[EmbeddedLanguage],
                swap_ratio: float,
                content_attr_swaps: bool = True,
                func_attr_swaps: bool = True):
    # 1. Setup Masker
    allowed_tags = set()
    if content_attr_swaps: allowed_tags.add(PossibleSwaps.CONTENT_SWAP)
    if func_attr_swaps: allowed_tags.add(PossibleSwaps.FUNCTION_SWAP)

    masker = POSMasker(matrix_language)

    # Returns [(Doc, [True, False, ...]), ...]
    batch_data = masker.get_docs_and_masks(input_sentences, allowed_tags)

    # 2. Stochastic Selection (The "Ratio" Logic)
    # We need to know WHICH words to translate to WHICH language before we call the API
    # Structure: unique_words_to_translate[word] = {lang_code_1, lang_code_2}
    unique_words_to_translate: Dict[str, Set[str]] = {}

    # This list will store the decision for every token: (is_swapped, target_lang)
    # matching the structure of the input sentences
    decisions = []

    for doc, mask in batch_data:
        sent_decisions = []
        for token, is_candidate in zip(doc, mask):

            # Logic: If candidate AND dice roll < ratio -> Swap
            if is_candidate and random.random() < swap_ratio:
                # Pick a random embedded language uniformly
                target_lang = random.choice(embedded_languages)

                # Record that we need this translation
                if token.text not in unique_words_to_translate:
                    unique_words_to_translate[token.text] = set()
                unique_words_to_translate[token.text].add(target_lang)

                sent_decisions.append((True, target_lang))
            else:
                sent_decisions.append((False, None))
        decisions.append(sent_decisions)

    # 3. Batch Translation (Perform the API calls efficiently)
    # Returns: translation_db['word']['lang'] -> 'translated_word'
    translation_db = batch_translate(unique_words_to_translate)

    # 4. Reconstruction
    final_sentences = []

    for (doc, _), sent_decs in zip(batch_data, decisions):
        output_tokens = []
        for token, (do_swap, target_lang) in zip(doc, sent_decs):

            text = token.text
            if do_swap:
                # Fetch translation
                text = translation_db.get(token.text, {}).get(target_lang, token.text)

            # Use token.whitespace_ to preserve original spacing (e.g. no space before punctuation)
            output_tokens.append(text + token.whitespace_)

        final_sentences.append("".join(output_tokens))

    # Output
    for original, final in zip(input_sentences, final_sentences):
        print(f"Matrix:   {original}")
        print(f"Switched: {final}")
        print("-" * 30)


if __name__ == "__main__":
    sentences = [
        "Life is beautiful honey!",
        "The quick brown fox jumps over the lazy dog."
    ]

    code_switch(sentences,
                MatrixLanguage.ENGLISH,
                [EmbeddedLanguage.GERMAN,
                 EmbeddedLanguage.ARABIC,
                 EmbeddedLanguage.GREEK],  # Mix German and French
                swap_ratio=0.4)
