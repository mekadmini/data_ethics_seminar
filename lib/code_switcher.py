import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Optional

from deep_translator import GoogleTranslator

from lib.custom_types import MatrixLanguage, EmbeddedLanguage, PossibleSwaps
from lib.tagger import POSMasker

# --- Configuration ---
MAX_WORKERS = 8
BATCH_SIZE = 50  # Process 50 words per request to keep it snappy


def translate_chunk(lang: str, words: List[str]) -> tuple:
    """
    Helper function to run inside a thread.
    Returns: (lang, {original: translated})
    """
    try:
        translator = GoogleTranslator(source='auto', target=lang)
        # deep_translator handles the list, but we chunked it safely first
        translated_list = translator.translate_batch(words)

        # Zip them back into a dict
        return lang, dict(zip(words, translated_list))
    except Exception as e:
        print(f"⚠️ Error on {lang} chunk: {e}")
        # Return original words on failure so we don't lose data
        return lang, {w: w for w in words}


def batch_translate(words_map: Dict[str, Set[str]]) -> Dict[str, Dict[str, str]]:
    results = {}
    if not words_map:
        return results

    # 1. Flatten and Group by Language
    # We want to create tasks like: ("es", ["word1", "word2"...])
    full_batches: Dict[str, List[str]] = {}

    for word, langs in words_map.items():
        results[word] = {}  # Init result container
        for lang in langs:
            if lang not in full_batches:
                full_batches[lang] = []
            full_batches[lang].append(word)

    # 2. Create smaller tasks (Chunks)
    # If we have 1000 words in Spanish, we split them into 20 tasks of 50 words.
    # This ensures one huge language doesn't block the threads.
    tasks = []
    for lang, all_words in full_batches.items():
        for i in range(0, len(all_words), BATCH_SIZE):
            chunk = all_words[i: i + BATCH_SIZE]
            tasks.append((lang, chunk))

    print(f"🚀 Dispatching {len(tasks)} translation tasks across {MAX_WORKERS} threads...")

    # 3. Execute in Parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(translate_chunk, lang, chunk): (lang, chunk)
            for lang, chunk in tasks
        }

        # Process as they finish
        for future in as_completed(future_to_task):
            lang, translation_dict = future.result()

            # 4. Map back to main results
            for original, translated in translation_dict.items():
                results[original][lang] = translated

    return results


def code_switch(input_sentences: List[str],
                matrix_language: MatrixLanguage,
                embedded_languages: List[EmbeddedLanguage],
                swap_ratio: float,
                language_weights: Optional[Dict[EmbeddedLanguage, float]] = None,
                masker: Optional[POSMasker] = None,
                content_attr_swaps: bool = True,
                func_attr_swaps: bool = True) -> List[str]:
    """
    Args:
        input_sentences: Source sentences.
        matrix_language: Base language.
        embedded_languages: List of allowed target languages.
        swap_ratio: Global probability (0.0 - 1.0) that a word is switched.
                    (e.g. 0.6 means 60% of candidate words are translated).
        language_weights: (Optional) A dictionary defining the ratio between target languages.
                          Example: {'el': 0.8, 'ar': 0.2}
                          If None, all embedded_languages have equal probability.
    """

    # 1. Prepare Weights for random selection
    # We convert the dict to a list of weights aligned with the embedded_languages list
    selection_weights = None
    if language_weights:
        # Default to 0.0 if a language is in the list but missing from the weights dict
        selection_weights = [language_weights.get(lang, 0.0) for lang in embedded_languages]

        # Safety check: avoid crash if weights sum to 0
        if sum(selection_weights) == 0:
            raise ValueError("Total sum of language_weights cannot be zero.")

    # 2. Setup Masker
    if masker is None:
        masker = POSMasker(matrix_language)

    allowed_tags = set()
    if content_attr_swaps:
        allowed_tags.add(PossibleSwaps.CONTENT_SWAP)
    if func_attr_swaps:
        allowed_tags.add(PossibleSwaps.FUNCTION_SWAP)

    # 3. Stochastic Selection
    batch_data = masker.get_docs_and_masks(input_sentences, allowed_tags)
    unique_words_to_translate: Dict[str, Set[str]] = {}
    decisions = []

    for doc, mask in batch_data:
        sent_decisions = []
        for token, is_candidate in zip(doc, mask):

            # Step A: Decide IF we swap (Global Ratio)
            if is_candidate and random.random() < swap_ratio:

                # Step B: Decide WHICH language (Relative Weights)
                if selection_weights:
                    # Weighted choice
                    target_lang = random.choices(
                        population=embedded_languages,
                        weights=selection_weights,
                        k=1
                    )[0]
                else:
                    # Uniform choice (default)
                    target_lang = random.choice(embedded_languages)

                # Register logic
                if token.text not in unique_words_to_translate:
                    unique_words_to_translate[token.text] = set()
                unique_words_to_translate[token.text].add(target_lang)

                sent_decisions.append((True, target_lang))
            else:
                sent_decisions.append((False, None))
        decisions.append(sent_decisions)

    # 4. Batch Translation
    translation_db = batch_translate(unique_words_to_translate)

    # 5. Reconstruction
    final_sentences = []
    for (doc, _), sent_decs in zip(batch_data, decisions):
        output_tokens = []
        for token, (do_swap, target_lang) in zip(doc, sent_decs):
            text = token.text
            if do_swap:
                word_translations = translation_db.get(token.text, {})
                translation = word_translations.get(target_lang, token.text)
                if translation is not None:
                    text = translation
            output_tokens.append(text + token.whitespace_)
        final_sentences.append("".join(output_tokens))

    return final_sentences
