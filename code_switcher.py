import random
from typing import List, Dict, Set, Optional

from deep_translator import GoogleTranslator

from lib.custom_types import MatrixLanguage, EmbeddedLanguage, PossibleSwaps
from lib.tagger import POSMasker


def batch_translate(words_map: Dict[str, Set[str]]) -> Dict[str, Dict[str, str]]:
    """
    Translates a set of words to multiple target languages efficiently.

    Args:
        words_map: A dictionary where keys are words to translate and values
                   are sets of target language codes.
                   Example: {'cat': {'de', 'fr'}, 'dog': {'de'}}

    Returns:
        A nested dictionary mapping words to their translations in specific languages.
        Example: {'cat': {'de': 'Katze', 'fr': 'chat'}, ...}
    """
    results = {}

    # Check if there is anything to translate to avoid empty processing
    if not words_map:
        return results

    # In a production environment, this is where you would batch API calls
    # to avoid hitting rate limits or high latency.
    for word, langs in words_map.items():
        results[word] = {}
        for lang in langs:
            try:
                # Using deep_translator.
                # Note: 'auto' source detection adds a small overhead;
                # passing the source lang explicitly is faster if known.
                translator = GoogleTranslator(source='auto', target=lang)
                trans = translator.translate(word)
                results[word][lang] = trans
            except Exception as e:
                # On failure, fallback to the original word
                # print(f"Translation failed for '{word}' to {lang}: {e}")
                results[word][lang] = word

    return results


def code_switch(input_sentences: List[str],
                matrix_language: MatrixLanguage,
                embedded_languages: List[EmbeddedLanguage],
                swap_ratio: float,
                masker: Optional[POSMasker] = None,
                content_attr_swaps: bool = True,
                func_attr_swaps: bool = True) -> List[str]:
    """
    Applies code-switching to a list of sentences based on POS tags and a probability ratio.

    Args:
        input_sentences: List of source sentences.
        matrix_language: The base language of the sentences.
        embedded_languages: List of languages to switch into.
        swap_ratio: Probability (0.0 to 1.0) that a candidate word gets switched.
        masker: (Optional) Pre-initialized POSMasker instance.
                Pass this to avoid reloading the spaCy model on every call.
        content_attr_swaps: Whether to swap content words (Nouns, Verbs, etc.).
        func_attr_swaps: Whether to swap function words (Det, Pronouns, etc.).

    Returns:
        List of code-switched sentences.
    """

    # 1. Setup Masker
    # If a masker instance is not provided, create a temporary one (slower).
    if masker is None:
        masker = POSMasker(matrix_language)

    allowed_tags = set()
    if content_attr_swaps:
        allowed_tags.add(PossibleSwaps.CONTENT_SWAP)
    if func_attr_swaps:
        allowed_tags.add(PossibleSwaps.FUNCTION_SWAP)

    # masker.get_docs_and_masks returns a list of tuples: [(Doc, [True, False, ...]), ...]
    batch_data = masker.get_docs_and_masks(input_sentences, allowed_tags)

    # 2. Stochastic Selection
    # Identify exactly which words need to be translated to which languages.
    unique_words_to_translate: Dict[str, Set[str]] = {}

    # Store decisions to reconstruct sentences later without re-rolling the dice.
    # Structure: decisions[sentence_idx][token_idx] = (should_swap: bool, target_lang: str)
    decisions = []

    for doc, mask in batch_data:
        sent_decisions = []
        for token, is_candidate in zip(doc, mask):

            # Logic: If it's a valid POS AND the random roll hits the ratio
            if is_candidate and random.random() < swap_ratio:
                # Pick a random embedded language uniformly
                target_lang = random.choice(embedded_languages)

                # Register this word for translation
                if token.text not in unique_words_to_translate:
                    unique_words_to_translate[token.text] = set()
                unique_words_to_translate[token.text].add(target_lang)

                sent_decisions.append((True, target_lang))
            else:
                sent_decisions.append((False, None))
        decisions.append(sent_decisions)

    # 3. Batch Translation
    # Perform the actual IO / API calls efficiently
    translation_db = batch_translate(unique_words_to_translate)

    # 4. Reconstruction
    final_sentences = []

    for (doc, _), sent_decs in zip(batch_data, decisions):
        output_tokens = []
        for token, (do_swap, target_lang) in zip(doc, sent_decs):

            text = token.text
            if do_swap:
                # Fetch translation from our pre-computed DB
                # Safe access: .get(word).get(lang, fallback_to_original)
                word_translations = translation_db.get(token.text, {})
                text = word_translations.get(target_lang, token.text)

            # Use token.whitespace_ to preserve original spacing (handles punctuation correctly)
            output_tokens.append(text + token.whitespace_)

        final_sentences.append("".join(output_tokens))

    return final_sentences
