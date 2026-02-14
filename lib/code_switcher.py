import random
from typing import List, Dict, Set, Optional

import argostranslate.package
import argostranslate.translate
from deep_translator import GoogleTranslator

from lib.custom_types import MatrixLanguage, EmbeddedLanguage, PossibleSwaps
from lib.tagger import POSMasker

# --- Configuration ---
# Argos Translate runs locally. Google runs online.

from functools import lru_cache

# ... imports ...

# Argos Translate runs locally. Google runs online.

@lru_cache(maxsize=None)
def get_argos_translator(from_code: str, to_code: str):
    """
    Returns a translation object for Argos.
    """
    try:
        return argostranslate.translate.get_translation_from_codes(from_code, to_code)
    except Exception:
        return None

def translate_text_argos(text: str, from_code: str, to_code: str) -> str:
    """
    Translates text using Argos Translate (Local).
    Tries direct translation first, then pivots through English.
    """
    # 1. Direct Translation
    translator = get_argos_translator(from_code, to_code)
    if translator:
        return translator.translate(text)
    
    # 2. Pivot through English
    if from_code != 'en' and to_code != 'en':
        t1 = get_argos_translator(from_code, 'en')
        t2 = get_argos_translator('en', to_code)
        if t1 and t2:
            return t2.translate(t1.translate(text))

    # 3. Fallback
    return text

def translate_text_google(text: str, to_code: str) -> str:
    """
    Translates text using Google Translate (Online).
    """
    try:
        # deep_translator handles 'auto' source well, or we can be explicit
        return GoogleTranslator(source='auto', target=to_code).translate(text)
    except Exception as e:
        print(f"⚠️ Google API Error: {e}")
        return text

def code_switch(input_sentences: List[str],
                matrix_language: MatrixLanguage,
                embedded_languages: List[EmbeddedLanguage],
                swap_ratio: float,
                language_weights: Optional[Dict[EmbeddedLanguage, float]] = None,
                masker: Optional[POSMasker] = None,
                content_attr_swaps: bool = True,
                func_attr_swaps: bool = True,
                use_google_api: bool = False) -> List[str]:
    """
    Args:
        use_google_api: If True, uses Google Translate (online). If False, uses Argos (local).
    """

    # 1. Prepare Weights
    selection_weights = None
    if language_weights:
        selection_weights = [language_weights.get(lang, 0.0) for lang in embedded_languages]
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
    
    # Map: word -> set of target_languages needed
    unique_words_to_translate: Dict[str, Set[str]] = {}
    decisions = []

    for doc, mask in batch_data:
        sent_decisions = []
        for token, is_candidate in zip(doc, mask):
            if is_candidate and random.random() < swap_ratio:
                if selection_weights:
                    target_lang = random.choices(embedded_languages, weights=selection_weights, k=1)[0]
                else:
                    target_lang = random.choice(embedded_languages)

                if token.text not in unique_words_to_translate:
                    unique_words_to_translate[token.text] = set()
                unique_words_to_translate[token.text].add(target_lang)

                sent_decisions.append((True, target_lang))
            else:
                sent_decisions.append((False, None))
        decisions.append(sent_decisions)

    # 4. Batch Translation (Sequential Loop)
    translation_db = {}
    source_code = matrix_language.value if hasattr(matrix_language, 'value') else str(matrix_language)
    
    engine_name = "Google API" if use_google_api else "Argos Local"
    print(f"🚀 Translating {len(unique_words_to_translate)} unique words from {source_code} using {engine_name}...")

    for word, target_langs in unique_words_to_translate.items():
        translation_db[word] = {}
        for target_lang in target_langs:
            target_code = target_lang.value if hasattr(target_lang, 'value') else str(target_lang)
            
            # Perform Translation
            if use_google_api:
                translated_word = translate_text_google(word, target_code)
            else:
                translated_word = translate_text_argos(word, source_code, target_code)
                
            translation_db[word][target_lang] = translated_word

    # 5. Reconstruction
    final_sentences = []
    for (doc, _), sent_decs in zip(batch_data, decisions):
        output_tokens = []
        for token, (do_swap, target_lang) in zip(doc, sent_decs):
            text = token.text
            if do_swap:
                # Retrieve translation
                translation = translation_db.get(token.text, {}).get(target_lang, token.text)
                if translation is None:
                    translation = token.text
                text = translation
            output_tokens.append(text + token.whitespace_)
        final_sentences.append("".join(output_tokens))

    return final_sentences

