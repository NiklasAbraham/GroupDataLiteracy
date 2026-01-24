"""
Concept extraction from text using sparse lexical weights.

Extracts semantic concepts by:
1. Extracting noun lemmas and their weights from lexical weights
2. Filtering by Zipf frequency to remove obscure words
3. Mapping to concepts using WordNet hypernyms or embedding similarity
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .concept_space import (
    ConceptSpace,
    DEFAULT_CONCEPT_DIR,
    DEFAULT_CONCEPT_MODEL,
    build_wordnet_concept_vocab,
    embed_and_save_concept_vocab,
    get_concept_space_filenames,
)

try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False


def build_noun_lemma_weights(text, lexical_weights, tokenizer, nlp):
    """
    Project BGE-M3 lexical_weights (sparse) onto noun lemmas in the text.
    Aggregates all overlapping subword tokens for each spaCy word.
    """
    doc = nlp(text)
    
    tok = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = tok['input_ids']
    offsets = tok['offset_mapping']
    
    id2w = {int(tid): w for tid, w in lexical_weights.items()}
    pos_weights = np.array([id2w.get(tid, 0.0) for tid in token_ids], dtype=np.float32)
    
    if pos_weights.sum() == 0:
        return {}
    
    pos_weights /= pos_weights.sum()
    
    lemma2weight = defaultdict(float)
    
    for token in doc:
        if token.pos_ != "NOUN":
            continue
        
        lemma = token.lemma_.lower()
        if len(lemma) <= 1:
            continue
        
        t_start = token.idx
        t_end = token.idx + len(token.text)
        
        idxs = []
        for i, (start, end) in enumerate(offsets):
            if not (end <= t_start or start >= t_end):
                idxs.append(i)
        
        if not idxs:
            continue
        
        w = pos_weights[idxs].sum()
        lemma2weight[lemma] += float(w)
    
    total = sum(lemma2weight.values())
    if total > 0:
        for k in list(lemma2weight.keys()):
            lemma2weight[k] /= total
    
    return dict(lemma2weight)


def filter_by_zipf(lemma2weight, zipf_threshold=2.5):
    """Filter lemma2weight to only include lemmas with Zipf frequency >= threshold."""
    if not HAS_WORDFREQ:
        return lemma2weight
    
    filtered = {}
    for lemma, weight in lemma2weight.items():
        try:
            zipf_score = wordfreq.zipf_frequency(lemma, 'en', wordlist='large')
            if zipf_score >= zipf_threshold:
                filtered[lemma] = weight
        except Exception:
            continue
    
    total = sum(filtered.values())
    if total > 0:
        filtered = {k: v / total for k, v in filtered.items()}
    
    return filtered


def extract_concepts_from_text(
    text: str,
    lexical_weights: Dict[Union[str, int], float],
    tokenizer,
    nlp,
    concept_space: Optional[ConceptSpace] = None,
    concept_dir: Optional[Path] = None,
    concept_model: str = DEFAULT_CONCEPT_MODEL,
    top_k: int = 30,
    zipf_threshold: float = 4.0,
    build_concept_space: bool = True,
    min_zipf_vocab: float = 4.0,
    max_vocab: int = 20000
) -> List[Tuple[str, float]]:
    """Extract top concepts from text using lexical weights and concept space mapping."""
    lemma2weight = build_noun_lemma_weights(text, lexical_weights, tokenizer, nlp)
    
    if not lemma2weight:
        return []
    
    filtered_lemma2weight = filter_by_zipf(lemma2weight, zipf_threshold=zipf_threshold)
    
    if not filtered_lemma2weight:
        return []
    
    if concept_space is None:
        if concept_dir is None:
            concept_dir = DEFAULT_CONCEPT_DIR
        
        concept_dir = Path(concept_dir)
        
        words_filename, vecs_filename = get_concept_space_filenames(
            min_zipf_vocab, max_vocab, concept_model
        )
        concept_words_path = concept_dir / words_filename
        concept_vecs_path = concept_dir / vecs_filename
        
        if not concept_words_path.exists() or not concept_vecs_path.exists():
            if build_concept_space:
                vocab = build_wordnet_concept_vocab(
                    min_zipf=min_zipf_vocab, 
                    max_vocab=max_vocab, 
                    filter_verbs=True, 
                    filter_generic=True
                )
                embed_and_save_concept_vocab(
                    vocab, concept_dir, 
                    model_name=concept_model,
                    min_zipf=min_zipf_vocab,
                    max_vocab=max_vocab
                )
            else:
                raise FileNotFoundError(f"Concept space files not found: {concept_words_path}")
        
        concept_space = ConceptSpace(concept_words_path, concept_vecs_path, model_name=concept_model)
    
    return concept_space.map_lemmas(filtered_lemma2weight, top_k=top_k)


def extract_concepts_from_embedding_results(
    text: str,
    embedding_results: Dict,
    tokenizer,
    nlp,
    concept_space: Optional[ConceptSpace] = None,
    concept_dir: Optional[Path] = None,
    concept_model: str = DEFAULT_CONCEPT_MODEL,
    top_k: int = 30,
    zipf_threshold: float = 4.0,
    build_concept_space: bool = True,
    min_zipf_vocab: float = 4.0,
    max_vocab: int = 20000
) -> List[Tuple[str, float]]:
    """
    Extract concepts from embedding service results.
    Convenience wrapper that extracts lexical_weights from embedding_results.
    """
    lexical_weights = embedding_results.get('lexical_weights')
    if lexical_weights is None:
        raise ValueError("embedding_results must contain 'lexical_weights' key")
    
    if isinstance(lexical_weights, (list, np.ndarray)):
        if len(lexical_weights) == 0:
            raise ValueError("lexical_weights is empty")
        lexical_weights = lexical_weights[0]
    
    if not isinstance(lexical_weights, dict):
        raise ValueError(f"lexical_weights must be a dict, got {type(lexical_weights)}")
    
    return extract_concepts_from_text(
        text=text,
        lexical_weights=lexical_weights,
        tokenizer=tokenizer,
        nlp=nlp,
        concept_space=concept_space,
        concept_dir=concept_dir,
        concept_model=concept_model,
        top_k=top_k,
        zipf_threshold=zipf_threshold,
        build_concept_space=build_concept_space,
        min_zipf_vocab=min_zipf_vocab,
        max_vocab=max_vocab
    )
