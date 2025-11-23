"""
Concept extraction from text using lexical weights and WordNet/embedding-based concept spaces.

This module provides functionality to extract semantic concepts from text by:
1. Extracting noun lemmas and their weights from lexical weights
2. Filtering by Zipf frequency to remove obscure words
3. Mapping to concepts using WordNet hypernyms or embedding similarity
"""

import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

# Setup logging
logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONCEPT_DIR = BASE_DIR / "data" / "concept_space"
DEFAULT_CONCEPT_MODEL = "BAAI/bge-small-en-v1.5"


def _get_concept_space_filenames(min_zipf, max_vocab, model_name):
    """
    Generate filenames for concept space files based on parameters.
    
    Args:
        min_zipf: Minimum Zipf frequency used for vocabulary
        max_vocab: Maximum vocabulary size
        model_name: Model name (slashes will be replaced with underscores)
    
    Returns:
        Tuple of (words_filename, vecs_filename)
    """
    # Sanitize model name for filename
    model_safe = model_name.replace("/", "_").replace("\\", "_")
    
    # Format: concept_words_zipf{min_zipf}_vocab{max_vocab}_{model}.npy
    words_filename = f"concept_words_zipf{min_zipf}_vocab{max_vocab}_{model_safe}.npy"
    vecs_filename = f"concept_vecs_zipf{min_zipf}_vocab{max_vocab}_{model_safe}.npy"
    
    return words_filename, vecs_filename


def build_noun_lemma_weights(text, lexical_weights, tokenizer, nlp):
    """
    Project BGE-M3 lexical_weights (sparse) onto noun lemmas in the text.
    Aggregates all overlapping subword tokens for each spaCy word.
    
    Args:
        text: Input text string
        lexical_weights: Dict mapping token_id (str or int) to weight (float)
        tokenizer: Transformer tokenizer
        nlp: spaCy language model
    
    Returns:
        dict[lemma] -> aggregated weight (normalized)
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
    """
    Filter lemma2weight to only include lemmas with Zipf frequency >= threshold.
    
    Args:
        lemma2weight: dict[lemma] -> weight
        zipf_threshold: Minimum Zipf frequency (default 2.5, higher = more common words)
    
    Returns:
        Filtered dict[lemma] -> weight, renormalized
    """
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
    
    # Renormalize weights
    total = sum(filtered.values())
    if total > 0:
        filtered = {k: v / total for k, v in filtered.items()}
    
    return filtered


def build_wordnet_concept_vocab(min_zipf=2, max_vocab=10000):
    """
    Build a concept vocabulary from WordNet nouns filtered by Zipf frequency.
    
    Args:
        min_zipf: Minimum Zipf frequency (default 2)
        max_vocab: Maximum vocabulary size (default 10000)
    
    Returns:
        List of concept words (noun lemmas)
    """
    if not HAS_WORDFREQ:
        raise ImportError("wordfreq is required for building concept vocabulary")
    
    lemmas = set()
    for syn in wn.all_synsets(pos=wn.NOUN):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ").lower()
            if len(w) <= 1:
                continue
            lemmas.add(w)
    
    filtered = []
    for w in lemmas:
        try:
            z = wordfreq.zipf_frequency(w, "en", wordlist='large')
            if z >= min_zipf:
                filtered.append((w, z))
        except Exception:
            continue
    
    filtered.sort(key=lambda x: -x[1])
    vocab = [w for w, _ in filtered[:max_vocab]]
    return vocab


def embed_and_save_concept_vocab(vocab, output_dir, model_name="BAAI/bge-small-en-v1.5", 
                                  batch_size=512, min_zipf=None, max_vocab=None):
    """
    Embed concept vocabulary and save to disk.
    
    Args:
        vocab: List of concept words
        output_dir: Directory to save .npy files
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding
        min_zipf: Minimum Zipf frequency used (for filename encoding)
        max_vocab: Maximum vocabulary size used (for filename encoding)
    
    Returns:
        Tuple of (words_path, vecs_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = SentenceTransformer(model_name)
    vecs = model.encode(vocab, batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    
    # Generate filenames with parameters encoded
    if min_zipf is not None and max_vocab is not None:
        words_filename, vecs_filename = _get_concept_space_filenames(min_zipf, max_vocab, model_name)
    else:
        # Fallback to old naming if parameters not provided
        words_filename = "concept_words.npy"
        vecs_filename = "concept_vecs.npy"
    
    words_path = output_dir / words_filename
    vecs_path = output_dir / vecs_filename
    
    np.save(words_path, np.array(vocab, dtype=object))
    np.save(vecs_path, vecs)
    
    logger.info(f"Saved {len(vocab)} concepts to {words_path} and {vecs_path}")
    return words_path, vecs_path


class ConceptSpace:
    """
    Concept space built from WordNet nouns with Zipf filtering.
    Maps movie noun lemmas to concept anchors using embedding similarity.
    """
    def __init__(self, concept_words_path, concept_vecs_path, model_name="BAAI/bge-small-en-v1.5"):
        """
        Initialize ConceptSpace from saved concept vocabulary.
        
        Args:
            concept_words_path: Path to .npy file containing concept words
            concept_vecs_path: Path to .npy file containing concept embeddings
            model_name: SentenceTransformer model name for embedding unknown lemmas
        """
        self.concept_words = np.load(concept_words_path, allow_pickle=True).tolist()
        self.concept_vecs = np.load(concept_vecs_path).astype(np.float32)  # Shape: (N, d)
        self.model = SentenceTransformer(model_name)
        self.word2idx = {w: i for i, w in enumerate(self.concept_words)}
        logger.info(f"Loaded concept space: {len(self.concept_words)} concepts")
    
    def map_lemmas(self, lemma2weight, top_k=10):
        """
        Map noun lemmas to concept anchors and aggregate weights.
        
        Args:
            lemma2weight: dict[lemma] -> weight (from lexical_weights)
            top_k: Number of top concepts to return
        
        Returns:
            List of (concept_word, score) tuples, sorted by score descending
        """
        if not lemma2weight:
            return []
        
        # Step 1: Normalize weights to probability distribution
        lemmas = list(lemma2weight.keys())
        weights = np.array([lemma2weight[l] for l in lemmas], dtype=np.float32)
        weights /= weights.sum()  # Normalize: Σw_i = 1
        
        # Step 2: Split into known (direct lookup) and unknown (need embedding)
        known_idx = []
        known_weights = []
        unknown_lemmas = []
        unknown_weights = []
        
        for l, w in zip(lemmas, weights):
            idx = self.word2idx.get(l)
            if idx is not None:
                known_idx.append(idx)
                known_weights.append(w)
            else:
                unknown_lemmas.append(l)
                unknown_weights.append(w)
        
        # Step 3: Initialize concept score vector
        concept_scores = np.zeros(self.concept_vecs.shape[0], dtype=np.float32)
        
        # Step 4a: Direct assignment for known lemmas
        for idx, w in zip(known_idx, known_weights):
            concept_scores[idx] += w
        
        # Step 4b: Embedding-based assignment for unknown lemmas
        if unknown_lemmas:
            unk_vecs = self.model.encode(unknown_lemmas, normalize_embeddings=True, show_progress_bar=False)
            unk_vecs = np.asarray(unk_vecs, dtype=np.float32)  # Shape: (U, d)
            
            # Cosine similarity: unk_vecs @ concept_vecs^T
            sims = unk_vecs @ self.concept_vecs.T  # Shape: (U, N)
            
            # Find nearest concept for each unknown lemma
            best = sims.argmax(axis=1)  # Shape: (U,)
            
            # Aggregate weights to nearest concepts
            for c_idx, w in zip(best, unknown_weights):
                concept_scores[c_idx] += w
        
        # Step 5: Rank concepts by aggregated scores
        ranked = np.argsort(-concept_scores)
        results = []
        for c_idx in ranked[:top_k]:
            score = float(concept_scores[c_idx])
            if score <= 0:
                continue
            concept_word = self.concept_words[c_idx]
            results.append((concept_word, score))
        
        return results


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
    """
    Extract top concepts from text using lexical weights and concept space mapping.
    
    This is the main high-level function for concept extraction.
    
    Args:
        text: Input text string
        lexical_weights: Dict mapping token_id (str or int) to weight (float)
        tokenizer: Transformer tokenizer
        nlp: spaCy language model
        concept_space: Optional pre-loaded ConceptSpace instance
        concept_dir: Directory containing concept space files (default: data/concept_space)
        concept_model: SentenceTransformer model name for concept space (default: BAAI/bge-small-en-v1.5)
        top_k: Number of top concepts to return (default: 30)
        zipf_threshold: Minimum Zipf frequency for filtering lemmas (default: 4.0)
        build_concept_space: If True, build concept space if it doesn't exist (default: True)
        min_zipf_vocab: Minimum Zipf frequency for building concept vocabulary (default: 4.0)
        max_vocab: Maximum vocabulary size for concept space (default: 20000)
    
    Returns:
        List of (concept, score) tuples sorted by score descending
    """
    # Step 1: Build word-level noun lemma weights
    lemma2weight = build_noun_lemma_weights(text, lexical_weights, tokenizer, nlp)
    
    if not lemma2weight:
        return []
    
    # Step 2: Filter by Zipf frequency
    filtered_lemma2weight = filter_by_zipf(lemma2weight, zipf_threshold=zipf_threshold)
    
    if not filtered_lemma2weight:
        return []
    
    # Step 3: Load or build concept space
    if concept_space is None:
        if concept_dir is None:
            concept_dir = DEFAULT_CONCEPT_DIR
        
        concept_dir = Path(concept_dir)
        
        # Generate filenames based on parameters
        words_filename, vecs_filename = _get_concept_space_filenames(
            min_zipf_vocab, max_vocab, concept_model
        )
        concept_words_path = concept_dir / words_filename
        concept_vecs_path = concept_dir / vecs_filename
        
        if not concept_words_path.exists() or not concept_vecs_path.exists():
            if build_concept_space:
                logger.info(f"Building WordNet concept vocabulary (this is a one-time operation)...")
                logger.info(f"Collecting WordNet nouns with Zipf >= {min_zipf_vocab}...")
                vocab = build_wordnet_concept_vocab(min_zipf=min_zipf_vocab, max_vocab=max_vocab)
                logger.info(f"Found {len(vocab)} concept words")
                logger.info(f"Embedding concepts...")
                embed_and_save_concept_vocab(
                    vocab, concept_dir, 
                    model_name=concept_model,
                    min_zipf=min_zipf_vocab,
                    max_vocab=max_vocab
                )
            else:
                raise FileNotFoundError(
                    f"Concept space files not found at {concept_words_path} and {concept_vecs_path}. "
                    "Set build_concept_space=True to build them automatically."
                )
        
        concept_space = ConceptSpace(concept_words_path, concept_vecs_path, model_name=concept_model)
    
    # Step 4: Map lemmas to concepts
    top_concepts = concept_space.map_lemmas(filtered_lemma2weight, top_k=top_k)
    
    return top_concepts


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
    
    Convenience wrapper that extracts lexical_weights from embedding_results
    and calls extract_concepts_from_text.
    
    Args:
        text: Input text string
        embedding_results: Dictionary returned by EmbeddingService.encode_corpus()
        tokenizer: Transformer tokenizer
        nlp: spaCy language model
        concept_space: Optional pre-loaded ConceptSpace instance
        concept_dir: Directory containing concept space files
        concept_model: SentenceTransformer model name for concept space
        top_k: Number of top concepts to return
        zipf_threshold: Minimum Zipf frequency for filtering lemmas
        build_concept_space: If True, build concept space if it doesn't exist
        min_zipf_vocab: Minimum Zipf frequency for building concept vocabulary
        max_vocab: Maximum vocabulary size for concept space
    
    Returns:
        List of (concept, score) tuples sorted by score descending
    """
    if 'lexical_weights' not in embedding_results:
        raise ValueError("embedding_results must contain 'lexical_weights' key")
    
    lexical_weights = embedding_results['lexical_weights']
    
    # Handle list/array of results (one per document)
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

