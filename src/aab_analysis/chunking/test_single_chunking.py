import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import spacy
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn
from collections import defaultdict

try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False
    print("Warning: wordfreq not installed. Install with: pip install wordfreq")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from src.utils.data_utils import load_final_dataset
from analysis.chunking.chunk_late_chunking import LateChunking
from embedding.embedding import EmbeddingService
from transformers import AutoTokenizer

DATA_DIR = str(BASE_DIR / "data" / "data_final")
CSV_PATH = str(BASE_DIR / "data" / "data_final" / "final_dataset.csv")
MODEL_NAME = "BAAI/bge-m3"
WINDOW_SIZE = 512
STRIDE = 256

GENERIC_CONCEPTS = {
    "entity", "object", "physical entity", "abstraction",
    "organism", "living thing", "person", "artifact", "act",
    "activity", "unit", "structure", "area", "location",
    "psychological feature", "attribute", "relation", "thing",
    "causal agent", "cause", "event", "happening", "phenomenon"
}

def build_noun_lemma_weights(text, lexical_weights, tokenizer, nlp):
    """
    Project BGE-M3 lexical_weights (sparse) onto noun lemmas in the text.
    Aggregates all overlapping subword tokens for each spaCy word.
    
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

def best_hypernym(noun, depth=3):
    """
    Get the best hypernym for a noun at a fixed depth, avoiding generic concepts.
    
    Args:
        noun: The noun lemma
        depth: Fixed depth in hypernym hierarchy (default 3)
    
    Returns:
        Hypernym lemma string, or None if only generic concepts found
    """
    synsets = wn.synsets(noun, pos=wn.NOUN)
    
    for syn in synsets:
        for path in syn.hypernym_paths():
            if len(path) > depth:
                h = path[depth]
                lemma = h.lemmas()[0].name().replace("_", " ").lower()
                if lemma not in GENERIC_CONCEPTS:
                    return lemma
    
    return None

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
        print(f"Warning: wordfreq not available, skipping Zipf filtering")
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

def top_concept_nouns_wordnet(lemma2weight, top_k=10, depth=3, zipf_threshold=2.5):
    """
    Map noun lemmas to concepts using WordNet hypernyms.
    Filters out obscure lemmas using Zipf frequency threshold.
    
    Args:
        lemma2weight: dict[lemma] -> weight
        top_k: Number of top concepts to return
        depth: Fixed depth for hypernym selection
        zipf_threshold: Minimum Zipf frequency for lemmas (default 2.5)
    
    Returns:
        List of (concept, score) tuples
    """
    # Filter by Zipf frequency first
    filtered_lemma2weight = filter_by_zipf(lemma2weight, zipf_threshold=zipf_threshold)
    
    if not filtered_lemma2weight:
        return []
    
    concept2w = defaultdict(float)
    
    for lemma, w in filtered_lemma2weight.items():
        concept = best_hypernym(lemma, depth=depth)
        if concept is None:
            continue
        concept2w[concept] += w
    
    ranked = sorted(concept2w.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def build_wordnet_concept_vocab(min_zipf=2, max_vocab=10000):
    """
    Build a concept vocabulary from WordNet nouns filtered by Zipf frequency.
    
    Args:
        min_zipf: Minimum Zipf frequency (default 4.5, very common words)
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

def embed_and_save_concept_vocab(vocab, output_dir, model_name="BAAI/bge-small-en-v1.5", batch_size=512):
    """
    Embed concept vocabulary and save to disk.
    
    Args:
        vocab: List of concept words
        output_dir: Directory to save .npy files
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = SentenceTransformer(model_name)
    vecs = model.encode(vocab, batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    
    words_path = output_dir / "concept_words.npy"
    vecs_path = output_dir / "concept_vecs.npy"
    
    np.save(words_path, np.array(vocab, dtype=object))
    np.save(vecs_path, vecs)
    
    print(f"Saved {len(vocab)} concepts to {words_path} and {vecs_path}")
    return words_path, vecs_path

class ConceptSpace:
    """
    Concept space built from WordNet nouns with Zipf filtering.
    Maps movie noun lemmas to concept anchors using embedding similarity.
    """
    def __init__(self, concept_words_path, concept_vecs_path, model_name="BAAI/bge-small-en-v1.5"):
        self.concept_words = np.load(concept_words_path, allow_pickle=True).tolist()
        self.concept_vecs = np.load(concept_vecs_path).astype(np.float32)  # Shape: (N, d)
        self.model = SentenceTransformer(model_name)
        self.word2idx = {w: i for i, w in enumerate(self.concept_words)}
        print(f"Loaded concept space: {len(self.concept_words)} concepts")
    
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
        # s[c_idx] += w for each known lemma
        for idx, w in zip(known_idx, known_weights):
            concept_scores[idx] += w
        
        # Step 4b: Embedding-based assignment for unknown lemmas
        # For each unknown lemma l_i:
        #   v_i = embed(l_i)  [L2-normalized]
        #   sim_i = v_i @ V_C^T  [cosine similarity with all concepts]
        #   c* = argmax_j sim_i[j]  [nearest concept]
        #   s[c*] += w_i
        if unknown_lemmas:
            unk_vecs = self.model.encode(unknown_lemmas, normalize_embeddings=True, show_progress_bar=False)
            unk_vecs = np.asarray(unk_vecs, dtype=np.float32)  # Shape: (U, d) where U = num unknown
            
            # Cosine similarity: unk_vecs @ concept_vecs^T
            # Since both are L2-normalized, dot product = cosine similarity
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

def top_concept_nouns_embedding(lemma2weight, concept_embedder, concepts, concept_vecs, top_k=10):
    """
    Map noun lemmas to concepts using embedding similarity.
    
    Args:
        lemma2weight: dict[lemma] -> weight
        concept_embedder: SentenceTransformer model
        concepts: List of concept names
        concept_vecs: Pre-computed concept embeddings
        top_k: Number of top concepts to return
    
    Returns:
        List of (concept, score) tuples
    """
    if len(lemma2weight) == 0:
        return []
    
    noun_lemmas = list(lemma2weight.keys())
    noun_weights = np.array([lemma2weight[lemma] for lemma in noun_lemmas])
    noun_weights = noun_weights / noun_weights.sum()
    
    noun_vecs = concept_embedder.encode(noun_lemmas, normalize_embeddings=True, show_progress_bar=False)
    noun_vecs = np.array(noun_vecs)
    
    sims = util.cos_sim(noun_vecs, concept_vecs).cpu().numpy()
    noun_to_concept = sims.argmax(axis=1)
    
    concept_scores = np.zeros(len(concepts))
    for noun_idx, concept_idx in enumerate(noun_to_concept):
        concept_scores[concept_idx] += noun_weights[noun_idx]
    
    ranked = np.argsort(-concept_scores)
    return [(concepts[i], concept_scores[i]) for i in ranked[:top_k] if concept_scores[i] > 0]


def extract_top_concepts(text, lexical_weights, tokenizer, nlp, concept_embedder, concept_vecs, concepts, top_k=10):
    """
    Extract top concepts from text using lexical weights and noun-to-concept mapping.
    
    Args:
        text: Input text
        lexical_weights: Dict mapping token_id (str) to weight (float)
        tokenizer: Transformer tokenizer
        nlp: spaCy language model
        concept_embedder: SentenceTransformer model for embedding nouns
        concept_vecs: Pre-computed concept embeddings
        concepts: List of concept names
        top_k: Number of top concepts to return
    
    Returns:
        List of (concept, score) tuples sorted by score
    """
    doc = nlp(text)
    
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = tokenized['input_ids']
    offsets = tokenized['offset_mapping']
    
    token_id_to_weight = {}
    for token_id_str, weight in lexical_weights.items():
        token_id_to_weight[int(token_id_str)] = weight
    
    token_weights = np.array([token_id_to_weight.get(tid, 0.0) for tid in token_ids])
    
    if token_weights.sum() == 0:
        return []
    
    token_weights = token_weights / token_weights.sum()
    
    nouns = []
    noun_token_indices = []
    
    for token in doc:
        if token.pos_ == "NOUN":
            lemma = token.lemma_.lower()
            if len(lemma) > 1:
                token_start = token.idx
                token_end = token.idx + len(token.text)
                
                matching_token_idx = None
                for idx, (start, end) in enumerate(offsets):
                    if start <= token_start < end or start < token_end <= end:
                        matching_token_idx = idx
                        break
                
                if matching_token_idx is not None:
                    nouns.append(lemma)
                    noun_token_indices.append(matching_token_idx)
    
    if len(nouns) == 0:
        return []
    
    noun_weights = np.array([token_weights[idx] for idx in noun_token_indices])
    noun_weights = noun_weights / noun_weights.sum()
    
    noun_vecs = concept_embedder.encode(nouns, normalize_embeddings=True, show_progress_bar=False)
    noun_vecs = np.array(noun_vecs)
    
    sims = util.cos_sim(noun_vecs, concept_vecs).cpu().numpy()
    noun_to_concept = sims.argmax(axis=1)
    
    concept_scores = np.zeros(len(concepts))
    for noun_idx, concept_idx in enumerate(noun_to_concept):
        concept_scores[concept_idx] += noun_weights[noun_idx]
    
    ranked = np.argsort(-concept_scores)
    return [(concepts[i], concept_scores[i]) for i in ranked[:top_k] if concept_scores[i] > 0]

if __name__ == '__main__':
    df = load_final_dataset(CSV_PATH, verbose=True)
    df = df[df['plot'].notna() & (df['plot'].str.len() > 2000)].copy()

    random_movie = df.sample(n=1, random_state=41).iloc[0]
    plot_text = random_movie['plot']
    print(f"\nSelected movie ID: {random_movie['movie_id']}")
    print(f"Plot length: {len(plot_text)} characters")
    
    print(f"\n{'='*80}")
    print("PLOT TEXT:")
    print(f"{'='*80}")
    print(plot_text)
    print(f"{'='*80}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_count = len(tokenizer(plot_text, add_special_tokens=False)["input_ids"])
    print(f"Token count: {token_count}\n")

    target_devices = ['cuda:0']
    embedding_service = EmbeddingService(MODEL_NAME, target_devices)
    late_chunking = LateChunking(embedding_service=embedding_service, 
                                 model_name=MODEL_NAME,
                                 window_size=WINDOW_SIZE,
                                 stride=STRIDE)

    results = embedding_service.encode_corpus([plot_text], batch_size=1)
    
    print("\nResults dictionary structure:")
    print(f"Keys: {list(results.keys())}")
    
    if 'dense_vecs' in results:
        print(f"dense_vecs type: {type(results['dense_vecs'])}, shape: {results['dense_vecs'].shape}")
    
    if 'lexical_weights' in results:
        print(f"\nlexical_weights type: {type(results['lexical_weights'])}, length: {len(results['lexical_weights'])}")
        lexical_weights = results['lexical_weights'][0]
        
        if isinstance(lexical_weights, dict):
            print("\nExtracting noun lemmas and concepts from lexical weights...")
            try:
                nlp = spacy.load("en_core_web_sm")
                
                # Step 1: Build word-level noun lemma weights
                lemma2weight = build_noun_lemma_weights(plot_text, lexical_weights, tokenizer, nlp)
                
                if not lemma2weight:
                    print("No noun lemmas found.\n")
                else:
                    # Filter by Zipf frequency
                    zipf_threshold = 4.0
                    filtered_lemma2weight = filter_by_zipf(lemma2weight, zipf_threshold=zipf_threshold)
                    
                    print(f"Filtered {len(lemma2weight)} lemmas to {len(filtered_lemma2weight)} with Zipf >= {zipf_threshold}")
                    
                    # Map to concepts using WordNet concept space (embedding-based)
                    try:
                        concept_dir = BASE_DIR / "data" / "concept_space"
                        concept_words_path = concept_dir / "concept_words.npy"
                        concept_vecs_path = concept_dir / "concept_vecs.npy"
                        
                        if not concept_words_path.exists() or not concept_vecs_path.exists():
                            print(f"\nBuilding WordNet concept vocabulary (this is a one-time operation)...")
                            print(f"Collecting WordNet nouns with Zipf >= 4.5...")
                            vocab = build_wordnet_concept_vocab(min_zipf=4, max_vocab=20000)
                            print(f"Found {len(vocab)} concept words")
                            print(f"Embedding concepts...")
                            embed_and_save_concept_vocab(vocab, concept_dir, 
                                                         model_name="BAAI/bge-small-en-v1.5")
                        
                        concept_space = ConceptSpace(concept_words_path, concept_vecs_path,
                                                     model_name="BAAI/bge-small-en-v1.5")
                        
                        top_concepts_wordnet_space = concept_space.map_lemmas(
                            filtered_lemma2weight, top_k=30
                        )
                        
                        if top_concepts_wordnet_space:
                            print(f"\n{'='*80}")
                            print(f"TOP 30 CONCEPTS (WORDNET 10K CONCEPT SPACE, Zipf >= {zipf_threshold}):")
                            print(f"{'='*80}")
                            for idx, (concept, score) in enumerate(top_concepts_wordnet_space, 1):
                                print(f"{idx:2d}. {concept:30s}: {score:.6f}")
                            print(f"{'='*80}\n")
                    except Exception as e:
                        print(f"WordNet concept space extraction failed: {e}")
                        import traceback
                        traceback.print_exc()
                        print()
                        
            except Exception as e:
                print(f"Error extracting concepts: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        top_n = 50
        
        if isinstance(lexical_weights, np.ndarray):
            print(f"  lexical_weights[0] type: {type(lexical_weights)}, shape: {lexical_weights.shape}, dtype: {lexical_weights.dtype}")
            
            top_indices = np.argsort(lexical_weights)[-top_n:][::-1]
            top_weights = lexical_weights[top_indices]
            
            print(f"\n{'='*80}")
            print(f"TOP {top_n} MAIN WORDS FROM LEXICAL WEIGHTS:")
            print(f"{'='*80}")
            for idx, (token_id, weight) in enumerate(zip(top_indices, top_weights), 1):
                try:
                    token = tokenizer.convert_ids_to_tokens(int(token_id))
                    print(f"{idx:3d}. Token ID {token_id:6d}: '{token}' (weight: {weight:.6f})")
                except Exception as e:
                    print(f"{idx:3d}. Token ID {token_id:6d}: <error converting> (weight: {weight:.6f})")
            print(f"{'='*80}\n")
        elif isinstance(lexical_weights, dict):
            print(f"  lexical_weights[0] type: dict, keys: {list(lexical_weights.keys())[:10]}... with a length of {len(lexical_weights)}")
            sorted_items = sorted(lexical_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            print(f"\n{'='*80}")
            print(f"TOP {top_n} MAIN WORDS FROM LEXICAL WEIGHTS:")
            print(f"{'='*80}")
            for idx, (token_id, weight) in enumerate(sorted_items, 1):
                try:
                    token_id_int = int(token_id)
                    token = tokenizer.convert_ids_to_tokens(token_id_int)
                    print(f"{idx:3d}. Token ID {token_id_int:6d}: '{token}' (weight: {weight:.6f})")
                except Exception as e:
                    token_id_int = int(token_id) if isinstance(token_id, str) else token_id
                    print(f"{idx:3d}. Token ID {token_id_int:6d}: <error converting> (weight: {weight:.6f})")
            print(f"{'='*80}\n")
        elif isinstance(lexical_weights, (list, tuple)):
            print(f"  lexical_weights[0] type: {type(lexical_weights)}, length: {len(lexical_weights)}")
            if len(lexical_weights) > 0:
                first_elem = lexical_weights[0]
                print(f"    first element type: {type(first_elem)}")
                if isinstance(first_elem, np.ndarray):
                    print(f"    first element shape: {first_elem.shape}")
                    if len(lexical_weights) == 1:
                        top_indices = np.argsort(first_elem)[-top_n:][::-1]
                        top_weights = first_elem[top_indices]
                        
                        print(f"\n{'='*80}")
                        print(f"TOP {top_n} MAIN WORDS FROM LEXICAL WEIGHTS:")
                        print(f"{'='*80}")
                        for idx, (token_id, weight) in enumerate(zip(top_indices, top_weights), 1):
                            try:
                                token = tokenizer.convert_ids_to_tokens(int(token_id))
                                print(f"{idx:3d}. Token ID {token_id:6d}: '{token}' (weight: {weight:.6f})")
                            except Exception as e:
                                print(f"{idx:3d}. Token ID {token_id:6d}: <error converting> (weight: {weight:.6f})")
                        print(f"{'='*80}\n")
                elif isinstance(first_elem, dict):
                    sorted_items = sorted(first_elem.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    
                    print(f"\n{'='*80}")
                    print(f"TOP {top_n} MAIN WORDS FROM LEXICAL WEIGHTS:")
                    print(f"{'='*80}")
                    for idx, (token_id, weight) in enumerate(sorted_items, 1):
                        try:
                            token_id_int = int(token_id)
                            token = tokenizer.convert_ids_to_tokens(token_id_int)
                            print(f"{idx:3d}. Token ID {token_id_int:6d}: '{token}' (weight: {weight:.6f})")
                        except Exception as e:
                            token_id_int = int(token_id) if isinstance(token_id, str) else token_id
                            print(f"{idx:3d}. Token ID {token_id_int:6d}: <error converting> (weight: {weight:.6f})")
                    print(f"{'='*80}\n")
        else:
            print(f"  lexical_weights[0] type: {type(lexical_weights)}, value: {lexical_weights}")
    
    if 'colbert_vecs' in results:
        print(f"\ncolbert_vecs type: {type(results['colbert_vecs'])}, length: {len(results['colbert_vecs'])}")
        if len(results['colbert_vecs']) > 0:
            for i, item in enumerate(results['colbert_vecs']):
                print(f"  colbert_vecs[{i}] type: {type(item)}")
                if isinstance(item, np.ndarray):
                    print(f"    shape: {item.shape}, dtype: {item.dtype}")
                elif isinstance(item, (list, tuple)):
                    print(f"    length: {len(item)}")
                    if len(item) > 0:
                        print(f"    first element type: {type(item[0])}")
                        if isinstance(item[0], np.ndarray):
                            print(f"    first element shape: {item[0].shape}")
                else:
                    print(f"    value: {item}")
    
    colbert_vecs = results['colbert_vecs'][0]
    print(f"\ncolbert_vecs[0] shape: {colbert_vecs.shape}")

    hidden_states = colbert_vecs
    seq_len = hidden_states.shape[0]

    window_embeddings = []
    start = 0
    while start < seq_len:
        end = min(start + WINDOW_SIZE, seq_len)
        window_hidden = hidden_states[start:end]
        window_mean = window_hidden.mean(axis=0)
        window_normalized = window_mean / np.linalg.norm(window_mean)
        window_embeddings.append(window_normalized)
        if end >= seq_len:
            break
        start += STRIDE

    window_embeddings = np.array(window_embeddings)
    print(f"window_embeddings shape: {window_embeddings.shape}")

    final_embedding_pre_norm = window_embeddings.mean(axis=0)
    print(f"final_embedding_pre_norm shape: {final_embedding_pre_norm.shape}")

    final_embedding = late_chunking.embed(plot_text)
    print(f"final_embedding shape: {final_embedding.shape}")

    embedding_service.cleanup()

