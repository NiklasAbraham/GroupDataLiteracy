"""
Shared concept space functionality for concept extraction methods.

Builds and loads concept spaces from WordNet nouns filtered by Zipf frequency.
Used by both sparse and dense embedding concept extraction methods.
"""

from pathlib import Path

import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONCEPT_DIR = BASE_DIR / "data" / "concept_space"
DEFAULT_CONCEPT_MODEL = "BAAI/bge-small-en-v1.5"


def get_concept_space_filenames(min_zipf, max_vocab, model_name):
    """Generate filenames for concept space files based on parameters."""
    model_safe = model_name.replace("/", "_").replace("\\", "_")
    words_filename = f"concept_words_zipf{min_zipf}_vocab{max_vocab}_{model_safe}.npy"
    vecs_filename = f"concept_vecs_zipf{min_zipf}_vocab{max_vocab}_{model_safe}.npy"
    return words_filename, vecs_filename


def _is_verb_like(word):
    """Check if a word is primarily a verb or verb-derived (gerund)."""
    verb_synsets = wn.synsets(word, pos=wn.VERB)
    noun_synsets = wn.synsets(word, pos=wn.NOUN)
    
    if word.endswith('ing') and verb_synsets:
        return True
    
    if verb_synsets and len(verb_synsets) > len(noun_synsets):
        return True
    
    return False


def _is_valid_english_word(word, min_length=2):
    """Check if a word meets minimum length requirement."""
    return len(word) >= min_length


def build_wordnet_concept_vocab(min_zipf=2, max_vocab=10000, filter_verbs=False, filter_generic=True):
    """
    Build concept vocabulary from WordNet nouns filtered by Zipf frequency.
    Only includes single-word nouns (excludes multi-word phrases).
    """
    if not HAS_WORDFREQ:
        raise ImportError("wordfreq is required for building concept vocabulary")
    
    generic_words = {
        # Movie/film terms
        'movie', 'film', 'story', 'storyline', 'plot', 'narrative', 'screenplay',
        'episode', 'summary', 'documentary', 'narration', 'narrator',
        'flashback', 'foreground', 'background', 'reenactment', 'spoiler',
        'scene', 'shot', 'camera', 'cameraman', 'video', 'footage', 'recording',
        'trailer', 'soundtrack', 'replay', 'watching', 'viewer', 'audience',
        
        # People terms
        'human', 'people', 'person', 'man', 'woman', 'character', 'protagonist',
        
        # Communication terms
        'conversation', 'dialogue', 'dialog', 'discussion', 'talking', 'speaking', 'talk',
        
        # Time/space terms
        'duration', 'time', 'moment', 'part', 'section', 'beginning', 'middle', 'end', 'ending',
        
        # Generic actions/states
        'working', 'going', 'coming', 'looking', 'seeing', 'hearing', 'feeling',
        'settling', 'journeying', 'walkabout',
        
        # Generic objects
        'thing', 'stuff', 'something', 'anything', 'everything', 'nothing',
        'situation', 'incident', 'seat', 'backseat',
        
        # Generic emotions/states
        'feeling', 'calmness', 'relaxation', 'sedation', 'hesitation', 'nostalgia',
        'awakening', 'eagerness', 'elation', 'enthusiasm', 'ambience', 'ambiance',
        
        # Other generic terms
        'lesson', 'peace', 'peacekeeping', 'peacetime', 'silent', 'funny',
        'reunion', 'rehearsal', 'negotiation', 'bargaining', 'bickering',
        'whispering', 'weeping', 'suspicion', 'apprehension', 'uneasiness',
        'proximity', 'introspection', 'hallucination', 'serendipity',
        'indiscretion', 'antisemitism', 'salient', 'gunshot', 'policeman',
        'backpacker', 'salesman', 'philistine', 'burglary'
    }
    
    words = set()
    for syn in wn.all_synsets(pos=wn.NOUN):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ").lower().strip()
            if " " in w or len(w) <= 1:
                continue
            words.add(w)
    
    filtered = []
    filtered_out_generic = 0
    for word in words:
        if len(word) < 2:
            continue
        
        if filter_generic and word in generic_words:
            filtered_out_generic += 1
            continue
        
        if filter_verbs and _is_verb_like(word):
            continue
            
        try:
            z = wordfreq.zipf_frequency(word, "en", wordlist='large')
            if z >= min_zipf:
                filtered.append((word, z))
        except Exception:
            continue
    
    if filter_generic:
        print(f"Filtered out {filtered_out_generic} generic words from concept vocabulary")
    
    filtered.sort(key=lambda x: -x[1])
    return [w for w, _ in filtered[:max_vocab]]


def embed_and_save_concept_vocab(vocab, output_dir, model_name="BAAI/bge-small-en-v1.5", 
                                  batch_size=512, min_zipf=None, max_vocab=None):
    """Embed concept vocabulary and save to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = SentenceTransformer(model_name)
    vecs = model.encode(vocab, batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    
    if min_zipf is not None and max_vocab is not None:
        words_filename, vecs_filename = get_concept_space_filenames(min_zipf, max_vocab, model_name)
    else:
        words_filename = "concept_words.npy"
        vecs_filename = "concept_vecs.npy"
    
    words_path = output_dir / words_filename
    vecs_path = output_dir / vecs_filename
    
    np.save(words_path, np.array(vocab, dtype=object))
    np.save(vecs_path, vecs)
    return words_path, vecs_path


class ConceptSpace:
    """Concept space built from WordNet nouns with Zipf filtering."""
    
    def __init__(self, concept_words_path, concept_vecs_path, model_name="BAAI/bge-small-en-v1.5"):
        """Initialize ConceptSpace from saved concept vocabulary."""
        self.concept_words = np.load(concept_words_path, allow_pickle=True).tolist()
        self.concept_vecs = np.load(concept_vecs_path).astype(np.float32)
        self.model = SentenceTransformer(model_name)
        self.word2idx = {w: i for i, w in enumerate(self.concept_words)}
    
    def map_lemmas(self, lemma2weight, top_k=10):
        """
        Map noun lemmas to concept anchors and aggregate weights.
        Used by sparse embedding method to map lemmas to concepts.
        """
        if not lemma2weight:
            return []
        
        lemmas = list(lemma2weight.keys())
        weights = np.array([lemma2weight[lemma] for lemma in lemmas], dtype=np.float32)
        weights /= weights.sum()
        
        known_idx = []
        known_weights = []
        unknown_lemmas = []
        unknown_weights = []
        
        for lemma, w in zip(lemmas, weights):
            idx = self.word2idx.get(lemma)
            if idx is not None:
                known_idx.append(idx)
                known_weights.append(w)
            else:
                unknown_lemmas.append(lemma)
                unknown_weights.append(w)
        
        concept_scores = np.zeros(self.concept_vecs.shape[0], dtype=np.float32)
        
        for idx, w in zip(known_idx, known_weights):
            concept_scores[idx] += w
        
        if unknown_lemmas:
            unk_vecs = self.model.encode(unknown_lemmas, normalize_embeddings=True, show_progress_bar=False)
            unk_vecs = np.asarray(unk_vecs, dtype=np.float32)
            
            sims = unk_vecs @ self.concept_vecs.T
            best = sims.argmax(axis=1)
            
            for c_idx, w in zip(best, unknown_weights):
                concept_scores[c_idx] += w
        
        ranked = np.argsort(-concept_scores)
        results = []
        for c_idx in ranked[:top_k]:
            score = float(concept_scores[c_idx])
            if score <= 0:
                continue
            concept_word = self.concept_words[c_idx]
            results.append((concept_word, score))
        
        return results
