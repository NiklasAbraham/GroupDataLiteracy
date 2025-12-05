# Analysis module

# Lazy imports to avoid requiring nltk when only chunking is needed
# Import concept_extraction from concept_words module only when actually used
def __getattr__(name):
    if name in ['extract_concepts_from_text', 'extract_concepts_from_embedding_results', 
                'build_noun_lemma_weights', 'filter_by_zipf']:
        from concept_words.concept_extraction_sparse import (
            extract_concepts_from_text,
            extract_concepts_from_embedding_results,
            build_noun_lemma_weights,
            filter_by_zipf,
        )
        globals().update({
            'extract_concepts_from_text': extract_concepts_from_text,
            'extract_concepts_from_embedding_results': extract_concepts_from_embedding_results,
            'build_noun_lemma_weights': build_noun_lemma_weights,
            'filter_by_zipf': filter_by_zipf,
        })
        return globals()[name]
    elif name in ['ConceptSpace', 'build_wordnet_concept_vocab', 'embed_and_save_concept_vocab']:
        from concept_words.concept_space import (
            ConceptSpace,
            build_wordnet_concept_vocab,
            embed_and_save_concept_vocab,
        )
        globals().update({
            'ConceptSpace': ConceptSpace,
            'build_wordnet_concept_vocab': build_wordnet_concept_vocab,
            'embed_and_save_concept_vocab': embed_and_save_concept_vocab,
        })
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'extract_concepts_from_text',
    'extract_concepts_from_embedding_results',
    'ConceptSpace',
    'build_noun_lemma_weights',
    'filter_by_zipf',
    'build_wordnet_concept_vocab',
    'embed_and_save_concept_vocab',
]

