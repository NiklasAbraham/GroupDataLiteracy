"""
Script to create a comparison table of dense and sparse concept extraction results.
"""

# Dense results (from example_concept_extraction_dense.py, lines 212-241)
dense_results = [
    ("arguing", 0.544049),
    ("filming", 0.531092),
    ("inside", 0.525644),
    ("socializing", 0.523859),
    ("reliving", 0.523364),
    ("yelling", 0.521545),
    ("shouting", 0.521472),
    ("pretending", 0.520833),
    ("communicating", 0.518198),
    ("someone", 0.516860),
    ("backstage", 0.516742),
    ("happening", 0.515222),
    ("humans", 0.513191),
    ("halftime", 0.513142),
    ("altercation", 0.512606),
    ("chattering", 0.511847),
    ("talks", 0.511800),
    ("sweating", 0.511625),
    ("roleplaying", 0.509357),
    ("sneezing", 0.509242),
    ("sit-in", 0.508355),
    ("check-in", 0.507920),
    ("daydreaming", 0.507528),
    ("hurrying", 0.506722),
    ("somebody", 0.506603),
    ("pleading", 0.506445),
    ("anecdote", 0.506024),
    ("sit-down", 0.505426),
    ("haggling", 0.504629),
    ("playback", 0.504442),
]

# Sparse results (from example_concept_extraction_sparse.py, lines 346-375)
sparse_results = [
    ("cinema", 0.140162),
    ("sectional", 0.045673),
    ("tourist", 0.041620),
    ("return", 0.041336),
    ("characteristic", 0.041081),
    ("female", 0.036733),
    ("police", 0.032159),
    ("feelings", 0.030896),
    ("shop", 0.023586),
    ("friend", 0.017115),
    ("slut", 0.016428),
    ("humor", 0.015926),
    ("hombre", 0.015551),
    ("search", 0.015023),
    ("terrorist", 0.014839),
    ("tranquillity", 0.014705),
    ("routine", 0.014684),
    ("discourse", 0.014459),
    ("life", 0.013988),
    ("television", 0.013844),
    ("scenery", 0.013648),
    ("light", 0.013422),
    ("period", 0.013034),
    ("parent", 0.012890),
    ("election", 0.012552),
    ("theatre", 0.012369),
    ("relation", 0.012162),
    ("fashion", 0.011937),
    ("boat", 0.011896),
    ("image", 0.011896),
]


def print_comparison_table():
    """Print a formatted comparison table of dense and sparse results."""
    print("=" * 100)
    print("COMPARISON TABLE: DENSE vs SPARSE CONCEPT EXTRACTION")
    print("=" * 100)
    print()
    print(f"{'Rank':<6} {'Dense Concept':<30} {'Dense Score':<12} {'Sparse Concept':<30} {'Sparse Score':<12}")
    print("-" * 100)
    
    for idx in range(len(dense_results)):
        dense_word, dense_score = dense_results[idx]
        sparse_word, sparse_score = sparse_results[idx]
        
        print(f"{idx+1:<6} {dense_word:<30} {dense_score:<12.6f} {sparse_word:<30} {sparse_score:<12.6f}")
    
    print("=" * 100)


def print_markdown_table():
    """Print a markdown formatted table."""
    print("| Rank | Dense Concept | Dense Score | Sparse Concept | Sparse Score |")
    print("|------|---------------|-------------|----------------|--------------|")
    
    for idx in range(len(dense_results)):
        dense_word, dense_score = dense_results[idx]
        sparse_word, sparse_score = sparse_results[idx]
        
        print(f"| {idx+1} | {dense_word} | {dense_score:.6f} | {sparse_word} | {sparse_score:.6f} |")


if __name__ == '__main__':
    print_comparison_table()
    print("\n\n")
    print("Markdown format:")
    print_markdown_table()
