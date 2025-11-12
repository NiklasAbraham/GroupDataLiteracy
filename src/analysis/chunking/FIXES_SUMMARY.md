# LateChunking Implementation Fixes - Summary

## Date: 2025-11-12

## Problem Statement
The initial experimental run showed that `LateChunking` was producing **identical** results to `MeanPooling` (cosine similarity = 1.0), indicating the implementation had collapsed algebraically into global mean pooling.

## Root Causes Identified

### 1. Missing Per-Window Normalization (CRITICAL)
**Problem:** Windows were mean-pooled but NOT normalized before aggregation.
```python
# BEFORE (WRONG):
window_embedding = window_hidden.mean(axis=0)  # Linear operation
final_embedding = np.average(window_embeddings, weights=...)  # Still linear
return self._normalize(final_embedding)  # Single normalization at end
```

This created a purely linear pipeline: `normalize(weighted_mean(mean(windows)))` which is algebraically equivalent to `normalize(global_mean(tokens))`.

**Fix:** Added L2 normalization to each window BEFORE aggregation:
```python
# AFTER (CORRECT):
window_embedding = self._normalize(window_hidden.mean(axis=0))  # Non-linear!
final_embedding = np.average(window_embeddings, weights=...)  # Avg of normalized
return self._normalize(final_embedding)  # Final stabilization
```

This **two-stage normalization** (per-window + final) is the critical non-linearity that distinguishes late chunking from mean pooling.

### 2. Inappropriate Window Sizes
**Problem:** Window sizes (512, 1024, 2048, 4096) exceeded the model's maximum sequence length.
- BGE-M3 truncates inputs to ~512 tokens
- `seq_len=511` even for 44,500 token inputs
- No windows were created because `seq_len <= window_size`

**Fix:** Adjusted window sizes to fit within model limits:
- Changed from: 512/1024/2048/4096
- Changed to: 128/256/384 (for LateChunking)
- ChunkFirstEmbed can use larger sizes since it chunks BEFORE embedding

### 3. Explicit Error Handling for Missing colbert_vecs
**Problem:** Silent fallback to dense vectors when `colbert_vecs` unavailable, changing the algorithm without warning.

**Fix:** Added explicit error checking:
```python
if 'colbert_vecs' not in results:
    raise ValueError(
        "LateChunking requires token-level hidden states (colbert_vecs). "
        "Cannot perform late chunking without token-level vectors."
    )
```

### 4. Overlap Weighting (ENHANCEMENT)
**Problem:** Simple mean of windows gives higher weight to tokens in overlapping regions (counted multiple times).

**Fix:** Implemented weighted averaging that accounts for overlap:
- Count how many windows contain each token
- Weight windows inversely proportional to average token coverage
- Windows with more unique tokens get higher weight

## Verification
Created sanity test (`test_late_chunking_fix.py`) that confirmed:
- **Before fix:** Cosine similarity = 1.000000 (identical)
- **After fix:** Cosine similarity = 0.999973 (different, but similar as expected)
- LateChunking now creates multiple windows (e.g., 7 windows for 511 tokens with window_size=128)
- Each window is normalized (norm = 1.0)

## Updated Configuration

### manager.py
```python
# LateChunking - embeds full text then chunks hidden states
# NOTE: BGE-M3 truncates to ~512 tokens, so windows operate within this limit
'LateChunking_128_64': LateChunking(window_size=128, stride=64),
'LateChunking_256_128': LateChunking(window_size=256, stride=128),
'LateChunking_384_192': LateChunking(window_size=384, stride=192),

# ChunkFirstEmbed - processes text in chunks before embedding
# Can use larger sizes since chunking happens BEFORE embedding
'ChunkFirstEmbed_512_256': ChunkFirstEmbed(chunk_size=512, stride=256),
'ChunkFirstEmbed_1024_512': ChunkFirstEmbed(chunk_size=1024, stride=512),
'ChunkFirstEmbed_2048_1024': ChunkFirstEmbed(chunk_size=2048, stride=1024),
```

## Expected Results After Fix

| Metric | MeanPool | CLS | ChunkFirst-512 | LateChunk (patched) |
|--------|----------|-----|----------------|---------------------|
| length-norm corr | ~0.02 | ~0.03 | **≈ -0.40** | **≈ 0.00** |
| 1st PC % (isotropy) | 11% | 5% | 6-7% | **≤ 8%** |
| within-var | 0 | 0 | 0.25-0.35 | **0.10-0.18** |
| between-dist | 0.65 | 0.61 | 0.60 | **0.50-0.55** |
| silhouette | -0.13 | -0.09 | -0.10 | **-0.03 → +0.05** |

Key predictions:
1. **LateChunking should NO LONGER equal MeanPooling**
2. Length-norm correlation should flatten (no length bias)
3. Isotropy should improve (lower first PC%)
4. Different window sizes should now produce distinct results
5. Genre clustering quality should improve

## Files Modified
1. `src/analysis/chunking/chunk_late_chunking.py` - Added per-window normalization, error handling, overlap weighting
2. `src/analysis/chunking/manager.py` - Updated window sizes to fit model limits
3. Created and removed `test_late_chunking_fix.py` - Sanity test (temporary)

## Next Steps
1. Run full experiment with `python manager.py`
2. Verify LateChunking metrics differ from MeanPooling
3. Check that different LateChunking configs produce varied results
4. Compare results to theoretical predictions above

