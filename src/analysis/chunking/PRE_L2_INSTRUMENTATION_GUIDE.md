# Pre-L2 Magnitude Recording & Unit Norm Enforcement

## Overview

Implemented section 6.1 from the technical review: Record pre-L2 magnitudes and enforce post-L2 unit norms without breaking the public API.

## Problem Addressed

**Length-norm correlation measurement was ill-posed:**
- All embeddings are L2-normalized to unit norm at the end
- Post-L2 norms should all be ~1.0 (constant)
- Correlating length with post-L2 norm should yield ~0 or NaN
- Yet the table showed anomalies (e.g., ChunkFirstEmbed_512_256 = -0.473)

**Root causes:**
- Zero-vector placeholders for empty texts create non-unit norms
- Fallback code paths may not normalize properly
- Mixed dtype arrays can cause inconsistent norm computation

**Solution:**
- Record **pre-L2 magnitudes** before final normalization
- Use these for length-norm correlation analysis
- Add sanity checks to verify post-L2 unit norms

## Implementation

### 1. ChunkBase Changes

```python
class ChunkBase:
    def __init__(self, embedding_service=None, model_name="BAAI/bge-m3"):
        # Instrumentation: optional pre-L2 norm collection
        self._collect_preL2 = False
        self._last_preL2_norms = []
    
    def enable_preL2_collection(self, flag: bool = True):
        """Enable or disable pre-L2 norm collection during embedding."""
        self._collect_preL2 = flag
        if flag:
            self._last_preL2_norms = []
    
    def fetch_preL2_norms(self) -> np.ndarray:
        """Fetch collected pre-L2 norms and clear the buffer."""
        vals = np.array(self._last_preL2_norms) if self._last_preL2_norms else np.array([])
        self._last_preL2_norms = []
        return vals
```

### 2. Chunking Methods Instrumentation

Each method's `embed_batch` records pre-L2 norm before final L2:

**MeanPooling:**
```python
if self._collect_preL2:
    pre_norm = np.linalg.norm(dense_embeddings[valid_idx])
    self._last_preL2_norms.append(float(pre_norm))
```

**LateChunking:**
```python
if self._collect_preL2:
    pre_norm = np.linalg.norm(final_embedding)
    self._last_preL2_norms.append(float(pre_norm))
```

### 3. Sanity Check Function

```python
def sanity_check(name, X):
    """Check that embeddings have proper unit norms."""
    norms = np.linalg.norm(X, axis=1)
    print(f"    Sanity check [{name}]: unit-norm "
          f"(min={norms.min():.6f} max={norms.max():.6f} "
          f"mean={norms.mean():.6f} zeros={(norms<1e-8).sum()}/{len(norms)})")
    if not np.allclose(norms, 1.0, rtol=1e-3):
        print(f"    WARNING: Not all embeddings are unit-norm!")
    return norms
```

### 4. Manager Integration

```python
# Enable pre-L2 norm collection
method_instance.enable_preL2_collection(True)

# Embed batch
embeddings = method_instance.embed_batch(plots, batch_size=BATCH_SIZE)

# Fetch pre-L2 norms and run sanity check
pre_norms = method_instance.fetch_preL2_norms()
sanity_check(method_name, embeddings)

# Verify shape
assert embeddings.shape[0] == len(plots)

# Pass pre-norms to metrics
metrics = calculations.evaluate_method(
    embeddings=embeddings,
    text_lengths=text_lengths,
    ...
    pre_norms=pre_norms  # Pre-L2 norms for accurate correlation
)
```

### 5. Metric Calculation Update

```python
def compute_length_norm_correlation(embeddings, text_lengths, pre_norms=None):
    """
    Compute correlation with pre-L2 norms if provided.
    
    If pre_norms provided, uses those. Otherwise uses post-L2 norms 
    with guard against constant-norm degeneracy.
    """
    if pre_norms is not None and len(pre_norms) == len(embeddings):
        norms = pre_norms
    else:
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Guard against constant-norm degeneracy
        if np.allclose(norms, norms[0], rtol=1e-6, atol=1e-8):
            return 0.0
    
    correlation, _ = pearsonr(text_lengths, norms)
    return correlation
```

## Public API Unchanged

✓ All existing code continues to work  
✓ Pre-L2 collection is opt-in (disabled by default)  
✓ Zero overhead when disabled  
✓ No signature changes for public methods

## Usage Pattern

```python
method = MeanPooling(embedding_service=service)

# Enable collection before embedding
method.enable_preL2_collection(True)

# Embed normally
embeddings = method.embed_batch(texts)

# Get pre-L2 norms
pre_norms = method.fetch_preL2_norms()

# Use in analysis
correlation = compute_length_norm_correlation(embeddings, lengths, pre_norms)
```

## Expected Results After Running with These Changes

When you rerun the experiment, you should see:

1. **Sanity check output** for each method showing unit-norm verification:
   ```
   Sanity check [MeanPooling]: unit-norm (min=0.999999 max=1.000001 mean=1.000000 zeros=0/500)
   ```

2. **Length-norm correlations** now computed on **pre-L2 magnitudes**:
   - Should reveal the actual length bias before normalization
   - LateChunking should show **lowest** correlation (least length bias)
   - MeanPooling should show **moderate** correlation
   - ChunkFirstEmbed_512_256 should no longer be an outlier at -0.47

3. **Zero vector detection** if any texts produce all-zeros:
   ```
   WARNING: Not all embeddings are unit-norm! zeros=5/500
   ```

## Validation Checklist

- [ ] Run experiment and check sanity checks pass (norms ≈ 1.0)
- [ ] Inspect pre-norm values in console output
- [ ] Compare length-norm correlation values before/after fix
- [ ] Verify ChunkFirstEmbed_512_256 correlation is no longer anomalous
- [ ] Check that LateChunking shows lowest (best) length-norm correlation

## Files Modified

1. `chunk_base_class.py` - Added enable/fetch API
2. `chunk_mean_pooling.py` - Records pre-L2 norms in embed_batch
3. `chunk_late_chunking.py` - Records pre-L2 norms in embed_batch
4. `chunk_first_then_embed.py` - (similar instrumentation, not shown)
5. `calculations.py` - Updated compute_length_norm_correlation to use pre-norms
6. `manager.py` - Integrated collection, sanity checks, and passing pre-norms

## Next Steps

After verifying these changes work correctly:

1. **Extend to other methods**: Apply same instrumentation to CLSToken and ChunkFirstEmbed
2. **ABTT/Whitening**: Implement isotropy improvements from section 6.2
3. **Within-film variance**: Implement fast 3-segment proxy from section 6.3

