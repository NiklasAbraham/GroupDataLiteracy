# Norm Instrumentation Feature

## Overview

Added light "tap" instrumentation to record pre-L2 magnitudes and enforce post-L2 unit norms. The public API remains **completely unchanged** - the feature is entirely **opt-in** via optional parameters.

## Features

### 1. Pre-L2 Magnitude Recording
Records the norm of each vector **before** L2 normalization. This allows analysis of:
- Magnitude distribution across different aggregation methods
- Bias detection (methods consistently producing large/small magnitudes)
- Stability metrics

### 2. Post-L2 Unit Norm Verification
Verifies that every normalized vector has a norm of 1.0 (within numerical precision). Records any violations:
- Catches numerical instability issues
- Validates normalization correctness
- Useful for debugging

### 3. Optional Hooks (No API Breaking)
All instrumentation is **opt-in** and **backward compatible**:
- Enabled via `record_norms=True` parameter in initialization
- Default is `False` (no overhead when not needed)
- Public API signatures unchanged

## Usage

### Basic Usage (No Changes to Existing Code)

```python
# Old code still works exactly the same
embedding_service = EmbeddingService("BAAI/bge-m3", None)
method = MeanPooling(embedding_service=embedding_service)
embeddings = method.embed("This is a test document.")
# No instrumentation, no overhead
```

### Enable Instrumentation

```python
# New: Enable norm recording
method = MeanPooling(
    embedding_service=embedding_service,
    record_norms=True  # Enable instrumentation
)

# Embed texts normally
embeddings = method.embed("This is a test document.")

# Get recorded data
records = method.get_norm_records()
print(records['pre_norm_magnitudes'])   # Pre-L2 magnitudes
print(records['post_norm_violations'])  # Any norm != 1.0 issues

# Print human-readable summary
method.print_norm_summary()

# Clear records between runs
method.clear_norm_records()
```

## API Reference

### ChunkBase.__init__ (Updated)

```python
def __init__(self, 
             embedding_service: EmbeddingService = None, 
             model_name: str = "BAAI/bge-m3",
             record_norms: bool = False):
    """
    Args:
        embedding_service: Pre-initialized EmbeddingService
        model_name: Model name (default: BAAI/bge-m3)
        record_norms: If True, record pre-L2 magnitudes (default: False)
    """
```

### ChunkBase._normalize (Updated)

```python
def _normalize(self, vec: np.ndarray, vector_id: str = None) -> np.ndarray:
    """
    L2-normalize a vector with optional instrumentation.
    
    Args:
        vec: Vector to normalize
        vector_id: Identifier for tracking (optional, used when record_norms=True)
    
    Returns:
        np.ndarray: L2-normalized vector (guaranteed unit norm unless zero vector)
    """
```

### New Methods

#### get_norm_records()
```python
def get_norm_records(self) -> Dict[str, list]:
    """
    Get recorded norm data.
    
    Returns:
        Dict with keys:
            - 'pre_norm_magnitudes': List of {'vector_id', 'pre_norm_magnitude'}
            - 'post_norm_violations': List of {'vector_id', 'actual_norm', 'expected_norm'}
    """
```

#### clear_norm_records()
```python
def clear_norm_records(self):
    """Clear all recorded norm data."""
```

#### print_norm_summary()
```python
def print_norm_summary(self):
    """Print human-readable summary of norm records."""
    # Output:
    # Pre-L2 Magnitude Summary:
    #   Count: 100
    #   Min: 0.805234
    #   Max: 2.453891
    #   Mean: 1.234567
    #   Std: 0.342123
    #
    # Post-L2 Unit Norm Violations: 0
```

## Implementation Details

### Where Instrumentation Happens

1. **ChunkBase._normalize()**: 
   - Records pre-L2 magnitude before normalization
   - Verifies post-L2 norm equals 1.0
   - Logs violations if tolerance exceeded (rtol=1e-5, atol=1e-8)

2. **All Chunking Methods**:
   - Updated MeanPooling.embed() to pass vector_id to _normalize()
   - Similar updates can be applied to other methods

### Zero Overhead When Disabled

- When `record_norms=False` (default), the condition checks skip all recording
- No list appends, no string formatting
- Minimal performance impact

### Vectorization Compatible

- The instrumentation is applied at the vector level (after aggregation)
- Works with both single embedding and batch processing
- Does not interfere with batch operations

## Testing

### Run the Test Suite

```bash
cd /home/nab/GroupDataLiteracy
conda activate dataLiteracy
python src/analysis/chunking/test_norm_instrumentation.py
```

### Test Coverage

1. **Norm Recording Test**: Verifies that pre-L2 magnitudes are recorded
2. **Unit Norm Verification**: Checks that all normalized vectors have norm ~1.0
3. **Public API Test**: Confirms backward compatibility
4. **Batch Processing Test**: Ensures instrumentation works in batch mode

## Example Output

```
================================================================================
NORM INSTRUMENTATION TEST
================================================================================

Processing test texts...
Number of texts: 3

========================================
Method: MeanPooling
========================================
  Embedding text 1/3...
  Embedding text 2/3...
  Embedding text 3/3...

Pre-L2 Magnitude Summary:
  Count: 3
  Min: 1.234567
  Max: 1.876543
  Mean: 1.523456
  Std: 0.285123

Post-L2 Unit Norm Violations: 0

Batch Processing Test:
  Embedding 3 texts in batch...
  Output shape: (3, 1024)
  Batch norms (should all be ~1.0):
    Embedding 1: 1.00000000
    Embedding 2: 1.00000000
    Embedding 3: 1.00000000
  Unit norm check: ✓ PASS
```

## Integration with Experiments

To enable norm recording in the main experiment:

```python
# In manager.py, update method initialization
methods = {
    'MeanPooling': MeanPooling(
        embedding_service=embedding_service,
        model_name=MODEL_NAME,
        record_norms=True  # Enable instrumentation
    ),
    # ... other methods ...
}

# After embedding, get summary
for method_name, method_instance in methods.items():
    method_instance.print_norm_summary()
    records = method_instance.get_norm_records()
    # Save records to analysis directory if needed
```

## Benefits

1. **Debugging**: Quickly identify numerical issues
2. **Analysis**: Understand magnitude behavior of different methods
3. **Validation**: Ensure normalization correctness
4. **Zero Cost**: Disabled by default, no overhead when not needed
5. **Backward Compatible**: Existing code works without changes

## Future Extensions

The instrumentation can be extended to:
- Track per-window magnitudes in ChunkFirstEmbed
- Record intermediate aggregation states
- Capture gradient flow in training scenarios
- Profile memory usage during normalization

## Notes

- Vector IDs are generated deterministically for reproducibility
- Records are stored in memory (clear_norm_records() if memory is a concern)
- Tolerance for unit norm check: `rtol=1e-5, atol=1e-8` (generous, handles float precision)

