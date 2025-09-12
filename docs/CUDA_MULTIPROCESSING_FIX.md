# CUDA Multiprocessing Compatibility Fix

## Overview

This document describes the fix implemented to resolve CUDA multiprocessing initialization errors in the OMOAI pipeline, specifically when using vLLM (Vectorized Large Language Model) in subprocesses.

## Problem Description

**Error**: `RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method`

**Context**: The error occurred when the API server attempted to run post-processing (punctuation and summarization) via subprocess calls to `scripts/post.py`, which initializes vLLM models.

**Root Cause**: On Linux systems, Python's default multiprocessing method is 'fork', which copies the parent process memory including CUDA context. CUDA does not allow re-initialization in forked processes, causing the vLLM engine initialization to fail.

## Technical Details

### Error Chain

1. Main API process initializes CUDA (through torch operations)
2. API calls [`run_postprocess_script()`](src/omoai/api/services.py:288) which spawns subprocess
3. Subprocess tries to initialize vLLM, which attempts CUDA initialization
4. CUDA detects forked process and refuses re-initialization

### Affected Components

- [`src/omoai/api/services.py`](src/omoai/api/services.py) - Subprocess call point
- [`scripts/post.py`](scripts/post.py) - vLLM initialization
- [`src/omoai/api/scripts/postprocess_wrapper.py`](src/omoai/api/scripts/postprocess_wrapper.py) - Wrapper function

## Solution Implementation

### 1. Multiprocessing Spawn Method Enforcement

**File**: [`scripts/post.py`](scripts/post.py:11-13)

```python
import multiprocessing

# Force spawn method for CUDA compatibility in subprocesses
if __name__ == "__main__":
    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
```

### 2. CUDA Context Isolation

**File**: [`src/omoai/api/scripts/postprocess_wrapper.py`](src/omoai/api/scripts/postprocess_wrapper.py:35-42)

```python
# Set environment variable to force spawn method for CUDA compatibility
env = os.environ.copy()
env["MULTIPROCESSING_START_METHOD"] = "spawn"
env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
env["TOKENIZERS_PARALLELISM"] = "false"
```

### 3. Enhanced Logging

Added diagnostic logging to track:

- Current multiprocessing start method
- CUDA availability in parent process
- Command execution details
- Environment variable settings

## Configuration Options

### Environment Variables

| Variable                       | Purpose                     | Default |
| ------------------------------ | --------------------------- | ------- |
| `MULTIPROCESSING_START_METHOD` | Force spawn method          | "spawn" |
| `VLLM_WORKER_MULTIPROC_METHOD` | vLLM multiprocessing method | "spawn" |
| `CUDA_VISIBLE_DEVICES`         | GPU visibility              | "0"     |
| `TOKENIZERS_PARALLELISM`       | Disable tokenizer warnings  | "false" |

### Performance Considerations

- **Spawn vs Fork**: Spawn method has slightly higher overhead than fork but ensures CUDA compatibility
- **Memory Usage**: Spawn creates fresh processes, potentially increasing memory usage
- **Startup Time**: Slightly longer subprocess startup due to fresh Python interpreter

## Testing

### Test Coverage

**File**: [`tests/test_cuda_multiprocessing_fix.py`](tests/test_cuda_multiprocessing_fix.py)

- ✅ Multiprocessing spawn method validation
- ✅ CUDA context isolation verification
- ✅ Environment variable configuration
- ✅ vLLM import isolation testing
- ✅ Torch CUDA context separation
- ✅ Integration test suite

### Running Tests

```bash
# Run CUDA multiprocessing tests
python -m pytest tests/test_cuda_multiprocessing_fix.py -v

# Run all tests including integration
python -m pytest tests/test_cuda_multiprocessing_fix.py::TestCUDAMultiprocessingIntegration -v
```

## Troubleshooting

### Common Issues

1. **"Cannot re-initialize CUDA" still occurs**

   - Verify environment variables are set correctly
   - Check that spawn method is being used
   - Ensure no CUDA operations in parent before subprocess

2. **Performance degradation**

   - Monitor subprocess startup times
   - Consider GPU memory utilization settings
   - Evaluate if fork method can be used for non-CUDA operations

3. **vLLM initialization failures**
   - Check vLLM version compatibility
   - Verify GPU memory availability
   - Review model configuration parameters

### Debug Steps

1. Enable debug logging:

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Check multiprocessing method:

   ```python
   import multiprocessing
   print(f"Start method: {multiprocessing.get_start_method()}")
   ```

3. Monitor CUDA state:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA initialized: {torch.cuda.is_initialized()}")
   ```

## Best Practices

### Code Guidelines

1. **Always use spawn method** when CUDA operations are involved in subprocesses
2. **Set environment variables** before subprocess creation
3. **Add logging** for multiprocessing and CUDA state
4. **Test on target hardware** with actual GPU workloads

### Future Considerations

1. **Monitor vLLM updates** for improved multiprocessing support
2. **Evaluate alternative approaches** such as process pools or service architecture
3. **Consider CUDA MPS** (Multi-Process Service) for advanced use cases
4. **Track performance metrics** to optimize spawn overhead

## References

- [PyTorch CUDA Multiprocessing Documentation](https://pytorch.org/docs/stable/multiprocessing.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Changelog

- **2025-09-12**: Initial fix implementation
- **2025-09-12**: Added comprehensive test suite
- **2025-09-12**: Documentation and troubleshooting guide
