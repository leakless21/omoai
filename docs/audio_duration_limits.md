# Audio Duration Limits and Performance Guide

## Overview

This guide explains the limitations, performance characteristics, and best practices for processing long audio files with the OMOAI API.

## Current Configuration Limits

### ✅ **Updated Limits (After Optimization)**

| Parameter | Limit | Reasoning |
|-----------|-------|-----------|
| **File Size** | 200MB | Supports ~3 hours of compressed audio |
| **Request Timeout** | 30 minutes | Matches ASR batch processing time |
| **ASR Batch Duration** | 1 hour | Increased from 30 minutes |
| **Recommended Max** | 2 hours | Balance of performance vs. memory |
| **Hard Warning** | 3 hours | API shows warning but processes |

### 🔧 **Technical Limits**

- **GPU Memory**: ~7GB available for vLLM models
- **Processing Speed**: ~50 tokens/s for punctuation
- **ASR Chunking**: 64-token chunks with context
- **Temp Storage**: Files stored in `/tmp` during processing

## Performance Estimates

### **Processing Time by Audio Duration**

| Audio Length | ASR Time | Punctuation | Summary | Total Est. |
|--------------|----------|-------------|---------|------------|
| 10 minutes   | ~30s     | ~30s        | ~3s     | **~1-2 min** |
| 30 minutes   | ~1-2min  | ~2min       | ~5s     | **~3-5 min** |
| 1 hour       | ~2-4min  | ~4-6min     | ~10s    | **~6-10 min** |
| 2 hours      | ~4-8min  | ~8-12min    | ~15s    | **~12-20 min** |

*Note: First-time model loading adds ~40 seconds*

### **Memory Usage by Audio Duration**

| Audio Length | RAM Usage | GPU Memory | Temp Disk |
|--------------|-----------|------------|-----------|
| 10 minutes   | ~500MB    | ~4GB       | ~20MB     |
| 30 minutes   | ~1GB      | ~5GB       | ~60MB     |
| 1 hour       | ~2GB      | ~6GB       | ~120MB    |
| 2 hours      | ~4GB      | ~7GB       | ~240MB    |

## Potential Issues & Solutions

### 🚨 **Critical Issues for Very Long Audio**

#### 1. **HTTP Timeout**
- **Problem**: Client may timeout before processing completes
- **Solution**: Increased timeout to 30 minutes
- **Recommendation**: Use async processing for 2+ hour files

#### 2. **Memory Pressure**
- **Problem**: Very long audio can exhaust GPU memory
- **Solution**: Pipeline uses chunked processing
- **Fallback**: Automatic fallback to script mode if memory insufficient

#### 3. **Disk Space**
- **Problem**: Large WAV files consume significant temp space
- **Solution**: Automatic cleanup after processing
- **Monitor**: `/tmp` usage during processing

#### 4. **Model Loading Time**
- **Problem**: Cold start adds ~40 seconds
- **Solution**: Models stay loaded between requests
- **Benefit**: Subsequent requests are much faster

### 💡 **Best Practices for Long Audio**

#### **For 1-2 Hour Audio:**
- ✅ Use the API directly
- ✅ Expect 10-20 minute processing time
- ✅ Monitor progress via logs
- ⚠️ Ensure stable network connection

#### **For 2+ Hour Audio:**
- 🔄 Consider splitting into 30-60 minute segments
- 🔄 Process segments separately and combine results
- 🔄 Use script mode if memory issues occur
- 🔄 Monitor server resources

#### **For 3+ Hour Audio:**
- ❌ Not recommended via API
- ✅ Use CLI pipeline directly
- ✅ Split into smaller segments
- ✅ Process offline

## Error Messages & Troubleshooting

### **Common Error Messages**

#### File Size Warnings:
```
"Audio file appears to be very long (~180.5 minutes estimated). 
Maximum recommended duration is 3 hours. 
Consider splitting large files for better performance."
```
**Action**: File will process but consider splitting for better performance.

#### Memory Errors:
```
"Enhanced pipeline processing failed: CUDA out of memory"
```
**Action**: API will automatically fallback to script mode.

#### Timeout Errors:
```
"Operation timed out after 1800001 milliseconds"
```
**Action**: Audio too long for API. Use CLI or split file.

## Monitoring & Optimization

### **Performance Monitoring**

Track these metrics for long audio processing:
- **Model Loading Time**: Should be ~40s first request, <5s subsequent
- **ASR Processing Speed**: ~Real-time factor of 0.1-0.2x
- **Punctuation Speed**: ~50 tokens/s
- **Memory Usage**: Should stay under 7GB GPU

### **Optimization Tips**

1. **Warm Up Models**: Send a short test file first
2. **Batch Processing**: Process multiple files in sequence to reuse loaded models
3. **Monitor Resources**: Watch GPU memory and disk space
4. **Quality vs Speed**: Longer audio = better model context but slower processing

## Configuration Reference

### **Key Configuration Settings**

```yaml
api:
  max_body_size_mb: 200          # 200MB max upload
  request_timeout_seconds: 1800   # 30 minute timeout

asr:
  total_batch_duration_s: 3600    # 1 hour ASR batches
  max_audio_duration_hours: 2     # 2 hour soft limit
```

### **Environment Variables**

```bash
# For very long audio processing
export CUDA_LAUNCH_BLOCKING=0
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_HOME=/path/to/large/cache
```

## Future Improvements

### **Planned Enhancements**

- [ ] **Async Processing**: Queue long audio for background processing
- [ ] **Progress Tracking**: Real-time progress updates for long files  
- [ ] **Smart Chunking**: Automatic optimal chunk size based on content
- [ ] **Streaming Output**: Return results as they're processed
- [ ] **Resource Management**: Dynamic memory allocation based on file size

### **Alternative Approaches**

For very long audio, consider:
- **CLI Processing**: Use command-line interface for unlimited duration
- **Segment Processing**: Split audio and combine results
- **Batch API**: Process multiple files efficiently
- **Custom Pipeline**: Direct Python pipeline usage

---

*Last Updated: September 8, 2025*
*Version: 1.0*
