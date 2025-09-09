# API Component Documentation

## Overview

The API component provides a comprehensive RESTful interface for the OmoAI audio processing pipeline. It implements a dual-mode architecture that balances performance and reliability through intelligent service selection.

## Domain Responsibilities

The API domain is responsible for:

- Exposing audio processing capabilities through HTTP endpoints
- Handling file uploads and multipart form data
- Managing request/response validation and serialization
- Providing flexible output formatting options
- Implementing intelligent service mode selection
- Ensuring robust error handling and status reporting
- Offering interactive API documentation

## Component Architecture

### Core Modules

#### 1. Application Layer (`src/omoai/api/app.py`)

**Purpose**: Litestar application setup and configuration

**Key Classes**:

- `create_app()`: Factory function that creates and configures the Litestar application
- `app`: Global application instance
- `main()`: Entry point for running the API server

**Responsibilities**:

- Application configuration with request size limits
- Route handler registration
- Logging configuration
- Server startup and management

**File Location**: [`src/omoai/api/app.py`](src/omoai/api/app.py:1)

#### 2. Controllers Layer

##### Main Controller (`src/omoai/api/main_controller.py`)

**Purpose**: Handles the primary `/pipeline` endpoint and root redirection

**Key Classes**:

- `MainController`: Litestar controller class with pipeline endpoint handlers

**Key Methods**:

- `root()`: Redirects root URL to OpenAPI schema documentation
- `pipeline()`: Main endpoint for full audio processing pipeline

**Responsibilities**:

- Processing multipart form data with audio file uploads
- Parsing and validating query parameters for output formatting
- Coordinating the complete pipeline execution
- Generating appropriate response formats based on request parameters
- Managing output persistence to disk when configured

**File Location**: [`src/omoai/api/main_controller.py`](src/omoai/api/main_controller.py:1)

##### Preprocess Controller (`src/omoai/api/preprocess_controller.py`)

**Purpose**: Handles audio preprocessing through the `/preprocess` endpoint

**Key Classes**:

- `PreprocessController`: Controller for audio preprocessing operations

**Responsibilities**:

- Audio format validation and conversion
- Audio normalization to 16kHz mono PCM16 format
- Temporary file management for processed audio

**File Location**: [`src/omoai/api/preprocess_controller.py`](src/omoai/api/preprocess_controller.py:1)

##### ASR Controller (`src/omoai/api/asr_controller.py`)

**Purpose**: Handles automatic speech recognition through the `/asr` endpoint

**Key Classes**:

- `ASRController`: Controller for ASR processing operations

**Responsibilities**:

- Loading and managing ASR models
- Processing preprocessed audio files
- Generating transcriptions with timestamps and confidence scores

**File Location**: [`src/omoai/api/asr_controller.py`](src/omoai/api/asr_controller.py:1)

##### Postprocess Controller (`src/omoai/api/postprocess_controller.py`)

**Purpose**: Handles text post-processing through the `/postprocess` endpoint

**Key Classes**:

- `PostprocessController`: Controller for text post-processing operations

**Responsibilities**:

- Applying punctuation and capitalization to raw transcripts
- Generating summaries in multiple formats
- Calculating quality metrics and diffs

**File Location**: [`src/omoai/api/postprocess_controller.py`](src/omoai/api/postprocess_controller.py:1)

#### 3. Services Layer

##### Enhanced Services (`src/omoai/api/services_enhanced.py`)

**Purpose**: Intelligent service management with automatic fallback capabilities

**Key Classes**:

- `ServiceMode`: Enumeration defining service modes (SCRIPT_BASED, IN_MEMORY, AUTO)
- `_BytesUploadFile`: Lightweight UploadFile wrapper for testing and benchmarks

**Key Functions**:

- `get_service_mode()`: Determines active service mode based on configuration
- `should_use_in_memory_service()`: Health check for in-memory service availability
- `preprocess_audio_service()`: Smart preprocessing with automatic fallback
- `asr_service()`: Smart ASR processing with automatic fallback
- `postprocess_service()`: Smart post-processing with automatic fallback
- `run_full_pipeline()`: Smart full pipeline execution with automatic fallback
- `get_service_status()`: Comprehensive service status reporting
- `warmup_services()`: Model preloading for performance optimization
- `benchmark_service_performance()`: Performance comparison between service modes

**Responsibilities**:

- Service mode selection and management
- Health monitoring and graceful fallback
- Performance optimization through model caching
- Service status reporting and diagnostics

**File Location**: [`src/omoai/api/services_enhanced.py`](src/omoai/api/services_enhanced.py:1)

##### V1 Services (`src/omoai/api/services.py`)

**Purpose**: Script-based service implementations for maximum compatibility

**Key Functions**:

- `preprocess_audio_service()`: Script-based audio preprocessing
- `asr_service()`: Script-based ASR processing
- `postprocess_service()`: Script-based post-processing
- `run_full_pipeline()`: Script-based full pipeline execution

**Responsibilities**:

- Executing external scripts for audio processing
- Managing temporary files and process communication
- Handling script failures and timeouts

**File Location**: [`src/omoai/api/services.py`](src/omoai/api/services.py:1)

##### V2 Services (`src/omoai/api/services_v2.py`)

**Purpose**: High-performance in-memory service implementations

**Key Functions**:

- `preprocess_audio_service_v2()`: In-memory audio preprocessing
- `asr_service_v2()`: In-memory ASR processing with cached models
- `postprocess_service_v2()`: In-memory post-processing with cached models
- `run_full_pipeline_v2()`: In-memory full pipeline execution
- `health_check_models()`: Health check for cached models

**Responsibilities**:

- High-performance in-memory processing
- Model caching and lifecycle management
- Memory-efficient processing without temporary files

**File Location**: [`src/omoai/api/services_v2.py`](src/omoai/api/services_v2.py:1)

#### 4. Models Layer (`src/omoai/api/models.py`)

**Purpose**: Pydantic models for request/response validation and serialization

**Key Classes**:

- `PipelineRequest`: Request model for pipeline endpoint
- `PreprocessRequest`: Request model for preprocessing endpoint
- `ASRRequest`: Request model for ASR endpoint
- `PostprocessRequest`: Request model for post-processing endpoint
- `OutputFormatParams`: Query parameter model for output formatting
- `QualityMetrics`: Quality metrics response model
- `HumanReadableDiff`: Diff visualization response model
- `PipelineResponse`: Standard pipeline response model
- `PreprocessResponse`: Preprocessing response model
- `ASRResponse`: ASR response model
- `PostprocessResponse`: Post-processing response model

**Responsibilities**:

- Request validation and type safety
- Response serialization and formatting
- API contract definition
- OpenAPI schema generation

**File Location**: [`src/omoai/api/models.py`](src/omoai/api/models.py:1)

#### 5. Support Modules

##### Health Module (`src/omoai/api/health.py`)

**Purpose**: Health check endpoint for system monitoring

**Key Functions**:

- `health_check()`: Comprehensive system health check

**Responsibilities**:

- Checking system dependencies (ffmpeg, config files, scripts)
- Reporting service mode and model status
- Providing system status information

**File Location**: [`src/omoai/api/health.py`](src/omoai/api/health.py:1)

##### Exceptions Module (`src/omoai/api/exceptions.py`)

**Purpose**: Custom exception classes for API error handling

**Key Classes**:

- `AudioProcessingException`: Base exception for audio processing errors
- `ConfigurationException`: Exception for configuration-related errors
- `ServiceUnavailableException`: Exception for service unavailability

**Responsibilities**:

- Providing meaningful error information
- Enabling proper HTTP status code mapping
- Supporting error recovery and debugging

**File Location**: [`src/omoai/api/exceptions.py`](src/omoai/api/exceptions.py:1)

##### Logging Module (`src/omoai/api/logging.py`)

**Purpose**: Logging configuration for the API component

**Key Functions**:

- `configure_logging()`: Sets up structured logging configuration

**Responsibilities**:

- Configuring log formats and levels
- Setting up log handlers and outputs
- Supporting structured logging for better observability

**File Location**: [`src/omoai/api/logging.py`](src/omoai/api/logging.py:1)

##### Singletons Module (`src/omoai/api/singletons.py`)

**Purpose**: Model singleton management for performance optimization

**Key Functions**:

- `get_asr_model()`: Retrieves or creates the ASR model singleton
- `get_llm_model()`: Retrieves or creates the LLM model singleton
- `preload_all_models()`: Preloads all models for optimal performance

**Responsibilities**:

- Managing model lifecycle and caching
- Ensuring thread-safe model access
- Optimizing performance through model reuse

**File Location**: [`src/omoai/api/singletons.py`](src/omoai/api/singletons.py:1)

## Data Flow

### Request Processing Flow

1. **Request Reception**: Litestar receives HTTP request at `/pipeline` endpoint
2. **Validation**: Pydantic models validate request data and query parameters
3. **Service Selection**: `services_enhanced` determines appropriate service mode
4. **Processing**: Selected service (v1 or v2) processes the audio file
5. **Response Formatting**: Response is formatted based on query parameters
6. **Response Return**: Formatted response is returned to client

### Service Mode Decision Flow

1. **Configuration Check**: Read service mode from config file
2. **Environment Override**: Check for environment variable override
3. **Health Assessment**: If in auto mode, check model availability
4. **Mode Selection**: Choose between memory, script, or auto mode
5. **Service Execution**: Execute pipeline with selected mode
6. **Fallback Handling**: Gracefully fallback if primary mode fails

## Integration Points

### External Dependencies

- **Litestar Framework**: Web framework providing routing, middleware, and request handling
- **Pydantic**: Data validation and serialization using Pydantic models
- **Uvicorn**: ASGI server for running the application
- **FFmpeg**: External dependency for audio processing (via scripts)

### Internal Dependencies

- **Configuration System**: Uses `src.omoai.config` for all configuration needs
- **Pipeline Components**: Integrates with `src.omoai.pipeline` for in-memory processing
- **Output System**: Uses `src.omoai.output` for generating formatted outputs
- **Logging System**: Integrates with `src.omoai.logging_system` for structured logging

## Configuration

### Service Mode Configuration

The API supports three service modes configurable in `config.yaml`:

```yaml
api:
  service_mode: "auto" # "auto", "memory", or "script"
```

### API Server Configuration

```yaml
api:
  host: "127.0.0.1"
  port: 8000
  max_body_size_mb: 100
  request_timeout_seconds: 300
  temp_dir: "/tmp"
  cleanup_temp_files: true
  enable_progress_output: false
```

## Error Handling Strategy

The API implements comprehensive error handling:

1. **Validation Errors**: 422 status codes with detailed field error information
2. **File Size Errors**: 413 status codes for oversized uploads
3. **Processing Errors**: 500 status codes with error details
4. **Service Unavailable**: 503 status codes when models cannot be loaded
5. **Client Errors**: 400 status codes for malformed requests

## Performance Considerations

### Optimization Strategies

1. **Model Caching**: Singletons ensure models are loaded only once
2. **Service Mode Selection**: Automatic selection of fastest available mode
3. **Memory Management**: Efficient handling of large audio files
4. **Concurrent Processing**: Support for multiple simultaneous requests

### Monitoring Points

1. **Request Processing Time**: Tracked for each endpoint
2. **Service Mode Performance**: Benchmarked between v1 and v2
3. **Model Loading Time**: Monitored during startup and warmup
4. **Memory Usage**: Tracked for large file processing

## Testing Strategy

### Unit Tests

- Controller request/response handling
- Model validation and serialization
- Service mode selection logic
- Error handling and status codes

### Integration Tests

- Full pipeline execution through API
- Service mode fallback behavior
- File upload and processing
- Output format generation

### Performance Tests

- Service mode comparison benchmarks
- Concurrent request handling
- Large file processing limits
- Model loading and warmup timing

## Security Considerations

### Input Validation

- File type and size validation
- Query parameter validation
- Path traversal protection
- Malformed request handling

### Configuration Security

- Secure defaults for sensitive settings
- Environment variable override support
- Configuration validation on startup
- Warning for dangerous settings (trust_remote_code)

### Output Security

- Sensitive data filtering in error messages
- Secure temporary file handling
- Configurable progress output control
- Request size limiting
