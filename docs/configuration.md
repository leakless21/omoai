# Configuration Management

This document describes the configuration system for the OMOAI audio processing pipeline and API.

## Configuration File Structure

The project uses a single `config.yaml` file located in the project root. This file contains all configuration for both the command-line tools and the REST API.

### Main Sections

#### 1. Paths Configuration

```yaml
paths:
  chunkformer_dir: /home/cetech/omoai/chunkformer
  chunkformer_checkpoint: /home/cetech/omoai/models/chunkformer/chunkformer-large-vie
  out_dir: /home/cetech/omoai/data/output
```

#### 2. ASR Configuration

```yaml
asr:
  total_batch_duration_s: 1800
  chunk_size: 64
  left_context_size: 128
  right_context_size: 128
  device: cuda
  autocast_dtype: fp16
```

#### 3. LLM Configuration

```yaml
llm:
  model_id: cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit
  quantization: auto
  max_model_len: 50000
  gpu_memory_utilization: 0.90
  max_num_seqs: 2
  max_num_batched_tokens: 512
  trust_remote_code: true
```

#### 4. API Configuration (New)

```yaml
api:
  # Server configuration
  host: "0.0.0.0"
  port: 8000
  # Request limits
  max_body_size_mb: 100
  request_timeout_seconds: 300
  # File handling
  temp_dir: "/tmp"
  cleanup_temp_files: true
  # Progress output
  enable_progress_output: true
  # Health check configuration
  health_check_dependencies:
    - ffmpeg
    - config_file
    - asr_script
    - postprocess_script
```

## API Configuration Management

### Configuration Loader

The API uses a dedicated configuration management system in `omoai/config.py`:

- **Automatic Discovery**: Searches for `config.yaml` in project root or current directory
- **Type Safety**: Uses dataclasses for configuration validation
- **Global Instance**: Provides singleton access pattern for efficient loading
- **Error Handling**: Clear error messages for missing or invalid configuration

### Key Features

1. **Dynamic Path Resolution**: The API automatically finds the config file relative to the project structure
2. **Environment Flexibility**: Works in different environments without hardcoded paths
3. **Validation**: Type-safe configuration with dataclass validation
4. **Fallback Values**: Sensible defaults for all API settings

### Usage in Services

All API services now use the configuration system:

```python
from omoai.config import get_config

def service_function():
    config = get_config()
    temp_dir = config.api.temp_dir
    config_path = config.config_path
    # ... use configuration values
```

### Benefits

1. **Centralized Configuration**: Single source of truth for all settings
2. **Environment Independence**: No hardcoded paths or values
3. **Easy Customization**: Simple YAML editing to change behavior
4. **Consistent Behavior**: Same configuration used by CLI tools and API
5. **Health Monitoring**: Configuration status visible in health checks

## Configuration Parameters

### API-Specific Parameters

| Parameter                   | Default   | Description                              |
| --------------------------- | --------- | ---------------------------------------- |
| `host`                      | "0.0.0.0" | Server bind address                      |
| `port`                      | 8000      | Server port                              |
| `max_body_size_mb`          | 100       | Maximum request body size in MB          |
| `request_timeout_seconds`   | 300       | Request timeout                          |
| `temp_dir`                  | "/tmp"    | Temporary file directory                 |
| `cleanup_temp_files`        | true      | Whether to clean up temp files           |
| `enable_progress_output`    | true      | Show progress in terminal                |
| `health_check_dependencies` | [list]    | Dependencies to check in health endpoint |

### Customization Examples

#### Production Environment

```yaml
api:
  host: "127.0.0.1" # More restrictive binding
  port: 8080
  max_body_size_mb: 500 # Larger files
  temp_dir: "/var/tmp/omoai" # Dedicated temp space
  cleanup_temp_files: true
```

#### Development Environment

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  max_body_size_mb: 50 # Smaller for testing
  temp_dir: "./tmp" # Local temp directory
  enable_progress_output: true # Verbose output
```

## Running with uv

The project uses `uv` for virtual environment and dependency management. All commands should be run with `uv run` to ensure the correct environment:

```bash
# Start the API server (uvicorn factory)
uv run uvicorn omoai.api.app:create_app --factory --host 0.0.0.0 --port 8000

# Test configuration loading
uv run python -c "from omoai.config import get_config; print(get_config().api.max_body_size_mb)"

# Run other scripts
uv run python scripts/asr.py --help
```

The `uv run` command automatically activates the virtual environment and ensures all dependencies are available.

## Health Check Integration

The health check endpoint (`/health`) uses the configuration to:

1. **Verify Configuration Loading**: Confirms config.yaml is accessible
2. **Check Dependencies**: Tests each dependency listed in `health_check_dependencies`
3. **Validate Paths**: Ensures temp directories and script paths exist
4. **Report Settings**: Shows current configuration values

Example health check response:

```json
{
  "status": "healthy",
  "details": {
    "ffmpeg": "available",
    "config_file": "found at /home/cetech/omoai/config.yaml",
    "asr_script": "found",
    "postprocess_script": "found",
    "temp_dir": "accessible at /tmp",
    "config_loaded": "yes",
    "max_body_size": "100MB"
  }
}
```

## Migration from Hardcoded Values

The configuration system replaces previous hardcoded values:

- **Before**: Hardcoded paths like `"/home/cetech/omoai/config.yaml"`
- **After**: Dynamic discovery with `get_config().config_path`

- **Before**: Fixed temp directory `"/tmp"`
- **After**: Configurable via `get_config().api.temp_dir`

- **Before**: Hardcoded 100MB limit
- **After**: Configurable via `get_config().api.max_body_size_mb`

This provides much better flexibility and maintainability for different deployment environments.

### Service Mode

The application does not provide an in-memory mode. All stages run via script-based services for reliability and predictable resource usage. Any previous `api.service_mode` options are no longer applicable.
