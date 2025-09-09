"""
Pydantic schemas for configuration validation.

This module provides comprehensive validation for the OMOAI configuration
with security defaults and type safety.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    """Configuration for file and directory paths."""
    
    chunkformer_dir: Path = Field(
        description="Path to the ChunkFormer source directory"
    )
    chunkformer_checkpoint: Path = Field(
        description="Path to the ChunkFormer model checkpoint"
    )
    out_dir: Path = Field(
        default=Path("data/output"),
        description="Output directory for artifacts"
    )
    
    @field_validator("chunkformer_dir", "chunkformer_checkpoint")
    @classmethod
    def validate_required_paths_exist(cls, v: Path) -> Path:
        """Validate that required paths exist."""
        if not v.exists():
            raise ValueError(f"Required path does not exist: {v}")
        return v
    
    @field_validator("out_dir")
    @classmethod
    def ensure_out_dir_exists(cls, v: Path) -> Path:
        """Ensure output directory exists or can be created."""
        try:
            v.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create output directory {v}: {e}")
        return v


class ASRConfig(BaseModel):
    """Configuration for Automatic Speech Recognition."""
    
    total_batch_duration_s: int = Field(
        default=1800,
        ge=60,  # Minimum 1 minute
        le=7200,  # Maximum 2 hours
        description="Maximum audio duration (seconds) per batch"
    )
    chunk_size: int = Field(
        default=64,
        ge=16,
        le=512,
        description="ChunkFormer chunk size"
    )
    left_context_size: int = Field(
        default=128,
        ge=0,
        le=1024,
        description="Left context size for ChunkFormer"
    )
    right_context_size: int = Field(
        default=128,
        ge=0,
        le=1024,
        description="Right context size for ChunkFormer"
    )
    device: Literal["cpu", "cuda", "auto"] = Field(
        default="auto",
        description="Device for ASR inference"
    )
    autocast_dtype: Optional[Literal["fp32", "bf16", "fp16"]] = Field(
        default="fp16",
        description="Autocast dtype for mixed precision"
    )
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device setting."""
        if v == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


class LLMConfig(BaseModel):
    """Configuration for Large Language Model inference."""
    
    model_id: Optional[str] = Field(
        default=None,
        description="Hugging Face model identifier"
    )
    quantization: Optional[str] = Field(
        default="auto",
        description="Quantization method (auto, awq, gptq, etc.)"
    )
    max_model_len: int = Field(
        default=8192,
        ge=512,
        le=200000,
        description="Maximum model context length"
    )
    gpu_memory_utilization: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization ratio"
    )
    max_num_seqs: int = Field(
        default=1,
        ge=1,
        le=256,
        description="Maximum number of concurrent sequences"
    )
    max_num_batched_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of batched tokens"
    )
    trust_remote_code: bool = Field(
        default=False,  # SECURITY: Default to False
        description="Whether to trust remote code execution (SECURITY RISK)"
    )
    
    @field_validator("trust_remote_code")
    @classmethod
    def warn_trust_remote_code(cls, v: bool) -> bool:
        """Warn about security implications of trust_remote_code."""
        if v:
            import warnings
            warnings.warn(
                "trust_remote_code=True enables arbitrary code execution from model repositories. "
                "This is a SECURITY RISK. Only enable if you trust the model source.",
                UserWarning,
                stacklevel=3
            )
        return v


class SamplingConfig(BaseModel):
    """Configuration for LLM sampling parameters."""
    
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )


class PunctuationConfig(BaseModel):
    """Configuration for punctuation restoration."""
    
    llm: LLMConfig
    preserve_original_words: bool = Field(
        default=True,
        description="Whether to preserve original word order"
    )
    auto_switch_ratio: float = Field(
        default=0.98,
        ge=0.5,
        le=1.0,
        description="Ratio of context used for token budget"
    )
    auto_margin_tokens: int = Field(
        default=128,
        ge=16,
        le=1024,
        description="Safety margin tokens"
    )
    enable_paragraphs: bool = Field(
        default=True,
        description="Enable paragraph breaks based on timing"
    )
    join_separator: str = Field(
        default=" ",
        description="Separator for joining segments"
    )
    paragraph_gap_seconds: float = Field(
        default=3.0,
        ge=0.5,
        le=30.0,
        description="Time gap threshold for paragraph breaks"
    )
    system_prompt: str = Field(
        description="System prompt for punctuation model"
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Sampling configuration"
    )


class SummarizationConfig(BaseModel):
    """Configuration for text summarization."""
    
    llm: LLMConfig
    map_reduce: bool = Field(
        default=False,
        description="Use map-reduce for very long texts"
    )
    auto_switch_ratio: float = Field(
        default=0.98,
        ge=0.5,
        le=1.0,
        description="Auto-switch to map-reduce ratio"
    )
    auto_margin_tokens: int = Field(
        default=256,
        ge=16,
        le=1024,
        description="Safety margin tokens"
    )
    system_prompt: str = Field(
        description="System prompt for summarization model"
    )
    sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(temperature=0.7),
        description="Sampling configuration"
    )


class TranscriptOutputConfig(BaseModel):
    """Configuration for transcript output options."""
    
    include_raw: bool = Field(
        default=True,
        description="Include raw transcript in outputs"
    )
    include_punct: bool = Field(
        default=True,
        description="Include punctuated transcript in outputs"
    )
    include_segments: bool = Field(
        default=True,
        description="Include timestamped segments in outputs"
    )
    timestamps: Literal["none", "s", "ms", "clock"] = Field(
        default="clock",
        description="Timestamp format (none, s=seconds, ms=milliseconds, clock=HH:MM:SS)"
    )
    wrap_width: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Text wrapping width (0=no wrapping)"
    )
    file_raw: str = Field(
        default="transcript.raw.txt",
        description="Raw transcript filename"
    )
    file_punct: str = Field(
        default="transcript.punct.txt", 
        description="Punctuated transcript filename"
    )
    file_srt: str = Field(
        default="transcript.srt",
        description="SRT subtitle filename"
    )
    file_vtt: str = Field(
        default="transcript.vtt",
        description="WebVTT subtitle filename"
    )
    file_segments: str = Field(
        default="segments.json",
        description="Segments JSON filename"
    )


class SummaryOutputConfig(BaseModel):
    """Configuration for summary output options."""
    
    mode: Literal["bullets", "abstract", "both", "none"] = Field(
        default="both",
        description="Summary generation mode"
    )
    bullets_max: int = Field(
        default=7,
        ge=1,
        le=20,
        description="Maximum number of bullet points"
    )
    abstract_max_chars: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum abstract length in characters"
    )
    language: str = Field(
        default="vi",
        description="Summary language (vi, en, etc.)"
    )
    file: str = Field(
        default="summary.md",
        description="Summary output filename"
    )


class OutputConfig(BaseModel):
    """Configuration for output formatting."""
    
    # Legacy fields (maintained for backward compatibility)
    write_separate_files: bool = Field(
        default=True,
        description="Write separate transcript and summary files"
    )
    transcript_file: str = Field(
        default="transcript.txt",
        description="Legacy transcript filename (use transcript.file_punct instead)"
    )
    summary_file: str = Field(
        default="summary.txt",
        description="Legacy summary filename (use summary.file instead)"
    )
    wrap_width: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Legacy text wrapping width (use transcript.wrap_width instead)"
    )
    
    # New structured configuration
    formats: List[Literal["json", "text", "srt", "vtt", "md"]] = Field(
        default=["json", "text"],
        description="Output formats to generate"
    )
    transcript: TranscriptOutputConfig = Field(
        default_factory=TranscriptOutputConfig,
        description="Transcript output configuration"
    )
    summary: SummaryOutputConfig = Field(
        default_factory=SummaryOutputConfig,
        description="Summary output configuration"
    )
    final_json: str = Field(
        default="final.json",
        description="Final JSON output filename"
    )

    # API-related output controls
    save_on_api: bool = Field(
        default=False,
        description="If true, the /pipeline API will persist configured outputs to disk"
    )
    save_formats_on_api: List[str] = Field(
        default_factory=lambda: ["final_json", "segments"],
        description="Which artifacts to persist when save_on_api is true (e.g. final_json, segments, transcript_punct, transcript_raw)"
    )
    api_output_dir: Optional[Path] = Field(
        default=None,
        description="Optional override directory to save API outputs (if not set, paths.out_dir is used)"
    )
    
    @model_validator(mode="after")
    def migrate_legacy_fields(self) -> "OutputConfig":
        """Migrate legacy fields to new structure for backward compatibility."""
        # Migrate wrap_width if it's non-default
        if self.wrap_width != 0 and self.transcript.wrap_width == 0:
            self.transcript.wrap_width = self.wrap_width
        
        # Migrate legacy filenames if they're non-default
        if self.transcript_file != "transcript.txt":
            self.transcript.file_punct = self.transcript_file
        
        if self.summary_file != "summary.txt":
            self.summary.file = self.summary_file
        
        # Ensure api_output_dir Path is resolved if provided as string in YAML
        if isinstance(self.api_output_dir, (str,)) and self.api_output_dir:
            try:
                self.api_output_dir = Path(self.api_output_dir)
            except Exception:
                # keep original value; validation of path happens later when used
                pass

        return self


class APIConfig(BaseModel):
    """Configuration for API server."""
    
    host: str = Field(
        default="127.0.0.1",  # SECURITY: Default to localhost
        description="API server host"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port"
    )
    max_body_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum request body size in MB"
    )
    request_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Request timeout in seconds"
    )
    temp_dir: Path = Field(
        default=Path("/tmp"),
        description="Temporary files directory"
    )
    cleanup_temp_files: bool = Field(
        default=True,
        description="Automatically cleanup temporary files"
    )
    enable_progress_output: bool = Field(
        default=False,  # SECURITY: Default to False for production
        description="Enable progress output (may leak information)"
    )
    health_check_dependencies: List[str] = Field(
        default_factory=lambda: ["ffmpeg", "config_file"],
        description="Dependencies to check in health endpoint"
    )
    
    service_mode: str = Field(
        default="auto",
        description="Service mode for API server ('auto', 'script', 'memory')"
    )
    @field_validator("temp_dir")
    @classmethod
    def validate_temp_dir(cls, v: Path) -> Path:
        """Validate temporary directory."""
        try:
            v.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = v / ".omoai_access_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise ValueError(f"Temporary directory {v} is not accessible: {e}")
        return v


class OmoAIConfig(BaseSettings):
    """Main OMOAI configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix="OMOAI_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",  # Reject unknown fields
    )
    
    paths: PathsConfig
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(
        description="Base LLM configuration (model_id required)"
    )
    punctuation: PunctuationConfig
    summarization: SummarizationConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    @model_validator(mode="after")
    def validate_config_consistency(self) -> "OmoAIConfig":
        """Validate cross-field consistency."""
        # Validate base LLM has model_id
        if not self.llm.model_id:
            raise ValueError("Base llm.model_id is required")
        
        # Ensure punctuation and summarization inherit base LLM model_id if not set
        if not self.punctuation.llm.model_id:
            self.punctuation.llm.model_id = self.llm.model_id
        
        if not self.summarization.llm.model_id:
            self.summarization.llm.model_id = self.llm.model_id
        
        # Validate trust_remote_code consistency
        if (self.llm.trust_remote_code or 
            self.punctuation.llm.trust_remote_code or 
            self.summarization.llm.trust_remote_code):
            import warnings
            warnings.warn(
                "At least one LLM configuration has trust_remote_code=True. "
                "This enables arbitrary code execution. Ensure you trust all model sources.",
                UserWarning,
                stacklevel=2
            )
        
        return self
    
    @classmethod
    def load_from_yaml(cls, config_path: Path) -> "OmoAIConfig":
        """Load configuration from YAML file with validation."""
        import yaml
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
        
        if not raw_config:
            raise ValueError(f"Empty configuration file: {config_path}")
        
        try:
            return cls(**raw_config)
        except Exception as e:
            raise ValueError(f"Configuration validation failed for {config_path}: {e}")
    
    def model_dump_yaml(self, **kwargs) -> str:
        """Export configuration as YAML string."""
        import yaml
        # Convert Path objects to strings for YAML serialization
        data = self.model_dump(mode="python", **kwargs)
        
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        data = convert_paths(data)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    def save_to_yaml(self, config_path: Path, **kwargs) -> None:
        """Save configuration to YAML file."""
        yaml_content = self.model_dump_yaml(**kwargs)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)


# Legacy compatibility functions
def load_config(config_path: Optional[Union[str, Path]] = None) -> OmoAIConfig:
    """Load and validate configuration from YAML file."""
    if config_path is None:
        # Search for config.yaml in common locations
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent.parent / "config.yaml",
        ]
        
        env_config = os.environ.get("OMOAI_CONFIG")
        if env_config:
            search_paths.insert(0, Path(env_config))
        
        config_path = None
        for path in search_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                f"Configuration file not found. Searched: {search_paths}. "
                f"Set OMOAI_CONFIG environment variable or create config.yaml"
            )
    
    return OmoAIConfig.load_from_yaml(Path(config_path))


def get_config() -> OmoAIConfig:
    """Get the global configuration instance (singleton pattern)."""
    if not hasattr(get_config, "_instance"):
        get_config._instance = load_config()
    return get_config._instance


def reload_config(config_path: Optional[Union[str, Path]] = None) -> OmoAIConfig:
    """Reload configuration and update global instance."""
    new_config = load_config(config_path)
    get_config._instance = new_config
    return new_config
