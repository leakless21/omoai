"""Configuration management for the API."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API-specific configuration."""
    host: str
    port: int
    max_body_size_mb: int
    request_timeout_seconds: int
    temp_dir: str
    cleanup_temp_files: bool
    enable_progress_output: bool
    health_check_dependencies: List[str]


@dataclass
class PathsConfig:
    """Paths configuration."""
    chunkformer_dir: str
    chunkformer_checkpoint: str
    out_dir: str


@dataclass
class AppConfig:
    """Main application configuration."""
    api: APIConfig
    paths: PathsConfig
    raw_config: Dict[str, Any]  # Keep the full config for script compatibility
    config_path: Path


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file. If None, searches for config.yaml
                    in the project root.
    
    Returns:
        AppConfig object with parsed configuration
        
    Raises:
        FileNotFoundError: If config file is not found
        ValueError: If config file is invalid
    """
    if config_path is None:
        # Search for config.yaml in the project root
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config.yaml"
        
        # Also check current working directory
        if not config_path.exists():
            config_path = Path.cwd() / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    
    # Extract API configuration with defaults
    api_config = raw_config.get("api", {})
    api = APIConfig(
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        max_body_size_mb=api_config.get("max_body_size_mb", 100),
        request_timeout_seconds=api_config.get("request_timeout_seconds", 300),
        temp_dir=api_config.get("temp_dir", "/tmp"),
        cleanup_temp_files=api_config.get("cleanup_temp_files", True),
        enable_progress_output=api_config.get("enable_progress_output", True),
        health_check_dependencies=api_config.get("health_check_dependencies", [
            "ffmpeg", "config_file", "asr_script", "postprocess_script"
        ])
    )
    
    # Extract paths configuration
    paths_config = raw_config.get("paths", {})
    paths = PathsConfig(
        chunkformer_dir=paths_config.get("chunkformer_dir", ""),
        chunkformer_checkpoint=paths_config.get("chunkformer_checkpoint", ""),
        out_dir=paths_config.get("out_dir", "data/output")
    )
    
    return AppConfig(
        api=api,
        paths=paths,
        raw_config=raw_config,
        config_path=config_path
    )


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config