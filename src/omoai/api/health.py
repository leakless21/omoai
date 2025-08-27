"""Health check module for the API."""
import subprocess
import sys
from pathlib import Path
from litestar import get
from litestar.response import Response

from omoai.api.config import get_config


@get("/health")
async def health_check() -> Response[dict]:
    """
    Comprehensive health check endpoint.
    
    Checks:
    - API configuration loading
    - External dependencies (ffmpeg, scripts)
    - Config file accessibility
    
    Returns:
        Response with health status and details
    """
    status = "healthy"
    details = {}
    status_code = 200
    
    try:
        config = get_config()
        
        # Check each dependency based on configuration
        for dependency in config.api.health_check_dependencies:
            if dependency == "ffmpeg":
                try:
                    subprocess.run(["ffmpeg", "-version"], 
                                 capture_output=True, check=True, timeout=10)
                    details["ffmpeg"] = "available"
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    details["ffmpeg"] = "unavailable"
                    status = "unhealthy"
                    
            elif dependency == "config_file":
                if config.config_path.exists():
                    details["config_file"] = f"found at {config.config_path}"
                else:
                    details["config_file"] = f"not found at {config.config_path}"
                    status = "unhealthy"
                    
            elif dependency == "asr_script":
                asr_script = Path("scripts/asr.py")
                if asr_script.exists():
                    details["asr_script"] = "found"
                else:
                    details["asr_script"] = "not found"
                    status = "unhealthy"
                    
            elif dependency == "postprocess_script":
                post_script = Path("scripts/post.py")
                if post_script.exists():
                    details["postprocess_script"] = "found"
                else:
                    details["postprocess_script"] = "not found"
                    status = "unhealthy"
        
        # Check temp directory
        temp_dir = Path(config.api.temp_dir)
        if temp_dir.exists() and temp_dir.is_dir():
            details["temp_dir"] = f"accessible at {temp_dir}"
        else:
            details["temp_dir"] = f"inaccessible at {temp_dir}"
            status = "degraded"  # Not critical but worth noting
            
        # Add configuration info
        details["config_loaded"] = "yes"
        details["max_body_size"] = f"{config.api.max_body_size_mb}MB"
        
    except Exception as e:
        status = "unhealthy"
        details["error"] = f"Configuration error: {str(e)}"
        status_code = 500
    
    if status == "unhealthy":
        status_code = 500
    elif status == "degraded":
        status_code = 200  # Still operational
        
    return Response(
        content={"status": status, "details": details}, 
        status_code=status_code
    )