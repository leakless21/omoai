"""Health check module for the API."""
import subprocess
import sys
import os
from pathlib import Path
from litestar import get
from litestar.response import Response

from omoai.config.schemas import get_config


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

        # Helper: determine config file path candidates (consistent with loader discovery)
        env_cfg = os.environ.get("OMOAI_CONFIG")
        config_candidates = []
        if env_cfg:
            config_candidates.append(Path(env_cfg))
        # Prefer current working directory and repository root
        config_candidates.append(Path.cwd() / "config.yaml")
        # Project root relative to this file (approximate same search as loaders)
        config_candidates.append(Path(__file__).resolve().parents[4] / "config.yaml")

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
                # Search for an existing config file from candidates
                found = None
                for p in config_candidates:
                    try:
                        if p and p.exists():
                            found = p
                            break
                    except Exception:
                        continue

                if found:
                    details["config_file"] = f"found at {found}"
                else:
                    details["config_file"] = f"not found (searched: {[str(p) for p in config_candidates]})"
                    status = "unhealthy"

            elif dependency == "asr_script":
                # Legacy script dependency - no longer needed with pipeline modules
                details["asr_script"] = "deprecated (using pipeline modules)"

            elif dependency == "postprocess_script":
                # Legacy script dependency - no longer needed with pipeline modules  
                details["postprocess_script"] = "deprecated (using pipeline modules)"

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