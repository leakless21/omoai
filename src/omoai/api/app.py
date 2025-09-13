import logging
import os

import uvicorn
from litestar import Litestar
from litestar.exceptions import HTTPException
from litestar.response import Response

from omoai.api.asr_controller import ASRController
from omoai.api.exceptions import AudioProcessingException
from omoai.api.health import health_check
from omoai.api.main_controller import MainController
from omoai.api.postprocess_controller import PostprocessController
from omoai.api.preprocess_controller import PreprocessController
from omoai.config.schemas import get_config
from omoai.logging_system.logger import setup_logging


def global_exception_handler(request, exc):
    """Global exception handler for all uncaught exceptions with structured logs."""
    logger = logging.getLogger(__name__)

    # Basic request context (best-effort)
    req_path = str(
        getattr(getattr(request, "url", None), "path", getattr(request, "url", ""))
    )
    method = getattr(request, "method", "")

    if isinstance(exc, AudioProcessingException):
        logger.error(
            "Audio processing error",
            exc_info=exc,
            extra={
                "type": "AudioProcessingException",
                "status_code": getattr(exc, "status_code", 500),
                "detail": str(exc),
                "path": req_path,
                "method": method,
            },
        )
        return Response(
            content={"error": str(exc), "type": "AudioProcessingException"},
            status_code=exc.status_code,
        )
    elif isinstance(exc, HTTPException):
        logger.error(
            "HTTP error",
            exc_info=exc,
            extra={
                "type": "HTTPException",
                "status_code": getattr(exc, "status_code", 500),
                "detail": str(exc),
                "path": req_path,
                "method": method,
            },
        )
        return Response(
            content={"error": str(exc), "type": "HTTPException"},
            status_code=exc.status_code,
        )
    else:
        logger.exception(
            "Unhandled exception",
            extra={
                "type": type(exc).__name__,
                "path": req_path,
                "method": method,
            },
        )
        return Response(
            content={"error": "Internal server error", "type": "InternalServerError"},
            status_code=500,
        )


def create_app() -> Litestar:
    """Create the Litestar application with configuration-based settings."""
    config = get_config()
    # Configure the unified logging system once, using config.yaml
    setup_logging()

    return Litestar(
        route_handlers=[
            MainController,
            PreprocessController,
            ASRController,
            PostprocessController,
            health_check,
        ],
        on_startup=[],
        # Use unified stdlib logging configured by setup_logging()
        logging_config=None,
        request_max_body_size=config.api.max_body_size_mb
        * 1024
        * 1024,  # Convert MB to bytes,
        exception_handlers={
            Exception: global_exception_handler,
            AudioProcessingException: global_exception_handler,
            HTTPException: global_exception_handler,
        },
    )


def main():
    uvicorn.run(
        "omoai.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src", "scripts"],
        reload_excludes=[".venv/*"],
        workers=int(os.environ.get("UVICORN_WORKERS", 1)),
    )


if __name__ == "__main__":
    main()
