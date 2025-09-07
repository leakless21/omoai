from litestar import Litestar
from pathlib import Path

from omoai.config.schemas import get_config
from omoai.api.main_controller import MainController
from omoai.api.preprocess_controller import PreprocessController
from omoai.api.asr_controller import ASRController
from omoai.api.postprocess_controller import PostprocessController
from omoai.api.health import health_check
from omoai.api.logging import configure_logging


def create_app() -> Litestar:
    """Create the Litestar application with configuration-based settings."""
    config = get_config()
    
    return Litestar(
        route_handlers=[MainController, PreprocessController, ASRController, PostprocessController, health_check],
        on_startup=[],
        logging_config=configure_logging(),
        request_max_body_size=config.api.max_body_size_mb * 1024 * 1024,  # Convert MB to bytes
    )


# Create the app instance
app = create_app()


def main() -> None:
    """Main entry point for the API server."""
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "omoai.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=True
    )