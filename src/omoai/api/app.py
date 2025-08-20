from litestar import Litestar
from pathlib import Path

from src.omoai.api.config import get_config
from src.omoai.api.main_controller import MainController
from src.omoai.api.preprocess_controller import PreprocessController
from src.omoai.api.asr_controller import ASRController
from src.omoai.api.postprocess_controller import PostprocessController
from src.omoai.api.health import health_check
from src.omoai.api.logging import configure_logging


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