from pathlib import Path
import json

from omoai.logging_system.logger import setup_logging, get_logger
from omoai.logging_system.config import LoggingConfig


def test_structured_logging_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "omoai-log.jsonl"
    cfg = LoggingConfig(
        level="INFO",
        enable_console=False,
        enable_file=True,
        log_file=log_file,
        rotation="1 MB",
        retention=1,
        enqueue=False,
    )

    # Initialize logging and emit a message
    setup_logging(cfg)
    logger = get_logger(__name__)
    logger.info("test-message", extra={"component": "test"})

    # Ensure file was written and JSON can be parsed
    assert log_file.exists() and log_file.stat().st_size > 0
    last_line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    obj = json.loads(last_line)
    assert obj.get("message") == "test-message"
    # Loguru JSON should contain 'level'/'timestamp'/'message' and extras
    assert obj.get("level")
    assert obj.get("timestamp")
    assert obj.get("extra", {}).get("component") == "test"

