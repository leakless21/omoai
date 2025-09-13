import sys
from pathlib import Path
import os
import pytest


# Ensure src/ is on sys.path once for tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



@pytest.fixture(scope="session")
def test_mp3_path() -> Path | None:
    """Locate a real MP3 test file if available.

    Search order:
    - env OMOAI_TEST_MP3
    - data/input/testaudio.mp3
    - tests/assets/testaudio.mp3
    - fixtures/testaudio.mp3
    Returns Path or None if not found.
    """
    candidates: list[Path] = []
    env_path = os.environ.get("OMOAI_TEST_MP3")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("tests/assets/testaudio.mp3"),
            Path("data/input/testaudio.mp3"),
            Path("fixtures/testaudio.mp3"),
        ]
    )
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None
