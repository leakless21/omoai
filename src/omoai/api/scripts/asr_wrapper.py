"""ASR script wrapper ensuring correct working directory and command structure.

On non-zero exit codes, raises AudioProcessingException with a detailed message
including return code, stdout and stderr. This aligns wrapper behavior with
tests that call the wrapper directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from omoai.api.exceptions import AudioProcessingException


def run_asr_script(
    audio_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
) -> None:
    """
    Invoke the top-level ASR script as a module with cwd set to project root.

    Expected command pattern (validated by tests):
      [sys.executable, "-m", "scripts.asr", "--audio", audio_path, "--out", output_path, ("--config", config_path)?]
    """
    audio_path = str(audio_path)
    output_path = str(output_path)
    cmd = [
        sys.executable,
        "-m",
        "scripts.asr",
        "--audio",
        audio_path,
        "--out",
        output_path,
    ]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    # Project root: src/omoai/api/scripts/asr_wrapper.py -> .../src -> project root
    project_root = Path(__file__).resolve().parents[4]

    # Let CalledProcessError propagate to callers for test assertions
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        # Provide detailed error context expected by tests
        message = (
            f"ASR processing failed with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        raise AudioProcessingException(message)
