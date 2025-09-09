"""ASR script wrapper ensuring correct working directory and command structure."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Optional, Union


def run_asr_script(
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Invoke the top-level ASR script as a module with cwd set to project root.

    Expected command pattern (validated by tests):
      [sys.executable, "-m", "scripts.asr", "--audio", audio_path, "--out", output_path, ("--config", config_path)?]
    """
    audio_path = str(audio_path)
    output_path = str(output_path)
    cmd = [sys.executable, "-m", "scripts.asr", "--audio", audio_path, "--out", output_path]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    # Project root: src/omoai/api/scripts/asr_wrapper.py -> .../src -> project root
    project_root = Path(__file__).resolve().parents[4]

    # Let CalledProcessError propagate to callers for test assertions
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ASR script failed with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )