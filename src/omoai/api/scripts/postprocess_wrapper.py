"""Postprocess script wrapper ensuring correct working directory and command structure."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Optional, Union


def run_postprocess_script(
    asr_json_path: Union[str, Path],
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Invoke the top-level postprocess script as a module with cwd set to project root.

    Expected command pattern (validated by tests):
      [sys.executable, "-m", "scripts.post", "--asr-json", asr_json_path, "--out", output_path, ("--config", config_path)?]
    """
    asr_json_path = str(asr_json_path)
    output_path = str(output_path)
    cmd = [sys.executable, "-m", "scripts.post", "--asr-json", asr_json_path, "--out", output_path]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    # Project root: src/omoai/api/scripts/postprocess_wrapper.py -> .../src -> project root
    project_root = Path(__file__).resolve().parents[4]

    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Postprocess script failed with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )