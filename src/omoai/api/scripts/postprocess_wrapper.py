"""Postprocess script wrapper ensuring correct working directory and command structure."""

from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_postprocess_script(
    asr_json_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
) -> None:
    """
    Invoke the top-level postprocess script as a module with cwd set to project root.

    Expected command pattern (validated by tests):
      [sys.executable, "-m", "scripts.post", "--asr-json", asr_json_path, "--out", output_path, ("--config", config_path)?]
    """
    # Log diagnostic information for CUDA multiprocessing issue
    # Keep diagnostics lightweight; avoid importing heavy CUDA libraries here
    try:
        logger.info(
            f"Current multiprocessing start method: {multiprocessing.get_start_method()}"
        )
    except Exception:
        logger.info("Current multiprocessing start method: unknown")

    asr_json_path = str(asr_json_path)
    output_path = str(output_path)
    cmd = [
        sys.executable,
        "-m",
        "scripts.post",
        "--asr-json",
        asr_json_path,
        "--out",
        output_path,
    ]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    # Project root: src/omoai/api/scripts/postprocess_wrapper.py -> .../src -> project root
    project_root = Path(__file__).resolve().parents[4]

    # Set environment variable to force spawn method for CUDA compatibility
    env = os.environ.copy()
    env["MULTIPROCESSING_START_METHOD"] = "spawn"

    # Add CUDA isolation to prevent re-initialization issues
    env["CUDA_VISIBLE_DEVICES"] = env.get(
        "CUDA_VISIBLE_DEVICES", "0"
    )  # Ensure consistent GPU visibility
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # Force vLLM to use spawn method
    env["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer warnings in subprocess

    logger.info(f"Running postprocess command: {' '.join(cmd)}")
    logger.info(f"Working directory: {project_root}")
    logger.info(
        f"CUDA environment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )

    result = subprocess.run(
        cmd, cwd=project_root, capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Postprocess script failed with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
