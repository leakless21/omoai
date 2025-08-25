"""Unit test to verify ASR script import fix and prevent regression."""

import sys
import subprocess
from pathlib import Path
import pytest


def test_asr_script_import():
    """Test that the ASR script can successfully import the omoai.chunkformer module."""
    
    # Test the import by running the ASR script with a minimal command that triggers the import
    # We'll use --help to avoid needing actual audio files
    cmd = [
        sys.executable,
        "-m",
        "scripts.asr",
        "--help",
    ]
    
    # Run from the project root directory
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    
    # The script should either:
    # 1. Show help text (return code 0) if imports work, or
    # 2. Fail with a different error (return code != 1) if imports work but other issues exist
    # We specifically want to ensure it doesn't fail with ModuleNotFoundError (return code 1)
    
    # If the script shows help text, imports are working
    if result.returncode == 0:
        assert "usage:" in result.stdout or "ASR wrapper for ChunkFormer" in result.stdout
        return
    
    # If it fails, make sure it's not due to ModuleNotFoundError
    assert result.returncode != 1, f"Script failed with ModuleNotFoundError: {result.stderr}"
    
    # If it fails with a different error, that's acceptable for this test
    # since we're only testing that imports work
    assert "ModuleNotFoundError: No module named 'omoai'" not in result.stderr


def test_ensure_chunkformer_on_path_function():
    """Test that ensure_chunkformer_on_path properly adds src directory to sys.path."""
    
    # Import the function directly
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from asr import ensure_chunkformer_on_path
    
    # Save original sys.path
    original_path = sys.path.copy()
    
    try:
        # Clear any existing paths that might interfere
        sys.path = [p for p in sys.path if not str(Path(__file__).parent.parent / "src") in p]
        
        # Call the function
        chunkformer_dir = Path(__file__).parent.parent / "chunkformer"
        ensure_chunkformer_on_path(chunkformer_dir)
        
        # Verify that src directory was added to sys.path
        src_dir = str(Path(__file__).parent.parent / "src")
        assert src_dir in sys.path, f"Expected {src_dir} to be in sys.path"
        
    finally:
        # Restore original sys.path
        sys.path = original_path


if __name__ == "__main__":
    test_asr_script_import()
    test_ensure_chunkformer_on_path_function()
    print("All tests passed!")