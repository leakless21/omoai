#!/usr/bin/env python3
"""
Test script to verify PyTorch optimizations are working correctly.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
from omoai.pipeline.asr import run_asr


class TestPyTorchOptimizations(unittest.TestCase):
    """Test PyTorch optimization changes."""

    def test_debug_empty_cache_environment_flag(self):
        """Test that DEBUG_EMPTY_CACHE respects environment variable."""
        # Test default (false)
        import importlib
        importlib.reload(asr)
        self.assertFalse(asr.DEBUG_EMPTY_CACHE)
        
        # Test when set to true
        with patch.dict(os.environ, {"OMOAI_DEBUG_EMPTY_CACHE": "true"}):
            importlib.reload(asr)
            self.assertTrue(asr.DEBUG_EMPTY_CACHE)
        
        # Test case insensitive
        with patch.dict(os.environ, {"OMOAI_DEBUG_EMPTY_CACHE": "TRUE"}):
            importlib.reload(asr)
            self.assertTrue(asr.DEBUG_EMPTY_CACHE)
            
        # Test false value
        with patch.dict(os.environ, {"OMOAI_DEBUG_EMPTY_CACHE": "false"}):
            importlib.reload(asr)
            self.assertFalse(asr.DEBUG_EMPTY_CACHE)

    def test_api_controller_debug_flag(self):
        """Test that API controller also respects debug flag."""
        with patch.dict(os.environ, {"OMOAI_DEBUG_EMPTY_CACHE": "true"}):
            # Import fresh to pick up env var
            import importlib
            if "src.omoai.api.asr_controller" in sys.modules:
                importlib.reload(sys.modules["src.omoai.api.asr_controller"])
            else:
                import src.omoai.api.asr_controller
            self.assertTrue(src.omoai.api.asr_controller.DEBUG_EMPTY_CACHE)

    def test_empty_cache_logic_in_code(self):
        """Test that empty_cache logic is correctly implemented in the code."""
        # Read the ASR script code and verify the logic
        asr_code = (project_root / "src" / "omoai" / "pipeline" / "asr.py").read_text()
        
        # Should have the debug flag check
        self.assertIn("DEBUG_EMPTY_CACHE", asr_code)
        self.assertIn("if DEBUG_EMPTY_CACHE and device.type == \"cuda\":", asr_code)
        
        # Should not have unconditional empty_cache calls
        lines = asr_code.split('\n')
        for i, line in enumerate(lines):
            if "torch.cuda.empty_cache()" in line:
                # Check that it's conditional (preceded by an if statement)
                prev_lines = lines[max(0, i-5):i]
                has_conditional = any("if" in prev_line and "DEBUG_EMPTY_CACHE" in prev_line for prev_line in prev_lines)
                self.assertTrue(has_conditional, f"Line {i+1}: empty_cache call should be conditional")

    def test_inference_mode_usage(self):
        """Test that inference_mode context manager is used."""
        # This is a structural test - we've already implemented the change
        # Just verify the change is in the code
        asr_code = (project_root / "src" / "omoai" / "pipeline" / "asr.py").read_text()
        self.assertIn("torch.inference_mode()", asr_code)
        self.assertNotIn("torch.no_grad()", asr_code.replace("# Use inference_mode for better performance over no_grad", ""))

    def test_autocast_explicit_usage(self):
        """Test that autocast uses explicit parameters."""
        asr_code = (project_root / "src" / "omoai" / "pipeline" / "asr.py").read_text()
        # Should use explicit dtype and enabled parameters
        self.assertIn("dtype=amp_dtype", asr_code)
        self.assertIn("enabled=(amp_dtype is not None)", asr_code)

if __name__ == "__main__":
    unittest.main()de)

if __name__ == "__main__":
    unittest.main()