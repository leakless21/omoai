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
from scripts.asr import run_asr


class TestPyTorchOptimizations(unittest.TestCase):
    """Test PyTorch optimization changes."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "paths": {
                "chunkformer_dir": str(project_root / "chunkformer"),
                "chunkformer_checkpoint": "dummy_checkpoint"
            },
            "asr": {
                "total_batch_duration_s": 60,
                "chunk_size": 16,
                "left_context_size": 32,
                "right_context_size": 32,
                "device": "cpu",
                "autocast_dtype": None
            }
        }

    def test_debug_empty_cache_environment_flag(self):
        """Test that DEBUG_EMPTY_CACHE respects environment variable."""
        # Test default (false)
        import scripts.asr
        # Reload to get fresh import
        import importlib
        importlib.reload(scripts.asr)
        self.assertFalse(scripts.asr.DEBUG_EMPTY_CACHE)
        
        # Test when set to true
        with patch.dict(os.environ, {"OMOAI_DEBUG_EMPTY_CACHE": "true"}):
            importlib.reload(scripts.asr)
            self.assertTrue(scripts.asr.DEBUG_EMPTY_CACHE)
        
        # Test case insensitive
        with patch.dict(os.environ, {"OMOAI_DEBUG_EMPTY_CACHE": "TRUE"}):
            importlib.reload(scripts.asr)
            self.assertTrue(scripts.asr.DEBUG_EMPTY_CACHE)
            
        # Test false value
        with patch.dict(os.environ, {"OMOAI_DEBUG_EMPTY_CACHE": "false"}):
            importlib.reload(scripts.asr)
            self.assertFalse(scripts.asr.DEBUG_EMPTY_CACHE)

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
        asr_code = (project_root / "scripts" / "asr.py").read_text()
        
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
        asr_code = (project_root / "scripts" / "asr.py").read_text()
        self.assertIn("torch.inference_mode()", asr_code)
        self.assertNotIn("torch.no_grad()", asr_code.replace("# Use inference_mode for better performance over no_grad", ""))

    def test_autocast_explicit_usage(self):
        """Test that autocast uses explicit parameters."""
        asr_code = (project_root / "scripts" / "asr.py").read_text()
        # Should use explicit dtype and enabled parameters
        self.assertIn("dtype=amp_dtype", asr_code)
        self.assertIn("enabled=(amp_dtype is not None)", asr_code)
        
    def test_post_script_optimizations(self):
        """Test that post.py script has proper optimizations."""
        post_code = (project_root / "scripts" / "post.py").read_text()
        
        # Should have debug flag
        self.assertIn("DEBUG_EMPTY_CACHE", post_code)
        
        # Should have conditional empty_cache calls
        self.assertIn("if DEBUG_EMPTY_CACHE", post_code)
        
        # Check that empty_cache calls are properly conditional
        lines = post_code.split('\n')
        for i, line in enumerate(lines):
            if "torch.cuda.empty_cache()" in line:
                # Look at preceding lines to ensure it's conditional
                context_lines = lines[max(0, i-5):i+1]
                context = '\n'.join(context_lines)
                # Should be inside an if block or with suppress block that's conditional
                has_conditional = any("if DEBUG_EMPTY_CACHE" in ctx_line or "if DEBUG_EMPTY_CACHE or" in ctx_line 
                                    for ctx_line in context_lines)
                self.assertTrue(has_conditional, f"Line {i+1}: empty_cache should be conditional on DEBUG_EMPTY_CACHE")


    def test_chunkformer_decode_optimizations(self):
        """Test that chunkformer decode.py has optimizations."""
        decode_code = (project_root / "src" / "omoai" / "chunkformer" / "decode.py").read_text()
        
        # Should use inference_mode instead of no_grad
        self.assertIn("@torch.inference_mode()", decode_code)
        
        # Count remaining @torch.no_grad() decorators - should be 0
        no_grad_count = decode_code.count("@torch.no_grad()")
        self.assertEqual(no_grad_count, 0, "All @torch.no_grad() should be replaced with @torch.inference_mode()")
        
        # Should not have per-chunk empty_cache
        self.assertNotIn("torch.cuda.empty_cache()", decode_code)


if __name__ == "__main__":
    unittest.main()
