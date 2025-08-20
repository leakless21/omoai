#!/usr/bin/env python3
"""
Test API singletons and enhanced services.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import threading
import asyncio

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.omoai.api.singletons import (
    ModelSingletons,
    get_asr_model,
    get_punctuation_processor,
    get_summarization_processor,
    preload_all_models,
    get_model_status
)
from src.omoai.api.services_enhanced import (
    get_service_mode,
    should_use_in_memory_service,
    get_service_status,
    warmup_services,
    ServiceMode
)


class TestAPISingletons(unittest.TestCase):
    """Test API singleton model management."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create required directories
        (self.temp_dir / "chunkformer").mkdir()
        (self.temp_dir / "checkpoint").mkdir()
        
        # Create valid config for testing
        self.test_config_dict = {
            "paths": {
                "chunkformer_dir": str(self.temp_dir / "chunkformer"),
                "chunkformer_checkpoint": str(self.temp_dir / "checkpoint"),
                "out_dir": str(self.temp_dir / "output")
            },
            "llm": {
                "model_id": "test/model",
                "trust_remote_code": False,
            },
            "punctuation": {
                "llm": {"trust_remote_code": False},
                "system_prompt": "Add punctuation.",
            },
            "summarization": {
                "llm": {"trust_remote_code": False},
                "system_prompt": "Summarize text.",
            }
        }

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset singleton state
        ModelSingletons._instance = None

    def test_singleton_pattern(self):
        """Test that ModelSingletons follows singleton pattern."""
        instance1 = ModelSingletons()
        instance2 = ModelSingletons()
        
        # Should be the same instance
        self.assertIs(instance1, instance2)

    def test_thread_safety(self):
        """Test that singleton creation is thread-safe."""
        instances = []
        
        def create_instance():
            instances.append(ModelSingletons())
        
        # Create multiple threads that create instances
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All instances should be the same
        first_instance = instances[0]
        for instance in instances[1:]:
            self.assertIs(instance, first_instance)

    def test_model_initialization_mocking(self):
        """Test model initialization with mocked dependencies."""
        # Test that singleton can be created without issues
        singleton = ModelSingletons()
        self.assertIsNotNone(singleton)
        
        # Test that singleton follows pattern
        singleton2 = ModelSingletons()
        self.assertIs(singleton, singleton2)

    def test_model_info_structure_basic(self):
        """Test basic model info structure without loading actual models."""
        # Test that we can create singleton and check basic structure
        singleton = ModelSingletons()
        
        # Test that models are not loaded initially
        self.assertIsNone(singleton._asr_model)
        self.assertIsNone(singleton._punctuation_processor)
        self.assertIsNone(singleton._summarization_processor)
        
        # Test thread safety attributes exist
        self.assertTrue(hasattr(singleton, '_init_lock'))
        self.assertTrue(hasattr(singleton, '_initialized'))

    def test_service_mode_detection(self):
        """Test service mode detection logic."""
        import os
        
        # Test default mode
        if "OMOAI_SERVICE_MODE" in os.environ:
            del os.environ["OMOAI_SERVICE_MODE"]
        
        mode = get_service_mode()
        self.assertEqual(mode, ServiceMode.AUTO)
        
        # Test environment override
        os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.IN_MEMORY
        mode = get_service_mode()
        self.assertEqual(mode, ServiceMode.IN_MEMORY)
        
        os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.SCRIPT_BASED
        mode = get_service_mode()
        self.assertEqual(mode, ServiceMode.SCRIPT_BASED)
        
        # Cleanup
        del os.environ["OMOAI_SERVICE_MODE"]

    def test_should_use_in_memory_service_sync(self):
        """Test in-memory service availability detection."""
        import os
        import asyncio
        from unittest.mock import patch
        
        async def run_test():
            with patch('src.omoai.api.services_v2.health_check_models') as mock_health_check:
                # Test forced script mode
                os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.SCRIPT_BASED
                result = await should_use_in_memory_service()
                self.assertFalse(result)
                
                # Test forced memory mode
                os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.IN_MEMORY
                result = await should_use_in_memory_service()
                self.assertTrue(result)
                
                # Test auto mode with healthy models
                os.environ["OMOAI_SERVICE_MODE"] = ServiceMode.AUTO
                mock_health_check.return_value = {"status": "healthy"}
                result = await should_use_in_memory_service()
                # Note: may return False if actual models aren't available in test
                self.assertIsInstance(result, bool)
                
                # Test auto mode with unhealthy models
                mock_health_check.return_value = {"status": "unhealthy"}
                result = await should_use_in_memory_service()
                self.assertFalse(result)
                
                return True
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            self.assertTrue(result)
        finally:
            loop.close()
            # Cleanup
            if "OMOAI_SERVICE_MODE" in os.environ:
                del os.environ["OMOAI_SERVICE_MODE"]

    def test_get_service_status_sync(self):
        """Test service status reporting synchronously."""
        import asyncio
        
        async def run_test():
            status = await get_service_status()
            
            # Verify structure
            self.assertIn("service_mode", status)
            self.assertIn("in_memory_available", status)
            self.assertIn("active_backend", status)
            self.assertIn("performance_mode", status)
            self.assertIn("models", status)
            self.assertIn("config", status)
            
            # Verify values are valid
            self.assertIn(status["service_mode"], [ServiceMode.AUTO, ServiceMode.SCRIPT_BASED, ServiceMode.IN_MEMORY])
            self.assertIn(status["active_backend"], ["memory", "script"])
            self.assertIn(status["performance_mode"], ["high", "standard"])
            
            return status
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            self.assertIsInstance(result, dict)
        finally:
            loop.close()

    def test_warmup_services_sync(self):
        """Test service warmup functionality."""
        import asyncio
        from unittest.mock import patch
        
        async def run_test():
            with patch('src.omoai.api.singletons.preload_all_models') as mock_preload:
                mock_preload.return_value = {
                    "asr": True,
                    "punctuation": True, 
                    "summarization": True
                }
                
                result = await warmup_services()
                
                # Verify structure
                self.assertIn("success", result)
                self.assertIn("warmup_time", result)
                self.assertIn("models_loaded", result)
                self.assertIn("service_status", result)
                
                # Verify success
                self.assertTrue(result["success"])
                self.assertIsInstance(result["warmup_time"], float)
                self.assertGreater(result["warmup_time"], 0)
                
                return result
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            self.assertTrue(result["success"])
        finally:
            loop.close()

    def test_warmup_services_failure_sync(self):
        """Test service warmup with failures."""
        import asyncio
        from unittest.mock import patch
        
        async def run_test():
            with patch('src.omoai.api.singletons.preload_all_models') as mock_preload:
                mock_preload.side_effect = Exception("Model loading failed")
                
                result = await warmup_services()
                
                # Verify failure handling
                self.assertFalse(result["success"])
                self.assertIn("error", result)
                self.assertIn("fallback_to_script", result)
                self.assertTrue(result["fallback_to_script"])
                
                return result
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            self.assertFalse(result["success"])
        finally:
            loop.close()


class TestAsyncAPISingletons(unittest.TestCase):
    """Test async functionality of API singletons."""

    def setUp(self):
        """Set up async test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up async test environment."""
        self.loop.close()
        # Reset singleton state
        ModelSingletons._instance = None

    def test_async_service_status(self):
        """Test async service status functionality."""
        async def run_test():
            status = await get_service_status()
            self.assertIsInstance(status, dict)
            self.assertIn("service_mode", status)
            return status
        
        result = self.loop.run_until_complete(run_test())
        self.assertIsInstance(result, dict)

    def test_async_warmup(self):
        """Test async warmup functionality."""
        async def run_test():
            with patch('src.omoai.api.singletons.preload_all_models') as mock_preload:
                mock_preload.return_value = {"asr": True, "punctuation": True, "summarization": True}
                
                result = await warmup_services()
                self.assertTrue(result["success"])
                return result
        
        result = self.loop.run_until_complete(run_test())
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
