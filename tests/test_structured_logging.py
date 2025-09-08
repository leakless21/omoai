#!/usr/bin/env python3
"""
Tests for OMOAI structured logging system.
"""
import json
import logging
import os
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from omoai.logging_system import (
    get_logger, setup_logging, log_performance, log_error,
    performance_context, timed, JSONFormatter, StructuredFormatter,
    LoggingConfig, get_logging_config, get_performance_logger
)


class TestLoggingConfiguration(unittest.TestCase):
    """Test logging configuration and setup."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear environment variables
        for env_var in [
            "OMOAI_LOG_LEVEL", "OMOAI_LOG_FORMAT", "OMOAI_LOG_CONSOLE",
            "OMOAI_LOG_FILE_ENABLED", "OMOAI_LOG_FILE", "OMOAI_DEBUG"
        ]:
            if env_var in os.environ:
                del os.environ[env_var]
    
    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        self.assertEqual(config.level, "INFO")
        self.assertEqual(config.format_type, "structured")
        self.assertTrue(config.enable_console)
        self.assertFalse(config.enable_file)
        self.assertTrue(config.enable_performance_logging)
        self.assertTrue(config.enable_request_tracing)
        self.assertTrue(config.enable_error_tracking)
    
    def test_environment_based_config(self):
        """Test configuration from environment variables."""
        os.environ["OMOAI_LOG_LEVEL"] = "DEBUG"
        os.environ["OMOAI_LOG_FORMAT"] = "json"
        os.environ["OMOAI_DEBUG"] = "true"
        
        config = LoggingConfig.from_environment()
        
        self.assertEqual(config.level, "DEBUG")
        self.assertEqual(config.format_type, "json")
        self.assertTrue(config.debug_mode)
        self.assertEqual(config.get_log_level(), logging.DEBUG)
    
    def test_performance_threshold(self):
        """Test performance threshold logic."""
        config = LoggingConfig(performance_threshold_ms=100.0)
        
        self.assertTrue(config.should_log_performance(150.0))
        self.assertFalse(config.should_log_performance(50.0))
        self.assertTrue(config.should_log_performance(100.0))  # Equal to threshold


class TestJSONFormatter(unittest.TestCase):
    """Test JSON logging formatter."""
    
    def setUp(self):
        """Set up test formatter."""
        self.formatter = JSONFormatter()
    
    def test_basic_formatting(self):
        """Test basic JSON log formatting."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        data = json.loads(formatted)
        
        self.assertIn("timestamp", data)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test.logger")
        self.assertEqual(data["message"], "Test message")
        self.assertEqual(data["line"], 42)
    
    def test_exception_formatting(self):
        """Test exception formatting in JSON logs."""
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info()
            )
        
        formatted = self.formatter.format(record)
        data = json.loads(formatted)
        
        self.assertIn("exception", data)
        self.assertEqual(data["exception"]["type"], "ValueError")
        self.assertEqual(data["exception"]["message"], "Test error")
        self.assertIn("traceback", data["exception"])
    
    def test_extra_fields(self):
        """Test extra fields in JSON logs."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.request_id = "req-123"
        record.operation = "test_operation"
        record.duration_ms = 150.5
        
        formatted = self.formatter.format(record)
        data = json.loads(formatted)
        
        self.assertIn("extra", data)
        self.assertEqual(data["extra"]["request_id"], "req-123")
        self.assertEqual(data["extra"]["operation"], "test_operation")
        self.assertEqual(data["extra"]["duration_ms"], 150.5)


class TestStructuredFormatter(unittest.TestCase):
    """Test structured logging formatter."""
    
    def setUp(self):
        """Set up test formatter."""
        self.formatter = StructuredFormatter(color=False)  # Disable color for testing
    
    def test_basic_formatting(self):
        """Test basic structured log formatting."""
        record = logging.LogRecord(
            name="src.omoai.pipeline.asr",
            level=logging.INFO,
            pathname="/src/omoai/pipeline/asr.py",
            lineno=42,
            msg="ASR inference completed",
            args=(),
            exc_info=None
        )
        record.funcName = "run_inference"
        record.module = "asr"
        
        formatted = self.formatter.format(record)
        
        self.assertIn("INFO", formatted)
        self.assertIn("pipeline.asr", formatted)  # Shortened logger name
        self.assertIn("asr:run_inference:42", formatted)  # Location info
        self.assertIn("ASR inference completed", formatted)
    
    def test_extra_fields_formatting(self):
        """Test extra fields in structured logs."""
        record = logging.LogRecord(
            name="omoai.test",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_func"
        record.module = "test"
        record.duration_ms = 150.5
        record.operation = "test_op"
        
        formatted = self.formatter.format(record)
        
        self.assertIn("duration_ms=150.5", formatted)
        self.assertIn("operation=test_op", formatted)


class TestPerformanceLogging(unittest.TestCase):
    """Test performance logging functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_capture = StringIO()
        
        # Create test logger
        self.logger = logging.getLogger("test.performance")
        self.logger.setLevel(logging.DEBUG)
        
        # Add string handler
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_performance_basic(self):
        """Test basic performance logging."""
        log_performance(
            operation="test_operation",
            duration_ms=150.0,
            logger=self.logger,
            extra_field="test_value"
        )
        
        log_output = self.log_capture.getvalue()
        data = json.loads(log_output.strip())
        
        self.assertEqual(data["extra"]["operation"], "test_operation")
        self.assertEqual(data["extra"]["duration_ms"], 150.0)
        self.assertEqual(data["extra"]["extra_field"], "test_value")
        self.assertIn("test_operation", data["message"])
        self.assertIn("150.00ms", data["message"])
    
    def test_performance_context_manager(self):
        """Test performance context manager."""
        with patch('omoai.logging_system.logger.log_performance') as mock_log:
            with performance_context("test_context", logger=self.logger):
                time.sleep(0.01)  # Small delay
        
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        self.assertEqual(call_args[1]["operation"], "test_context")
        self.assertGreater(call_args[1]["duration_ms"], 5)  # At least 5ms
    
    def test_timed_decorator(self):
        """Test timed decorator."""
        @timed(operation="test_decorated_function", logger=self.logger)
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        with patch('omoai.logging_system.logger.log_performance') as mock_log:
            result = test_function(1, 2)
        
        self.assertEqual(result, 3)
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        self.assertEqual(call_args[1]["operation"], "test_decorated_function")
    
    def test_performance_logger_integration(self):
        """Test performance logger integration."""
        perf_logger = get_performance_logger()
        
        perf_logger.log_operation(
            operation="test_integration",
            duration_ms=200.0,
            success=True,
            custom_metric=42
        )
        
        # Check that operation was recorded
        stats = perf_logger.get_stats("test_integration")
        self.assertEqual(stats["operation"], "test_integration")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["min_ms"], 200.0)


class TestErrorLogging(unittest.TestCase):
    """Test error logging functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.log_capture = StringIO()
        
        # Create test logger
        self.logger = logging.getLogger("test.error")
        self.logger.setLevel(logging.DEBUG)
        
        # Add string handler
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def test_log_error_basic(self):
        """Test basic error logging."""
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            log_error(
                message="Test error occurred",
                error=e,
                error_type="TEST_ERROR",
                error_code="ERR001",
                remediation="Fix the test",
                logger=self.logger,
                context_field="test_context"
            )
        
        log_output = self.log_capture.getvalue()
        data = json.loads(log_output.strip())
        
        self.assertEqual(data["level"], "ERROR")
        self.assertEqual(data["message"], "Test error occurred")
        self.assertEqual(data["extra"]["error_type"], "TEST_ERROR")
        self.assertEqual(data["extra"]["error_code"], "ERR001")
        self.assertEqual(data["extra"]["remediation"], "Fix the test")
        self.assertEqual(data["extra"]["context_field"], "test_context")
        self.assertIn("exception", data)
    
    def test_log_error_without_exception(self):
        """Test error logging without exception object."""
        log_error(
            message="Error without exception",
            error_type="GENERAL_ERROR",
            logger=self.logger
        )
        
        log_output = self.log_capture.getvalue()
        data = json.loads(log_output.strip())
        
        self.assertEqual(data["level"], "ERROR")
        self.assertEqual(data["message"], "Error without exception")
        self.assertEqual(data["extra"]["error_type"], "GENERAL_ERROR")
        self.assertNotIn("exception", data)


class TestMetricsCollection(unittest.TestCase):
    """Test metrics collection functionality."""
    
    def test_metrics_collector_basic(self):
        """Test basic metrics collection."""
        from omoai.logging_system.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_performance("operation1", 100.0, success=True)
        collector.record_performance("operation1", 150.0, success=True)
        collector.record_performance("operation2", 200.0, success=False, error_type="TEST_ERROR")
        
        # Get stats
        stats1 = collector.get_operation_stats("operation1")
        self.assertEqual(stats1["count"], 2)
        self.assertEqual(stats1["min_ms"], 100.0)
        self.assertEqual(stats1["max_ms"], 150.0)
        self.assertEqual(stats1["mean_ms"], 125.0)
        
        stats2 = collector.get_operation_stats("operation2")
        self.assertEqual(stats2["count"], 1)
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        from omoai.logging_system.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record system metrics
        system_metrics = collector.record_system_metrics()
        
        self.assertIsNotNone(system_metrics.cpu_percent)
        self.assertIsNotNone(system_metrics.memory_percent)
        self.assertIsNotNone(system_metrics.memory_used_gb)
        self.assertGreaterEqual(system_metrics.cpu_percent, 0)
        self.assertGreaterEqual(system_metrics.memory_percent, 0)
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        from omoai.logging_system.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record some recent metrics
        collector.record_performance("test_op", 100.0, success=True)
        collector.record_performance("test_op", 200.0, success=False, error_type="TEST_ERROR")
        
        # Generate report
        report = collector.get_performance_report(hours=1)
        
        self.assertIn("summary", report)
        self.assertIn("operations", report)
        self.assertIn("system_health", report)
        
        summary = report["summary"]
        self.assertEqual(summary["total_operations"], 2)
        self.assertEqual(summary["successful_operations"], 1)
        self.assertEqual(summary["failed_operations"], 1)
        self.assertEqual(summary["success_rate_percent"], 50.0)


class TestLoggingIntegration(unittest.TestCase):
    """Test complete logging system integration."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_file = self.temp_dir / "test.log"
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_logging_setup(self):
        """Test complete logging system setup."""
        # Configure logging with file output
        config = LoggingConfig(
            level="DEBUG",
            format_type="json",
            enable_file=True,
            log_file=self.log_file,
            enable_performance_logging=True,
            enable_error_tracking=True,
        )
        
        setup_logging(config)
        
        # Test basic logging
        logger = get_logger("test.integration")
        logger.info("Test integration message", extra={"test_field": "test_value"})
        
        # Test performance logging
        with performance_context("integration_test", logger=logger):
            time.sleep(0.01)
        
        # Test error logging
        try:
            raise RuntimeError("Integration test error")
        except RuntimeError as e:
            log_error(
                message="Integration test error occurred",
                error=e,
                error_type="INTEGRATION_ERROR",
                logger=logger
            )
        
        # Verify log file was created and contains data
        self.assertTrue(self.log_file.exists())
        
        with open(self.log_file, 'r') as f:
            log_lines = f.readlines()
        
        self.assertGreater(len(log_lines), 0)
        
        # Verify JSON format
        for line in log_lines:
            if line.strip():
                data = json.loads(line.strip())
                self.assertIn("timestamp", data)
                self.assertIn("level", data)
                self.assertIn("message", data)
    
    def test_request_id_tracing(self):
        """Test request ID tracing functionality."""
        from omoai.logging_system.logger import with_request_context, generate_request_id
        
        # Generate request ID
        request_id = generate_request_id()
        self.assertIsInstance(request_id, str)
        self.assertGreater(len(request_id), 10)
        
        # Test context decorator
        @with_request_context(request_id=request_id, user_id="test_user")
        def test_function():
            logger = get_logger("test.tracing")
            logger.info("Test message with context")
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")


if __name__ == "__main__":
    unittest.main()
