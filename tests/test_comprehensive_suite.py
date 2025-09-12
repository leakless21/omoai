#!/usr/bin/env python3
"""
Comprehensive test suite for OMOAI with golden fixtures and performance testing.

This module provides the complete testing framework including:
- Golden audio fixture testing
- Performance regression testing  
- Load testing and stress testing
- End-to-end pipeline validation
"""
import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import our fixtures and testing framework
from tests.fixtures import (
    AudioFixtureManager, ReferenceDataManager, PerformanceTestSuite,
    LoadTestRunner, create_test_audio
)

# Import OMOAI components
from src.omoai.pipeline import run_full_pipeline_memory, PipelineResult
from src.omoai.config import get_config, OmoAIConfig


class TestGoldenFixtures(unittest.TestCase):
    """Test golden audio fixtures for accuracy and consistency."""
    
    def setUp(self):
        """Set up test environment with fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fixture_manager = AudioFixtureManager(self.temp_dir / "fixtures")
        self.reference_manager = ReferenceDataManager(self.temp_dir / "reference")
        
        # Create test config
        self.config = self._create_test_config()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self) -> OmoAIConfig:
        """Create test configuration."""
        # Ensure required paths exist so OmoAIConfig validation passes
        (self.temp_dir / "chunkformer").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        # Prefer using local real ChunkFormer model if available, otherwise fall back to temp dirs
        model_path = Path("models/chunkformer/chunkformer-large-vie")
        if model_path.exists():
            checkpoint_path = model_path
            chunk_dir = model_path.parent
        else:
            checkpoint_path = self.temp_dir / "checkpoint"
            chunk_dir = self.temp_dir / "chunkformer"
        return OmoAIConfig(
            paths={
                "chunkformer_dir": chunk_dir,
                "chunkformer_checkpoint": checkpoint_path
            },
            llm={
                "model_id": "test/model",
                "trust_remote_code": False
            },
            punctuation={
                "llm": {},
                "system_prompt": "Add punctuation."
            },
            summarization={
                "llm": {},
                "system_prompt": "Summarize text."
            }
        )
    
    def test_fixture_creation_and_loading(self):
        """Test that fixtures are created and can be loaded."""
        # Get all available fixtures
        fixtures = self.fixture_manager.get_all_fixtures()
        self.assertGreater(len(fixtures), 0, "Should have created some fixtures")
        
        # Test each fixture
        for name, fixture in fixtures.items():
            with self.subTest(fixture=name):
                # Validate fixture exists and is loadable
                self.assertTrue(
                    self.fixture_manager.validate_fixture(name),
                    f"Fixture {name} should be valid"
                )
                
                # Load audio tensor
                waveform, sample_rate = self.fixture_manager.load_audio_tensor(name)
                self.assertIsNotNone(waveform, f"Should load waveform for {name}")
                self.assertGreater(sample_rate, 0, f"Should have valid sample rate for {name}")
                
                # Check duration matches expectation
                actual_duration = waveform.shape[-1] / sample_rate
                expected_duration = fixture.duration_seconds
                self.assertAlmostEqual(
                    actual_duration, expected_duration, places=1,
                    msg=f"Duration mismatch for {name}"
                )
    
    def test_fixture_difficulty_levels(self):
        """Test that fixtures are properly categorized by difficulty."""
        easy_fixtures = self.fixture_manager.get_fixtures_by_difficulty("easy")
        medium_fixtures = self.fixture_manager.get_fixtures_by_difficulty("medium")
        
        self.assertGreater(len(easy_fixtures), 0, "Should have easy fixtures")
        
        # Easy fixtures should be shorter or simpler
        for fixture in easy_fixtures:
            self.assertLessEqual(
                fixture.duration_seconds, 5.0,
                f"Easy fixture {fixture.name} should be short"
            )
    
    @patch('src.omoai.pipeline.pipeline.get_audio_info')
    @patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor')
    @patch('src.omoai.pipeline.pipeline.run_asr_inference')
    @patch('src.omoai.pipeline.pipeline.postprocess_transcript')
    def test_golden_fixture_pipeline_processing(
        self, mock_postprocess, mock_asr, mock_preprocess, mock_audio_info
    ):
        """Test pipeline processing with golden fixtures."""
        from src.omoai.pipeline.asr import ASRResult, ASRSegment
        from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
        
        # Setup mocks
        mock_audio_info.return_value = {"duration": 3.0, "sample_rate": 16000, "channels": 1}
        mock_preprocess.return_value = (create_test_audio(3.0, 16000), 16000)
        
        mock_asr_result = ASRResult(
            segments=[ASRSegment(0.0, 3.0, "test audio", 0.95)],
            transcript="test audio",
            audio_duration_seconds=3.0,
            sample_rate=16000,
            metadata={"model": "test"}
        )
        mock_asr.return_value = mock_asr_result
        
        mock_postprocess_result = PostprocessResult(
            segments=[ASRSegment(0.0, 3.0, "Test audio.", 0.95)],
            transcript_punctuated="Test audio.",
            summary=SummaryResult(["Test audio"], "Audio test", {}),
            metadata={"quality": "good"}
        )
        mock_postprocess.return_value = mock_postprocess_result
        
        # Test processing with a golden fixture
        fixtures = self.fixture_manager.get_all_fixtures()
        test_fixture_name = list(fixtures.keys())[0]  # Use first available fixture
        
        waveform, sample_rate = self.fixture_manager.load_audio_tensor(test_fixture_name)
        waveform_bytes = waveform.numpy().tobytes()
        
        # Process through pipeline
        result = run_full_pipeline_memory(
            audio_input=waveform_bytes,
            config=self.config,
            validate_input=False
        )
        
        # Validate result structure
        self.assertIsInstance(result, PipelineResult)
        self.assertIsInstance(result.transcript_raw, str)
        self.assertIsInstance(result.transcript_punctuated, str)
        self.assertIsNotNone(result.summary)
        
        # Validate timing data
        self.assertIn("total", result.timing)
        self.assertGreater(result.timing["total"], 0)
    
    def test_custom_fixture_creation(self):
        """Test creating custom fixtures."""
        # Create a custom audio tensor
        custom_audio = create_test_audio(duration_seconds=2.0, frequency=880.0)
        
        # Create custom fixture
        fixture = self.fixture_manager.create_custom_fixture(
            name="custom_test",
            waveform=custom_audio,
            sample_rate=16000,
            expected_transcript="custom test audio",
            metadata={"type": "custom", "frequency": 880.0}
        )
        
        # Validate fixture was created
        self.assertEqual(fixture.name, "custom_test")
        self.assertEqual(fixture.expected_transcript, "custom test audio")
        self.assertEqual(fixture.sample_rate, 16000)
        self.assertAlmostEqual(fixture.duration_seconds, 2.0, places=1)
        
        # Validate fixture can be loaded
        loaded_waveform, loaded_sr = self.fixture_manager.load_audio_tensor("custom_test")
        self.assertEqual(loaded_sr, 16000)
        self.assertEqual(loaded_waveform.shape, custom_audio.shape)


class TestPerformanceRegression(unittest.TestCase):
    """Test performance regression detection."""
    
    def setUp(self):
        """Set up performance testing environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fixture_manager = AudioFixtureManager(self.temp_dir / "fixtures")
        self.perf_suite = PerformanceTestSuite()
        
        # Create test config
        self.config = self._create_test_config()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self) -> OmoAIConfig:
        """Create test configuration."""
        # Ensure required paths exist so OmoAIConfig validation passes
        (self.temp_dir / "chunkformer").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        # Prefer using local real ChunkFormer model if available, otherwise fall back to temp dirs
        model_path = Path("models/chunkformer/chunkformer-large-vie")
        if model_path.exists():
            checkpoint_path = model_path
            chunk_dir = model_path.parent
        else:
            checkpoint_path = self.temp_dir / "checkpoint"
            chunk_dir = self.temp_dir / "chunkformer"
        return OmoAIConfig(
            paths={
                "chunkformer_dir": chunk_dir,
                "chunkformer_checkpoint": checkpoint_path
            },
            llm={
                "model_id": "test/model",
                "trust_remote_code": False
            },
            punctuation={"llm": {}, "system_prompt": "Add punctuation."},
            summarization={"llm": {}, "system_prompt": "Summarize text."}
        )
    
    def _mock_fast_pipeline(self, audio_input, **kwargs):
        """Mock fast pipeline operation."""
        time.sleep(0.01)  # 10ms processing time
        from src.omoai.pipeline.asr import ASRResult, ASRSegment
        from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
        
        # Return mock result structure
        class MockPipelineResult:
            def __init__(self):
                self.transcript_raw = "mock transcript"
                self.transcript_punctuated = "Mock transcript."
                self.summary = SummaryResult(["Mock"], "Summary", {})
                self.timing = {"total": 0.01, "asr": 0.005, "postprocessing": 0.003}
                self.metadata = {"test": True}
        
        return MockPipelineResult()
    
    def _mock_slow_pipeline(self, audio_input, **kwargs):
        """Mock slow pipeline operation (simulates regression)."""
        time.sleep(0.05)  # 50ms processing time (5x slower)
        return self._mock_fast_pipeline(audio_input, **kwargs)
    
    def test_baseline_establishment(self):
        """Test establishing performance baselines."""
        with patch('src.omoai.pipeline.run_full_pipeline_memory', self._mock_fast_pipeline):
            test_functions = {
                "simple_pipeline": lambda: run_full_pipeline_memory(
                    b"fake_audio", config=self.config, validate_input=False
                )
            }
            
            baselines = self.perf_suite.run_baseline_tests(test_functions)
            
            self.assertIn("simple_pipeline", baselines)
            baseline = baselines["simple_pipeline"]
            
            # Should be around 10ms
            self.assertLess(baseline.duration_ms, 50, "Baseline should be fast")
            self.assertGreater(baseline.duration_ms, 5, "Baseline should take some time")
    
    def test_regression_detection(self):
        """Test performance regression detection."""
        # First establish baseline with fast mock
        with patch('src.omoai.pipeline.run_full_pipeline_memory', self._mock_fast_pipeline):
            test_functions = {
                "pipeline_test": lambda: run_full_pipeline_memory(
                    b"fake_audio", config=self.config, validate_input=False
                )
            }
            
            self.perf_suite.run_baseline_tests(test_functions)
        
        # Then test with slow mock (simulates regression)
        with patch('src.omoai.pipeline.run_full_pipeline_memory', self._mock_slow_pipeline):
            regression_results = self.perf_suite.run_regression_tests(test_functions)
            
            self.assertIn("pipeline_test", regression_results)
            result = regression_results["pipeline_test"]
            
            # Should detect regression
            self.assertTrue(result["duration_regression"], "Should detect duration regression")
            self.assertGreater(result["duration_ratio"], 1.5, "Should show significant slowdown")
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        with patch('src.omoai.pipeline.run_full_pipeline_memory', self._mock_fast_pipeline):
            test_functions = {
                "test_pipeline": lambda: run_full_pipeline_memory(
                    b"fake_audio", config=self.config, validate_input=False
                )
            }
            
            baseline_results = self.perf_suite.run_baseline_tests(test_functions)
            regression_results = self.perf_suite.run_regression_tests(test_functions)
            
            report = self.perf_suite.generate_performance_report({
                "baseline": baseline_results,
                "regression": regression_results
            })
            
            self.assertIn("Performance Test Report", report)
            self.assertIn("Baseline Performance", report)
            self.assertIn("Regression Test Results", report)
            self.assertIn("PASSED", report)  # Should pass since we use same mock


class TestLoadTesting(unittest.TestCase):
    """Test load testing capabilities."""
    
    def setUp(self):
        """Set up load testing environment.""" 
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = self._create_test_config()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self) -> OmoAIConfig:
        """Create test configuration."""
        # Ensure required paths exist so OmoAIConfig validation passes
        (self.temp_dir / "chunkformer").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        # Prefer using local real ChunkFormer model if available, otherwise fall back to temp dirs
        model_path = Path("models/chunkformer/chunkformer-large-vie")
        if model_path.exists():
            checkpoint_path = model_path
            chunk_dir = model_path.parent
        else:
            checkpoint_path = self.temp_dir / "checkpoint"
            chunk_dir = self.temp_dir / "chunkformer"
        return OmoAIConfig(
            paths={
                "chunkformer_dir": chunk_dir,
                "chunkformer_checkpoint": checkpoint_path
            },
            llm={
                "model_id": "test/model",
                "trust_remote_code": False
            },
            punctuation={"llm": {}, "system_prompt": "Add punctuation."},
            summarization={"llm": {}, "system_prompt": "Summarize text."}
        )
    
    def _mock_pipeline_operation(self, audio_input):
        """Mock pipeline operation for load testing."""
        # Simulate variable processing time
        import random
        processing_time = random.uniform(0.01, 0.03)  # 10-30ms
        time.sleep(processing_time)
        
        class MockResult:
            def __init__(self):
                self.transcript_raw = "load test transcript"
                self.timing = {"total": processing_time}
        
        return MockResult()
    
    def test_single_operation_performance(self):
        """Test single operation performance measurement."""
        with patch('src.omoai.pipeline.run_full_pipeline_memory', self._mock_pipeline_operation):
            runner = LoadTestRunner(
                operation_func=lambda x: run_full_pipeline_memory(x, config=self.config),
                operation_name="single_op_test"
            )
            
            metrics = runner.run_single_operation(b"test_audio")
            
            self.assertGreater(metrics.duration_ms, 0)
            self.assertIsInstance(metrics.memory_usage_mb, (int, float))
            self.assertIsInstance(metrics.cpu_percent, (int, float))
            self.assertTrue(metrics.metadata.get("success", False))
    
    def test_concurrent_load_testing(self):
        """Test concurrent load testing."""
        with patch('src.omoai.pipeline.run_full_pipeline_memory', self._mock_pipeline_operation):
            runner = LoadTestRunner(
                operation_func=lambda x: run_full_pipeline_memory(x, config=self.config),
                operation_name="load_test"
            )
            
            # Small load test (3 users, 2 operations each)
            test_data = [b"audio1", b"audio2", b"audio3"]
            
            result = runner.run_load_test(
                concurrent_users=3,
                operations_per_user=2,
                test_data=test_data,
                timeout_seconds=10
            )
            
            # Validate results
            self.assertEqual(result.total_operations, 6)  # 3 users * 2 ops
            self.assertGreaterEqual(result.successful_operations, 0)
            self.assertGreater(result.throughput_ops_per_sec, 0)
            self.assertGreaterEqual(result.error_rate_percent, 0)
            self.assertLessEqual(result.error_rate_percent, 100)
            
            # Performance metrics
            self.assertGreater(result.average_operation_time_ms, 0)
            self.assertGreaterEqual(result.p50_ms, result.min_operation_time_ms)
            self.assertGreaterEqual(result.p90_ms, result.p50_ms)
            self.assertGreaterEqual(result.p95_ms, result.p90_ms)
    
    def test_stress_test_configuration(self):
        """Test stress test configuration and execution."""
        from tests.fixtures.performance_fixtures import StressTestConfig
        
        config = StressTestConfig(
            max_concurrent_users=5,
            ramp_up_duration_seconds=5,
            steady_state_duration_seconds=10,
            ramp_down_duration_seconds=5,
            operation_timeout_seconds=5
        )
        
        # Validate config
        self.assertEqual(config.max_concurrent_users, 5)
        self.assertEqual(config.steady_state_duration_seconds, 10)
        
        # Test stress test execution (limited)
        with patch('src.omoai.pipeline.run_full_pipeline_memory', self._mock_pipeline_operation):
            runner = LoadTestRunner(
                operation_func=lambda x: run_full_pipeline_memory(x, config=self.config),
                operation_name="stress_test"
            )
            
            test_data = [b"audio1", b"audio2"]
            
            # Run limited stress test
            results = runner.run_stress_test(config, test_data)
            
            # Should have multiple test phases
            self.assertIn("steady_state", results)
            self.assertIsInstance(results["steady_state"].throughput_ops_per_sec, (int, float))


class TestEndToEndValidation(unittest.TestCase):
    """End-to-end validation of the complete testing framework."""
    
    def setUp(self):
        """Set up comprehensive testing environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fixture_manager = AudioFixtureManager(self.temp_dir / "fixtures")
        self.reference_manager = ReferenceDataManager(self.temp_dir / "reference")
        self.perf_suite = PerformanceTestSuite()
        
        self.config = self._create_test_config()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self) -> OmoAIConfig:
        """Create test configuration."""
        # Ensure required directories exist so OmoAIConfig path validation passes
        chunk_dir = self.temp_dir / "chunkformer"
        chkpt_dir = self.temp_dir / "checkpoint"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chkpt_dir.mkdir(parents=True, exist_ok=True)
        # Prefer using local real ChunkFormer model if available, otherwise fall back to temp dirs
        model_path = Path("models/chunkformer/chunkformer-large-vie")
        if model_path.exists():
            checkpoint_path = model_path
            chunkformer_dir = model_path.parent
        else:
            checkpoint_path = chkpt_dir
            chunkformer_dir = chunk_dir
        return OmoAIConfig(
            paths={
                "chunkformer_dir": chunkformer_dir,
                "chunkformer_checkpoint": checkpoint_path
            },
            llm={
                "model_id": "test/model",
                "trust_remote_code": False
            },
            punctuation={"llm": {}, "system_prompt": "Add punctuation."},
            summarization={"llm": {}, "system_prompt": "Summarize text."}
        )
    
    def test_complete_testing_workflow(self):
        """Test the complete testing workflow."""
        # 1. Validate fixture creation
        fixtures = self.fixture_manager.get_all_fixtures()
        self.assertGreater(len(fixtures), 0, "Should create fixtures")
        
        # 2. Validate reference data creation
        transcripts = self.reference_manager.transcripts
        self.assertGreater(len(transcripts), 0, "Should have reference transcripts")
        
        # 3. Test fixture loading and validation
        for name in list(fixtures.keys())[:2]:  # Test first 2 fixtures
            self.assertTrue(
                self.fixture_manager.validate_fixture(name),
                f"Fixture {name} should be valid"
            )
        
        # 4. Test reference data integrity
        for name, transcript in list(transcripts.items())[:2]:  # Test first 2
            self.assertIsInstance(transcript.expected_transcript, str)
            self.assertIsInstance(transcript.expected_word_count, int)
            self.assertGreaterEqual(transcript.expected_confidence_min, 0.0)
    
    def test_framework_integration(self):
        """Test integration between all framework components."""
        # Get a fixture
        fixtures = self.fixture_manager.get_all_fixtures()
        if not fixtures:
            self.skipTest("No fixtures available")
        
        fixture_name = list(fixtures.keys())[0]
        fixture = fixtures[fixture_name]
        
        # Get reference data
        reference = self.reference_manager.get_reference_transcript(fixture_name)
        
        if reference:
            # Validate data consistency
            self.assertEqual(fixture.name, reference.audio_fixture)
            
            # Test benchmark validation
            mock_results = {
                "duration_ms": 100.0,
                "audio_duration_seconds": fixture.duration_seconds,
                "real_time_factor": 100.0 / (fixture.duration_seconds * 1000),
                "confidence_avg": 0.9
            }
            
            # Update benchmarks
            self.reference_manager.update_benchmark_from_results(
                fixture_name, mock_results
            )
            
            # Validate against benchmarks
            validation = self.reference_manager.validate_against_benchmark(
                fixture_name, mock_results
            )
            
            # Should have validation results
            self.assertIsInstance(validation, dict)
    
    def test_testing_framework_robustness(self):
        """Test framework handles edge cases and errors gracefully."""
        # Test non-existent fixture
        self.assertIsNone(self.fixture_manager.get_fixture("nonexistent"))
        self.assertFalse(self.fixture_manager.validate_fixture("nonexistent"))
        
        # Test invalid audio data
        with self.assertRaises(Exception):
            self.fixture_manager.load_audio_tensor("nonexistent")
        
        # Test empty reference data
        empty_reference = self.reference_manager.get_reference_transcript("nonexistent")
        self.assertIsNone(empty_reference)
        
        # Test validation with missing benchmarks
        validation = self.reference_manager.validate_against_benchmark(
            "nonexistent", {"duration_ms": 100.0}
        )
        self.assertEqual(validation, {})


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
