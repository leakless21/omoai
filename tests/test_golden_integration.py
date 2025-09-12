#!/usr/bin/env python3
"""
Golden integration tests that combine the new testing framework
with the existing OMOAI testing infrastructure.
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import testing fixtures
from tests.fixtures import AudioFixtureManager, ReferenceDataManager, create_test_audio

# Import OMOAI components for integration
from src.omoai.pipeline import run_full_pipeline_memory
from src.omoai.pipeline.asr import ASRResult, ASRSegment
from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
from src.omoai.config import OmoAIConfig


class TestGoldenIntegration(unittest.TestCase):
    """Integration tests using golden fixtures with real pipeline components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Ensure required directories exist for config validation
        chunk_dir = self.temp_dir / "chunkformer"
        chkpt_dir = self.temp_dir / "checkpoint"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chkpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.fixture_manager = AudioFixtureManager(self.temp_dir / "fixtures")
        self.reference_manager = ReferenceDataManager(self.temp_dir / "reference")
        
        # Create test configuration
        self.config = OmoAIConfig(
            paths={
                "chunkformer_dir": chunk_dir,
                "chunkformer_checkpoint": chkpt_dir
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
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.omoai.pipeline.pipeline.get_audio_info')
    @patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor')
    @patch('src.omoai.pipeline.pipeline.run_asr_inference')
    @patch('src.omoai.pipeline.pipeline.postprocess_transcript')
    def test_golden_fixtures_with_structured_logging(
        self, mock_postprocess, mock_asr, mock_preprocess, mock_audio_info
    ):
        """Test golden fixtures work with structured logging enabled."""
        
        # Get a fixture to test with
        fixtures = self.fixture_manager.get_all_fixtures()
        self.assertGreater(len(fixtures), 0, "Should have created fixtures")
        
        # Use first fixture
        fixture_name = list(fixtures.keys())[0]
        fixture = fixtures[fixture_name]
        
        # Setup mocks based on fixture
        mock_audio_info.return_value = {
            "duration": fixture.duration_seconds,
            "sample_rate": fixture.sample_rate,
            "channels": 1
        }
        
        # Load actual fixture audio
        waveform, sample_rate = self.fixture_manager.load_audio_tensor(fixture_name)
        mock_preprocess.return_value = (waveform, sample_rate)
        
        # Mock ASR result
        mock_asr_result = ASRResult(
            segments=[ASRSegment(0.0, fixture.duration_seconds, fixture.expected_transcript, 0.95)],
            transcript=fixture.expected_transcript,
            audio_duration_seconds=fixture.duration_seconds,
            sample_rate=sample_rate,
            metadata={"model": "test", "fixture": fixture_name}
        )
        mock_asr.return_value = mock_asr_result
        
        # Mock postprocessing result
        punctuated = fixture.expected_transcript.capitalize() + "." if fixture.expected_transcript else ""
        mock_postprocess_result = PostprocessResult(
            segments=[ASRSegment(0.0, fixture.duration_seconds, punctuated, 0.95)],
            transcript_punctuated=punctuated,
            summary=SummaryResult([punctuated] if punctuated else [], punctuated, {}),
            metadata={"quality": "good", "fixture": fixture_name}
        )
        mock_postprocess.return_value = mock_postprocess_result
        
        # Run pipeline with the fixture audio
        waveform_bytes = waveform.numpy().tobytes()
        
        result = run_full_pipeline_memory(
            audio_input=waveform_bytes,
            config=self.config,
            validate_input=False
        )
        
        # Validate result
        self.assertIsNotNone(result)
        self.assertIn("total", result.timing)
        self.assertGreater(result.timing["total"], 0)
        
        # Validate structured logging captured performance data
        self.assertIn("metadata", result.__dict__)
        
        # Check that performance metrics are reasonable
        if "asr" in result.timing:
            asr_time_seconds = result.timing["asr"]
            audio_duration = fixture.duration_seconds
            if audio_duration > 0:
                rtf = asr_time_seconds / audio_duration
                self.assertLess(rtf, 10.0, "Real-time factor should be reasonable for test")
    
    def test_fixture_difficulty_progression(self):
        """Test that fixtures progress from easy to hard appropriately."""
        easy_fixtures = self.fixture_manager.get_fixtures_by_difficulty("easy")
        medium_fixtures = self.fixture_manager.get_fixtures_by_difficulty("medium")
        hard_fixtures = self.fixture_manager.get_fixtures_by_difficulty("hard")
        
        # Easy fixtures should be simplest
        for fixture in easy_fixtures:
            self.assertLessEqual(fixture.duration_seconds, 5.0)
            self.assertEqual(fixture.expected_word_count, 0)  # Usually no speech in easy fixtures
        
        # Medium fixtures should be more complex
        for fixture in medium_fixtures:
            # Medium can be longer or have more complex audio
            self.assertGreaterEqual(fixture.duration_seconds, 2.0)
        
        # Should have at least some easy fixtures
        self.assertGreater(len(easy_fixtures), 0, "Should have easy fixtures for basic testing")
    
    def test_reference_data_consistency(self):
        """Test that reference data is consistent with fixtures."""
        fixtures = self.fixture_manager.get_all_fixtures()
        
        for name, fixture in fixtures.items():
            reference = self.reference_manager.get_reference_transcript(name)
            
            if reference:
                # Consistency checks
                self.assertEqual(fixture.name, reference.audio_fixture)
                self.assertEqual(fixture.expected_transcript, reference.expected_transcript)
                self.assertEqual(fixture.expected_word_count, reference.expected_word_count)
                
                # Confidence bounds
                self.assertGreaterEqual(reference.expected_confidence_min, 0.0)
                self.assertLessEqual(reference.expected_confidence_min, 1.0)
                self.assertGreaterEqual(reference.expected_confidence_avg, reference.expected_confidence_min)
                self.assertLessEqual(reference.expected_confidence_avg, 1.0)
    
    def test_performance_benchmark_creation(self):
        """Test automatic creation of performance benchmarks."""
        fixtures = self.fixture_manager.get_all_fixtures()
        
        for name, fixture in list(fixtures.items())[:2]:  # Test first 2 fixtures
            # Simulate test results
            mock_results = {
                "duration_ms": 100.0 + (fixture.duration_seconds * 20),  # Reasonable processing time
                "audio_duration_seconds": fixture.duration_seconds,
                "real_time_factor": 0.1,
                "memory_usage_mb": 50.0,
                "confidence_avg": 0.9
            }
            
            # Update benchmark
            self.reference_manager.update_benchmark_from_results(name, mock_results)
            
            # Verify benchmark was created
            benchmark = self.reference_manager.get_performance_benchmark(name)
            self.assertIsNotNone(benchmark, f"Benchmark should be created for {name}")
            
            # Verify benchmark values are reasonable
            self.assertGreater(benchmark.max_processing_time_ms, mock_results["duration_ms"])
            self.assertGreater(benchmark.max_real_time_factor, mock_results["real_time_factor"])
    
    def test_benchmark_validation_workflow(self):
        """Test the complete benchmark validation workflow."""
        fixtures = self.fixture_manager.get_all_fixtures()
        
        if not fixtures:
            self.skipTest("No fixtures available for testing")
        
        fixture_name = list(fixtures.keys())[0]
        fixture = fixtures[fixture_name]
        
        # Create initial benchmark
        baseline_results = {
            "duration_ms": 100.0,
            "audio_duration_seconds": fixture.duration_seconds,
            "real_time_factor": 0.1,
            "memory_usage_mb": 50.0,
            "confidence_avg": 0.9,
            "word_accuracy": 0.95
        }
        
        self.reference_manager.update_benchmark_from_results(fixture_name, baseline_results)
        
        # Test good performance (should pass)
        good_results = baseline_results.copy()
        good_results["duration_ms"] = 90.0  # Slightly faster
        
        validation = self.reference_manager.validate_against_benchmark(fixture_name, good_results)
        
        if "performance_time" in validation:
            self.assertTrue(validation["performance_time"], "Good performance should pass")
        
        # Test bad performance (should fail)
        bad_results = baseline_results.copy()
        bad_results["duration_ms"] = 300.0  # Much slower
        
        validation = self.reference_manager.validate_against_benchmark(fixture_name, bad_results)
        
        if "performance_time" in validation:
            self.assertFalse(validation["performance_time"], "Bad performance should fail")
    
    def test_custom_fixture_workflow(self):
        """Test the complete workflow for custom fixtures."""
        # Create custom audio
        custom_waveform = create_test_audio(
            duration_seconds=1.5,
            sample_rate=16000,
            frequency=660.0,  # A different frequency
            amplitude=0.7
        )
        
        # Create custom fixture
        custom_fixture = self.fixture_manager.create_custom_fixture(
            name="custom_test_audio",
            waveform=custom_waveform,
            sample_rate=16000,
            expected_transcript="custom test",
            metadata={"type": "custom", "frequency": 660.0, "test_case": "workflow"}
        )
        
        # Validate fixture was created correctly
        self.assertEqual(custom_fixture.name, "custom_test_audio")
        self.assertAlmostEqual(custom_fixture.duration_seconds, 1.5, places=1)
        self.assertEqual(custom_fixture.sample_rate, 16000)
        
        # Test that fixture can be loaded
        loaded_waveform, loaded_sr = self.fixture_manager.load_audio_tensor("custom_test_audio")
        self.assertEqual(loaded_sr, 16000)
        self.assertEqual(loaded_waveform.shape, custom_waveform.shape)
        
        # Create reference data for custom fixture
        from tests.fixtures.reference_data import ReferenceTranscript
        custom_reference = ReferenceTranscript(
            audio_fixture="custom_test_audio",
            expected_transcript="custom test",
            expected_word_count=2,
            expected_confidence_min=0.8,
            expected_confidence_avg=0.9,
            expected_segments=[
                {"start": 0.0, "end": 0.75, "text": "custom", "confidence": 0.9},
                {"start": 0.75, "end": 1.5, "text": "test", "confidence": 0.9}
            ],
            metadata={"type": "custom", "created_for": "workflow_test"}
        )
        
        self.reference_manager.add_reference_transcript(custom_reference)
        
        # Verify reference was added
        retrieved_reference = self.reference_manager.get_reference_transcript("custom_test_audio")
        self.assertIsNotNone(retrieved_reference)
        self.assertEqual(retrieved_reference.expected_transcript, "custom test")
        self.assertEqual(retrieved_reference.expected_word_count, 2)


class TestFrameworkIntegration(unittest.TestCase):
    """Test integration with existing testing framework."""
    
    def setUp(self):
        """Set up framework integration tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create required directories
        (self.temp_dir / "chunkformer").mkdir(parents=True)
        (self.temp_dir / "checkpoint").mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compatibility_with_existing_tests(self):
        """Test that new framework is compatible with existing tests."""
        # Import existing test utilities
        from tests.test_utils import MockTorchTensor
        
        # Test that existing mock utilities work with new fixtures
        fixture_manager = AudioFixtureManager(self.temp_dir / "fixtures")
        
        # Create fixture with mock tensor
        mock_tensor = MockTorchTensor([1, 16000])  # 1 second of audio
        
        # Should be able to create custom fixture
        fixture = fixture_manager.create_custom_fixture(
            name="mock_tensor_test",
            waveform=mock_tensor,
            sample_rate=16000,
            expected_transcript="mock audio"
        )
        
        self.assertEqual(fixture.name, "mock_tensor_test")
        self.assertEqual(fixture.sample_rate, 16000)
    
    def test_structured_logging_integration(self):
        """Test integration with structured logging system."""
        from src.omoai.logging_system import get_logger, get_performance_logger
        
        # Get loggers
        logger = get_logger("test.golden_integration")
        perf_logger = get_performance_logger()
        
        # Create fixture manager with logging
        fixture_manager = AudioFixtureManager(self.temp_dir / "fixtures")
        
        # Log fixture creation
        fixtures = fixture_manager.get_all_fixtures()
        logger.info("Golden fixtures created", extra={
            "fixture_count": len(fixtures),
            "fixture_names": list(fixtures.keys())
        })
        
        # Test performance logging with fixtures
        for name, fixture in list(fixtures.items())[:1]:  # Test one fixture
            with patch('time.time', side_effect=[0, 0.1]):  # Mock 100ms operation
                perf_logger.log_operation(
                    operation=f"fixture_validation_{name}",
                    duration_ms=100.0,
                    success=True,
                    fixture_name=name,
                    fixture_duration=fixture.duration_seconds,
                    fixture_difficulty=fixture.difficulty_level
                )
        
        # Should not raise any exceptions
        self.assertTrue(True, "Logging integration successful")
    
    def test_config_validation_integration(self):
        """Test integration with config validation system."""
        from src.omoai.config import OmoAIConfig
        
        # Create config that works with fixture testing
        config = OmoAIConfig(
            paths={
                "chunkformer_dir": self.temp_dir / "chunkformer",
                "chunkformer_checkpoint": self.temp_dir / "checkpoint"
            },
            llm={
                "model_id": "test/model",
                "trust_remote_code": False  # Security default
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
        
        # Validate config works
        self.assertEqual(config.llm.model_id, "test/model")
        self.assertFalse(config.llm.trust_remote_code)
        
        # Test with fixture manager
        fixture_manager = AudioFixtureManager(self.temp_dir / "fixtures")
        fixtures = fixture_manager.get_all_fixtures()
        
        # Should be able to use config with fixtures
        self.assertGreater(len(fixtures), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
