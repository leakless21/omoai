#!/usr/bin/env python3
"""
Integration tests for the OMOAI pipeline module.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import numpy as np
import torch
import json

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.omoai.pipeline import (
    run_full_pipeline_memory,
    preprocess_audio_to_tensor,
    run_asr_inference,
    postprocess_transcript,
    ASRResult,
    ASRSegment,
    PipelineResult,
)
from src.omoai.config import OmoAIConfig


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create required directories
        (self.temp_dir / "chunkformer").mkdir()
        (self.temp_dir / "checkpoint").mkdir()
        
        # Create mock audio data (1 second of silence at 16kHz)
        self.mock_audio_data = np.zeros(16000, dtype=np.float32)
        self.mock_audio_tensor = torch.from_numpy(self.mock_audio_data).unsqueeze(0)
        
        # Create minimal valid config for testing
        self.valid_config_dict = {
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

    def create_test_config(self) -> OmoAIConfig:
        """Create a valid test configuration."""
        return OmoAIConfig(**self.valid_config_dict)

    def test_preprocessing_integration(self):
        """Test preprocessing with real audio processing."""
        # Mock pydub for consistent test behavior
        with patch('src.omoai.pipeline.preprocess.AudioSegment') as mock_audio:
            mock_segment = MagicMock()
            mock_segment.set_frame_rate.return_value = mock_segment
            mock_segment.set_sample_width.return_value = mock_segment
            mock_segment.set_channels.return_value = mock_segment
            mock_segment.get_array_of_samples.return_value = self.mock_audio_data * 32768
            mock_audio.from_file.return_value = mock_segment
            
            # Test with bytes input
            result_tensor, sample_rate = preprocess_audio_to_tensor(
                b"fake_audio_bytes",
                return_sample_rate=True
            )
            
            # Verify results
            self.assertIsInstance(result_tensor, torch.Tensor)
            self.assertEqual(result_tensor.shape, (1, 16000))
            self.assertEqual(sample_rate, 16000)

    @patch('src.omoai.pipeline.asr.ChunkFormerASR')
    def test_asr_integration(self, mock_asr_class):
        """Test ASR integration with configuration."""
        # Mock ChunkFormer ASR
        mock_asr = MagicMock()
        mock_result = ASRResult(
            segments=[ASRSegment(0.0, 1.0, "hello world")],
            transcript="hello world",
            audio_duration_seconds=1.0,
            sample_rate=16000,
            metadata={"test": True}
        )
        mock_asr.process_tensor.return_value = mock_result
        mock_asr_class.return_value = mock_asr
        
        # Test ASR with configuration
        config = self.create_test_config()
        result = run_asr_inference(
            self.mock_audio_tensor,
            config=config
        )
        
        # Verify results
        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.transcript, "hello world")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "hello world")

    @patch('src.omoai.pipeline.postprocess.VLLMProcessor')
    def test_postprocessing_integration(self, mock_processor_class):
        """Test postprocessing integration."""
        # Mock vLLM processor
        mock_processor = MagicMock()
        mock_processor.generate_text.side_effect = [
            "Hello, world!",  # Punctuation response
            '{"bullets": ["Hello world"], "abstract": "A greeting"}'  # Summary response
        ]
        mock_processor_class.return_value = mock_processor
        
        # Create test ASR result
        asr_result = ASRResult(
            segments=[ASRSegment(0.0, 1.0, "hello world")],
            transcript="hello world",
            audio_duration_seconds=1.0,
            sample_rate=16000,
            metadata={}
        )
        
        # Test postprocessing
        config = self.create_test_config()
        result = postprocess_transcript(asr_result, config)
        
        # Verify results
        self.assertEqual(len(result.segments), 1)
        self.assertIn("Hello", result.transcript_punctuated)
        self.assertEqual(len(result.summary.bullets), 1)
        self.assertIn("greeting", result.summary.abstract)

    @patch('src.omoai.pipeline.pipeline.postprocess_transcript')
    @patch('src.omoai.pipeline.pipeline.run_asr_inference')
    @patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor')
    @patch('src.omoai.pipeline.pipeline.get_audio_info')
    def test_full_pipeline_integration(self, mock_audio_info, mock_preprocess, mock_asr, mock_postprocess):
        """Test the complete pipeline integration."""
        # Mock audio info
        mock_audio_info.return_value = {
            "duration_seconds": 1.0,
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
            "frame_count": 16000,
        }
        
        # Mock preprocessing
        mock_preprocess.return_value = (self.mock_audio_tensor, 16000)
        
        # Mock ASR
        mock_asr_result = ASRResult(
            segments=[ASRSegment(0.0, 1.0, "hello world")],
            transcript="hello world",
            audio_duration_seconds=1.0,
            sample_rate=16000,
            metadata={"model": "test"}
        )
        mock_asr.return_value = mock_asr_result
        
        # Mock postprocessing
        from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
        mock_postprocess_result = PostprocessResult(
            segments=[ASRSegment(0.0, 1.0, "Hello, world!")],
            transcript_punctuated="Hello, world!",
            summary=SummaryResult(["Hello world"], "A greeting", {}),
            metadata={"quality": "good"}
        )
        mock_postprocess.return_value = mock_postprocess_result
        
        # Create valid configuration
        config = self.create_test_config()
        
        # Test complete pipeline
        result = run_full_pipeline_memory(
            audio_input=b"fake_audio_bytes",
            config=config,
            validate_input=False  # Skip validation to avoid mocking complexity
        )
        
        # Verify pipeline result structure
        self.assertIsInstance(result, PipelineResult)
        
        # Verify final outputs
        self.assertEqual(result.transcript_raw, "hello world")
        self.assertEqual(result.transcript_punctuated, "Hello, world!")
        self.assertEqual(len(result.summary.bullets), 1)
        self.assertEqual(result.summary.abstract, "A greeting")
        
        # Verify intermediate results are preserved
        self.assertIsInstance(result.asr_result, ASRResult)
        self.assertIsInstance(result.postprocess_result, PostprocessResult)
        
        # Verify timing information
        self.assertIn("preprocessing", result.timing)
        self.assertIn("asr", result.timing)
        self.assertIn("postprocessing", result.timing)
        self.assertIn("total", result.timing)
        
        # Verify metadata structure
        self.assertIn("pipeline_version", result.metadata)
        self.assertIn("audio_info", result.metadata)
        self.assertIn("performance", result.metadata)
        self.assertIn("quality_metrics", result.metadata)
        
        # Verify performance metrics
        performance = result.metadata["performance"]
        self.assertIn("real_time_factor", performance)
        self.assertIn("audio_duration", performance)
        self.assertEqual(performance["audio_duration"], 1.0)
        
        # Verify quality metrics
        quality = result.metadata["quality_metrics"]
        self.assertIn("segments_count", quality)
        self.assertIn("transcript_length", quality)
        self.assertEqual(quality["segments_count"], 1)

    def test_pipeline_with_intermediate_saving(self):
        """Test pipeline with intermediate file saving."""
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess, \
             patch('src.omoai.pipeline.pipeline.run_asr_inference') as mock_asr, \
             patch('src.omoai.pipeline.pipeline.postprocess_transcript') as mock_postprocess, \
             patch('src.omoai.pipeline.pipeline.get_audio_info') as mock_audio_info, \
             patch('torchaudio.save') as mock_save:
            
            # Setup mocks
            mock_audio_info.return_value = {"duration_seconds": 1.0, "sample_rate": 16000, "channels": 1, "format": "wav", "frame_count": 16000}
            mock_preprocess.return_value = (self.mock_audio_tensor, 16000)
            
            mock_asr_result = ASRResult(
                [ASRSegment(0.0, 1.0, "test")], "test", 1.0, 16000, {}
            )
            mock_asr.return_value = mock_asr_result
            
            from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
            mock_postprocess.return_value = PostprocessResult(
                [ASRSegment(0.0, 1.0, "Test.")], "Test.",
                SummaryResult(["Test bullet"], "Test abstract", {}), {}
            )
            
            # Create output directory
            output_dir = self.temp_dir / "pipeline_output"
            
            # Test pipeline with intermediate saving
            config = self.create_test_config()
            result = run_full_pipeline_memory(
                audio_input=b"fake_audio",
                config=config,
                save_intermediates=True,
                output_dir=output_dir,
                validate_input=False
            )
            
            # Verify pipeline completed
            self.assertIsInstance(result, PipelineResult)
            
            # Verify intermediate files would be created
            mock_save.assert_called_once()  # Preprocessed audio saved
            
            # Verify output directory was created
            self.assertTrue(output_dir.exists())

    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess, \
             patch('src.omoai.pipeline.pipeline.get_audio_info') as mock_audio_info, \
             patch('src.omoai.pipeline.pipeline.validate_audio_input') as mock_validate:
            
            # Mock validation to pass
            mock_validate.return_value = True
            
            # Mock audio info to pass validation
            mock_audio_info.return_value = {
                "duration_seconds": 1.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "frame_count": 16000,
            }
            
            # Simulate preprocessing failure
            mock_preprocess.side_effect = ValueError("Audio preprocessing failed")
            
            config = self.create_test_config()
            
            # Test that error is properly propagated
            with self.assertRaises(RuntimeError) as ctx:
                run_full_pipeline_memory(
                    audio_input=b"invalid_audio",
                    config=config,
                    validate_input=True  # Enable validation but with mocked validation
                )
            
            # Verify error message includes context
            error_msg = str(ctx.exception)
            self.assertIn("Pipeline failed", error_msg)
            self.assertIn("Audio preprocessing failed", error_msg)

    def test_pipeline_performance_tracking(self):
        """Test that pipeline accurately tracks performance metrics."""
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess, \
             patch('src.omoai.pipeline.pipeline.run_asr_inference') as mock_asr, \
             patch('src.omoai.pipeline.pipeline.postprocess_transcript') as mock_postprocess, \
             patch('src.omoai.pipeline.pipeline.get_audio_info') as mock_audio_info:
            
            import time
            
            # Setup mocks with timing delays
            def slow_preprocess(*args, **kwargs):
                time.sleep(0.01)
                return self.mock_audio_tensor, 16000
            
            def slow_asr(*args, **kwargs):
                time.sleep(0.02)
                return ASRResult([], "test", 10.0, 16000, {})  # 10 second audio
            
            def slow_postprocess(*args, **kwargs):
                time.sleep(0.01)
                from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
                return PostprocessResult([], "Test.", SummaryResult([], "", {}), {})
            
            mock_audio_info.return_value = {"duration_seconds": 10.0, "sample_rate": 16000, "channels": 1, "format": "wav", "frame_count": 160000}
            mock_preprocess.side_effect = slow_preprocess
            mock_asr.side_effect = slow_asr
            mock_postprocess.side_effect = slow_postprocess
            
            config = self.create_test_config()
            
            # Measure pipeline execution
            start_time = time.time()
            result = run_full_pipeline_memory(
                audio_input=b"fake_audio",
                config=config,
                validate_input=False
            )
            total_time = time.time() - start_time
            
            # Verify timing accuracy (within reasonable tolerance)
            self.assertGreater(result.timing["preprocessing"], 0.005)
            self.assertGreater(result.timing["asr"], 0.015)
            self.assertGreater(result.timing["postprocessing"], 0.005)
            self.assertLess(abs(result.timing["total"] - total_time), 0.01)  # 10ms tolerance
            
            # Verify performance metrics
            rtf = result.metadata["performance"]["real_time_factor"]
            expected_rtf = result.timing["total"] / 10.0  # 10 second audio
            self.assertAlmostEqual(rtf, expected_rtf, places=3)

    def test_configuration_validation_integration(self):
        """Test that pipeline properly validates and uses configuration."""
        # Test with minimal config (should inherit model IDs)
        minimal_config = {
            "paths": {
                "chunkformer_dir": str(self.temp_dir / "chunkformer"),
                "chunkformer_checkpoint": str(self.temp_dir / "checkpoint"),
            },
            "llm": {
                "model_id": "base/model",
                "trust_remote_code": False,
            },
            "punctuation": {
                "llm": {"trust_remote_code": False},  # Should inherit model_id
                "system_prompt": "Add punctuation.",
            },
            "summarization": {
                "llm": {"trust_remote_code": False},  # Should inherit model_id
                "system_prompt": "Summarize text.",
            }
        }
        
        config = OmoAIConfig(**minimal_config)
        
        # Verify inheritance worked
        self.assertEqual(config.punctuation.llm.model_id, "base/model")
        self.assertEqual(config.summarization.llm.model_id, "base/model")
        
        # Test that pipeline accepts this configuration
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess, \
             patch('src.omoai.pipeline.pipeline.run_asr_inference') as mock_asr, \
             patch('src.omoai.pipeline.pipeline.postprocess_transcript') as mock_postprocess, \
             patch('src.omoai.pipeline.pipeline.get_audio_info') as mock_audio_info:
            
            # Setup mocks
            mock_audio_info.return_value = {"duration_seconds": 1.0, "sample_rate": 16000, "channels": 1, "format": "wav", "frame_count": 16000}
            mock_preprocess.return_value = (self.mock_audio_tensor, 16000)
            mock_asr.return_value = ASRResult([], "test", 1.0, 16000, {})
            
            from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
            mock_postprocess.return_value = PostprocessResult([], "Test.", SummaryResult([], "", {}), {})
            
            # Should work without errors
            result = run_full_pipeline_memory(
                audio_input=b"fake_audio",
                config=config,
                validate_input=False
            )
            
            self.assertIsInstance(result, PipelineResult)


if __name__ == "__main__":
    unittest.main()
