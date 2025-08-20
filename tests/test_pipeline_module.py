#!/usr/bin/env python3
"""
Test the in-memory pipeline module for OMOAI.
"""
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import numpy as np
import torch

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.omoai.pipeline import (
    preprocess_audio_to_tensor,
    preprocess_audio_bytes, 
    get_audio_info,
    validate_audio_input,
    run_asr_inference,
    ASRResult,
    ASRSegment,
    postprocess_transcript,
    run_full_pipeline_memory,
    PipelineResult,
)


class TestPipelineModule(unittest.TestCase):
    """Test the in-memory pipeline processing functions."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create required directories for path validation
        (self.temp_dir / "chunkformer").mkdir()
        (self.temp_dir / "checkpoint").mkdir()
        
        # Create mock audio data (1 second of silence at 16kHz)
        self.mock_audio_data = np.zeros(16000, dtype=np.float32)
        self.mock_audio_tensor = torch.from_numpy(self.mock_audio_data).unsqueeze(0)
        
        # Create a minimal config for testing
        self.test_config = {
            "paths": {
                "chunkformer_dir": str(self.temp_dir / "chunkformer"),
                "chunkformer_checkpoint": str(self.temp_dir / "checkpoint"),
            },
            "llm": {
                "model_id": "test/model",
                "trust_remote_code": False,
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.85,
            },
            "asr": {
                "device": "cpu",
                "autocast_dtype": None,
                "chunk_size": 16,
                "left_context_size": 32,
                "right_context_size": 32,
                "total_batch_duration_s": 60,
            },
            "punctuation": {
                "llm": {
                    "trust_remote_code": False,  # Inherit model_id from base
                },
                "system_prompt": "Add punctuation.",
                "sampling": {"temperature": 0.0},
            },
            "summarization": {
                "llm": {
                    "trust_remote_code": False,  # Inherit model_id from base
                },
                "system_prompt": "Summarize text.",
                "sampling": {"temperature": 0.7},
            },
            "output": {
                "write_separate_files": True,
                "transcript_file": "transcript.txt",
                "summary_file": "summary.txt",
            }
        }

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audio_preprocessing_tensor(self):
        """Test audio preprocessing to tensor format."""
        # Mock pydub AudioSegment
        with patch('src.omoai.pipeline.preprocess.AudioSegment') as mock_audio:
            mock_segment = MagicMock()
            mock_segment.set_frame_rate.return_value = mock_segment
            mock_segment.set_sample_width.return_value = mock_segment  
            mock_segment.set_channels.return_value = mock_segment
            mock_segment.get_array_of_samples.return_value = self.mock_audio_data * 32768
            mock_audio.from_file.return_value = mock_segment
            
            # Test with bytes input
            audio_bytes = b"fake_audio_data"
            tensor, sample_rate = preprocess_audio_to_tensor(
                audio_bytes, 
                return_sample_rate=True
            )
            
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape[0], 1)  # Batch dimension
            self.assertEqual(sample_rate, 16000)
            
            # Verify the preprocessing chain was called
            mock_audio.from_file.assert_called_once()
            mock_segment.set_frame_rate.assert_called_with(16000)
            mock_segment.set_channels.assert_called_with(1)

    def test_audio_preprocessing_bytes(self):
        """Test audio preprocessing to bytes format."""
        input_audio = b"fake_audio_input"
        
        # Mock subprocess.run for ffmpeg
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = b"processed_wav_data"
            mock_run.return_value.returncode = 0
            
            result = preprocess_audio_bytes(input_audio)
            
            self.assertEqual(result, b"processed_wav_data")
            mock_run.assert_called_once()
            
            # Verify ffmpeg command structure
            call_args = mock_run.call_args[0][0]
            self.assertIn("ffmpeg", call_args)
            self.assertIn("-ac", call_args)
            self.assertIn("1", call_args)  # Mono
            self.assertIn("-ar", call_args) 
            self.assertIn("16000", call_args)  # 16kHz

    def test_audio_validation(self):
        """Test audio input validation."""
        # Mock get_audio_info
        with patch('src.omoai.pipeline.preprocess.get_audio_info') as mock_info:
            # Test valid audio
            mock_info.return_value = {
                "duration_seconds": 5.0,
                "sample_rate": 44100,
                "channels": 2,
            }
            
            self.assertTrue(validate_audio_input(b"audio_data"))
            
            # Test audio too short
            mock_info.return_value["duration_seconds"] = 0.05
            with self.assertRaises(ValueError) as ctx:
                validate_audio_input(b"audio_data", min_duration_seconds=0.1)
            self.assertIn("too short", str(ctx.exception))
            
            # Test audio too long
            mock_info.return_value["duration_seconds"] = 3700.0
            with self.assertRaises(ValueError) as ctx:
                validate_audio_input(b"audio_data", max_duration_seconds=3600.0)
            self.assertIn("too long", str(ctx.exception))

    def test_asr_result_structure(self):
        """Test ASR result data structures."""
        # Test ASRSegment
        segment = ASRSegment(
            start=0.0,
            end=1.0, 
            text="test text",
            confidence=0.95
        )
        
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 1.0)
        self.assertEqual(segment.text, "test text")
        self.assertEqual(segment.confidence, 0.95)
        
        # Test ASRResult
        result = ASRResult(
            segments=[segment],
            transcript="test text",
            audio_duration_seconds=1.0,
            sample_rate=16000,
            metadata={"test": "data"}
        )
        
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.transcript, "test text")
        self.assertEqual(result.audio_duration_seconds, 1.0)

    @patch('src.omoai.pipeline.asr.ChunkFormerASR')
    def test_asr_inference_tensor_input(self, mock_asr_class):
        """Test ASR inference with tensor input."""
        # Mock ChunkFormer ASR
        mock_asr = MagicMock()
        mock_result = ASRResult(
            segments=[ASRSegment(0.0, 1.0, "test")],
            transcript="test",
            audio_duration_seconds=1.0,
            sample_rate=16000,
            metadata={}
        )
        mock_asr.process_tensor.return_value = mock_result
        mock_asr_class.return_value = mock_asr
        
        # Test tensor input
        result = run_asr_inference(
            self.mock_audio_tensor,
            config=self.test_config,
            sample_rate=16000
        )
        
        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.transcript, "test")
        mock_asr.process_tensor.assert_called_once()

    @patch('src.omoai.pipeline.postprocess.VLLMProcessor')
    def test_postprocessing(self, mock_processor_class):
        """Test postprocessing functionality."""
        # Mock vLLM processor
        mock_processor = MagicMock()
        mock_processor.generate_text.side_effect = [
            "Test text with punctuation.",  # Punctuation response
            '{"bullets": ["Test bullet"], "abstract": "Test abstract"}'  # Summary response
        ]
        mock_processor_class.return_value = mock_processor
        
        # Create test ASR result
        asr_result = ASRResult(
            segments=[ASRSegment(0.0, 1.0, "test text")],
            transcript="test text", 
            audio_duration_seconds=1.0,
            sample_rate=16000,
            metadata={}
        )
        
        # Test postprocessing
        result = postprocess_transcript(asr_result, self.test_config)
        
        self.assertIn("punctuation", result.transcript_punctuated.lower())
        self.assertEqual(len(result.summary.bullets), 1)
        self.assertIn("Test abstract", result.summary.abstract)

    @patch('src.omoai.pipeline.pipeline.run_asr_inference')
    @patch('src.omoai.pipeline.pipeline.postprocess_transcript')  
    @patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor')
    def test_full_pipeline_memory(self, mock_preprocess, mock_postprocess, mock_asr):
        """Test the complete in-memory pipeline."""
        # Mock preprocessing
        mock_preprocess.return_value = (self.mock_audio_tensor, 16000)
        
        # Mock ASR
        mock_asr_result = ASRResult(
            segments=[ASRSegment(0.0, 1.0, "test")],
            transcript="test",
            audio_duration_seconds=1.0, 
            sample_rate=16000,
            metadata={}
        )
        mock_asr.return_value = mock_asr_result
        
        # Mock postprocessing
        from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
        mock_postprocess_result = PostprocessResult(
            segments=[ASRSegment(0.0, 1.0, "Test.")],
            transcript_punctuated="Test.",
            summary=SummaryResult(["Test bullet"], "Test abstract", {}),
            metadata={}
        )
        mock_postprocess.return_value = mock_postprocess_result
        
        # Test pipeline
        with patch('src.omoai.config.get_config') as mock_config:
            from src.omoai.config import OmoAIConfig
            mock_config.return_value = OmoAIConfig(**self.test_config)
            
            result = run_full_pipeline_memory(
                audio_input=b"fake_audio",
                validate_input=False  # Skip validation for test
            )
        
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.transcript_raw, "test")
        self.assertEqual(result.transcript_punctuated, "Test.")
        self.assertIn("total", result.timing)
        self.assertIn("preprocessing", result.timing)
        self.assertIn("asr", result.timing)
        self.assertIn("postprocessing", result.timing)

    def test_pipeline_timing_tracking(self):
        """Test that pipeline tracks timing information correctly."""
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess, \
             patch('src.omoai.pipeline.pipeline.run_asr_inference') as mock_asr, \
             patch('src.omoai.pipeline.pipeline.postprocess_transcript') as mock_postprocess, \
             patch('src.omoai.config.get_config') as mock_config:
            
            # Setup mocks with delays to test timing
            import time
            
            def slow_preprocess(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                return self.mock_audio_tensor, 16000
            
            def slow_asr(*args, **kwargs):
                time.sleep(0.02)  # 20ms delay
                return ASRResult([], "", 1.0, 16000, {})
            
            def slow_postprocess(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
                return PostprocessResult([], "", SummaryResult([], "", {}), {})
            
            mock_preprocess.side_effect = slow_preprocess
            mock_asr.side_effect = slow_asr
            mock_postprocess.side_effect = slow_postprocess
            
            from src.omoai.config import OmoAIConfig
            mock_config.return_value = OmoAIConfig(**self.test_config)
            
            result = run_full_pipeline_memory(
                audio_input=b"fake_audio",
                validate_input=False
            )
            
            # Verify timing information is captured
            self.assertGreater(result.timing["preprocessing"], 0.005)  # At least 5ms
            self.assertGreater(result.timing["asr"], 0.015)  # At least 15ms  
            self.assertGreater(result.timing["postprocessing"], 0.005)  # At least 5ms
            self.assertGreater(result.timing["total"], 0.035)  # At least 35ms total

    def test_error_handling(self):
        """Test error handling in pipeline functions."""
        # Test preprocessing error
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Error")
            
            with self.assertRaises(subprocess.CalledProcessError):
                preprocess_audio_bytes(b"invalid_audio")
        
        # Test validation error
        with self.assertRaises(ValueError):
            validate_audio_input(b"")  # Empty audio
        
        # Test pipeline error propagation
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess:
            mock_preprocess.side_effect = ValueError("Preprocessing failed")
            
            with patch('src.omoai.config.get_config') as mock_config:
                from src.omoai.config import OmoAIConfig
                mock_config.return_value = OmoAIConfig(**self.test_config)
                
                with self.assertRaises(RuntimeError) as ctx:
                    run_full_pipeline_memory(b"fake_audio", validate_input=False)
                
                self.assertIn("Pipeline failed", str(ctx.exception))
                self.assertIn("Preprocessing failed", str(ctx.exception))

    def test_intermediate_file_saving(self):
        """Test saving intermediate files during pipeline processing."""
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess, \
             patch('src.omoai.pipeline.pipeline.run_asr_inference') as mock_asr, \
             patch('src.omoai.pipeline.pipeline.postprocess_transcript') as mock_postprocess, \
             patch('src.omoai.config.get_config') as mock_config, \
             patch('torchaudio.save') as mock_save, \
             patch('builtins.open', mock_open()) as mock_file:
            
            # Setup mocks
            mock_preprocess.return_value = (self.mock_audio_tensor, 16000)
            mock_asr.return_value = ASRResult(
                [ASRSegment(0.0, 1.0, "test")], "test", 1.0, 16000, {}
            )
            
            from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
            mock_postprocess.return_value = PostprocessResult(
                [ASRSegment(0.0, 1.0, "Test.")], "Test.", 
                SummaryResult(["Test"], "Abstract", {}), {}
            )
            
            from src.omoai.config import OmoAIConfig
            mock_config.return_value = OmoAIConfig(**self.test_config)
            
            # Test with intermediate file saving
            result = run_full_pipeline_memory(
                audio_input=b"fake_audio",
                save_intermediates=True,
                output_dir=self.temp_dir,
                validate_input=False
            )
            
            # Verify files would be saved
            mock_save.assert_called_once()  # Preprocessed audio
            self.assertGreater(mock_file.call_count, 0)  # JSON files

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        with patch('src.omoai.pipeline.pipeline.preprocess_audio_to_tensor') as mock_preprocess, \
             patch('src.omoai.pipeline.pipeline.run_asr_inference') as mock_asr, \
             patch('src.omoai.pipeline.pipeline.postprocess_transcript') as mock_postprocess, \
             patch('src.omoai.config.get_config') as mock_config:
            
            # Setup mocks
            mock_preprocess.return_value = (self.mock_audio_tensor, 16000)
            mock_asr.return_value = ASRResult(
                [ASRSegment(0.0, 1.0, "test")], "test", 10.0, 16000, {}  # 10 second audio
            )
            
            from src.omoai.pipeline.postprocess import PostprocessResult, SummaryResult
            mock_postprocess.return_value = PostprocessResult(
                [ASRSegment(0.0, 1.0, "Test.")], "Test.", 
                SummaryResult(["Test"], "Abstract", {}), {}
            )
            
            from src.omoai.config import OmoAIConfig
            mock_config.return_value = OmoAIConfig(**self.test_config)
            
            result = run_full_pipeline_memory(
                audio_input=b"fake_audio",
                validate_input=False
            )
            
            # Check performance metrics
            self.assertIn("real_time_factor", result.metadata["performance"])
            self.assertIn("audio_duration", result.metadata["performance"])
            self.assertEqual(result.metadata["performance"]["audio_duration"], 10.0)
            
            # Real-time factor should be calculated correctly
            rtf = result.metadata["performance"]["real_time_factor"]
            self.assertIsInstance(rtf, float)
            self.assertGreater(rtf, 0)


if __name__ == "__main__":
    unittest.main()
