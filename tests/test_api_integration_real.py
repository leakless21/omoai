#!/usr/bin/env python3
"""
Integration tests for API endpoints using real audio files.

Tests the actual API endpoints with real audio files to simulate real-world usage.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import wave
import requests
from io import BytesIO

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from litestar import Litestar
from litestar.testing import TestClient
from omoai.api.app import create_app
from omoai.config import get_config


def create_test_wav_file(file_path, duration_seconds=3.0, sample_rate=16000, frequency=440):
    """Create a realistic WAV file with speech-like characteristics for testing."""
    # Generate audio with multiple frequencies to simulate speech
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    
    # Create a more complex signal that resembles speech patterns
    # Base frequency with modulation
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics
    audio_data += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    audio_data += 0.2 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Add amplitude modulation to simulate speech syllables
    mod_freq = 3  # 3 syllables per second
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    audio_data = audio_data * envelope
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.02, len(t))
    audio_data += noise
    
    # Normalize and convert to 16-bit PCM
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(str(file_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return file_path


def create_vietnamese_speech_wav(file_path, duration_seconds=5.0, sample_rate=16000):
    """Create a WAV file that simulates Vietnamese speech patterns."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    
    # Vietnamese has tones and specific pitch patterns
    # Simulate different tones with frequency variations
    
    # Base frequencies for different Vietnamese phonemes
    base_freqs = [220, 247, 262, 294, 330, 349, 392]  # A3 to G4
    
    # Create segments with different tones
    segments = []
    segment_duration = duration_seconds / 7  # 7 segments
    
    audio_data = np.zeros_like(t)
    
    for i, freq in enumerate(base_freqs):
        start_idx = int(i * segment_duration * sample_rate)
        end_idx = int((i + 1) * segment_duration * sample_rate)
        if end_idx > len(t):
            end_idx = len(t)
        
        segment_t = t[start_idx:end_idx]
        
        # Create tone with Vietnamese characteristics
        # Vietnamese has 6 tones: level, rising, falling, dipping, broken, heavy
        tone_patterns = [
            lambda t: np.ones_like(t),  # level
            lambda t: 1 + 0.3 * t / segment_duration,  # rising
            lambda t: 1 - 0.3 * t / segment_duration,  # falling
            lambda t: 1 + 0.2 * np.sin(2 * np.pi * 2 * t / segment_duration),  # dipping
            lambda t: 1 + 0.4 * np.sin(2 * np.pi * 10 * t / segment_duration),  # broken
            lambda t: 1 - 0.3 * t / segment_duration + 0.1 * np.sin(2 * np.pi * 5 * t / segment_duration)  # heavy
        ]
        
        tone_pattern = tone_patterns[i % len(tone_patterns)]
        modulated_freq = freq * tone_pattern(segment_t)
        
        # Generate the signal for this segment
        segment_signal = np.sin(2 * np.pi * modulated_freq * segment_t)
        
        # Add harmonics for richness
        segment_signal += 0.3 * np.sin(2 * np.pi * modulated_freq * 2 * segment_t)
        segment_signal += 0.1 * np.sin(2 * np.pi * modulated_freq * 3 * segment_t)
        
        # Apply envelope
        envelope = np.exp(-segment_t / (segment_duration * 0.3))
        segment_signal = segment_signal * envelope
        
        audio_data[start_idx:end_idx] = segment_signal
    
    # Add some background noise
    noise = np.random.normal(0, 0.01, len(t))
    audio_data += noise
    
    # Normalize and convert to 16-bit PCM
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.7
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(str(file_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return file_path


class TestAPIIntegrationWithRealAudio(unittest.TestCase):
    """Integration tests for API endpoints using real audio files."""

    def setUp(self):
        """Set up test environment with real audio files."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test audio files
        self.short_audio_path = self.temp_dir / "short_test.wav"
        create_test_wav_file(self.short_audio_path, duration_seconds=1.0)
        
        self.medium_audio_path = self.temp_dir / "medium_test.wav"
        create_test_wav_file(self.medium_audio_path, duration_seconds=3.0)
        
        self.vietnamese_audio_path = self.temp_dir / "vietnamese_test.wav"
        create_vietnamese_speech_wav(self.vietnamese_audio_path, duration_seconds=5.0)
        
        # Create test client
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("details", data)

    

    @patch('omoai.api.services.run_preprocess_script')
    def test_preprocess_endpoint_with_real_audio(self, mock_run_script):
        """Test preprocess endpoint with real audio file."""
        # Mock the script to simulate successful preprocessing
        def mock_preprocess(input_path, output_path):
            # Create a dummy output file to simulate preprocessing
            Path(output_path).touch()
        
        mock_run_script.side_effect = mock_preprocess
        
        # Read the real audio file
        with open(self.short_audio_path, 'rb') as f:
            audio_content = f.read()
        
        # Make request
        response = self.client.post(
            "/preprocess",
            files={"audio_file": ("test.wav", audio_content, "audio/wav")}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("output_path", data)
        self.assertTrue(data["output_path"].endswith(".wav"))
        
        # Verify the mock was called
        mock_run_script.assert_called_once()

    @patch('omoai.api.services.run_asr_script')
    def test_asr_endpoint_with_real_audio(self, mock_run_script):
        """Test ASR endpoint with real audio file."""
        # Mock the script to simulate successful ASR processing
        def mock_asr(audio_path, output_path, config_path):
            # Create a realistic ASR output JSON file
            asr_result = {
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": "xin chào"},
                    {"start": 0.5, "end": 1.0, "text": "thế giới"}
                ],
                "transcript_raw": "xin chào thế giới"
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asr_result, f, ensure_ascii=False)
        
        mock_run_script.side_effect = mock_asr
        
        # First, preprocess the audio to get a preprocessed file path
        with patch('omoai.api.services.run_preprocess_script') as mock_preprocess:
            preprocessed_path = self.temp_dir / "preprocessed.wav"
            mock_preprocess.side_effect = lambda input_path, output_path: Path(output_path).touch()
            
            with open(self.short_audio_path, 'rb') as f:
                audio_content = f.read()
            
            preprocess_response = self.client.post(
                "/preprocess",
                files={"audio_file": ("test.wav", audio_content, "audio/wav")}
            )
            preprocessed_path = preprocess_response.json()["output_path"]
        
        # Now test ASR endpoint
        asr_request = {"preprocessed_path": preprocessed_path}
        response = self.client.post("/asr", json=asr_request)
        
        # Verify response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("segments", data)
        self.assertIsInstance(data["segments"], list)
        
        # Verify the mock was called
        mock_run_script.assert_called_once()

    @patch('omoai.api.services.run_postprocess_script')
    def test_postprocess_endpoint_with_real_data(self, mock_run_script):
        """Test postprocess endpoint with realistic ASR data."""
        # Mock the script to simulate successful post-processing
        def mock_postprocess(asr_json_path, output_path, config_path):
            # Create a realistic post-processing output JSON file
            postprocess_result = {
                "summary": {
                    "bullets": ["Xin chào thế giới", "Đây là một bài kiểm tra"],
                    "abstract": "Một đoạn văn bản ngắn bằng tiếng Việt."
                },
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": "Xin chào,"},
                    {"start": 0.5, "end": 1.0, "text": "thế giới."},
                    {"start": 1.0, "end": 1.5, "text": "Đây là một bài kiểm tra."}
                ]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(postprocess_result, f, ensure_ascii=False)
        
        mock_run_script.side_effect = mock_postprocess
        
        # Create realistic ASR data
        asr_output = {
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "xin chào"},
                {"start": 0.5, "end": 1.0, "text": "thế giới"},
                {"start": 1.0, "end": 1.5, "text": "đây là một bài kiểm tra"}
            ],
            "transcript_raw": "xin chào thế giới đây là một bài kiểm tra"
        }
        
        # Make request - wrap ASR output in PostprocessRequest format
        postprocess_request = {"asr_output": asr_output}
        response = self.client.post("/postprocess", json=postprocess_request)
        
        # Verify response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("summary", data)
        self.assertIn("segments", data)
        self.assertIn("bullets", data["summary"])
        self.assertIn("abstract", data["summary"])
        
        # Verify the mock was called
        mock_run_script.assert_called_once()

    @patch('omoai.api.services.run_preprocess_script')
    @patch('omoai.api.services.run_asr_script')
    @patch('omoai.api.services.run_postprocess_script')
    def test_full_pipeline_endpoint_with_real_audio(self, mock_postprocess, mock_asr, mock_preprocess):
        """Test full pipeline endpoint with real audio file."""
        # Mock preprocessing
        def mock_preprocess_func(input_path, output_path):
            Path(output_path).touch()
        
        mock_preprocess.side_effect = mock_preprocess_func
        
        # Mock ASR
        def mock_asr_func(audio_path, output_path, config_path):
            asr_result = {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "xin chào thế giới"},
                    {"start": 1.0, "end": 2.0, "text": "đây là bài kiểm tra"},
                    {"start": 2.0, "end": 3.0, "text": "chúc bạn một ngày tốt lành"}
                ],
                "transcript_raw": "xin chào thế giới đây là bài kiểm tra chúc bạn một ngày tốt lành"
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asr_result, f, ensure_ascii=False)
        
        mock_asr.side_effect = mock_asr_func
        
        # Mock post-processing
        def mock_postprocess_func(asr_json_path, output_path, config_path):
            postprocess_result = {
                "summary": {
                    "bullets": [
                        "Xin chào thế giới",
                        "Đây là bài kiểm tra",
                        "Chúc bạn một ngày tốt lành"
                    ],
                    "abstract": "Một đoạn văn bản tiếng Việt với lời chào và bài kiểm tra."
                },
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Xin chào thế giới."},
                    {"start": 1.0, "end": 2.0, "text": "Đây là bài kiểm tra."},
                    {"start": 2.0, "end": 3.0, "text": "Chúc bạn một ngày tốt lành."}
                ]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(postprocess_result, f, ensure_ascii=False)
        
        mock_postprocess.side_effect = mock_postprocess_func
        
        # Read the real audio file
        with open(self.medium_audio_path, 'rb') as f:
            audio_content = f.read()
        
        # Make request
        response = self.client.post(
            "/pipeline",
            files={"audio_file": ("test.wav", audio_content, "audio/wav")}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("summary", data)
        self.assertIn("segments", data)
        self.assertIn("bullets", data["summary"])
        self.assertIn("abstract", data["summary"])
        
        # Verify all mocks were called
        mock_preprocess.assert_called_once()
        mock_asr.assert_called_once()
        mock_postprocess.assert_called_once()

    @patch('omoai.api.services.run_preprocess_script')
    @patch('omoai.api.services.run_asr_script')
    @patch('omoai.api.services.run_postprocess_script')
    def test_full_pipeline_with_output_parameters(self, mock_postprocess, mock_asr, mock_preprocess):
        """Test full pipeline with output parameters."""
        # Mock preprocessing
        def mock_preprocess_func(input_path, output_path):
            Path(output_path).touch()
        
        mock_preprocess.side_effect = mock_preprocess_func
        
        # Mock ASR
        def mock_asr_func(audio_path, output_path, config_path):
            asr_result = {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "xin chào thế giới"},
                    {"start": 1.0, "end": 2.0, "text": "đây là bài kiểm tra"},
                    {"start": 2.0, "end": 3.0, "text": "chúc bạn một ngày tốt lành"}
                ]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asr_result, f, ensure_ascii=False)
        
        mock_asr.side_effect = mock_asr_func
        
        # Mock post-processing
        def mock_postprocess_func(asr_json_path, output_path, config_path):
            postprocess_result = {
                "summary": {
                    "bullets": [
                        "Xin chào thế giới",
                        "Đây là bài kiểm tra",
                        "Chúc bạn một ngày tốt lành",
                        "Một ngày tuyệt vời",
                        "Hãy tận hưởng nó"
                    ],
                    "abstract": "Một đoạn văn bản tiếng Việt với lời chào và bài kiểm tra."
                },
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Xin chào thế giới."},
                    {"start": 1.0, "end": 2.0, "text": "Đây là bài kiểm tra."},
                    {"start": 2.0, "end": 3.0, "text": "Chúc bạn một ngày tốt lành."}
                ]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(postprocess_result, f, ensure_ascii=False)
        
        mock_postprocess.side_effect = mock_postprocess_func
        
        # Read the real audio file
        with open(self.medium_audio_path, 'rb') as f:
            audio_content = f.read()
        
        # Make request with output parameters
        response = self.client.post(
            "/pipeline?summary=bullets&summary_bullets_max=3&include=segments",
            files={"audio_file": ("test.wav", audio_content, "audio/wav")}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        
        # Verify filtering was applied
        self.assertEqual(len(data["summary"]["bullets"]), 3)  # Limited to 3 bullets
        self.assertNotIn("abstract", data["summary"])  # Only bullets kept
        self.assertEqual(len(data["segments"]), 3)  # Segments included

    @patch('omoai.api.services.run_preprocess_script')
    def test_preprocess_with_vietnamese_audio(self, mock_run_script):
        """Test preprocessing with Vietnamese audio characteristics."""
        def mock_preprocess(input_path, output_path):
            Path(output_path).touch()
        
        mock_run_script.side_effect = mock_preprocess
        
        # Read the Vietnamese audio file
        with open(self.vietnamese_audio_path, 'rb') as f:
            audio_content = f.read()
        
        # Make request
        response = self.client.post(
            "/preprocess",
            files={"audio_file": ("vietnamese.wav", audio_content, "audio/wav")}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("output_path", data)
        
        # Verify the mock was called
        mock_run_script.assert_called_once()

    def test_error_handling_invalid_audio_format(self):
        """Test error handling for invalid audio format."""
        # Create invalid audio data
        invalid_audio = b"this is not a valid audio file"
        
        # Make request
        response = self.client.post(
            "/preprocess",
            files={"audio_file": ("invalid.txt", invalid_audio, "text/plain")}
        )
        
        # The response should indicate an error
        # Note: The actual status code depends on the error handling implementation
        self.assertIn(response.status_code, [400, 422, 500])

    def test_error_handling_missing_audio_file(self):
        """Test error handling for missing audio file."""
        # Make request without audio file
        response = self.client.post("/preprocess")
        
        # Should return an error
        self.assertIn(response.status_code, [400, 422])

    @patch('omoai.api.services.run_preprocess_script')
    def test_error_handling_preprocess_failure(self, mock_run_script):
        """Test error handling when preprocessing fails."""
        # Mock the script to raise an exception
        mock_run_script.side_effect = Exception("Preprocessing failed")
        
        # Read the real audio file
        with open(self.short_audio_path, 'rb') as f:
            audio_content = f.read()
        
        # Make request
        response = self.client.post(
            "/preprocess",
            files={"audio_file": ("test.wav", audio_content, "audio/wav")}
        )
        
        # Should return an error
        self.assertIn(response.status_code, [500])


if __name__ == "__main__":
    unittest.main()