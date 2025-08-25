"""
Unit test to validate ASR configuration fix and prevent regression.
This test ensures that the ASR script can run successfully with proper configuration.
"""
import json
import tempfile
import pytest
from pathlib import Path
from src.omoai.api.scripts.asr_wrapper import run_asr_script
from src.omoai.api.exceptions import AudioProcessingException


def test_asr_script_with_valid_config():
    """Test that ASR script runs successfully with valid configuration."""
    # Use the existing test audio file
    test_audio_path = Path(__file__).parent.parent / "src" / "omoai" / "chunkformer" / "data" / "common_voice_vi_23397238.wav"
    
    # Skip test if audio file doesn't exist
    if not test_audio_path.exists():
        pytest.skip(f"Test audio file not found: {test_audio_path}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        # Get the config file path
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        # Run ASR script - this should not raise an exception
        run_asr_script(
            audio_path=test_audio_path,
            output_path=output_path,
            config_path=config_path
        )
        
        # Verify output file was created and contains valid JSON
        assert output_path.exists(), "ASR output file should be created"
        
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Validate the structure of the output
        assert "audio" in result, "Output should contain audio metadata"
        assert "segments" in result, "Output should contain segments"
        assert "transcript_raw" in result, "Output should contain raw transcript"
        assert "metadata" in result, "Output should contain metadata"
        
        # Validate audio metadata
        audio_info = result["audio"]
        assert "sr" in audio_info, "Audio info should contain sample rate"
        assert "path" in audio_info, "Audio info should contain file path"
        assert "duration_s" in audio_info, "Audio info should contain duration"
        
        # Validate metadata
        metadata = result["metadata"]
        assert "asr_model" in metadata, "Metadata should contain model name"
        assert "params" in metadata, "Metadata should contain parameters"
        
        # Verify model name indicates correct configuration
        assert "chunkformer" in metadata["asr_model"].lower(), "Model should be chunkformer"
        
        # Verify segments exist (even if empty list)
        assert isinstance(result["segments"], list), "Segments should be a list"
        
        # Verify transcript is a string
        assert isinstance(result["transcript_raw"], str), "Transcript should be a string"
        
    finally:
        # Clean up temporary file
        if output_path.exists():
            output_path.unlink()


def test_asr_script_with_invalid_config():
    """Test that ASR script fails gracefully with invalid configuration."""
    # Use the existing test audio file
    test_audio_path = Path(__file__).parent.parent / "src" / "omoai" / "chunkformer" / "data" / "common_voice_vi_23397238.wav"
    
    # Skip test if audio file doesn't exist
    if not test_audio_path.exists():
        pytest.skip(f"Test audio file not found: {test_audio_path}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = Path(f.name)
    
    # Create invalid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
paths:
  chunkformer_dir: /invalid/path/to/chunkformer
  chunkformer_checkpoint: /invalid/path/to/checkpoint
""")
        invalid_config_path = Path(f.name)
    
    try:
        # This should raise an AudioProcessingException
        with pytest.raises(AudioProcessingException) as exc_info:
            run_asr_script(
                audio_path=test_audio_path,
                output_path=output_path,
                config_path=invalid_config_path
            )
        
        # Verify the exception message indicates the failure
        assert "ASR processing failed" in str(exc_info.value)
        
    finally:
        # Clean up temporary files
        if output_path.exists():
            output_path.unlink()
        if invalid_config_path.exists():
            invalid_config_path.unlink()


def test_asr_script_error_logging():
    """Test that ASR script provides detailed error information."""
    # Use the existing test audio file
    test_audio_path = Path(__file__).parent.parent / "src" / "omoai" / "chunkformer" / "data" / "common_voice_vi_23397238.wav"
    
    # Skip test if audio file doesn't exist
    if not test_audio_path.exists():
        pytest.skip(f"Test audio file not found: {test_audio_path}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = Path(f.name)
    
    # Create invalid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
paths:
  chunkformer_dir: /invalid/path/to/chunkformer
  chunkformer_checkpoint: /invalid/path/to/checkpoint
""")
        invalid_config_path = Path(f.name)
    
    try:
        # This should raise an AudioProcessingException with detailed error info
        with pytest.raises(AudioProcessingException) as exc_info:
            run_asr_script(
                audio_path=test_audio_path,
                output_path=output_path,
                config_path=invalid_config_path
            )
        
        # Verify the exception contains detailed error information
        error_message = str(exc_info.value)
        assert "return code" in error_message, "Error should contain return code"
        # The enhanced error logging should capture stderr/stdout
        assert len(error_message) > 20, "Error message should be detailed"
        
    finally:
        # Clean up temporary files
        if output_path.exists():
            output_path.unlink()
        if invalid_config_path.exists():
            invalid_config_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])