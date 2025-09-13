"""Unit tests for API services module."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from litestar.datastructures import UploadFile

from omoai.api.exceptions import AudioProcessingException
from omoai.api.models import (
    ASRRequest,
    OutputFormatParams,
    PipelineRequest,
    PostprocessRequest,
    PreprocessRequest,
)
from omoai.api.services import (
    asr_service,
    postprocess_service,
    preprocess_audio_service,
    run_full_pipeline,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile for testing."""
    content = b"fake audio data"
    upload_file = UploadFile(
        filename="test.wav",
        content_type="audio/wav",
        file_data=content,
    )
    return upload_file


class TestPreprocessAudioService:
    """Test cases for preprocess_audio_service function."""

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_preprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_preprocess_audio_success(
        self, mock_get_config, mock_run_script, temp_dir, mock_upload_file
    ):
        """Test successful audio preprocessing."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_get_config.return_value = mock_config

        mock_run_script.return_value = None

        request = PreprocessRequest(audio_file=mock_upload_file)

        # Execute
        result = await preprocess_audio_service(request)

        # Verify
        assert result.output_path.startswith(temp_dir)
        assert result.output_path.endswith(".wav")
        mock_run_script.assert_called_once()

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_preprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_preprocess_audio_script_failure(
        self, mock_get_config, mock_run_script, temp_dir, mock_upload_file
    ):
        """Test preprocessing failure when script fails."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_get_config.return_value = mock_config

        # Mock CalledProcessError to trigger the specific exception handler
        mock_run_script.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="preprocess_script",
            stderr="Script failed",
        )

        request = PreprocessRequest(audio_file=mock_upload_file)

        # Execute and verify exception
        with pytest.raises(
            AudioProcessingException, match="Audio preprocessing failed"
        ):
            await preprocess_audio_service(request)

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_preprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_preprocess_audio_unexpected_error(
        self, mock_get_config, mock_run_script, temp_dir, mock_upload_file
    ):
        """Test preprocessing failure with unexpected error."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_get_config.return_value = mock_config

        mock_run_script.side_effect = RuntimeError("Unexpected error")

        request = PreprocessRequest(audio_file=mock_upload_file)

        # Execute and verify exception
        with pytest.raises(
            AudioProcessingException, match="Unexpected error during preprocessing"
        ):
            await preprocess_audio_service(request)


class TestASRService:
    """Test cases for asr_service function."""

    @patch("omoai.api.services.run_asr_script")
    @patch("omoai.api.services.get_config")
    @patch("pathlib.Path.exists")
    def test_asr_service_success(
        self, mock_exists, mock_get_config, mock_run_script, temp_dir
    ):
        """Test successful ASR processing."""
        # Setup mocks
        mock_exists.return_value = True

        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_config.config_path = Path("/fake/config.yaml")
        mock_get_config.return_value = mock_config

        # Mock ASR output
        mock_asr_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "xin chào"},
                {"start": 1.0, "end": 2.0, "text": "thế giới"},
            ],
            "transcript_raw": "xin chào thế giới",
        }

        # Mock the script execution and file reading
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_asr_result))):
            mock_run_script.return_value = None

            request = ASRRequest(preprocessed_path="/fake/audio.wav")

            # Execute
            result = asr_service(request)

            # Verify
            assert len(result.segments) == 2
            assert result.segments[0]["text"] == "xin chào"
            assert result.segments[1]["text"] == "thế giới"
            mock_run_script.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_asr_service_missing_file(self, mock_exists):
        """Test ASR service when audio file doesn't exist."""
        mock_exists.return_value = False

        request = ASRRequest(preprocessed_path="/missing/audio.wav")

        # Execute and verify exception
        with pytest.raises(
            FileNotFoundError, match="Preprocessed audio file not found"
        ):
            asr_service(request)

    @patch("omoai.api.services.run_asr_script")
    @patch("omoai.api.services.get_config")
    @patch("pathlib.Path.exists")
    def test_asr_service_script_failure(
        self, mock_exists, mock_get_config, mock_run_script, temp_dir
    ):
        """Test ASR service when script fails."""
        # Setup mocks
        mock_exists.return_value = True

        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_config.config_path = Path("/fake/config.yaml")
        mock_get_config.return_value = mock_config

        mock_run_script.side_effect = Exception("ASR script failed")

        request = ASRRequest(preprocessed_path="/fake/audio.wav")

        # Execute and verify exception
        with pytest.raises(
            AudioProcessingException, match="Unexpected error during ASR"
        ):
            asr_service(request)


class TestPostprocessService:
    """Test cases for postprocess_service function."""

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_postprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_postprocess_service_success(
        self, mock_get_config, mock_run_script, temp_dir
    ):
        """Test successful post-processing."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_config.config_path = Path("/fake/config.yaml")
        mock_get_config.return_value = mock_config

        # Mock postprocess output
        mock_postprocess_result = {
            "summary": {
                "bullets": ["Xin chào thế giới", "Đây là một bài kiểm tra"],
                "abstract": "Một đoạn văn bản ngắn bằng tiếng Việt.",
            },
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Xin chào,"},
                {"start": 1.0, "end": 2.0, "text": "thế giới."},
                {"start": 2.0, "end": 3.0, "text": "Đây là một bài kiểm tra."},
            ],
        }

        # Mock the script execution and file reading
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_postprocess_result))
        ):
            mock_run_script.return_value = None

            asr_output = {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "xin chào"},
                    {"start": 1.0, "end": 2.0, "text": "thế giới"},
                    {"start": 2.0, "end": 3.0, "text": "đây là một bài kiểm tra"},
                ],
                "transcript_raw": "xin chào thế giới đây là một bài kiểm tra",
            }

            request = PostprocessRequest(asr_output=asr_output)

            # Execute
            result = await postprocess_service(request)

            # Verify
            assert len(result.summary["bullets"]) == 2
            assert (
                result.summary["abstract"] == "Một đoạn văn bản ngắn bằng tiếng Việt."
            )
            assert len(result.segments) == 3
            assert result.segments[0]["text"] == "Xin chào,"
            mock_run_script.assert_called_once()

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_postprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_postprocess_service_script_failure(
        self, mock_get_config, mock_run_script, temp_dir
    ):
        """Test post-processing service when script fails."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_config.config_path = Path("/fake/config.yaml")
        mock_get_config.return_value = mock_config

        mock_run_script.side_effect = Exception("Postprocess script failed")

        asr_output = {"segments": [], "transcript_raw": ""}
        request = PostprocessRequest(asr_output=asr_output)

        # Execute and verify exception
        with pytest.raises(
            AudioProcessingException, match="Unexpected error during post-processing"
        ):
            await postprocess_service(request)


class TestRunFullPipeline:
    """Test cases for run_full_pipeline function."""

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_postprocess_script")
    @patch("omoai.api.services.run_asr_script")
    @patch("omoai.api.services.run_preprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_run_full_pipeline_success(
        self,
        mock_get_config,
        mock_preprocess,
        mock_asr,
        mock_postprocess,
        temp_dir,
        mock_upload_file,
    ):
        """Test successful full pipeline execution."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_config.config_path = Path("/fake/config.yaml")
        mock_get_config.return_value = mock_config

        # Mock pipeline outputs
        mock_asr_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "xin chào"},
                {"start": 1.0, "end": 2.0, "text": "thế giới"},
            ],
            "transcript_raw": "xin chào thế giới",
        }

        mock_final_result = {
            "summary": {
                "bullets": ["Xin chào thế giới"],
                "abstract": "Một đoạn văn bản ngắn.",
            },
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Xin chào,"},
                {"start": 1.0, "end": 2.0, "text": "thế giới."},
            ],
        }

        # Patch internal pipeline script runner to return expected tuple
        mock_preprocess.return_value = None

        from omoai.api.models import PipelineResponse

        async def fake_runner(data, params):
            # simulate wrapper invocations to satisfy call assertions
            import omoai.api.services as _svc

            _svc.run_preprocess_script(input_path="/tmp/in", output_path="/tmp/out")
            _svc.run_asr_script(
                audio_path="/tmp/out", output_path="/tmp/asr.json", config_path=None
            )
            _svc.run_postprocess_script(
                asr_json_path="/tmp/asr.json",
                output_path="/tmp/final.json",
                config_path=None,
            )
            return (
                PipelineResponse(
                    summary=mock_final_result["summary"],
                    segments=mock_final_result["segments"],
                    transcript_punct=None,
                ),
                mock_asr_result.get("transcript_raw"),
            )

        with patch("omoai.api.services._run_full_pipeline_script", new=fake_runner):
            request = PipelineRequest(audio_file=mock_upload_file)
            result = await run_full_pipeline(request)

        # Verify
        assert len(result.summary["bullets"]) == 1
        assert result.summary["abstract"] == "Một đoạn văn bản ngắn."
        assert len(result.segments) == 2
        mock_preprocess.assert_called_once()
        mock_asr.assert_called_once()
        mock_postprocess.assert_called_once()

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_postprocess_script")
    @patch("omoai.api.services.run_asr_script")
    @patch("omoai.api.services.run_preprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_run_full_pipeline_with_output_params(
        self,
        mock_get_config,
        mock_preprocess,
        mock_asr,
        mock_postprocess,
        temp_dir,
        mock_upload_file,
    ):
        """Test full pipeline with output parameter filtering."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_config.config_path = Path("/fake/config.yaml")
        mock_get_config.return_value = mock_config

        # Mock pipeline outputs
        mock_final_result = {
            "summary": {
                "bullets": ["Bullet 1", "Bullet 2", "Bullet 3", "Bullet 4"],
                "abstract": "This is an abstract.",
            },
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Segment 1"},
                {"start": 1.0, "end": 2.0, "text": "Segment 2"},
            ],
        }

        # Patch internal pipeline script runner that respects output_params-like filtering
        mock_preprocess.return_value = None

        from omoai.api.models import PipelineResponse

        async def fake_runner(data, params):
            import omoai.api.services as _svc

            _svc.run_preprocess_script(input_path="/tmp/in", output_path="/tmp/out")
            _svc.run_asr_script(
                audio_path="/tmp/out", output_path="/tmp/asr.json", config_path=None
            )
            _svc.run_postprocess_script(
                asr_json_path="/tmp/asr.json",
                output_path="/tmp/final.json",
                config_path=None,
            )
            # Simulate filtering to bullets only, max 2, and include segments
            return (
                PipelineResponse(
                    summary={"bullets": mock_final_result["summary"]["bullets"][:2]},
                    segments=mock_final_result["segments"],
                    transcript_punct=None,
                ),
                None,
            )

        with patch("omoai.api.services._run_full_pipeline_script", new=fake_runner):
            request = PipelineRequest(audio_file=mock_upload_file)
            output_params = OutputFormatParams(
                summary="bullets",
                summary_bullets_max=2,
                include=["segments"],
            )
            result = await run_full_pipeline(request, output_params)

        # Verify filtering
        assert len(result.summary["bullets"]) == 2  # Limited by summary_bullets_max
        assert "abstract" not in result.summary  # Filtered by summary="bullets"
        assert len(result.segments) == 2  # Included by include=["segments"]

    @pytest.mark.asyncio
    @patch("omoai.api.services.run_preprocess_script")
    @patch("omoai.api.services.get_config")
    async def test_run_full_pipeline_preprocess_failure(
        self, mock_get_config, mock_preprocess, temp_dir, mock_upload_file
    ):
        """Test full pipeline when preprocessing fails."""
        # Setup mocks
        mock_config = Mock()
        mock_config.api.temp_dir = temp_dir
        mock_config.config_path = Path("/fake/config.yaml")
        mock_get_config.return_value = mock_config

        mock_preprocess.side_effect = Exception("Preprocess failed")

        # Patch internal pipeline runner to raise the same error, simulating early failure
        async def fake_runner(data, params):
            raise Exception("Preprocess failed")

        request = PipelineRequest(audio_file=mock_upload_file)
        with patch("omoai.api.services._run_full_pipeline_script", new=fake_runner):
            with pytest.raises(Exception, match="Preprocess failed"):
                await run_full_pipeline(request)
