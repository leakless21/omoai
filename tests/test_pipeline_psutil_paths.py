import json
import os
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from litestar.datastructures import UploadFile

from omoai.api.models import OutputFormatParams, PipelineRequest
from omoai.api.services import run_full_pipeline


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


def _fake_psutil_module():
    mod = types.SimpleNamespace()

    class _Mem:
        def __init__(self):
            self.rss = 512 * 1024 * 1024  # 512 MB

    class _Proc:
        def memory_info(self):
            return _Mem()

    mod.Process = _Proc
    return mod


@pytest.mark.asyncio
async def test_full_pipeline_runs_with_psutil_present_and_unpatched_open(
    monkeypatch, temp_dir
):
    # Make psutil importable and working
    monkeypatch.setitem(os.sys.modules, "psutil", _fake_psutil_module())

    # Point config temp_dir to our tmp
    class _Cfg:
        pass

    class _Api:
        pass

    cfg = _Cfg()
    cfg.api = _Api()
    cfg.api.temp_dir = temp_dir
    monkeypatch.setattr("omoai.api.services.get_config", lambda: cfg)

    # Track that preprocess saw an existing input_path
    seen_existing_input = {"ok": False}

    def _preprocess(input_path, output_path):
        # Ensure the uploaded file path exists at this point
        if Path(input_path).exists():
            seen_existing_input["ok"] = True
        Path(output_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    def _asr(audio_path, output_path, config_path=None):
        asr_obj = {
            "segments": [{"start": 0, "end": 1, "text": "hello"}],
            "text": "hello",
        }
        Path(output_path).write_text(json.dumps(asr_obj), encoding="utf-8")

    def _post(asr_json_path, output_path, config_path=None):
        final = {
            "summary": {"bullets": ["greeting"], "abstract": "hello world"},
            "segments": [{"start": 0, "end": 1, "text": "Hello."}],
            "transcript_punct": "Hello.",
        }
        Path(output_path).write_text(json.dumps(final), encoding="utf-8")

    monkeypatch.setattr("omoai.api.services.run_preprocess_script", _preprocess)
    monkeypatch.setattr("omoai.api.services.run_asr_script", _asr)
    monkeypatch.setattr("omoai.api.services.run_postprocess_script", _post)

    # Create a realistic UploadFile
    upload = UploadFile(
        filename="test.wav", content_type="audio/wav", file_data=b"dummy-bytes"
    )
    req = PipelineRequest(audio_file=upload)

    result = await run_full_pipeline(req)

    assert result is not None
    assert result.summary
    assert result.summary.get("bullets") == ["greeting"]
    assert result.transcript_punct == "Hello."
    # Ensure input existed when preprocess ran
    assert seen_existing_input["ok"] is True


@pytest.mark.asyncio
async def test_full_pipeline_runs_with_open_patched(monkeypatch, temp_dir):
    # Simulate open being patched; pipeline should still run and skip psutil
    monkeypatch.setattr("builtins.open", MagicMock())

    # Ensure psutil is NOT present to validate the open-patched path
    if "psutil" in os.sys.modules:
        del os.sys.modules["psutil"]

    class _Cfg:
        pass

    class _Api:
        pass

    cfg = _Cfg()
    cfg.api = _Api()
    cfg.api.temp_dir = temp_dir
    monkeypatch.setattr("omoai.api.services.get_config", lambda: cfg)

    def _preprocess(input_path, output_path):
        Path(output_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    def _asr(audio_path, output_path, config_path=None):
        asr_obj = {
            "segments": [{"start": 0, "end": 1, "text": "xin chao"}],
            "text": "xin chao",
        }
        Path(output_path).write_text(json.dumps(asr_obj), encoding="utf-8")

    def _post(asr_json_path, output_path, config_path=None):
        final = {
            "summary": {"bullets": ["xin chào"], "abstract": "chao the gioi"},
            "segments": [{"start": 0, "end": 1, "text": "Xin chào."}],
            "transcript_punct": "Xin chào.",
        }
        Path(output_path).write_text(json.dumps(final), encoding="utf-8")

    monkeypatch.setattr("omoai.api.services.run_preprocess_script", _preprocess)
    monkeypatch.setattr("omoai.api.services.run_asr_script", _asr)
    monkeypatch.setattr("omoai.api.services.run_postprocess_script", _post)

    upload = UploadFile(
        filename="test.wav", content_type="audio/wav", file_data=b"dummy-bytes"
    )
    req = PipelineRequest(audio_file=upload)

    params = OutputFormatParams(
        summary="bullets", summary_bullets_max=1, include=["segments"]
    )
    result = await run_full_pipeline(req, params)

    assert list(result.summary.keys()) == ["bullets"]
    assert result.summary["bullets"] == ["xin chào"]
    assert len(result.segments) == 1
