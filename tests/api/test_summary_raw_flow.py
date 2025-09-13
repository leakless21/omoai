import types


def test_output_format_params_parse_raw():
    from omoai.api.models import OutputFormatParams

    p = OutputFormatParams(return_summary_raw=True)
    assert p.return_summary_raw is True


def test_summarize_text_includes_raw(monkeypatch):
    import scripts.post as post

    # stub generate_chat to a known content
    called = {}

    def fake_generate_chat(llm, messages, temperature, max_tokens):
        called["messages"] = messages
        return "Title: Demo\nSummary: This is the abstract.\nPoints:\n- A\n- B"

    monkeypatch.setattr(post, "generate_chat", fake_generate_chat)

    out = post.summarize_text(llm=object(), text="hello", system_prompt="sys")
    assert out["title"].lower().startswith("demo")
    assert "abstract" in out["summary"] or isinstance(out["summary"], str)
    assert out.get("raw").startswith("Title: Demo")


def test_summarize_map_reduce_exposes_raw_single_reduce(monkeypatch):
    import scripts.post as post

    # Force a single reduce chunk
    monkeypatch.setattr(post, "_split_text_by_token_budget", lambda llm, text, max_input_tokens: [text])

    def fake_generate_chat(llm, messages, temperature, max_tokens):
        return '{"bullets": ["x"], "abstract": "y"}'

    monkeypatch.setattr(post, "generate_chat", fake_generate_chat)

    out = post.summarize_long_text_map_reduce(llm=object(), text="abc", system_prompt="sys")
    # We only guarantee presence of fields and raw exposure here
    assert "bullets" in out and isinstance(out["bullets"], list)
    assert "abstract" in out and isinstance(out["abstract"], str)
    assert out.get("raw") is not None


def test_summarize_map_reduce_exposes_raw_multi_reduce(monkeypatch):
    import scripts.post as post

    # Force multiple reduce chunks
    monkeypatch.setattr(post, "_split_text_by_token_budget", lambda llm, text, max_input_tokens: [text[:1], text[1:2]])

    # Single prompt path for map step
    monkeypatch.setattr(post, "tqdm", None)

    def fake_generate_chat(llm, messages, temperature, max_tokens):
        return '{"bullets": ["a"], "abstract": "b"}'

    def fake_generate_chat_batch(llm, list_of_messages, temperature, max_tokens):
        return [
            '{"bullets": ["c"], "abstract": "d"}',
            '{"bullets": ["e"], "abstract": "f"}',
        ]

    monkeypatch.setattr(post, "generate_chat", fake_generate_chat)
    monkeypatch.setattr(post, "generate_chat_batch", fake_generate_chat_batch)

    out = post.summarize_long_text_map_reduce(llm=object(), text="abcd", system_prompt="sys")
    assert "bullets" in out and isinstance(out["bullets"], list)
    assert "abstract" in out and isinstance(out["abstract"], str)
    assert out.get("raw") is not None  # joined batch outputs

def test_asr_transcript_raw_exposed(monkeypatch, tmp_path):
    import json
    from unittest.mock import Mock, mock_open, patch
    from pathlib import Path
    from omoai.api import services
    from omoai.api.models import ASRRequest

    # Mock config and environment
    mock_config = Mock()
    mock_config.api.temp_dir = str(tmp_path)
    mock_config.config_path = Path("/fake/config.yaml")
    monkeypatch.setattr(services, "get_config", lambda: mock_config)

    # Pretend preprocessed file exists
    monkeypatch.setattr(Path, "exists", lambda self: True)

    # ASR output JSON contains 'text' key (raw transcript)
    asr_output = {"segments": [], "text": "dummy raw transcript"}
    mo = mock_open(read_data=json.dumps(asr_output))

    # Mock the run_asr_script to not actually invoke external process
    monkeypatch.setattr(services, "run_asr_script", lambda audio_path, output_path, config_path=None: None)

    with patch("builtins.open", mo):
        request = ASRRequest(preprocessed_path="/fake/audio.wav")
        result = services.asr_service(request)
        assert result.transcript_raw == "dummy raw transcript"

import pytest
from types import SimpleNamespace
from unittest.mock import patch

@pytest.mark.asyncio
async def test_pipeline_exposes_transcript_raw():
    """
    Verify that the /pipeline endpoint response includes the transcript_raw field
    when the pipeline service returns a raw transcript.
    """
    from omoai.api.app import create_app
    from litestar.testing import TestClient

    # Prepare dummy pipeline result that includes transcript_raw on response
    expected_raw = "dummy raw transcript"
    async def fake_run_full_pipeline(data, params):
        return SimpleNamespace(
            transcript_punct="This is punctuated.",
            summary={"title": "T", "summary": "A", "points": ["p"]},
            segments=[],
            transcript_raw=expected_raw,
        )

    # Patch the run_full_pipeline used by the controller and hit the endpoint
    with patch("omoai.api.main_controller.run_full_pipeline", new=fake_run_full_pipeline):
        app = create_app()
        with TestClient(app=app) as client:
            resp = client.post("/pipeline", files={"audio_file": ("a.wav", b"123", "audio/wav")})
            assert resp.status_code in (200, 201)
            data = resp.json()
            assert data.get("transcript_raw") == expected_raw
