import json
from pathlib import Path

import types

from omoai.pipeline.postprocess import punctuate_transcript, summarize_text


class StubTokenizer:
    def encode(self, text: str):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Simplified: concatenate user content
        return "\n\n".join([m["content"] for m in messages if m.get("role") == "user"])  # type: ignore


class StubLLM:
    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, params):
        class _Out:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Res:
            def __init__(self, text: str) -> None:
                self.outputs = [types.SimpleNamespace(text=text)]

        # Echo prompt back as deterministic output
        return [_Res(prompts[0])]


def test_punctuate_text_stub():
    llm = StubLLM()
    text = "xin chao moi nguoi"
    from omoai.pipeline.asr import ASRResult, ASRSegment
    from omoai.config import PunctuationConfig, LLMConfig
    asr_result = ASRResult(segments=[ASRSegment(start=0, end=1, text=text)], transcript=text, audio_duration_seconds=1, sample_rate=16000, metadata={})
    config = PunctuationConfig(llm=LLMConfig(model_id="test"), system_prompt="test")
    out = punctuate_transcript(asr_result, config)
    assert "xin chao moi nguoi" in out[0].text


def test_summarize_text_stub_json_shape():
    text = "day la doan van can tom tat"
    from omoai.config import SummarizationConfig, LLMConfig
    config = SummarizationConfig(llm=LLMConfig(model_id="test"), system_prompt="stub")
    res = summarize_text(text, config)
    assert "bullets" in res and "abstract" in res



