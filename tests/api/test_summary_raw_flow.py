import types


def test_output_format_params_parse_raw():
    from src.omoai.api.models import OutputFormatParams

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
