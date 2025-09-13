def test_normalize_summary_from_dict_bullets_and_abstract():
    from omoai.api.services import _normalize_summary

    raw = {
        "title": "Sample Title",
        "bullets": ["A", "B"],
        "abstract": "This is an abstract.",
    }

    out = _normalize_summary(raw)
    assert isinstance(out, dict)
    assert out["title"] == "Sample Title"
    assert out["summary"] == "This is an abstract."
    assert out["abstract"] == "This is an abstract."
    assert out["bullets"] == ["A", "B"]


def test_normalize_summary_from_labeled_text_vi():
    from omoai.api.services import _normalize_summary

    text = (
        "Tiêu đề: Demo\nTóm tắt: Đây là phần tóm tắt.\nĐiểm chính:\n- Mục 1\n- Mục 2\n"
    )

    out = _normalize_summary(text)
    assert isinstance(out, dict)
    assert out["title"].lower().startswith("demo")
    # Should not include the label prefix 'Tóm tắt:'
    assert not out["summary"].lower().startswith("tóm tắt:")
    assert out["bullets"] == ["Mục 1", "Mục 2"]
