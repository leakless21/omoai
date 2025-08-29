import sys
import os
import pytest

# Add the script's directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from post import (
    _force_preserve_with_alignment,
    _align_chars,
    _compute_wer,
    _compute_cer,
    _compute_per,
    _compute_uwer_fwer,
    _generate_human_readable_diff,
    _distribute_punct_to_segments,
)

@pytest.mark.parametrize("original_text, llm_text, adopt_case, expected_output", [
    # Test case 1: Simple punctuation
    ("hello world", "Hello, world.", True, "Hello, world."),
    # Test case 2: Punctuation with no case adoption
    ("hello world", "Hello, world.", False, "hello, world."),
    # Test case 3: LLM inserts a word
    ("i am going to the store", "i am going to the big store.", True, "i am going to the big store."),
    # Test case 4: LLM deletes a word
    ("i am going to the big store", "i am going to the store.", True, "i am going to the store."),
    # Test case 5: LLM replaces a word
    ("the cat sat on the mat", "The dog sat on the mat.", True, "The dog sat on the mat."),
    # Test case 6: Complex case with multiple changes
    ("this is a test of the emergency broadcast system", "This is a test of the new emergency broadcast system.", True, "This is a test of the new emergency broadcast system."),
])
def test_force_preserve_with_alignment(original_text, llm_text, adopt_case, expected_output):
    assert _force_preserve_with_alignment(original_text, llm_text, adopt_case) == expected_output

# --- Tests for _align_chars ---
@pytest.mark.parametrize("orig_word, llm_word, expected_orig_chars, expected_llm_chars, expected_tags", [
    # Test case 1: Exact match
    ("hello", "hello", ["h","e","l","l","o"], ["h","e","l","l","o"], ["equal","equal","equal","equal","equal"]),
    # Test case 2: Case change
    ("hello", "Hello", ["h","e","l","l","o"], ["H","e","l","l","o"], ["replace","equal","equal","equal","equal"]),
    # Test case 3: Substitution
    ("cat", "bat", ["c","a","t"], ["b","a","t"], ["replace","equal","equal"]),
    # Test case 4: Insertion in LLM
    ("hello", "helllo", ["h","e","l","l","o"], ["h","e","l","l","l","o"], ["equal","equal","equal","equal","insert","equal"]),
    # Test case 5: Deletion in LLM
    ("hello", "helo", ["h","e","l","l","o"], ["h","e","l","o"], ["equal","equal","equal","delete","equal"]),
    # Test case 6: Complex change
    ("kitten", "sitting", ["k","i","t","t","e","n"], ["s","i","t","t","i","n","g"], ["replace","equal","equal","equal","replace","equal","insert"]),
])
def test_align_chars(orig_word, llm_word, expected_orig_chars, expected_llm_chars, expected_tags):
    orig_chars, llm_chars, tags = _align_chars(orig_word, llm_word)
    assert orig_chars == expected_orig_chars
    assert llm_chars == expected_llm_chars
    assert tags == expected_tags

# --- Tests for _compute_wer ---
@pytest.mark.parametrize("orig_words, llm_words, expected_wer", [
    # Test case 1: Exact match
    (["hello", "world"], ["hello", "world"], 0.0),
    # Test case 2: One substitution
    (["hello", "world"], ["hell", "world"], 0.5),
    # Test case 3: One insertion
    (["hello", "world"], ["hello", "big", "world"], 1/3),
    # Test case 4: One deletion
    (["hello", "big", "world"], ["hello", "world"], 1/3),
    # Test case 5: Empty original
    ([], ["hello"], 1.0),
    # Test case 6: Empty both
    ([], [], 0.0),
])
def test_compute_wer(orig_words, llm_words, expected_wer):
    assert _compute_wer(orig_words, llm_words) == pytest.approx(expected_wer)

# --- Tests for _compute_cer ---
@pytest.mark.parametrize("orig_text, llm_text, expected_cer", [
    # Test case 1: Exact match
    ("hello", "hello", 0.0),
    # Test case 2: One substitution
    ("hello", "hellx", 1/5),
    # Test case 3: One insertion
    ("hello", "helllo", 1/5),
    # Test case 4: One deletion
    ("hello", "helo", 1/5),
    # Test case 5: Empty original
    ("", "hello", 1.0),
    # Test case 6: Empty both
    ("", "", 0.0),
])
def test_compute_cer(orig_text, llm_text, expected_cer):
    assert _compute_cer(orig_text, llm_text) == pytest.approx(expected_cer)

# --- Tests for _compute_per ---
@pytest.mark.parametrize("orig_text, llm_text, expected_per", [
    # Test case 1: No punctuation
    ("hello world", "hello world", 0.0),
    # Test case 2: Exact punctuation match
    ("Hello, world!", "Hello, world!", 0.0),
    # Test case 3: One punctuation substitution
    ("Hello, world!", "Hello. world!", 1/2),
    # Test case 4: One punctuation insertion
    ("Hello world", "Hello, world", 1.0), # Original has 0, LLM has 1
    # Test case 5: One punctuation deletion
    ("Hello, world", "Hello world", 1.0), # Original has 1, LLM has 0
    # Test case 6: Empty original
    ("", "Hello!", 1.0),
    # Test case 7: Empty both
    ("", "", 0.0),
])
def test_compute_per(orig_text, llm_text, expected_per):
    assert _compute_per(orig_text, llm_text) == pytest.approx(expected_per)

# --- Tests for _compute_uwer_fwer ---
@pytest.mark.parametrize("orig_text, llm_text, expected_uwer, expected_fwer", [
    # Test case 1: Exact match
    ("Hello, world!", "Hello, world!", 0.0, 0.0),
    # Test case 2: Punctuation difference only
    ("Hello, world", "Hello. world", 0.0, 0.5), # U-WER should be 0, F-WER should be 0.5 (substitution of , for .)
    # Test case 3: Word and punctuation difference
    ("Hello, world", "Hi. world", 0.5, 0.5), # U-WER: "Hello" -> "Hi", F-WER: "Hello," -> "Hi." (1 edit on 2 words)
    # Test case 4: Empty original
    ("", "Hello!", 1.0, 1.0),
    # Test case 5: Empty both
    ("", "", 0.0, 0.0),
])
def test_compute_uwer_fwer(orig_text, llm_text, expected_uwer, expected_fwer):
    uwer, fwer = _compute_uwer_fwer(orig_text, llm_text)
    assert uwer == pytest.approx(expected_uwer)
    assert fwer == pytest.approx(expected_fwer)

# --- Tests for _generate_human_readable_diff ---
@pytest.mark.parametrize("orig_text, llm_text, expected_diff", [
    # Test case 1: Exact match
    ("hello world", "hello world", "  hello world"),
    # Test case 2: Substitution
    ("hello world", "hellx world", "  hell\n- o\n+ x\n   world"),
    # Test case 3: Insertion
    ("hello world", "hello big world", "  hello \n+ big \n  world"),
    # Test case 4: Deletion
    ("hello big world", "hello world", "  hello \n- big \n  world"),
    # Test case 5: Complex
    ("the cat sat", "the dog sat on mat", "  the \n- cat\n+ dog\n   sat\n+  on mat"),
    # Test case 6: Empty original
    ("", "hello", "+ hello"),
    # Test case 7: Empty LLM
    ("hello", "", "- hello"),
])
def test_generate_human_readable_diff(orig_text, llm_text, expected_diff):
    assert _generate_human_readable_diff(orig_text, llm_text) == expected_diff

# --- Tests for _distribute_punct_to_segments ---
@pytest.mark.parametrize("punctuated_text, segments, expected_segments", [
    # Test case 1: Exact word count match
    ("Hello, world.", [{"start": 0, "end": 1, "text_raw": "hello"}, {"start": 1, "end": 2, "text_raw": "world"}], [{"start": 0, "end": 1, "text_raw": "hello", "text_punct": "Hello,"}, {"start": 1, "end": 2, "text_raw": "world", "text_punct": "world."}]),
    # Test case 2: LLM inserts a word
    ("Hello, big world.", [{"start": 0, "end": 1, "text_raw": "hello"}, {"start": 1, "end": 2, "text_raw": "world"}], [{"start": 0, "end": 1, "text_raw": "hello", "text_punct": "Hello, big"}, {"start": 1, "end": 2, "text_raw": "world", "text_punct": "world."}]),
    # Test case 3: LLM deletes a word
    ("Hello, world.", [{"start": 0, "end": 1, "text_raw": "hello"}, {"start": 1, "end": 2, "text_raw": "big"}, {"start": 2, "end": 3, "text_raw": "world"}], [{"start": 0, "end": 1, "text_raw": "hello", "text_punct": "Hello,"}, {"start": 1, "end": 2, "text_raw": "big", "text_punct": ""}, {"start": 2, "end": 3, "text_raw": "world", "text_punct": "world."}]),
    # Test case 4: Empty segments
    ("Hello, world.", [], []),
    # Test case 5: Empty punctuated text
    ("", [{"start": 0, "end": 1, "text_raw": "hello"}], [{"start": 0, "end": 1, "text_raw": "hello", "text_punct": ""}]),
])
def test_distribute_punct_to_segments(punctuated_text, segments, expected_segments):
    result = _distribute_punct_to_segments(punctuated_text, segments)
    assert result == expected_segments

# --- Vietnamese Language Tests ---
# Tests for Vietnamese text with mixed English content

# --- Tests for _force_preserve_with_alignment with Vietnamese ---
@pytest.mark.parametrize("original_text, llm_text, adopt_case, expected_output", [
    # Test case 1: Vietnamese with English name
    ("tôi là john", "tôi là John.", True, "tôi là John."),
    # Test case 2: Vietnamese with English name, no case adoption
    ("tôi là john", "tôi là John.", False, "tôi là john."),
    # Test case 3: Vietnamese with technical term
    ("hôm nay học machine learning", "Hôm nay học machine learning.", True, "Hôm nay học machine learning."),
    # Test case 4: Vietnamese with company name
    ("tôi làm việc tại fpt", "Tôi làm việc tại FPT.", True, "Tôi làm việc tại FPT."),
    # Test case 5: Vietnamese with time expression
    ("lớp học bắt đầu lúc 9 am", "Lớp học bắt đầu lúc 9 AM.", True, "Lớp học bắt đầu lúc 9 AM."),
    # Test case 6: Vietnamese with multiple English terms
    ("chúng ta học deep learning và neural network", "Chúng ta học deep learning và neural network.", True, "Chúng ta học deep learning và neural network."),
    # Test case 7: Vietnamese with English question
    ("bạn có biết what is ai không", "Bạn có biết what is AI không?", True, "Bạn có biết what is AI không?"),
    # Test case 8: Vietnamese with English acronym
    ("tôi làm việc trong lĩnh vực ai", "Tôi làm việc trong lĩnh vực AI.", True, "Tôi làm việc trong lĩnh vực AI."),
])
def test_force_preserve_with_alignment_vietnamese(original_text, llm_text, adopt_case, expected_output):
    assert _force_preserve_with_alignment(original_text, llm_text, adopt_case) == expected_output

# --- Tests for _align_chars with Vietnamese ---
@pytest.mark.parametrize("orig_word, llm_word, expected_orig_chars, expected_llm_chars, expected_tags", [
    # Test case 1: Vietnamese exact match with diacritics
    ("xin", "xin", ["x","i","n"], ["x","i","n"], ["equal","equal","equal"]),
    # Test case 2: Vietnamese with diacritics
    ("xin", "Xin", ["x","i","n"], ["X","i","n"], ["replace","equal","equal"]),
    # Test case 3: Vietnamese word with diacritics
    ("chào", "chào", ["c","h","à","o"], ["c","h","à","o"], ["equal","equal","equal","equal"]),
    # Test case 4: Vietnamese with English word
    ("john", "John", ["j","o","h","n"], ["J","o","h","n"], ["replace","equal","equal","equal"]),
    # Test case 5: Vietnamese technical term
    ("learning", "Learning", ["l","e","a","r","n","i","n","g"], ["L","e","a","r","n","i","n","g"], ["replace","equal","equal","equal","equal","equal","equal","equal"]),
    # Test case 6: Vietnamese with complex diacritics
    ("học", "học", ["h","ọ","c"], ["h","ọ","c"], ["equal","equal","equal"]),
    # Test case 7: Vietnamese with English substitution
    ("ai", "AI", ["a","i"], ["A","I"], ["replace","replace"]),
    # Test case 8: Vietnamese with insertion
    ("tôi", "tôi ơi", ["t","ô","i"], ["t","ô","i"," ","ơ","i"], ["equal","equal","equal","insert","insert","insert"]),
])
def test_align_chars_vietnamese(orig_word, llm_word, expected_orig_chars, expected_llm_chars, expected_tags):
    orig_chars, llm_chars, tags = _align_chars(orig_word, llm_word)
    assert orig_chars == expected_orig_chars
    assert llm_chars == expected_llm_chars
    assert tags == expected_tags

# --- Tests for _compute_wer with Vietnamese ---
@pytest.mark.parametrize("orig_words, llm_words, expected_wer", [
    # Test case 1: Vietnamese exact match
    (["xin", "chào"], ["xin", "chào"], 0.0),
    # Test case 2: Vietnamese with English name
    (["tôi", "là", "john"], ["tôi", "là", "John"], 1/3),
    # Test case 3: Vietnamese with technical term
    (["hôm", "nay", "học", "machine", "learning"], ["hôm", "nay", "học", "Machine", "Learning"], 2/5),
    # Test case 4: Vietnamese with insertion
    (["tôi", "học"], ["tôi", "học", "AI"], 1/3),
    # Test case 5: Vietnamese with deletion
    (["tôi", "học", "AI"], ["tôi", "học"], 1/3),
    # Test case 6: Vietnamese mixed with English
    (["chúng", "ta", "học", "deep", "learning"], ["chúng", "ta", "học", "deep", "learning"], 0.0),
    # Test case 7: Empty Vietnamese
    ([], ["xin", "chào"], 1.0),
    # Test case 8: Vietnamese with complex diacritics
    (["học", "tập"], ["học", "tập"], 0.0),
])
def test_compute_wer_vietnamese(orig_words, llm_words, expected_wer):
    assert _compute_wer(orig_words, llm_words) == pytest.approx(expected_wer)

# --- Tests for _compute_cer with Vietnamese ---
@pytest.mark.parametrize("orig_text, llm_text, expected_cer", [
    # Test case 1: Vietnamese exact match
    ("xin chào", "xin chào", 0.0),
    # Test case 2: Vietnamese with diacritic change (1 substitution in 8 chars)
    ("xin chào", "xin chao", 1/8),
    # Test case 3: Vietnamese with English capitalization (1 substitution in 11 chars)
    ("tôi là john", "tôi là John", 1/11),
    # Test case 4: Vietnamese with insertion (3 insertions in 7 chars)
    ("tôi học", "tôi học AI", 3/7),
    # Test case 5: Vietnamese with deletion (3 deletions in 10 chars)
    ("tôi học AI", "tôi học", 3/10),
    # Test case 6: Vietnamese mixed with English (2 substitutions in 20 chars including space)
    ("học machine learning", "học Machine Learning", 2/20),
    # Test case 7: Empty Vietnamese
    ("", "xin chào", 1.0),
    # Test case 8: Vietnamese with complex diacritics (1 substitution in 7 chars)
    ("học tập", "học tạp", 1/7),
])
def test_compute_cer_vietnamese(orig_text, llm_text, expected_cer):
    assert _compute_cer(orig_text, llm_text) == pytest.approx(expected_cer)

# --- Tests for _compute_per with Vietnamese ---
@pytest.mark.parametrize("orig_text, llm_text, expected_per", [
    # Test case 1: Vietnamese no punctuation
    ("xin chào", "xin chào", 0.0),
    # Test case 2: Vietnamese exact punctuation match
    ("Xin chào!", "Xin chào!", 0.0),
    # Test case 3: Vietnamese punctuation substitution
    ("Xin chào,", "Xin chào.", 1.0),
    # Test case 4: Vietnamese punctuation insertion
    ("Xin chào", "Xin chào!", 1.0),
    # Test case 5: Vietnamese punctuation deletion
    ("Xin chào!", "Xin chào", 1.0),
    # Test case 6: Vietnamese with English mixed punctuation
    ("tôi là John", "tôi là John.", 1.0),
    # Test case 7: Vietnamese question
    ("bạn có khỏe không", "bạn có khỏe không?", 1.0),
    # Test case 8: Empty Vietnamese
    ("", "Xin chào!", 1.0),
])
def test_compute_per_vietnamese(orig_text, llm_text, expected_per):
    assert _compute_per(orig_text, llm_text) == pytest.approx(expected_per)

# --- Tests for _compute_uwer_fwer with Vietnamese ---
@pytest.mark.parametrize("orig_text, llm_text, expected_uwer, expected_fwer", [
    # Test case 1: Vietnamese exact match
    ("Xin chào!", "Xin chào!", 0.0, 0.0),
    # Test case 2: Vietnamese punctuation difference only (U-WER ignores punctuation, F-WER sees it as substitution)
    ("Xin chào,", "Xin chào.", 0.0, 0.5),
    # Test case 3: Vietnamese word and punctuation difference (U-WER: "Xin chào" -> "Chào bạn", F-WER: "Xin chào," -> "Chào bạn.")
    ("Xin chào,", "Chào bạn.", 1.0, 1.0),
    # Test case 4: Vietnamese with English name (U-WER ignores punctuation, F-WER includes it)
    ("tôi là john", "tôi là John.", 1/3, 1/3),
    # Test case 5: Vietnamese with technical term (U-WER: "hôm nay học machine learning" -> "Hôm nay học Machine Learning", F-WER includes punctuation)
    ("hôm nay học machine learning", "Hôm nay học Machine Learning.", 3/5, 3/5),
    # Test case 6: Vietnamese question (U-WER ignores punctuation, F-WER sees it as insertion)
    ("bạn có khỏe không", "bạn có khỏe không?", 0.0, 0.25),
    # Test case 7: Empty Vietnamese
    ("", "Xin chào!", 1.0, 1.0),
    # Test case 8: Vietnamese complex sentence (U-WER: "tôi học AI và machine learning" -> "Tôi học AI và Machine Learning", F-WER includes punctuation)
    ("tôi học AI và machine learning", "Tôi học AI và Machine Learning.", 3/6, 3/6),
])
def test_compute_uwer_fwer_vietnamese(orig_text, llm_text, expected_uwer, expected_fwer):
    uwer, fwer = _compute_uwer_fwer(orig_text, llm_text)
    assert uwer == pytest.approx(expected_uwer)
    assert fwer == pytest.approx(expected_fwer)

# --- Tests for _generate_human_readable_diff with Vietnamese ---
@pytest.mark.parametrize("orig_text, llm_text, expected_diff", [
    # Test case 1: Vietnamese exact match
    ("xin chào", "xin chào", "  xin chào"),
    # Test case 2: Vietnamese with diacritic change
    ("xin chào", "xin chao", "  xin ch\n- à\n+ a\n  o"),
    # Test case 3: Vietnamese with English capitalization
    ("tôi là john", "tôi là John", "  tôi là \n- j\n+ J\n  ohn"),
    # Test case 4: Vietnamese with insertion
    ("tôi học", "tôi học AI", "  tôi học\n+  AI"),
    # Test case 5: Vietnamese with deletion
    ("tôi học AI", "tôi học", "  tôi học\n-  AI"),
    # Test case 6: Vietnamese mixed with English
    ("học machine learning", "học Machine Learning", "  học \n- m\n+ M\n  achine \n- l\n+ L\n  earning"),
    # Test case 7: Vietnamese complex sentence
    ("tôi học AI", "tôi học AI và ML", "  tôi học AI\n+  và ML"),
    # Test case 8: Empty Vietnamese
    ("", "xin chào", "+ xin chào"),
])
def test_generate_human_readable_diff_vietnamese(orig_text, llm_text, expected_diff):
    assert _generate_human_readable_diff(orig_text, llm_text) == expected_diff

# --- Tests for _distribute_punct_to_segments with Vietnamese ---
@pytest.mark.parametrize("punctuated_text, segments, expected_segments", [
    # Test case 1: Vietnamese with English name
    ("tôi là John.", [{"start": 0, "end": 1, "text_raw": "tôi"}, {"start": 1, "end": 2, "text_raw": "là"}, {"start": 2, "end": 3, "text_raw": "john"}], [{"start": 0, "end": 1, "text_raw": "tôi", "text_punct": "tôi"}, {"start": 1, "end": 2, "text_raw": "là", "text_punct": "là"}, {"start": 2, "end": 3, "text_raw": "john", "text_punct": "John."}]),
    # Test case 2: Vietnamese with technical term
    ("Hôm nay học machine learning.", [{"start": 0, "end": 1, "text_raw": "hôm"}, {"start": 1, "end": 2, "text_raw": "nay"}, {"start": 2, "end": 3, "text_raw": "học"}, {"start": 3, "end": 4, "text_raw": "machine"}, {"start": 4, "end": 5, "text_raw": "learning"}], [{"start": 0, "end": 1, "text_raw": "hôm", "text_punct": "Hôm"}, {"start": 1, "end": 2, "text_raw": "nay", "text_punct": "nay"}, {"start": 2, "end": 3, "text_raw": "học", "text_punct": "học"}, {"start": 3, "end": 4, "text_raw": "machine", "text_punct": "machine"}, {"start": 4, "end": 5, "text_raw": "learning", "text_punct": "learning."}]),
    # Test case 3: Vietnamese with company name
    ("Tôi làm việc tại FPT.", [{"start": 0, "end": 1, "text_raw": "tôi"}, {"start": 1, "end": 2, "text_raw": "làm"}, {"start": 2, "end": 3, "text_raw": "việc"}, {"start": 3, "end": 4, "text_raw": "tại"}, {"start": 4, "end": 5, "text_raw": "fpt"}], [{"start": 0, "end": 1, "text_raw": "tôi", "text_punct": "Tôi"}, {"start": 1, "end": 2, "text_raw": "làm", "text_punct": "làm"}, {"start": 2, "end": 3, "text_raw": "việc", "text_punct": "việc"}, {"start": 3, "end": 4, "text_raw": "tại", "text_punct": "tại"}, {"start": 4, "end": 5, "text_raw": "fpt", "text_punct": "FPT."}]),
    # Test case 4: Vietnamese with time expression
    ("Lớp học bắt đầu lúc 9 AM.", [{"start": 0, "end": 1, "text_raw": "lớp"}, {"start": 1, "end": 2, "text_raw": "học"}, {"start": 2, "end": 3, "text_raw": "bắt"}, {"start": 3, "end": 4, "text_raw": "đầu"}, {"start": 4, "end": 5, "text_raw": "lúc"}, {"start": 5, "end": 6, "text_raw": "9"}, {"start": 6, "end": 7, "text_raw": "am"}], [{"start": 0, "end": 1, "text_raw": "lớp", "text_punct": "Lớp"}, {"start": 1, "end": 2, "text_raw": "học", "text_punct": "học"}, {"start": 2, "end": 3, "text_raw": "bắt", "text_punct": "bắt"}, {"start": 3, "end": 4, "text_raw": "đầu", "text_punct": "đầu"}, {"start": 4, "end": 5, "text_raw": "lúc", "text_punct": "lúc"}, {"start": 5, "end": 6, "text_raw": "9", "text_punct": "9"}, {"start": 6, "end": 7, "text_raw": "am", "text_punct": "AM."}]),
    # Test case 5: Vietnamese with multiple English terms
    ("Chúng ta học deep learning và neural network.", [{"start": 0, "end": 1, "text_raw": "chúng"}, {"start": 1, "end": 2, "text_raw": "ta"}, {"start": 2, "end": 3, "text_raw": "học"}, {"start": 3, "end": 4, "text_raw": "deep"}, {"start": 4, "end": 5, "text_raw": "learning"}, {"start": 5, "end": 6, "text_raw": "và"}, {"start": 6, "end": 7, "text_raw": "neural"}, {"start": 7, "end": 8, "text_raw": "network"}], [{"start": 0, "end": 1, "text_raw": "chúng", "text_punct": "Chúng"}, {"start": 1, "end": 2, "text_raw": "ta", "text_punct": "ta"}, {"start": 2, "end": 3, "text_raw": "học", "text_punct": "học"}, {"start": 3, "end": 4, "text_raw": "deep", "text_punct": "deep"}, {"start": 4, "end": 5, "text_raw": "learning", "text_punct": "learning"}, {"start": 5, "end": 6, "text_raw": "và", "text_punct": "và"}, {"start": 6, "end": 7, "text_raw": "neural", "text_punct": "neural"}, {"start": 7, "end": 8, "text_raw": "network", "text_punct": "network."}]),
    # Test case 6: Vietnamese with English question
    ("Bạn có biết what is AI không?", [{"start": 0, "end": 1, "text_raw": "bạn"}, {"start": 1, "end": 2, "text_raw": "có"}, {"start": 2, "end": 3, "text_raw": "biết"}, {"start": 3, "end": 4, "text_raw": "what"}, {"start": 4, "end": 5, "text_raw": "is"}, {"start": 5, "end": 6, "text_raw": "ai"}, {"start": 6, "end": 7, "text_raw": "không"}], [{"start": 0, "end": 1, "text_raw": "bạn", "text_punct": "Bạn"}, {"start": 1, "end": 2, "text_raw": "có", "text_punct": "có"}, {"start": 2, "end": 3, "text_raw": "biết", "text_punct": "biết"}, {"start": 3, "end": 4, "text_raw": "what", "text_punct": "what"}, {"start": 4, "end": 5, "text_raw": "is", "text_punct": "is"}, {"start": 5, "end": 6, "text_raw": "ai", "text_punct": "AI"}, {"start": 6, "end": 7, "text_raw": "không", "text_punct": "không?"}]),
    # Test case 7: Vietnamese with English acronym
    ("Tôi làm việc trong lĩnh vực AI.", [{"start": 0, "end": 1, "text_raw": "tôi"}, {"start": 1, "end": 2, "text_raw": "làm"}, {"start": 2, "end": 3, "text_raw": "việc"}, {"start": 3, "end": 4, "text_raw": "trong"}, {"start": 4, "end": 5, "text_raw": "lĩnh"}, {"start": 5, "end": 6, "text_raw": "vực"}, {"start": 6, "end": 7, "text_raw": "ai"}], [{"start": 0, "end": 1, "text_raw": "tôi", "text_punct": "Tôi"}, {"start": 1, "end": 2, "text_raw": "làm", "text_punct": "làm"}, {"start": 2, "end": 3, "text_raw": "việc", "text_punct": "việc"}, {"start": 3, "end": 4, "text_raw": "trong", "text_punct": "trong"}, {"start": 4, "end": 5, "text_raw": "lĩnh", "text_punct": "lĩnh"}, {"start": 5, "end": 6, "text_raw": "vực", "text_punct": "vực"}, {"start": 6, "end": 7, "text_raw": "ai", "text_punct": "AI."}]),
    # Test case 8: Vietnamese empty segments
    ("Xin chào!", [], []),
    # Test case 9: Vietnamese empty punctuated text
    ("", [{"start": 0, "end": 1, "text_raw": "xin"}], [{"start": 0, "end": 1, "text_raw": "xin", "text_punct": ""}]),
])
def test_distribute_punct_to_segments_vietnamese(punctuated_text, segments, expected_segments):
    result = _distribute_punct_to_segments(punctuated_text, segments)
    assert result == expected_segments