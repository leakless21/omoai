import sys
import os
import pytest

# Add the script's directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from post import _force_preserve_with_alignment

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