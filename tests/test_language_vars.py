"""Unit tests for nupunkt language variables module."""

import pytest

from nupunkt.core.language_vars import PunktLanguageVars


def test_punkt_language_vars_basic():
    """Test basic properties of PunktLanguageVars."""
    lang_vars = PunktLanguageVars()

    # Test sentence end characters
    assert "." in lang_vars.sent_end_chars
    assert "?" in lang_vars.sent_end_chars
    assert "!" in lang_vars.sent_end_chars

    # Test internal punctuation
    assert "," in lang_vars.internal_punctuation
    assert ":" in lang_vars.internal_punctuation
    assert ";" in lang_vars.internal_punctuation


def test_punkt_language_vars_word_tokenize():
    """Test word tokenization in PunktLanguageVars."""
    lang_vars = PunktLanguageVars()

    # Test simple word tokenization
    words = lang_vars.word_tokenize("Hello world")
    assert words == ["Hello", "world"]

    # Test with punctuation
    words = lang_vars.word_tokenize("Hello, world!")
    assert words == ["Hello", ",", "world", "!"]

    # Test with period - accepting current behavior which doesn't split period
    words = lang_vars.word_tokenize("End.")
    assert words == ["End."]


def test_punkt_language_vars_properties():
    """Test properties of PunktLanguageVars."""
    lang_vars = PunktLanguageVars()

    # Test regex properties
    assert hasattr(lang_vars, "word_tokenize_pattern")
    assert hasattr(lang_vars, "period_context_pattern")

    # Test case detection via first_upper/first_lower
    from nupunkt.core.tokens import PunktToken

    upper_token = PunktToken("Word")
    assert upper_token.first_upper
    assert not upper_token.first_lower

    lower_token = PunktToken("word")
    assert not lower_token.first_upper
    assert lower_token.first_lower


def test_punkt_language_vars_pattern_matching():
    """Test pattern matching in PunktLanguageVars."""
    lang_vars = PunktLanguageVars()

    # Test period context pattern (need to access the property first to initialize it)
    pattern = lang_vars.period_context_pattern
    assert pattern.search("Mr. Smith")
    assert pattern.search("U.S.A. ")

    # Test word tokenize pattern
    assert lang_vars.word_tokenize_pattern is not None
    assert "Word" in lang_vars.word_tokenize("Word")
    assert "123" in lang_vars.word_tokenize("123")

    # In current implementation, "Word." is a single token
    tokens = lang_vars.word_tokenize("Word.")
    assert "Word." in tokens  # Period is attached to the word

    # Test pattern with special chars
    tokens = lang_vars.word_tokenize("A. Smith")
    assert "A." in tokens
    assert "Smith" in tokens


def test_punkt_language_vars_custom():
    """Test customizing PunktLanguageVars."""

    class CustomLanguageVars(PunktLanguageVars):
        """A custom language vars class with different sentence endings."""

        sent_end_chars = (".", "?", "!", ";")  # Add semicolon as sentence end

    custom_vars = CustomLanguageVars()

    # Check that our custom class has semicolon as sentence end
    assert ";" in custom_vars.sent_end_chars
    assert ";" not in PunktLanguageVars().sent_end_chars


@pytest.mark.benchmark(group="language_vars")
def test_word_tokenize_benchmark(benchmark):
    """Benchmark the word_tokenize method."""
    lang_vars = PunktLanguageVars()

    text = """
    This is a benchmark test for word tokenization functionality in PunktLanguageVars.
    It contains multiple sentences, with various punctuation marks like commas, periods, 
    question marks, and exclamation points! Does it handle all of these correctly?
    Numbers like 3.14 and abbreviations like Dr. Smith should be handled properly.
    U.S.A. is a country. This sentence ends the paragraph.
    
    This starts a new paragraph. It should be tokenized correctly as well.
    """

    # Run the benchmark
    tokens = benchmark(lambda: lang_vars.word_tokenize(text))

    # Verify we got reasonable results
    assert len(tokens) > 0
    assert "This" in tokens
    assert "benchmark" in tokens

    # Verify specific tokens that include periods
    assert any(token.endswith(".") for token in tokens)
    assert "U.S.A." in tokens or "U.S.A" in tokens
