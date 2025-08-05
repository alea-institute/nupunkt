"""Tests for the PunktSentenceTokenizer, focusing on deterministic behavior."""

import pytest

from nupunkt.core.parameters import PunktParameters
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer


@pytest.fixture
def params() -> PunktParameters:
    """Returns a manually configured PunktParameters object for testing."""
    p = PunktParameters()
    # Add common abbreviations that might appear in tests
    p.add_abbreviation("dr")
    p.add_abbreviation("mr")
    p.add_abbreviation("mrs")
    p.add_abbreviation("inc")
    p.add_abbreviation("d.c")
    # Add a sentence starter to enable basic splitting
    p.add_sent_starter("this")
    p.add_sent_starter("is")
    p.add_sent_starter("he")
    p.add_sent_starter("she")
    p.add_sent_starter("then")
    return p


@pytest.fixture
def tokenizer(params: PunktParameters) -> PunktSentenceTokenizer:
    """Returns a tokenizer configured with the manual params fixture."""
    return PunktSentenceTokenizer(params)


class TestPunktSentenceTokenizerInitialization:
    """Tests for the initialization of the PunktSentenceTokenizer."""

    def test_default_initialization(self):
        """Test that the tokenizer can be initialized with default parameters."""
        tokenizer = PunktSentenceTokenizer()
        assert tokenizer is not None
        assert isinstance(tokenizer._params, PunktParameters)
        assert len(tokenizer._params.abbrev_types) == 1  # '...' is a default abbrev

    def test_initialization_with_params(self, params: PunktParameters):
        """Test initialization with a pre-configured PunktParameters object."""
        tokenizer = PunktSentenceTokenizer(params)
        assert tokenizer._params is params
        assert "dr" in tokenizer._params.abbrev_types
        assert "d.c" in tokenizer._params.abbrev_types


class TestSimpleTokenization:
    """Tests for basic tokenization scenarios using a controlled tokenizer."""

    def test_standard_sentences(self, tokenizer: PunktSentenceTokenizer):
        """Test with a simple paragraph containing multiple standard sentences."""
        text = "Hello world. This is a test. Is it working?"
        expected: list[str] = ["Hello world.", "This is a test.", "Is it working?"]
        assert tokenizer.tokenize(text) == expected

    def test_no_sentence_breaks(self, tokenizer: PunktSentenceTokenizer):
        """Test with text that contains no sentence breaks."""
        text = "This is just one long line of text without any periods"
        expected: list[str] = ["This is just one long line of text without any periods"]
        assert tokenizer.tokenize(text) == expected

    def test_empty_string(self, tokenizer: PunktSentenceTokenizer):
        """Test with an empty string."""
        text = ""
        expected = []
        assert tokenizer.tokenize(text) == expected

    def test_whitespace_string(self, tokenizer: PunktSentenceTokenizer):
        """Test with text containing only whitespace."""
        text = "   \t\n   "
        expected = []
        assert tokenizer.tokenize(text) == expected

    def test_span_tokenize_simple(self, tokenizer: PunktSentenceTokenizer):
        """Test the span_tokenize method with simple text."""
        text = "First sentence. Second sentence."
        spans = list(tokenizer.span_tokenize(text))
        assert len(spans) == 2
        assert spans[0] == (0, 15)
        assert spans[1] == (16, 32)
        assert text[spans[0][0] : spans[0][1]] == "First sentence."
        assert text[spans[1][0] : spans[1][1]] == "Second sentence."

    def test_span_tokenize_no_breaks(self, tokenizer: PunktSentenceTokenizer):
        """Test span_tokenize on text with no breaks."""
        text = "A single sentence"
        spans = list(tokenizer.span_tokenize(text))
        assert len(spans) == 1
        assert spans[0] == (0, 17)
        assert text[spans[0][0] : spans[0][1]] == "A single sentence"

    def test_span_tokenize_empty_and_whitespace(self, tokenizer: PunktSentenceTokenizer):
        """Test span_tokenize with empty and whitespace-only strings."""
        assert list(tokenizer.span_tokenize("")) == []
        assert list(tokenizer.span_tokenize("   \n\t ")) == []


class TestAbbreviationAndEllipsisHandling:
    """Tests for more complex scenarios like abbreviations and ellipses."""

    def test_abbreviation_handling(self, tokenizer: PunktSentenceTokenizer):
        """Test that sentences are not split after a known abbreviation."""
        text = "Dr. Smith lives in Washington, D.C. He is a doctor."
        expected = ["Dr. Smith lives in Washington, D.C.", "He is a doctor."]
        assert tokenizer.tokenize(text) == expected

    def test_ellipsis_multi_period(self, tokenizer: PunktSentenceTokenizer):
        """Test that ellipsis (...) is handled correctly."""
        text = "This is the first part... and this is the second. A new sentence."
        expected = ["This is the first part... and this is the second.", "A new sentence."]
        assert tokenizer.tokenize(text) == expected

    def test_ellipsis_spaced(self, tokenizer: PunktSentenceTokenizer):
        """Test that spaced ellipsis (. . .) is handled correctly."""
        text = "This is the first part . . . and this is the second. A new sentence."
        expected = ["This is the first part . . . and this is the second.", "A new sentence."]
        assert tokenizer.tokenize(text) == expected

    def test_ellipsis_unicode(self, tokenizer: PunktSentenceTokenizer):
        """Test that unicode ellipsis (…) is handled correctly."""
        text = "This is the first part… and this is the second. A new sentence."
        expected = ["This is the first part… and this is the second.", "A new sentence."]
        assert tokenizer.tokenize(text) == expected

    def test_ellipsis_followed_by_uppercase_is_break(self, tokenizer: PunktSentenceTokenizer):
        """Test that an ellipsis followed by an uppercase word creates a break."""
        text = "The story trails off... Then it begins again."
        expected = ["The story trails off...", "Then it begins again."]
        assert tokenizer.tokenize(text) == expected


class TestOrthographicHeuristicsAndReAlignment:
    """Tests for orthographic heuristics and boundary realignment."""

    def test_abbreviation_followed_by_lowercase_is_not_break(
        self, tokenizer: PunktSentenceTokenizer
    ):
        """An abbreviation followed by a lowercase word should not be a sentence break."""
        # Manually add 'ft' as an abbreviation for this test
        tokenizer._params.add_abbreviation("ft")
        text = "The building is 50 ft. tall."
        expected = ["The building is 50 ft. tall."]
        assert tokenizer.tokenize(text) == expected

    def test_boundary_realignment_with_quotes(self, tokenizer: PunktSentenceTokenizer):
        """Test realignment of boundaries with quotes."""
        text = 'He said, "This is a sentence." Then he left.'
        expected = ['He said, "This is a sentence."', "Then he left."]
        spans = list(tokenizer.span_tokenize(text))
        sentences = [text[start:end] for start, end in spans]
        assert sentences == expected
        assert spans[0] == (0, 30)
        assert spans[1] == (31, 44)

    def test_boundary_realignment_with_brackets(self, tokenizer: PunktSentenceTokenizer):
        """Test realignment of boundaries with brackets."""
        text = "This is a sentence. (A new one begins here.)"
        expected = ["This is a sentence.", "(A new one begins here.)"]
        assert tokenizer.tokenize(text) == expected

    def test_no_break_before_internal_punctuation(self, tokenizer: PunktSentenceTokenizer):
        """Test that no break occurs before internal punctuation like commas."""
        text = "This is sentence one. And this, sentence two."
        expected = ["This is sentence one.", "And this, sentence two."]
        assert tokenizer.tokenize(text) == expected
