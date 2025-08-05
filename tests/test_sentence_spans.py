"""
Tests for sentence span functionality.
"""

import pytest

import nupunkt


@pytest.fixture
def tokenizer():
    """Create a sentence tokenizer instance."""
    from nupunkt import load

    return load("default")


class TestSentenceSpans:
    """Test cases for sentence span functionality."""

    def test_tokenize_with_spans_basic(self, tokenizer):
        """Test tokenize_with_spans returns both text and spans."""
        text = "First sentence. Second sentence."
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 2

        sent1, span1 = result[0]
        assert sent1 == "First sentence. "
        assert span1 == (0, 16)

        sent2, span2 = result[1]
        assert sent2 == "Second sentence."
        assert span2 == (16, 32)

    def test_tokenize_with_spans_empty(self, tokenizer):
        """Test tokenize_with_spans with empty string."""
        result = tokenizer.tokenize_with_spans("")
        assert result == []

    def test_tokenize_with_spans_single_sentence(self, tokenizer):
        """Test tokenize_with_spans with single sentence."""
        text = "Just one sentence"
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 1
        sent, span = result[0]
        assert sent == "Just one sentence"
        assert span == (0, 17)
        assert text[span[0] : span[1]] == sent

    def test_tokenize_with_spans_coverage(self, tokenizer):
        """Test that spans cover the entire text without gaps."""
        text = "First. Second. Third."
        result = tokenizer.tokenize_with_spans(text)
        spans = [span for _, span in result]

        # Check no gaps between spans
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0]

        # Check full coverage
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)

        # Verify sentences are extracted correctly (including trailing whitespace)
        sentences = [sent for sent, _ in result]
        assert sentences[0] == "First. "  # includes space
        assert sentences[1] == "Second. "  # includes space
        assert sentences[2] == "Third."  # last one has no trailing space

    def test_tokenize_with_spans_preserves_whitespace(self, tokenizer):
        """Test that tokenize_with_spans preserves all whitespace."""
        text = "First sentence.   Second sentence.  "
        result = tokenizer.tokenize_with_spans(text)

        # Verify sentences include inter-sentence whitespace
        assert len(result) == 2
        sent1, span1 = result[0]
        sent2, span2 = result[1]

        assert sent1 == "First sentence.   "  # includes trailing spaces
        assert sent2 == "Second sentence.  "  # includes trailing spaces
        assert text[span1[0] : span1[1]] == sent1
        assert text[span2[0] : span2[1]] == sent2

        # Verify contiguity
        assert span1[1] == span2[0]

        # Reconstruct text from spans
        reconstructed = "".join(sent for sent, _ in result)
        assert reconstructed == text

    def test_tokenize_with_spans_complex_punctuation(self, tokenizer):
        """Test handling of complex punctuation scenarios."""
        text = 'He said, "Hello there." Then he left.'
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 2
        sent1, span1 = result[0]
        sent2, span2 = result[1]

        assert sent1 == 'He said, "Hello there." '
        assert sent2 == "Then he left."
        assert text[span1[0] : span1[1]] == sent1
        assert text[span2[0] : span2[1]] == sent2

    def test_tokenize_with_spans_with_abbreviations(self, tokenizer):
        """Test handling of abbreviations."""
        text = "Dr. Smith arrived at 3 p.m. today."
        result = tokenizer.tokenize_with_spans(text)

        # Should be one sentence due to abbreviations
        assert len(result) == 1
        sent, span = result[0]
        assert sent == text
        assert span == (0, len(text))

    def test_tokenize_with_spans_realign_boundaries(self, tokenizer):
        """Test boundary realignment parameter."""
        text = "This is a test. (With parentheses.)"

        # With realignment (default)
        result_aligned = tokenizer.tokenize_with_spans(text, realign_boundaries=True)
        assert len(result_aligned) == 2
        assert result_aligned[0][0] == "This is a test. "
        assert result_aligned[1][0] == "(With parentheses.)"

        # Without realignment might produce different results
        result_no_align = tokenizer.tokenize_with_spans(text, realign_boundaries=False)
        # The exact behavior depends on the model, but we should get valid sentences
        assert len(result_no_align) >= 2

    def test_tokenize_with_spans_multiline(self, tokenizer):
        """Test handling of multiline text."""
        text = """First sentence.
Second sentence on new line.
Third sentence."""
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 3
        # Verify sentences include newlines
        sentences = [sent for sent, _ in result]
        assert sentences[0] == "First sentence.\n"
        assert sentences[1] == "Second sentence on new line.\n"
        assert sentences[2] == "Third sentence."

        # Verify spans correctly extract from original text
        for sent, span in result:
            assert text[span[0] : span[1]] == sent

        # Verify contiguity
        spans = [span for _, span in result]
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0]


class TestConvenienceFunctions:
    """Test cases for convenience functions in __init__.py."""

    def test_sent_spans_basic(self):
        """Test sent_spans convenience function."""
        text = "First sentence. Second sentence."
        spans = nupunkt.sent_spans(text)

        assert len(spans) == 2
        assert spans[0] == (0, 16)
        assert spans[1] == (16, 32)
        assert text[spans[0][0] : spans[0][1]] == "First sentence. "
        assert text[spans[1][0] : spans[1][1]] == "Second sentence."

    def test_sent_spans_empty(self):
        """Test sent_spans with empty string."""
        spans = nupunkt.sent_spans("")
        assert spans == []

    def test_sent_spans_with_text_basic(self):
        """Test sent_spans_with_text convenience function."""
        text = "First sentence. Second sentence."
        result = nupunkt.sent_spans_with_text(text)

        assert len(result) == 2

        sent1, span1 = result[0]
        assert sent1 == "First sentence. "
        assert span1 == (0, 16)

        sent2, span2 = result[1]
        assert sent2 == "Second sentence."
        assert span2 == (16, 32)

    def test_sent_spans_with_text_empty(self):
        """Test sent_spans_with_text with empty string."""
        result = nupunkt.sent_spans_with_text("")
        assert result == []

    def test_sent_spans_consistency(self):
        """Test consistency between different span functions."""
        text = "Test sentence one. Test sentence two. Test three."

        # Get results from all methods
        spans = nupunkt.sent_spans(text)
        spans_with_text = nupunkt.sent_spans_with_text(text)
        sentences = nupunkt.sent_tokenize(text)

        # Verify consistency
        assert len(spans) == len(spans_with_text) == len(sentences)

        for i, (sent, span) in enumerate(spans_with_text):
            assert span == spans[i]
            # sent_tokenize returns sentences without trailing whitespace,
            # but tokenize_with_spans includes it for contiguity
            assert sent.rstrip() == sentences[i]
            assert text[span[0] : span[1]] == sent

    def test_sent_spans_complex_document(self):
        """Test span functions on a complex document."""
        text = """Dr. Smith arrived at 9 a.m. today. He was late.

"Hello," he said. "How are you?"

The meeting ended at 5 p.m. Everyone left."""

        spans = nupunkt.sent_spans(text)
        spans_with_text = nupunkt.sent_spans_with_text(text)

        # Verify all spans are valid
        for sent, span in spans_with_text:
            assert text[span[0] : span[1]] == sent

        # Verify coverage
        reconstructed = "".join(sent for sent, _ in spans_with_text)
        assert reconstructed == text

        # Verify contiguity
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0]

    def test_sent_spans_with_ellipsis(self):
        """Test handling of ellipsis."""
        text = "First sentence... Second sentence."
        spans = nupunkt.sent_spans(text)
        spans_with_text = nupunkt.sent_spans_with_text(text)

        assert len(spans) == 2
        sent1, span1 = spans_with_text[0]
        assert sent1 == "First sentence... "
        assert text[span1[0] : span1[1]] == sent1


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_text(self, tokenizer):
        """Test handling of Unicode text."""
        text = "Première phrase. Deuxième phrase. 第三句。"
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 3
        for sent, span in result:
            assert text[span[0] : span[1]] == sent

    def test_very_long_sentence(self, tokenizer):
        """Test handling of very long sentences."""
        # Create a long sentence without periods
        words = ["word"] * 100
        text = " ".join(words)
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 1
        sent, span = result[0]
        assert sent == text
        assert span == (0, len(text))

    def test_mixed_sentence_endings(self, tokenizer):
        """Test various sentence ending punctuation."""
        text = "Question? Exclamation! Statement."
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 3
        sentences = [sent for sent, _ in result]
        assert sentences[0] == "Question? "
        assert sentences[1] == "Exclamation! "
        assert sentences[2] == "Statement."

    def test_spans_with_quotes_and_parentheses(self, tokenizer):
        """Test complex nesting of quotes and parentheses."""
        text = 'She said, "Look (over there)." Then she left.'
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 2
        sent1, _ = result[0]
        sent2, _ = result[1]
        assert sent1 == 'She said, "Look (over there)." '
        assert sent2 == "Then she left."
