"""
Tests for adaptive sentence span functions.
"""

from typing import List, Tuple

import pytest

import nupunkt


class TestAdaptiveSpans:
    """Test the adaptive span functions."""

    def test_sent_spans_adaptive_basic(self):
        """Test basic adaptive span functionality."""
        text = "Dr. Smith studied at M.I.T. in Cambridge. She graduated in 2020."

        # Get adaptive spans
        spans = nupunkt.sent_spans_adaptive(text)

        # Should recognize M.I.T. as abbreviation and not split
        assert len(spans) == 2
        assert spans[0] == (0, 42)  # "Dr. Smith studied at M.I.T. in Cambridge. "
        assert spans[1] == (42, 64)  # "She graduated in 2020."

        # Verify contiguity
        assert spans[0][1] == spans[1][0]
        assert spans[0][0] == 0
        assert spans[1][1] == len(text)

    def test_sent_spans_with_text_adaptive_basic(self):
        """Test adaptive spans with text."""
        text = "Dr. Smith studied at M.I.T. in Cambridge. She graduated in 2020."

        # Get spans with text (without confidence)
        results: List[Tuple[str, Tuple[int, int]]] = nupunkt.sent_spans_with_text_adaptive(text)

        assert len(results) == 2

        # Check first sentence
        sent1, span1 = results[0]
        assert sent1 == "Dr. Smith studied at M.I.T. in Cambridge. "
        assert span1 == (0, 42)
        assert text[span1[0] : span1[1]] == sent1

        # Check second sentence
        sent2, span2 = results[1]
        assert sent2 == "She graduated in 2020."
        assert span2 == (42, 64)
        assert text[span2[0] : span2[1]] == sent2

    def test_sent_spans_adaptive_unknown_abbreviations(self):
        """Test handling of unknown abbreviations."""
        # Test with various unknown abbreviations
        test_cases = [
            ("She works at I.B.M. headquarters.", 1),  # Should stay as one
            ("Visit X.Y.Z. corporation today.", 1),  # Unknown pattern
            ("See pg. 5 for details.", 2),  # Unknown lowercase - should split
        ]

        for text, expected_count in test_cases:
            spans = nupunkt.sent_spans_adaptive(text)
            assert len(spans) == expected_count, f"Failed for: {text}"

    def test_sent_spans_with_confidence(self):
        """Test spans with confidence scores."""
        text = "Dr. Smith arrived. She studied at M.I.T. yesterday."

        # Get spans with confidence
        results = nupunkt.sent_spans_with_text_adaptive(text, return_confidence=True)

        assert len(results) == 2

        # Each result should have 3 elements: sentence, span, confidence
        for result in results:
            assert len(result) == 3
            sentence, span, confidence = result
            assert isinstance(sentence, str)
            assert isinstance(span, tuple) and len(span) == 2
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

            # Verify span matches text
            assert text[span[0] : span[1]] == sentence

    def test_threshold_parameter(self):
        """Test that threshold parameter affects results."""
        text = "End with etc. The next sentence."

        # High threshold - more conservative
        spans_high = nupunkt.sent_spans_adaptive(text, threshold=0.9)

        # Low threshold - more aggressive
        spans_low = nupunkt.sent_spans_adaptive(text, threshold=0.3)

        # Both should produce valid results
        assert len(spans_high) >= 1
        assert len(spans_low) >= 1

        # Verify spans are contiguous
        for spans in [spans_high, spans_low]:
            for i in range(len(spans) - 1):
                assert spans[i][1] == spans[i + 1][0]

    def test_model_parameter(self):
        """Test using different models."""
        text = "First sentence. Second sentence."

        # Should work with default model
        spans = nupunkt.sent_spans_adaptive(text, model="default")
        assert len(spans) == 2

        # Test with invalid model should raise error
        with pytest.raises(FileNotFoundError):
            nupunkt.sent_spans_adaptive(text, model="nonexistent_model")

    def test_dynamic_abbrev_parameter(self):
        """Test disabling dynamic abbreviation detection."""
        text = "She studied at M.I.T. yesterday."

        # With dynamic abbreviations (default)
        spans_dynamic = nupunkt.sent_spans_adaptive(text, dynamic_abbrev=True)

        # Without dynamic abbreviations
        spans_static = nupunkt.sent_spans_adaptive(text, dynamic_abbrev=False)

        # Both should produce valid results
        assert len(spans_dynamic) >= 1
        assert len(spans_static) >= 1

    def test_consistency_with_tokenize(self):
        """Test that spans match the tokenization results."""
        text = "First sentence. Second one! Third?"

        # Get sentences
        sentences = nupunkt.sent_tokenize_adaptive(text)

        # Get spans with text
        spans_with_text = nupunkt.sent_spans_with_text_adaptive(text)

        # Should have same number of results
        assert len(sentences) == len(spans_with_text)

        # Sentences should match (modulo whitespace)
        for sent, (span_sent, _) in zip(sentences, spans_with_text):
            assert sent.strip() == span_sent.strip()

    def test_empty_text(self):
        """Test handling of empty text."""
        assert nupunkt.sent_spans_adaptive("") == []
        assert nupunkt.sent_spans_with_text_adaptive("") == []
        assert nupunkt.sent_spans_with_text_adaptive("", return_confidence=True) == []

    def test_whitespace_only(self):
        """Test handling of whitespace-only text."""
        text = "   \n\n   "
        spans = nupunkt.sent_spans_adaptive(text)
        assert len(spans) == 1
        assert spans[0] == (0, len(text))


class TestAdaptiveSpanEdgeCases:
    """Test edge cases for adaptive spans."""

    def test_complex_abbreviations(self):
        """Test complex abbreviation patterns."""
        text = "The C.E.O.'s decision regarding Ph.D. candidates was final."

        spans_with_text: List[Tuple[str, Tuple[int, int]]] = nupunkt.sent_spans_with_text_adaptive(text)

        # Should recognize as single sentence
        assert len(spans_with_text) == 1
        sent, span = spans_with_text[0]
        assert span == (0, len(text))

    def test_mixed_punctuation(self):
        """Test mixed punctuation scenarios."""
        text = 'He asked "Really?" She nodded. "Yes!" she exclaimed.'

        spans = nupunkt.sent_spans_adaptive(text)

        # Should handle quotes properly
        assert len(spans) >= 2

        # Verify full coverage
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)

    def test_ellipsis_handling(self):
        """Test ellipsis handling."""
        text = "She paused... Then continued. But..."

        spans_with_text = nupunkt.sent_spans_with_text_adaptive(text)

        # Should produce valid spans
        assert len(spans_with_text) >= 1

        # Reconstruct text
        reconstructed = "".join(sent for sent, _ in spans_with_text)
        assert reconstructed == text


if __name__ == "__main__":
    # Run basic tests
    test = TestAdaptiveSpans()
    test.test_sent_spans_adaptive_basic()
    test.test_sent_spans_with_text_adaptive_basic()
    test.test_sent_spans_with_confidence()
    print("✓ Basic adaptive span tests passed!")
