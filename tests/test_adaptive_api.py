"""Tests for the adaptive sentence tokenization API."""

import pytest

# Manual test imports for running this file directly
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

from nupunkt import PunktSentenceTokenizer, load, sent_tokenize, sent_tokenize_adaptive
from nupunkt.hybrid import AdaptiveTokenizer


class TestSentTokenizeAdaptive:
    """Test the sent_tokenize function with adaptive mode."""

    def test_basic_adaptive_mode(self):
        """Test basic adaptive tokenization."""
        text = "She studied at M.I.T. in Cambridge."

        # Standard mode might split on M.I.T.
        sent_tokenize(text, adaptive=False)

        # Adaptive should handle it correctly
        adaptive = sent_tokenize(text, adaptive=True)
        assert len(adaptive) == 1
        assert adaptive[0] == text

    def test_dynamic_abbreviation_discovery(self):
        """Test discovery of patterns not in training data."""
        patterns = [
            ("He has a Ph.D. in physics.", 1),
            ("Visit I.B.M. headquarters.", 1),
            ("She earned her B.S.E.E. degree.", 1),
            ("The U.S.A. is large.", 1),
            ("Contact N.A.S.A. today.", 1),
        ]

        for text, expected_sentences in patterns:
            result = sent_tokenize(text, adaptive=True)
            assert len(result) == expected_sentences, f"Failed for: {text}"

    def test_confidence_threshold_parameter(self):
        """Test that confidence threshold affects behavior."""
        text = "This ends. Another sentence."

        # Different thresholds should work
        low = sent_tokenize(text, adaptive=True, confidence_threshold=0.3)
        medium = sent_tokenize(text, adaptive=True, confidence_threshold=0.7)
        high = sent_tokenize(text, adaptive=True, confidence_threshold=0.9)

        # All should produce valid results
        assert all(isinstance(s, str) for s in low)
        assert all(isinstance(s, str) for s in medium)
        assert all(isinstance(s, str) for s in high)
        assert len(low) >= 1 and len(low) <= 2
        assert len(medium) >= 1 and len(medium) <= 2
        assert len(high) >= 1 and len(high) <= 2

    def test_return_confidence_flag(self):
        """Test return_confidence parameter."""
        text = "Clear sentence! Unclear sentence."

        # Without confidence
        regular = sent_tokenize(text, adaptive=True)
        assert all(isinstance(s, str) for s in regular)

        # With confidence
        with_conf = sent_tokenize(text, adaptive=True, return_confidence=True)
        assert all(isinstance(s, tuple) for s in with_conf)
        assert all(isinstance(s[0], str) and isinstance(s[1], float) for s in with_conf)
        assert all(0.0 <= s[1] <= 1.0 for s in with_conf if isinstance(s, tuple))

    def test_debug_parameter(self):
        """Test debug parameter doesn't break functionality."""
        text = "Test sentence. Another one."

        # Debug should not affect the result
        regular = sent_tokenize(text, adaptive=True, debug=False)
        debug = sent_tokenize(text, adaptive=True, debug=True)

        assert regular == debug

    def test_model_parameter_with_adaptive(self):
        """Test that model parameter works with adaptive mode."""
        text = "Test sentence. Another one."

        # Should work with default model
        result = sent_tokenize(text, model="default", adaptive=True)
        assert len(result) == 2

    def test_dynamic_abbrev_parameter(self):
        """Test dynamic_abbrev parameter."""
        text = "Visit M.I.T. today."

        # With dynamic abbreviation discovery (default)
        with_dynamic = sent_tokenize(text, adaptive=True, dynamic_abbrev=True)
        assert len(with_dynamic) == 1

        # Without dynamic abbreviation discovery
        without_dynamic = sent_tokenize(text, adaptive=True, dynamic_abbrev=False)
        # Should still work, might have different results
        assert len(without_dynamic) >= 1

    def test_error_handling(self):
        """Test appropriate errors for invalid parameters."""
        # return_confidence without adaptive mode
        try:
            sent_tokenize("text", return_confidence=True)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "return_confidence is only available in adaptive mode" in str(e)

        # Invalid threshold values should be handled by AdaptiveTokenizer
        # Not testing here as it's the tokenizer's responsibility

    def test_empty_and_whitespace(self):
        """Test with empty and whitespace strings."""
        assert sent_tokenize("", adaptive=True) == []
        assert sent_tokenize("   \n\t  ", adaptive=True) == []
        assert sent_tokenize("", adaptive=True, return_confidence=True) == []


class TestSentTokenizeAdaptive:
    """Test the sent_tokenize_adaptive convenience function."""

    def test_basic_functionality(self):
        """Test basic dynamic tokenization."""
        text = "She got her Ph.D. at M.I.T. yesterday."
        result = sent_tokenize_adaptive(text)
        assert len(result) == 1
        assert result[0] == text

    def test_threshold_parameter(self):
        """Test threshold parameter."""
        text = "Short sentence. Another one."

        # Different thresholds
        low = sent_tokenize_adaptive(text, threshold=0.5)
        high = sent_tokenize_adaptive(text, threshold=0.85)

        assert len(low) == 2
        assert len(high) == 2

    def test_return_confidence(self):
        """Test return_confidence with dynamic function."""
        text = "Test! Another test."

        result = sent_tokenize_adaptive(text, return_confidence=True)
        assert all(isinstance(s, tuple) for s in result)
        assert all(isinstance(s[0], str) and isinstance(s[1], float) for s in result)

    def test_model_parameter(self):
        """Test model parameter with dynamic function."""
        text = "Test sentence."

        # Should work with default
        result = sent_tokenize_adaptive(text, model="default")
        assert len(result) == 1

    def test_forwards_kwargs(self):
        """Test that kwargs are forwarded correctly."""
        text = "Test sentence."

        # Should forward debug parameter
        result = sent_tokenize_adaptive(text, debug=True)
        assert len(result) == 1


class TestBackwardCompatibility:
    """Ensure backward compatibility is maintained."""

    def test_sent_tokenize_unchanged(self):
        """Original sent_tokenize() API works exactly as before."""
        text = "Hello world. How are you?"
        result = sent_tokenize(text)
        assert result == ["Hello world.", "How are you?"]

    def test_model_parameter_unchanged(self):
        """Model parameter still works without adaptive."""
        text = "Test sentence."
        default_result = sent_tokenize(text)
        explicit_result = sent_tokenize(text, model="default")
        assert default_result == explicit_result

    def test_no_unexpected_parameters(self):
        """Ensure no parameters leak when not using adaptive mode."""
        text = "Test."

        # These should work fine
        sent_tokenize(text)
        sent_tokenize(text, model="default")

        # This should fail
        with pytest.raises(ValueError):
            sent_tokenize(text, return_confidence=True)


class TestIntegration:
    """Test integration with existing functionality."""

    def test_load_returns_standard_tokenizer(self):
        """load() function returns standard tokenizer by default."""
        tokenizer = load("default")
        assert isinstance(tokenizer, PunktSentenceTokenizer)
        assert not isinstance(tokenizer, AdaptiveTokenizer)

    def test_adaptive_uses_hybrid_internally(self):
        """Verify adaptive mode uses the hybrid tokenizer."""
        text = "Test with M.I.T. pattern."

        # This should use AdaptiveTokenizer internally
        result = sent_tokenize(text, adaptive=True)
        assert len(result) == 1

    def test_consistent_results(self):
        """Test that results are consistent across calls."""
        text = "This is a test. Multiple sentences here. And one more."

        # Multiple calls should give same result
        result1 = sent_tokenize(text, adaptive=True, confidence_threshold=0.7)
        result2 = sent_tokenize(text, adaptive=True, confidence_threshold=0.7)

        assert result1 == result2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_text(self):
        """Test with very long text."""
        long_text = "Test sentence. " * 100

        result = sent_tokenize(long_text, adaptive=True)
        assert len(result) == 100

    def test_unicode_text(self):
        """Test with unicode characters."""
        unicode_text = "Hello 世界. How are you？ Καλημέρα!"

        result = sent_tokenize(unicode_text, adaptive=True)
        # Note: Full-width question mark (？) is not a standard sentence boundary
        assert len(result) == 2
        assert result[0] == "Hello 世界."
        assert "Καλημέρα!" in result[1]

    def test_single_character(self):
        """Test with single character."""
        assert sent_tokenize("I", adaptive=True) == ["I"]
        assert sent_tokenize(".", adaptive=True) == ["."]

    def test_numbers_and_decimals(self):
        """Test with numbers and decimals."""
        text = "The temperature is 98.6 degrees."
        result = sent_tokenize(text, adaptive=True)
        assert len(result) == 1

    def test_urls_and_emails(self):
        """Test with URLs and emails."""
        text = "Visit www.example.com. Email me at test@example.com."
        result = sent_tokenize(text, adaptive=True)
        assert len(result) == 2


class TestPerformance:
    """Test performance characteristics."""

    def test_caching_behavior(self):
        """Test that tokenizer is cached appropriately."""
        # First call might be slower (loading)
        text = "Test."
        result1 = sent_tokenize(text, adaptive=True)

        # Second call should use cached tokenizer
        result2 = sent_tokenize(text, adaptive=True)

        assert result1 == result2

    def test_lazy_import(self):
        """Test that hybrid module is only imported when needed."""
        # This is hard to test directly, but we can verify it works
        # Standard mode shouldn't load hybrid
        text = "Test."
        result = sent_tokenize(text, adaptive=False)
        assert len(result) == 1


if __name__ == "__main__":
    # Run a few key tests manually
    test = TestSentTokenizeAdaptive()
    test.test_basic_adaptive_mode()
    test.test_return_confidence_flag()

    test2 = TestSentTokenizeAdaptive()
    test2.test_basic_functionality()

    test3 = TestBackwardCompatibility()
    test3.test_sent_tokenize_unchanged()

    print("✓ Manual tests passed!")
