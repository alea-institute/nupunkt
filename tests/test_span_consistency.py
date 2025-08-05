"""
Tests to verify consistency between sentence and paragraph span behavior.
"""

import pytest

import nupunkt


def test_sentence_spans_are_contiguous():
    """Test that sentence spans are contiguous like paragraph spans."""
    text = "First sentence. Second sentence. Third sentence."

    # Get sentence spans
    sent_spans = nupunkt.sent_spans(text)

    # Verify contiguity
    for i in range(len(sent_spans) - 1):
        assert sent_spans[i][1] == sent_spans[i + 1][0], "Sentence spans should be contiguous"

    # Verify full coverage
    assert sent_spans[0][0] == 0, "First span should start at 0"
    assert sent_spans[-1][1] == len(text), "Last span should end at text length"


def test_paragraph_spans_are_contiguous():
    """Test that paragraph spans are contiguous."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    # Get paragraph spans
    para_spans = nupunkt.para_spans(text)

    # Verify contiguity
    for i in range(len(para_spans) - 1):
        assert para_spans[i][1] == para_spans[i + 1][0], "Paragraph spans should be contiguous"

    # Verify full coverage
    assert para_spans[0][0] == 0, "First span should start at 0"
    assert para_spans[-1][1] == len(text), "Last span should end at text length"


def test_sentence_and_paragraph_consistency():
    """Test that sentence and paragraph spans follow the same contiguity pattern."""
    text = """First sentence. Second sentence.

Third sentence in new paragraph. Fourth sentence.

Fifth sentence in third paragraph."""

    # Get both types of spans
    sent_spans = nupunkt.sent_spans(text)
    para_spans = nupunkt.para_spans(text)

    # Both should cover the entire text
    assert sent_spans[0][0] == para_spans[0][0] == 0
    assert sent_spans[-1][1] == para_spans[-1][1] == len(text)

    # Both should be contiguous
    for spans, span_type in [(sent_spans, "sentence"), (para_spans, "paragraph")]:
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0], f"{span_type} spans should be contiguous"


def test_spans_with_text_consistency():
    """Test that spans_with_text functions also maintain contiguity."""
    text = "First. Second.\n\nThird."

    # Get spans with text
    sent_spans_with_text = nupunkt.sent_spans_with_text(text)
    para_spans_with_text = nupunkt.para_spans_with_text(text)

    # Reconstruct text from both
    sent_reconstructed = "".join(sent for sent, _ in sent_spans_with_text)
    para_reconstructed = "".join(para for para, _ in para_spans_with_text)

    # Both should perfectly reconstruct the original text
    assert sent_reconstructed == text, "Sentence spans should reconstruct original text"
    assert para_reconstructed == text, "Paragraph spans should reconstruct original text"


def test_whitespace_handling_consistency():
    """Test that both tokenizers handle whitespace consistently."""
    text = "Sentence one.   \n\n   Sentence two."

    sent_spans_with_text = nupunkt.sent_spans_with_text(text)

    # Verify whitespace is preserved
    for sent, span in sent_spans_with_text:
        assert text[span[0] : span[1]] == sent

    # Verify contiguity with whitespace
    reconstructed = "".join(sent for sent, _ in sent_spans_with_text)
    assert reconstructed == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
