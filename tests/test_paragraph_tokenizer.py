"""
Tests for the paragraph tokenizer module.
"""

import tempfile
from pathlib import Path

import pytest

from nupunkt.tokenizers.paragraph_tokenizer import PunktParagraphTokenizer
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer


@pytest.fixture
def tokenizer():
    """Create a paragraph tokenizer instance."""
    return PunktParagraphTokenizer()


class TestParagraphTokenizer:
    """Test cases for PunktParagraphTokenizer."""

    def test_basic_paragraph_split(self, tokenizer):
        """Test basic paragraph splitting with double newlines."""
        text = "First paragraph.\n\nSecond paragraph."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 2
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "\n\nSecond paragraph."

    def test_single_newline_no_split(self, tokenizer):
        """Test that single newlines don't split paragraphs."""
        text = "First line.\nSecond line."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    def test_multiple_newlines(self, tokenizer):
        """Test handling of multiple consecutive newlines."""
        text = "First paragraph.\n\n\nSecond paragraph.\n\n\n\nThird paragraph."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 3
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "\n\n\nSecond paragraph."
        assert paragraphs[2] == "\n\n\n\nThird paragraph."

    def test_mixed_newline_types(self, tokenizer):
        """Test handling of different newline characters."""
        # Windows-style newlines
        text = "First paragraph.\r\n\r\nSecond paragraph."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 2
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "\r\n\r\nSecond paragraph."

    def test_leading_newlines(self, tokenizer):
        """Test text starting with newlines."""
        text = "\n\nStarting with newlines."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    def test_trailing_newlines(self, tokenizer):
        """Test text ending with newlines."""
        text = "Ending with newlines.\n\n"
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 2
        assert paragraphs[0] == "Ending with newlines."
        assert paragraphs[1] == "\n\n"

    def test_empty_string(self, tokenizer):
        """Test empty string input."""
        text = ""
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 0

    def test_only_newlines(self, tokenizer):
        """Test string containing only newlines."""
        text = "\n\n\n"
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    def test_no_newlines(self, tokenizer):
        """Test text with no newlines."""
        text = "Single paragraph with no newlines."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    def test_sentence_boundary_interaction(self, tokenizer):
        """Test that paragraphs split only at sentence boundaries."""
        # Multiple sentences, then paragraph break
        text = "First sentence. Second sentence.\n\nNew paragraph."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 2
        assert paragraphs[0] == "First sentence. Second sentence."
        assert paragraphs[1] == "\n\nNew paragraph."

        # Newline mid-sentence should not split
        text = "This is a sentence that\n\ncontinues here."
        paragraphs = tokenizer.tokenize(text)

        # Should be one paragraph since the break is mid-sentence
        assert len(paragraphs) == 1

    def test_span_tokenize_basic(self, tokenizer):
        """Test span_tokenize returns correct character offsets."""
        text = "First paragraph.\n\nSecond paragraph."
        spans = tokenizer.span_tokenize(text)

        assert len(spans) == 2
        assert spans[0] == (0, 16)
        assert spans[1] == (16, 35)

        # Verify spans reconstruct the text
        assert text[spans[0][0] : spans[0][1]] == "First paragraph."
        assert text[spans[1][0] : spans[1][1]] == "\n\nSecond paragraph."

    def test_span_tokenize_coverage(self, tokenizer):
        """Test that spans cover the entire text without gaps."""
        text = "Para 1.\n\nPara 2.\n\n\nPara 3."
        spans = tokenizer.span_tokenize(text)

        # Check no gaps between spans
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0]

        # Check full coverage
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)

    def test_tokenize_with_spans(self, tokenizer):
        """Test tokenize_with_spans returns both text and spans."""
        text = "First paragraph.\n\nSecond paragraph."
        result = tokenizer.tokenize_with_spans(text)

        assert len(result) == 2

        para1, span1 = result[0]
        assert para1 == "First paragraph."
        assert span1 == (0, 16)

        para2, span2 = result[1]
        assert para2 == "\n\nSecond paragraph."
        assert span2 == (16, 35)

    def test_complex_document(self, tokenizer):
        """Test a more complex document with various paragraph styles."""
        text = """Introduction paragraph with multiple sentences. This is the second sentence.

Body paragraph 1. It has several sentences too. Here's another one.

Body paragraph 2.
With a line break inside.

Conclusion paragraph."""

        paragraphs = tokenizer.tokenize(text)
        assert len(paragraphs) == 4

        # Check that line break within paragraph doesn't split it
        assert "Body paragraph 2.\nWith a line break inside." in paragraphs[2]

    def test_whitespace_after_sentence(self, tokenizer):
        """Test handling of whitespace after sentence before paragraph break."""
        text = "First paragraph.  \n\nSecond paragraph."
        paragraphs = tokenizer.tokenize(text)

        assert len(paragraphs) == 2
        # The tokenizer splits at sentence boundaries, so trailing spaces go to next paragraph
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "  \n\nSecond paragraph."

    def test_save_load_functionality(self, tokenizer):
        """Test saving and loading the tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_tokenizer"

            # Save the tokenizer (will add .json.xz extension)
            tokenizer.save(save_path, compress=True)

            # The actual saved file has .json.xz extension
            actual_path = Path(str(save_path) + ".json.xz")
            assert actual_path.exists()

            # Load it back
            loaded_tokenizer = PunktParagraphTokenizer.load(actual_path)

            # Test that it works the same
            text = "First paragraph.\n\nSecond paragraph."
            original_result = tokenizer.tokenize(text)
            loaded_result = loaded_tokenizer.tokenize(text)

            assert original_result == loaded_result

    def test_custom_sentence_tokenizer(self):
        """Test using a custom sentence tokenizer."""
        # Create a custom sentence tokenizer
        custom_sent_tokenizer = PunktSentenceTokenizer()

        # Create paragraph tokenizer with custom sentence tokenizer
        para_tokenizer = PunktParagraphTokenizer(sentence_tokenizer=custom_sent_tokenizer)

        text = "First paragraph.\n\nSecond paragraph."
        paragraphs = para_tokenizer.tokenize(text)

        assert len(paragraphs) == 2
