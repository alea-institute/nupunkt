"""
Tests for evaluation dataset handling.

These tests ensure we can correctly load, parse, and save evaluation datasets.
"""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from nupunkt.evaluation.dataset import (
    TestCase,
    create_test_cases,
    load_evaluation_data,
    load_jsonl_evaluation_data,
    parse_annotated_text,
    save_evaluation_dataset,
)


class TestAnnotationParsing:
    """Test parsing of annotated text."""

    def test_parse_simple_annotated_text(self):
        """Test parsing text with default delimiter."""
        text = "First sentence.<|sentence|>Second sentence.<|sentence|>Third."
        sentences = parse_annotated_text(text)

        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third."

    def test_parse_custom_delimiter(self):
        """Test parsing with custom delimiter."""
        text = "First.###BREAK###Second.###BREAK###Third."
        sentences = parse_annotated_text(text, delimiter="###BREAK###")

        assert len(sentences) == 3
        assert sentences[0] == "First."
        assert sentences[1] == "Second."
        assert sentences[2] == "Third."

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace correctly."""
        text = "  First sentence.  <|sentence|>  Second sentence.  <|sentence|>  Third.  "
        sentences = parse_annotated_text(text)

        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third."

    def test_parse_empty_segments(self):
        """Test parsing skips empty segments."""
        text = "<|sentence|>First.<|sentence|><|sentence|>Second.<|sentence|>"
        sentences = parse_annotated_text(text)

        assert len(sentences) == 2  # Empty segments ignored
        assert sentences[0] == "First."
        assert sentences[1] == "Second."

    def test_parse_no_delimiter(self):
        """Test parsing text without delimiters."""
        text = "This is just plain text with no delimiters."
        sentences = parse_annotated_text(text)

        assert len(sentences) == 1
        assert sentences[0] == "This is just plain text with no delimiters."


class TestDatasetLoading:
    """Test loading evaluation datasets."""

    def test_load_jsonl_basic(self):
        """Test loading basic JSONL data."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Hello.<|sentence|>World."}\n')
            f.write('{"text": "Test.<|sentence|>Data."}\n')
            temp_path = Path(f.name)

        try:
            # Load data
            test_cases = list(load_jsonl_evaluation_data(temp_path))

            assert len(test_cases) == 2

            # First test case
            assert test_cases[0].text == "Hello.World."  # Delimiter removed
            assert test_cases[0].sentences == ["Hello.", "World."]

            # Second test case
            assert test_cases[1].text == "Test.Data."
            assert test_cases[1].sentences == ["Test.", "Data."]
        finally:
            temp_path.unlink()

    def test_load_jsonl_with_metadata(self):
        """Test loading JSONL with metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Test.<|sentence|>", "source": "unit_test", "id": 123}\n')
            temp_path = Path(f.name)

        try:
            test_cases = list(load_jsonl_evaluation_data(temp_path))

            assert len(test_cases) == 1
            assert test_cases[0].metadata is not None
            assert test_cases[0].metadata["source"] == "unit_test"
            assert test_cases[0].metadata["id"] == 123
        finally:
            temp_path.unlink()

    def test_load_jsonl_gzipped(self):
        """Test loading gzipped JSONL data."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as f:
            temp_path = Path(f.name)

        # Write gzipped data
        with gzip.open(temp_path, "wt", encoding="utf-8") as f:
            f.write('{"text": "Compressed.<|sentence|>Data."}\n')

        try:
            test_cases = list(load_jsonl_evaluation_data(temp_path))

            assert len(test_cases) == 1
            assert test_cases[0].text == "Compressed.Data."
            assert test_cases[0].sentences == ["Compressed.", "Data."]
        finally:
            temp_path.unlink()

    def test_load_jsonl_max_samples(self):
        """Test loading with max_samples limit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                f.write(f'{{"text": "Sentence {i}.<|sentence|>"}}\n')
            temp_path = Path(f.name)

        try:
            test_cases = list(load_jsonl_evaluation_data(temp_path, max_samples=3))

            assert len(test_cases) == 3  # Limited to 3
            assert test_cases[0].sentences == ["Sentence 0."]
            assert test_cases[2].sentences == ["Sentence 2."]
        finally:
            temp_path.unlink()

    def test_load_evaluation_data_auto_format(self):
        """Test auto-detection of file format."""
        # Test JSONL detection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Test.<|sentence|>"}\n')
            jsonl_path = Path(f.name)

        try:
            test_cases = load_evaluation_data(jsonl_path, format="auto")
            assert len(test_cases) == 1
            assert test_cases[0].sentences == ["Test."]
        finally:
            jsonl_path.unlink()

        # Test gzip detection
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            gz_path = Path(f.name)

        with gzip.open(gz_path, "wt") as f:
            f.write('{"text": "Gzipped.<|sentence|>"}\n')

        try:
            test_cases = load_evaluation_data(gz_path, format="auto")
            assert len(test_cases) == 1
            assert test_cases[0].sentences == ["Gzipped."]
        finally:
            gz_path.unlink()


class TestDatasetCreation:
    """Test creating and saving datasets."""

    def test_create_test_cases(self):
        """Test creating test cases from components."""
        texts = [
            "First document. With two sentences.",
            "Second document. Also two. Sentences here.",
        ]
        sentence_lists = [
            ["First document.", "With two sentences."],
            ["Second document.", "Also two.", "Sentences here."],
        ]
        metadata = [{"id": 1, "source": "test"}, {"id": 2, "source": "test"}]

        test_cases = create_test_cases(texts, sentence_lists, metadata)

        assert len(test_cases) == 2

        # Check first case
        assert test_cases[0].text == texts[0]
        assert test_cases[0].sentences == sentence_lists[0]
        assert test_cases[0].metadata is not None
        assert test_cases[0].metadata["id"] == 1

        # Check second case
        assert test_cases[1].text == texts[1]
        assert test_cases[1].sentences == sentence_lists[1]
        assert test_cases[1].metadata is not None
        assert test_cases[1].metadata["id"] == 2

    def test_create_test_cases_without_metadata(self):
        """Test creating test cases without metadata."""
        texts = ["Text one.", "Text two."]
        sentences = [["Text one."], ["Text two."]]

        test_cases = create_test_cases(texts, sentences)

        assert len(test_cases) == 2
        assert test_cases[0].metadata is None
        assert test_cases[1].metadata is None

    def test_create_test_cases_length_mismatch(self):
        """Test error on length mismatch."""
        texts = ["One", "Two", "Three"]
        sentences = [["One"], ["Two"]]  # Missing one

        with pytest.raises(ValueError, match="must match"):
            create_test_cases(texts, sentences)

    def test_save_and_load_dataset(self):
        """Test round-trip save and load."""
        # Create test cases
        original_cases = [
            TestCase(
                text="First test case. Two sentences.",
                sentences=["First test case.", "Two sentences."],
                metadata={"id": 1},
            ),
            TestCase(text="Second case here.", sentences=["Second case here."], metadata={"id": 2}),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            save_evaluation_dataset(original_cases, temp_path)

            # Load back
            loaded_cases = load_evaluation_data(temp_path)

            assert len(loaded_cases) == len(original_cases)

            # Verify content
            for orig, loaded in zip(original_cases, loaded_cases):
                # Text might have normalized whitespace after save/load cycle
                assert loaded.text.strip() == orig.text.strip()
                assert loaded.sentences == orig.sentences
                assert loaded.metadata["id"] == orig.metadata["id"]
        finally:
            temp_path.unlink()

    def test_save_dataset_gzipped(self):
        """Test saving dataset as gzipped file."""
        test_cases = [
            TestCase(text="Gzip test.", sentences=["Gzip test."]),
            TestCase(
                text="Multiple sentences. Here's another.",
                sentences=["Multiple sentences.", "Here's another."],
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save as gzip
            save_evaluation_dataset(test_cases, temp_path)

            # Verify it's gzipped and content is correct
            with gzip.open(temp_path, "rt") as f:
                # First test case - single sentence
                data1 = json.loads(f.readline())
                assert data1["text"] == "Gzip test."

                # Second test case - multiple sentences with delimiter
                data2 = json.loads(f.readline())
                assert " <|sentence|> " in data2["text"]
                assert data2["text"] == "Multiple sentences. <|sentence|> Here's another."
        finally:
            temp_path.unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test handling empty datasets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Empty file
            temp_path = Path(f.name)

        try:
            test_cases = load_evaluation_data(temp_path)
            assert len(test_cases) == 0
        finally:
            temp_path.unlink()

    def test_malformed_json(self):
        """Test handling malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Valid line.<|sentence|>"}\n')
            f.write("This is not JSON\n")  # Malformed
            f.write('{"text": "Another valid.<|sentence|>"}\n')
            temp_path = Path(f.name)

        try:
            # Should skip malformed lines with warning
            with pytest.warns(UserWarning, match="Skipping malformed JSON"):
                test_cases = list(load_jsonl_evaluation_data(temp_path))

            # Should load the two valid lines
            assert len(test_cases) == 2
            assert test_cases[0].sentences == ["Valid line."]
            assert test_cases[1].sentences == ["Another valid."]
        finally:
            temp_path.unlink()

    def test_missing_text_field(self):
        """Test handling missing 'text' field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"sentences": ["Test"], "no_text": true}\n')
            temp_path = Path(f.name)

        try:
            test_cases = list(load_jsonl_evaluation_data(temp_path))
            # Should skip entries without 'text' field
            assert len(test_cases) == 0
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    # Run some tests manually
    test_instance = TestAnnotationParsing()
    test_instance.test_parse_simple_annotated_text()
    test_instance.test_parse_with_whitespace()

    test_instance2 = TestDatasetCreation()
    test_instance2.test_create_test_cases()

    print("Manual dataset tests passed!")
