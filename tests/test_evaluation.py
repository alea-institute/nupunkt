"""Tests for the evaluation module."""

import tempfile
from pathlib import Path

from nupunkt import load
from nupunkt.evaluation.dataset import (
    TestCase,
    create_test_cases,
    load_evaluation_data,
    parse_annotated_text,
    save_evaluation_dataset,
)
from nupunkt.evaluation.evaluator import evaluate_single_example
from nupunkt.evaluation.metrics import (
    boundary_accuracy,
    calculate_metrics,
    get_sentence_boundaries,
    precision_recall_f1,
)


class TestMetrics:
    """Test evaluation metrics calculation."""

    def test_precision_recall_f1(self):
        """Test basic metric calculations."""
        # Perfect case
        p, r, f1 = precision_recall_f1(10, 0, 0)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

        # No true positives
        p, r, f1 = precision_recall_f1(0, 10, 10)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

        # Mixed case
        p, r, f1 = precision_recall_f1(6, 2, 2)
        assert p == 0.75  # 6/(6+2)
        assert r == 0.75  # 6/(6+2)
        assert f1 == 0.75

    def test_get_sentence_boundaries(self):
        """Test boundary extraction."""
        text = "Hello world. How are you? I'm fine."
        sentences = ["Hello world.", "How are you?", "I'm fine."]

        boundaries = get_sentence_boundaries(sentences, text)

        # Should find boundaries at end of each sentence
        assert 12 in boundaries  # After "Hello world."
        assert 25 in boundaries  # After "How are you?"
        assert 35 in boundaries  # After "I'm fine."
        assert len(boundaries) == 3

    def test_boundary_accuracy(self):
        """Test boundary accuracy calculation."""
        pred = {10, 20, 30}
        true = {10, 21, 30}

        # Exact matching
        tp, fp, fn = boundary_accuracy(pred, true, tolerance=0)
        assert tp == 2  # 10 and 30 match
        assert fp == 1  # 20 doesn't match
        assert fn == 1  # 21 not predicted

        # With tolerance
        tp, fp, fn = boundary_accuracy(pred, true, tolerance=1)
        assert tp == 3  # All match with tolerance
        assert fp == 0
        assert fn == 0

    def test_calculate_metrics(self):
        """Test full metrics calculation."""
        pred_sentences = ["Hello world.", "How are you?", "I'm fine."]
        true_sentences = ["Hello world.", "How are you?", "I'm fine."]
        text = "Hello world. How are you? I'm fine."

        metrics = calculate_metrics(pred_sentences, true_sentences, text, 0.1)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.exact_match_accuracy == 1.0
        assert metrics.over_segmentation_rate == 0.0
        assert metrics.under_segmentation_rate == 0.0


class TestDataset:
    """Test dataset handling."""

    def test_parse_annotated_text(self):
        """Test parsing text with annotations."""
        text = "Hello world.<|sentence|>How are you?<|sentence|>I'm fine."
        sentences = parse_annotated_text(text)

        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I'm fine."

    def test_create_test_cases(self):
        """Test creating test cases."""
        texts = ["Hello world. How are you?", "I'm fine. Thanks."]
        sentence_lists = [["Hello world.", "How are you?"], ["I'm fine.", "Thanks."]]

        test_cases = create_test_cases(texts, sentence_lists)

        assert len(test_cases) == 2
        assert test_cases[0].text == texts[0]
        assert test_cases[0].sentences == sentence_lists[0]
        assert test_cases[1].text == texts[1]
        assert test_cases[1].sentences == sentence_lists[1]

    def test_save_and_load_dataset(self):
        """Test saving and loading evaluation datasets."""
        test_cases = [
            TestCase(
                text="Hello world. How are you?",
                sentences=["Hello world.", "How are you?"],
                metadata={"source": "test"},
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            save_evaluation_dataset(test_cases, temp_path)

            # Load
            loaded = load_evaluation_data(temp_path)

            assert len(loaded) == 1
            # Text might have normalized whitespace after reconstruction
            assert loaded[0].text.replace("  ", " ") == test_cases[0].text.replace("  ", " ")
            assert loaded[0].sentences == test_cases[0].sentences
            assert loaded[0].metadata is not None
            assert loaded[0].metadata["source"] == "test"
        finally:
            temp_path.unlink()


class TestEvaluator:
    """Test evaluation functionality."""

    def test_evaluate_single_example(self):
        """Test evaluating a single example."""
        tokenizer = load("default")
        test_case = TestCase(
            text="Hello world. How are you?", sentences=["Hello world.", "How are you?"]
        )

        predicted, time_taken, error = evaluate_single_example(tokenizer, test_case)

        assert len(predicted) == 2
        assert predicted[0].strip() == "Hello world."
        assert predicted[1].strip() == "How are you?"
        assert time_taken > 0
        assert error is None

    def test_evaluate_with_error(self):
        """Test evaluation with count mismatch."""
        tokenizer = load("default")
        test_case = TestCase(
            text="Hello world how are you",  # No punctuation
            sentences=["Hello world.", "How are you?"],  # Expects 2 sentences
        )

        predicted, time_taken, error = evaluate_single_example(tokenizer, test_case)

        assert len(predicted) == 1  # Only one sentence predicted
        assert error is not None
        assert error["type"] == "count_mismatch"
        assert error["predicted_count"] == 1
        assert error["true_count"] == 2
