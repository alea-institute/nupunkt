"""
Comprehensive tests for evaluation metrics.

These tests ensure the evaluation system produces correct metrics
and handles edge cases properly.
"""

import pytest

from nupunkt.evaluation.metrics import (
    EvaluationMetrics,
    boundary_accuracy,
    calculate_metrics,
    get_sentence_boundaries,
    precision_recall_f1,
)


class TestMetricsCalculation:
    """Test core metrics calculation functions."""

    def test_precision_recall_f1_perfect(self):
        """Test perfect prediction case."""
        p, r, f1 = precision_recall_f1(tp=100, fp=0, fn=0)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_precision_recall_f1_no_predictions(self):
        """Test when no predictions are made."""
        p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=100)
        assert p == 0.0  # No predictions made
        assert r == 0.0  # Missed all positives
        assert f1 == 0.0

    def test_precision_recall_f1_no_true_positives(self):
        """Test when all predictions are wrong."""
        p, r, f1 = precision_recall_f1(tp=0, fp=50, fn=50)
        assert p == 0.0  # All predictions wrong
        assert r == 0.0  # Found none of the true positives
        assert f1 == 0.0

    def test_precision_recall_f1_typical(self):
        """Test typical case with mixed results."""
        # 80 correct, 20 wrong predictions, 20 missed
        p, r, f1 = precision_recall_f1(tp=80, fp=20, fn=20)
        assert p == pytest.approx(0.8)  # 80/(80+20)
        assert r == pytest.approx(0.8)  # 80/(80+20)
        assert f1 == pytest.approx(0.8)  # 2*0.8*0.8/(0.8+0.8)

    def test_get_sentence_boundaries_simple(self):
        """Test boundary extraction from simple sentences."""
        text = "Hello world. How are you? I am fine."
        sentences = ["Hello world.", "How are you?", "I am fine."]

        boundaries = get_sentence_boundaries(sentences, text)

        assert boundaries == {12, 25, 36}  # Positions after each sentence
        assert len(boundaries) == len(sentences)

    def test_get_sentence_boundaries_with_whitespace(self):
        """Test boundaries with extra whitespace."""
        text = "Hello world.  How are you?   I am fine."
        sentences = ["Hello world.", "How are you?", "I am fine."]

        boundaries = get_sentence_boundaries(sentences, text)

        # Should find correct positions despite extra spaces
        assert 12 in boundaries  # After "Hello world."
        assert len(boundaries) == 3

    def test_get_sentence_boundaries_edge_cases(self):
        """Test boundary extraction edge cases."""
        # Empty case
        assert get_sentence_boundaries([], "") == set()

        # Single sentence
        text = "Just one sentence."
        sentences = ["Just one sentence."]
        boundaries = get_sentence_boundaries(sentences, text)
        assert boundaries == {18}

    def test_boundary_accuracy_exact(self):
        """Test boundary accuracy with exact matches."""
        pred = {10, 20, 30, 40}
        true = {10, 20, 30, 40}

        tp, fp, fn = boundary_accuracy(pred, true, tolerance=0)
        assert tp == 4  # All match
        assert fp == 0  # No false positives
        assert fn == 0  # No false negatives

    def test_boundary_accuracy_with_tolerance(self):
        """Test boundary accuracy with tolerance."""
        pred = {10, 20, 30}
        true = {11, 19, 31}  # Off by 1

        # Without tolerance
        tp, fp, fn = boundary_accuracy(pred, true, tolerance=0)
        assert tp == 0  # None match exactly
        assert fp == 3  # All predictions are wrong
        assert fn == 3  # All true boundaries missed

        # With tolerance=1
        tp, fp, fn = boundary_accuracy(pred, true, tolerance=1)
        assert tp == 3  # All match within tolerance
        assert fp == 0
        assert fn == 0

    def test_boundary_accuracy_partial_match(self):
        """Test boundary accuracy with partial matches."""
        pred = {10, 20, 30, 99}  # 99 is wrong
        true = {10, 20, 30, 40}  # 40 is missed

        tp, fp, fn = boundary_accuracy(pred, true, tolerance=0)
        assert tp == 3  # 10, 20, 30 match
        assert fp == 1  # 99 is false positive
        assert fn == 1  # 40 is false negative


class TestFullMetricsCalculation:
    """Test the complete metrics calculation pipeline."""

    def test_calculate_metrics_perfect_match(self):
        """Test metrics for perfect prediction."""
        text = "First sentence. Second sentence. Third sentence."
        pred = ["First sentence.", "Second sentence.", "Third sentence."]
        true = ["First sentence.", "Second sentence.", "Third sentence."]

        metrics = calculate_metrics(pred, true, text, processing_time=1.0)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.exact_match_accuracy == 1.0
        assert metrics.over_segmentation_rate == 0.0
        assert metrics.under_segmentation_rate == 0.0
        assert metrics.total_sentences_pred == 3
        assert metrics.total_sentences_true == 3

    def test_calculate_metrics_under_segmentation(self):
        """Test metrics when model under-segments."""
        text = "First sentence. Second sentence. Third sentence."
        pred = ["First sentence. Second sentence.", "Third sentence."]  # Merged first two
        true = ["First sentence.", "Second sentence.", "Third sentence."]

        metrics = calculate_metrics(pred, true, text, processing_time=1.0)

        assert metrics.total_sentences_pred == 2
        assert metrics.total_sentences_true == 3
        assert metrics.under_segmentation_rate > 0  # Missing boundaries
        assert metrics.over_segmentation_rate == 0
        assert metrics.precision < 1.0
        assert metrics.recall < 1.0

    def test_calculate_metrics_over_segmentation(self):
        """Test metrics when model over-segments."""
        text = "First sentence. Second sentence."
        pred = ["First", "sentence.", "Second sentence."]  # Split first sentence
        true = ["First sentence.", "Second sentence."]

        metrics = calculate_metrics(pred, true, text, processing_time=1.0)

        assert metrics.total_sentences_pred == 3
        assert metrics.total_sentences_true == 2
        assert metrics.over_segmentation_rate > 0  # Extra boundaries
        assert metrics.under_segmentation_rate == 0

    def test_calculate_metrics_sentence_length_diff(self):
        """Test average sentence length difference calculation."""
        text = "Short. This is a much longer sentence with many words."
        pred = ["Short.", "This is a much longer sentence with many words."]
        true = ["Short. This is a much longer sentence with many words."]  # All one sentence

        metrics = calculate_metrics(pred, true, text, processing_time=1.0)

        # Average lengths should differ significantly
        assert metrics.avg_sentence_length_diff > 0

    def test_calculate_metrics_performance(self):
        """Test performance metrics calculation."""
        text = "Test sentence."
        pred = ["Test sentence."]
        true = ["Test sentence."]

        metrics = calculate_metrics(pred, true, text, processing_time=0.01)

        assert metrics.processing_time == 0.01
        assert metrics.sentences_per_second == 100  # 1 sentence / 0.01s


class TestEvaluationMetricsClass:
    """Test the EvaluationMetrics dataclass."""

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = EvaluationMetrics(
            precision=0.8765,
            recall=0.7654,
            f1=0.8171,
            accuracy=0.8,
            boundary_precision=0.85,
            boundary_recall=0.82,
            boundary_f1=0.835,
            exact_match_accuracy=0.75,
            over_segmentation_rate=0.1,
            under_segmentation_rate=0.05,
            avg_sentence_length_diff=5.5,
            processing_time=1.234,
            sentences_per_second=123.45,
            total_sentences_pred=100,
            total_sentences_true=95,
            total_boundaries_pred=100,
            total_boundaries_true=95,
        )

        d = metrics.to_dict()

        # Check rounding
        assert d["precision"] == 0.8765
        assert d["recall"] == 0.7654
        assert d["f1"] == 0.8171
        assert d["processing_time"] == 1.234
        assert d["sentences_per_second"] == 123.45

        # Check integer values
        assert d["total_sentences_pred"] == 100
        assert d["total_sentences_true"] == 95

    def test_metrics_summary(self):
        """Test human-readable summary generation."""
        metrics = EvaluationMetrics(
            precision=0.85,
            recall=0.80,
            f1=0.825,
            accuracy=0.82,
            boundary_precision=0.86,
            boundary_recall=0.81,
            boundary_f1=0.835,
            exact_match_accuracy=0.75,
            over_segmentation_rate=0.1,
            under_segmentation_rate=0.05,
            avg_sentence_length_diff=5.5,
            processing_time=1.5,
            sentences_per_second=100,
            total_sentences_pred=100,
            total_sentences_true=95,
            total_boundaries_pred=100,
            total_boundaries_true=95,
        )

        summary = metrics.summary()

        # Check key information is present
        assert "Core Metrics:" in summary
        assert "85.00%" in summary  # Precision
        assert "80.00%" in summary  # Recall
        assert "82.50%" in summary  # F1
        assert "Boundary Detection:" in summary
        assert "Performance:" in summary
        assert "100" in summary  # Sentences/second


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_instance = TestMetricsCalculation()
    test_instance.test_precision_recall_f1_perfect()
    test_instance.test_precision_recall_f1_typical()
    test_instance.test_get_sentence_boundaries_simple()
    test_instance.test_boundary_accuracy_exact()
    test_instance.test_boundary_accuracy_with_tolerance()

    print("All manual tests passed!")
