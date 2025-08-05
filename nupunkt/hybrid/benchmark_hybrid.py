#!/usr/bin/env python3
"""
Benchmark script to compare original Punkt with confidence-based hybrid approach.
"""

import time
from typing import Any, Dict

from nupunkt import PunktSentenceTokenizer
from nupunkt.hybrid import (
    CHALLENGE_SET,
    ConfidenceSentenceTokenizer,
    create_domain_tokenizer,
    evaluate_tokenizer,
)


def benchmark_performance(tokenizer, text: str, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark tokenizer performance."""
    start = time.perf_counter()

    for _ in range(iterations):
        sentences = tokenizer.tokenize(text)

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / iterations
    chars_per_sec = len(text) * iterations / total_time

    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "chars_per_sec": chars_per_sec,
        "sentences_found": len(sentences),
    }


def compare_tokenizers():
    """Compare different tokenizer configurations."""

    print("Comparing Tokenizers on Challenge Set")
    print("=" * 80)

    # Test text for performance benchmarking
    perf_test_text = " ".join([item[0] for item in CHALLENGE_SET]) * 10

    tokenizers = {
        "Original Punkt": PunktSentenceTokenizer(),
        "Confidence (Default)": ConfidenceSentenceTokenizer(),
        "Confidence (Legal)": create_domain_tokenizer("legal"),
        "Confidence (Scientific)": create_domain_tokenizer("scientific"),
        "Confidence (High Threshold)": ConfidenceSentenceTokenizer(confidence_threshold=0.7),
        "Confidence (Low Threshold)": ConfidenceSentenceTokenizer(confidence_threshold=0.3),
    }

    results = {}

    for name, tokenizer in tokenizers.items():
        print(f"\n{name}:")
        print("-" * 40)

        # Accuracy evaluation
        correct, total, errors = evaluate_tokenizer(tokenizer, verbose=False)
        accuracy = correct / total * 100

        # Performance benchmark
        perf = benchmark_performance(tokenizer, perf_test_text, iterations=10)

        results[name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": len(errors),
            "chars_per_sec": perf["chars_per_sec"],
        }

        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"Speed: {perf['chars_per_sec']:,.0f} chars/sec")

        # Show first few errors if any
        if errors and len(errors) <= 3:
            print(f"Errors on: {[e['description'] for e in errors]}")

    # Summary table
    print("\n\nSummary Comparison")
    print("=" * 80)
    print(f"{'Tokenizer':<30} {'Accuracy':<12} {'Speed (chars/s)':<15} {'Errors'}")
    print("-" * 80)

    for name, res in results.items():
        print(
            f"{name:<30} {res['accuracy']:>6.1f}% {res['chars_per_sec']:>15,.0f} {res['errors']:>6}"
        )

    # Find best accuracy
    best_accuracy = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.1f}%)")

    # Find best speed
    best_speed = max(results.items(), key=lambda x: x[1]["chars_per_sec"])
    print(f"Best Speed: {best_speed[0]} ({best_speed[1]['chars_per_sec']:,.0f} chars/s)")


def debug_specific_case():
    """Debug a specific challenging case with confidence scores."""

    print("\n\nDebug Mode: Analyzing Specific Cases")
    print("=" * 80)

    # Pick a challenging case
    test_case = "Dr. Smith studied at M.I.T. in Cambridge."

    tokenizer = ConfidenceSentenceTokenizer(debug=True)
    results = tokenizer.tokenize_with_confidence(test_case)

    print(f"Text: {test_case}")
    print("Expected sentences: 1")
    print(f"Found sentences: {len(results)}")
    print()

    for i, (sentence, confidence) in enumerate(results, 1):
        print(f"Sentence {i}: {sentence}")
        print(f"  Total confidence: {confidence.total:.3f}")
        print(f"  Threshold: {confidence.decision_threshold}")
        print(f"  Decision: {'BREAK' if confidence.is_boundary else 'NO BREAK'}")
        print("  Components:")
        for comp, score in confidence.components.items():
            print(f"    {comp:<15}: {score:.3f}")
        print()


def test_threshold_tuning():
    """Test different confidence thresholds to find optimal value."""

    print("\n\nThreshold Tuning Analysis")
    print("=" * 80)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print(f"{'Threshold':<12} {'Accuracy':<12} {'Errors'}")
    print("-" * 36)

    for threshold in thresholds:
        tokenizer = ConfidenceSentenceTokenizer(confidence_threshold=threshold)
        correct, total, errors = evaluate_tokenizer(tokenizer, verbose=False)
        accuracy = correct / total * 100
        print(f"{threshold:<12.1f} {accuracy:>6.1f}%      {len(errors)}")

    print("\nNote: Lower thresholds = more sentence breaks")
    print("      Higher thresholds = fewer sentence breaks")


if __name__ == "__main__":
    compare_tokenizers()
    debug_specific_case()
    test_threshold_tuning()
