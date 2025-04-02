#!/usr/bin/env python3
"""
Script to benchmark different cache size configurations for nupunkt.

This script tests different cache size settings and reports their performance.
"""

import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.core.constants import (
    DOC_TOKENIZE_CACHE_SIZE,
    ORTHO_CACHE_SIZE,
    PARA_TOKENIZE_CACHE_SIZE,
    SENT_STARTER_CACHE_SIZE,
    WHITESPACE_CACHE_SIZE,
)
from nupunkt.models import load_default_model
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

# Use a subset of test cases from test_default_model.py
TEST_CASES = [
    "This is a simple sentence. This is another one.",
    "The contract was signed on June 15, 2024. Both parties received copies.",
    "The market closed at 4 p.m. yesterday. Trading volume was high.",
    "Pursuant to 17 U.S.C. § 506(a), copyright infringement is a federal crime. Penalties can be severe.",
    "As stated in Fed. R. Civ. P. 56(c), summary judgment is appropriate. This standard is well established.",
    "In Smith v. Jones, 123 F.3d 456 (9th Cir. 1997), the court ruled for the plaintiff. This set a precedent.",
    "According to Cal. Civ. Code § 1624, certain contracts must be in writing. This requirement is strictly enforced.",
    "The company reported Q1 earnings of $2.5B. This exceeded analyst expectations.",
    "Apple Inc. (AAPL) closed at $189.84 on Friday. The stock has gained 15.3% YTD.",
    "The Federal Reserve raised interest rates by 25 bps. Markets reacted positively to the announcement.",
    "As per the 10-K filing, revenue increased by 12.4% YoY. Operating expenses remained stable.",
    "Prof. Williams presented the findings at the conference. His research was well-received.",
    "The merger between Corp. A and Inc. B was approved. Shareholders will vote next month.",
    "Dr. Johnson et al. published their analysis in the J. Fin. Econ. The paper examined market anomalies.",
    "Ms. Garcia, Esq. filed the motion on behalf of XYZ Co. The hearing is scheduled for next week.",
    "The SEC issued Rule 10b-5, which prohibits securities fraud. Violations can result in severe penalties.",
    "The prospectus (dated March 1, 2024) contains important disclosures. Investors should read it carefully.",
    "The meeting is at 2:30 p.m. EST. Please prepare all required documentation.",
    'He stated, "The merger will be completed by Q3." The timeline seems ambitious.',
    "The court granted cert. Review is pending before the Supreme Court.",
    "Inflation rose to 3.2% in Jan. Feb. and Mar. showed similar trends.",
    "Visit the company website at www.example.com. There you'll find investor information.",
    "The stock trades at 15x vs. 20x for peers. This valuation gap represents an opportunity.",
    "The bond yields 5.2% p.a. This rate is competitive in the current market.",
]

# Generate a larger test text by repeating test cases
LARGE_TEST_TEXT = " ".join(TEST_CASES * 20)  # Create a significant amount of text to process


def test_model_accuracy(tokenizer: PunktSentenceTokenizer, text: str, iterations: int = 5) -> float:
    """
    Test the tokenizer's performance on text with multiple iterations.

    Args:
        tokenizer: The sentence tokenizer to test
        text: Text sample to process
        iterations: Number of test iterations for more reliable results

    Returns:
        Average characters processed per second
    """
    # Create variations of the test text to avoid document caching
    variations = []
    for i in range(iterations):
        # Add a unique prefix to create a new string that won't be cached
        variations.append(f"Test iteration {i}: {text}")

    # Warm up run to initialize any JIT optimizations
    _ = tokenizer.tokenize(variations[0][:1000])

    # Clear anything that might be cached
    gc.collect()

    speeds = []

    # Run iterations with different variations of text
    for i in range(iterations):
        variation = variations[i]

        # Force garbage collection before each run
        gc.collect()

        # Time the tokenization
        start_time = time.time()
        _ = tokenizer.tokenize(variation)
        end_time = time.time()

        processing_time = end_time - start_time
        chars_per_second = len(variation) / processing_time if processing_time > 0 else 0
        speeds.append(chars_per_second)

        print(f"    Iteration {i + 1}: {chars_per_second:,.0f} chars/sec")

    # Return average speed
    return sum(speeds) / len(speeds)


def create_tokenizer_with_settings(settings: Dict[str, int]) -> PunktSentenceTokenizer:
    """Create a new tokenizer with the given cache size settings."""
    tokenizer = load_default_model()

    # Apply cache settings
    return PunktSentenceTokenizer(
        train_text=tokenizer._params,
        cache_size=settings.get("doc_cache", DOC_TOKENIZE_CACHE_SIZE),
        paragraph_cache_size=settings.get("para_cache", PARA_TOKENIZE_CACHE_SIZE),
        ortho_cache_size=settings.get("ortho_cache", ORTHO_CACHE_SIZE),
        sent_starter_cache_size=settings.get("sent_starter_cache", SENT_STARTER_CACHE_SIZE),
        whitespace_cache_size=settings.get("whitespace_cache", WHITESPACE_CACHE_SIZE),
    )


def test_cache_settings() -> List[Tuple[Dict[str, int], float]]:
    """Test different cache size configurations and return their performances."""
    test_text = LARGE_TEST_TEXT
    print(f"Using test text of {len(test_text):,} characters")

    # Cache size configurations to test
    configs = [
        # Default settings for baseline
        {
            "name": "Default",
            "doc_cache": DOC_TOKENIZE_CACHE_SIZE,
            "para_cache": PARA_TOKENIZE_CACHE_SIZE,
            "ortho_cache": ORTHO_CACHE_SIZE,
            "sent_starter_cache": SENT_STARTER_CACHE_SIZE,
            "whitespace_cache": WHITESPACE_CACHE_SIZE,
        },
        # Test larger cache sizes (2x)
        {
            "name": "Large (2x)",
            "doc_cache": DOC_TOKENIZE_CACHE_SIZE * 2,
            "para_cache": PARA_TOKENIZE_CACHE_SIZE * 2,
            "ortho_cache": ORTHO_CACHE_SIZE * 2,
            "sent_starter_cache": SENT_STARTER_CACHE_SIZE * 2,
            "whitespace_cache": WHITESPACE_CACHE_SIZE * 2,
        },
        # Test smaller cache sizes (0.5x)
        {
            "name": "Small (0.5x)",
            "doc_cache": max(DOC_TOKENIZE_CACHE_SIZE // 2, 10),
            "para_cache": max(PARA_TOKENIZE_CACHE_SIZE // 2, 10),
            "ortho_cache": max(ORTHO_CACHE_SIZE // 2, 10),
            "sent_starter_cache": max(SENT_STARTER_CACHE_SIZE // 2, 10),
            "whitespace_cache": max(WHITESPACE_CACHE_SIZE // 2, 10),
        },
        # Test much larger document cache
        {
            "name": "Large Doc Cache",
            "doc_cache": DOC_TOKENIZE_CACHE_SIZE * 5,
            "para_cache": PARA_TOKENIZE_CACHE_SIZE,
            "ortho_cache": ORTHO_CACHE_SIZE,
            "sent_starter_cache": SENT_STARTER_CACHE_SIZE,
            "whitespace_cache": WHITESPACE_CACHE_SIZE,
        },
        # Test larger orthographic cache
        {
            "name": "Large Ortho Cache",
            "doc_cache": DOC_TOKENIZE_CACHE_SIZE,
            "para_cache": PARA_TOKENIZE_CACHE_SIZE,
            "ortho_cache": ORTHO_CACHE_SIZE * 5,
            "sent_starter_cache": SENT_STARTER_CACHE_SIZE,
            "whitespace_cache": WHITESPACE_CACHE_SIZE,
        },
        # Test larger paragraph cache
        {
            "name": "Large Para Cache",
            "doc_cache": DOC_TOKENIZE_CACHE_SIZE,
            "para_cache": PARA_TOKENIZE_CACHE_SIZE * 5,
            "ortho_cache": ORTHO_CACHE_SIZE,
            "sent_starter_cache": SENT_STARTER_CACHE_SIZE,
            "whitespace_cache": WHITESPACE_CACHE_SIZE,
        },
        # Test larger sentence starter cache
        {
            "name": "Large Sent Starter Cache",
            "doc_cache": DOC_TOKENIZE_CACHE_SIZE,
            "para_cache": PARA_TOKENIZE_CACHE_SIZE,
            "ortho_cache": ORTHO_CACHE_SIZE,
            "sent_starter_cache": SENT_STARTER_CACHE_SIZE * 5,
            "whitespace_cache": WHITESPACE_CACHE_SIZE,
        },
        # Test larger whitespace cache
        {
            "name": "Large Whitespace Cache",
            "doc_cache": DOC_TOKENIZE_CACHE_SIZE,
            "para_cache": PARA_TOKENIZE_CACHE_SIZE,
            "ortho_cache": ORTHO_CACHE_SIZE,
            "sent_starter_cache": SENT_STARTER_CACHE_SIZE,
            "whitespace_cache": WHITESPACE_CACHE_SIZE * 5,
        },
    ]

    results = []

    # Randomize the order of configurations to avoid any systematic bias
    import random

    random.shuffle(configs)

    # Test each configuration
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        print(f"  Doc Cache: {config['doc_cache']}, Para Cache: {config['para_cache']}")
        print(
            f"  Ortho Cache: {config['ortho_cache']}, Sent Starter Cache: {config['sent_starter_cache']}"
        )
        print(f"  Whitespace Cache: {config['whitespace_cache']}")

        # Create a fresh tokenizer with these settings
        tokenizer = create_tokenizer_with_settings(config)

        # Test performance
        speed = test_model_accuracy(tokenizer, test_text)
        print(f"  Average: {speed:,.0f} characters/second")

        results.append((config, speed))

    return results


def main():
    """Run the cache size benchmarks."""
    print("=== Testing Cache Size Configurations ===")
    results = test_cache_settings()

    if results:
        # Sort by performance (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        print("\n=== Performance Ranking ===")
        baseline_config = next(
            (config for config, _ in results if config["name"] == "Default"), None
        )
        baseline_speed = next(
            (speed for config, speed in results if config["name"] == "Default"), None
        )

        for i, (config, speed) in enumerate(results, 1):
            improvement = ""
            if baseline_speed and config["name"] != "Default":
                percent = ((speed - baseline_speed) / baseline_speed) * 100
                improvement = f" ({percent:+.1f}% vs. default)"

            print(f"{i}. {config['name']}: {speed:,.0f} chars/sec{improvement}")

        best_config = results[0][0]
        print("\n=== Recommended Cache Sizes ===")
        print(f"Document Cache: {best_config['doc_cache']} (was {DOC_TOKENIZE_CACHE_SIZE})")
        print(f"Paragraph Cache: {best_config['para_cache']} (was {PARA_TOKENIZE_CACHE_SIZE})")
        print(f"Orthographic Cache: {best_config['ortho_cache']} (was {ORTHO_CACHE_SIZE})")
        print(
            f"Sent Starter Cache: {best_config['sent_starter_cache']} (was {SENT_STARTER_CACHE_SIZE})"
        )
        print(f"Whitespace Cache: {best_config['whitespace_cache']} (was {WHITESPACE_CACHE_SIZE})")

        # For each setting, show performance impact
        print("\n=== Detailed Performance Impact ===")
        if baseline_config:

            def compare_setting(setting_name, config_key, default_value):
                configs_with_different_value = [
                    (c, s) for c, s in results if c[config_key] != default_value
                ]
                if configs_with_different_value:
                    largest_improvement = max(configs_with_different_value, key=lambda x: x[1])
                    percent = ((largest_improvement[1] - baseline_speed) / baseline_speed) * 100
                    best_value = largest_improvement[0][config_key]
                    print(
                        f"{setting_name}: Best value {best_value} ({percent:+.1f}% vs. default {default_value})"
                    )

            compare_setting("Document Cache", "doc_cache", DOC_TOKENIZE_CACHE_SIZE)
            compare_setting("Paragraph Cache", "para_cache", PARA_TOKENIZE_CACHE_SIZE)
            compare_setting("Orthographic Cache", "ortho_cache", ORTHO_CACHE_SIZE)
            compare_setting("Sent Starter Cache", "sent_starter_cache", SENT_STARTER_CACHE_SIZE)
            compare_setting("Whitespace Cache", "whitespace_cache", WHITESPACE_CACHE_SIZE)
    else:
        print("No results available. Make sure test data is available.")


if __name__ == "__main__":
    main()
