#!/usr/bin/env python3
"""
Script to profile the performance of the default model for nupunkt.

This script:
1. Loads the default model from nupunkt/models/
2. Profiles its performance using cProfile and line_profiler
3. Generates reports showing where time is spent
"""

import argparse
import cProfile
import gzip
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

# Test cases from test_default_model.py
TEST_CASES = {
    "basic": [
        "This is a simple sentence. This is another one.",
        "Hello world! How are you today? I'm doing well.",
        "The quick brown fox jumps over the lazy dog. The fox was very quick.",
    ],
    "abbreviations": [
        "Dr. Smith went to Washington, D.C. He was very excited about the trip.",
        "The company (Ltd.) was founded in 1997. It has grown significantly since then.",
        "Mr. Johnson and Mrs. Lee will attend the meeting at 3 p.m. They will discuss the agenda.",
        "She has a B.A. in English. She also studied French in college.",
        "The U.S. economy is growing. Many industries are showing improvement.",
    ],
    "legal_citations": [
        "Under 18 U.S.C. 12, this is a legal citation. The next sentence begins here.",
        "As stated in Fed. R. Civ. P. 56(c), summary judgment is appropriate. This standard is well established.",
        "In Smith v. Jones, 123 F.3d 456 (9th Cir. 1997), the court ruled in favor of the plaintiff. This set a precedent.",
        "According to Cal. Civ. Code § 123, the contract must be in writing. This requirement is strict.",
    ],
    "ellipsis": [
        "This text contains an ellipsis... And this is a new sentence.",
        "The story continues... But not for long.",
        "He paused for a moment... Then he continued walking.",
        "She thought about it for a while... Then she made her decision.",
    ],
    "other_punctuation": [
        "Let me give you an example, e.g. this one. Did you understand it?",
        "The company (formerly known as Tech Solutions, Inc.) was acquired last year. The new owners rebranded it.",
        "The meeting is at 3 p.m. Don't be late!",
        'He said, "I\'ll be there at 5 p.m." Then he hung up the phone.',
    ],
    "challenging": [
        "The patient presented with abd. pain. CT scan was ordered.",
        "The table shows results for Jan. Feb. and Mar. Each month shows improvement.",
        "Visit the website at www.example.com. There you'll find more information.",
        "She scored 92 vs. 85 in the previous match. Her performance has improved.",
        "The temperature was 32 deg. C. It was quite hot that day.",
    ],
}


def load_test_data(file_path: Path) -> List[str]:
    """Load text examples from a test file."""
    print(f"Loading test data from {file_path}...")
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            if file_path.suffix == ".jsonl.gz":
                # Handle JSONL format
                texts = []
                for line in f:
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            texts.append(data["text"])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in {file_path}")
                return texts
            else:
                # Plain text in gzip
                return f.read().split("\n\n")
    else:
        # Regular text file
        with open(file_path, encoding="utf-8") as f:
            return f.read().split("\n\n")


def test_model_accuracy(tokenizer: PunktSentenceTokenizer, texts: List[str]) -> Dict[str, Any]:
    """Test the tokenizer's performance on a list of texts."""
    start_time = time.time()
    total_chars = sum(len(text) for text in texts)
    total_sentences = 0

    for text in texts:
        sentences = tokenizer.tokenize(text)
        total_sentences += len(sentences)

    end_time = time.time()
    processing_time = end_time - start_time
    chars_per_second = total_chars / processing_time if processing_time > 0 else 0

    return {
        "total_texts": len(texts),
        "total_chars": total_chars,
        "total_sentences": total_sentences,
        "processing_time_seconds": processing_time,
        "chars_per_second": chars_per_second,
    }


def run_cprofile(tokenizer: PunktSentenceTokenizer, texts: List[str], output_path: Path) -> None:
    """Run cProfile on the tokenizer with the given texts."""
    print("\n=== Running cProfile Analysis ===")

    # Create profile output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the actual test
    for text in texts:
        tokenizer.tokenize(text)

    profiler.disable()

    # Save stats to file
    stats_path = output_path.with_suffix(".prof")
    profiler.dump_stats(str(stats_path))
    print(f"cProfile data saved to: {stats_path}")

    # Generate readable report
    stats = pstats.Stats(str(stats_path))
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        sys.stdout = f  # Redirect stdout to file
        stats.sort_stats("cumulative").print_stats(30)
        sys.stdout = sys.__stdout__  # Reset stdout

    print(f"cProfile report saved to: {txt_path}")

    # Print summary to console
    print("\nTop 10 functions by cumulative time:")
    stats.sort_stats("cumulative").print_stats(10)


def run_line_profiler(
    tokenizer: PunktSentenceTokenizer, texts: List[str], output_path: Path
) -> None:
    """Run line_profiler on the tokenizer with the given texts."""
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("line_profiler not installed. Skipping line-by-line profiling.")
        print("Install with: pip install line_profiler")
        return

    print("\n=== Running Line Profiler Analysis ===")

    # Create a line profiler and add functions to profile
    lp = LineProfiler()
    lp.add_function(tokenizer.tokenize)

    # Try to profile important internal methods if they exist
    for method_name in [
        "_slices_from_text",
        "_annotate_tokens",
        "_tokenize_words",
        "_handle_abbrev",
        "_handle_potential_sentence_break",
    ]:
        if hasattr(tokenizer, method_name):
            method = getattr(tokenizer, method_name)
            lp.add_function(method)

    # Profile the tokenization process
    lp_wrapper = lp(lambda: [tokenizer.tokenize(text) for text in texts])
    lp_wrapper()

    # Save the line profiler stats to text file
    txt_path = output_path.with_suffix(".line.txt")

    # Generate text report
    with open(txt_path, "w") as f:
        sys.stdout = f  # Redirect stdout to file
        lp.print_stats()
        sys.stdout = sys.__stdout__  # Reset stdout

    print(f"Line profiler report saved to: {txt_path}")

    # Print summary to console
    print("\nLine-by-line profiling results:")
    lp.print_stats()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the nupunkt default model")
    parser.add_argument(
        "--examples-only", action="store_true", help="Only use predefined examples (faster)"
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        default=True,
        help="Run cProfile profiling (default: True)",
    )
    parser.add_argument(
        "--line-profiler",
        action="store_true",
        help="Run line_profiler profiling (requires line_profiler package)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profile_results",
        help="Base name for output files (without extension)",
    )
    return parser.parse_args()


def main():
    """Profile the default model and report results."""
    args = parse_args()

    # Set paths
    models_dir = root_dir / "nupunkt" / "models"
    model_path = models_dir / "default_model.bin"
    test_path = root_dir / "data" / "test.jsonl.gz"
    output_dir = root_dir / "profiles"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / args.output

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    print(f"Loading model from {model_path}...")
    tokenizer = PunktSentenceTokenizer.load(model_path)
    print("Model loaded successfully")

    # Determine which texts to use for profiling
    if args.examples_only:
        # Use only the predefined examples
        all_examples = []
        for examples in TEST_CASES.values():
            all_examples.extend(examples)
        texts = all_examples
        print(f"Using {len(texts)} predefined examples for profiling")
    else:
        # Use test data if available
        if test_path.exists():
            texts = load_test_data(test_path)
            print(f"Using {len(texts)} test documents for profiling")
        else:
            print(f"Test data file not found: {test_path}")
            print("Falling back to predefined examples")
            all_examples = []
            for examples in TEST_CASES.values():
                all_examples.extend(examples)
            texts = all_examples
            print(f"Using {len(texts)} predefined examples for profiling")

    # Run requested profiling
    if args.cprofile:
        run_cprofile(tokenizer, texts, output_path)

    if args.line_profiler:
        run_line_profiler(tokenizer, texts, output_path)

    print("\nProfiling completed successfully.")


if __name__ == "__main__":
    main()
