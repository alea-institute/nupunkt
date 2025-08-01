#!/usr/bin/env python3
"""
Script to test the nupunkt tokenizer on custom text.

This script:
1. Loads the default model
2. Tokenizes text provided by the user
3. Shows the resulting sentences
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.models import load_default_model


def get_test_text() -> str:
    """Return sample test text if none is provided."""
    return """
Dr. Smith went to Washington, D.C. He was very excited about the trip.
The company (Ltd.) was founded in 1997. It has grown significantly since then.
This text contains an ellipsis... And this is a new sentence.
Let me give you an example, e.g. this one. Did you understand it?
The meeting is at 3 p.m. Don't be late!
Under 18 U.S.C. 12, this is a legal citation. The next sentence begins here.
The patient presented with abd. pain. CT scan was ordered.
The table shows results for Jan. Feb. and Mar. Each month shows improvement.
Visit the website at www.example.com. There you'll find more information.
She scored 92 vs. 85 in the previous match. Her performance has improved.
The temperature was 32 deg. C. It was quite hot that day.
    """


def tokenize_text(text: str, model_path: Path | None = None) -> None:
    """
    Tokenize the given text and print the results.

    Args:
        text: The text to tokenize
        model_path: Optional path to a custom model
    """
    # Load the tokenizer
    print("Loading default model...")
    tokenizer = load_default_model()
    print("Model loaded successfully.")

    # Tokenize the text
    print("\n=== Tokenizing Text ===")
    print(f"Input text:\n{text}")

    sentences = tokenizer.tokenize(text)

    print("\n=== Tokenization Results ===")
    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence.strip()}")

    print(f"\nFound {len(sentences)} sentences.")


def main() -> None:
    """Process command-line arguments and tokenize text."""
    parser = argparse.ArgumentParser(description="Test the nupunkt tokenizer on custom text")
    parser.add_argument(
        "--text", type=str, default=None, help="Text to tokenize (default: use sample text)"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Path to a file containing text to tokenize"
    )
    parser.add_argument("--model", type=str, default=None, help="Path to a custom model file")

    args = parser.parse_args()

    # Get the text to tokenize
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = get_test_text()

    # Tokenize the text
    tokenize_text(text, args.model)


if __name__ == "__main__":
    main()
