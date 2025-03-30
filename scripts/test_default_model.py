#!/usr/bin/env python3
"""
Script to test the default model for nupunkt.

This script:
1. Loads the default model from nupunkt/models/
2. Tests its performance on a variety of sentence examples
3. Optionally runs evaluations on test data if available
"""

import sys
import os
import time
from pathlib import Path
import gzip
import json
from typing import List, Dict, Tuple, Any

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

# Test cases organized by category
TEST_CASES = {
    "basic": [
        "This is a simple sentence. This is another one.",
        "Hello world! How are you today? I'm doing well.",
        "The quick brown fox jumps over the lazy dog. The fox was very quick."
    ],
    "abbreviations": [
        "Dr. Smith went to Washington, D.C. He was very excited about the trip.",
        "The company (Ltd.) was founded in 1997. It has grown significantly since then.",
        "Mr. Johnson and Mrs. Lee will attend the meeting at 3 p.m. They will discuss the agenda.",
        "She has a B.A. in English. She also studied French in college.",
        "The U.S. economy is growing. Many industries are showing improvement."
    ],
    "legal_citations": [
        "Under 18 U.S.C. 12, this is a legal citation. The next sentence begins here.",
        "As stated in Fed. R. Civ. P. 56(c), summary judgment is appropriate. This standard is well established.",
        "In Smith v. Jones, 123 F.3d 456 (9th Cir. 1997), the court ruled in favor of the plaintiff. This set a precedent.",
        "According to Cal. Civ. Code § 123, the contract must be in writing. This requirement is strict."
    ],
    "ellipsis": [
        "This text contains an ellipsis... And this is a new sentence.",
        "The story continues... But not for long.",
        "He paused for a moment... Then he continued walking.",
        "She thought about it for a while... Then she made her decision."
    ],
    "other_punctuation": [
        "Let me give you an example, e.g. this one. Did you understand it?",
        "The company (formerly known as Tech Solutions, Inc.) was acquired last year. The new owners rebranded it.",
        "The meeting is at 3 p.m. Don't be late!",
        "He said, \"I'll be there at 5 p.m.\" Then he hung up the phone."
    ],
    "challenging": [
        "The patient presented with abd. pain. CT scan was ordered.",
        "The table shows results for Jan. Feb. and Mar. Each month shows improvement.",
        "Visit the website at www.example.com. There you'll find more information.",
        "She scored 92 vs. 85 in the previous match. Her performance has improved.",
        "The temperature was 32 deg. C. It was quite hot that day."
    ]
}

def load_test_data(file_path: Path) -> List[str]:
    """Load text examples from a test file."""
    print(f"Loading test data from {file_path}...")
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            if file_path.suffix == '.jsonl.gz':
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
                return f.read().split('\n\n')
    else:
        # Regular text file
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().split('\n\n')

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
        "chars_per_second": chars_per_second
    }

def test_examples(tokenizer: PunktSentenceTokenizer) -> None:
    """Test the tokenizer on the predefined examples."""
    print("\n=== Testing Default Model on Examples ===")
    
    for category, examples in TEST_CASES.items():
        print(f"\n== {category.upper()} EXAMPLES ==")
        for text in examples:
            print(f"\nText: {text}")
            sentences = tokenizer.tokenize(text)
            for i, sentence in enumerate(sentences, 1):
                print(f"  Sentence {i}: {sentence.strip()}")

def main():
    """Test the default model and report results."""
    # Set paths
    models_dir = root_dir / "nupunkt" / "models"
    model_path = models_dir / "default_model.json.xz"
    test_path = root_dir / "data" / "test.jsonl.gz"
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    tokenizer = PunktSentenceTokenizer.load(model_path)
    print("Model loaded successfully")
    
    # Test on examples
    test_examples(tokenizer)
    
    # Test on test data if available
    if test_path.exists():
        print(f"\n=== Performance Evaluation on Test Data ===")
        test_texts = load_test_data(test_path)
        
        # Run benchmark
        print(f"Running benchmark on {len(test_texts)} test documents...")
        results = test_model_accuracy(tokenizer, test_texts)
        
        # Print results
        print("\nPerformance Results:")
        print(f"  Documents processed:      {results['total_texts']}")
        print(f"  Total characters:         {results['total_chars']:,}")
        print(f"  Total sentences found:    {results['total_sentences']:,}")
        print(f"  Processing time:          {results['processing_time_seconds']:.2f} seconds")
        print(f"  Processing speed:         {results['chars_per_second']:,.0f} characters/second")
        print(f"  Average sentence length:  {results['total_chars'] / results['total_sentences']:.1f} characters")
    else:
        print(f"\nNote: Test data file not found: {test_path}")
        print("Skipping performance evaluation.")
    
    print("\nTesting completed successfully.")

if __name__ == "__main__":
    main()