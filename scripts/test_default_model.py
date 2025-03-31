#!/usr/bin/env python3
"""
Script to test and verify the default model for nupunkt.

This script:
1. Tests the default model with sample legal and financial texts
2. Exports the model to different formats as needed
"""

import sys
import os
import time
import argparse
from pathlib import Path
import json
import gzip
from typing import List, Dict, Any

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.models import load_default_model, get_default_model_path, optimize_default_model
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

# Test cases organized by category
TEST_CASES = {
    "basic": [
        "This is a simple sentence. This is another one.",
        "The contract was signed on June 15, 2024. Both parties received copies.",
        "The market closed at 4 p.m. yesterday. Trading volume was high."
    ],
    "legal_citations": [
        "Pursuant to 17 U.S.C. § 506(a), copyright infringement is a federal crime. Penalties can be severe.",
        "As stated in Fed. R. Civ. P. 56(c), summary judgment is appropriate. This standard is well established.",
        "In Smith v. Jones, 123 F.3d 456 (9th Cir. 1997), the court ruled for the plaintiff. This set a precedent.",
        "According to Cal. Civ. Code § 1624, certain contracts must be in writing. This requirement is strictly enforced."
    ],
    "financial": [
        "The company reported Q1 earnings of $2.5B. This exceeded analyst expectations.",
        "Apple Inc. (AAPL) closed at $189.84 on Friday. The stock has gained 15.3% YTD.",
        "The Federal Reserve raised interest rates by 25 bps. Markets reacted positively to the announcement.",
        "As per the 10-K filing, revenue increased by 12.4% YoY. Operating expenses remained stable."
    ],
    "abbreviations": [
        "Prof. Williams presented the findings at the conference. His research was well-received.",
        "The merger between Corp. A and Inc. B was approved. Shareholders will vote next month.",
        "Dr. Johnson et al. published their analysis in the J. Fin. Econ. The paper examined market anomalies.",
        "Ms. Garcia, Esq. filed the motion on behalf of XYZ Co. The hearing is scheduled for next week."
    ],
    "punctuation": [
        "The SEC issued Rule 10b-5, which prohibits securities fraud. Violations can result in severe penalties.",
        "The prospectus (dated March 1, 2024) contains important disclosures. Investors should read it carefully.",
        "The meeting is at 2:30 p.m. EST. Please prepare all required documentation.",
        "He stated, \"The merger will be completed by Q3.\" The timeline seems ambitious."
    ],
    "challenging": [
        "The court granted cert. Review is pending before the Supreme Court.",
        "Inflation rose to 3.2% in Jan. Feb. and Mar. showed similar trends.",
        "Visit the company website at www.example.com. There you'll find investor information.",
        "The stock trades at 15x vs. 20x for peers. This valuation gap represents an opportunity.",
        "The bond yields 5.2% p.a. This rate is competitive in the current market."
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

def test_model():
    """Test the default model with sample texts and run performance evaluation."""
    print("Loading default model...")
    model_path = get_default_model_path()
    tokenizer = load_default_model()
    
    print(f"Loaded model from: {model_path}")
    
    # Test on examples
    test_examples(tokenizer)
    
    # Test on test data if available
    test_path = root_dir / "data" / "test.jsonl.gz"
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

def export_model(format_type, compression_method, compression_level, output_path=None):
    """Export the default model to a specified format."""
    print(f"\n=== Exporting Model to {format_type} Format ===")
    print(f"Compression method: {compression_method}, Level: {compression_level}")
    
    # Export the model
    result_path = optimize_default_model(
        output_path=output_path,
        format_type=format_type,
        compression_method=compression_method,
        compression_level=compression_level
    )
    
    print(f"Model exported successfully to: {result_path}")
    if os.path.exists(result_path):
        size_kb = os.path.getsize(result_path) / 1024
        print(f"File size: {size_kb:.2f} KB")

def main():
    """Run the default model testing and format optimization."""
    parser = argparse.ArgumentParser(description="Test and optimize the default model for nupunkt")
    parser.add_argument("--test", action="store_true", help="Test the model with sample texts")
    parser.add_argument("--export", action="store_true", help="Export the model to a specified format")
    parser.add_argument("--format", type=str, default="binary", 
                        choices=["json", "json_xz", "binary"],
                        help="Format to save the model in")
    parser.add_argument("--compression", type=str, default="lzma", 
                        choices=["none", "zlib", "lzma", "gzip"],
                        help="Compression method for binary format")
    parser.add_argument("--level", type=int, default=6, 
                        help="Compression level (0-9)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path for the exported model")
    
    args = parser.parse_args()
    
    # If no arguments provided, run all tests
    if not (args.test or args.export):
        args.test = True
    
    # Run tests if requested
    if args.test:
        test_model()
    
    # Export model if requested
    if args.export:
        export_model(
            format_type=args.format,
            compression_method=args.compression,
            compression_level=args.level,
            output_path=args.output
        )

if __name__ == "__main__":
    main()