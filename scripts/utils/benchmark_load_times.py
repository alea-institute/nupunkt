#!/usr/bin/env python3
"""
Script to benchmark loading times for different model formats.

This script:
1. Creates test models in different formats
2. Measures loading time for each format
3. Reports the results
"""

import sys
import os
import time
import tempfile
from pathlib import Path
import argparse
import statistics

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.utils.compression import (
    load_compressed_json, 
    save_compressed_json, 
    save_binary_model, 
    load_binary_model
)
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

def measure_load_time(file_path: Path, iterations: int = 5) -> dict:
    """Measure the time it takes to load a model."""
    load_times = []
    
    for i in range(iterations):
        start_time = time.time()
        if str(file_path).endswith('.bin'):
            # Load using binary loader
            _ = load_binary_model(file_path)
        else:
            # Load using standard method
            _ = load_compressed_json(file_path)
        end_time = time.time()
        load_times.append(end_time - start_time)
    
    return {
        "iterations": iterations,
        "min": min(load_times),
        "max": max(load_times),
        "mean": statistics.mean(load_times),
        "median": statistics.median(load_times)
    }

def measure_tokenize_time(file_path: Path, text: str, iterations: int = 5) -> dict:
    """Measure the time it takes to tokenize text with a model."""
    # Load the model
    if str(file_path).endswith('.bin'):
        from nupunkt.core.parameters import PunktParameters
        params_dict = load_binary_model(file_path) 
        params = PunktParameters.from_json(params_dict)
        tokenizer = PunktSentenceTokenizer(params)
    else:
        from nupunkt.core.parameters import PunktParameters
        params_dict = load_compressed_json(file_path)
        if "parameters" in params_dict:
            params = PunktParameters.from_json(params_dict["parameters"])
        else:
            params = PunktParameters.from_json(params_dict)
        tokenizer = PunktSentenceTokenizer(params)
    
    # Measure tokenization time
    tokenize_times = []
    for i in range(iterations):
        start_time = time.time()
        sentences = tokenizer.tokenize(text)
        end_time = time.time()
        tokenize_times.append(end_time - start_time)
    
    return {
        "iterations": iterations,
        "min": min(tokenize_times),
        "max": max(tokenize_times),
        "mean": statistics.mean(tokenize_times),
        "median": statistics.median(tokenize_times)
    }

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    kb = size_bytes / 1024
    if kb < 1000:
        return f"{kb:.2f} KB"
    mb = kb / 1024
    return f"{mb:.2f} MB"

def run_benchmark(model_path: Path, output_dir: Path, text: str) -> None:
    """Run benchmark tests on different model formats."""
    print(f"Loading source model from {model_path}...")
    source_data = load_compressed_json(model_path)
    
    # Format combinations to test
    formats = [
        ("json", "none", 0, "model.json"),
        ("json_xz", "none", 6, "model.json.xz"),
        ("binary", "none", 0, "model_raw.bin"),
        ("binary", "zlib", 6, "model_zlib.bin"),
        ("binary", "lzma", 6, "model_lzma.bin"),
        ("binary", "gzip", 6, "model_gzip.bin")
    ]
    
    # Create test models
    test_files = []
    print("\nCreating test models...")
    for format_type, compression, level, filename in formats:
        test_path = output_dir / filename
        
        print(f"  Creating {format_type} model with {compression} compression (level {level})...")
        
        if format_type == "binary":
            save_binary_model(
                source_data,
                test_path,
                compression_method=compression,
                level=level
            )
        else:
            save_compressed_json(
                source_data,
                test_path,
                level=level,
                use_compression=(format_type == "json_xz")
            )
        
        if os.path.exists(test_path):
            test_files.append((format_type, compression, level, test_path))
            print(f"    Created: {test_path} ({format_size(os.path.getsize(test_path))})")
        else:
            print(f"    Failed to create {test_path}")
    
    # Benchmark loading times
    print("\n=== Loading Time Benchmark ===")
    print(f"{'Format':<10} {'Compression':<12} {'Level':<6} {'Size':<10} {'Mean (s)':<10} {'Median (s)':<10} {'Relative':<10}")
    print("-" * 80)
    
    # First measure times for all formats
    results_map = {}
    for format_type, compression, level, path in test_files:
        size = os.path.getsize(path)
        results = measure_load_time(path)
        results_map[(format_type, compression, level)] = (size, results)
    
    # Find the fastest loading format to use as baseline
    baseline_time = min(r[1]['mean'] for r in results_map.values())
    
    # Print results with relative performance
    for format_type, compression, level, path in test_files:
        size, results = results_map[(format_type, compression, level)]
        relative = results['mean'] / baseline_time
        
        print(f"{format_type:<10} {compression:<12} {level:<6} {format_size(size):<10} "
              f"{results['mean']:<10.5f} {results['median']:<10.5f} {relative:<10.2f}")
    
    # Benchmark tokenization times
    print("\n=== Tokenization Time Benchmark ===")
    print(f"{'Format':<10} {'Compression':<12} {'Level':<6} {'Size':<10} {'Mean (s)':<10} {'Median (s)':<10} {'Relative':<10}")
    print("-" * 80)
    
    # First measure times for all formats
    tokenize_results = {}
    for format_type, compression, level, path in test_files:
        size = os.path.getsize(path)
        results = measure_tokenize_time(path, text)
        tokenize_results[(format_type, compression, level)] = (size, results)
    
    # Find the fastest tokenization format to use as baseline
    baseline_time = min(r[1]['mean'] for r in tokenize_results.values())
    
    # Print results with relative performance
    for format_type, compression, level, path in test_files:
        size, results = tokenize_results[(format_type, compression, level)]
        relative = results['mean'] / baseline_time
        
        print(f"{format_type:<10} {compression:<12} {level:<6} {format_size(size):<10} "
              f"{results['mean']:<10.5f} {results['median']:<10.5f} {relative:<10.2f}")

def main():
    """Run benchmarks on different model formats."""
    parser = argparse.ArgumentParser(
        description="Benchmark loading times for different model formats"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to the model to benchmark (default: use the default model)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save test models (default: use a temporary directory)"
    )
    
    args = parser.parse_args()
    
    # Use default model if not specified
    if args.model:
        model_path = Path(args.model)
    else:
        from nupunkt.models import get_default_model_path
        model_path = get_default_model_path()
    
    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)
    
    # Test text for tokenization benchmark
    test_text = """
    Dr. Smith went to Washington, D.C. He was very excited about the trip.
    The company (Ltd.) was founded in 1997. It has grown significantly since then.
    This text contains an ellipsis... And this is a new sentence.
    Let me give you an example, e.g. this one. Did you understand it?
    The meeting is at 3 p.m. Don't be late!
    Under 18 U.S.C. 12, this is a legal citation. The next sentence begins here.
    """ * 50  # Multiply to create larger text
    
    try:
        # Run the benchmark
        run_benchmark(model_path, output_dir, test_text)
    finally:
        # Clean up if using temporary directory
        if not args.output_dir:
            temp_dir.cleanup()

if __name__ == "__main__":
    main()