#!/usr/bin/env python3
"""
Base utilities for profiling nupunkt functions.

Provides shared functionality for loading data, running profiles,
and formatting results across different profiling scripts.
"""

import argparse
import cProfile
import gzip
import io
import json
import pstats
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple


@dataclass
class ProfilingResult:
    """Container for profiling results."""
    
    function_name: str
    total_time: float
    total_chars: int
    chars_per_second: float
    profile_stats: pstats.Stats
    sample_count: int


def load_jsonl_texts(filepath: Path, limit: Optional[int] = None) -> Iterator[str]:
    """
    Load texts from a JSONL file (gzipped or plain).
    
    Args:
        filepath: Path to the JSONL file
        limit: Maximum number of texts to load (None for all)
        
    Yields:
        Text strings from the file
    """
    open_fn = gzip.open if filepath.suffix == '.gz' else open
    
    count = 0
    with open_fn(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
            
            data = json.loads(line.strip())
            if 'text' in data:
                # Remove sentence markers if present
                text = data['text'].replace('<|sentence|>', '')
                yield text
                count += 1


def profile_function(
    func: Callable,
    texts: List[str],
    func_args: dict = None,
    warmup_samples: int = 10
) -> ProfilingResult:
    """
    Profile a function on a list of texts.
    
    Args:
        func: Function to profile
        texts: List of texts to process
        func_args: Additional arguments to pass to the function
        warmup_samples: Number of warmup iterations before profiling
        
    Returns:
        ProfilingResult with timing and profiling data
    """
    if func_args is None:
        func_args = {}
    
    # Warmup to ensure any caching/initialization is done
    warmup_texts = texts[:warmup_samples] if len(texts) > warmup_samples else texts
    for text in warmup_texts:
        _ = func(text, **func_args)
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the function
    total_chars = sum(len(text) for text in texts)
    
    start_time = time.perf_counter()
    profiler.enable()
    
    for text in texts:
        _ = func(text, **func_args)
    
    profiler.disable()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    chars_per_second = total_chars / total_time if total_time > 0 else 0
    
    # Get stats
    stats_buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_buffer)
    
    return ProfilingResult(
        function_name=func.__name__,
        total_time=total_time,
        total_chars=total_chars,
        chars_per_second=chars_per_second,
        profile_stats=stats,
        sample_count=len(texts)
    )


def print_profiling_summary(result: ProfilingResult):
    """Print a formatted summary of profiling results."""
    print(f"\n{'='*60}")
    print(f"Profiling Summary: {result.function_name}")
    print(f"{'='*60}")
    print(f"Total samples:     {result.sample_count:,}")
    print(f"Total characters:  {result.total_chars:,}")
    print(f"Total time:        {result.total_time:.3f} seconds")
    print(f"Throughput:        {result.chars_per_second:,.0f} chars/sec")
    print(f"{'='*60}\n")


def print_top_functions(result: ProfilingResult, limit: int = 20):
    """Print the top time-consuming functions."""
    print(f"\nTop {limit} time-consuming functions:")
    print("-" * 80)
    result.profile_stats.sort_stats('cumulative').print_stats(limit)


def save_profile_data(result: ProfilingResult, output_path: Path):
    """Save raw profile data to a file."""
    result.profile_stats.dump_stats(str(output_path))
    print(f"\nProfile data saved to: {output_path}")


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with common options."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/test.jsonl.gz'),
        help='Path to JSONL data file'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of texts to process (None for all)'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup samples before profiling'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top functions to display'
    )
    
    parser.add_argument(
        '--save-profile',
        type=Path,
        default=None,
        help='Path to save raw profile data'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed profile output'
    )
    
    return parser


def run_profiling(
    func: Callable,
    args: argparse.Namespace,
    func_args: dict = None,
    custom_summary: Callable[[ProfilingResult], None] = None
):
    """
    Run profiling with standard workflow.
    
    Args:
        func: Function to profile
        args: Parsed command line arguments
        func_args: Additional arguments to pass to the function
        custom_summary: Optional custom summary printer
    """
    # Load data
    print(f"Loading data from: {args.data}")
    texts = list(load_jsonl_texts(args.data, limit=args.limit))
    print(f"Loaded {len(texts)} texts")
    
    if not texts:
        print("Error: No texts loaded!", file=sys.stderr)
        sys.exit(1)
    
    # Run profiling
    print(f"\nProfiling {func.__name__}...")
    result = profile_function(
        func=func,
        texts=texts,
        func_args=func_args,
        warmup_samples=args.warmup
    )
    
    # Print summary
    print_profiling_summary(result)
    
    # Custom summary if provided
    if custom_summary:
        custom_summary(result)
    
    # Print top functions if not quiet
    if not args.quiet:
        print_top_functions(result, limit=args.top)
    
    # Save profile if requested
    if args.save_profile:
        save_profile_data(result, args.save_profile)


def compare_functions(
    functions: List[Tuple[Callable, dict]],
    texts: List[str],
    warmup_samples: int = 10
) -> List[ProfilingResult]:
    """
    Compare multiple functions on the same dataset.
    
    Args:
        functions: List of (function, func_args) tuples
        texts: List of texts to process
        warmup_samples: Number of warmup iterations
        
    Returns:
        List of ProfilingResult objects
    """
    results = []
    
    for func, func_args in functions:
        print(f"\nProfiling {func.__name__}...")
        result = profile_function(
            func=func,
            texts=texts,
            func_args=func_args or {},
            warmup_samples=warmup_samples
        )
        results.append(result)
        print_profiling_summary(result)
    
    return results


def print_comparison_table(results: List[ProfilingResult]):
    """Print a comparison table of multiple profiling results."""
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)
    print(f"{'Function':<30} {'Time (s)':<12} {'Chars/sec':<15} {'Relative':>10}")
    print("-"*80)
    
    # Find baseline (first result)
    baseline_speed = results[0].chars_per_second if results else 1
    
    for result in results:
        relative = result.chars_per_second / baseline_speed if baseline_speed > 0 else 0
        print(f"{result.function_name:<30} "
              f"{result.total_time:<12.3f} "
              f"{result.chars_per_second:<15,.0f} "
              f"{relative:>10.2f}x")
    
    print("="*80)