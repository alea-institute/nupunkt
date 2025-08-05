#!/usr/bin/env python3
"""
Profile the sent_tokenize_adaptive function from nupunkt.

This script profiles the adaptive sentence tokenization function
with various threshold settings to understand performance characteristics.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nupunkt
from scripts.profiling.profiling_utils import (
    create_base_parser,
    run_profiling,
    ProfilingResult,
    profile_function,
    print_profiling_summary,
    print_comparison_table,
    load_jsonl_texts
)


def print_adaptive_stats(result: ProfilingResult):
    """Print adaptive tokenization specific statistics."""
    # Estimate performance impact vs standard tokenization
    print(f"\nAdaptive tokenization specific metrics:")
    print(f"  - Dynamic abbreviation detection enabled")
    print(f"  - Confidence scoring enabled")
    
    # Calculate overhead estimate
    avg_ms_per_text = (result.total_time * 1000) / result.sample_count
    print(f"  - Avg time per text: {avg_ms_per_text:.2f} ms")


def profile_threshold_variations(texts, args):
    """Profile different threshold settings."""
    print("\n" + "="*60)
    print("Profiling Different Threshold Settings")
    print("="*60)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for threshold in thresholds:
        func_args = {
            'threshold': threshold,
            'model': args.model,
            'dynamic_abbrev': True
        }
        
        result = profile_function(
            func=nupunkt.sent_tokenize_adaptive,
            texts=texts,
            func_args=func_args,
            warmup_samples=args.warmup
        )
        
        # Modify function name to include threshold
        result.function_name = f"adaptive (threshold={threshold})"
        results.append(result)
        
        print(f"\nThreshold {threshold}:")
        print(f"  Time: {result.total_time:.3f}s")
        print(f"  Speed: {result.chars_per_second:,.0f} chars/sec")
    
    return results


def main():
    """Main profiling function."""
    import cProfile
    import pstats
    
    parser = create_base_parser(
        "Profile nupunkt.sent_tokenize_adaptive performance"
    )
    
    # Add adaptive-specific arguments
    parser.add_argument(
        '--model',
        type=str,
        default='default',
        help='Model to use for tokenization'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for adaptive mode'
    )
    
    parser.add_argument(
        '--no-dynamic-abbrev',
        action='store_true',
        help='Disable dynamic abbreviation detection'
    )
    
    parser.add_argument(
        '--profile-thresholds',
        action='store_true',
        help='Profile different threshold settings'
    )
    
    parser.add_argument(
        '--compare-standard',
        action='store_true',
        help='Compare with standard tokenization'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable tokenizer caching'
    )
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.no_cache:
        nupunkt.load.cache_clear()
        nupunkt._get_default_model.cache_clear()
        nupunkt._get_adaptive_tokenizer.cache_clear()
    
    # Load data
    print(f"Loading data from: {args.data}")
    texts = list(load_jsonl_texts(args.data, limit=args.limit))
    print(f"Loaded {len(texts)} texts")
    
    if not texts:
        print("Error: No texts loaded!", file=sys.stderr)
        sys.exit(1)
    
    # Run cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    for text in texts:
        _ = nupunkt.sent_tokenize_adaptive(
            text,
            threshold=args.threshold,
            model=args.model,
            dynamic_abbrev=not args.no_dynamic_abbrev
        )
    
    profiler.disable()
    
    # Print stats directly
    print("cProfile Results:")
    print("="*80)
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(args.top)
    
    # Profile threshold variations if requested
    if args.profile_thresholds:
        threshold_results = profile_threshold_variations(texts, args)
        print_comparison_table(threshold_results)
    
    # Compare with standard tokenization if requested
    if args.compare_standard:
        print("\n" + "="*60)
        print("Comparing Adaptive vs Standard Tokenization")
        print("="*60)
        
        # Profile standard
        standard_result = profile_function(
            func=nupunkt.sent_tokenize,
            texts=texts,
            func_args={'model': args.model},
            warmup_samples=args.warmup
        )
        standard_result.function_name = "sent_tokenize (standard)"
        
        # Profile adaptive
        adaptive_result = profile_function(
            func=nupunkt.sent_tokenize_adaptive,
            texts=texts,
            func_args=func_args,
            warmup_samples=args.warmup
        )
        adaptive_result.function_name = f"sent_tokenize_adaptive ({args.threshold})"
        
        # Show comparison
        print_comparison_table([standard_result, adaptive_result])
        
        # Calculate overhead
        overhead = ((adaptive_result.total_time / standard_result.total_time) - 1) * 100
        print(f"\nAdaptive overhead: {overhead:.1f}%")


if __name__ == '__main__':
    main()