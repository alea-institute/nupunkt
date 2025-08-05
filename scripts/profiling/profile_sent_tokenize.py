#!/usr/bin/env python3
"""
Profile the sent_tokenize function from nupunkt.

This script profiles the standard sentence tokenization function
to identify performance bottlenecks and measure throughput.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nupunkt
from scripts.profiling.profiling_utils import (
    create_base_parser,
    run_profiling,
    ProfilingResult
)


def print_tokenization_stats(result: ProfilingResult):
    """Print additional tokenization-specific statistics."""
    # We can estimate sentences per second by re-running on a sample
    # This is a quick approximation
    avg_chars_per_text = result.total_chars / result.sample_count
    est_sents_per_text = avg_chars_per_text / 50  # Rough estimate: 50 chars/sentence
    sents_per_second = (result.sample_count * est_sents_per_text) / result.total_time
    
    print(f"Estimated sentences/sec:   {sents_per_second:,.0f}")
    print(f"Avg chars/text:            {avg_chars_per_text:,.0f}")


def main():
    """Main profiling function."""
    import cProfile
    import pstats
    
    parser = create_base_parser(
        "Profile nupunkt.sent_tokenize performance"
    )
    
    # Add sent_tokenize specific arguments
    parser.add_argument(
        '--model',
        type=str,
        default='default',
        help='Model to use for tokenization'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable model caching (force reload each time)'
    )
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.no_cache:
        nupunkt.load.cache_clear()
        nupunkt._get_default_model.cache_clear()
    
    # Load data
    from scripts.profiling.profiling_utils import load_jsonl_texts
    print(f"Loading data from: {args.data}")
    texts = list(load_jsonl_texts(args.data, limit=args.limit))
    print(f"Loaded {len(texts)} texts\n")
    
    # Run cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    for text in texts:
        _ = nupunkt.sent_tokenize(text, model=args.model)
    
    profiler.disable()
    
    # Print stats directly
    print("cProfile Results:")
    print("="*80)
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(args.top)


if __name__ == '__main__':
    main()