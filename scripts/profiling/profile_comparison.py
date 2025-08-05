#!/usr/bin/env python3
"""
Compare performance of different nupunkt tokenization methods.

This script provides comprehensive performance comparison between:
- Standard sent_tokenize
- Adaptive sent_tokenize_adaptive with various settings
- Different threshold values
- Impact of dynamic abbreviation detection
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nupunkt
from scripts.profiling.profiling_utils import (
    create_base_parser,
    load_jsonl_texts,
    compare_functions,
    print_comparison_table,
    profile_function,
    ProfilingResult,
    save_profile_data
)


def create_detailed_report(results: list[ProfilingResult], output_path: Path = None):
    """Create a detailed performance report."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("NUPUNKT PERFORMANCE ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Summary table
    report_lines.append("Performance Summary")
    report_lines.append("-"*80)
    report_lines.append(f"{'Configuration':<40} {'Time (s)':<12} {'Chars/sec':<15} {'Relative':>10}")
    report_lines.append("-"*80)
    
    baseline_speed = results[0].chars_per_second if results else 1
    
    for result in results:
        relative = result.chars_per_second / baseline_speed if baseline_speed > 0 else 0
        report_lines.append(
            f"{result.function_name:<40} "
            f"{result.total_time:<12.3f} "
            f"{result.chars_per_second:<15,.0f} "
            f"{relative:>10.2f}x"
        )
    
    report_lines.append("="*80)
    report_lines.append("")
    
    # Detailed analysis
    report_lines.append("Detailed Analysis")
    report_lines.append("-"*80)
    
    # Find fastest and slowest
    fastest = max(results, key=lambda r: r.chars_per_second)
    slowest = min(results, key=lambda r: r.chars_per_second)
    
    report_lines.append(f"Fastest configuration: {fastest.function_name}")
    report_lines.append(f"  - Speed: {fastest.chars_per_second:,.0f} chars/sec")
    report_lines.append(f"  - Total time: {fastest.total_time:.3f} seconds")
    report_lines.append("")
    
    report_lines.append(f"Slowest configuration: {slowest.function_name}")
    report_lines.append(f"  - Speed: {slowest.chars_per_second:,.0f} chars/sec")
    report_lines.append(f"  - Total time: {slowest.total_time:.3f} seconds")
    report_lines.append("")
    
    # Performance ratio
    perf_ratio = fastest.chars_per_second / slowest.chars_per_second
    report_lines.append(f"Performance ratio (fastest/slowest): {perf_ratio:.2f}x")
    
    # Adaptive overhead analysis
    standard_result = next((r for r in results if "standard" in r.function_name), None)
    if standard_result:
        report_lines.append("")
        report_lines.append("Adaptive Mode Overhead Analysis")
        report_lines.append("-"*40)
        
        for result in results:
            if "adaptive" in result.function_name:
                overhead = ((result.total_time / standard_result.total_time) - 1) * 100
                report_lines.append(f"{result.function_name}: {overhead:+.1f}%")
    
    report_text = "\n".join(report_lines)
    
    # Print report
    print("\n" + report_text)
    
    # Save if requested
    if output_path:
        output_path.write_text(report_text)
        print(f"\nReport saved to: {output_path}")
    
    return report_text


def main():
    """Main comparison profiling function."""
    parser = create_base_parser(
        "Compare performance of nupunkt tokenization methods"
    )
    
    # Add comparison-specific arguments
    parser.add_argument(
        '--model',
        type=str,
        default='default',
        help='Model to use for all tokenization methods'
    )
    
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.5, 0.7, 0.9],
        help='Threshold values to test for adaptive mode'
    )
    
    parser.add_argument(
        '--save-report',
        type=Path,
        default=None,
        help='Save detailed report to file'
    )
    
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include all variations (with/without dynamic abbrev)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable all caching'
    )
    
    args = parser.parse_args()
    
    # Clear caches if requested
    if args.no_cache:
        nupunkt.load.cache_clear()
        nupunkt._get_default_model.cache_clear()
        nupunkt._get_adaptive_tokenizer.cache_clear()
    
    # Load data
    print(f"Loading data from: {args.data}")
    texts = list(load_jsonl_texts(args.data, limit=args.limit))
    print(f"Loaded {len(texts)} texts")
    print(f"Total characters: {sum(len(t) for t in texts):,}")
    print()
    
    if not texts:
        print("Error: No texts loaded!", file=sys.stderr)
        sys.exit(1)
    
    # Build list of configurations to test
    results = []
    
    # 1. Standard tokenization (baseline)
    print("Profiling standard tokenization...")
    standard_result = profile_function(
        func=nupunkt.sent_tokenize,
        texts=texts,
        func_args={'model': args.model},
        warmup_samples=args.warmup
    )
    standard_result.function_name = "sent_tokenize (standard)"
    results.append(standard_result)
    
    # 2. Adaptive with different thresholds
    for threshold in args.thresholds:
        print(f"\nProfiling adaptive tokenization (threshold={threshold})...")
        
        # With dynamic abbreviation
        adaptive_result = profile_function(
            func=nupunkt.sent_tokenize_adaptive,
            texts=texts,
            func_args={
                'threshold': threshold,
                'model': args.model,
                'dynamic_abbrev': True
            },
            warmup_samples=args.warmup
        )
        adaptive_result.function_name = f"sent_tokenize_adaptive (t={threshold})"
        results.append(adaptive_result)
        
        # Without dynamic abbreviation (if requested)
        if args.include_all:
            adaptive_no_dyn = profile_function(
                func=nupunkt.sent_tokenize_adaptive,
                texts=texts,
                func_args={
                    'threshold': threshold,
                    'model': args.model,
                    'dynamic_abbrev': False
                },
                warmup_samples=args.warmup
            )
            adaptive_no_dyn.function_name = f"adaptive (t={threshold}, no dyn abbrev)"
            results.append(adaptive_no_dyn)
    
    # Create detailed report
    create_detailed_report(results, output_path=args.save_report)
    
    # Save individual profile data if requested
    if args.save_profile:
        for i, result in enumerate(results):
            profile_path = args.save_profile.with_suffix(f".{i}.prof")
            save_profile_data(result, profile_path)
    
    # Memory usage estimate (optional)
    print("\n" + "="*60)
    print("Resource Usage Estimates")
    print("="*60)
    print(f"Texts processed: {len(texts)}")
    print(f"Total characters: {sum(len(t) for t in texts):,}")
    print(f"Average text length: {sum(len(t) for t in texts) / len(texts):.0f} chars")
    
    # Estimate sentences processed
    total_sentences = sum(len(nupunkt.sent_tokenize(t)) for t in texts[:10])
    avg_sentences = total_sentences / 10
    est_total_sentences = avg_sentences * len(texts)
    print(f"Estimated sentences: {est_total_sentences:,.0f}")


if __name__ == '__main__':
    main()