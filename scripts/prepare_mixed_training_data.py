#!/usr/bin/env python3
"""
Prepare mixed training data from multiple HuggingFace datasets.

This script streams and shuffles data from multiple alea-institute datasets
to create a diverse training corpus for nupunkt. It's kept for reproducibility
of training data generation.

Features:
- Streaming mode for memory efficiency
- Shuffled sampling for better data mixture
- Configurable sample sizes per dataset
- Reproducible with seed setting
- Output in JSONL.gz format compatible with existing training pipeline

REQUIRES EXTERNAL DEPENDENCIES (not part of nupunkt):
  uv run --with datasets python scripts/prepare_mixed_training_data.py

Usage:
    # Default: 1000 samples from each default dataset
    uv run --with datasets python scripts/prepare_mixed_training_data.py
    
    # Custom samples per dataset (5000 each)
    uv run --with datasets python scripts/prepare_mixed_training_data.py --samples 5000
    
    # Different samples for each dataset
    uv run --with datasets python scripts/prepare_mixed_training_data.py \
        --datasets alea-institute/kl3m-data-usc alea-institute/kl3m-data-ecfr \
        --samples 5000 2000
        
    # Custom output path
    uv run --with datasets python scripts/prepare_mixed_training_data.py \
        --samples 1000 \
        --output data/my_training_data.jsonl.gz

Note: The main nupunkt CLI can train directly from HuggingFace datasets:
    python -m nupunkt train hf:alea-institute/kl3m-data-usc --output model.bin
    
This script allows creating mixed datasets with specific sampling ratios.
"""

import argparse
import gzip
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# Check if HuggingFace datasets is available
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("ERROR: HuggingFace datasets not available.")
    print("Run this script with:")
    print("  uv run --with datasets python scripts/prepare_mixed_training_data.py")
    sys.exit(1)


def stream_dataset_samples(
    dataset_name: str,
    max_samples: int,
    text_field: str | None = None,
    seed: int | None = None
) -> Iterator[Dict[str, Any]]:
    """
    Stream samples from a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        max_samples: Maximum number of samples to yield
        text_field: Field name containing text (auto-detected if None)
        seed: Random seed for shuffling
        
    Yields:
        Dictionary with 'text' field containing the document text
    """
    print(f"Loading {dataset_name} in streaming mode...")
    
    # Load dataset in streaming mode
    dataset = load_dataset(dataset_name, streaming=True, split="train")
    
    # Shuffle the dataset if seed is provided
    if seed is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    samples_yielded = 0
    
    for record in dataset:
        # Extract text from record
        text = None
        
        # Try specified field first
        if text_field and text_field in record:
            text = record[text_field]
        # Auto-detect text field
        elif 'text' in record:
            text = record['text']
        else:
            # Try common field names
            for field in ['content', 'document', 'sentence', 'paragraph']:
                if field in record:
                    text = record[field]
                    break
        
        if text:
            yield {'text': text}
            samples_yielded += 1
            
            if samples_yielded >= max_samples:
                break
    
    print(f"  Extracted {samples_yielded} samples from {dataset_name}")


def mix_datasets(
    dataset_configs: List[Tuple[str, int]],
    seed: int | None = None
) -> List[Dict[str, Any]]:
    """
    Mix samples from multiple datasets.
    
    Args:
        dataset_configs: List of (dataset_name, num_samples) tuples
        seed: Random seed for reproducibility
        
    Returns:
        List of mixed samples
    """
    all_samples = []
    
    # Collect samples from each dataset
    for dataset_name, num_samples in dataset_configs:
        samples = list(stream_dataset_samples(
            dataset_name,
            num_samples,
            seed=seed
        ))
        all_samples.extend(samples)
    
    # Shuffle all samples together
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_samples)
    
    print(f"\nTotal samples collected: {len(all_samples)}")
    return all_samples


def save_samples(
    samples: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save samples to a JSONL.gz file.
    
    Args:
        samples: List of sample dictionaries
        output_path: Path to output file
    """
    print(f"Saving {len(samples)} samples to {output_path}")
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Output file size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed training data from HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "alea-institute/kl3m-data-usc",
            "alea-institute/kl3m-data-ecfr",
            "alea-institute/kl3m-data-fr",
            "alea-institute/kl3m-data-edgar-agreements",
            "alea-institute/kl3m-data-govinfo-crecb",
            "alea-institute/kl3m-data-pacer-docs",
            "alea-institute/kl3m-data-govinfo-chrg",
            "alea-institute/kl3m-data-govinfo-govpub",
            "alea-institute/kl3m-data-edgar-10-k"
        ],
        help="List of HuggingFace dataset names"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        help="Number of samples per dataset (single value for all, or one per dataset)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: data/train-{timestamp}.jsonl.gz)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Handle samples argument
    if not args.samples:
        # Default: 1000 samples per dataset
        samples_per_dataset = [1000] * len(args.datasets)
    elif len(args.samples) == 1:
        # Single value: use for all datasets
        samples_per_dataset = args.samples * len(args.datasets)
    elif len(args.samples) == len(args.datasets):
        # One value per dataset
        samples_per_dataset = args.samples
    else:
        parser.error(f"--samples must be either a single value or {len(args.datasets)} values")
    
    # Create dataset configurations
    dataset_configs = list(zip(args.datasets, samples_per_dataset))
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data") / f"train-{timestamp}.jsonl.gz"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("Mixed Training Data Preparation")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")
    print("\nDatasets and samples:")
    for dataset, samples in dataset_configs:
        print(f"  - {dataset}: {samples} samples")
    print()
    
    # Mix datasets
    mixed_samples = mix_datasets(dataset_configs, seed=args.seed)
    
    # Save results
    save_samples(mixed_samples, output_path)
    
    print("\nDone!")
    
    # Print example usage for training
    print("\nTo train a model with this data:")
    print(f"  python -m nupunkt train {output_path} \\")
    print("    --hyperparameters conservative \\")
    print("    --output models/mixed_model.bin")


if __name__ == "__main__":
    main()