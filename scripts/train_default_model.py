#!/usr/bin/env python3
"""
Script to train the default model for nupunkt.

This script:
1. Trains a general-purpose English sentence tokenizer using local data files
2. Loads abbreviations from data/legal_abbreviations.json
3. Saves the model to the models directory as the package's default model

Requires data/train.jsonl.gz, data/train2.jsonl.gz, and data/legal_abbreviations.json to be present.
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import time

# For progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Provide a simple fallback if tqdm is not installed
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        return lambda x: x

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer
from nupunkt.trainers.base_trainer import PunktTrainer
from nupunkt.utils.compression import compare_formats


def load_abbreviations(file_path: Path) -> List[str]:
    """Load abbreviations from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        abbreviations = json.load(f)

    # Convert abbreviations to lowercase and remove trailing periods
    cleaned_abbrevs = []
    for abbr in abbreviations:
        # Remove trailing period if present
        if abbr.endswith("."):
            abbr = abbr[:-1]
        # Convert to lowercase
        cleaned_abbrevs.append(abbr.lower())

    return cleaned_abbrevs


def load_jsonl_text(file_path: Path, max_samples: Optional[int] = None) -> str:
    """Load text from a compressed JSONL file with progress bar."""
    combined_text = ""
    sample_count = 0
    
    # Get total line count for the progress bar if max_samples is not set
    total = max_samples or sum(1 for _ in gzip.open(file_path, "rt", encoding="utf-8"))
    
    # Use tqdm to show progress
    desc = f"Loading from {file_path.name}"
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        # Create progress bar
        for line in tqdm(f, total=total, desc=desc, unit="samples"):
            try:
                data = json.loads(line)
                if "text" in data:
                    combined_text += data["text"] + "\n\n"
                    sample_count += 1
                    if max_samples and sample_count >= max_samples:
                        break
            except json.JSONDecodeError:
                tqdm.write(f"Warning: Could not parse line in {file_path}")
                continue

    # Print summary at the end
    tqdm.write(f"Loaded {sample_count} text samples, total size: {len(combined_text):,} characters")
    return combined_text


def print_training_details(trainer: PunktTrainer, initial_abbrevs: set) -> None:
    """Print detailed information about the trained model."""
    tqdm.write("\n" + "=" * 80)
    tqdm.write("TRAINING RESULTS SUMMARY")
    tqdm.write("=" * 80)

    # Print hyperparameters
    tqdm.write("\nHyperparameters:")
    tqdm.write(f"  - Abbreviation threshold:      {trainer.ABBREV}")
    tqdm.write(f"  - Abbreviation backoff:        {trainer.ABBREV_BACKOFF}")
    tqdm.write(f"  - Collocation threshold:       {trainer.COLLOCATION}")
    tqdm.write(f"  - Sentence starter threshold:  {trainer.SENT_STARTER}")
    tqdm.write(f"  - Minimum collocation freq:    {trainer.MIN_COLLOC_FREQ}")
    tqdm.write(f"  - Maximum abbreviation length: {trainer.MAX_ABBREV_LENGTH}")

    # Memory optimization parameters
    if trainer.MEMORY_EFFICIENT:
        tqdm.write("\nMemory Optimization Settings:")
        tqdm.write(f"  - Min type frequency:          {trainer.TYPE_FDIST_MIN_FREQ}")
        tqdm.write(f"  - Min sentence starter freq:   {trainer.SENT_STARTER_MIN_FREQ}")
        tqdm.write(f"  - Min collocation frequency:   {trainer.COLLOC_FDIST_MIN_FREQ}")
        tqdm.write(f"  - Pruning interval:            {trainer.PRUNE_INTERVAL}")

    # Get parameters and stats
    params = trainer.get_params()
    loaded_abbrev_count = len(initial_abbrevs)
    current_abbrevs = set(params.abbrev_types)
    learned_abbrevs = current_abbrevs - initial_abbrevs
    learned_abbrev_count = len(learned_abbrevs)

    # Print model statistics
    tqdm.write("\nModel Statistics:")
    tqdm.write(f"  - Abbreviations:               {len(params.abbrev_types):,}")
    tqdm.write(f"    - Loaded from file:          {loaded_abbrev_count:,}")
    tqdm.write(f"    - Learned during training:   {learned_abbrev_count:,}")
    tqdm.write(f"  - Collocations:                {len(params.collocations):,}")
    tqdm.write(f"  - Sentence starters:           {len(params.sent_starters):,}")

    # Print some learned abbreviations (if any)
    if learned_abbrevs:
        tqdm.write("\nSample of learned abbreviations (up to 20):")
        for abbr in sorted(list(learned_abbrevs))[:20]:
            tqdm.write(f"  - {abbr}")

    # Print some sample collocations
    if params.collocations:
        tqdm.write("\nSample of collocations (up to 20):")
        sorted_collocs = sorted(params.collocations)
        for w1, w2 in sorted_collocs[:20]:
            tqdm.write(f"  - {w1} {w2}")

    # Print some sample sentence starters
    if params.sent_starters:
        tqdm.write("\nSample of sentence starters (up to 20):")
        sorted_starters = sorted(params.sent_starters)
        for starter in sorted_starters[:20]:
            tqdm.write(f"  - {starter}")

    tqdm.write("\n" + "=" * 80)


def compare_storage_formats(trainer: PunktTrainer, output_dir: Path) -> None:
    """Compare different storage formats for the model with progress bar."""
    tqdm.write("\n" + "=" * 80)
    tqdm.write("STORAGE FORMAT COMPARISON")
    tqdm.write("=" * 80)

    # Get model data
    data = trainer.get_params().to_json()

    # Compare formats
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timer to measure comparison time
    start_time = time.time()
    tqdm.write("\nComparing storage formats (this may take a moment)...")
    
    # Run comparison with formats
    format_sizes = compare_formats(data, output_dir)
    
    elapsed_time = time.time() - start_time
    tqdm.write(f"Comparison completed in {elapsed_time:.2f} seconds")

    # Print sizes in human-readable format
    tqdm.write("\nStorage Format Comparison:")
    tqdm.write(f"{'Format':<30} {'Size':<10} {'Ratio':<10}")
    tqdm.write("-" * 50)

    # Get the json size as reference for compression ratio
    json_size = format_sizes.get("json", 1)

    # Sort by size for easier comparison
    for format_name, size in sorted(format_sizes.items(), key=lambda x: x[1]):
        ratio = size / json_size if json_size else 0
        size_str = f"{size / 1024:.2f} KB"
        ratio_str = f"{ratio:.3f}"
        tqdm.write(f"{format_name:<30} {size_str:<10} {ratio_str:<10}")

    tqdm.write("\nRecommended format: binary_zlib_level6 or binary_lzma_level6")
    tqdm.write("Note: Binary formats require less deserialization overhead than JSON")
    tqdm.write("\n" + "=" * 80)


def main() -> None:
    """
    Train a default model and save it to the package models directory.
    
    This script now uses memory optimization techniques by default:
    - Memory-efficient mode with early frequency pruning
    - Batch training to process text in manageable chunks
    
    These optimizations significantly reduce memory usage while maintaining model quality.
    Use --no-memory-efficient or --no-batches to disable these optimizations if needed.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train the default model for nupunkt with memory-efficient processing enabled by default"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from each training file",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="binary",
        choices=["json", "json_xz", "binary"],
        help="Format to save the model in",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zlib",
        choices=["none", "zlib", "lzma", "gzip"],
        help="Compression method for binary format",
    )
    parser.add_argument("--level", type=int, default=6, help="Compression level (0-9)")
    parser.add_argument("--compare", action="store_true", help="Compare different storage formats")
    # Memory optimization options
    parser.add_argument(
        "--no-memory-efficient", 
        action="store_true",
        help="Disable memory-efficient training mode (on by default)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1000000,
        help="Batch size (in characters) for batch training"
    )
    parser.add_argument(
        "--prune-freq", 
        type=int, 
        default=10000,
        help="How often to prune distributions (token count)"
    )
    parser.add_argument(
        "--min-type-freq", 
        type=int, 
        default=3,
        help="Minimum frequency to keep a type (higher values reduce memory usage)"
    )
    parser.add_argument(
        "--min-starter-freq", 
        type=int, 
        default=5,
        help="Minimum frequency to keep a sentence starter"
    )
    parser.add_argument(
        "--min-colloc-freq", 
        type=int, 
        default=3,
        help="Minimum frequency to keep a collocation"
    )
    parser.add_argument(
        "--no-batches", 
        action="store_true",
        help="Disable batch training mode (on by default)"
    )
    args = parser.parse_args()

    # Set paths
    data_dir = root_dir / "data"
    models_dir = root_dir / "nupunkt" / "models"

    # Determine model path based on format
    if args.format == "binary":
        model_path = models_dir / "default_model.bin"
    elif args.format == "json_xz":
        model_path = models_dir / "default_model.json.xz"
    else:
        model_path = models_dir / "default_model.json"

    # Make sure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Check if we have required files
    train_path = data_dir / "train.jsonl.gz"
    train2_path = data_dir / "train2.jsonl.gz"
    legal_abbrev_path = data_dir / "legal_abbreviations.json"
    general_abbrev_path = data_dir / "general_abbreviations.json"

    # Throw exception if required files are not present
    if not train_path.exists():
        raise FileNotFoundError(f"Required training file not found: {train_path}")

    if not train2_path.exists():
        raise FileNotFoundError(f"Required training file not found: {train2_path}")

    if not legal_abbrev_path.exists():
        raise FileNotFoundError(f"Required abbreviations file not found: {legal_abbrev_path}")

    # Load legal abbreviations (required)
    legal_abbreviations = load_abbreviations(legal_abbrev_path)
    general_abbreviations = load_abbreviations(general_abbrev_path)

    # Combine them
    abbreviations = sorted(list(set(legal_abbreviations + general_abbreviations)))

    # Load training data
    train_text1 = load_jsonl_text(train_path, max_samples=args.max_samples)
    train_text2 = load_jsonl_text(train2_path, max_samples=args.max_samples)

    # combined_text = train_text1 + "\n\n" + train_text2
    combined_text = train_text1 + "\n\n" + train_text2

    # Configure memory optimization settings - now enabled by default
    memory_settings = {"memory_efficient": True}
    if args.no_memory_efficient:
        memory_settings["memory_efficient"] = False
        tqdm.write("\n=== Memory-Efficient Training Mode DISABLED ===")
    else:
        tqdm.write("\n=== Using Memory-Efficient Training Mode ===")
        
    # Create trainer with memory settings
    tqdm.write("\n=== Training Default Model ===")
    # Make trainer non-verbose since we'll use our own progress reporting
    trainer = PunktTrainer(verbose=False, **memory_settings)
    
    # Configure memory optimization parameters
    if memory_settings["memory_efficient"]:
        trainer.TYPE_FDIST_MIN_FREQ = args.min_type_freq
        trainer.SENT_STARTER_MIN_FREQ = args.min_starter_freq
        trainer.COLLOC_FDIST_MIN_FREQ = args.min_colloc_freq
        trainer.PRUNE_INTERVAL = args.prune_freq
        
        tqdm.write("Memory optimization parameters:")
        tqdm.write(f"  - min_type_freq: {args.min_type_freq}")
        tqdm.write(f"  - min_starter_freq: {args.min_starter_freq}")
        tqdm.write(f"  - min_colloc_freq: {args.min_colloc_freq}")
        tqdm.write(f"  - prune_interval: {args.prune_freq}")

    # Add abbreviations from loaded files with progress bar
    initial_abbrevs = set()
    tqdm.write(f"\nAdding {len(abbreviations)} abbreviations...")
    
    # Use tqdm for abbreviation loading
    for abbr in tqdm(abbreviations, desc="Adding abbreviations", unit="abbr"):
        trainer._params.abbrev_types.add(abbr.lower())
        initial_abbrevs.add(abbr.lower())

    # Start timing training process
    start_time = time.time()

    # Train using batch mode by default unless disabled
    if args.no_batches:
        # Train on the combined text in one go
        tqdm.write("\nBatch training disabled, processing text in one pass")
        
        # Since the trainer is not verbose, we need to show progress here
        tqdm.write("Starting training...")
        trainer.train(combined_text)
        
    else:
        # Use batch mode by default
        # Convert text to batches
        tqdm.write(f"\nUsing batch training with batch size: {args.batch_size:,} characters")
        tqdm.write("Splitting text into batches...")
        
        # Get batches with progress tracking
        batches = list(PunktTrainer.text_to_batches(combined_text, batch_size=args.batch_size))
        tqdm.write(f"Split text into {len(batches)} batches, starting training")
        
        # Create progress bar for batches
        for i, batch in enumerate(tqdm(batches, desc="Training on batches", unit="batch")):
            # Train on this batch non-verbosely (we're showing progress with tqdm)
            trainer.train(batch, verbose=False, finalize=(i == len(batches)-1))
            
    # Calculate total training time
    training_time = time.time() - start_time
    tqdm.write(f"\nTraining completed in {training_time:.2f} seconds")

    # Print detailed statistics about the trained model
    print_training_details(trainer, initial_abbrevs)

    # Compare storage formats if requested
    if args.compare:
        compare_storage_formats(trainer, models_dir)

    # Save model in the requested format
    tqdm.write(f"\n=== Saving Model to {model_path} ===")
    
    # Use tqdm to display progress indication during saving
    with tqdm(total=1, desc="Saving model", unit="model") as pbar:
        trainer.get_params().save(
            model_path,
            format_type=args.format,
            compression_level=args.level,
            compression_method=args.compression,
        )
        pbar.update(1)
    
    tqdm.write(f"Model saved successfully to {model_path}")

    # Test the model
    tqdm.write("\n=== Testing Model Performance ===")
    tokenizer = PunktSentenceTokenizer(trainer.get_params())

    test_cases = [
        "Dr. Smith went to Washington, D.C. He was very excited about the trip.",
        "The company (Ltd.) was founded in 1997. It has grown significantly since then.",
        "This text contains an ellipsis... And this is a new sentence.",
        "Let me give you an example, e.g. this one. Did you understand it?",
        "The meeting is at 3 p.m. Don't be late!",
        "Under 18 U.S.C. 12, this is a legal citation. The next sentence begins here.",
    ]

    tqdm.write("\nTokenizing sample texts:")
    
    # Use enumeration with tqdm for test case progress
    for i, test in enumerate(tqdm(test_cases, desc="Testing tokenization", unit="text")):
        # Use tqdm.write to avoid overwriting progress bar
        tqdm.write(f"\nText {i+1}: {test}")
        sentences = tokenizer.tokenize(test)
        for j, sentence in enumerate(sentences, 1):
            tqdm.write(f"  Sentence {j}: {sentence.strip()}")

    # Calculate total file size
    model_size_bytes = model_path.stat().st_size
    model_size_kb = model_size_bytes / 1024
    
    tqdm.write("\n" + "=" * 80)
    tqdm.write("TRAINING COMPLETED SUCCESSFULLY")
    tqdm.write(f"Model saved to: {model_path}")
    tqdm.write(f"Model size: {model_size_kb:.2f} KB")
    tqdm.write("=" * 80)


if __name__ == "__main__":
    main()
