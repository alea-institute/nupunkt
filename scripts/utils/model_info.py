#!/usr/bin/env python3
"""
Script to display information about nupunkt models.

This script:
1. Loads a model from a specified file or the default model
2. Displays key information about the model
3. Provides options to convert between formats or display details

Usage:
    python model_info.py --input model.bin
    python model_info.py --input model.bin --stats
    python model_info.py --input model.bin --convert output.json --format json
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.utils.compression import (
    load_binary_model,
    load_compressed_json,
    save_binary_model,
    save_compressed_json,
)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    kb = size_bytes / 1024
    if kb < 1000:
        return f"{kb:.2f} KB"
    mb = kb / 1024
    return f"{mb:.2f} MB"


def load_model(input_path: Path) -> Dict[str, Any]:
    """
    Load a model from the specified path.

    Args:
        input_path: Path to the model file

    Returns:
        The model data as a dictionary
    """
    start_time = time.time()

    # Load the model data
    if str(input_path).endswith(".bin"):
        data = load_binary_model(input_path)
        model_format = "binary"
    else:
        data = load_compressed_json(input_path)
        if str(input_path).endswith(".json.xz"):
            model_format = "json_xz"
        else:
            model_format = "json"

    load_time = time.time() - start_time

    # Handle the trainer format if present
    if "parameters" in data:
        params = data["parameters"]
        trainer_params = {k: v for k, v in data.items() if k != "parameters"}
    else:
        params = data
        trainer_params = {}

    return {
        "path": input_path,
        "format": model_format,
        "size": os.path.getsize(input_path),
        "load_time": load_time,
        "params": params,
        "trainer_params": trainer_params,
    }


def display_model_info(model_data: Dict[str, Any], show_stats: bool = False) -> None:
    """
    Display information about the model.

    Args:
        model_data: Dictionary containing model information
        show_stats: Whether to display detailed statistics
    """
    print("\n=== Model Information ===")
    print(f"File: {model_data['path']}")
    print(f"Format: {model_data['format']}")
    print(f"Size: {format_size(model_data['size'])}")
    print(f"Load time: {model_data['load_time']:.3f} seconds")

    params = model_data["params"]
    print("\nParameters:")
    print(f"  Abbreviation types: {len(params.get('abbrev_types', []))}")
    print(f"  Collocations: {len(params.get('collocations', []))}")
    print(f"  Sentence starters: {len(params.get('sent_starters', []))}")
    print(f"  Orthographic context: {len(params.get('ortho_context', {}))}")

    trainer_params = model_data["trainer_params"]
    if trainer_params:
        print("\nTrainer parameters:")
        for key, value in trainer_params.items():
            # Skip large collections like common_abbrevs
            if isinstance(value, (list, dict, set)) and len(value) > 10:
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

    if show_stats:
        display_model_stats(model_data)


def display_model_stats(model_data: Dict[str, Any]) -> None:
    """
    Display detailed statistics about the model.

    Args:
        model_data: Dictionary containing model information
    """
    params = model_data["params"]

    print("\n=== Model Statistics ===")

    # Abbreviation statistics
    abbrev_types = params.get("abbrev_types", [])
    if abbrev_types:
        print(f"\nAbbreviation types ({len(abbrev_types)}):")
        print("  Most common types by first letter:")
        first_letter_counts = Counter(abbr[0] if abbr else "" for abbr in abbrev_types)
        for letter, count in first_letter_counts.most_common(10):
            print(f"    {letter}: {count}")

        print("\n  Abbreviation examples:")
        for abbr in sorted(list(abbrev_types))[:10]:
            print(f"    {abbr}")

    # Collocation statistics
    collocations = params.get("collocations", [])
    if collocations:
        print(f"\nCollocations ({len(collocations)}):")
        print("  Examples:")
        for w1, w2 in sorted(collocations)[:10]:
            print(f"    {w1} {w2}")

    # Sentence starter statistics
    sent_starters = params.get("sent_starters", [])
    if sent_starters:
        print(f"\nSentence starters ({len(sent_starters)}):")
        print("  Examples:")
        for starter in sorted(sent_starters)[:10]:
            print(f"    {starter}")

    # Orthographic context statistics
    ortho_context = params.get("ortho_context", {})
    if ortho_context:
        print(f"\nOrthographic context ({len(ortho_context)}):")
        print("  Most common flag values:")
        flags = Counter(ortho_context.values())
        for flag, count in flags.most_common(5):
            print(f"    Flag {flag}: {count} types")


def convert_model(
    model_data: Dict[str, Any],
    output_path: Path,
    format_type: str,
    compression_method: str,
    compression_level: int,
) -> None:
    """
    Convert a model to a different format.

    Args:
        model_data: Dictionary containing model information
        output_path: Path to save the converted model to
        format_type: Output format type ('binary', 'json_xz', 'json')
        compression_method: Compression method for binary format
        compression_level: Compression level (0-9)
    """
    print("\n=== Converting Model ===")
    print(f"Source: {model_data['path']}")
    print(f"Target: {output_path}")
    print(f"Format: {format_type}")
    if format_type == "binary":
        print(f"Compression: {compression_method} (level {compression_level})")
    elif format_type == "json_xz":
        print(f"Compression level: {compression_level}")

    start_time = time.time()

    # Reconstruct original data format
    if model_data["trainer_params"]:
        # This was a trainer format
        data = model_data["trainer_params"].copy()
        data["parameters"] = model_data["params"]
    else:
        # This was a direct format
        data = model_data["params"]

    # Convert the model
    if format_type == "binary":
        save_binary_model(
            data, output_path, compression_method=compression_method, level=compression_level
        )
    else:
        save_compressed_json(
            data, output_path, level=compression_level, use_compression=(format_type == "json_xz")
        )

    convert_time = time.time() - start_time
    print(f"Conversion completed in {convert_time:.3f} seconds")

    # Get output file size
    output_size = os.path.getsize(output_path)
    print(f"Output file size: {format_size(output_size)}")

    # Show compression ratio
    ratio = output_size / model_data["size"]
    print(f"Size ratio: {ratio:.3f} (< 1 means smaller)")

    print(f"Model successfully converted and saved to {output_path}")


def main():
    """Process command-line arguments and run the requested operations."""
    parser = argparse.ArgumentParser(description="Display information about nupunkt models")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input model file (default: use the default model)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display detailed statistics about the model"
    )
    parser.add_argument(
        "--convert",
        type=str,
        default=None,
        help="Convert the model to a different format and save to the specified path",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="binary",
        choices=["json", "json_xz", "binary"],
        help="Format to convert the model to",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="lzma",
        choices=["none", "zlib", "lzma", "gzip"],
        help="Compression method for binary format",
    )
    parser.add_argument("--level", type=int, default=6, help="Compression level (0-9)")

    args = parser.parse_args()

    # Use default model if input not specified
    if args.input:
        input_path = Path(args.input)
    else:
        from nupunkt.models import get_default_model_path

        input_path = get_default_model_path()

    # Load the model
    try:
        model_data = load_model(input_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Display model information
    display_model_info(model_data, args.stats)

    # Convert the model if requested
    if args.convert:
        output_path = Path(args.convert)

        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)

        try:
            convert_model(
                model_data=model_data,
                output_path=output_path,
                format_type=args.format,
                compression_method=args.compression,
                compression_level=args.level,
            )
        except Exception as e:
            print(f"Error converting model: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
