#!/usr/bin/env python3
"""
Script to optimize the model storage format for nupunkt.

This script:
1. Loads the current default model
2. Converts it to binary format with compression
3. Shows the size difference
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.models import get_default_model_path
from nupunkt.utils.compression import load_compressed_json, save_binary_model


def optimize_model(format_type, compression_method, compression_level, output_path=None):
    """Optimize the model storage format."""
    # Get the current model path
    current_model_path = get_default_model_path()
    print(f"Current model: {current_model_path}")

    # Load the model data
    print("Loading model data...")
    data = load_compressed_json(current_model_path)

    # Get original file size
    original_size = os.path.getsize(current_model_path)
    print(f"Original size: {original_size / 1024:.2f} KB")

    # Determine output path
    if output_path is None:
        if format_type == "binary":
            output_path = root_dir / "nupunkt" / "models" / "default_model.bin"
        else:
            output_path = current_model_path
    else:
        output_path = Path(output_path)

    # Save in binary format
    print(
        f"Saving in {format_type} format with {compression_method} compression (level {compression_level})..."
    )

    if format_type == "binary":
        save_binary_model(
            data, output_path, compression_method=compression_method, level=compression_level
        )
    else:
        from nupunkt.utils.compression import save_compressed_json

        save_compressed_json(
            data, output_path, level=compression_level, use_compression=(format_type == "json_xz")
        )

    # Get new file size
    new_size = os.path.getsize(output_path)
    print(f"New size: {new_size / 1024:.2f} KB")

    # Calculate compression ratio
    ratio = new_size / original_size
    print(f"Compression ratio: {ratio:.3f} (smaller is better)")

    # Verify the model can be loaded
    print("Verifying model can be loaded...")
    start_time = time.time()
    if format_type == "binary":
        from nupunkt.utils.compression import load_binary_model

        _ = load_binary_model(output_path)
    else:
        _ = load_compressed_json(output_path)
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.5f} seconds")

    return output_path


def main():
    """Run the model optimization."""
    parser = argparse.ArgumentParser(description="Optimize the model storage format for nupunkt")
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
    parser.add_argument(
        "--output", type=str, default=None, help="Custom output path for the optimized model"
    )

    args = parser.parse_args()

    # Optimize the model
    output_path = optimize_model(
        format_type=args.format,
        compression_method=args.compression,
        compression_level=args.level,
        output_path=args.output,
    )

    print(f"Model optimized successfully and saved to: {output_path}")


if __name__ == "__main__":
    main()
