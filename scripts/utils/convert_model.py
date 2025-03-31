#!/usr/bin/env python3
"""
Script to convert nupunkt models between different storage formats.

This script:
1. Loads a model from a specified file or the default model
2. Converts it to a different format (binary, json_xz, json)
3. Saves it to a specified location

Usage:
    python convert_model.py --input model.json.xz --output model.bin --format binary --compression zlib
"""

import sys
import os
import time
from pathlib import Path
import argparse

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

def convert_model(input_path: Path, output_path: Path, format_type: str, 
                 compression_method: str, compression_level: int) -> None:
    """
    Convert a model from one format to another.
    
    Args:
        input_path: Path to the input model file
        output_path: Path to save the converted model to
        format_type: Output format type ('binary', 'json_xz', 'json')
        compression_method: Compression method for binary format
        compression_level: Compression level (0-9)
    """
    print(f"Loading model from {input_path}...")
    start_time = time.time()
    
    # Load the model data
    if str(input_path).endswith('.bin'):
        data = load_binary_model(input_path)
    else:
        data = load_compressed_json(input_path)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.3f} seconds")
    
    # Get input file size
    input_size = os.path.getsize(input_path)
    print(f"Input file size: {input_size/1024:.2f} KB")
    
    # Convert the model
    print(f"Converting to {format_type} format with {compression_method} compression (level {compression_level})...")
    start_time = time.time()
    
    if format_type == 'binary':
        save_binary_model(
            data, 
            output_path, 
            compression_method=compression_method,
            level=compression_level
        )
    else:
        save_compressed_json(
            data, 
            output_path, 
            level=compression_level, 
            use_compression=(format_type == 'json_xz')
        )
    
    convert_time = time.time() - start_time
    print(f"Conversion completed in {convert_time:.3f} seconds")
    
    # Get output file size
    output_size = os.path.getsize(output_path)
    print(f"Output file size: {output_size/1024:.2f} KB")
    
    # Show compression ratio
    ratio = output_size / input_size
    print(f"Compression ratio: {ratio:.3f} (smaller is better)")
    
    # Verify the model can be loaded
    print("Verifying model can be loaded...")
    start_time = time.time()
    
    if format_type == 'binary':
        _ = load_binary_model(output_path)
    else:
        _ = load_compressed_json(output_path)
    
    verify_time = time.time() - start_time
    print(f"Model verified in {verify_time:.3f} seconds")
    
    print(f"Model successfully converted and saved to {output_path}")

def main():
    """Run the model conversion."""
    parser = argparse.ArgumentParser(description="Convert nupunkt models between different formats")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to the input model file (default: use the default model)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the converted model to")
    parser.add_argument("--format", type=str, default="binary", 
                        choices=["json", "json_xz", "binary"],
                        help="Format to convert the model to")
    parser.add_argument("--compression", type=str, default="zlib", 
                        choices=["none", "zlib", "lzma", "gzip"],
                        help="Compression method for binary format")
    parser.add_argument("--level", type=int, default=6, 
                        help="Compression level (0-9)")
    
    args = parser.parse_args()
    
    # Use default model if input not specified
    if args.input:
        input_path = Path(args.input)
    else:
        from nupunkt.models import get_default_model_path
        input_path = get_default_model_path()
    
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Convert the model
    convert_model(
        input_path=input_path,
        output_path=output_path,
        format_type=args.format,
        compression_method=args.compression,
        compression_level=args.level
    )

if __name__ == "__main__":
    main()