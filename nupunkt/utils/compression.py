"""
Compression utilities for nupunkt.

This module provides functions for compressing and decompressing data using LZMA.
"""

import json
import lzma
from pathlib import Path
from typing import Any, Dict, Union, Optional


def save_compressed_json(data: Dict[str, Any], file_path: Union[str, Path], 
                          level: int = 1, use_compression: bool = True) -> None:
    """
    Save data as a compressed JSON file using LZMA.
    
    Args:
        data: The data to save
        file_path: The path to save the file to
        level: Compression level (0-9), lower is faster but less compressed
        use_compression: Whether to use compression (if False, saves as regular JSON)
    """
    # Convert Path to string if needed
    if isinstance(file_path, Path):
        file_path = str(file_path)
        
    # Ensure the file path has the correct extension
    if use_compression and not file_path.endswith('.json.xz'):
        file_path = file_path + '.xz' if file_path.endswith('.json') else file_path + '.json.xz'
    elif not use_compression and not file_path.endswith('.json'):
        file_path = file_path + '.json'
    
    # Serialize the data
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    
    if use_compression:
        # Use LZMA compression
        filters = [{"id": lzma.FILTER_LZMA2, "preset": level}]
        with lzma.open(file_path, 'wt', encoding='utf-8', format=lzma.FORMAT_XZ, filters=filters) as f:
            f.write(json_str)
    else:
        # Save as regular JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)


def load_compressed_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load data from a JSON file, which may be compressed with LZMA.
    
    Args:
        file_path: The path to the file
        encoding: The text encoding to use
        
    Returns:
        The loaded data
    """
    # Convert Path to string if needed
    if isinstance(file_path, Path):
        file_path = str(file_path)
    
    try:
        # Try loading as compressed file
        if file_path.endswith('.xz'):
            with lzma.open(file_path, 'rt', encoding=encoding) as f:
                return json.loads(f.read())
        else:
            # Try to detect if it's a compressed file without the extension
            try:
                with lzma.open(file_path, 'rt', encoding=encoding) as f:
                    return json.loads(f.read())
            except (lzma.LZMAError, json.JSONDecodeError):
                # Not a compressed file, load as regular JSON
                with open(file_path, 'r', encoding=encoding) as f:
                    return json.load(f)
    except Exception as e:
        # As a fallback, try regular JSON
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)