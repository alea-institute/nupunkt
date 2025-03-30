"""
Model package for nupunkt.

This module provides functionality for loading the default pre-trained model.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Union

from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer


def get_default_model_path() -> Path:
    """
    Get the path to the default pre-trained model.
    
    The function searches for both compressed (.json.xz) and uncompressed (.json) versions
    of the default model, preferring the compressed version if available.
    
    Returns:
        Path: The path to the default model file
    """
    base_dir = Path(__file__).parent
    
    # Check for compressed model first
    compressed_path = base_dir / "default_model.json.xz"
    if compressed_path.exists():
        return compressed_path
    
    # Fall back to uncompressed model
    return base_dir / "default_model.json"


def load_default_model() -> PunktSentenceTokenizer:
    """
    Load the default pre-trained model.
    
    Returns:
        PunktSentenceTokenizer: A tokenizer initialized with the default model
    """
    model_path = get_default_model_path()
    return PunktSentenceTokenizer.load(model_path)


def compress_default_model(output_path: Optional[Union[str, Path]] = None, 
                           compression_level: int = 1) -> Path:
    """
    Compress the default model using LZMA.
    
    Args:
        output_path: Optional path to save the compressed model. If None,
                    saves to the default location.
        compression_level: LZMA compression level (0-9). Lower means faster 
                          compression but larger file size.
                          
    Returns:
        Path: The path to the compressed model file
    """
    from nupunkt.utils.compression import save_compressed_json, load_compressed_json
    
    # Get the path to the default model
    default_path = Path(__file__).parent / "default_model.json"
    
    # Determine output path
    if output_path is None:
        output_path = default_path.with_suffix(".json.xz")
    else:
        output_path = Path(output_path)
    
    # Load the model data
    try:
        with open(default_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Try loading using the utility function (in case it's already compressed)
        data = load_compressed_json(default_path)
    
    # Save compressed model
    save_compressed_json(data, output_path, level=compression_level, use_compression=True)
    
    return output_path