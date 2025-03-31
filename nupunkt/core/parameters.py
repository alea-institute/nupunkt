"""
PunktParameters module - Contains the parameters for the Punkt algorithm.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Set, Tuple, Union

from nupunkt.utils.compression import (
    save_compressed_json, 
    load_compressed_json, 
    save_binary_model,
    load_binary_model
)


@dataclass
class PunktParameters:
    """
    Stores the parameters that Punkt uses for sentence boundary detection.
    
    This includes:
    - Abbreviation types
    - Collocations
    - Sentence starters
    - Orthographic context
    """
    abbrev_types: Set[str] = field(default_factory=set)
    collocations: Set[Tuple[str, str]] = field(default_factory=set)
    sent_starters: Set[str] = field(default_factory=set)
    ortho_context: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_ortho_context(self, typ: str, flag: int) -> None:
        """
        Add an orthographic context flag to a token type.
        
        Args:
            typ: The token type
            flag: The orthographic context flag
        """
        self.ortho_context[typ] |= flag
        
    def to_json(self) -> Dict[str, Any]:
        """Convert parameters to a JSON-serializable dictionary."""
        return {
            "abbrev_types": sorted(list(self.abbrev_types)),
            "collocations": sorted([[c[0], c[1]] for c in self.collocations]),
            "sent_starters": sorted(list(self.sent_starters)),
            "ortho_context": {k: v for k, v in self.ortho_context.items()}
        }
        
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PunktParameters":
        """Create a PunktParameters instance from a JSON dictionary."""
        params = cls()
        params.abbrev_types = set(data.get("abbrev_types", []))
        params.collocations = set(tuple(c) for c in data.get("collocations", []))
        params.sent_starters = set(data.get("sent_starters", []))
        params.ortho_context = defaultdict(int)
        for k, v in data.get("ortho_context", {}).items():
            params.ortho_context[k] = int(v)  # Ensure value is int
        return params
        
    def save(self, file_path: Union[str, Path], 
             format_type: str = "json_xz", 
             compression_level: int = 1, 
             compression_method: str = "zlib") -> None:
        """
        Save parameters to a file using the specified format and compression.
        
        Args:
            file_path: The path to save the file to
            format_type: The format type to use ('json', 'json_xz', 'binary')
            compression_level: Compression level (0-9), lower is faster but less compressed
            compression_method: Compression method for binary format ('none', 'zlib', 'lzma', 'gzip')
        """
        if format_type == "binary":
            save_binary_model(
                self.to_json(), 
                file_path, 
                compression_method=compression_method,
                level=compression_level
            )
        else:
            save_compressed_json(
                self.to_json(), 
                file_path, 
                level=compression_level, 
                use_compression=(format_type == "json_xz")
            )
            
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PunktParameters":
        """
        Load parameters from a file in any supported format.
        
        This method automatically detects the file format based on extension
        and loads the parameters accordingly.
        
        Args:
            file_path: The path to the file
            
        Returns:
            A new PunktParameters instance
        """
        # The load_compressed_json function will try to detect if it's a binary file
        data = load_compressed_json(file_path)
        return cls.from_json(data)