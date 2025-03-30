"""
PunktParameters module - Contains the parameters for the Punkt algorithm.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Set, Tuple, Union

from nupunkt.utils.compression import save_compressed_json, load_compressed_json


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
        
    def save(self, file_path: Union[str, Path], compress: bool = True, compression_level: int = 1) -> None:
        """
        Save parameters to a JSON file, optionally with LZMA compression.
        
        Args:
            file_path: The path to save the file to
            compress: Whether to compress the file using LZMA (default: True)
            compression_level: LZMA compression level (0-9), lower is faster but less compressed
        """
        save_compressed_json(
            self.to_json(), 
            file_path, 
            level=compression_level, 
            use_compression=compress
        )
            
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PunktParameters":
        """
        Load parameters from a JSON file, which may be compressed with LZMA.
        
        Args:
            file_path: The path to the file
            
        Returns:
            A new PunktParameters instance
        """
        data = load_compressed_json(file_path)
        return cls.from_json(data)