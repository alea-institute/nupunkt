"""
PunktParameters module - Contains the parameters for the Punkt algorithm.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Pattern, Set, Tuple, Union

from nupunkt.utils.compression import (
    load_compressed_json,
    save_binary_model,
    save_compressed_json,
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

    # Cached regex patterns for efficient lookups
    _abbrev_pattern: Optional[Pattern] = field(default=None, repr=False)
    _sent_starter_pattern: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize any derived attributes after instance creation."""
        # Patterns will be compiled on first use
        # Initialize frozen sets to empty frozensets
        self._frozen_abbrev_types = frozenset()
        self._frozen_collocations = frozenset()
        self._frozen_sent_starters = frozenset()

    def get_abbrev_pattern(self) -> Pattern:
        """
        Get a compiled regex pattern for matching abbreviations.

        The pattern is compiled on first use and cached for subsequent calls.

        Returns:
            A compiled regex pattern that matches any abbreviation in abbrev_types
        """
        if not self._abbrev_pattern or len(self._abbrev_pattern.pattern) == 0:
            if not self.abbrev_types:
                # If no abbreviations, create a pattern that will never match
                self._abbrev_pattern = re.compile(r"^$")
            else:
                # Escape abbreviations and sort by length (longest first) to ensure proper matching
                escaped_abbrevs = [re.escape(abbr) for abbr in self.abbrev_types]
                sorted_abbrevs = sorted(escaped_abbrevs, key=len, reverse=True)
                pattern = r"^(?:" + "|".join(sorted_abbrevs) + r")$"
                self._abbrev_pattern = re.compile(pattern, re.IGNORECASE)
        return self._abbrev_pattern

    def get_sent_starter_pattern(self) -> Pattern:
        """
        Get a compiled regex pattern for matching sentence starters.

        The pattern is compiled on first use and cached for subsequent calls.

        Returns:
            A compiled regex pattern that matches any sentence starter
        """
        if not self._sent_starter_pattern or len(self._sent_starter_pattern.pattern) == 0:
            if not self.sent_starters:
                # If no sentence starters, create a pattern that will never match
                self._sent_starter_pattern = re.compile(r"^$")
            else:
                # Escape sentence starters and sort by length (longest first)
                escaped_starters = [re.escape(starter) for starter in self.sent_starters]
                sorted_starters = sorted(escaped_starters, key=len, reverse=True)
                pattern = r"^(?:" + "|".join(sorted_starters) + r")$"
                self._sent_starter_pattern = re.compile(pattern, re.IGNORECASE)
        return self._sent_starter_pattern

    def add_ortho_context(self, typ: str, flag: int) -> None:
        """
        Add an orthographic context flag to a token type.

        Args:
            typ: The token type
            flag: The orthographic context flag
        """
        self.ortho_context[typ] |= flag

    def add_abbreviation(self, abbrev: str) -> None:
        """
        Add a single abbreviation and invalidate the cached pattern.

        Args:
            abbrev: The abbreviation to add
        """
        self.abbrev_types.add(abbrev)
        self._abbrev_pattern = None

    def add_sent_starter(self, starter: str) -> None:
        """
        Add a single sentence starter and invalidate the cached pattern.

        Args:
            starter: The sentence starter to add
        """
        self.sent_starters.add(starter)
        self._sent_starter_pattern = None

    def invalidate_patterns(self) -> None:
        """Invalidate cached regex patterns when sets are modified."""
        self._abbrev_pattern = None
        self._sent_starter_pattern = None

    def freeze_sets(self) -> None:
        """
        Freeze the mutable sets to create immutable frozensets for faster lookups.

        Call this method after training is complete to optimize for inference speed.
        """
        self._frozen_abbrev_types = frozenset(self.abbrev_types)
        self._frozen_collocations = frozenset(self.collocations)
        self._frozen_sent_starters = frozenset(self.sent_starters)

    def update_abbrev_types(self, abbrevs: Set[str]) -> None:
        """
        Update abbreviation types and invalidate the cached pattern.

        Args:
            abbrevs: Set of abbreviations to add
        """
        self.abbrev_types.update(abbrevs)
        self._abbrev_pattern = None

    def update_sent_starters(self, starters: Set[str]) -> None:
        """
        Update sentence starters and invalidate the cached pattern.

        Args:
            starters: Set of sentence starters to add
        """
        self.sent_starters.update(starters)
        self._sent_starter_pattern = None

    def to_json(self) -> Dict[str, Any]:
        """Convert parameters to a JSON-serializable dictionary."""
        return {
            "abbrev_types": sorted(self.abbrev_types),
            "collocations": sorted([[c[0], c[1]] for c in self.collocations]),
            "sent_starters": sorted(self.sent_starters),
            "ortho_context": dict(self.ortho_context.items()),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PunktParameters":
        """Create a PunktParameters instance from a JSON dictionary."""
        params = cls()
        params.abbrev_types = set(data.get("abbrev_types", []))
        params.collocations = {tuple(c) for c in data.get("collocations", [])}
        params.sent_starters = set(data.get("sent_starters", []))
        params.ortho_context = defaultdict(int)
        for k, v in data.get("ortho_context", {}).items():
            params.ortho_context[k] = int(v)  # Ensure value is int

        # Don't pre-compile patterns by default
        # Direct set lookup is faster based on benchmarks

        # Create frozen sets for faster lookups during inference
        params.freeze_sets()

        return params

    def save(
        self,
        file_path: Union[str, Path],
        format_type: str = "json_xz",
        compression_level: int = 1,
        compression_method: str = "zlib",
    ) -> None:
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
                level=compression_level,
            )
        else:
            save_compressed_json(
                self.to_json(),
                file_path,
                level=compression_level,
                use_compression=(format_type == "json_xz"),
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

        # Handle binary format which is wrapped in a "parameters" key
        if "parameters" in data:
            return cls.from_json(data["parameters"])

        return cls.from_json(data)
