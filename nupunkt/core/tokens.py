"""
Token module for nupunkt.

This module provides the PunktToken class, which represents a token
in the Punkt algorithm and calculates various derived properties.
"""

import re
from functools import lru_cache
from typing import Dict, Tuple

# Compiled regex patterns for better performance
_RE_NON_WORD_DOT = re.compile(r"[^\w.]")
_RE_NUMBER = re.compile(r"^-?[\.,]?\d[\d,\.-]*\.?$")
_RE_ELLIPSIS = re.compile(r"\.\.+$")
_RE_SPACED_ELLIPSIS = re.compile(r"\.\s+\.\s+\.")
_RE_INITIAL = re.compile(r"[^\W\d]\.")
_RE_ALPHA = re.compile(r"[^\W\d]+")
_RE_NON_PUNCT = re.compile(r"[^\W\d]")


# LRU-cached functions for token classification to improve performance
# Use a smaller cache for common tokens only
@lru_cache(maxsize=500)
def _check_is_ellipsis(tok: str) -> bool:
    """
    Cached function to check if a token represents an ellipsis.

    Args:
        tok: The token to check

    Returns:
        True if the token is an ellipsis, False otherwise
    """
    # Check for standard ellipsis (... or longer)
    if bool(_RE_ELLIPSIS.search(tok)):
        return True

    # Check for unicode ellipsis
    if tok == "\u2026" or tok.endswith("\u2026"):
        return True

    # Check for spaced ellipsis (. . ., . .  ., etc.)
    return bool(_RE_SPACED_ELLIPSIS.search(tok))


@lru_cache(maxsize=500)
def _check_is_initial(tok: str) -> bool:
    """
    Cached function to check if a token is an initial.

    Args:
        tok: The token to check

    Returns:
        True if the token is an initial, False otherwise
    """
    return bool(_RE_INITIAL.fullmatch(tok))


@lru_cache(maxsize=1000)
def _check_is_alpha(tok: str) -> bool:
    """
    Cached function to check if a token is alphabetic.

    Args:
        tok: The token to check

    Returns:
        True if the token is alphabetic, False otherwise
    """
    return bool(_RE_ALPHA.fullmatch(tok))


@lru_cache(maxsize=1000)
def _check_is_non_punct(typ: str) -> bool:
    """
    Cached function to check if a token type contains non-punctuation.

    Args:
        typ: The token type to check

    Returns:
        True if the token type contains non-punctuation, False otherwise
    """
    return bool(_RE_NON_PUNCT.search(typ))


@lru_cache(maxsize=2000)  # Increased cache size for token types
def _get_token_type(tok: str) -> str:
    """
    Get the normalized type of a token (cached for better performance).

    Args:
        tok: The token string

    Returns:
        The normalized type (##number## for numbers, lowercase form for others)
    """
    # Normalize numbers
    if _RE_NUMBER.match(tok):
        return "##number##"
    return tok.lower()


@lru_cache(maxsize=1000)
def _get_type_no_period(type_str: str) -> str:
    """Get the token type without a trailing period (cached)."""
    return type_str[:-1] if type_str.endswith(".") and len(type_str) > 1 else type_str


# Module-level cache for PunktToken instances
_token_instance_cache: Dict[Tuple[str, bool, bool], "PunktToken"] = {}
_TOKEN_CACHE_SIZE = 2000  # Increased from original 1000


def create_punkt_token(tok: str, parastart: bool = False, linestart: bool = False) -> "PunktToken":
    """
    Factory function to create PunktToken instances with caching.

    Args:
        tok: Token text
        parastart: Whether the token starts a paragraph
        linestart: Whether the token starts a line

    Returns:
        A new or cached PunktToken instance
    """
    # Only cache smaller tokens (most common case)
    if len(tok) < 15:
        cache_key = (tok, parastart, linestart)
        token = _token_instance_cache.get(cache_key)
        if token is not None:
            return token

        token = PunktToken(tok, parastart, linestart)

        # Add to cache if not full
        if len(_token_instance_cache) < _TOKEN_CACHE_SIZE:
            _token_instance_cache[cache_key] = token
        return token

    # For longer tokens, just create a new instance
    return PunktToken(tok, parastart, linestart)


class PunktToken:
    """
    Represents a token in the Punkt algorithm.

    This class contains the token string and various properties and flags that
    indicate its role in sentence boundary detection.

    Uses __slots__ for memory efficiency, especially for large documents
    where millions of token instances are created.
    """

    __slots__ = (
        "tok",
        "parastart",
        "linestart",
        "sentbreak",
        "abbr",
        "ellipsis",
        "period_final",
        "type",
        "valid_abbrev_candidate",
        "_first_upper",
        "_first_lower",
        "_type_no_period",
        "_type_no_sentperiod",
        "_is_ellipsis",
        "_is_number",
        "_is_initial",
        "_is_alpha",
        "_is_non_punct",
    )

    # Define allowed characters for fast punctuation check (alphanumeric + period)
    _ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")

    def __init__(self, tok: str, parastart: bool = False, linestart: bool = False) -> None:
        """
        Initialize a new PunktToken instance.

        Args:
            tok: The token string
            parastart: Whether this token starts a paragraph
            linestart: Whether this token starts a line
        """
        # Initialize base attributes
        self.tok = tok
        self.parastart = parastart
        self.linestart = linestart
        self.sentbreak = False
        self.abbr = False
        self.ellipsis = False

        # Initialize computed attributes
        self.period_final = tok.endswith(".")
        self.type = _get_token_type(tok)

        # Pre-compute frequently accessed properties
        tok_len = len(tok)
        self._first_upper = tok_len > 0 and tok[0].isupper()
        self._first_lower = tok_len > 0 and tok[0].islower()

        # Initialize lazily computed properties (will be set on first access)
        self._type_no_period = None
        self._type_no_sentperiod = None
        self._is_ellipsis = None
        self._is_number = None
        self._is_initial = None
        self._is_alpha = None
        self._is_non_punct = None

        # Fast check for invalid characters (non-alphanumeric and non-period)
        has_invalid_char = False
        for c in tok:
            if c not in self._ALLOWED_CHARS:
                has_invalid_char = True
                break

        if self.period_final and not has_invalid_char:
            # For tokens with internal periods (like U.S.C), get non-period chars
            # Use more efficient counting method
            alpha_count = 0
            digit_count = 0
            for c in tok:
                if c != ".":
                    if c.isalpha():
                        alpha_count += 1
                    elif c.isdigit():
                        digit_count += 1

            self.valid_abbrev_candidate = (
                self.type != "##number##" and alpha_count >= digit_count and alpha_count > 0
            )
        else:
            self.valid_abbrev_candidate = False

        # If token has a period but isn't valid candidate, reset abbr flag
        if self.period_final and not self.valid_abbrev_candidate:
            self.abbr = False

    @property
    def type_no_period(self) -> str:
        """Get the token type without a trailing period."""
        if self._type_no_period is None:
            self._type_no_period = _get_type_no_period(self.type)
        return self._type_no_period

    @property
    def type_no_sentperiod(self) -> str:
        """Get the token type without a sentence-final period."""
        if self._type_no_sentperiod is None:
            self._type_no_sentperiod = self.type_no_period if self.sentbreak else self.type
        return self._type_no_sentperiod

    @property
    def first_upper(self) -> bool:
        """Check if the first character of the token is uppercase."""
        return self._first_upper

    @property
    def first_lower(self) -> bool:
        """Check if the first character of the token is lowercase."""
        return self._first_lower

    @property
    def first_case(self) -> str:
        """Get the case of the first character of the token."""
        if self.first_lower:
            return "lower"
        if self.first_upper:
            return "upper"
        return "none"

    @property
    def is_ellipsis(self) -> bool:
        """
        Check if the token is an ellipsis (any of the following patterns):
        1. Multiple consecutive periods (..., ......)
        2. Unicode ellipsis character (…)
        3. Periods separated by spaces (. . ., .  .  .)
        """
        if self._is_ellipsis is None:
            self._is_ellipsis = _check_is_ellipsis(self.tok)
        return self._is_ellipsis

    @property
    def is_number(self) -> bool:
        """Check if the token is a number."""
        if self._is_number is None:
            self._is_number = self.type.startswith("##number##")
        return self._is_number

    @property
    def is_initial(self) -> bool:
        """Check if the token is an initial (single letter followed by a period)."""
        if self._is_initial is None:
            self._is_initial = _check_is_initial(self.tok)
        return self._is_initial

    @property
    def is_alpha(self) -> bool:
        """Check if the token is alphabetic (contains only letters)."""
        if self._is_alpha is None:
            self._is_alpha = _check_is_alpha(self.tok)
        return self._is_alpha

    @property
    def is_non_punct(self) -> bool:
        """Check if the token contains non-punctuation characters."""
        if self._is_non_punct is None:
            self._is_non_punct = _check_is_non_punct(self.type)
        return self._is_non_punct

    def __str__(self) -> str:
        """Get a string representation of the token with annotation flags."""
        s = self.tok
        if self.abbr:
            s += "<A>"
        if self.ellipsis:
            s += "<E>"
        if self.sentbreak:
            s += "<S>"
        return s

    def __repr__(self) -> str:
        """Get a detailed string representation of the token."""
        return (
            f"PunktToken(tok='{self.tok}', parastart={self.parastart}, "
            f"linestart={self.linestart}, sentbreak={self.sentbreak}, "
            f"abbr={self.abbr}, ellipsis={self.ellipsis})"
        )
