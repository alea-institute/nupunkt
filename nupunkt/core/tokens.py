"""
Token module for nupunkt.

This module provides the PunktToken class, which represents a token
in the Punkt algorithm and calculates various derived properties.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

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
@lru_cache(maxsize=250)
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
    if _RE_SPACED_ELLIPSIS.search(tok):
        return True

    return False


@lru_cache(maxsize=250)
def _check_is_initial(tok: str) -> bool:
    """
    Cached function to check if a token is an initial.
    
    Args:
        tok: The token to check
        
    Returns:
        True if the token is an initial, False otherwise
    """
    return bool(_RE_INITIAL.fullmatch(tok))


@lru_cache(maxsize=500)
def _check_is_alpha(tok: str) -> bool:
    """
    Cached function to check if a token is alphabetic.
    
    Args:
        tok: The token to check
        
    Returns:
        True if the token is alphabetic, False otherwise
    """
    return bool(_RE_ALPHA.fullmatch(tok))


@lru_cache(maxsize=500)
def _check_is_non_punct(typ: str) -> bool:
    """
    Cached function to check if a token type contains non-punctuation.
    
    Args:
        typ: The token type to check
        
    Returns:
        True if the token type contains non-punctuation, False otherwise
    """
    return bool(_RE_NON_PUNCT.search(typ))


@dataclass
class PunktToken:
    """
    Represents a token in the Punkt algorithm.

    This class contains the token string and various properties and flags that
    indicate its role in sentence boundary detection.
    """

    tok: str
    parastart: bool = False
    linestart: bool = False
    sentbreak: bool = False
    abbr: bool = False
    ellipsis: bool = False

    # Derived attributes (set in __post_init__)
    period_final: bool = field(init=False)
    type: str = field(init=False)
    valid_abbrev_candidate: bool = field(init=False)
    
    # Pre-computed properties
    _first_upper: bool = field(init=False, repr=False)
    _first_lower: bool = field(init=False, repr=False)
    
    # Define allowed characters for fast punctuation check (alphanumeric + period)
    _ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")
    
    # Class-level token cache for common tokens
    _token_cache = {}
    _TOKEN_CACHE_SIZE = 1000

    def __post_init__(self) -> None:
        """
        Initialize derived attributes after instance creation.

        This method calculates:
        - Whether the token ends with a period
        - The token type (normalized form)
        - Whether the token is a valid abbreviation candidate
        - Pre-computes frequently accessed properties
        """
        # Try to get from cache for common tokens
        if len(self.tok) < 15:  # Only cache smaller tokens
            cache_key = (self.tok, self.parastart, self.linestart)
            cached_attrs = self._token_cache.get(cache_key)
            if cached_attrs is not None:
                self.period_final = cached_attrs[0]
                self.type = cached_attrs[1]
                self.valid_abbrev_candidate = cached_attrs[2]
                self._first_upper = cached_attrs[3]
                self._first_lower = cached_attrs[4]
                return
        
        # Basic properties
        self.period_final = self.tok.endswith(".")
        self.type = self._get_type(self.tok)
        
        # Pre-compute frequently accessed properties
        tok_len = len(self.tok)
        self._first_upper = tok_len > 0 and self.tok[0].isupper()
        self._first_lower = tok_len > 0 and self.tok[0].islower()
        
        # Fast check for invalid characters (non-alphanumeric and non-period)
        # This is much faster than regex for common cases
        has_invalid_char = False
        for c in self.tok:
            if c not in self._ALLOWED_CHARS:
                has_invalid_char = True
                break
        
        if self.period_final and not has_invalid_char:
            # For tokens with internal periods (like U.S.C), get non-period chars
            # Use more efficient counting method
            alpha_count = 0
            digit_count = 0
            for c in self.tok:
                if c != '.':
                    if c.isalpha():
                        alpha_count += 1
                    elif c.isdigit():
                        digit_count += 1
            
            self.valid_abbrev_candidate = (
                not (self.type == "##number##")
                and alpha_count >= digit_count
                and alpha_count > 0
            )
        else:
            self.valid_abbrev_candidate = False
        
        # If token has a period but isn't valid candidate, reset abbr flag
        if self.period_final and not self.valid_abbrev_candidate:
            self.abbr = False
            
        # Cache small, common tokens
        if len(self.tok) < 15 and len(self._token_cache) < self._TOKEN_CACHE_SIZE:
            cache_key = (self.tok, self.parastart, self.linestart)
            self._token_cache[cache_key] = (
                self.period_final,
                self.type,
                self.valid_abbrev_candidate,
                self._first_upper,
                self._first_lower
            )

    @staticmethod
    @lru_cache(maxsize=1000)  # Cache the most frequent 1,000 token types
    def _get_type(tok: str) -> str:
        """
        Get the normalized type of a token.

        Args:
            tok: The token string

        Returns:
            The normalized type (##number## for numbers, lowercase form for others)
        """
        # Normalize numbers
        if _RE_NUMBER.match(tok):
            return "##number##"
        return tok.lower()

    @property
    def type_no_period(self) -> str:
        """Get the token type without a trailing period."""
        return self.type[:-1] if self.type.endswith(".") and len(self.type) > 1 else self.type

    @property
    def type_no_sentperiod(self) -> str:
        """Get the token type without a sentence-final period."""
        return self.type_no_period if self.sentbreak else self.type

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
        return _check_is_ellipsis(self.tok)

    @property
    def is_number(self) -> bool:
        """Check if the token is a number."""
        return self.type.startswith("##number##")

    @property
    def is_initial(self) -> bool:
        """Check if the token is an initial (single letter followed by a period)."""
        return _check_is_initial(self.tok)

    @property
    def is_alpha(self) -> bool:
        """Check if the token is alphabetic (contains only letters)."""
        return _check_is_alpha(self.tok)

    @property
    def is_non_punct(self) -> bool:
        """Check if the token contains non-punctuation characters."""
        return _check_is_non_punct(self.type)

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