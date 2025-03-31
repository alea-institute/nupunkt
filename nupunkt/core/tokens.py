"""
Token module for nupunkt.

This module provides the PunktToken class, which represents a token
in the Punkt algorithm and calculates various derived properties.
"""

import re
from dataclasses import dataclass, field

# Compiled regex patterns for better performance
_RE_NON_WORD_DOT = re.compile(r"[^\w.]")
_RE_NUMBER = re.compile(r"^-?[\.,]?\d[\d,\.-]*\.?$")
_RE_ELLIPSIS = re.compile(r"\.\.+$")
_RE_SPACED_ELLIPSIS = re.compile(r"\.\s+\.\s+\.")
_RE_INITIAL = re.compile(r"[^\W\d]\.")
_RE_ALPHA = re.compile(r"[^\W\d]+")
_RE_NON_PUNCT = re.compile(r"[^\W\d]")


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

    def __post_init__(self) -> None:
        """
        Initialize derived attributes after instance creation.

        This method calculates:
        - Whether the token ends with a period
        - The token type (normalized form)
        - Whether the token is a valid abbreviation candidate
        """
        self.period_final = self.tok.endswith(".")
        self.type = self._get_type(self.tok)

        # Determine if token could be a valid abbreviation candidate
        # Rules:
        # 1. Must end with a period
        # 2. Only alphanumeric characters and periods allowed (no other special punctuation)
        # 3. Not a pure number
        # 4. Must have at least as many alphabet chars as digits

        # For tokens with internal periods (like U.S.C), get the non-period characters for counting
        token_no_periods = self.tok.replace(".", "")

        # Count alphabet and digit characters in the non-period version
        alpha_count = sum(1 for c in token_no_periods if c.isalpha())
        digit_count = sum(1 for c in token_no_periods if c.isdigit())

        self.valid_abbrev_candidate = (
            self.period_final
            and not _RE_NON_WORD_DOT.search(self.tok)
            and not (self.type == "##number##")
            and alpha_count >= digit_count  # Must have at least as many letters as digits
            and alpha_count > 0  # Must have at least one letter
        )

        # If token has a period but contains other punctuation, it can't be an abbreviation
        if self.period_final and not self.valid_abbrev_candidate:
            self.abbr = False

    @staticmethod
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
        return bool(self.tok) and self.tok[0].isupper()

    @property
    def first_lower(self) -> bool:
        """Check if the first character of the token is lowercase."""
        return bool(self.tok) and self.tok[0].islower()

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
        # Check for standard ellipsis (... or longer)
        if bool(_RE_ELLIPSIS.search(self.tok)):
            return True

        # Check for unicode ellipsis
        if self.tok == "\u2026" or self.tok.endswith("\u2026"):
            return True

        # Check for spaced ellipsis (. . ., . .  ., etc.)
        if _RE_SPACED_ELLIPSIS.search(self.tok):
            return True

        return False

    @property
    def is_number(self) -> bool:
        """Check if the token is a number."""
        return self.type.startswith("##number##")

    @property
    def is_initial(self) -> bool:
        """Check if the token is an initial (single letter followed by a period)."""
        return bool(_RE_INITIAL.fullmatch(self.tok))

    @property
    def is_alpha(self) -> bool:
        """Check if the token is alphabetic (contains only letters)."""
        return bool(_RE_ALPHA.fullmatch(self.tok))

    @property
    def is_non_punct(self) -> bool:
        """Check if the token contains non-punctuation characters."""
        return bool(_RE_NON_PUNCT.search(self.type))

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
