"""
Token module for nupunkt.

This module provides the PunktToken class, which represents a token 
in the Punkt algorithm and calculates various derived properties.
"""

import re
from dataclasses import dataclass, field


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
    # Using private fields for lazy-loaded properties
    _type: str = field(default=None, init=False)
    _valid_abbrev_candidate: bool = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Initialize derived attributes after instance creation.
        
        This method calculates core properties needed immediately,
        while deferring other calculations until needed.
        """
        # Only calculate period_final immediately - it's used frequently
        self.period_final = self.tok.endswith(".")
        
        # If token doesn't end with period, we know it's not a valid abbreviation
        # This quick check avoids further calculation for most tokens
        if not self.period_final:
            self._valid_abbrev_candidate = False
            
        # Cache for property values
        self._property_cache = {}

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
        if re.match(r"^-?[\.,]?\d[\d,\.-]*\.?$", tok):
            return "##number##"
        return tok.lower()
        
    @property
    def type(self) -> str:
        """Get the normalized type of the token, with lazy loading."""
        if self._type is None:
            self._type = self._get_type(self.tok)
        return self._type

    @property
    def type_no_period(self) -> str:
        """Get the token type without a trailing period."""
        cache_key = 'type_no_period'
        if cache_key not in self._property_cache:
            result = self.type[:-1] if self.type.endswith(".") and len(self.type) > 1 else self.type
            self._property_cache[cache_key] = result
        return self._property_cache[cache_key]

    @property
    def type_no_sentperiod(self) -> str:
        """Get the token type without a sentence-final period."""
        cache_key = 'type_no_sentperiod'
        if cache_key not in self._property_cache:
            result = self.type_no_period if self.sentbreak else self.type
            self._property_cache[cache_key] = result
        return self._property_cache[cache_key]

    @property
    def valid_abbrev_candidate(self) -> bool:
        """Determine if token could be a valid abbreviation candidate."""
        # Use cached value if available
        if self._valid_abbrev_candidate is not None:
            return self._valid_abbrev_candidate
            
        # Quick return for non-period-final tokens
        if not self.period_final:
            self._valid_abbrev_candidate = False
            return False
            
        # Fast path for very common abbreviation patterns
        tok = self.tok.lower()
        if len(tok) <= 5:  # Most abbreviations are short
            # Common abbreviations (Mr., Dr., etc.)
            if tok in ("mr.", "dr.", "ms.", "jr.", "sr.", "co.", "ltd.", "inc."):
                self._valid_abbrev_candidate = True
                return True
                
            # Single letter abbreviations with period (A., B., etc.)
            if len(tok) == 2 and tok[0].isalpha() and tok[1] == '.':
                self._valid_abbrev_candidate = True
                return True
                
            # Common U.S. state abbreviations (ca., ga., etc.)
            if tok in ("al.", "ak.", "az.", "ar.", "ca.", "co.", "ct.", "de.", "fl.", 
                       "ga.", "hi.", "id.", "il.", "in.", "ia.", "ks.", "ky.", "la.", 
                       "me.", "md.", "ma.", "mi.", "mn.", "ms.", "mo.", "mt.", "ne.", 
                       "nv.", "nh.", "nj.", "nm.", "ny.", "nc.", "nd.", "oh.", "ok.", 
                       "or.", "pa.", "ri.", "sc.", "sd.", "tn.", "tx.", "ut.", "vt.", 
                       "va.", "wa.", "wv.", "wi.", "wy."):
                self._valid_abbrev_candidate = True
                return True
        
        # Fast path for common non-abbreviation patterns
        # Check for invalid characters (anything not alphanumeric or period)
        for c in self.tok:
            if not (c.isalnum() or c == '.'):
                self._valid_abbrev_candidate = False
                return False
            
        # Rules:
        # 1. Must end with a period (already checked)
        # 2. Only alphanumeric characters and periods allowed (checked above)
        # 3. Not a pure number
        # 4. Must have at least as many alphabet chars as digits
        
        # For tokens with internal periods (like U.S.C), get the non-period characters for counting
        # Use direct counting instead of creating a new string
        alpha_count = 0
        digit_count = 0
        has_alpha = False
        
        for c in self.tok:
            if c != '.':
                if c.isalpha():
                    alpha_count += 1
                    has_alpha = True
                elif c.isdigit():
                    digit_count += 1
        
        # Early exit if no alphabet characters
        if not has_alpha:
            self._valid_abbrev_candidate = False
            return False
            
        # Final check: not a number and at least as many alpha as digits
        self._valid_abbrev_candidate = (
            not (self.type == "##number##") and 
            alpha_count >= digit_count
        )
        
        return self._valid_abbrev_candidate
        
    @valid_abbrev_candidate.setter
    def valid_abbrev_candidate(self, value: bool) -> None:
        """Set the valid_abbrev_candidate flag."""
        self._valid_abbrev_candidate = value

    @property
    def first_upper(self) -> bool:
        """Check if the first character of the token is uppercase."""
        cache_key = 'first_upper'
        if cache_key not in self._property_cache:
            self._property_cache[cache_key] = bool(self.tok) and self.tok[0].isupper()
        return self._property_cache[cache_key]

    @property
    def first_lower(self) -> bool:
        """Check if the first character of the token is lowercase."""
        cache_key = 'first_lower'
        if cache_key not in self._property_cache:
            self._property_cache[cache_key] = bool(self.tok) and self.tok[0].islower()
        return self._property_cache[cache_key]

    @property
    def first_case(self) -> str:
        """Get the case of the first character of the token."""
        cache_key = 'first_case'
        if cache_key not in self._property_cache:
            if self.first_lower:
                result = "lower"
            elif self.first_upper:
                result = "upper"
            else:
                result = "none"
            self._property_cache[cache_key] = result
        return self._property_cache[cache_key]

    @property
    def is_ellipsis(self) -> bool:
        """
        Check if the token is an ellipsis (any of the following patterns):
        1. Multiple consecutive periods (..., ......)
        2. Unicode ellipsis character (…)
        3. Periods separated by spaces (. . ., .  .  .)
        """
        cache_key = 'is_ellipsis'
        if cache_key not in self._property_cache:
            # Fast path for common patterns
            if self.tok == "..." or self.tok == "\u2026":
                self._property_cache[cache_key] = True
            elif ".." in self.tok:
                # Check for standard ellipsis (... or longer)
                self._property_cache[cache_key] = bool(re.search(r"\.\.+$", self.tok))
            elif ". " in self.tok and "." in self.tok[self.tok.index(". ")+2:]:
                # Check for spaced ellipsis (. . ., . .  ., etc.)
                self._property_cache[cache_key] = bool(re.search(r"\.\s+\.\s+\.", self.tok))
            else:
                self._property_cache[cache_key] = False
        return self._property_cache[cache_key]

    @property
    def is_number(self) -> bool:
        """Check if the token is a number."""
        cache_key = 'is_number'
        if cache_key not in self._property_cache:
            self._property_cache[cache_key] = self.type.startswith("##number##")
        return self._property_cache[cache_key]

    @property
    def is_initial(self) -> bool:
        """Check if the token is an initial (single letter followed by a period)."""
        cache_key = 'is_initial'
        if cache_key not in self._property_cache:
            # Fast path for common case
            if len(self.tok) == 2 and self.tok[1] == '.' and self.tok[0].isalpha():
                self._property_cache[cache_key] = True
            else:
                self._property_cache[cache_key] = bool(re.fullmatch(r"[^\W\d]\.", self.tok))
        return self._property_cache[cache_key]

    @property
    def is_alpha(self) -> bool:
        """Check if the token is alphabetic (contains only letters)."""
        cache_key = 'is_alpha'
        if cache_key not in self._property_cache:
            # Fast path without regex
            if self.tok.isalpha():
                self._property_cache[cache_key] = True
            else:
                self._property_cache[cache_key] = bool(re.fullmatch(r"[^\W\d]+", self.tok))
        return self._property_cache[cache_key]

    @property
    def is_non_punct(self) -> bool:
        """Check if the token contains non-punctuation characters."""
        cache_key = 'is_non_punct'
        if cache_key not in self._property_cache:
            # Fast path for common case - check if any character is alphanumeric
            for c in self.type:
                if c.isalnum():
                    self._property_cache[cache_key] = True
                    break
            else:
                self._property_cache[cache_key] = bool(re.search(r"[^\W\d]", self.type))
        return self._property_cache[cache_key]

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