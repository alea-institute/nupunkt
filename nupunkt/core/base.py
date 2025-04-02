"""
Base module for nupunkt.

This module provides the base class for Punkt tokenizers and trainers.
"""

from functools import lru_cache
from typing import Iterator, Optional, Type

from nupunkt.core.constants import ABBREV_CACHE_SIZE
from nupunkt.core.language_vars import PunktLanguageVars
from nupunkt.core.parameters import PunktParameters
from nupunkt.core.tokens import PunktToken, create_punkt_token


@lru_cache(maxsize=ABBREV_CACHE_SIZE)
def is_abbreviation(abbrev_set: frozenset, candidate: str) -> bool:
    """
    Check if a candidate is a known abbreviation, using cached lookups.

    Args:
        abbrev_set: A frozenset of known abbreviations
        candidate: The candidate string to check

    Returns:
        True if the candidate is a known abbreviation, False otherwise
    """
    # Check if the token itself is a known abbreviation
    if candidate in abbrev_set:
        return True

    # Check if the last part after a dash is a known abbreviation
    if "-" in candidate:
        dash_part = candidate.split("-")[-1]
        if dash_part in abbrev_set:
            return True

    # Special handling for period-separated abbreviations like U.S.C.
    # Check if the version without internal periods is in abbrev_types
    if "." in candidate:
        no_periods = candidate.replace(".", "")
        if no_periods in abbrev_set:
            return True

    return False


class PunktBase:
    """
    Base class for Punkt tokenizers and trainers.

    This class provides common functionality used by both the trainer and tokenizer,
    including tokenization and first-pass annotation of tokens.
    """

    def __init__(
        self,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Type[PunktToken] = PunktToken,
        params: Optional[PunktParameters] = None,
    ) -> None:
        """
        Initialize the PunktBase instance.

        Args:
            lang_vars: Language-specific variables
            token_cls: The token class to use
            params: Punkt parameters
        """
        self._lang_vars = lang_vars or PunktLanguageVars()
        self._Token = token_cls
        self._params = params or PunktParameters()

    def _tokenize_words(self, plaintext: str) -> Iterator[PunktToken]:
        """
        Tokenize text into words, maintaining paragraph and line-start information.

        Args:
            plaintext: The text to tokenize

        Yields:
            PunktToken instances for each token
        """
        # Quick check for empty text
        if not plaintext:
            return

        parastart = False
        # Split by lines - this is more efficient than using splitlines()
        for line in plaintext.split("\n"):
            # Check if line has any content
            if line.strip():
                tokens = self._lang_vars.word_tokenize(line)
                if tokens:
                    # First token gets parastart and linestart flags
                    # Use the factory function to benefit from caching
                    if issubclass(self._Token, PunktToken):
                        # Use our optimized factory function if the token class is PunktToken
                        yield create_punkt_token(tokens[0], parastart=parastart, linestart=True)

                        # Process remaining tokens in a batch when possible
                        for tok in tokens[1:]:
                            yield create_punkt_token(tok)
                    else:
                        # Fallback for custom token classes
                        yield self._Token(tokens[0], parastart=parastart, linestart=True)

                        # Process remaining tokens in a batch when possible
                        for tok in tokens[1:]:
                            yield self._Token(tok)
                parastart = False
            else:
                parastart = True

    def _annotate_first_pass(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Perform first-pass annotation on tokens.

        This annotates tokens with sentence breaks, abbreviations, and ellipses.

        Args:
            tokens: The tokens to annotate

        Yields:
            Annotated tokens
        """
        for token in tokens:
            self._first_pass_annotation(token)
            yield token

    def _first_pass_annotation(self, token: PunktToken) -> None:
        """
        Annotate a token with sentence breaks, abbreviations, and ellipses.

        Args:
            token: The token to annotate
        """
        if token.tok in self._lang_vars.sent_end_chars:
            token.sentbreak = True
        elif token.is_ellipsis:
            token.ellipsis = True
            # Don't mark as sentence break now - will be decided in second pass
            # based on what follows the ellipsis
            token.sentbreak = False
        elif token.period_final and not token.tok.endswith(".."):
            # If token is not a valid abbreviation candidate, mark it as a sentence break
            if not token.valid_abbrev_candidate:
                token.sentbreak = True
            else:
                # For valid candidates, check if they are known abbreviations
                candidate = token.tok[:-1].lower()

                # Use frozen set for faster lookups if available
                abbrev_set = (
                    getattr(self._params, "_frozen_abbrev_types", None) or self._params.abbrev_types
                )

                # Convert to frozenset if it's not already (for caching)
                if not isinstance(abbrev_set, frozenset):
                    abbrev_set = frozenset(abbrev_set)

                # Use the module-level cached function
                if is_abbreviation(abbrev_set, candidate):
                    token.abbr = True
                else:
                    token.sentbreak = True

    def _is_abbreviation(self, candidate: str) -> bool:
        """
        Check if a candidate is a known abbreviation.

        This is a wrapper around the module-level cached function.

        Args:
            candidate: The candidate string to check

        Returns:
            True if the candidate is a known abbreviation, False otherwise
        """
        # Use frozen set for faster lookups if available
        abbrev_set = (
            getattr(self._params, "_frozen_abbrev_types", None) or self._params.abbrev_types
        )

        # Convert to frozenset if it's not already (for caching)
        if not isinstance(abbrev_set, frozenset):
            abbrev_set = frozenset(abbrev_set)

        return is_abbreviation(abbrev_set, candidate)
