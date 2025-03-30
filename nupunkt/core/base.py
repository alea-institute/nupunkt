"""
Base module for nupunkt.

This module provides the base class for Punkt tokenizers and trainers.
"""

from typing import Iterator, Optional, Type

from nupunkt.core.constants import ORTHO_MAP
from nupunkt.core.language_vars import PunktLanguageVars
from nupunkt.core.parameters import PunktParameters
from nupunkt.core.tokens import PunktToken


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
        parastart = False
        for line in plaintext.splitlines():
            if line.strip():
                tokens = self._lang_vars.word_tokenize(line)
                if tokens:
                    yield self._Token(tokens[0], parastart=parastart, linestart=True)
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
                
                # Check if the token itself is a known abbreviation
                if candidate in self._params.abbrev_types:
                    token.abbr = True
                # Check if the last part after a dash is a known abbreviation
                elif "-" in candidate and candidate.split("-")[-1] in self._params.abbrev_types:
                    token.abbr = True
                # Special handling for period-separated abbreviations like U.S.C.
                # Check if the version without internal periods is in abbrev_types
                elif "." in candidate and candidate.replace(".", "") in self._params.abbrev_types:
                    token.abbr = True
                else:
                    token.sentbreak = True