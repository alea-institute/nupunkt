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
        # Early exit for empty text
        if not plaintext:
            return
            
        parastart = False
        
        # Fast path for simple cases (no line breaks)
        if '\n' not in plaintext:
            # Quick check if line contains any non-whitespace characters
            line_has_content = False
            for c in plaintext:
                if not c.isspace():
                    line_has_content = True
                    break
            
            if line_has_content:
                tokens = self._lang_vars.word_tokenize(plaintext)
                if tokens:
                    yield self._Token(tokens[0], parastart=parastart, linestart=True)
                    for tok in tokens[1:]:
                        yield self._Token(tok)
            return
            
        # Process line by line
        for line in plaintext.splitlines():
            # Skip empty lines but mark the start of paragraphs
            # Check if line contains any non-whitespace characters without creating a new string
            line_has_content = False
            for c in line:
                if not c.isspace():
                    line_has_content = True
                    break
                    
            if not line_has_content:
                parastart = True
                continue
                
            # Process non-empty lines
            tokens = self._lang_vars.word_tokenize(line)
            if tokens:
                # First token gets paragraph and line start flags
                yield self._Token(tokens[0], parastart=parastart, linestart=True)
                
                # Remaining tokens in the line
                for tok in tokens[1:]:
                    yield self._Token(tok)
                    
            # Reset paragraph start flag after processing a non-empty line
            parastart = False

    def _annotate_first_pass(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Perform first-pass annotation on tokens.
        
        This annotates tokens with sentence breaks, abbreviations, and ellipses.
        
        Args:
            tokens: The tokens to annotate
            
        Yields:
            Annotated tokens
        """
        # Fast batch processing if tokens is a list
        if isinstance(tokens, list):
            for token in tokens:
                self._first_pass_annotation(token)
            return tokens
        
        # Regular iterator processing
        for token in tokens:
            self._first_pass_annotation(token)
            yield token

    def _first_pass_annotation(self, token: PunktToken) -> None:
        """
        Annotate a token with sentence breaks, abbreviations, and ellipses.
        
        Args:
            token: The token to annotate
        """
        # Quick check for empty tokens
        if not token.tok:
            return
            
        # Fast path for sentence ending characters (., !, ?)
        if len(token.tok) == 1 and token.tok in self._lang_vars.sent_end_chars:
            token.sentbreak = True
            return
            
        # Check for ellipsis pattern
        if token.is_ellipsis:
            token.ellipsis = True
            # Don't mark as sentence break now - will be decided in second pass
            # based on what follows the ellipsis
            token.sentbreak = False
            return
            
        # Handle period-final tokens
        if token.period_final:
            # Skip double-period tokens (likely part of an ellipsis)
            if token.tok.endswith(".."):
                return
                
            # If token is not a valid abbreviation candidate, mark it as a sentence break
            if not token.valid_abbrev_candidate:
                token.sentbreak = True
                return
                
            # For valid candidates, check if they are known abbreviations - use direct string ops
            # for better performance instead of creating a new string
            tok_len = len(token.tok)
            if tok_len > 1:
                # Get lowercase version of token without final period
                candidate = token.tok[:tok_len-1].lower()
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