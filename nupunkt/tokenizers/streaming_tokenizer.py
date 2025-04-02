"""
Streaming sentence tokenizer module for nupunkt.

This module provides a token-free streaming implementation of the sentence tokenizer
that processes text as a byte stream without creating intermediate token objects.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from nupunkt.core.constants import (
    ABBREV_CACHE_SIZE,
    DOC_TOKENIZE_CACHE_SIZE,
    ORTHO_BEG_LC,
    ORTHO_CACHE_SIZE,
    ORTHO_LC,
    ORTHO_MID_UC,
    ORTHO_UC,
    SENT_STARTER_CACHE_SIZE,
    WHITESPACE_CACHE_SIZE,
)
from nupunkt.core.language_vars import PunktLanguageVars
from nupunkt.core.parameters import PunktParameters
from nupunkt.utils.compression import load_compressed_json, save_compressed_json
from nupunkt.utils.iteration import pair_iter
from nupunkt.tokenizers.sentence_tokenizer import (
    cached_ortho_heuristic,  # Reuse from standard tokenizer
    is_sent_starter,         # Reuse from standard tokenizer
)


@lru_cache(maxsize=ABBREV_CACHE_SIZE)
def is_abbreviation(abbrev_types: frozenset, candidate: str) -> bool:
    """
    Check if a candidate is a known abbreviation, using cached lookups.

    This method attempts to match the behavior of the standard tokenizer's
    abbreviation detection logic to ensure consistent results.

    Args:
        abbrev_types: A frozenset of known abbreviations
        candidate: The candidate string to check

    Returns:
        True if the candidate is a known abbreviation, False otherwise
    """
    # Quick check for empty candidate
    if not candidate:
        return False
    
    # Special handling for legal abbreviations like "v." vs "v"
    # Many legal abbreviations are stored without their trailing period
    if candidate.endswith('.'):
        # If the candidate ends with period, also check without period
        if candidate[:-1] in abbrev_types:
            return True
    
    # Check if the token itself is a known abbreviation
    if candidate in abbrev_types:
        return True
        
    # Try with period at the end
    if not candidate.endswith('.') and candidate + '.' in abbrev_types:
        return True

    # Check if the last part after a dash is a known abbreviation
    if "-" in candidate:
        dash_part = candidate.split("-")[-1]
        if dash_part in abbrev_types:
            return True
        if not dash_part.endswith('.') and dash_part + '.' in abbrev_types:
            return True
        # Also check without trailing period for the dash part
        if dash_part.endswith('.') and dash_part[:-1] in abbrev_types:
            return True

    # Special handling for period-separated abbreviations like U.S.C.
    # Check if the version without internal periods is in abbrev_types
    if "." in candidate:
        no_periods = candidate.replace(".", "")
        if no_periods in abbrev_types:
            return True
        
        # Try with a period at the end
        if not no_periods.endswith('.') and no_periods + '.' in abbrev_types:
            return True

    return False


# Pre-compiled regex for numbers (matching _RE_NUMBER in tokens.py)
_RE_NUMBER = re.compile(r"^-?[\.,]?\d[\d,\.-]*\.?$")

@lru_cache(maxsize=ABBREV_CACHE_SIZE)
def is_valid_abbrev_candidate(token: str) -> bool:
    """
    Check if a token is a valid abbreviation candidate based on character content.
    
    Uses the exact same logic as the standard PunktToken class.
    
    Args:
        token: The token to check
        
    Returns:
        True if the token is a valid abbreviation candidate
    """
    # Must end with period
    if not token.endswith('.'):
        return False
        
    # Check for number pattern (using same regex as standard tokenizer)
    if _RE_NUMBER.match(token):
        return False
        
    # Count alpha vs digit characters
    alpha_count = 0
    digit_count = 0
    
    # Skip the final period
    for c in token[:-1]:
        if c != '.':  # Skip internal periods too
            if c.isalpha():
                alpha_count += 1
            elif c.isdigit():
                digit_count += 1
    
    # Valid if more alphabetic than digits and at least one alphabetic
    return alpha_count >= digit_count and alpha_count > 0


class StreamingSentenceTokenizer:
    """
    Streaming sentence tokenizer using the Punkt algorithm.

    This implementation processes text as a byte stream without creating token objects,
    using direct span detection for better performance and memory efficiency.
    """

    # Pre-compiled regex patterns for boundary detection
    # Using the same patterns as the standard tokenizer
    _RE_ELLIPSIS_MULTI = re.compile(r"\.\.+")
    _RE_ELLIPSIS_SPACED = re.compile(r"\.\s+\.\s+\.")
    _RE_UNICODE_ELLIPSIS = re.compile("\u2026")
    
    # Match simple numbers with decimal point (avoid splitting 3.14 into two sentences)
    _RE_DECIMAL_NUMBER = re.compile(r"\d+\.\d+")
    
    # Set of common punctuation marks for fast lookup - same as standard tokenizer
    _PUNCT_CHARS = frozenset([";", ":", ",", ".", "!", "?"])
    
    # Common sentence-ending punctuation as a frozenset for O(1) lookups
    _SENT_END_CHARS = frozenset([".", "!", "?"])

    def __init__(
        self,
        params: Optional[Union[Dict, PunktParameters]] = None,
        lang_vars: Optional[PunktLanguageVars] = None,
        include_common_abbrevs: bool = True,
        cache_size: int = DOC_TOKENIZE_CACHE_SIZE,
        precision_mode: str = "balanced",
        context_window: int = 5,
        decimal_handling: bool = True,
        ortho_cache_size: int = ORTHO_CACHE_SIZE,
        sent_starter_cache_size: int = SENT_STARTER_CACHE_SIZE,
    ) -> None:
        """
        Initialize the streaming tokenizer with parameters.

        Args:
            params: Parameters or dict containing parameters
            lang_vars: Language-specific variables
            include_common_abbrevs: Whether to include common abbreviations
            cache_size: Size of the document-level tokenization cache
            precision_mode: Detection precision mode: "fast", "balanced", or "precise"
            context_window: Size of the context window for boundary analysis
            decimal_handling: Whether to avoid splitting sentences at decimal points
            ortho_cache_size: Size of the orthographic heuristic cache
            sent_starter_cache_size: Size of the sentence starter cache
        """
        # Initialize language variables
        self._lang_vars = lang_vars or PunktLanguageVars()
        
        # Initialize parameters
        if params is None:
            self._params = PunktParameters()
        elif isinstance(params, dict):
            self._params = PunktParameters.from_json(params)
        else:
            self._params = params
            
        # Cache size for tokenization
        self._cache_size = cache_size
        
        # Use the abbreviations already in the parameters
        # (This will include all abbreviations from the default model if loaded from it)
        abbrev_types = set(self._params.abbrev_types)
        
        # If explicitly requested to include common abbreviations and creating from scratch,
        # add them from the trainer
        if include_common_abbrevs and len(abbrev_types) == 0:
            # Import here to avoid circular import
            from nupunkt.trainers.base_trainer import PunktTrainer
            if hasattr(PunktTrainer, "COMMON_ABBREVS"):
                abbrev_types.update(PunktTrainer.COMMON_ABBREVS)
        
        # Convert sets to frozensets for faster lookups and caching
        self._abbrev_types = frozenset(abbrev_types)
        self._sent_starters = frozenset(self._params.sent_starters)
        self._collocations = frozenset(self._params.collocations)
        
        # Set precision mode settings
        self.set_precision_mode(precision_mode)
        
        # Set context window size (how much context to consider for boundary decisions)
        self._context_window = context_window
        
        # Special handling options
        self._decimal_handling = decimal_handling
        
        # Initialize punctuation sets from language variables
        if hasattr(self._lang_vars, 'sent_end_chars'):
            self._sent_end_chars = frozenset(self._lang_vars.sent_end_chars)
        else:
            self._sent_end_chars = self._SENT_END_CHARS
            
        if hasattr(self._lang_vars, 'punct_chars'):
            self._punct_chars = frozenset(self._lang_vars.punct_chars)
        else:
            self._punct_chars = self._PUNCT_CHARS

    def set_precision_mode(self, mode: str) -> None:
        """
        Set the precision mode for sentence boundary detection.
        
        Args:
            mode: "fast", "balanced", or "precise"
        """
        if mode not in ("fast", "balanced", "precise"):
            raise ValueError("Precision mode must be 'fast', 'balanced', or 'precise'")
        
        self._precision_mode = mode
        
        # Configure settings based on precision mode
        if mode == "fast":
            # Fast mode prioritizes speed over accuracy
            self._check_collocations = False
            self._check_orthographic_context = False
            self._use_abbreviation_heuristics = False
            self._check_sentence_starters = False
            self._perform_decimal_number_check = False
            self._realign_boundaries = False
        elif mode == "balanced":
            # Balanced mode attempts a good trade-off
            self._check_collocations = True
            self._check_orthographic_context = False
            self._use_abbreviation_heuristics = True
            self._check_sentence_starters = True
            self._perform_decimal_number_check = True
            self._realign_boundaries = True
        else:  # precise
            # Precise mode attempts to match standard implementation closely
            self._check_collocations = True
            self._check_orthographic_context = True
            self._use_abbreviation_heuristics = True
            self._check_sentence_starters = True
            self._perform_decimal_number_check = True
            self._realign_boundaries = True

    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
        lang_vars: Optional[PunktLanguageVars] = None,
    ) -> "StreamingSentenceTokenizer":
        """
        Load a StreamingSentenceTokenizer from a model file.

        Args:
            file_path: Path to the model file (can be JSON, compressed JSON, or binary)
            lang_vars: Optional language variables to use

        Returns:
            A new StreamingSentenceTokenizer instance
        """
        # Load the model data
        data = load_compressed_json(file_path)
        
        # Create a new tokenizer with the loaded parameters
        return cls(params=data, lang_vars=lang_vars)

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the tokenizer to a JSON-serializable dictionary.

        Returns:
            A JSON-serializable dictionary
        """
        # Create a trainer to handle serialization
        from nupunkt.trainers.base_trainer import PunktTrainer
        trainer = PunktTrainer(lang_vars=self._lang_vars)

        # Set the parameters
        trainer._params = self._params
        trainer._finalized = True

        return trainer.to_json()

    def save(
        self, file_path: Union[str, Path], compress: bool = True, compression_level: int = 1
    ) -> None:
        """
        Save the tokenizer to a JSON file, optionally with LZMA compression.

        Args:
            file_path: The path to save the file to
            compress: Whether to compress the file using LZMA (default: True)
            compression_level: LZMA compression level (0-9), lower is faster but less compressed
        """
        save_compressed_json(
            self.to_json(), file_path, level=compression_level, use_compression=compress
        )

    def _is_next_char_uppercase(self, text: str, pos: int, text_len: int) -> bool:
        """
        Check if the next non-whitespace character after a position is uppercase.
        
        Identical to the standard tokenizer's implementation.

        Args:
            text: The text to check
            pos: The position to start checking from
            text_len: The length of the text

        Returns:
            True if the next non-whitespace character is uppercase
        """
        i = pos
        while i < text_len and text[i].isspace():
            i += 1
        return i < text_len and text[i].isupper()

    @staticmethod
    @lru_cache(maxsize=WHITESPACE_CACHE_SIZE)
    def _cached_whitespace_index(text: str) -> int:
        """
        Cached implementation of finding the last whitespace index.
        
        Identical to the standard tokenizer's implementation.

        Args:
            text: The text to search

        Returns:
            The index of the last whitespace character, or 0 if none
        """
        for i in range(len(text) - 1, -1, -1):
            if text[i].isspace():
                return i
        return 0

    def _get_last_whitespace_index(self, text: str) -> int:
        """
        Find the index of the last whitespace character in a string.
        
        Identical to the standard tokenizer's implementation.

        Args:
            text: The text to search

        Returns:
            The index of the last whitespace character, or 0 if none
        """
        return self._cached_whitespace_index(text)

    def _ortho_heuristic(self, word: str, first_upper: bool, first_lower: bool) -> Union[bool, str]:
        """
        Apply orthographic heuristics using the same logic as the standard tokenizer.

        Args:
            word: The word to check
            first_upper: Whether the first character is uppercase
            first_lower: Whether the first character is lowercase

        Returns:
            True if the word starts a sentence, False if not, "unknown" if uncertain
        """
        # Simple case for punctuation tokens - use set lookup instead of tuple comparison
        if word in self._punct_chars:
            return False

        # If orthographic context checking is disabled in this precision mode, return unknown
        if not self._check_orthographic_context:
            return "unknown"

        # Get orthographic context from parameters
        ortho = self._params.ortho_context.get(word.lower(), 0)

        # Use the cached ortho heuristic from the standard tokenizer
        return cached_ortho_heuristic(ortho, word.lower(), first_upper, first_lower)

    @lru_cache(maxsize=DOC_TOKENIZE_CACHE_SIZE)
    def _tokenize_cached(self, text: str) -> List[Tuple[int, int]]:
        """
        Cached implementation of text tokenization.

        This allows repeated tokenization of the same text to be cached,
        which is useful for applications that repeatedly tokenize the same documents.

        Args:
            text: The text to tokenize

        Returns:
            List of (start, end) spans for each sentence
        """
        # Get raw sentence boundaries
        boundaries = list(self._find_sentence_boundaries(text))
        
        # Apply boundary realignment if enabled
        if self._realign_boundaries:
            boundaries = self._realign_sentence_boundaries(text, boundaries)
            
        return boundaries

    def tokenize(self, text: str, realign_boundaries: bool = True) -> List[str]:
        """
        Tokenize text into sentences.

        Args:
            text: The text to tokenize
            realign_boundaries: Whether to realign sentence boundaries (used for API compatibility)

        Returns:
            A list of sentences
        """
        # Use cached boundary detection for repeated texts
        boundaries = self._tokenize_cached(text)
        
        # Extract sentence strings from boundaries
        return [text[start:end] for start, end in boundaries]

    def span_tokenize(self, text: str, realign_boundaries: bool = True) -> Iterator[Tuple[int, int]]:
        """
        Tokenize text into sentence spans.

        Args:
            text: The text to tokenize
            realign_boundaries: Whether to realign sentence boundaries (used for API compatibility)

        Yields:
            Tuples of (start, end) character offsets for each sentence
        """
        # Use cached results if available
        boundaries = self._tokenize_cached(text)
        yield from boundaries

    def _realign_sentence_boundaries(
        self, text: str, boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Realign sentence boundaries to match standard tokenizer behavior.
        
        Based on the standard tokenizer's _realign_boundaries method.

        Args:
            text: The text
            boundaries: The sentence boundaries as (start, end) tuples

        Returns:
            Realigned sentence boundaries
        """
        if not boundaries:
            return boundaries
            
        # Skip realignment in fast mode
        if not self._realign_boundaries:
            return boundaries
            
        # Convert to slices for compatibility with standard tokenizer
        slices = [slice(start, end) for start, end in boundaries]
        
        # Apply realignment
        realigned_slices = []
        realign = 0
        
        # Process each pair of slices
        for slice1, slice2 in pair_iter(iter(slices)):
            slice1 = slice(slice1.start + realign, slice1.stop)
            if slice2 is None:
                if text[slice1]:
                    realigned_slices.append(slice1)
                continue
                
            # Use the language vars boundary realignment regex
            m = self._lang_vars.re_boundary_realignment.match(text[slice2])
            if m:
                realigned_slices.append(slice(slice1.start, slice2.start + len(m.group(0).rstrip())))
                realign = m.end()
            else:
                realign = 0
                if text[slice1]:
                    realigned_slices.append(slice1)
                    
        # Convert back to (start, end) tuples
        return [(s.start, s.stop) for s in realigned_slices]

    def _find_sentence_boundaries(self, text: str) -> Iterator[Tuple[int, int]]:
        """
        Find sentence boundaries in text without creating token objects.

        This is the core streaming implementation that processes text directly
        without creating intermediate objects.

        Args:
            text: The text to process

        Yields:
            Tuples of (start, end) spans for each detected sentence
        """
        # Skip processing empty text
        if not text:
            return

        # Quick check for sentence-ending characters
        if not any(end_char in text for end_char in self._sent_end_chars):
            # If no sentence-ending chars, the whole text is one sentence
            yield (0, len(text))
            return

        # Pre-compute text length
        text_len = len(text)
        
        # Track the current position in the text
        pos = 0
        sentence_start = 0
        
        # Precompile a regex to find decimal numbers based on current settings
        decimal_number_pattern = self._RE_DECIMAL_NUMBER if self._perform_decimal_number_check and self._decimal_handling else None
        
        # Direct character-by-character processing
        while pos < text_len:
            # Fast-forward to potential sentence boundary
            while pos < text_len and text[pos] not in self._sent_end_chars:
                pos += 1
                
            # If we reached the end without finding boundary, yield remaining text
            if pos >= text_len:
                if sentence_start < text_len:
                    yield (sentence_start, text_len)
                break
                
            # Found a potential boundary character
            boundary_char = text[pos]
            
            # Check for ellipsis patterns
            if boundary_char == '.' and pos + 1 < text_len and text[pos + 1] == '.':
                # Skip ellipsis (find end of consecutive periods)
                while pos < text_len and text[pos] == '.':
                    pos += 1
                continue
            
            # Special handling for ellipsis
            for ellipsis_pattern in [self._RE_ELLIPSIS_MULTI, self._RE_ELLIPSIS_SPACED, self._RE_UNICODE_ELLIPSIS]:
                # Look back up to 20 chars to find potential ellipsis
                look_back = max(0, pos - 20)
                context = text[look_back:pos + 5]
                
                if ellipsis_pattern.search(context):
                    # Check if there's a capital letter after the ellipsis
                    if self._is_next_char_uppercase(text, pos + 1, text_len):
                        # This is likely an ellipsis followed by a new sentence
                        is_boundary = True
                    else:
                        # Ellipsis without a capital letter after it - not a boundary
                        is_boundary = False
                        
                    if is_boundary:
                        break
            
            # Check for decimal numbers (3.14) if enabled
            if (decimal_number_pattern and 
                boundary_char == '.' and 
                pos > 0 and pos < text_len - 1):
                
                # Look back up to 20 chars to find start of potential decimal number
                look_back = max(0, pos - 20)
                context = text[look_back:pos + 10]
                
                # Check if we're in the middle of a decimal number
                if decimal_number_pattern.search(context) and context.find('.') > 0:
                    # Found a decimal point within a number - not a sentence boundary
                    pos += 1
                    continue
                
            # Get more context around potential boundary by looking at surrounding words
            # Look back for the word before the boundary
            word_start = pos
            while word_start > 0 and word_start > pos - 30 and text[word_start - 1].isalnum():
                word_start -= 1
                
            # Extract the word that might be an abbreviation
            potential_abbrev = text[word_start:pos].lower() if word_start < pos else ""

            # Check if next character is uppercase (potential sentence start)
            next_is_upper = False
            next_pos = pos + 1
            while next_pos < text_len and text[next_pos].isspace():
                next_pos += 1
            if next_pos < text_len:
                next_is_upper = text[next_pos].isupper()
                next_is_lower = text[next_pos].islower() if not next_is_upper else False
            else:
                next_is_upper = False
                next_is_lower = False
                
            # Get the next word for sentence starter check
            next_word_end = next_pos
            while next_word_end < text_len and next_word_end < next_pos + 30 and text[next_word_end].isalnum():
                next_word_end += 1
            next_word = text[next_pos:next_word_end].lower() if next_pos < next_word_end else ""
            
            # Decision logic for sentence boundary
            is_boundary = False
            
            # Exclamation and question marks almost always indicate boundaries
            if boundary_char in ('!', '?'):
                is_boundary = True
            
            # Check for periods
            elif boundary_char == '.':
                # Default case - periods typically end sentences
                is_boundary = True
                
                # Special handling for abbreviations
                if potential_abbrev:
                    # First check if it's a valid abbreviation candidate by character content
                    # This mimics the standard tokenizer's behavior
                    is_valid_candidate = is_valid_abbrev_candidate(potential_abbrev + ".")
                    
                    # Special handling for very short abbreviations like "v." in legal text
                    # These are always valid abbreviation candidates regardless of character content
                    if len(potential_abbrev) <= 2:
                        # For very short abbreviations, prioritize the abbreviation set lookup
                        is_short_abbrev = self._use_abbreviation_heuristics and is_abbreviation(self._abbrev_types, potential_abbrev)
                        if is_short_abbrev:
                            is_valid_candidate = True
                    
                    if is_valid_candidate:
                        # For valid candidates, check if they are known abbreviations
                        is_abbrev = self._use_abbreviation_heuristics and is_abbreviation(self._abbrev_types, potential_abbrev)
                        
                        if is_abbrev:
                            # For known abbreviations, by default they don't end sentences
                            is_boundary = False
                            
                            # But an uppercase letter following might indicate a new sentence,
                            # except when it's a known collocation or not a sentence starter
                            if next_is_upper:
                                # Check collocations first (e.g., "U.S. Government")
                                if self._check_collocations and (potential_abbrev, next_word) in self._collocations:
                                    is_boundary = False
                                # Special handling for legal citations and other short abbreviations
                                # These are particularly prone to incorrect splits when followed by capitals
                                elif len(potential_abbrev) <= 2 and is_abbrev and self._check_collocations:
                                    # Short abbreviations followed by capital words often form a phrase
                                    # Examples: "v. Jones", "Dr. Smith", "Mr. Johnson"
                                    is_boundary = False
                                # Then check sentence starters
                                elif self._check_sentence_starters and is_sent_starter(self._sent_starters, next_word):
                                    is_boundary = True
                                # Apply orthographic heuristics if enabled
                                elif self._check_orthographic_context:
                                    ortho_result = self._ortho_heuristic(next_word, next_is_upper, next_is_lower)
                                    if ortho_result is True:
                                        # Strong orthographic evidence for sentence start
                                        is_boundary = True
                                    elif ortho_result is False:
                                        # Strong orthographic evidence against sentence start
                                        is_boundary = False
                                    # Otherwise keep current boundary decision
                                else:
                                    # Default is to treat it as a sentence boundary in precise mode,
                                    # since uppercase words following abbreviations are often new sentences
                                    is_boundary = self._precision_mode == "precise"
                        else:
                            # This is a valid abbreviation candidate but not in our known abbreviations
                            # The standard tokenizer would mark this as a sentence break
                            is_boundary = True
                    else:
                        # This is not a valid abbreviation candidate
                        # Special cases for non-abbreviations with periods
                        
                        # Handle initials (single letter followed by period)
                        if len(potential_abbrev) == 1 and potential_abbrev.isalpha() and next_is_upper:
                            # Probably an initial like in "J. Smith" - not a boundary
                            is_boundary = False
                            
                            # But if orthographic context suggests sentence start, respect that
                            if self._check_orthographic_context:
                                ortho_result = self._ortho_heuristic(next_word, next_is_upper, next_is_lower)
                                if ortho_result is True:
                                    is_boundary = True
                            
                        # Handle decimal numbers
                        elif potential_abbrev.isdigit() and next_word and next_word[0].isdigit():
                            # This could be a decimal number like "3.14" or "100.000"
                            is_boundary = False
                            
                        # Handle domain names and URLs (partial implementation)
                        elif "." in potential_abbrev and any(c.isalpha() for c in potential_abbrev):
                            # Might be a domain name or part of an email/URL
                            parts = potential_abbrev.split(".")
                            if len(parts) >= 2 and all(len(p) > 0 for p in parts):
                                # This looks like a domain name structure - not a boundary
                                is_boundary = False
            
            # If we found a boundary, yield the sentence
            if is_boundary:
                # Include the boundary character itself
                yield (sentence_start, pos + 1)
                
                # Update the start position for the next sentence
                sentence_start = pos + 1
                
                # Skip whitespace after boundary
                while sentence_start < text_len and text[sentence_start].isspace():
                    sentence_start += 1
                
            # Move to next position
            pos += 1
            
        # Yield any remaining text as the final sentence
        if sentence_start < text_len:
            yield (sentence_start, text_len)