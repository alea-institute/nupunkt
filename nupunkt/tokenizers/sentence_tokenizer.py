"""
Sentence tokenizer module for nupunkt.

This module provides the main tokenizer class for sentence boundary detection.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from nupunkt.core.base import PunktBase
from nupunkt.core.constants import ORTHO_BEG_LC, ORTHO_BEG_UC, ORTHO_LC, ORTHO_MID_UC, ORTHO_UC
from nupunkt.core.language_vars import PunktLanguageVars
from nupunkt.core.parameters import PunktParameters
from nupunkt.core.tokens import PunktToken
from nupunkt.trainers.base_trainer import PunktTrainer
from nupunkt.utils.iteration import pair_iter


class PunktSentenceTokenizer(PunktBase):
    """
    Sentence tokenizer using the Punkt algorithm.
    
    This class uses trained parameters to tokenize text into sentences,
    handling abbreviations, collocations, and other special cases.
    """
    # Precompiled regex patterns for ellipsis detection
    _re_ellipsis_patterns = [
        re.compile(r"\.\.+"),  
        re.compile(r"\.\s+\.\s+\."), 
        re.compile("\u2026")  # Unicode ellipsis character
    ]
    
    def __init__(
        self,
        train_text: Optional[Any] = None,
        verbose: bool = False,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Type[PunktToken] = PunktToken,
        include_common_abbrevs: bool = True,  # Whether to include common abbreviations
        cache_size: int = 100,  # Size of the sentence tokenization cache
    ) -> None:
        """
        Initialize the tokenizer, optionally with training text or parameters.
        
        Args:
            train_text: Training text or pre-trained parameters
            verbose: Whether to show verbose training information
            lang_vars: Language-specific variables
            token_cls: The token class to use
            include_common_abbrevs: Whether to include common abbreviations
            cache_size: Size of the cache for sentence tokenization results
        """
        super().__init__(lang_vars, token_cls)
        # If a training text (or pre-trained parameters) is provided,
        # use it to set the parameters.
        if train_text:
            if isinstance(train_text, str):
                trainer = PunktTrainer(
                    train_text, 
                    verbose=verbose, 
                    lang_vars=self._lang_vars, 
                    token_cls=self._Token,
                    include_common_abbrevs=include_common_abbrevs
                )
                self._params = trainer.get_params()
            else:
                self._params = train_text
                
        # Add common abbreviations if using an existing parameter set
        if include_common_abbrevs and not isinstance(train_text, str) and hasattr(PunktTrainer, 'COMMON_ABBREVS'):
            for abbr in PunktTrainer.COMMON_ABBREVS:
                self._params.abbrev_types.add(abbr)
            if verbose:
                print(f"Added {len(PunktTrainer.COMMON_ABBREVS)} common abbreviations to tokenizer.")
                
        # Initialize cache for tokenization results
        self._tokenize_cache = {}
        self._cache_size = cache_size
        self._cache_keys = []  # Track keys for LRU eviction
                
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the tokenizer to a JSON-serializable dictionary.
        
        Returns:
            A JSON-serializable dictionary
        """
        # Create a trainer to handle serialization
        trainer = PunktTrainer(lang_vars=self._lang_vars, token_cls=self._Token)
        
        # Set the parameters
        trainer._params = self._params
        trainer._finalized = True
        
        return trainer.to_json()
        
    @classmethod
    def from_json(cls, data: Dict[str, Any], lang_vars: Optional[PunktLanguageVars] = None,
                 token_cls: Optional[Type[PunktToken]] = None) -> "PunktSentenceTokenizer":
        """
        Create a PunktSentenceTokenizer from a JSON dictionary.
        
        Args:
            data: The JSON dictionary
            lang_vars: Optional language variables
            token_cls: Optional token class
            
        Returns:
            A new PunktSentenceTokenizer instance
        """
        # First create a trainer from the JSON data
        trainer = PunktTrainer.from_json(data, lang_vars, token_cls)
        
        # Then create a tokenizer with the parameters
        return cls(trainer.get_params(), lang_vars=lang_vars, token_cls=token_cls or PunktToken)
        
    def save(self, file_path: Union[str, Path], compress: bool = True, compression_level: int = 1) -> None:
        """
        Save the tokenizer to a JSON file, optionally with LZMA compression.
        
        Args:
            file_path: The path to save the file to
            compress: Whether to compress the file using LZMA (default: True)
            compression_level: LZMA compression level (0-9), lower is faster but less compressed
        """
        from nupunkt.utils.compression import save_compressed_json
        save_compressed_json(
            self.to_json(), 
            file_path, 
            level=compression_level, 
            use_compression=compress
        )
            
    @classmethod
    def load(cls, file_path: Union[str, Path], lang_vars: Optional[PunktLanguageVars] = None,
            token_cls: Optional[Type[PunktToken]] = None) -> "PunktSentenceTokenizer":
        """
        Load a PunktSentenceTokenizer from a JSON file, which may be compressed with LZMA.
        
        Args:
            file_path: The path to load the file from
            lang_vars: Optional language variables
            token_cls: Optional token class
            
        Returns:
            A new PunktSentenceTokenizer instance
        """
        from nupunkt.utils.compression import load_compressed_json
        data = load_compressed_json(file_path)
        return cls.from_json(data, lang_vars, token_cls)
        
    def reconfigure(self, config: Dict[str, Any]) -> None:
        """
        Reconfigure the tokenizer with new settings.
        
        Args:
            config: A dictionary with configuration settings
        """
        # Create a temporary trainer
        trainer = PunktTrainer.from_json(config, self._lang_vars, self._Token)
        
        # If parameters are present in the config, use them
        if "parameters" in config:
            self._params = PunktParameters.from_json(config["parameters"])
        else:
            # Otherwise just keep our current parameters
            trainer._params = self._params
            trainer._finalized = True

    def tokenize(self, text: str, realign_boundaries: bool = True) -> List[str]:
        """
        Tokenize text into sentences with caching for improved performance.
        
        Args:
            text: The text to tokenize
            realign_boundaries: Whether to realign sentence boundaries
            
        Returns:
            A list of sentences
        """
        # Use caching for frequently tokenized texts
        cache_key = (text, realign_boundaries)
        
        # Check cache first
        if cache_key in self._tokenize_cache:
            # Move this key to the end of the LRU list
            self._cache_keys.remove(cache_key)
            self._cache_keys.append(cache_key)
            return self._tokenize_cache[cache_key]
        
        # Not in cache, tokenize the text
        result = list(self.sentences_from_text(text, realign_boundaries))
        
        # Add to cache
        self._tokenize_cache[cache_key] = result
        self._cache_keys.append(cache_key)
        
        # Evict oldest item if cache is full
        if len(self._cache_keys) > self._cache_size:
            oldest_key = self._cache_keys.pop(0)
            del self._tokenize_cache[oldest_key]
            
        return result

    def span_tokenize(self, text: str, realign_boundaries: bool = True) -> Iterator[Tuple[int, int]]:
        """
        Tokenize text into sentence spans.
        
        Args:
            text: The text to tokenize
            realign_boundaries: Whether to realign sentence boundaries
            
        Yields:
            Tuples of (start, end) character offsets for each sentence
        """
        slices = list(self._slices_from_text(text))
        if realign_boundaries:
            slices = list(self._realign_boundaries(text, slices))
        for s in slices:
            yield (s.start, s.stop)

    def sentences_from_text(self, text: str, realign_boundaries: bool = True) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: The text to tokenize
            realign_boundaries: Whether to realign sentence boundaries
            
        Returns:
            A list of sentences
        """
        return [text[start:stop] for start, stop in self.span_tokenize(text, realign_boundaries)]

    def _get_last_whitespace_index(self, text: str) -> int:
        """
        Find the index of the last whitespace character in a string.
        
        Args:
            text: The text to search
            
        Returns:
            The index of the last whitespace character, or 0 if none
        """
        # Fast path for empty or very short text
        if not text or len(text) < 3:
            return 0
            
        # Most whitespace in text is spaces
        # rfind is implemented in C and much faster than Python loop
        last_space = text.rfind(' ')
        
        # If we found a space, return it directly
        if last_space >= 0:
            return last_space
            
        # Check for tab and newline only if space wasn't found
        # In most text, spaces are much more common
        last_tab = text.rfind('\t')
        if last_tab >= 0:
            return last_tab
            
        last_newline = text.rfind('\n')
        if last_newline >= 0:
            return last_newline
            
        # Fall back to slower method for other whitespace characters
        # This is very rare in normal text
        for i in range(len(text) - 1, -1, -1):
            if text[i].isspace():
                return i
                
        # No whitespace found
        return 0

    def _fast_lstrip_index(self, text: str, start_pos: int = 0) -> int:
        """
        Fast implementation to find the index after leading whitespace.
        
        Args:
            text: The text to process
            start_pos: The starting position in the text
            
        Returns:
            The index of the first non-whitespace character
        """
        i = start_pos
        while i < len(text) and text[i].isspace():
            i += 1
        return i
    
    def _match_potential_end_contexts(self, text: str) -> Iterator[Tuple[re.Match, str]]:
        """
        Find potential sentence end contexts in text.
        
        Args:
            text: The text to search
            
        Yields:
            Tuples of (match, context) for potential sentence ends
        """
        previous_slice = slice(0, 0)
        previous_match: Optional[re.Match] = None
        
        # Special handling for ellipsis followed by capital letter
        ellipsis_positions = []
        
        # Find positions of all ellipsis patterns in the text using precompiled patterns
        for pattern in self._re_ellipsis_patterns:
            for match in pattern.finditer(text):
                end_pos = match.end()
                # Check if there's a capital letter after the ellipsis
                if end_pos < len(text) - 1:
                    # Use our fast lstrip implementation that doesn't create new strings
                    first_non_ws = self._fast_lstrip_index(text, end_pos)
                    # Check if there's text after whitespace and it starts with uppercase
                    if first_non_ws < len(text) and text[first_non_ws].isupper():
                        ellipsis_positions.append(end_pos - 1)  # Position of the last period
        
        # Standard processing for period contexts
        for match in self._lang_vars.period_context_pattern.finditer(text):
            before_text = text[previous_slice.stop : match.start()]
            idx = self._get_last_whitespace_index(before_text)
            index_after_last_space = previous_slice.stop + idx + 1 if idx else previous_slice.start
            prev_word_slice = slice(index_after_last_space, match.start())
            if previous_match and previous_slice.stop <= prev_word_slice.start:
                yield previous_match, text[previous_slice] + previous_match.group() + previous_match.group("after_tok")
            previous_match = match
            previous_slice = prev_word_slice
        if previous_match:
            yield previous_match, text[previous_slice] + previous_match.group() + previous_match.group("after_tok")

    def _slices_from_text(self, text: str) -> Iterator[slice]:
        """
        Find slices of sentences in text.
        
        Args:
            text: The text to slice
            
        Yields:
            slice objects for each sentence
        """
        # Cache frequently used text slices for better performance
        cache_key = f"slices_{hash(text)}"
        if hasattr(self, '_slice_cache') and cache_key in self._slice_cache:
            # Return cached slices directly
            yield from self._slice_cache[cache_key]
            return
            
        slices = []
        last_break = 0
        
        # Process batches of potential end contexts for efficiency
        potential_ends = list(self._match_potential_end_contexts(text))
        
        for match, context in potential_ends:
            # Check if this is a sentence break
            if self.text_contains_sentbreak(context):
                current_slice = slice(last_break, match.end())
                slices.append(current_slice)
                yield current_slice
                
                # Update the last break position
                if match.group("next_tok"):
                    last_break = match.start("next_tok")
                else:
                    last_break = match.end()
        
        # Add the final slice without creating a new string via rstrip()
        # Find the last non-whitespace character index directly
        last_non_ws = len(text) - 1
        while last_non_ws >= 0 and text[last_non_ws].isspace():
            last_non_ws -= 1
        
        # If we found a non-whitespace character, add 1 to include it
        if last_non_ws >= 0:
            last_non_ws += 1
            
        final_slice = slice(last_break, last_non_ws)
        slices.append(final_slice)
        yield final_slice
        
        # Cache the results (if we have a reasonable number of slices)
        if len(slices) < 1000:  # Don't cache extremely large documents
            if not hasattr(self, '_slice_cache'):
                self._slice_cache = {}
                
            # Limited cache size
            if len(self._slice_cache) > 50:
                # Simple eviction strategy - just clear the cache
                self._slice_cache.clear()
                
            self._slice_cache[cache_key] = slices

    def _realign_boundaries(self, text: str, slices: List[slice]) -> Iterator[slice]:
        """
        Realign sentence boundaries to handle trailing punctuation.
        
        Args:
            text: The text
            slices: The sentence slices
            
        Yields:
            Realigned sentence slices
        """
        realign = 0
        for slice1, slice2 in pair_iter(iter(slices)):
            slice1 = slice(slice1.start + realign, slice1.stop)
            if slice2 is None:
                # Check if slice has content without creating a string
                if slice1.stop > slice1.start:
                    yield slice1
                continue
            m = self._lang_vars.re_boundary_realignment.match(text[slice2])
            if m:
                yield slice(slice1.start, slice2.start + len(m.group(0).rstrip()))
                realign = m.end()
            else:
                realign = 0
                # Check if slice has content without creating a string
                if slice1.stop > slice1.start:
                    yield slice1

    def text_contains_sentbreak(self, text: str) -> bool:
        """
        Check if text contains a sentence break.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text contains a sentence break
        """
        # Micro-optimization: Empty or very short texts cannot contain sentence breaks
        if not text or len(text) < 2:
            return False
            
        # Quick check for sentence ending characters
        for char in self._lang_vars.sent_end_chars:
            if char in text:
                # Fast path for obvious sentence breaks (most common case)
                if char != '.':  # For ! and ? we can be more certain
                    return True
                
                # For periods, we need to handle special cases
                # But continue with checks below
                break
        else:
            # No sentence ending chars found at all
            return False
        
        # Quick check for ellipsis pattern followed by uppercase
        # This is a common pattern that we can detect efficiently
        for pattern in self._re_ellipsis_patterns:
            match = pattern.search(text)
            if match:
                # Check if there's text after the ellipsis using our fast lstrip implementation
                end_pos = match.end()
                if end_pos < len(text):
                    first_non_ws = self._fast_lstrip_index(text, end_pos)
                    if first_non_ws < len(text) and text[first_non_ws].isupper():
                        return True
        
        # Cache the tokenization result if available
        # This helps with repeated text fragments that occur in the text
        cache_key = hash(text)
        if hasattr(self, '_sentbreak_cache'):
            if cache_key in self._sentbreak_cache:
                return self._sentbreak_cache[cache_key]
        else:
            self._sentbreak_cache = {}
            
        # Fall back to full tokenization and annotation for ambiguous cases
        tokens = list(self._annotate_tokens(self._tokenize_words(text)))
        result = any(token.sentbreak for token in tokens)
        
        # Cache the result (limited size)
        if len(self._sentbreak_cache) > 1000:
            # Simple eviction strategy - just clear the cache when it gets too big
            self._sentbreak_cache.clear()
        self._sentbreak_cache[cache_key] = result
        
        return result

    def _annotate_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Perform full annotation on tokens.
        
        Args:
            tokens: The tokens to annotate
            
        Yields:
            Fully annotated tokens
        """
        # Convert to list for better performance with pair_iter
        tokens_list = list(self._annotate_first_pass(tokens))
        
        # Perform second pass annotation
        # This is more efficient than using an iterator-based approach
        return self._annotate_second_pass(tokens_list)

    def _annotate_second_pass(self, tokens: List[PunktToken]) -> Iterator[PunktToken]:
        """
        Perform second-pass annotation on tokens.
        
        This applies collocational and orthographic heuristics.
        
        Args:
            tokens: The tokens to annotate (as a list for better performance)
            
        Yields:
            Tokens with second-pass annotation
        """
        # Use a more efficient approach for list inputs
        if not tokens:
            return
            
        # Bulk process all pairs at once
        for i in range(len(tokens) - 1):
            self._second_pass_annotation(tokens[i], tokens[i + 1])
            yield tokens[i]
            
        # Don't forget the last token
        if tokens:
            self._second_pass_annotation(tokens[-1], None)
            yield tokens[-1]

    def _second_pass_annotation(
        self, token1: PunktToken, token2: Optional[PunktToken]
    ) -> Optional[str]:
        """
        Perform second-pass annotation on a token.
        
        Args:
            token1: The current token
            token2: The next token
            
        Returns:
            A string describing the decision, or None
        """
        if token2 is None:
            return None
            
        # Special handling for ellipsis - check this before period_final check
        if token1.is_ellipsis:
            # If next token starts with uppercase and is a known sentence starter,
            # or has strong orthographic evidence of being a sentence starter,
            # then mark this as a sentence break
            is_sent_starter = self._ortho_heuristic(token2)
            next_typ = token2.type_no_sentperiod
            
            # Default behavior: ellipsis followed by uppercase letter is a sentence break
            if token2.first_upper:
                token1.sentbreak = True
                if is_sent_starter is True:
                    return "Ellipsis followed by orthographic sentence starter"
                elif next_typ in self._params.sent_starters:
                    return "Ellipsis followed by known sentence starter"
                else:
                    return "Ellipsis followed by uppercase word"
            else:
                token1.sentbreak = False
                return "Ellipsis not followed by sentence starter"
        
        # For tokens with periods but not ellipsis
        if not token1.period_final:
            return None
            
        typ = token1.type_no_period
        next_typ = token2.type_no_sentperiod
        tok_is_initial = token1.is_initial

        # Collocation heuristic: if the pair is known, mark token as abbreviation.
        if (typ, next_typ) in self._params.collocations:
            token1.sentbreak = False
            token1.abbr = True
            return "Known collocation"

        # If token is marked as an abbreviation, decide based on orthographic evidence.
        if token1.abbr and (not tok_is_initial):
            is_sent_starter = self._ortho_heuristic(token2)
            if is_sent_starter is True:
                token1.sentbreak = True
                return "Abbreviation with orthographic heuristic"
            if token2.first_upper and next_typ in self._params.sent_starters:
                token1.sentbreak = True
                return "Abbreviation with sentence starter"

        # Check for initials or ordinals.
        if tok_is_initial or typ == "##number##":
            is_sent_starter = self._ortho_heuristic(token2)
            if is_sent_starter is False:
                token1.sentbreak = False
                token1.abbr = True
                return "Initial with orthographic heuristic"
            if is_sent_starter == "unknown" and tok_is_initial and token2.first_upper and not (self._params.ortho_context.get(next_typ, 0) & ORTHO_LC):
                token1.sentbreak = False
                token1.abbr = True
                return "Initial with special orthographic heuristic"
        return None

    def _ortho_heuristic(self, token: PunktToken) -> bool | str:
        """
        Apply orthographic heuristics to determine if a token starts a sentence.
        
        Args:
            token: The token to check
            
        Returns:
            True if the token starts a sentence, False if not, "unknown" if uncertain
        """
        if token.tok in (";", ":", ",", ".", "!", "?"):
            return False
        ortho = self._params.ortho_context.get(token.type_no_sentperiod, 0)
        if token.first_upper and (ortho & ORTHO_LC) and not (ortho & ORTHO_MID_UC):
            return True
        if token.first_lower and ((ortho & ORTHO_UC) or not (ortho & ORTHO_BEG_LC)):
            return False
        return "unknown"