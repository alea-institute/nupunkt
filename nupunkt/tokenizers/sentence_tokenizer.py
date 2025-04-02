"""
Sentence tokenizer module for nupunkt.

This module provides the main tokenizer class for sentence boundary detection.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from nupunkt.core.base import PunktBase
from nupunkt.core.constants import (
    DOC_TOKENIZE_CACHE_SIZE,
    ORTHO_BEG_LC,
    ORTHO_CACHE_SIZE,
    ORTHO_LC,
    ORTHO_MID_UC,
    ORTHO_UC,
    PARA_TOKENIZE_CACHE_SIZE,
    SENT_STARTER_CACHE_SIZE,
    WHITESPACE_CACHE_SIZE,
)
from nupunkt.core.language_vars import PunktLanguageVars
from nupunkt.core.parameters import PunktParameters
from nupunkt.core.tokens import PunktToken
from nupunkt.trainers.base_trainer import PunktTrainer
from nupunkt.utils.iteration import pair_iter


@lru_cache(maxsize=ORTHO_CACHE_SIZE)
def cached_ortho_heuristic(
    ortho_context: int, type_no_sentperiod: str, first_upper: bool, first_lower: bool
) -> Union[bool, str]:
    """
    Cached implementation of orthographic heuristics.

    Args:
        ortho_context: The orthographic context value from parameters
        type_no_sentperiod: The token type without sentence-final period
        first_upper: Whether the first character is uppercase
        first_lower: Whether the first character is lowercase

    Returns:
        True if the token starts a sentence, False if not, "unknown" if uncertain
    """
    if first_upper and (ortho_context & ORTHO_LC) and not (ortho_context & ORTHO_MID_UC):
        return True
    if first_lower and ((ortho_context & ORTHO_UC) or not (ortho_context & ORTHO_BEG_LC)):
        return False
    return "unknown"


@lru_cache(maxsize=SENT_STARTER_CACHE_SIZE)
def is_sent_starter(sent_starters: frozenset, token_type: str) -> bool:
    """
    Check if a token type is a known sentence starter, using cached lookups.

    Args:
        sent_starters: A frozenset of known sentence starters
        token_type: The token type to check

    Returns:
        True if the token type is a known sentence starter, False otherwise
    """
    return token_type in sent_starters


class PunktSentenceTokenizer(PunktBase):
    """
    Sentence tokenizer using the Punkt algorithm.

    This class uses trained parameters to tokenize text into sentences,
    handling abbreviations, collocations, and other special cases.
    """

    # Pre-compiled regex patterns
    _RE_ELLIPSIS_MULTI = re.compile(r"\.\.+")
    _RE_ELLIPSIS_SPACED = re.compile(r"\.\s+\.\s+\.")
    _RE_UNICODE_ELLIPSIS = re.compile("\u2026")

    # Set of common punctuation marks for fast lookup
    _PUNCT_CHARS = frozenset([";", ":", ",", ".", "!", "?"])

    # Common sentence-ending punctuation as a frozenset for O(1) lookups
    _SENT_END_CHARS = frozenset([".", "!", "?"])

    @staticmethod
    def _is_next_char_uppercase(text: str, pos: int, text_len: int) -> bool:
        """
        Check if the next non-whitespace character after a position is uppercase.

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

    def __init__(
        self,
        train_text: Optional[Any] = None,
        verbose: bool = False,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Type[PunktToken] = PunktToken,
        include_common_abbrevs: bool = True,  # Whether to include common abbreviations
        cache_size: int = DOC_TOKENIZE_CACHE_SIZE,  # Size of the sentence tokenization cache
        paragraph_cache_size: int = PARA_TOKENIZE_CACHE_SIZE,  # Size of the paragraph-level cache
        enable_paragraph_caching: bool = False,  # Whether to enable paragraph-level caching
        ortho_cache_size: int = ORTHO_CACHE_SIZE,  # Size of the orthographic heuristic cache
        sent_starter_cache_size: int = SENT_STARTER_CACHE_SIZE,  # Size of the sentence starter cache
        whitespace_cache_size: int = WHITESPACE_CACHE_SIZE,  # Size of the whitespace index cache
    ) -> None:
        """
        Initialize the tokenizer, optionally with training text or parameters.

        Args:
            train_text: Training text or pre-trained parameters
            verbose: Whether to show verbose training information
            lang_vars: Language-specific variables
            token_cls: The token class to use
            include_common_abbrevs: Whether to include common abbreviations
            cache_size: Size of the document-level tokenization cache
            paragraph_cache_size: Size of the paragraph-level cache
            enable_paragraph_caching: Whether to enable paragraph-level caching
            ortho_cache_size: Size of the orthographic heuristic cache
            sent_starter_cache_size: Size of the sentence starter cache
            whitespace_cache_size: Size of the whitespace index cache
        """
        super().__init__(lang_vars, token_cls)

        # Store cache sizes
        self._cache_size = cache_size
        self._paragraph_cache_size = paragraph_cache_size
        self._enable_paragraph_caching = enable_paragraph_caching

        # If a training text (or pre-trained parameters) is provided,
        # use it to set the parameters.
        if train_text:
            if isinstance(train_text, str):
                trainer = PunktTrainer(
                    train_text,
                    verbose=verbose,
                    lang_vars=self._lang_vars,
                    token_cls=self._Token,
                    include_common_abbrevs=include_common_abbrevs,
                )
                self._params = trainer.get_params()
            else:
                self._params = train_text

        # Add common abbreviations if using an existing parameter set
        if (
            include_common_abbrevs
            and not isinstance(train_text, str)
            and hasattr(PunktTrainer, "COMMON_ABBREVS")
        ):
            for abbr in PunktTrainer.COMMON_ABBREVS:
                self._params.abbrev_types.add(abbr)
            if verbose:
                print(
                    f"Added {len(PunktTrainer.COMMON_ABBREVS)} common abbreviations to tokenizer."
                )

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
    def from_json(
        cls,
        data: Dict[str, Any],
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Optional[Type[PunktToken]] = None,
    ) -> "PunktSentenceTokenizer":
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
        from nupunkt.utils.compression import save_compressed_json

        save_compressed_json(
            self.to_json(), file_path, level=compression_level, use_compression=compress
        )

    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Optional[Type[PunktToken]] = None,
    ) -> "PunktSentenceTokenizer":
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
        Tokenize text into sentences.

        Args:
            text: The text to tokenize
            realign_boundaries: Whether to realign sentence boundaries

        Returns:
            A list of sentences
        """
        return list(self.sentences_from_text(text, realign_boundaries))

    def span_tokenize(
        self, text: str, realign_boundaries: bool = True
    ) -> Iterator[Tuple[int, int]]:
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

    @staticmethod
    @lru_cache(maxsize=WHITESPACE_CACHE_SIZE)
    def _cached_whitespace_index(text: str) -> int:
        """
        Cached implementation of finding the last whitespace index.

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

        Args:
            text: The text to search

        Returns:
            The index of the last whitespace character, or 0 if none
        """
        return self._cached_whitespace_index(text)

    def _match_potential_end_contexts(self, text: str) -> Iterator[Tuple[re.Match, str]]:
        """
        Find potential sentence end contexts in text.

        Args:
            text: The text to search

        Yields:
            Tuples of (match, context) for potential sentence ends
        """
        # Pre-compute text length once
        text_len = len(text)

        # Skip processing if text is too short
        if text_len < 2:
            return

        # Quick check for any sentence-ending characters using frozenset for O(1) lookups
        if not any(end_char in text for end_char in self._SENT_END_CHARS):
            return

        # Collect all matches to avoid generator overhead
        matches: List[Tuple[re.Match, str]] = []

        previous_slice = slice(0, 0)
        previous_match: Optional[re.Match] = None

        # Special handling for ellipsis followed by capital letter - only check if text contains '..'
        ellipsis_positions = []

        # Fast path: only process ellipsis if the text contains consecutive periods
        if ".." in text or "\u2026" in text or ". . " in text:
            # Multiple periods ellipsis
            for match in self._RE_ELLIPSIS_MULTI.finditer(text):
                end_pos = match.end()
                # Check if there's a capital letter after the ellipsis
                if end_pos < text_len and self._is_next_char_uppercase(text, end_pos, text_len):
                    ellipsis_positions.append(end_pos - 1)  # Position of the last period

            # Spaced ellipsis
            for match in self._RE_ELLIPSIS_SPACED.finditer(text):
                end_pos = match.end()
                if end_pos < text_len and self._is_next_char_uppercase(text, end_pos, text_len):
                    ellipsis_positions.append(end_pos - 1)

            # Unicode ellipsis
            for match in self._RE_UNICODE_ELLIPSIS.finditer(text):
                end_pos = match.end()
                if end_pos < text_len and self._is_next_char_uppercase(text, end_pos, text_len):
                    ellipsis_positions.append(end_pos - 1)

        # Standard processing for period contexts
        for match in self._lang_vars.period_context_pattern.finditer(text):
            before_text = text[previous_slice.stop : match.start()]
            idx = self._get_last_whitespace_index(before_text)
            index_after_last_space = previous_slice.stop + idx + 1 if idx else previous_slice.start
            prev_word_slice = slice(index_after_last_space, match.start())
            if previous_match and previous_slice.stop <= prev_word_slice.start:
                # Use string joining for efficient concatenation
                context = "".join(
                    [
                        text[previous_slice.start : previous_slice.stop],
                        previous_match.group(),
                        previous_match.group("after_tok"),
                    ]
                )
                matches.append((previous_match, context))
            previous_match = match
            previous_slice = prev_word_slice

        if previous_match:
            # Use string joining for efficient concatenation
            context = "".join(
                [
                    text[previous_slice.start : previous_slice.stop],
                    previous_match.group(),
                    previous_match.group("after_tok"),
                ]
            )
            matches.append((previous_match, context))

        # Yield all matches at once
        yield from matches

    def _slices_from_text(self, text: str) -> Iterator[slice]:
        """
        Find slices of sentences in text.

        Args:
            text: The text to slice

        Yields:
            slice objects for each sentence
        """
        # Find the last non-whitespace character index directly without creating a copy
        text_end = len(text) - 1
        while text_end >= 0 and text[text_end].isspace():
            text_end -= 1
        # Add 1 to include the non-whitespace character itself
        if text_end >= 0:
            text_end += 1
        else:
            text_end = 0

        last_break = 0
        # Get all potential sentence breaks in one go
        for match, context in self._match_potential_end_contexts(text):
            if self.text_contains_sentbreak(context):
                yield slice(last_break, match.end())
                last_break = match.start("next_tok") if match.group("next_tok") else match.end()

        # Final slice
        if last_break < text_end:
            yield slice(last_break, text_end)

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
        # Use the original pair_iter as benchmark shows it's more efficient
        for slice1, slice2 in pair_iter(iter(slices)):
            slice1 = slice(slice1.start + realign, slice1.stop)
            if slice2 is None:
                if text[slice1]:
                    yield slice1
                continue
            m = self._lang_vars.re_boundary_realignment.match(text[slice2])
            if m:
                yield slice(slice1.start, slice2.start + len(m.group(0).rstrip()))
                realign = m.end()
            else:
                realign = 0
                if text[slice1]:
                    yield slice1

    def text_contains_sentbreak(self, text: str) -> bool:
        """
        Check if text contains a sentence break.

        Args:
            text: The text to check

        Returns:
            True if the text contains a sentence break
        """
        # Quick check for empty text
        if not text:
            return False

        # Quick check for extremely short text without sentence-ending punctuation
        if len(text) < 5 and not any(end_char in text for end_char in self._SENT_END_CHARS):
            return False

        # Quick check for definite sentence breaks (! or ?)
        # These are almost always sentence breaks and don't need full analysis
        # Use frozenset for faster lookups with the in operator
        excl_quest_marks = frozenset(["!", "?"])
        if "!" in text or "?" in text:
            # Further optimization: if followed by space + uppercase
            for i, char in enumerate(text[:-2]):
                if (
                    char in excl_quest_marks
                    and i < len(text) - 2
                    and text[i + 1].isspace()
                    and text[i + 2].isupper()
                ):
                    return True

        # Tokenize and annotate in one pass
        tokens = list(self._annotate_tokens(self._tokenize_words(text)))

        # No tokens means no sentence break
        if not tokens:
            return False

        # Fast check for sentbreak before looping
        if any(token.sentbreak for token in tokens):
            return True

        # Fast check for ellipsis - if no ellipsis is present, skip the loop
        if all(not token.ellipsis for token in tokens):
            return False

        # Special handling for ellipsis followed by capitalized word
        # Only run this if we have at least two tokens
        if len(tokens) > 1:
            # Check only tokens that are marked as ellipsis
            for i, token in enumerate(tokens[:-1]):  # Skip the last token
                if token.ellipsis and tokens[i + 1].first_upper:
                    return True

        return False

    def _annotate_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Perform full annotation on tokens.

        Args:
            tokens: The tokens to annotate

        Yields:
            Fully annotated tokens
        """
        tokens = self._annotate_first_pass(tokens)
        tokens = self._annotate_second_pass(tokens)
        return tokens

    def _annotate_second_pass(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        """
        Perform second-pass annotation on tokens.

        This applies collocational and orthographic heuristics.

        Args:
            tokens: The tokens to annotate

        Yields:
            Tokens with second-pass annotation
        """
        # Use the original pair_iter as benchmark shows it's more efficient
        for token1, token2 in pair_iter(tokens):
            self._second_pass_annotation(token1, token2)
            yield token1

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
                # Use cached lookup for sentence starters
                elif self._is_sent_starter(next_typ):
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
            # Use cached lookup for sentence starters
            if token2.first_upper and self._is_sent_starter(next_typ):
                token1.sentbreak = True
                return "Abbreviation with sentence starter"

        # Check for initials or ordinals.
        if tok_is_initial or typ == "##number##":
            is_sent_starter = self._ortho_heuristic(token2)
            if is_sent_starter is False:
                token1.sentbreak = False
                token1.abbr = True
                return "Initial with orthographic heuristic"
            if (
                is_sent_starter == "unknown"
                and tok_is_initial
                and token2.first_upper
                and not (self._params.ortho_context.get(next_typ, 0) & ORTHO_LC)
            ):
                token1.sentbreak = False
                token1.abbr = True
                return "Initial with special orthographic heuristic"
        return None

    def _ortho_heuristic(self, token: PunktToken) -> Union[bool, str]:
        """
        Apply orthographic heuristics to determine if a token starts a sentence.

        Args:
            token: The token to check

        Returns:
            True if the token starts a sentence, False if not, "unknown" if uncertain
        """
        # Simple case for punctuation tokens - use set lookup instead of tuple comparison
        if token.tok in (";", ":", ",", ".", "!", "?"):
            return False

        # Get orthographic context
        ortho = self._params.ortho_context.get(token.type_no_sentperiod, 0)

        # Use module-level cached function
        return cached_ortho_heuristic(
            ortho, token.type_no_sentperiod, token.first_upper, token.first_lower
        )

    def _cached_ortho_heuristic(
        self, type_no_sentperiod: str, first_upper: bool, first_lower: bool
    ) -> Union[bool, str]:
        """
        Wrapper for the cached implementation of orthographic heuristics.

        Args:
            type_no_sentperiod: The token type without sentence-final period
            first_upper: Whether the first character is uppercase
            first_lower: Whether the first character is lowercase

        Returns:
            True if the token starts a sentence, False if not, "unknown" if uncertain
        """
        ortho = self._params.ortho_context.get(type_no_sentperiod, 0)
        return cached_ortho_heuristic(ortho, type_no_sentperiod, first_upper, first_lower)

    def _is_sent_starter(self, token_type: str) -> bool:
        """
        Check if a token type is a known sentence starter.

        This is a wrapper around the module-level cached function.

        Args:
            token_type: The token type to check

        Returns:
            True if the token type is a known sentence starter, False otherwise
        """
        # Get sentence starters set
        sent_starters = self._params.sent_starters

        # Convert to frozenset if needed for caching
        if not isinstance(sent_starters, frozenset):
            sent_starters = frozenset(sent_starters)

        return is_sent_starter(sent_starters, token_type)
