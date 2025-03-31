"""
Trainer module for nupunkt.

This module provides the trainer class for the Punkt algorithm, which learns
parameters from training text to use for sentence boundary detection.
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

from nupunkt.core.base import PunktBase
from nupunkt.core.constants import (
    ORTHO_BEG_UC,
    ORTHO_MAP,
    ORTHO_MID_UC,
)
from nupunkt.core.language_vars import PunktLanguageVars
from nupunkt.core.parameters import PunktParameters
from nupunkt.core.tokens import PunktToken
from nupunkt.utils.iteration import pair_iter
from nupunkt.utils.statistics import collocation_log_likelihood, dunning_log_likelihood


class PunktTrainer(PunktBase):
    """
    Trainer for Punkt sentence boundary detection parameters.

    This class learns from training text to identify abbreviations, collocations,
    and sentence starters, which are then used for sentence boundary detection.
    """

    # Customization parameters (tweak as needed)
    ABBREV: float = 0.1  # Very low threshold to reliably capture abbreviations
    ABBREV_BACKOFF: int = 10  # Lower frequency threshold for rare abbreviations
    COLLOCATION: float = 5
    SENT_STARTER: float = 25.0
    INCLUDE_ALL_COLLOCS: bool = False
    INCLUDE_ABBREV_COLLOCS: bool = False
    MIN_COLLOC_FREQ: int = 5  # Minimum frequency for collocations
    MAX_ABBREV_LENGTH: int = 9  # Maximum length for abbreviation detection

    # Stability settings
    ABBREV_CONSISTENCY: float = (
        0.25  # How consistent an abbreviation's sentence-boundary behavior must be
    )
    PERSIST_ABBREVS: bool = True  # Whether to persist abbreviations between training runs

    # Common English abbreviations that should always be detected
    COMMON_ABBREVS: ClassVar[List[str]] = ["..."]  # Include ellipsis as a common "abbreviation"

    # JSON serialization keys
    CONFIG_ABBREV: str = "abbrev_threshold"
    CONFIG_ABBREV_BACKOFF: str = "abbrev_backoff"
    CONFIG_COLLOCATION: str = "collocation_threshold"
    CONFIG_SENT_STARTER: str = "sent_starter_threshold"
    CONFIG_INCLUDE_ALL_COLLOCS: str = "include_all_collocs"
    CONFIG_INCLUDE_ABBREV_COLLOCS: str = "include_abbrev_collocs"
    CONFIG_MIN_COLLOC_FREQ: str = "min_colloc_freq"
    CONFIG_MAX_ABBREV_LENGTH: str = "max_abbrev_length"
    CONFIG_COMMON_ABBREVS: str = "common_abbrevs"
    CONFIG_LANGUAGE: str = "language"

    def __init__(
        self,
        train_text: Optional[str] = None,
        verbose: bool = False,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Type[PunktToken] = PunktToken,
        include_common_abbrevs: bool = True,  # Whether to include common abbreviations by default
    ) -> None:
        """
        Initialize the trainer, optionally with training text.

        Args:
            train_text: Optional training text to immediately train on
            verbose: Whether to show verbose training information
            lang_vars: Language-specific variables
            token_cls: The token class to use
            include_common_abbrevs: Whether to include common abbreviations by default
        """
        super().__init__(lang_vars, token_cls)
        self._type_fdist: Counter[str] = Counter()
        self._num_period_toks: int = 0
        self._collocation_fdist: Counter[Tuple[str, str]] = Counter()
        self._sent_starter_fdist: Counter[str] = Counter()
        self._sentbreak_count: int = 0
        self._finalized: bool = True

        # Pre-load common abbreviations for better handling
        if include_common_abbrevs:
            for abbr in self.COMMON_ABBREVS:
                self._params.abbrev_types.add(abbr)
                if verbose:
                    print(f"Pre-loaded common abbreviation: {abbr}")

        if train_text:
            self.train(train_text, verbose=verbose, finalize=True)

    def get_params(self) -> PunktParameters:
        """
        Get the trained parameters.

        Returns:
            The trained Punkt parameters
        """
        if not self._finalized:
            self.finalize_training()
        return self._params

    def train(
        self, text: str, verbose: bool = False, finalize: bool = True, preserve_abbrevs: bool = None
    ) -> None:
        """
        Train the model on the given text.

        Args:
            text: The training text
            verbose: Whether to display progress information
            finalize: Whether to finalize training after this run
            preserve_abbrevs: Whether to preserve existing abbreviations (overrides self.PERSIST_ABBREVS)
        """
        # Store current abbreviations if preserving them
        should_preserve = self.PERSIST_ABBREVS if preserve_abbrevs is None else preserve_abbrevs
        original_abbrevs = set()
        if should_preserve:
            original_abbrevs = set(self._params.abbrev_types)
            if verbose:
                print(f"Preserving {len(original_abbrevs)} existing abbreviations")

        if verbose:
            try:
                from tqdm import tqdm

                print("Tokenizing text...")
            except ImportError:
                tqdm = lambda x, **kwargs: x
                print("Note: Install tqdm for progress bars during training.")

        # Tokenize text
        tokens = list(self._tokenize_words(text))

        if verbose:
            print(f"Found {len(tokens)} tokens in text.")

        self._train_tokens(tokens, verbose)

        # Reapply preserved abbreviations if needed
        if should_preserve and original_abbrevs:
            # Restore original abbreviations, but respect newly learned ones too
            for abbrev in original_abbrevs:
                # Only add valid abbreviation candidates (alphanumeric only)
                if not re.search(r"[^\w]", abbrev) and len(abbrev) <= self.MAX_ABBREV_LENGTH:
                    self._params.abbrev_types.add(abbrev)

            if verbose:
                preserved_count = len(self._params.abbrev_types & original_abbrevs)
                print(f"Preserved {preserved_count} abbreviations")

        if finalize:
            self.finalize_training(verbose)

    def _train_tokens(self, tokens: List[PunktToken], verbose: bool) -> None:
        """
        Train on a list of tokens.

        Args:
            tokens: The tokens to train on
            verbose: Whether to display progress information
        """
        self._finalized = False

        if verbose:
            try:
                from tqdm import tqdm

                token_iter = tqdm(tokens, desc="Counting tokens", unit="token")
            except ImportError:
                token_iter = tokens
                if verbose:
                    print("Counting tokens...")
        else:
            token_iter = tokens

        # First pass: count tokens and build frequency distribution
        for token in token_iter:
            self._type_fdist[token.type] += 1
            if token.period_final:
                self._num_period_toks += 1

        # Identify abbreviation types
        unique_types = {token.type for token in tokens}
        if verbose:
            print(f"Found {len(unique_types)} unique token types.")

            # Print the most frequent tokens with periods
            print("\nMost frequent tokens ending with period:")
            period_tokens = [
                (t, c)
                for t, c in self._type_fdist.items()
                if t.endswith(".") and c >= self.ABBREV_BACKOFF
            ]
            period_tokens.sort(key=lambda x: x[1], reverse=True)
            for token, count in period_tokens[:20]:
                print(f"  {token:<15} {count:>5}")

            print("\nIdentifying abbreviations...")
            try:
                from tqdm import tqdm

                abbrev_iter = tqdm(
                    list(self._reclassify_abbrev_types(unique_types)),
                    desc="Classifying abbreviations",
                    unit="type",
                )
            except ImportError:
                abbrev_iter = self._reclassify_abbrev_types(unique_types)
        else:
            abbrev_iter = self._reclassify_abbrev_types(unique_types)

        for typ, score, is_add in abbrev_iter:
            if score >= self.ABBREV:
                if is_add:
                    self._params.abbrev_types.add(typ)
            else:
                if not is_add and typ in self._params.abbrev_types:
                    self._params.abbrev_types.remove(typ)

        # Annotate tokens with sentence breaks
        if verbose:
            print("Annotating tokens...")
        tokens = list(self._annotate_first_pass(tokens))

        # Gather orthographic data
        if verbose:
            print("Gathering orthographic data...")
        self._get_orthography_data(tokens)
        self._sentbreak_count += sum(1 for t in tokens if t.sentbreak)

        # Analyze token pairs for collocations and sentence starters
        if verbose:
            print("Analyzing token pairs...")
            try:
                from tqdm import tqdm

                pairs = list(pair_iter(tokens))
                pair_iter_with_progress = tqdm(pairs, desc="Analyzing token pairs", unit="pair")
            except ImportError:
                pair_iter_with_progress = pair_iter(tokens)
        else:
            pair_iter_with_progress = pair_iter(tokens)

        for token1, token2 in pair_iter_with_progress:
            if not token1.period_final or token2 is None:
                continue
            if self._is_rare_abbrev_type(token1, token2):
                self._params.abbrev_types.add(token1.type_no_period)
            if self._is_potential_sent_starter(token2, token1):
                self._sent_starter_fdist[token2.type] += 1
            if self._is_potential_collocation(token1, token2):
                pair = (token1.type_no_period, token2.type_no_sentperiod)
                self._collocation_fdist[pair] += 1

    def _reclassify_abbrev_types(self, types: Set[str]) -> Iterator[Tuple[str, float, bool]]:
        """
        Reevaluate which token types should be classified as abbreviations.

        Args:
            types: Set of token types to evaluate

        Yields:
            Tuples of (token_type, score, is_add) where is_add indicates whether to add or remove
        """
        for typ in types:
            if not re.search(r"[^\W\d]", typ) or typ == "##number##":
                continue

            # Skip tokens with non-alphanumeric characters (except periods)
            # This excludes tokens with punctuation like #, %, $, etc.
            if re.search(r"[^\w.]", typ):
                continue

            if typ.endswith("."):
                if typ in self._params.abbrev_types:
                    continue
                candidate = typ[:-1]
                is_add = True
            else:
                if typ not in self._params.abbrev_types:
                    continue
                candidate = typ
                is_add = False

            # Skip if candidate length exceeds maximum allowed length
            if len(candidate) > self.MAX_ABBREV_LENGTH:
                if not is_add and candidate in self._params.abbrev_types:
                    # If it's already in abbrev_types but too long, remove it
                    yield candidate, 0.0, False
                continue

            # Allow periods within abbreviation candidates (like U.S.C.)
            # but still reject other non-alphanumeric characters
            if re.search(r"[^\w.]", candidate):
                if not is_add and candidate in self._params.abbrev_types:
                    # If it's already in abbrev_types but has invalid chars, remove it
                    yield candidate, 0.0, False
                continue

            # For candidate with internal periods (like U.S.C), get the non-period characters
            candidate_no_periods = candidate.replace(".", "")

            # Count alphabet and digit characters in the non-period version
            alpha_count = sum(1 for c in candidate_no_periods if c.isalpha())
            digit_count = sum(1 for c in candidate_no_periods if c.isdigit())

            # Must have at least as many letters as digits and at least one letter
            if alpha_count < digit_count or alpha_count == 0:
                if not is_add and candidate in self._params.abbrev_types:
                    # If it's already in abbrev_types but doesn't meet the criteria, remove it
                    yield candidate, 0.0, False
                continue

            num_periods = candidate.count(".") + 1
            num_nonperiods = len(candidate) - candidate.count(".") + 1
            count_with_period = self._type_fdist[candidate + "."]
            count_without_period = self._type_fdist[candidate]
            total = sum(self._type_fdist.values())

            # Check existing abbreviation status
            is_existing_abbrev = candidate in self._params.abbrev_types

            # Apply more lenient scoring for existing abbreviations
            if is_existing_abbrev and self.PERSIST_ABBREVS and not is_add:
                # For existing abbreviations, we use a lower threshold to maintain consistency
                # Only remove if there's strong evidence against it being an abbreviation
                consistency = (
                    count_with_period / (count_with_period + count_without_period)
                    if (count_with_period + count_without_period) > 0
                    else 0
                )
                if consistency >= self.ABBREV_CONSISTENCY:
                    # If word appears consistently with a period, keep it as an abbreviation
                    score = self.ABBREV + 0.1  # Ensure it stays above threshold
                    yield candidate, score, is_add
                    continue

            # Normal calculation for new abbreviations or those that lost consistency
            log_likelihood = dunning_log_likelihood(
                count_with_period + count_without_period,
                self._num_period_toks,
                count_with_period,
                total,
            )
            f_length = math.exp(-num_nonperiods)
            f_periods = num_periods

            # Less aggressive penalty for short words
            if len(candidate) <= 3:
                f_penalty = 1.0  # No penalty for very short words
            else:
                f_penalty = (
                    math.pow(num_nonperiods, -count_without_period * 0.5)
                    if count_without_period
                    else 1
                )

            # Boost score for consistent period usage
            consistency_boost = (
                count_with_period / (count_with_period + count_without_period)
                if (count_with_period + count_without_period) > 0
                else 0
            )

            # Calculate final score with improvements
            score = log_likelihood * f_length * f_periods * f_penalty * (1.0 + consistency_boost)

            yield candidate, score, is_add

    def _get_orthography_data(self, tokens: List[PunktToken]) -> None:
        """
        Gather orthographic context data from tokens.

        Args:
            tokens: The tokens to gather data from
        """
        context = "internal"
        for token in tokens:
            if token.parastart and context != "unknown":
                context = "initial"
            if token.linestart and context == "internal":
                context = "unknown"
            typ = token.type_no_sentperiod
            flag = ORTHO_MAP.get((context, token.first_case), 0)
            if flag:
                self._params.add_ortho_context(typ, flag)
            if token.sentbreak:
                context = "initial" if not (token.is_number or token.is_initial) else "unknown"
            elif token.ellipsis or token.abbr:
                context = "unknown"
            else:
                context = "internal"

    def _is_rare_abbrev_type(self, cur_tok: PunktToken, next_tok: PunktToken) -> bool:
        """
        Check if a token appears to be a rare abbreviation, based on context.

        Args:
            cur_tok: The current token
            next_tok: The next token

        Returns:
            True if the current token is likely a rare abbreviation
        """
        if cur_tok.abbr or not cur_tok.sentbreak:
            return False
        typ = cur_tok.type_no_sentperiod

        # Skip tokens with non-alphanumeric characters (except periods)
        if re.search(r"[^\w.]", typ):
            return False

        # Allow internal periods in abbreviations (like U.S.C.)
        # but still reject other non-alphanumeric characters
        base_typ = typ[:-1] if typ.endswith(".") else typ
        if re.search(r"[^\w.]", base_typ):
            return False

        # For tokens with internal periods (like U.S.C), get the non-period characters
        base_typ_no_periods = base_typ.replace(".", "")

        # Check alphabet vs digit ratio on the non-period version
        alpha_count = sum(1 for c in base_typ_no_periods if c.isalpha())
        digit_count = sum(1 for c in base_typ_no_periods if c.isdigit())
        if alpha_count < digit_count or alpha_count == 0:
            return False

        # Check if the token exceeds maximum abbreviation length
        if len(typ) > self.MAX_ABBREV_LENGTH:
            return False

        count = self._type_fdist[typ] + self._type_fdist.get(typ[:-1], 0)
        if typ in self._params.abbrev_types or count >= self.ABBREV_BACKOFF:
            return False
        if next_tok.tok[0] in self._lang_vars.internal_punctuation:
            return True
        if next_tok.first_lower:
            ortho = self._params.ortho_context.get(next_tok.type_no_sentperiod, 0)
            if (ortho & ORTHO_BEG_UC) and not (ortho & ORTHO_MID_UC):
                return True
        return False

    def _is_potential_collocation(self, tok1: PunktToken, tok2: PunktToken) -> bool:
        """
        Check if two tokens form a potential collocation.

        Args:
            tok1: The first token
            tok2: The second token

        Returns:
            True if the tokens form a potential collocation
        """
        return (
            (
                self.INCLUDE_ALL_COLLOCS
                or (self.INCLUDE_ABBREV_COLLOCS and tok1.abbr)
                or (tok1.sentbreak and (tok1.is_number or tok1.is_initial))
            )
            and tok1.is_non_punct
            and tok2.is_non_punct
        )

    def _is_potential_sent_starter(self, cur_tok: PunktToken, prev_tok: PunktToken) -> bool:
        """
        Check if a token is a potential sentence starter.

        Args:
            cur_tok: The current token
            prev_tok: The previous token

        Returns:
            True if the current token is a potential sentence starter
        """
        return (
            prev_tok.sentbreak
            and not (prev_tok.is_number or prev_tok.is_initial)
            and cur_tok.is_alpha
        )

    def finalize_training(
        self, verbose: bool = False, preserve_common_abbrevs: bool = True
    ) -> None:
        """
        Finalize the training by identifying sentence starters and collocations.

        Args:
            verbose: Whether to display progress information
            preserve_common_abbrevs: Whether to preserve common abbreviations
        """
        if verbose:
            print("Finalizing training...")
            print("Identifying sentence starters...")

        # Store common abbreviations to ensure they're preserved
        common_abbrevs = set(self.COMMON_ABBREVS) if preserve_common_abbrevs else set()

        self._params.sent_starters.clear()

        if verbose:
            try:
                from tqdm import tqdm

                # Convert to list to show progress
                sent_starters = list(self._find_sent_starters())
                starter_iter = tqdm(sent_starters, desc="Finding sentence starters", unit="starter")
            except ImportError:
                starter_iter = self._find_sent_starters()
        else:
            starter_iter = self._find_sent_starters()

        for typ, ll in starter_iter:
            self._params.sent_starters.add(typ)

        if verbose:
            print(f"Found {len(self._params.sent_starters)} sentence starters.")
            print("Identifying collocations...")

        self._params.collocations.clear()

        if verbose:
            try:
                from tqdm import tqdm

                # Convert to list to show progress
                collocations = list(self._find_collocations())
                collocation_iter = tqdm(
                    collocations, desc="Finding collocations", unit="collocation"
                )
            except ImportError:
                collocation_iter = self._find_collocations()
        else:
            collocation_iter = self._find_collocations()

        for (typ1, typ2), ll in collocation_iter:
            self._params.collocations.add((typ1, typ2))

        # Ensure common abbreviations are preserved after statistical analysis
        if preserve_common_abbrevs:
            original_count = len(self._params.abbrev_types)
            for abbr in common_abbrevs:
                self._params.abbrev_types.add(abbr)
            if verbose and len(self._params.abbrev_types) > original_count:
                print(
                    f"Restored {len(self._params.abbrev_types) - original_count} common abbreviations."
                )

        if verbose:
            print(f"Found {len(self._params.collocations)} collocations.")
            print(f"Final abbreviation count: {len(self._params.abbrev_types)}")

            # Sort abbreviations by frequency
            if self._params.abbrev_types:
                print("\nMost common abbreviations (with frequency):")
                abbrev_freqs = [
                    (abbr, self._type_fdist.get(abbr, 0) + self._type_fdist.get(abbr + ".", 0))
                    for abbr in self._params.abbrev_types
                ]
                abbrev_freqs.sort(key=lambda x: x[1], reverse=True)

                # Show top 20 abbreviations or all if fewer
                for abbr, freq in abbrev_freqs[:20]:
                    source = " (built-in)" if abbr in common_abbrevs else ""
                    print(f"  {abbr:<10} {freq:>5}{source}")

            # Sort collocations by frequency
            if self._params.collocations:
                print("\nMost common collocations (with frequency):")
                colloc_freqs = [
                    (colloc, self._collocation_fdist.get(colloc, 0))
                    for colloc in self._params.collocations
                ]
                colloc_freqs.sort(key=lambda x: x[1], reverse=True)

                # Show top 20 collocations or all if fewer
                for (word1, word2), freq in colloc_freqs[:20]:
                    print(f"  {word1} {word2:<15} {freq:>5}")

            print("\nTraining complete.")

        self._finalized = True

    def _find_collocations(self) -> Iterator[Tuple[Tuple[str, str], float]]:
        """
        Find collocations in the training data.

        Yields:
            Tuples of ((token1, token2), score) for collocations
        """
        total = sum(self._type_fdist.values())
        for pair, col_count in self._collocation_fdist.items():
            typ1, typ2 = pair
            if typ2 in self._params.sent_starters:
                continue
            typ1_count = self._type_fdist[typ1] + self._type_fdist[typ1 + "."]
            typ2_count = self._type_fdist[typ2] + self._type_fdist[typ2 + "."]
            if (
                typ1_count > 1
                and typ2_count > 1
                and col_count >= self.MIN_COLLOC_FREQ
                and col_count <= min(typ1_count, typ2_count)
            ):
                ll = collocation_log_likelihood(typ1_count, typ2_count, col_count, total)
                if ll >= self.COLLOCATION and (total / typ1_count > typ2_count / col_count):
                    yield (typ1, typ2), ll

    def _find_sent_starters(self) -> Iterator[Tuple[str, float]]:
        """
        Find sentence starters in the training data.

        Yields:
            Tuples of (token_type, score) for sentence starters
        """
        total = sum(self._type_fdist.values())
        for typ, count in self._sent_starter_fdist.items():
            if not typ:
                continue
            typ_count = self._type_fdist[typ] + self._type_fdist[typ + "."]
            # Apply minimum frequency threshold and ensure consistency
            if typ_count < count or count < self.MIN_COLLOC_FREQ:
                continue
            ll = collocation_log_likelihood(self._sentbreak_count, typ_count, count, total)
            if ll >= self.SENT_STARTER and (total / self._sentbreak_count > typ_count / count):
                yield typ, ll

    def to_json(self) -> Dict[str, Any]:
        """
        Convert trainer configuration and parameters to a JSON-serializable dictionary.

        Returns:
            A JSON-serializable dictionary with trainer config and parameters
        """
        # Make sure training is finalized
        if not self._finalized:
            self.finalize_training()

        config = {
            # Configuration parameters
            self.CONFIG_ABBREV: self.ABBREV,
            self.CONFIG_ABBREV_BACKOFF: self.ABBREV_BACKOFF,
            self.CONFIG_COLLOCATION: self.COLLOCATION,
            self.CONFIG_SENT_STARTER: self.SENT_STARTER,
            self.CONFIG_INCLUDE_ALL_COLLOCS: self.INCLUDE_ALL_COLLOCS,
            self.CONFIG_INCLUDE_ABBREV_COLLOCS: self.INCLUDE_ABBREV_COLLOCS,
            self.CONFIG_MIN_COLLOC_FREQ: self.MIN_COLLOC_FREQ,
            self.CONFIG_MAX_ABBREV_LENGTH: self.MAX_ABBREV_LENGTH,
            self.CONFIG_COMMON_ABBREVS: self.COMMON_ABBREVS,
            # Current parameters (trained model)
            "parameters": self._params.to_json(),
            # Metadata
            "version": "0.2.0",
            "description": "nupunkt sentence tokenizer model",
        }
        return config

    @classmethod
    def from_json(
        cls,
        data: Dict[str, Any],
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Optional[Type[PunktToken]] = None,
    ) -> "PunktTrainer":
        """
        Create a PunktTrainer instance from a JSON dictionary.

        Args:
            data: The JSON dictionary
            lang_vars: Optional language variables
            token_cls: Optional token class

        Returns:
            A new PunktTrainer instance
        """
        # Create a new instance
        trainer = cls(lang_vars=lang_vars, token_cls=token_cls or PunktToken)

        # Set configuration parameters
        trainer.ABBREV = data.get(cls.CONFIG_ABBREV, cls.ABBREV)
        trainer.ABBREV_BACKOFF = data.get(cls.CONFIG_ABBREV_BACKOFF, cls.ABBREV_BACKOFF)
        trainer.COLLOCATION = data.get(cls.CONFIG_COLLOCATION, cls.COLLOCATION)
        trainer.SENT_STARTER = data.get(cls.CONFIG_SENT_STARTER, cls.SENT_STARTER)
        trainer.INCLUDE_ALL_COLLOCS = data.get(
            cls.CONFIG_INCLUDE_ALL_COLLOCS, cls.INCLUDE_ALL_COLLOCS
        )
        trainer.INCLUDE_ABBREV_COLLOCS = data.get(
            cls.CONFIG_INCLUDE_ABBREV_COLLOCS, cls.INCLUDE_ABBREV_COLLOCS
        )
        trainer.MIN_COLLOC_FREQ = data.get(cls.CONFIG_MIN_COLLOC_FREQ, cls.MIN_COLLOC_FREQ)
        trainer.MAX_ABBREV_LENGTH = data.get(cls.CONFIG_MAX_ABBREV_LENGTH, cls.MAX_ABBREV_LENGTH)

        # Load custom common abbreviations if provided
        if cls.CONFIG_COMMON_ABBREVS in data:
            trainer.COMMON_ABBREVS = data[cls.CONFIG_COMMON_ABBREVS]

        # Load parameters if available
        if "parameters" in data:
            trainer._params = PunktParameters.from_json(data["parameters"])
            trainer._finalized = True

        return trainer

    def save(
        self, file_path: Union[str, Path], compress: bool = True, compression_level: int = 1
    ) -> None:
        """
        Save trainer configuration and parameters to a JSON file, optionally with LZMA compression.

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
    ) -> "PunktTrainer":
        """
        Load trainer configuration and parameters from a JSON file, which may be compressed with LZMA.

        Args:
            file_path: The path to load the file from
            lang_vars: Optional language variables
            token_cls: Optional token class

        Returns:
            A new PunktTrainer instance
        """
        from nupunkt.utils.compression import load_compressed_json

        data = load_compressed_json(file_path)
        return cls.from_json(data, lang_vars, token_cls)
