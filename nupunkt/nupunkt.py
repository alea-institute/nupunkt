from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

# -------------------------------------------------------------------
# Orthographic Context Constants
# -------------------------------------------------------------------

_ORTHO_BEG_UC = 1 << 1
_ORTHO_MID_UC = 1 << 2
_ORTHO_UNK_UC = 1 << 3
_ORTHO_BEG_LC = 1 << 4
_ORTHO_MID_LC = 1 << 5
_ORTHO_UNK_LC = 1 << 6

_ORTHO_UC = _ORTHO_BEG_UC | _ORTHO_MID_UC | _ORTHO_UNK_UC
_ORTHO_LC = _ORTHO_BEG_LC | _ORTHO_MID_LC | _ORTHO_UNK_LC

_ORTHO_MAP: Dict[Tuple[str, str], int] = {
    ("initial", "upper"): _ORTHO_BEG_UC,
    ("internal", "upper"): _ORTHO_MID_UC,
    ("unknown", "upper"): _ORTHO_UNK_UC,
    ("initial", "lower"): _ORTHO_BEG_LC,
    ("internal", "lower"): _ORTHO_MID_LC,
    ("unknown", "lower"): _ORTHO_UNK_LC,
}

# -------------------------------------------------------------------
# Language Variables
# -------------------------------------------------------------------


class PunktLanguageVars:
    sent_end_chars: Tuple[str, ...] = (".", "?", "!")
    internal_punctuation: str = ",:;"
    re_boundary_realignment: re.Pattern = re.compile(r'["\')\]}]+?(?:\s+|(?=--)|$)', re.MULTILINE)
    _re_word_start: str = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    _re_multi_char_punct: str = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"

    def __init__(self) -> None:
        self._re_period_context: Optional[re.Pattern] = None
        self._re_word_tokenizer: Optional[re.Pattern] = None

    @property
    def _re_sent_end_chars(self) -> str:
        return f"[{re.escape(''.join(self.sent_end_chars))}]"

    @property
    def _re_non_word_chars(self) -> str:
        # Exclude characters that can never start a word
        nonword = "".join(set(self.sent_end_chars) - {"."})
        return rf"(?:[)\";}}\]\*:@\'\({{[\s{re.escape(nonword)}])"

    @property
    def word_tokenize_pattern(self) -> re.Pattern:
        if self._re_word_tokenizer is None:
            pattern = rf"""(
                {self._re_multi_char_punct}
                |
                (?={self._re_word_start})\S+?
                (?=
                    \s|
                    $|
                    {self._re_non_word_chars}|
                    {self._re_multi_char_punct}|
                    ,(?=$|\s|{self._re_non_word_chars}|{self._re_multi_char_punct})
                )
                |
                \S
            )"""
            self._re_word_tokenizer = re.compile(pattern, re.UNICODE | re.VERBOSE)
        return self._re_word_tokenizer

    def word_tokenize(self, text: str) -> List[str]:
        return self.word_tokenize_pattern.findall(text)

    @property
    def period_context_pattern(self) -> re.Pattern:
        if self._re_period_context is None:
            pattern = rf"""
                {self._re_sent_end_chars}
                (?=(?P<after_tok>
                    {self._re_non_word_chars}|
                    \s+(?P<next_tok>\S+)
                ))
            """
            self._re_period_context = re.compile(pattern, re.UNICODE | re.VERBOSE)
        return self._re_period_context


# -------------------------------------------------------------------
# Punkt Parameters and Token Data Structures
# -------------------------------------------------------------------


@dataclass
class PunktParameters:
    abbrev_types: Set[str] = field(default_factory=set)
    collocations: Set[Tuple[str, str]] = field(default_factory=set)
    sent_starters: Set[str] = field(default_factory=set)
    ortho_context: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_ortho_context(self, typ: str, flag: int) -> None:
        self.ortho_context[typ] |= flag

    def to_json(self) -> Dict[str, Any]:
        """Convert parameters to a JSON-serializable dictionary."""
        return {
            "abbrev_types": sorted(self.abbrev_types),
            "collocations": sorted([[c[0], c[1]] for c in self.collocations]),
            "sent_starters": sorted(self.sent_starters),
            "ortho_context": dict(self.ortho_context.items()),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> PunktParameters:
        """Create a PunktParameters instance from a JSON dictionary."""
        params = cls()
        params.abbrev_types = set(data.get("abbrev_types", []))
        params.collocations = {tuple(c) for c in data.get("collocations", [])}
        params.sent_starters = set(data.get("sent_starters", []))
        params.ortho_context = defaultdict(int)
        for k, v in data.get("ortho_context", {}).items():
            params.ortho_context[k] = int(v)  # Ensure value is int
        return params

    def save(self, file_path: str) -> None:
        """Save parameters to a JSON file."""
        with Path(file_path).open("w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> PunktParameters:
        """Load parameters from a JSON file."""
        with Path(file_path).open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data)


@dataclass
class PunktToken:
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
            and not re.search(r"[^\w.]", self.tok)
            and self.type != "##number##"
            and alpha_count >= digit_count  # Must have at least as many letters as digits
            and alpha_count > 0  # Must have at least one letter
        )

        # If token has a period but contains other punctuation, it can't be an abbreviation
        if self.period_final and not self.valid_abbrev_candidate:
            self.abbr = False

    @staticmethod
    def _get_type(tok: str) -> str:
        # Normalize numbers
        if re.match(r"^-?[\.,]?\d[\d,\.-]*\.?$", tok):
            return "##number##"
        return tok.lower()

    @property
    def type_no_period(self) -> str:
        return self.type[:-1] if self.type.endswith(".") and len(self.type) > 1 else self.type

    @property
    def type_no_sentperiod(self) -> str:
        return self.type_no_period if self.sentbreak else self.type

    @property
    def first_upper(self) -> bool:
        return bool(self.tok) and self.tok[0].isupper()

    @property
    def first_lower(self) -> bool:
        return bool(self.tok) and self.tok[0].islower()

    @property
    def first_case(self) -> str:
        if self.first_lower:
            return "lower"
        if self.first_upper:
            return "upper"
        return "none"

    @property
    def is_ellipsis(self) -> bool:
        return bool(re.search(r"\.\.+$", self.tok))

    @property
    def is_number(self) -> bool:
        return self.type.startswith("##number##")

    @property
    def is_initial(self) -> bool:
        return bool(re.fullmatch(r"[^\W\d]\.", self.tok))

    @property
    def is_alpha(self) -> bool:
        return bool(re.fullmatch(r"[^\W\d]+", self.tok))

    @property
    def is_non_punct(self) -> bool:
        return bool(re.search(r"[^\W\d]", self.type))

    def __str__(self) -> str:
        s = self.tok
        if self.abbr:
            s += "<A>"
        if self.ellipsis:
            s += "<E>"
        if self.sentbreak:
            s += "<S>"
        return s


# -------------------------------------------------------------------
# Helper: Pairwise Iteration
# -------------------------------------------------------------------


def pair_iter(iterable: Iterator[Any]) -> Iterator[Tuple[Any, Optional[Any]]]:
    it = iter(iterable)
    prev = next(it, None)
    if prev is None:
        return
    for current in it:
        yield prev, current
        prev = current
    yield prev, None


# -------------------------------------------------------------------
# Punkt Base Class
# -------------------------------------------------------------------


class PunktBase:
    def __init__(
        self,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: type[PunktToken] = PunktToken,
        params: Optional[PunktParameters] = None,
    ) -> None:
        self._lang_vars = lang_vars or PunktLanguageVars()
        self._Token = token_cls
        self._params = params or PunktParameters()

    def _tokenize_words(self, plaintext: str) -> Iterator[PunktToken]:
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
        for token in tokens:
            self._first_pass_annotation(token)
            yield token

    def _first_pass_annotation(self, token: PunktToken) -> None:
        if token.tok in self._lang_vars.sent_end_chars:
            token.sentbreak = True
        elif token.is_ellipsis:
            token.ellipsis = True
        elif token.period_final and not token.tok.endswith(".."):
            # If token is not a valid abbreviation candidate, mark it as a sentence break
            if not token.valid_abbrev_candidate:
                token.sentbreak = True
            else:
                # For valid candidates, check if they are known abbreviations
                candidate = token.tok[:-1].lower()

                # Check if the token itself is a known abbreviation
                if (
                    candidate in self._params.abbrev_types
                    or "-" in candidate
                    and candidate.split("-")[-1] in self._params.abbrev_types
                    or "." in candidate
                    and candidate.replace(".", "") in self._params.abbrev_types
                ):
                    token.abbr = True
                else:
                    token.sentbreak = True


# -------------------------------------------------------------------
# Punkt Trainer (Learning Parameters)
# -------------------------------------------------------------------


class PunktTrainer(PunktBase):
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
    COMMON_ABBREVS: ClassVar[List[str]] = []

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
        token_cls: type[PunktToken] = PunktToken,
        include_common_abbrevs: bool = True,  # Whether to include common abbreviations by default
    ) -> None:
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
            print("Tokenizing text...")
            # Check for tqdm using importlib instead of direct import
            try:
                import importlib.util

                if importlib.util.find_spec("tqdm") is not None:
                    pass  # tqdm is available
                else:
                    print("Note: Install tqdm for progress bars during training.")
            except (ImportError, AttributeError):
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
            log_likelihood = self._dunning_log_likelihood(
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

            # Debugger for common abbreviations
            if candidate in ["dr", "mr", "inc"] and not is_add:
                # We deliberately do not print this in regular usage, but it's
                # preserved for development debugging when needed
                pass

            yield candidate, score, is_add

    def _dunning_log_likelihood(self, count_a: int, count_b: int, count_ab: int, N: int) -> float:
        """
        Modified Dunning log-likelihood calculation that gives higher weight to
        potential abbreviations. This makes the model more likely to detect abbreviations,
        especially in larger datasets where evidence may be diluted.
        """
        p1 = count_b / N
        p2 = 0.99
        null_hypo = count_ab * math.log(p1 + 1e-8) + (count_a - count_ab) * math.log(
            1.0 - p1 + 1e-8
        )
        alt_hypo = count_ab * math.log(p2) + (count_a - count_ab) * math.log(1.0 - p2)

        # Basic log likelihood calculation
        ll = -2.0 * (null_hypo - alt_hypo)

        # Boosting factor for short tokens (likely abbreviations)
        # This makes the algorithm more sensitive to abbreviation detection
        return ll * 1.5

    @staticmethod
    def _col_log_likelihood(count_a: int, count_b: int, count_ab: int, N: int) -> float:
        p = count_b / N
        p1 = count_ab / count_a if count_a else 0
        try:
            p2 = (count_b - count_ab) / (N - count_a) if (N - count_a) else 0
        except ZeroDivisionError:
            p2 = 1
        try:
            summand1 = count_ab * math.log(p) + (count_a - count_ab) * math.log(1.0 - p)
        except ValueError:
            summand1 = 0
        try:
            summand2 = (count_b - count_ab) * math.log(p) + (
                N - count_a - count_b + count_ab
            ) * math.log(1.0 - p)
        except ValueError:
            summand2 = 0
        summand3 = (
            0
            if count_a == count_ab or p1 <= 0 or p1 >= 1
            else count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)
        )
        summand4 = (
            0
            if count_b == count_ab or p2 <= 0 or p2 >= 1
            else (count_b - count_ab) * math.log(p2)
            + (N - count_a - count_b + count_ab) * math.log(1.0 - p2)
        )
        return -2.0 * (summand1 + summand2 - summand3 - summand4)

    def _get_orthography_data(self, tokens: List[PunktToken]) -> None:
        context = "internal"
        for token in tokens:
            if token.parastart and context != "unknown":
                context = "initial"
            if token.linestart and context == "internal":
                context = "unknown"
            typ = token.type_no_sentperiod
            flag = _ORTHO_MAP.get((context, token.first_case), 0)
            if flag:
                self._params.add_ortho_context(typ, flag)
            if token.sentbreak:
                context = "initial" if not (token.is_number or token.is_initial) else "unknown"
            elif token.ellipsis or token.abbr:
                context = "unknown"
            else:
                context = "internal"

    def _is_rare_abbrev_type(self, cur_tok: PunktToken, next_tok: PunktToken) -> bool:
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
            if (ortho & _ORTHO_BEG_UC) and not (ortho & _ORTHO_MID_UC):
                return True
        return False

    def _is_potential_collocation(self, tok1: PunktToken, tok2: PunktToken) -> bool:
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
        return (
            prev_tok.sentbreak
            and not (prev_tok.is_number or prev_tok.is_initial)
            and cur_tok.is_alpha
        )

    def finalize_training(
        self, verbose: bool = False, preserve_common_abbrevs: bool = True
    ) -> None:
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

        for typ, _ll in starter_iter:
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

        for (typ1, typ2), _ll in collocation_iter:
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
                ll = self._col_log_likelihood(typ1_count, typ2_count, col_count, total)
                if ll >= self.COLLOCATION and (total / typ1_count > typ2_count / col_count):
                    yield (typ1, typ2), ll

    def _find_sent_starters(self) -> Iterator[Tuple[str, float]]:
        total = sum(self._type_fdist.values())
        for typ, count in self._sent_starter_fdist.items():
            if not typ:
                continue
            typ_count = self._type_fdist[typ] + self._type_fdist[typ + "."]
            # Apply minimum frequency threshold and ensure consistency
            if typ_count < count or count < self.MIN_COLLOC_FREQ:
                continue
            ll = self._col_log_likelihood(self._sentbreak_count, typ_count, count, total)
            if ll >= self.SENT_STARTER and (total / self._sentbreak_count > typ_count / count):
                yield typ, ll

    def to_json(self) -> Dict[str, Any]:
        """Convert trainer configuration and parameters to a JSON-serializable dictionary."""
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
    ) -> PunktTrainer:
        """Create a PunktTrainer instance from a JSON dictionary."""
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

    def save(self, file_path: str) -> None:
        """Save trainer configuration and parameters to a JSON file."""
        with Path(file_path).open("w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        file_path: str,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Optional[Type[PunktToken]] = None,
    ) -> PunktTrainer:
        """Load trainer configuration and parameters from a JSON file."""
        with Path(file_path).open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data, lang_vars, token_cls)

    # End of PunktTrainer


# -------------------------------------------------------------------
# Punkt Sentence Tokenizer
# -------------------------------------------------------------------


class PunktSentenceTokenizer(PunktBase):
    def __init__(
        self,
        train_text: Optional[str] = None,
        verbose: bool = False,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: type[PunktToken] = PunktToken,
        include_common_abbrevs: bool = True,  # Whether to include common abbreviations
    ) -> None:
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
        """Convert the tokenizer to a JSON-serializable dictionary."""
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
    ) -> PunktSentenceTokenizer:
        """Create a PunktSentenceTokenizer from a JSON dictionary."""
        # First create a trainer from the JSON data
        trainer = PunktTrainer.from_json(data, lang_vars, token_cls)

        # Then create a tokenizer with the parameters
        return cls(trainer.get_params(), lang_vars=lang_vars, token_cls=token_cls or PunktToken)

    def save(self, file_path: str) -> None:
        """Save the tokenizer to a JSON file."""
        with Path(file_path).open("w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        file_path: str,
        lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Optional[Type[PunktToken]] = None,
    ) -> PunktSentenceTokenizer:
        """Load a PunktSentenceTokenizer from a JSON file."""
        with Path(file_path).open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json(data, lang_vars, token_cls)

    def reconfigure(self, config: Dict[str, Any]) -> None:
        """Reconfigure the tokenizer with new settings."""
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
        return list(self.sentences_from_text(text, realign_boundaries))

    def span_tokenize(
        self, text: str, realign_boundaries: bool = True
    ) -> Iterator[Tuple[int, int]]:
        slices = list(self._slices_from_text(text))
        if realign_boundaries:
            slices = list(self._realign_boundaries(text, slices))
        for s in slices:
            yield (s.start, s.stop)

    def sentences_from_text(self, text: str, realign_boundaries: bool = True) -> List[str]:
        return [text[start:stop] for start, stop in self.span_tokenize(text, realign_boundaries)]

    def _get_last_whitespace_index(self, text: str) -> int:
        for i in range(len(text) - 1, -1, -1):
            if text[i].isspace():
                return i
        return 0

    def _match_potential_end_contexts(self, text: str) -> Iterator[Tuple[re.Match, str]]:
        previous_slice = slice(0, 0)
        previous_match: Optional[re.Match] = None
        for match in self._lang_vars.period_context_pattern.finditer(text):
            before_text = text[previous_slice.stop : match.start()]
            idx = self._get_last_whitespace_index(before_text)
            index_after_last_space = previous_slice.stop + idx + 1 if idx else previous_slice.start
            prev_word_slice = slice(index_after_last_space, match.start())
            if previous_match and previous_slice.stop <= prev_word_slice.start:
                yield (
                    previous_match,
                    text[previous_slice]
                    + previous_match.group()
                    + previous_match.group("after_tok"),
                )
            previous_match = match
            previous_slice = prev_word_slice
        if previous_match:
            yield (
                previous_match,
                text[previous_slice] + previous_match.group() + previous_match.group("after_tok"),
            )

    def _slices_from_text(self, text: str) -> Iterator[slice]:
        last_break = 0
        for match, context in self._match_potential_end_contexts(text):
            if self.text_contains_sentbreak(context):
                yield slice(last_break, match.end())
                last_break = match.start("next_tok") if match.group("next_tok") else match.end()
        yield slice(last_break, len(text.rstrip()))

    def _realign_boundaries(self, text: str, slices: List[slice]) -> Iterator[slice]:
        realign = 0
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
        tokens = list(self._annotate_tokens(self._tokenize_words(text)))
        return any(token.sentbreak for token in tokens)

    def _annotate_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        tokens = self._annotate_first_pass(tokens)
        tokens = self._annotate_second_pass(tokens)
        return tokens

    def _annotate_second_pass(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
        for token1, token2 in pair_iter(tokens):
            self._second_pass_annotation(token1, token2)
            yield token1

    def _second_pass_annotation(
        self, token1: PunktToken, token2: Optional[PunktToken]
    ) -> Optional[str]:
        if token2 is None or not token1.period_final:
            return None
        typ = token1.type_no_period
        next_typ = token2.type_no_sentperiod
        tok_is_initial = token1.is_initial

        # Collocation heuristic: if the pair is known, mark token as abbreviation.
        if (typ, next_typ) in self._params.collocations:
            token1.sentbreak = False
            token1.abbr = True
            return "Known collocation"

        # If token is marked as an abbreviation/ellipsis, decide based on orthographic evidence.
        if (token1.abbr or token1.ellipsis) and (not tok_is_initial):
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
            if (
                is_sent_starter == "unknown"
                and tok_is_initial
                and token2.first_upper
                and not (self._params.ortho_context.get(next_typ, 0) & _ORTHO_LC)
            ):
                token1.sentbreak = False
                token1.abbr = True
                return "Initial with special orthographic heuristic"
        return None

    def _ortho_heuristic(self, token: PunktToken) -> Union[bool, str]:
        if token.tok in (";", ":", ",", ".", "!", "?"):
            return False
        ortho = self._params.ortho_context.get(token.type_no_sentperiod, 0)
        if token.first_upper and (ortho & _ORTHO_LC) and not (ortho & _ORTHO_MID_UC):
            return True
        if token.first_lower and ((ortho & _ORTHO_UC) or not (ortho & _ORTHO_BEG_LC)):
            return False
        return "unknown"


# -------------------------------------------------------------------
# Example Usage (Demo)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import gzip
    import json

    train_text = ""
    with gzip.open("data/train2.jsonl.gz", "rt", encoding="utf-8") as input_file:
        for i, line in enumerate(input_file):
            train_text += json.loads(line).get("text", "") + "\n"
            if i > 100:
                break

    # Train a model on the demo text
    trainer = PunktTrainer(train_text, verbose=True)
    params = trainer.get_params()

    # Create a tokenizer with the trained parameters
    tokenizer = PunktSentenceTokenizer(params)
    sentences = tokenizer.tokenize(
        "This is a test. I hope Dr. Johnson likes his first million. Under 18 U.S.C. 12, he won't believe it! Let's see if it works. "
    )

    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")
