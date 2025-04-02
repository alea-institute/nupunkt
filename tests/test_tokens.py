"""Unit tests for nupunkt token module."""

import pytest

from nupunkt.core.tokens import PunktToken, create_punkt_token


def test_punkt_token_basic():
    """Test basic properties of PunktToken."""
    # Simple token
    token = PunktToken("word")
    assert token.tok == "word"
    assert token.type == "word"
    assert not token.period_final

    # Token with period
    token = PunktToken("word.")
    assert token.tok == "word."
    assert token.type == "word."
    assert token.period_final


def test_punkt_token_attributes():
    """Test PunktToken attributes."""
    # Token with parameters
    token = PunktToken("word", parastart=True, linestart=True)
    assert token.parastart
    assert token.linestart
    assert not token.sentbreak
    assert not token.abbr
    assert not token.ellipsis


def test_punkt_token_type_methods():
    """Test PunktToken type methods."""
    # Test type_no_period
    token = PunktToken("word.")
    assert token.type_no_period == "word"

    # Test type_no_sentperiod
    token = PunktToken("word.")
    token.sentbreak = True
    assert token.type_no_sentperiod == "word"

    token = PunktToken("word.")
    token.sentbreak = False
    assert token.type_no_sentperiod == "word."


def test_punkt_token_case_properties():
    """Test PunktToken case detection properties."""
    # Test first_upper
    token = PunktToken("Word")
    assert token.first_upper
    assert not token.first_lower

    # Test first_lower
    token = PunktToken("word")
    assert not token.first_upper
    assert token.first_lower

    # Test first_case
    token = PunktToken("Word")
    assert token.first_case == "upper"

    token = PunktToken("word")
    assert token.first_case == "lower"

    token = PunktToken("123")
    assert token.first_case == "none"


def test_punkt_token_special_types():
    """Test special type detection in PunktToken."""
    # Test is_ellipsis with standard ASCII ellipsis
    token = PunktToken("...")
    assert token.is_ellipsis

    # Test is_ellipsis with Unicode ellipsis character
    token = PunktToken("\u2026")
    assert token.is_ellipsis

    # Test is_ellipsis with Unicode ellipsis at end of word
    token = PunktToken("word\u2026")
    assert token.is_ellipsis

    # Test is_number
    token = PunktToken("123")
    assert token.is_number

    token = PunktToken("3.14")
    assert token.is_number

    # Test is_initial
    token = PunktToken("A.")
    assert token.is_initial

    # Test is_alpha
    token = PunktToken("word")
    assert token.is_alpha

    # Test is_non_punct
    token = PunktToken("word")
    assert token.is_non_punct

    token = PunktToken(".")
    assert not token.is_non_punct


@pytest.mark.benchmark(group="tokens")
def test_token_creation_benchmark(benchmark):
    """Benchmark token creation."""

    def create_tokens():
        tokens = []
        for i in range(1000):
            if i % 3 == 0:
                # Regular word
                token = PunktToken(f"word{i}")
            elif i % 3 == 1:
                # Word with period
                token = PunktToken(f"abbrev{i}.")
            else:
                # Mixed case with punctuation
                token = PunktToken(f"Mixed{i}!")
            tokens.append(token)
        return tokens

    # Run the benchmark
    tokens = benchmark(create_tokens)

    # Simple verification
    assert len(tokens) == 1000
    assert tokens[0].tok == "word0"
    assert tokens[1].tok == "abbrev1."
    assert tokens[2].tok == "Mixed2!"


@pytest.mark.benchmark(group="tokens")
def test_token_factory_benchmark(benchmark):
    """Benchmark token creation through the factory function."""

    def create_tokens_with_factory():
        tokens = []
        for i in range(1000):
            if i % 3 == 0:
                # Regular word
                token = create_punkt_token(f"word{i}")
            elif i % 3 == 1:
                # Word with period
                token = create_punkt_token(f"abbrev{i}.")
            else:
                # Mixed case with punctuation
                token = create_punkt_token(f"Mixed{i}!")
            tokens.append(token)
        return tokens

    # Run the benchmark
    tokens = benchmark(create_tokens_with_factory)

    # Simple verification
    assert len(tokens) == 1000
    assert tokens[0].tok == "word0"
    assert tokens[1].tok == "abbrev1."
    assert tokens[2].tok == "Mixed2!"


@pytest.mark.benchmark(group="tokens")
def test_token_factory_with_cache_benchmark(benchmark):
    """Benchmark token creation through the factory function with cache hits."""

    # Pre-populate the token cache with some common tokens
    common_tokens = [
        "the",
        "of",
        "and",
        "a",
        "to",
        "in",
        "is",
        "that",
        "it",
        "was",
        "for",
        "on",
        "are",
        "as",
        "with",
        "his",
        "they",
        "at",
        "be",
        "this",
        "from",
        "have",
        "or",
        "by",
        "one",
        "had",
        "not",
        "but",
        "what",
        "all",
    ]
    for tok in common_tokens:
        create_punkt_token(tok)

    def create_tokens_with_cache():
        tokens = []
        # First add common tokens that should hit the cache
        for tok in common_tokens:
            tokens.append(create_punkt_token(tok))

        # Then add some new tokens
        for i in range(500):
            if i % 3 == 0:
                # Regular word
                token = create_punkt_token(f"word{i}")
            elif i % 3 == 1:
                # Word with period
                token = create_punkt_token(f"abbrev{i}.")
            else:
                # Mixed case with punctuation
                token = create_punkt_token(f"Mixed{i}!")
            tokens.append(token)
        return tokens

    # Run the benchmark
    tokens = benchmark(create_tokens_with_cache)

    # Simple verification
    assert len(tokens) == 30 + 500
    assert tokens[0].tok == "the"


@pytest.mark.benchmark(group="tokens")
def test_token_property_access_benchmark(benchmark):
    """Benchmark access to token properties."""
    # Create a variety of tokens first
    tokens = []
    for i in range(1000):
        if i % 4 == 0:
            tokens.append(PunktToken(f"Word{i}"))
        elif i % 4 == 1:
            tokens.append(PunktToken(f"abbrev{i}."))
        elif i % 4 == 2:
            tokens.append(PunktToken(f"{i}.{i}"))
        else:
            tokens.append(PunktToken(f"...{i}"))

    def access_properties():
        results = []
        for token in tokens:
            # Access various properties
            props = (
                token.type,
                token.period_final,
                token.is_ellipsis,
                token.is_number,
                token.is_initial,
                token.is_alpha,
                token.is_non_punct,
                token.first_case,
            )
            results.append(props)
        return results

    # Run the benchmark
    results = benchmark(access_properties)

    # Simple verification
    assert len(results) == 1000


@pytest.mark.benchmark(group="tokens")
def test_lazy_property_access_benchmark(benchmark):
    """Benchmark access to lazily evaluated properties."""
    # Create tokens using the factory function
    tokens = []
    for i in range(1000):
        if i % 4 == 0:
            tokens.append(create_punkt_token(f"Word{i}"))
        elif i % 4 == 1:
            tokens.append(create_punkt_token(f"abbrev{i}."))
        elif i % 4 == 2:
            tokens.append(create_punkt_token(f"{i}.{i}"))
        else:
            tokens.append(create_punkt_token(f"...{i}"))

    def access_lazy_properties():
        results = []
        for token in tokens:
            # Access lazily evaluated properties
            props = (
                token.type_no_period,
                token.type_no_sentperiod,
                token.is_ellipsis,
                token.is_number,
                token.is_initial,
                token.is_alpha,
                token.is_non_punct,
            )
            results.append(props)
        return results

    # Run the benchmark
    results = benchmark(access_lazy_properties)

    # Simple verification
    assert len(results) == 1000


def test_slots_vs_dict():
    """Test memory efficiency of slots vs dict."""
    # Create a token using the slots-based class
    token = PunktToken("test")

    # Verify it uses slots
    assert not hasattr(token, "__dict__")

    # Verify all attributes are accessible
    assert token.tok == "test"
    assert token.parastart is False
    assert token.linestart is False
    assert token.sentbreak is False
    assert token.abbr is False
    assert token.ellipsis is False
    assert token.period_final is False
    assert token.type == "test"

    # Test lazy property initialization
    assert token.is_ellipsis is False
    # After access, the property should be calculated and cached
    assert token._is_ellipsis is False
