"""
nupunkt is a Python library for sentence and paragraph boundary detection based on the Punkt algorithm.

It learns to identify sentence boundaries in text, even when periods are used for
abbreviations, ellipses, and other non-sentence-ending contexts. It also supports
paragraph detection based on sentence boundaries and newlines.
"""

# Core classes
from functools import lru_cache
from pathlib import Path

# Import for type annotations
from typing import Any, List, Literal, Tuple, Union, overload

from nupunkt._version import __version__
from nupunkt.core.language_vars import PunktLanguageVars
from nupunkt.core.parameters import PunktParameters
from nupunkt.core.tokens import PunktToken

# Models
from nupunkt.models import load_default_model
from nupunkt.tokenizers.paragraph_tokenizer import PunktParagraphTokenizer

# Tokenizers
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

# Trainers
from nupunkt.trainers.base_trainer import PunktTrainer


@lru_cache(maxsize=64)  # Increased 8x from 8
def load(model: str) -> PunktSentenceTokenizer:
    """
    Load a Punkt model by name or path.

    Args:
        model: Either:
            - "default" for the built-in model
            - A file path to a model (.bin, .json, .json.xz)
            - A model name to search in standard locations

    Returns:
        A PunktSentenceTokenizer initialized with the model

    Model Search Paths:
        When loading by name, models are searched in order:
        1. Package models directory (built-in models)
        2. Platform-specific user data directory:
           - Linux: $XDG_DATA_HOME/nupunkt/models or ~/.local/share/nupunkt/models
           - macOS: ~/Library/Application Support/nupunkt/models
           - Windows: %LOCALAPPDATA%\\nupunkt\\models
        3. Legacy ~/.nupunkt/models (for backward compatibility)
        4. Current working directory/models

    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the model format is unsupported
    """
    # Handle default model
    if model == "default":
        return load_default_model()

    # Try as direct file path first
    path = Path(model)
    if path.is_file():
        return PunktSentenceTokenizer.load(path)

    # Search for model by name using platform-specific paths
    from nupunkt.utils.paths import get_model_search_paths

    search_paths = get_model_search_paths()
    for search_dir in search_paths:
        for ext in (".json.gz", ".json.xz", ".bin", ".json"):
            model_path = search_dir / f"{model}{ext}"
            if model_path.exists():
                return PunktSentenceTokenizer.load(model_path)

    # Build helpful error message
    searched_locations = "\n  - ".join(str(p) for p in search_paths)
    raise FileNotFoundError(
        f"Model '{model}' not found. Searched in:\n  - {searched_locations}\n"
        f"To install a model, place it in one of these directories."
    )


# Backward compatibility - keep the old internal functions
@lru_cache(maxsize=8)  # Increased 8x from 1
def _get_default_model():
    """Get the default model, loading it only once."""
    return load("default")


@lru_cache(maxsize=8)  # Increased 8x from 1
def _get_paragraph_tokenizer():
    """Get the paragraph tokenizer with the default model, loading it only once."""
    return PunktParagraphTokenizer(_get_default_model())


@lru_cache(maxsize=64)  # Increased 8x from 8
def _get_adaptive_tokenizer(
    model: str, confidence_threshold: float, enable_dynamic_abbrev: bool
) -> Any:  # Returns AdaptiveTokenizer but avoid import at module level
    """Get an adaptive tokenizer with caching based on parameters."""
    from nupunkt.hybrid import AdaptiveTokenizer

    # Load base model - always use load() to get cached version
    base_model = load(model)

    return AdaptiveTokenizer(
        model_or_text=base_model._params,
        confidence_threshold=confidence_threshold,
        enable_dynamic_abbrev=enable_dynamic_abbrev,
        debug=False,  # Don't cache debug mode
    )


# Function for quick and easy sentence tokenization
def sent_tokenize(
    text: str,
    model: str = "default",
    adaptive: bool = False,
    confidence_threshold: float = 0.7,
    dynamic_abbrev: bool = True,
    return_confidence: bool = False,
    debug: bool = False,
) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Tokenize text into sentences.

    Args:
        text: The text to tokenize
        model: Model to use - "default", a file path, or a model name
        adaptive: Enable adaptive tokenization with dynamic pattern recognition
        confidence_threshold: Decision threshold for adaptive mode (0.0-1.0)
                            Lower = more sentence breaks, Higher = fewer breaks
        dynamic_abbrev: Discover abbreviation patterns at runtime (M.I.T., Ph.D.)
        return_confidence: Return (sentence, confidence) tuples instead of just sentences
        debug: Enable debug output showing decision reasoning

    Returns:
        List of sentences, or list of (sentence, confidence) tuples if return_confidence=True

    Examples:
        >>> sent_tokenize("Hello world. How are you?")
        ['Hello world.', 'How are you?']

        >>> # Adaptive mode - dynamically recognizes patterns
        >>> sent_tokenize("She studied at M.I.T. in Cambridge.", adaptive=True)
        ['She studied at M.I.T. in Cambridge.']

        >>> # Get confidence scores
        >>> sent_tokenize("Dr. Smith arrived.", adaptive=True, return_confidence=True)
        [('Dr. Smith arrived.', 0.92)]

        >>> # Tune for high precision (fewer breaks)
        >>> sent_tokenize(text, adaptive=True, confidence_threshold=0.85)

        >>> # Tune for high recall (more breaks)
        >>> sent_tokenize(text, adaptive=True, confidence_threshold=0.5)
    """
    if adaptive:
        # Get cached tokenizer if not in debug mode
        if debug:
            # Debug mode creates a new instance each time
            from nupunkt.hybrid import AdaptiveTokenizer

            base_model = load(model) if model != "default" else None
            tokenizer = AdaptiveTokenizer(
                model_or_text=base_model._params if base_model else None,
                confidence_threshold=confidence_threshold,
                enable_dynamic_abbrev=dynamic_abbrev,
                debug=True,
            )
        else:
            # Use cached tokenizer for better performance
            tokenizer = _get_adaptive_tokenizer(
                model=model,
                confidence_threshold=confidence_threshold,
                enable_dynamic_abbrev=dynamic_abbrev,
            )

        if return_confidence:
            return tokenizer.tokenize_with_confidence(text)
        else:
            return list(tokenizer.tokenize(text))
    else:
        # Standard mode
        if return_confidence:
            raise ValueError("return_confidence is only available in adaptive mode")
        tokenizer = load(model)
        return list(tokenizer.tokenize(text))


# Convenience function for adaptive tokenization
def sent_tokenize_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    return_confidence: bool = False,
    debug: bool = False,
    **kwargs,
) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Adaptive sentence tokenization with confidence scoring.

    This is a convenience wrapper for sent_tokenize with adaptive=True.

    Args:
        text: Text to tokenize
        threshold: Confidence threshold (0.0-1.0)
        model: Model to use - "default", a file path, or a model name
        return_confidence: Return (sentence, confidence) tuples
        debug: Enable debug output
        **kwargs: Additional arguments passed to sent_tokenize

    Returns:
        List of sentences, or list of (sentence, confidence) tuples if return_confidence=True

    Examples:
        >>> # Adaptively handle abbreviations
        >>> sent_tokenize_adaptive("She got her Ph.D. at M.I.T. yesterday.")
        ['She got her Ph.D. at M.I.T. yesterday.']

        >>> # Tune for your use case
        >>> sent_tokenize_adaptive(legal_text, threshold=0.85)  # High precision
        >>> sent_tokenize_adaptive(tweets, threshold=0.5)       # High recall
    """
    return sent_tokenize(
        text,
        model=model,
        adaptive=True,
        confidence_threshold=threshold,
        return_confidence=return_confidence,
        debug=debug,
        **kwargs,
    )


# Function for paragraph tokenization
def para_tokenize(text: str) -> List[str]:
    """
    Tokenize text into paragraphs using the default pre-trained model.

    Paragraph breaks are identified at sentence boundaries that are
    immediately followed by two or more newlines.

    Args:
        text: The text to tokenize

    Returns:
        A list of paragraphs
    """
    paragraph_tokenizer = _get_paragraph_tokenizer()
    return list(paragraph_tokenizer.tokenize(text))


# Function for getting sentence spans
def sent_spans(text: str) -> List[Tuple[int, int]]:
    """
    Get sentence spans (start, end character positions) using the default pre-trained model.

    This is a convenience function for getting sentence spans without having
    to explicitly load a model. The spans are guaranteed to be contiguous,
    covering the entire input text without gaps.

    Args:
        text: The text to segment

    Returns:
        A list of sentence spans as (start_index, end_index) tuples
    """
    tokenizer = _get_default_model()
    return [span for _, span in tokenizer.tokenize_with_spans(text)]


# Function for getting sentence spans with text
def sent_spans_with_text(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Get sentences with their spans using the default pre-trained model.

    This is a convenience function for getting sentences with their character spans
    without having to explicitly load a model. The spans are guaranteed to be
    contiguous, covering the entire input text without gaps.

    Args:
        text: The text to segment

    Returns:
        A list of tuples containing (sentence, (start_index, end_index))
    """
    tokenizer = _get_default_model()
    return tokenizer.tokenize_with_spans(text)


# Function for getting paragraph spans
def para_spans(text: str) -> List[Tuple[int, int]]:
    """
    Get paragraph spans (start, end character positions) using the default pre-trained model.

    This is a convenience function for getting paragraph spans without having
    to explicitly load a model. The spans are guaranteed to be contiguous,
    covering the entire input text without gaps.

    Args:
        text: The text to segment

    Returns:
        A list of paragraph spans as (start_index, end_index) tuples
    """
    paragraph_tokenizer = _get_paragraph_tokenizer()
    return list(paragraph_tokenizer.span_tokenize(text))


# Function for getting paragraph spans with text
def para_spans_with_text(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Get paragraphs with their spans using the default pre-trained model.

    This is a convenience function for getting paragraphs with their character spans
    without having to explicitly load a model. The spans are guaranteed to be
    contiguous, covering the entire input text without gaps.

    Args:
        text: The text to segment

    Returns:
        A list of tuples containing (paragraph, (start_index, end_index))
    """
    paragraph_tokenizer = _get_paragraph_tokenizer()
    return list(paragraph_tokenizer.tokenize_with_spans(text))


# Function for getting sentence spans with adaptive tokenization
def sent_spans_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    dynamic_abbrev: bool = True,
    **kwargs,
) -> List[Tuple[int, int]]:
    """
    Get sentence spans using adaptive tokenization with confidence scoring.

    This function provides character-level sentence boundary detection using
    the adaptive algorithm that dynamically recognizes abbreviation patterns.

    Args:
        text: The text to segment
        threshold: Confidence threshold (0.0-1.0)
        model: Model to use - "default", a file path, or a model name
        dynamic_abbrev: Discover abbreviation patterns at runtime
        **kwargs: Additional arguments passed to the tokenizer

    Returns:
        A list of sentence spans as (start_index, end_index) tuples

    Examples:
        >>> # Get spans for text with unknown abbreviations
        >>> spans = sent_spans_adaptive("She studied at M.I.T. in Cambridge.")
        >>> [(0, 35)]  # Single sentence preserved

        >>> # Tune for high precision
        >>> spans = sent_spans_adaptive(legal_text, threshold=0.85)
    """
    # Get cached tokenizer
    tokenizer = _get_adaptive_tokenizer(
        model=model,
        confidence_threshold=threshold,
        enable_dynamic_abbrev=dynamic_abbrev,
    )
    return [span for _, span in tokenizer.tokenize_with_spans(text)]


# Function for getting sentence spans with text using adaptive tokenization
@overload
def sent_spans_with_text_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    dynamic_abbrev: bool = True,
    return_confidence: Literal[False] = False,
    **kwargs,
) -> List[Tuple[str, Tuple[int, int]]]:
    ...


@overload
def sent_spans_with_text_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    dynamic_abbrev: bool = True,
    return_confidence: Literal[True] = True,
    **kwargs,
) -> List[Tuple[str, Tuple[int, int], float]]:
    ...


def sent_spans_with_text_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    dynamic_abbrev: bool = True,
    return_confidence: bool = False,
    **kwargs,
) -> Union[List[Tuple[str, Tuple[int, int]]], List[Tuple[str, Tuple[int, int], float]]]:
    """
    Get sentences with their spans using adaptive tokenization.

    This function provides both the sentence text and character positions using
    the adaptive algorithm. Optionally includes confidence scores.

    Args:
        text: The text to segment
        threshold: Confidence threshold (0.0-1.0)
        model: Model to use - "default", a file path, or a model name
        dynamic_abbrev: Discover abbreviation patterns at runtime
        return_confidence: Include confidence scores in the output
        **kwargs: Additional arguments passed to the tokenizer

    Returns:
        If return_confidence is False:
            List of (sentence, (start_index, end_index)) tuples
        If return_confidence is True:
            List of (sentence, (start_index, end_index), confidence) tuples

    Examples:
        >>> # Get sentences with spans
        >>> results = sent_spans_with_text_adaptive("Dr. Smith studied at M.I.T. today.")
        >>> [('Dr. Smith studied at M.I.T. today.', (0, 34))]

        >>> # With confidence scores
        >>> results = sent_spans_with_text_adaptive(text, return_confidence=True)
        >>> [('First sentence.', (0, 15), 0.92), ('Second one.', (15, 26), 0.88)]
    """
    # Get cached tokenizer
    tokenizer = _get_adaptive_tokenizer(
        model=model,
        confidence_threshold=threshold,
        enable_dynamic_abbrev=dynamic_abbrev,
    )

    if return_confidence:
        # Get sentences with confidence scores
        sentences_with_conf = tokenizer.tokenize_with_confidence(text)

        # Get spans
        spans_with_text = tokenizer.tokenize_with_spans(text)

        # Combine confidence scores with spans
        results = []
        for (sent_conf, conf), (sent_span, span) in zip(sentences_with_conf, spans_with_text):
            # Verify sentences match (they should)
            if sent_conf.strip() != sent_span.strip():
                # Handle potential whitespace differences
                conf_idx = next(
                    (
                        i
                        for i, (s, _) in enumerate(sentences_with_conf)
                        if sent_span.strip() == s.strip()
                    ),
                    None,
                )
                if conf_idx is not None:
                    conf = sentences_with_conf[conf_idx][1]
            results.append((sent_span, span, conf))
        return results
    else:
        return tokenizer.tokenize_with_spans(text)


__all__ = [
    "__version__",
    "PunktParameters",
    "PunktLanguageVars",
    "PunktToken",
    "PunktTrainer",
    "PunktSentenceTokenizer",
    "PunktParagraphTokenizer",
    "load",
    "load_default_model",
    "sent_tokenize",
    "sent_tokenize_adaptive",
    "sent_spans",
    "sent_spans_with_text",
    "sent_spans_adaptive",
    "sent_spans_with_text_adaptive",
    "para_tokenize",
    "para_spans",
    "para_spans_with_text",
]
