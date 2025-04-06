"""
nupunkt is a Python library for sentence and paragraph boundary detection based on the Punkt algorithm.

It learns to identify sentence boundaries in text, even when periods are used for
abbreviations, ellipses, and other non-sentence-ending contexts. It also supports
paragraph detection based on sentence boundaries and newlines.
"""

__version__ = "0.5.1"

# Core classes
from functools import lru_cache

# Import for type annotations
from typing import List, Tuple

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


# Singleton pattern to load model only once
@lru_cache(maxsize=1)
def _get_default_model():
    """Get the default model, loading it only once."""
    return load_default_model()


@lru_cache(maxsize=1)
def _get_paragraph_tokenizer():
    """Get the paragraph tokenizer with the default model, loading it only once."""
    return PunktParagraphTokenizer(_get_default_model())


# Function for quick and easy sentence tokenization
def sent_tokenize(text: str) -> List[str]:
    """
    Tokenize text into sentences using the default pre-trained model.

    This is a convenience function for quick sentence tokenization
    without having to explicitly load a model.

    Args:
        text: The text to tokenize

    Returns:
        A list of sentences
    """
    tokenizer = _get_default_model()
    return list(tokenizer.tokenize(text))


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
    from sentences import SentenceSegmenter

    segmenter = _get_default_model()
    return list(SentenceSegmenter.get_sentence_spans(segmenter, text))


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
    from sentences import SentenceSegmenter

    segmenter = _get_default_model()
    return list(SentenceSegmenter.get_sentence_spans_with_text(segmenter, text))


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


__all__ = [
    "PunktParameters",
    "PunktLanguageVars",
    "PunktToken",
    "PunktTrainer",
    "PunktSentenceTokenizer",
    "PunktParagraphTokenizer",
    "load_default_model",
    "sent_tokenize",
    "sent_spans",
    "sent_spans_with_text",
    "para_tokenize",
    "para_spans",
    "para_spans_with_text",
]