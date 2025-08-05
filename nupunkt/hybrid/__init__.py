"""
Hybrid sentence boundary detection experiments.

This module contains research prototypes for enhanced sentence tokenization
that build upon the core Punkt algorithm.

The adaptive tokenizer (AdaptiveTokenizer) is recommended for use
as it properly integrates with the base Punkt algorithm while adding:
- Dynamic abbreviation detection for patterns like M.I.T., Ph.D., etc.
- Context-aware boundary decisions
- Confidence-based adaptive refinement of edge cases

Example:
    from nupunkt.hybrid import AdaptiveTokenizer

    tokenizer = AdaptiveTokenizer()
    sentences = tokenizer.tokenize("Dr. Smith studied at M.I.T. in Cambridge.")
    # Returns: ["Dr. Smith studied at M.I.T. in Cambridge."]
"""

# Original experimental tokenizer (not recommended)
# Evaluation utilities
# Adaptive tokenizer (recommended)
from nupunkt.hybrid.adaptive_tokenizer import (
    AdaptiveTokenizer,
    BoundaryDecision,
    create_adaptive_tokenizer,
)
from nupunkt.hybrid.challenge_set import (
    CHALLENGE_SET,
    evaluate_tokenizer,
    print_detailed_errors,
)
from nupunkt.hybrid.confidence_tokenizer import (
    DOMAIN_PRESETS,
    ConfidenceScore,
    ConfidenceSentenceTokenizer,
    create_domain_tokenizer,
)

__all__ = [
    # Adaptive tokenizer (recommended)
    "AdaptiveTokenizer",
    "BoundaryDecision",
    "create_adaptive_tokenizer",
    # Original tokenizer (kept for compatibility)
    "ConfidenceSentenceTokenizer",
    "ConfidenceScore",
    "create_domain_tokenizer",
    "DOMAIN_PRESETS",
    # Evaluation utilities
    "CHALLENGE_SET",
    "evaluate_tokenizer",
    "print_detailed_errors",
]
