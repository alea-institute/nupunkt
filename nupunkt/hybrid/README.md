# Hybrid Sentence Tokenization

This module contains experimental enhancements to the core Punkt sentence tokenization algorithm.

## Overview

The hybrid tokenizers aim to address cases where the standard Punkt algorithm struggles, particularly with:
- Abbreviations not in the training data (e.g., M.I.T., B.B.C.)
- Domain-specific patterns
- Edge cases requiring contextual analysis

## Tokenizers

### AdaptiveTokenizer (Recommended)

The `AdaptiveTokenizer` properly integrates with the base Punkt algorithm while adding intelligent enhancements:

```python
from nupunkt.hybrid import AdaptiveTokenizer

tokenizer = AdaptiveTokenizer()
sentences = tokenizer.tokenize("Dr. Smith studied at M.I.T. in Cambridge.")
# Returns: ["Dr. Smith studied at M.I.T. in Cambridge."]
```

**Key Features:**
- **Dynamic abbreviation detection**: Recognizes common patterns (academic degrees, organizations, months, etc.)
- **Context-aware decisions**: Considers what follows a potential boundary
- **Confidence-based refinement**: Only overrides base decisions when highly confident
- **Debug mode**: Provides detailed decision explanations

### ConfidenceSentenceTokenizer (Original)

The original experimental tokenizer that uses confidence scoring but has limitations:
- Overly aggressive in creating sentence boundaries
- Doesn't properly integrate with base Punkt decisions
- Kept for backward compatibility only

## Performance Comparison

On our challenge set of 27 difficult cases:

| Tokenizer | Accuracy | Notes |
|-----------|----------|-------|
| Standard Punkt | 77.8% | Baseline |
| Original Hybrid | 51.9% | Too aggressive |
| Adaptive | 77.8% | Matches baseline, handles additional cases |

The adaptive tokenizer matches the standard tokenizer's accuracy while correctly handling cases like "M.I.T." that aren't in the default abbreviation list.

## Usage Examples

### Basic Usage

```python
from nupunkt.hybrid import AdaptiveTokenizer

# Create tokenizer
tokenizer = AdaptiveTokenizer()

# Tokenize text
text = "Prof. Johnson works at I.B.M. headquarters. She has a Ph.D. from Stanford."
sentences = list(tokenizer.tokenize(text))
# Returns: ["Prof. Johnson works at I.B.M. headquarters.", "She has a Ph.D. from Stanford."]
```

### Debug Mode

```python
# Enable debug mode to see decision reasoning
tokenizer = AdaptiveTokenizer(debug=True)
sentences = tokenizer.tokenize("Visit M.I.T. today.")

# Outputs decision information:
# Token: 'M.I.T.' -> 'today'
#   Base: True, Final: False
#   Confidence: 0.00, Overridden: True
#   Reasons: Base algorithm: yes, Likely abbrev: Multiple internal periods, ...
```

### Domain-Specific Tokenizers

```python
from nupunkt.hybrid import create_adaptive_tokenizer

# Create domain-specific tokenizers
general_tokenizer = create_adaptive_tokenizer(domain='general')
legal_tokenizer = create_adaptive_tokenizer(domain='legal')
scientific_tokenizer = create_adaptive_tokenizer(domain='scientific')
```

### Custom Configuration

```python
# Create tokenizer with custom settings
tokenizer = AdaptiveTokenizer(
    confidence_threshold=0.8,  # More conservative
    enable_dynamic_abbrev=True,  # Enable pattern detection
    debug=False
)
```

## How It Works

The adaptive tokenizer enhances Punkt through:

1. **Pattern-Based Abbreviation Detection**
   - Academic degrees: B.A., M.S., Ph.D., J.D., etc.
   - Organizations: U.S., U.K., M.I.T., I.B.M., etc.
   - Common titles: Mr., Dr., Prof., etc.
   - Business terms: Inc., Corp., Ltd., LLC, etc.
   - Months: Jan., Feb., etc.

2. **Contextual Analysis**
   - Checks if the next word is a continuation word (and, or, but, etc.)
   - Identifies known sentence starters
   - Considers capitalization patterns

3. **Confidence Scoring**
   - Starts with base Punkt decision (high weight)
   - Adjusts based on punctuation type, abbreviation likelihood, and context
   - Only overrides when confidence exceeds threshold

## Evaluation

Use the included evaluation utilities to test tokenizer performance:

```python
from nupunkt.hybrid import evaluate_tokenizer, CHALLENGE_SET

tokenizer = AdaptiveTokenizer()
correct, total, errors = evaluate_tokenizer(tokenizer)
print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")
```

## Development Status

This module is experimental and under active development. The improved tokenizer is stable and recommended for use, but the API may change in future versions.

## Known Limitations

Even the improved tokenizer struggles with:
- Quotes containing sentence-ending punctuation
- Complex list patterns (1., 2., a), b))
- Some domain-specific abbreviations not matching patterns

For production use, consider training a domain-specific Punkt model in addition to using the hybrid enhancements.