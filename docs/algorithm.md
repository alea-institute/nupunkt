# The Punkt Algorithm and Adaptive Enhancements

This document explains how nupunkt's sentence boundary detection works, covering both the traditional Punkt algorithm and the new adaptive enhancements in v0.6.0.

## Overview

nupunkt implements an optimized version of the Punkt algorithm (Kiss & Strunk, 2006) for unsupervised sentence boundary detection. The algorithm uses statistical methods to identify:

1. **Abbreviations** - tokens that end with periods but don't end sentences
2. **Collocations** - word pairs that frequently occur together across sentence boundaries
3. **Sentence starters** - words that commonly begin sentences

## The Core Punkt Algorithm

### How It Works

The Punkt algorithm is based on the observation that a period following a token has three possible interpretations:

1. **Abbreviation marker** (e.g., "Dr.", "U.S.")
2. **Sentence boundary** (e.g., "The end.")
3. **Both** (e.g., "He works at M.I.T." - abbreviation at sentence end)

The algorithm learns to distinguish these cases statistically from training text.

### Key Components

#### 1. Abbreviation Detection

The algorithm uses the Dunning log-likelihood ratio to identify abbreviations:

```python
# Example: Testing if "Dr" is an abbreviation
count_dr = 50          # Times "Dr" appears
count_dr_period = 48   # Times "Dr." appears
count_total = 10000    # Total tokens

# High log-likelihood score indicates likely abbreviation
score = dunning_log_likelihood(count_dr, count_dr_period, ...)
```

Key features:
- Tokens appearing frequently with periods are likely abbreviations
- Short tokens (≤9 characters) are candidates
- The algorithm uses a low threshold (0.1) to capture most abbreviations
- A boosting factor (1.5x) helps detect abbreviations in large datasets

#### 2. Orthographic Context

The algorithm tracks capitalization patterns around potential sentence boundaries:

```python
# Orthographic contexts
ORTHO_BEG_UC = 1 << 1  # Uppercase at sentence beginning
ORTHO_MID_UC = 1 << 2  # Uppercase mid-sentence
ORTHO_UNK_UC = 1 << 3  # Unknown position uppercase
ORTHO_BEG_LC = 1 << 4  # Lowercase at sentence beginning
# ... etc
```

This helps distinguish:
- "Dr. Smith" (uppercase after abbreviation - normal)
- "end. She" (uppercase after period - sentence boundary)

#### 3. Collocations

Word pairs that span sentence boundaries without periods:

```python
# Example: "said Mr." often spans boundaries
# "Thank you," said Mr. Smith.
#              ^^^^ ^^^ 
#           collocation
```

The algorithm identifies these using collocation log-likelihood scores.

### Training Process

```python
from nupunkt import PunktTrainer

# 1. Tokenize text and count frequencies
trainer = PunktTrainer(training_text)

# 2. Identify abbreviations using log-likelihood
# 3. Find sentence starters and collocations
# 4. Build orthographic context model

params = trainer.get_params()
```

## Adaptive Tokenization (New in v0.6.0)

The adaptive tokenizer enhances the base algorithm by dynamically identifying abbreviation patterns and using confidence scoring.

### Key Improvements

#### 1. Pattern-Based Abbreviation Detection

Recognizes common patterns even if not in training data:

```python
# Patterns detected dynamically:
- Academic degrees: B.A., M.S., Ph.D.
- Organizations: U.S., N.A.S.A., I.B.M.
- Legal citations: Fed., Civ., Cal.
- Time: a.m., p.m.
- Titles: Mr., Dr., Prof.
```

Example difference:
```python
text = "She works at N.A.S.A. headquarters."

# Base algorithm (N.A.S.A. not in training)
base = ['She works at N.A.S.A.', 'headquarters.']

# Adaptive (recognizes pattern)
adaptive = ['She works at N.A.S.A. headquarters.']
```

#### 2. Confidence Scoring

Each boundary decision gets a confidence score:

```python
text = "Dr. Smith arrived at 3 p.m. She was late."

# With confidence scores
results = sent_tokenize_adaptive(text, return_confidence=True)
# [('Dr. Smith arrived at 3 p.m. She was late.', 0.33),
#  ('The meeting started.', 1.00)]
```

Lower confidence indicates uncertainty about the boundary.

#### 3. Context-Aware Decisions

The adaptive tokenizer considers:
- Whether the following word is lowercase (likely continuation)
- Whether the following word is a common continuation word ("of", "in", "at")
- Token length and capitalization patterns

### How Adaptive Mode Works

```python
class AdaptiveTokenizer:
    def _is_likely_abbreviation(self, token, next_token):
        # 1. Check against regex patterns
        # 2. Consider token characteristics
        # 3. Analyze following context
        # 4. Return confidence score
```

The algorithm:
1. Runs the base Punkt algorithm
2. Re-evaluates uncertain boundaries
3. Applies pattern matching and context rules
4. Makes final decision based on confidence threshold

### When to Use Adaptive Mode

**Use adaptive mode when:**
- Processing text with domain-specific abbreviations
- Handling text with unconventional punctuation patterns  
- You need confidence scores for downstream processing
- The base model makes consistent errors on certain patterns

**Use standard mode when:**
- Processing speed is critical
- Your text matches the training domain well
- You have a custom-trained model for your domain

## Performance Characteristics

### Base Algorithm
- **Speed**: ~33M characters/second on legal text
- **Memory**: O(n) where n is vocabulary size
- **Accuracy**: 91.1% precision on legal benchmarks

### Adaptive Algorithm
- **Speed**: ~10-15% slower than base
- **Memory**: Same as base + pattern matching overhead
- **Accuracy**: Higher on texts with novel abbreviation patterns

## Implementation Details

### Optimization Techniques

1. **Token Caching**: Common tokens are cached for speed
2. **Fast Paths**: Text without periods bypasses full processing
3. **Compiled Regexes**: All patterns pre-compiled at startup
4. **Lazy Loading**: Models loaded only when needed

### Memory-Efficient Training

For large corpora:
```python
trainer = train_model(
    huge_corpus,
    memory_efficient=True,
    batch_size=1000000,  # Process 1M chars at a time
    prune_freq=10000,    # Prune rare tokens periodically
)
```

## References

- Kiss, T. and Strunk, J. (2006). Unsupervised Multilingual Sentence Boundary Detection. Computational Linguistics 32: 485-525.
- Bommarito, M.J., Katz, D.M., and Bommarito, J. (2025). Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary. arXiv:2504.04131.