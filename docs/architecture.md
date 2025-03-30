# nupunkt Architecture

This document describes the architecture and modules of the nupunkt package.

## Package Structure

The nupunkt package is organized into the following modules:

```
nupunkt/
├── __init__.py          # Package initialization and public API
├── nupunkt.py           # Main implementation file
├── py.typed             # Type checking marker
├── core/                # Core components
│   ├── __init__.py
│   ├── base.py          # Base classes
│   ├── constants.py     # Constant definitions
│   ├── language_vars.py # Language variables
│   ├── parameters.py    # Algorithm parameters
│   └── tokens.py        # Token representation
├── models/              # Model handling
│   ├── __init__.py
│   └── default_model.json # Pre-trained model
├── tokenizers/          # Tokenization components
│   ├── __init__.py
│   └── sentence_tokenizer.py # Sentence tokenizer
├── trainers/            # Training components
│   ├── __init__.py
│   └── base_trainer.py  # Trainer implementation
└── utils/               # Utility functions
    ├── __init__.py
    ├── iteration.py     # Iteration helpers
    └── statistics.py    # Statistical functions
```

## Module Descriptions

### Core Module

The `core` module provides the fundamental building blocks for the sentence tokenization process:

- **base.py**: Contains the `PunktBase` class, which provides common functionality used by both trainers and tokenizers.
- **constants.py**: Defines orthographic context constants used to track capitalization patterns.
- **language_vars.py**: Provides the `PunktLanguageVars` class, which encapsulates language-specific behaviors.
- **parameters.py**: Contains the `PunktParameters` class that stores the learned parameters (abbreviations, collocations, etc.).
- **tokens.py**: Defines the `PunktToken` class, which represents tokens with various attributes.

### Models Module

The `models` module handles model loading and provides a pre-trained default model:

- **__init__.py**: Contains functions to load the default model.
- **default_model.json**: A pre-trained model ready for general use.

### Tokenizers Module

The `tokenizers` module contains the sentence tokenizer implementation:

- **sentence_tokenizer.py**: Implements the `PunktSentenceTokenizer` class, which performs the actual sentence boundary detection using trained parameters.

### Trainers Module

The `trainers` module handles training new models from text:

- **base_trainer.py**: Contains the `PunktTrainer` class, which learns parameters from training text.

### Utils Module

The `utils` module provides utility functions used throughout the package:

- **iteration.py**: Contains utilities for iteration, like `pair_iter` for iterating through pairs of items.
- **statistics.py**: Provides statistical functions for calculating log-likelihood and other measures.

## Main Classes

### PunktSentenceTokenizer

The primary class for tokenizing text into sentences. It uses trained parameters to identify sentence boundaries.

```python
tokenizer = PunktSentenceTokenizer()
sentences = tokenizer.tokenize(text)
```

### PunktTrainer

Used to train new models on domain-specific text:

```python
trainer = PunktTrainer(train_text, verbose=True)
params = trainer.get_params()
tokenizer = PunktSentenceTokenizer(params)
```

### PunktParameters

Stores the learned parameters that guide the tokenization process:

- **abbrev_types**: Set of known abbreviations
- **collocations**: Set of word pairs that often occur across sentence boundaries
- **sent_starters**: Set of words that often start sentences
- **ortho_context**: Dictionary tracking capitalization patterns

### PunktLanguageVars

Contains language-specific variables and settings that can be customized:

- **sent_end_chars**: Characters that can end sentences
- **internal_punctuation**: Characters considered internal punctuation
- **word_tokenize_pattern**: Pattern for tokenizing words

## Data Flow

1. Text is tokenized into words using `PunktLanguageVars.word_tokenize()`.
2. Tokens are annotated in the first pass to identify sentence breaks, abbreviations, and ellipses.
3. Tokens are annotated in the second pass using collocational and orthographic heuristics.
4. Sentence boundaries are determined based on the annotated tokens.
5. Boundaries are optionally realigned to handle trailing punctuation.
6. The resulting sentences or spans are returned.

## Algorithm Workflow

### Training

1. Count token frequencies in the training text.
2. Identify potential abbreviations based on statistical measures.
3. Annotate tokens with sentence breaks, abbreviations, and ellipses.
4. Gather orthographic context data (capitalization patterns).
5. Identify collocations and sentence starters.
6. Finalize the parameters for use in tokenization.

### Tokenization

1. Break the text into words.
2. Annotate tokens with sentence breaks, abbreviations, and ellipses.
3. Apply collocational and orthographic heuristics to refine annotations.
4. Use sentence breaks to slice the text into sentences.
5. Realign boundaries if requested.
6. Return sentences or character spans.