# Getting Started with nupunkt

This guide will help you install nupunkt and get started with basic sentence tokenization.

## Installation

### Using pip

```bash
pip install nupunkt
```

### From source

```bash
git clone https://github.com/alea-institute/nupunkt.git
cd nupunkt
pip install -e .
```

For development, install with additional dependencies:

```bash
pip install -e ".[dev]"
```

## Basic Usage

### Quick Sentence Tokenization

For quick tokenization using the default pre-trained model:

```python
from nupunkt import sent_tokenize

text = "Hello world. This is a test. Mr. Smith went to Washington D.C. yesterday."
sentences = sent_tokenize(text)

for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")
```

Output:
```
Sentence 1: Hello world.
Sentence 2: This is a test.
Sentence 3: Mr. Smith went to Washington D.C. yesterday.
```

### Creating a Tokenizer Instance

If you need more control or plan to tokenize multiple texts:

```python
from nupunkt import PunktSentenceTokenizer
from nupunkt.models import load_default_model

# Load the default pre-trained model
tokenizer = load_default_model()

# Or create a new tokenizer instance
# tokenizer = PunktSentenceTokenizer()

text = "Hello world! Is this a sentence? Yes, it is. Dr. Smith teaches at the U.S.C."
sentences = tokenizer.tokenize(text)

for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")
```

Output:
```
Sentence 1: Hello world!
Sentence 2: Is this a sentence?
Sentence 3: Yes, it is.
Sentence 4: Dr. Smith teaches at the U.S.C.
```

### Getting Sentence Spans

If you need character offsets for the sentences:

```python
spans = list(tokenizer.span_tokenize(text))
for start, end in spans:
    print(f"Span ({start}, {end}): {text[start:end]}")
```

## Common Options

### Handling Boundary Realignment

By default, nupunkt realigns sentence boundaries to handle trailing punctuation. You can disable this behavior:

```python
sentences = tokenizer.tokenize(text, realign_boundaries=False)
```

### Custom Language Variables

You can customize language-specific behavior:

```python
from nupunkt import PunktLanguageVars, PunktSentenceTokenizer

class GermanLanguageVars(PunktLanguageVars):
    # Add German-specific customizations
    sent_end_chars = (".", "?", "!", ":")

german_vars = GermanLanguageVars()
tokenizer = PunktSentenceTokenizer(lang_vars=german_vars)
```

## Next Steps

- See [Training Models](training_models.md) for training your own model
- Explore [Advanced Usage](advanced_usage.md) for more customization options
- Check the [API Reference](api_reference.md) for detailed information on all classes and methods