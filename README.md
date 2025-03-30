# nupunkt

**nupunkt** is a next-generation implementation of the Punkt algorithm for sentence boundary detection with zero runtime dependencies.

[![PyPI version](https://badge.fury.io/py/nupunkt.svg)](https://badge.fury.io/py/nupunkt)
[![Python Version](https://img.shields.io/pypi/pyversions/nupunkt.svg)](https://pypi.org/project/nupunkt/)
[![License](https://img.shields.io/github/license/alea-institute/nupunkt.svg)](https://github.com/alea-institute/nupunkt/blob/main/LICENSE)

## Overview

nupunkt accurately detects sentence boundaries in text, even in challenging cases where periods are used for abbreviations, ellipses, and other non-sentence-ending contexts. It's built on the statistical principles of the Punkt algorithm, with modern enhancements for improved handling of edge cases.

Key features:
- **Minimal dependencies**: Only requires Python 3.11+ and tqdm for progress bars
- **Pre-trained model**: Ready to use out of the box
- **Fast and accurate**: Optimized implementation of the Punkt algorithm
- **Trainable**: Can be trained on domain-specific text
- **Full support for ellipsis**: Handles various ellipsis patterns
- **Type annotations**: Complete type hints for better IDE integration

## Installation

```bash
pip install nupunkt
```

## Quick Start

```python
from nupunkt import sent_tokenize

text = """
Hello world! This is a test of nupunkt. It handles abbreviations like Dr. Smith 
and Mrs. Jones. It also handles ellipsis... as well as other punctuation?!
U.S.C. § 42 defines the law.
"""

# Tokenize into sentences
sentences = sent_tokenize(text)

# Print the results
for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")
```

Output:
```
Sentence 1: Hello world!
Sentence 2: This is a test of nupunkt.
Sentence 3: It handles abbreviations like Dr. Smith and Mrs. Jones.
Sentence 4: It also handles ellipsis... as well as other punctuation?!
Sentence 5: U.S.C. § 42 defines the law.
```

## Documentation

For more detailed documentation, see the [docs](./docs) directory:

- [Overview](./docs/overview.md)
- [Getting Started](./docs/getting_started.md)
- [API Reference](./docs/api_reference.md)
- [Architecture](./docs/architecture.md)
- [Training Models](./docs/training_models.md)
- [Advanced Usage](./docs/advanced_usage.md)

## Advanced Example

```python
from nupunkt import PunktTrainer, PunktSentenceTokenizer

# Train a new model on domain-specific text
with open("legal_corpus.txt", "r", encoding="utf-8") as f:
    legal_text = f.read()

trainer = PunktTrainer(legal_text, verbose=True)
params = trainer.get_params()

# Save the trained model
trainer.save("legal_model.json")

# Create a tokenizer with the trained parameters
tokenizer = PunktSentenceTokenizer(params)

# Tokenize legal text
legal_sample = "The court ruled in favor of the plaintiff. 28 U.S.C. § 1332 provides jurisdiction."
sentences = tokenizer.tokenize(legal_sample)

for s in sentences:
    print(s)
```

## Performance

nupunkt is designed to be both accurate and efficient. It can process large volumes of text quickly, making it suitable for production NLP pipelines.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

nupunkt is based on the Punkt algorithm originally developed by Tibor Kiss and Jan Strunk.