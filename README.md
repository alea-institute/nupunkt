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
Employee also specifically and forever releases the Acme Inc. (Company) and the Company Parties (except where and 
to the extent that such a release is expressly prohibited or made void by law) from any claims based on unlawful 
employment discrimination or harassment, including, but not limited to, the Federal Age Discrimination in 
Employment Act (29 U.S.C. § 621 et. seq.). This release does not include Employee’s right to indemnification, 
and related insurance coverage, under Sec. 7.1.4 or Ex. 1-1 of the Employment Agreement, his right to equity awards,
or continued exercise, pursuant to the terms of any specific equity award (or similar) agreement between 
Employee and the Company nor to Employee’s right to benefits under any Company plan or program in which
Employee participated and is due a benefit in accordance with the terms of the plan or program as of the Effective
Date and ending at 11:59 p.m. Eastern Time on Sep. 15, 2013.
"""

# Tokenize into sentences
sentences = sent_tokenize(text)

# Print the results
for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}\n")
```

Output:
```
Sentence 1:
Employee also specifically and forever releases the Acme Inc. (Company) and the Company Parties (except where and
to the extent that such a release is expressly prohibited or made void by law) from any claims based on unlawful
employment discrimination or harassment, including, but not limited to, the Federal Age Discrimination in
Employment Act (29 U.S.C. § 621 et. seq.).

Sentence 2:  This release does not include Employee’s right to indemnification,
and related insurance coverage, under Sec. 7.1.4 or Ex. 1-1 of the Employment Agreement, his right to equity awards,
or continued exercise, pursuant to the terms of any specific equity award (or similar) agreement between
Employee and the Company nor to Employee’s right to benefits under any Company plan or program in which
Employee participated and is due a benefit in accordance with the terms of the plan or program as of the Effective
Date and ending at 11:59 p.m. Eastern Time on Sep. 15, 2013.
```

## Documentation

For more detailed documentation, see the [docs](./docs) directory:

- [Overview](./docs/overview.md)
- [Getting Started](./docs/getting_started.md)
- [API Reference](./docs/api_reference.md)
- [Architecture](./docs/architecture.md)
- [Training Models](./docs/training_models.md)
- [Advanced Usage](./docs/advanced_usage.md)

## Command-line Tools

nupunkt comes with several utility scripts for working with models:

- **check_abbreviation.py**: Check if a token is in the model's abbreviation list
  ```bash
  python -m scripts.utils.check_abbreviation "U.S." 
  python -m scripts.utils.check_abbreviation --list   # List all abbreviations
  python -m scripts.utils.check_abbreviation --count  # Count abbreviations
  ```

- **test_tokenizer.py**: Test the tokenizer on sample text
- **model_info.py**: Display information about a model file

See the [scripts/utils/README.md](./scripts/utils/README.md) for more details on available tools.

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

### Highly Optimized

The tokenizer has been extensively optimized for performance:
- **Token caching** for common tokens
- **Fast path processing** for texts without sentence boundaries (up to 1.4B chars/sec)
- **Pre-computed properties** to avoid repeated calculations
- **Efficient character processing** and string handling in hot spots

### Example Legal Domain Benchmark
```
Performance Results:
  Documents processed:      1
  Total characters:         16,567,769
  Total sentences found:    16,095
  Processing time:          0.49 seconds
  Processing speed:         33,927,693 characters/second
  Average sentence length:  1029.4 characters
```

### Specialized Use Cases
- Normal text processing: ~31M characters/second
- Text without sentence boundaries: ~1.4B characters/second 
- Short text fragments: Extremely fast with early exit paths

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

nupunkt is based on the Punkt algorithm originally developed by Tibor Kiss and Jan Strunk.