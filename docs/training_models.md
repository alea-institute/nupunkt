# Training nupunkt Models

This guide explains how to train custom nupunkt models for domain-specific text.

## Why Train Custom Models?

While nupunkt comes with a pre-trained default model that works well for general text, you may want to train a custom model when:

- Working with specialized domains (legal, medical, scientific)
- Processing text with unusual abbreviation patterns
- Dealing with text in specific formats or styles
- Working with languages other than English

## Basic Training

### Training from Scratch

```python
from nupunkt import PunktTrainer, PunktSentenceTokenizer

# Load your training text
with open("training_corpus.txt", "r", encoding="utf-8") as f:
    training_text = f.read()

# Train a model (this will take some time with large corpora)
trainer = PunktTrainer(training_text, verbose=True)

# Get the trained parameters
params = trainer.get_params()

# Create a tokenizer with the trained parameters
tokenizer = PunktSentenceTokenizer(params)

# Save the model for later use
trainer.save("my_custom_model.json")
```

### Incremental Training

You can incrementally train an existing model on new data:

```python
from nupunkt import PunktTrainer, PunktSentenceTokenizer

# Load an existing model
trainer = PunktTrainer.load("existing_model.json")

# Load additional training text
with open("additional_corpus.txt", "r", encoding="utf-8") as f:
    more_training_text = f.read()

# Train on the new text
trainer.train(more_training_text, verbose=True, preserve_abbrevs=True)

# Get the updated parameters
params = trainer.get_params()

# Create a tokenizer with the trained parameters
tokenizer = PunktSentenceTokenizer(params)

# Save the updated model
trainer.save("updated_model.json")
```

## Training Options

### Verbose Output

Setting `verbose=True` provides detailed information during training:

```python
trainer = PunktTrainer(training_text, verbose=True)
```

This will show:
- Number of tokens found
- Most frequent tokens ending with periods
- Abbreviations identified
- Collocations identified
- Sentence starters identified

### Preserving Abbreviations

By default, abbreviations from previous training runs are preserved:

```python
trainer.train(more_text, preserve_abbrevs=True)  # Default
```

To start fresh with each training run:

```python
trainer.train(more_text, preserve_abbrevs=False)
```

### Customizing Common Abbreviations

nupunkt supports pre-loading abbreviation lists to improve tokenization accuracy. The default model training process uses two abbreviation sources:

1. **Legal Abbreviations**: Located at `data/legal_abbreviations.json`, this file contains a comprehensive list of legal abbreviations commonly found in legal documents.

2. **General Abbreviations**: Located at `data/general_abbreviations.json`, this file contains common English abbreviations including:
   - Months and days (Jan., Feb., Mon., Tue.)
   - Titles (Mr., Mrs., Dr., Prof.)
   - Academic degrees (Ph.D., M.A., B.Sc.)
   - Corporate designations (Inc., Ltd., Corp.)
   - Street suffixes (Ave., Blvd., Rd.)
   - Units and measurements (ft., kg., min.)
   - And more

You can extend either list to improve tokenization for your specific domain. When training a custom model, you can programmatically add abbreviations:

```python
from nupunkt import PunktTrainer
import json

# Create a trainer
trainer = PunktTrainer(verbose=True)

# Load abbreviations from a file
with open("my_abbreviations.json", "r", encoding="utf-8") as f:
    abbreviations = json.load(f)

# Add abbreviations to the model
for abbr in abbreviations:
    clean_abbr = abbr.lower()
    if abbr.endswith('.'):
        clean_abbr = clean_abbr[:-1]
    trainer._params.abbrev_types.add(clean_abbr)

# Then continue with normal training
trainer.train(training_text)
```

You can also specify common abbreviations that should always be recognized by subclassing:

```python
from nupunkt import PunktTrainer

# Create a subclass with custom common abbreviations
class LegalTrainer(PunktTrainer):
    COMMON_ABBREVS = ["art.", "sec.", "para.", "fig.", "p.", "pp."]

trainer = LegalTrainer(training_text)
```

## Training Parameters

nupunkt's training process is controlled by several hyperparameters that affect how abbreviations, collocations, and sentence starters are identified. These parameters balance precision and recall in different ways.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ABBREV` | 0.1 | Abbreviation threshold - log-likelihood score required to consider a word an abbreviation (lower values are more aggressive) |
| `ABBREV_BACKOFF` | 10 | Minimum frequency for rare abbreviations - how many occurrences needed to consider a rare token as a potential abbreviation |
| `COLLOCATION` | 5.0 | Collocation threshold - log-likelihood score required to consider two words a collocation |
| `SENT_STARTER` | 25.0 | Sentence starter threshold - log-likelihood score required to consider a word a sentence starter |
| `MIN_COLLOC_FREQ` | 5 | Minimum frequency for collocations - how many times a pair must appear to be considered |
| `MAX_ABBREV_LENGTH` | 9 | Maximum length for abbreviations - words longer than this are not considered abbreviations |
| `IGNORE_ABBREV_PENALTY` | 1.01 | Penalty factor for non-abbreviation words that end with a period (multiplies the collocation threshold) |
| `INCLUDE_ALL_COLLOCS` | False | Whether to include all collocations or just those following period-final tokens |
| `INCLUDE_ABBREV_COLLOCS` | False | Whether to include collocations that include abbreviations |
| `PERSIST_ABBREVS` | True | Whether to keep abbreviations from previous training when training with new data |

### Customizing Parameters

You can customize these parameters by subclassing `PunktTrainer`:

```python
from nupunkt import PunktTrainer

# Create a custom trainer for legal text - more aggressive with abbreviations
class LegalTrainer(PunktTrainer):
    # Lower threshold for abbreviations (more aggressive detection)
    ABBREV = 0.08
    
    # Higher collocation threshold (more conservative)
    COLLOCATION = 10.0
    
    # Longer abbreviations common in legal text
    MAX_ABBREV_LENGTH = 10
    
    # Allow more abbreviations in collocations
    IGNORE_ABBREV_PENALTY = 0.8
    
    # Keep abbreviations between training runs
    PERSIST_ABBREVS = True

trainer = LegalTrainer(training_text, verbose=True)
```

### Parameter Tuning Guidelines

The default parameters in nupunkt are already tuned to err on the side of caution when splitting sentences, favoring more abbreviation detection to avoid false sentence breaks. This makes it well-suited for legal and technical texts.

When tuning these parameters for your specific use case, consider these guidelines:

- **`ABBREV` (default: 0.1)**: The lower this value, the more words will be considered abbreviations.
  - Decrease further (e.g., 0.05) for extremely conservative sentence splitting
  - Increase (e.g., 0.2-0.3) if you're getting too few sentence breaks

- **`MAX_ABBREV_LENGTH` (default: 9)**: Longer values allow more words to be considered abbreviations.
  - Increase for domains with many long abbreviations (legal, scientific)
  - Decrease for general text where most abbreviations are short

- **`COLLOCATION` (default: 5.0)**: Controls how readily word pairs are kept together.
  - Decrease to be more aggressive about keeping word pairs together
  - Increase if the model is missing sentence boundaries

- **`SENT_STARTER` (default: 25.0)**: Controls which words can start sentences.
  - Increase to be more conservative about sentence breaks
  - Decrease if too many sentence boundaries are being missed

- **`ABBREV_BACKOFF` and `MIN_COLLOC_FREQ`**: Control how frequency affects decisions.
  - Decrease for smaller training corpora
  - Increase for larger, more representative corpora

Remember that the optimal values depend on your specific domain and the relative cost of false positives (incorrectly split sentences) versus false negatives (missed sentence boundaries).

## Model Storage Formats

nupunkt supports multiple storage formats for trained models, each with different trade-offs between file size, loading speed, and human readability.

### Available Formats

| Format | File Extension | Pros | Cons |
|--------|----------------|------|------|
| JSON | `.json` | Human-readable, easy to inspect | Largest file size |
| JSON with LZMA | `.json.xz` | Smaller file size, still inspectable when decompressed | Slower loading than binary formats |
| Binary | `.bin` | Smallest file size, fastest loading | Not human-readable |

### Compression Options for Binary Format

The binary format supports multiple compression methods:

- `none`: No compression (fastest loading, largest size)
- `zlib`: Good balance of speed and size
- `lzma`: Best compression (smallest size, slightly slower loading)
- `gzip`: Similar to zlib, widely compatible

### Saving in Different Formats

You can save your trained model in any of these formats:

```python
# Save in different formats
trainer.get_params().save("model.json", format_type="json")
trainer.get_params().save("model.json.xz", format_type="json_xz", compression_level=6)
trainer.get_params().save("model.bin", format_type="binary", 
                          compression_method="lzma", compression_level=6)
```

The default format is binary with LZMA compression (level 6), which provides the best balance of file size and performance.

## Loading and Using Custom Models

Models can be loaded regardless of their format - nupunkt automatically detects and handles the format based on the file extension:

```python
from nupunkt import PunktSentenceTokenizer

# Load a tokenizer with a custom model (any format)
tokenizer = PunktSentenceTokenizer.load("my_custom_model.bin")  # or .json or .json.xz

# Use the tokenizer
sentences = tokenizer.tokenize("Your text here.")
```

## Evaluating Models

To evaluate a model's performance, compare its output against a gold standard:

```python
from nupunkt import PunktSentenceTokenizer

# Load your model
tokenizer = PunktSentenceTokenizer.load("my_custom_model.json")

# Load test data and ground truth
with open("test_text.txt", "r", encoding="utf-8") as f:
    test_text = f.read()

with open("ground_truth.txt", "r", encoding="utf-8") as f:
    ground_truth = f.read().splitlines()

# Tokenize the test text
predicted = tokenizer.tokenize(test_text)

# Calculate metrics (e.g., accuracy, precision, recall)
correct = sum(1 for p in predicted if p in ground_truth)
accuracy = correct / len(ground_truth)
precision = correct / len(predicted)
recall = correct / len(ground_truth)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```

## Utility Scripts

nupunkt provides several utility scripts to help with model training, analysis, and optimization:

### Training a Default Model

```bash
python -m scripts.train_default_model --format binary --compression lzma --level 6 --compare
```

This script trains a default model using the provided training data and abbreviation lists from both `data/legal_abbreviations.json` and `data/general_abbreviations.json`. Options:
- `--format`: Output format (binary, json_xz, json)
- `--compression`: Compression method for binary format (none, zlib, lzma, gzip)
- `--level`: Compression level (0-9)
- `--compare`: Compare different storage formats after training
- `--max-samples`: Maximum number of samples to use from training files

### Testing Models

```bash
python -m scripts.test_default_model --test
```

This script tests the default model on sample legal and financial texts. Options:
- `--test`: Run the model tests
- `--export`: Export the model to a different format
- `--format`, `--compression`, `--level`: Format options for export

### Advanced Utilities

Located in the `scripts/utils` directory:

- `model_info.py`: Displays detailed information about a model file
- `optimize_model.py`: Converts a model to the most efficient storage format
- `convert_model.py`: Converts between different model formats
- `benchmark_load_times.py`: Benchmarks loading and tokenization performance
- `test_tokenizer.py`: Tests a model with custom text

Example:
```bash
python -m scripts.utils.model_info --stats  # Show detailed model statistics
```

## Tips for Effective Training

1. **Use representative text**: The training corpus should be representative of the text you'll be processing.
2. **Size matters**: Larger training corpora generally lead to better results (10,000+ sentences recommended).
3. **Quality over quantity**: Clean, well-formatted text is better than a larger but noisy corpus.
4. **Domain-specific abbreviations**: Provide known abbreviations for your domain to improve performance.
5. **Tune hyperparameters**: Adjust parameters based on your domain's characteristics.
6. **Inspect the results**: After training, inspect the identified abbreviations, collocations, and sentence starters.
7. **Iterative refinement**: Start with a base model, then incrementally train on problematic examples.
8. **Model compression**: Use the binary format with LZMA compression for production deployment.
9. **Preserve abbreviations**: When incrementally training, usually keep `preserve_abbrevs=True`.