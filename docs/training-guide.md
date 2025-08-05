# Training Custom Models

This guide covers how to train custom nupunkt models for your specific domain or use case.

## Overview

nupunkt can be trained on your own text to learn domain-specific:
- Abbreviations
- Sentence starters
- Collocations
- Orthographic patterns

The default model is trained on legal text, but you can create models optimized for medical, scientific, social media, or any other domain.

## Quick Start

### Basic Training

```bash
# Train from a text file
nupunkt train corpus.txt -o my_model.bin

# Train from multiple files
nupunkt train file1.txt file2.txt file3.txt -o combined_model.bin

# Train without default abbreviations
nupunkt train corpus.txt --no-default-abbreviations -o minimal_model.bin
```

### Python API

```python
from nupunkt.training import train_model

# Train from text
trainer = train_model(
    ["corpus.txt"],
    output_path="my_model.bin"
)

# Get training statistics
params = trainer._params
print(f"Learned {len(params.abbrev_types)} abbreviations")
print(f"Learned {len(params.sent_starters)} sentence starters")
print(f"Learned {len(params.collocations)} collocations")
```

## Training Data Formats

### Plain Text Files

The simplest format - just provide text files:

```bash
nupunkt train document.txt -o model.bin
```

### JSONL Format

For structured data, use JSONL with a "text" field:

```json
{"text": "First document. It has multiple sentences."}
{"text": "Second document. Another example."}
```

```bash
nupunkt train data.jsonl.gz -o model.bin
```

### HuggingFace Datasets

Train directly from HuggingFace datasets:

```bash
# Use hf: prefix
nupunkt train hf:alea-institute/kl3m-data-usc -o legal_model.bin

# Limit samples for testing
nupunkt train hf:alea-institute/kl3m-data-usc --max-samples 1000 -o test_model.bin
```

## Hyperparameter Tuning

### Using Presets

nupunkt provides three hyperparameter presets:

```bash
# Conservative: Fewer sentence breaks (high precision)
nupunkt train corpus.txt --hyperparameters conservative -o precise_model.bin

# Balanced: Good balance (recommended)
nupunkt train corpus.txt --hyperparameters balanced -o balanced_model.bin

# Aggressive: More sentence breaks (high recall)
nupunkt train corpus.txt --hyperparameters aggressive -o sensitive_model.bin
```

### Understanding Presets

| Preset | Abbrev Threshold | Sent Starter | Collocation | Use Case |
|--------|-----------------|--------------|-------------|----------|
| Conservative | 0.05 | 30.0 | 10.0 | Formal text, legal documents |
| Balanced | 0.1 | 15.0 | 7.88 | General purpose (default) |
| Aggressive | 0.3 | 5.0 | 3.0 | Informal text, social media |

### Custom Hyperparameters

Fine-tune individual thresholds:

```bash
# Lower abbreviation threshold = more abbreviations detected
nupunkt train corpus.txt --abbrev-threshold 0.05 -o model.bin

# Higher sentence starter threshold = fewer starters
nupunkt train corpus.txt --sent-starter-threshold 30.0 -o model.bin
```

Or use a JSON file:

```json
{
  "abbrev_threshold": 0.08,
  "sent_starter_threshold": 20.0,
  "collocation_threshold": 8.0,
  "min_colloc_freq": 5
}
```

```bash
nupunkt train corpus.txt --hyperparameters my_params.json -o model.bin
```

## Memory-Efficient Training

For large corpora (>1GB), use memory-efficient mode:

```bash
nupunkt train huge_corpus.txt \
  --batch-size 1000000 \
  --prune-freq 10000 \
  --min-type-freq 5 \
  -o large_model.bin
```

Options:
- `--batch-size`: Process text in chunks (default: 1M chars)
- `--prune-freq`: Remove rare tokens periodically
- `--min-type-freq`: Minimum frequency to keep tokens

## Adding Custom Abbreviations

### From JSON Files

Create an abbreviations file:

```json
{
  "abbreviations": [
    "Ph.D.", "M.D.", "CEO", "CFO", "Ltd.", "Inc.",
    "Jan.", "Feb.", "Mar.", "Apr.", "etc."
  ]
}
```

Use during training:

```bash
nupunkt train corpus.txt \
  --abbreviations my_abbrevs.json \
  -o model_with_abbrevs.bin
```

### Default Abbreviations

By default, nupunkt loads abbreviations from:
- `data/general_abbreviations.json` (1,006 entries)
- `data/legal_abbreviations.json` (1,045 entries)

To train without these:

```bash
nupunkt train corpus.txt --no-default-abbreviations -o minimal_model.bin
```

## Preparing Training Data

### Using the Data Preparation Script

For creating mixed training datasets from multiple sources:

```bash
# Install datasets dependency
# Note: This is only needed for data preparation, not runtime
uv run --with datasets python scripts/prepare_mixed_training_data.py \
  --samples 1000 \
  --output training_data.jsonl.gz
```

Custom dataset selection:

```bash
uv run --with datasets python scripts/prepare_mixed_training_data.py \
  --datasets \
    alea-institute/kl3m-data-usc \
    alea-institute/kl3m-data-edgar-10-k \
  --samples 5000 2000 \
  --output mixed_training.jsonl.gz
```

### Best Practices for Training Data

1. **Volume**: At least 100K words for good results
2. **Diversity**: Include various document types
3. **Quality**: Clean, properly formatted text
4. **Balance**: Mix formal and informal if needed

## Model Formats

### Binary Format (Recommended)

Fast loading, compact size:

```bash
nupunkt train corpus.txt --format binary -o model.bin
```

### JSON Format

Human-readable, larger size:

```bash
nupunkt train corpus.txt --format json -o model.json

# Compressed JSON
nupunkt train corpus.txt --format json --compression gzip -o model.json.gz
```

### Converting Formats

```bash
# Binary to JSON
nupunkt convert model.bin model.json

# JSON to binary
nupunkt convert model.json model.bin
```

## Evaluating Your Model

### Quick Test

Use `--test` flag to test immediately after training:

```bash
nupunkt train corpus.txt -o model.bin --test
```

### Comprehensive Evaluation

Prepare evaluation data in JSONL format with sentence boundaries marked:

```json
{"text": "First sentence.<|sentence|>Second sentence.<|sentence|>Third."}
```

Run evaluation:

```bash
nupunkt evaluate test_data.jsonl -m my_model.bin

# Compare models
nupunkt evaluate test_data.jsonl --compare \
  --models default.bin custom.bin
```

### Hyperparameter Optimization

Find optimal parameters for your dataset:

```bash
nupunkt optimize-params train.jsonl test.jsonl \
  -o optimized_model.bin \
  --trials 20
```

## Domain-Specific Examples

### Legal Text

```bash
# Use conservative preset for high precision
nupunkt train legal_corpus.txt \
  --hyperparameters conservative \
  --abbreviations legal_abbrevs.json \
  -o legal_model.bin
```

### Medical Text

```bash
# Add medical abbreviations
nupunkt train medical_corpus.txt \
  --abbreviations medical_abbrevs.json \
  --abbrev-threshold 0.05 \
  -o medical_model.bin
```

### Social Media

```bash
# Use aggressive preset for informal text
nupunkt train tweets.jsonl \
  --hyperparameters aggressive \
  --no-default-abbreviations \
  -o social_model.bin
```

### Academic Text

```bash
# Balanced with academic abbreviations
nupunkt train papers.txt \
  --hyperparameters balanced \
  --abbreviations academic_abbrevs.json \
  -o academic_model.bin
```

## Troubleshooting

### Model is splitting too much

- Use conservative hyperparameters
- Lower the abbreviation threshold
- Increase sentence starter threshold

### Model is not splitting enough

- Use aggressive hyperparameters  
- Increase abbreviation threshold
- Decrease sentence starter threshold

### Out of memory

- Enable memory-efficient mode
- Reduce batch size
- Increase prune frequency

### Specific abbreviations not recognized

- Add them explicitly via JSON file
- Lower abbreviation threshold
- Check if they appear in training data

## Advanced Training

### Python API with Custom Settings

```python
from nupunkt.training import train_model
from nupunkt.training.hyperparameters import PunktHyperparameters

# Custom hyperparameters
params = PunktHyperparameters(
    abbrev_threshold=0.08,
    sent_starter_threshold=20.0,
    collocation_threshold=8.0
)

# Train with progress callback
def progress(stage, current, total):
    print(f"{stage}: {current}/{total}")

trainer = train_model(
    training_texts=["corpus1.txt", "corpus2.txt"],
    hyperparameters=params,
    memory_efficient=True,
    progress_callback=progress,
    output_path="custom_model.bin"
)

# Inspect learned parameters
print(f"Abbreviations: {len(trainer._params.abbrev_types)}")
print(f"Sent starters: {len(trainer._params.sent_starters)}")
```

### Incremental Training

While nupunkt doesn't support true incremental training, you can:

1. Extract parameters from an existing model
2. Use them as a starting point
3. Train on new data

```python
# Load existing model
existing = nupunkt.load("existing_model.bin")

# Extract abbreviations
abbrevs = list(existing._params.abbrev_types)

# Train new model with these abbreviations
trainer = train_model(
    ["new_corpus.txt"],
    abbreviations=abbrevs,
    output_path="updated_model.bin"
)
```

## Next Steps

- Test your model with the [evaluation tools](getting-started.md#command-line-usage)
- Use [adaptive mode](algorithm.md#adaptive-tokenization-new-in-v060) for runtime improvements
- Check the [API reference](api-reference.md) for advanced usage