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

You can specify common abbreviations that should always be recognized:

```python
from nupunkt import PunktTrainer

# Create a subclass with custom common abbreviations
class LegalTrainer(PunktTrainer):
    COMMON_ABBREVS = ["art.", "sec.", "para.", "fig.", "p.", "pp."]

trainer = LegalTrainer(training_text)
```

## Customizing Training Parameters

You can customize various parameters that control the training process:

```python
from nupunkt import PunktTrainer

# Create a custom trainer
class CustomTrainer(PunktTrainer):
    # Threshold for identifying abbreviations (lower = more aggressive)
    ABBREV = 0.08
    
    # Minimum frequency for rare abbreviations
    ABBREV_BACKOFF = 5
    
    # Threshold for identifying collocations
    COLLOCATION = 7.5
    
    # Threshold for identifying sentence starters
    SENT_STARTER = 30.0
    
    # Minimum frequency for collocations
    MIN_COLLOC_FREQ = 7
    
    # Maximum length for abbreviation detection
    MAX_ABBREV_LENGTH = 10

trainer = CustomTrainer(training_text, verbose=True)
```

## Loading and Using Custom Models

```python
from nupunkt import PunktSentenceTokenizer

# Load a tokenizer with a custom model
tokenizer = PunktSentenceTokenizer.load("my_custom_model.json")

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

## Tips for Effective Training

1. **Use representative text**: The training corpus should be representative of the text you'll be processing.
2. **Size matters**: Larger training corpora generally lead to better results.
3. **Quality over quantity**: Clean, well-formatted text is better than a larger but noisy corpus.
4. **Inspect the results**: After training, inspect the identified abbreviations, collocations, and sentence starters.
5. **Iterative refinement**: Start with a base model, then incrementally train on problematic examples.
6. **Preserve abbreviations**: When incrementally training, usually keep `preserve_abbrevs=True`.