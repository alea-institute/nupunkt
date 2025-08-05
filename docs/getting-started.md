# Getting Started with nupunkt

This guide covers common use cases and provides detailed examples for using nupunkt effectively.

## Installation

```bash
pip install nupunkt
```

No additional dependencies are required. The library comes with a pre-trained model optimized for legal text.

## Basic Sentence Tokenization

### Simple Usage

```python
from nupunkt import sent_tokenize

text = "This is the first sentence. This is the second one."
sentences = sent_tokenize(text)
print(sentences)
# ['This is the first sentence.', 'This is the second one.']
```

### Handling Abbreviations

nupunkt's pre-trained model handles common abbreviations automatically:

```python
text = """Dr. Smith went to Washington, D.C. yesterday. 
She met with Sen. Johnson at 2 p.m. to discuss the proposal."""

sentences = sent_tokenize(text)
for i, sent in enumerate(sentences, 1):
    print(f"{i}: {sent}")
# 1: Dr. Smith went to Washington, D.C. yesterday.
# 2: She met with Sen. Johnson at 2 p.m. to discuss the proposal.
```

### Legal Text Examples

The model is specifically optimized for legal text with citations and complex punctuation:

```python
text = """Pursuant to 28 U.S.C. § 1332, diversity jurisdiction exists.
The court in Smith v. Jones, 123 F.3d 456 (9th Cir. 1997), held that
the statute applies broadly. See also Johnson v. Davis, 789 F.2d 123."""

sentences = sent_tokenize(text)
for i, sent in enumerate(sentences, 1):
    print(f"{i}: {sent}")
# 1: Pursuant to 28 U.S.C. § 1332, diversity jurisdiction exists.
# 2: The court in Smith v. Jones, 123 F.3d 456 (9th Cir. 1997), held that
# the statute applies broadly.
# 3: See also Johnson v. Davis, 789 F.2d 123.
```

## Adaptive Tokenization

The adaptive mode dynamically recognizes abbreviation patterns not in the training data:

### Basic Adaptive Usage

```python
from nupunkt import sent_tokenize_adaptive

# Text with abbreviations that might not be in training data
text = "She got her Ph.D. at M.I.T. in 2020. Now she works at N.A.S.A. headquarters."

# Standard mode may split incorrectly
standard = sent_tokenize(text)
print("Standard:", standard)
# Standard: ['She got her Ph.D. at M.I.T. in 2020.', 'Now she works at N.A.S.A.', 'headquarters.']

# Adaptive mode recognizes the patterns better
adaptive = sent_tokenize_adaptive(text)
print("Adaptive:", adaptive)
# Adaptive: ['She got her Ph.D. at M.I.T. in 2020.', 'Now she works at N.A.S.A. headquarters.']
```

### Understanding Confidence Scores

The adaptive tokenizer provides confidence scores for each sentence boundary decision:

```python
text = "Dr. Smith arrived at 3 p.m. She was late. The meeting started without her."

sentences_with_confidence = sent_tokenize_adaptive(
    text, 
    return_confidence=True
)

for sentence, confidence in sentences_with_confidence:
    print(f"[{confidence:.2f}] {sentence}")
# [0.33] Dr. Smith arrived at 3 p.m. She was late.
# [1.00] The meeting started without her.
```

Note: Lower confidence scores indicate the tokenizer is less certain about the boundary.

### Tuning the Confidence Threshold

The confidence threshold parameter affects how the adaptive tokenizer makes decisions. Counter-intuitively, higher thresholds can lead to more aggressive splitting:

```python
text = "Dr. Smith arrived at 3 p.m. She was late."

# Lower threshold (default 0.7)
print("Threshold 0.7:", sent_tokenize_adaptive(text, threshold=0.7))
# Threshold 0.7: ['Dr. Smith arrived at 3 p.m. She was late.']

# Higher threshold
print("Threshold 0.9:", sent_tokenize_adaptive(text, threshold=0.9))
# Threshold 0.9: ['Dr. Smith arrived at 3 p.m.', 'She was late.']
```

### Debug Mode

Enable debug mode to see the reasoning behind each decision:

```python
text = "Visit I.B.M. today."

sentences = sent_tokenize_adaptive(text, debug=True)
# === Tokenization Decisions ===
# 
# Token: 'I.B.M.' -> 'today'
#   Base: False, Final: False
#   Confidence: 0.00, Overridden: False
#   Reasons: Base algorithm: no, Known abbreviation, Lowercase follows

print("Result:", sentences)
# Result: ['Visit I.B.M. today.']
```

## Paragraph Tokenization

nupunkt can also split text into paragraphs based on sentence boundaries and newlines:

### Basic Paragraph Detection

```python
from nupunkt import para_tokenize

text = """This is the first paragraph. It has multiple sentences.

This is the second paragraph. It discusses a different topic."""

paragraphs = para_tokenize(text)
print(f"Found {len(paragraphs)} paragraphs")
# Found 2 paragraphs

for i, para in enumerate(paragraphs, 1):
    print(f"Paragraph {i}: {repr(para)}")
# Paragraph 1: 'This is the first paragraph. It has multiple sentences.'
# Paragraph 2: '\n\nThis is the second paragraph. It discusses a different topic.'
```

Note: Paragraphs include leading newlines when present.

### Getting Paragraph Spans

For applications that need to know the exact position of paragraphs:

```python
from nupunkt import para_spans, para_spans_with_text

text = """First paragraph here.

Second paragraph here."""

# Just the spans
spans = para_spans(text)
print("Spans:", spans)
# Spans: [(0, 21), (21, 45)]

# Spans with text
spans_with_text = para_spans_with_text(text)
for para_text, (start, end) in spans_with_text:
    print(f"[{start}:{end}] {repr(para_text)}")
# [0:21] 'First paragraph here.'
# [21:45] '\n\nSecond paragraph here.'
```

The spans are contiguous and cover the entire input text without gaps.

## Using Custom Models

### Loading Models

```python
import nupunkt

# Load the default model
tokenizer = nupunkt.load("default")

# Load a custom model from file
# tokenizer = nupunkt.load("path/to/my_model.bin")

# Use with sent_tokenize
sentences = nupunkt.sent_tokenize(text, model="default")
```

### Model Search Paths

When loading by name (not full path), nupunkt searches in:
1. Package models directory (built-in models)
2. Platform-specific user data directory:
   - Linux: `~/.local/share/nupunkt/models`
   - macOS: `~/Library/Application Support/nupunkt/models`
   - Windows: `%LOCALAPPDATA%\nupunkt\models`
3. Current working directory/models

### Dynamic Abbreviation Management

Add or remove abbreviations at runtime:

```python
tokenizer = nupunkt.load("default")

# Check if abbreviation exists (lowercase without period)
print("'dr' is abbreviation:", 'dr' in tokenizer._params.abbrev_types)
# 'dr' is abbreviation: True

# Add custom abbreviations
tokenizer.add_abbreviation("Ph.D")
tokenizer.add_abbreviations(["M.D.", "B.S.", "M.S."])

# Remove an abbreviation
tokenizer.remove_abbreviation("Mr.")

# Now tokenize with updated abbreviations
text = "She has a Ph.D. from Harvard."
sentences = list(tokenizer.tokenize(text))
print(sentences)
# ['She has a Ph.D. from Harvard.']
```

## Command-Line Usage

### Basic CLI

```bash
# Using Python directly for tokenization
echo "Hello world. How are you?" | python -c "import sys; from nupunkt import sent_tokenize; print('\n'.join(sent_tokenize(sys.stdin.read())))"

# From file
python -c "import sys; from nupunkt import sent_tokenize; print('\n'.join(sent_tokenize(sys.stdin.read())))" < input.txt

# With adaptive mode
echo "Dr. Smith arrived." | python -c "import sys; from nupunkt import sent_tokenize_adaptive; print('\n'.join(sent_tokenize_adaptive(sys.stdin.read())))"
```

### Training and Evaluation

See the [Training Guide](training-guide.md) for detailed information on:
- Training custom models
- Using HuggingFace datasets
- Evaluating model performance
- Optimizing hyperparameters

## Performance Tips

1. **Model caching**: The `sent_tokenize` function automatically caches models for reuse
2. **Use standard mode when possible**: Adaptive mode is slightly slower
3. **Batch processing**: Process multiple texts together when possible
4. **Avoid debug mode in production**: It significantly impacts performance

```python
# Good - model is cached automatically
for text in documents:
    sentences = sent_tokenize(text)  # Reuses cached model

# Less efficient - creates new tokenizer each time
for text in documents:
    tokenizer = PunktSentenceTokenizer.load("model.bin")
    sentences = tokenizer.tokenize(text)
```

## Common Issues and Solutions

### Issue: Incorrect splits with domain-specific abbreviations

**Solution**: Use adaptive mode or add custom abbreviations

```python
# Option 1: Adaptive mode
sentences = sent_tokenize_adaptive(text)

# Option 2: Add abbreviations to tokenizer
tokenizer = nupunkt.load("default")
tokenizer.add_abbreviation("Ltd.")
sentences = list(tokenizer.tokenize(text))
```

### Issue: Memory usage with large files

**Solution**: Process in chunks

```python
def process_large_file(filename):
    tokenizer = nupunkt.load("default")
    with open(filename, 'r') as f:
        for line in f:
            sentences = tokenizer.tokenize(line)
            for sentence in sentences:
                yield sentence
```

### Issue: Version warnings

You may see warnings like:
```
UserWarning: Model was created with nupunkt 0.5.1, but current version is 0.6.0
```

This is informational and the model will still work correctly.

## Next Steps

- Learn about [training custom models](training-guide.md)
- Understand the [Punkt algorithm and hybrid approaches](algorithm.md)
- Explore the [full API reference](api-reference.md)