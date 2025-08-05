# API Reference

Complete reference for nupunkt's Python API.

## Package Contents

```python
import nupunkt

# Version
nupunkt.__version__  # '0.6.0'

# Functions
nupunkt.sent_tokenize()
nupunkt.sent_tokenize_adaptive()
nupunkt.sent_spans()
nupunkt.sent_spans_with_text()
nupunkt.sent_spans_adaptive()
nupunkt.sent_spans_with_text_adaptive()
nupunkt.para_tokenize()
nupunkt.para_spans()
nupunkt.para_spans_with_text()
nupunkt.load()
nupunkt.load_default_model()

# Classes
nupunkt.PunktSentenceTokenizer
nupunkt.PunktParagraphTokenizer
nupunkt.PunktTrainer
nupunkt.PunktParameters
nupunkt.PunktLanguageVars
nupunkt.PunktToken
```

## Core Functions

### sent_tokenize

```python
sent_tokenize(
    text: str,
    model: str = "default",
    adaptive: bool = False,
    confidence_threshold: float = 0.7,
    dynamic_abbrev: bool = True,
    return_confidence: bool = False,
    debug: bool = False,
) -> Union[List[str], List[Tuple[str, float]]]
```

Tokenize text into sentences.

**Parameters:**
- `text`: The text to tokenize
- `model`: Model to use - "default", a file path, or a model name
- `adaptive`: Enable adaptive tokenization with dynamic pattern recognition
- `confidence_threshold`: Decision threshold for adaptive mode (0.0-1.0)
- `dynamic_abbrev`: Discover abbreviation patterns at runtime
- `return_confidence`: Return (sentence, confidence) tuples instead of just sentences
- `debug`: Enable debug output showing decision reasoning

**Returns:**
- List of sentences, or list of (sentence, confidence) tuples if `return_confidence=True`

**Example:**
```python
# Basic usage
sentences = sent_tokenize("Hello world. How are you?")

# Adaptive mode
sentences = sent_tokenize(text, adaptive=True)

# With confidence scores
results = sent_tokenize(text, adaptive=True, return_confidence=True)
for sentence, confidence in results:
    print(f"[{confidence:.2f}] {sentence}")
```

### sent_tokenize_adaptive

```python
sent_tokenize_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    return_confidence: bool = False,
    debug: bool = False,
    **kwargs,
) -> Union[List[str], List[Tuple[str, float]]]
```

Convenience wrapper for adaptive sentence tokenization.

**Parameters:**
- `text`: Text to tokenize
- `threshold`: Confidence threshold (0.0-1.0)
- `model`: Model to use
- `return_confidence`: Return confidence scores
- `debug`: Enable debug output
- `**kwargs`: Additional arguments passed to sent_tokenize

**Example:**
```python
# Simple adaptive tokenization
sentences = sent_tokenize_adaptive("She got her Ph.D. at M.I.T. yesterday.")

# With custom threshold
sentences = sent_tokenize_adaptive(text, threshold=0.85)
```

### sent_spans

```python
sent_spans(text: str) -> List[Tuple[int, int]]
```

Get sentence spans (start, end character positions) using the default model.

**Parameters:**
- `text`: The text to segment

**Returns:**
- List of (start_index, end_index) tuples

**Notes:**
- Spans are guaranteed to be contiguous (no gaps)
- Spans cover the entire input text
- Includes whitespace between sentences

**Example:**
```python
spans = sent_spans("First sentence. Second sentence.")
# Returns: [(0, 16), (16, 32)]

for start, end in spans:
    sentence = text[start:end]
    print(f"[{start}:{end}] {sentence}")
```

### sent_spans_with_text

```python
sent_spans_with_text(text: str) -> List[Tuple[str, Tuple[int, int]]]
```

Get sentences with their spans using the default model.

**Parameters:**
- `text`: The text to segment

**Returns:**
- List of (sentence_text, (start_index, end_index)) tuples

**Notes:**
- Spans are guaranteed to be contiguous
- Preserves all whitespace for perfect text reconstruction

**Example:**
```python
results = sent_spans_with_text("Hello world. How are you?")
# Returns: [("Hello world. ", (0, 13)), ("How are you?", (13, 25))]

for sentence, (start, end) in results:
    print(f"[{start}:{end}] {sentence}")
    assert text[start:end] == sentence  # Always true
```

### sent_spans_adaptive

```python
sent_spans_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    dynamic_abbrev: bool = True,
    **kwargs,
) -> List[Tuple[int, int]]
```

Get sentence spans using adaptive tokenization with confidence scoring.

**Parameters:**
- `text`: The text to segment
- `threshold`: Confidence threshold (0.0-1.0)
- `model`: Model to use - "default", a file path, or a model name
- `dynamic_abbrev`: Discover abbreviation patterns at runtime
- `**kwargs`: Additional arguments passed to the tokenizer

**Returns:**
- List of (start_index, end_index) tuples

**Notes:**
- Uses adaptive algorithm that dynamically recognizes abbreviations
- Spans are guaranteed to be contiguous
- Better handles unknown abbreviations like M.I.T., Ph.D., etc.

**Example:**
```python
# Get spans for text with unknown abbreviations
spans = sent_spans_adaptive("She studied at M.I.T. in Cambridge.")
# Returns: [(0, 35)] - single sentence preserved

# Tune for high precision
spans = sent_spans_adaptive(legal_text, threshold=0.85)
```

### sent_spans_with_text_adaptive

```python
sent_spans_with_text_adaptive(
    text: str,
    threshold: float = 0.7,
    model: str = "default",
    dynamic_abbrev: bool = True,
    return_confidence: bool = False,
    **kwargs,
) -> Union[List[Tuple[str, Tuple[int, int]]], List[Tuple[str, Tuple[int, int], float]]]
```

Get sentences with their spans using adaptive tokenization.

**Parameters:**
- `text`: The text to segment
- `threshold`: Confidence threshold (0.0-1.0)
- `model`: Model to use - "default", a file path, or a model name
- `dynamic_abbrev`: Discover abbreviation patterns at runtime
- `return_confidence`: Include confidence scores in the output
- `**kwargs`: Additional arguments passed to the tokenizer

**Returns:**
- If `return_confidence=False`: List of (sentence, (start_index, end_index)) tuples
- If `return_confidence=True`: List of (sentence, (start_index, end_index), confidence) tuples

**Notes:**
- Provides both sentence text and character positions
- Optionally includes confidence scores for each sentence
- Spans are contiguous and cover the entire text

**Example:**
```python
# Get sentences with spans
results = sent_spans_with_text_adaptive("Dr. Smith studied at M.I.T. today.")
# Returns: [('Dr. Smith studied at M.I.T. today.', (0, 34))]

# With confidence scores
results = sent_spans_with_text_adaptive(text, return_confidence=True)
for sentence, (start, end), confidence in results:
    print(f"[{confidence:.2f}] [{start}:{end}] {sentence}")
```

### para_tokenize

```python
para_tokenize(text: str) -> List[str]
```

Tokenize text into paragraphs using the default model.

**Parameters:**
- `text`: The text to tokenize

**Returns:**
- List of paragraphs

**Example:**
```python
paragraphs = para_tokenize("""First paragraph.

Second paragraph.""")
```

### para_spans

```python
para_spans(text: str) -> List[Tuple[int, int]]
```

Get paragraph spans (start, end character positions).

**Parameters:**
- `text`: The text to segment

**Returns:**
- List of (start_index, end_index) tuples

**Example:**
```python
spans = para_spans(text)
for start, end in spans:
    paragraph = text[start:end]
```

### para_spans_with_text

```python
para_spans_with_text(text: str) -> List[Tuple[str, Tuple[int, int]]]
```

Get paragraphs with their spans.

**Parameters:**
- `text`: The text to segment

**Returns:**
- List of (paragraph_text, (start_index, end_index)) tuples

**Example:**
```python
for para_text, (start, end) in para_spans_with_text(text):
    print(f"[{start}:{end}] {para_text}")
```

### load

```python
load(model: str) -> PunktSentenceTokenizer
```

Load a Punkt model by name or path.

**Parameters:**
- `model`: Either:
  - "default" for the built-in model
  - A file path to a model (.bin, .json, .json.gz, .json.xz)
  - A model name to search in standard locations

**Returns:**
- A PunktSentenceTokenizer initialized with the model

**Model Search Paths:**
1. Package models directory (built-in models)
2. Platform-specific user data directory:
   - Linux: `~/.local/share/nupunkt/models`
   - macOS: `~/Library/Application Support/nupunkt/models`
   - Windows: `%LOCALAPPDATA%\nupunkt\models`
3. Current working directory/models

**Example:**
```python
# Load default model
tokenizer = nupunkt.load("default")

# Load from file
tokenizer = nupunkt.load("/path/to/model.bin")

# Load by name (searches standard paths)
tokenizer = nupunkt.load("legal_custom")
```

### load_default_model

```python
load_default_model() -> PunktSentenceTokenizer
```

Load the default pre-trained model.

**Returns:**
- The default PunktSentenceTokenizer

**Example:**
```python
tokenizer = load_default_model()
sentences = list(tokenizer.tokenize(text))
```

## Classes

### PunktSentenceTokenizer

The main tokenizer class for sentence segmentation.

#### Methods

##### tokenize
```python
tokenize(text: str, realign_boundaries: bool = True) -> List[str]
```
Tokenize text into sentences.

##### span_tokenize
```python
span_tokenize(text: str, realign_boundaries: bool = True) -> Iterator[Tuple[int, int]]
```
Return sentence spans as (start, end) tuples.

##### tokenize_with_spans
```python
tokenize_with_spans(text: str, realign_boundaries: bool = True) -> List[Tuple[str, Tuple[int, int]]]
```
Return sentences with their character spans. Spans are contiguous and cover the entire text.

##### add_abbreviation
```python
add_abbreviation(abbrev: str) -> None
```
Add a single abbreviation to the model.

##### add_abbreviations
```python
add_abbreviations(abbrevs: List[str]) -> None
```
Add multiple abbreviations to the model.

##### remove_abbreviation
```python
remove_abbreviation(abbrev: str) -> None
```
Remove an abbreviation from the model.

##### save
```python
save(
    file_path: Union[str, Path],
    compress: bool = True,
    compression_level: int = 1
) -> None
```
Save the model to a file.

##### text_contains_sentbreak
```python
text_contains_sentbreak(text: str) -> bool
```
Check if text contains a sentence break.

**Example Usage:**
```python
# Create/load tokenizer
tokenizer = nupunkt.load("default")

# Add custom abbreviations
tokenizer.add_abbreviation("Ph.D")
tokenizer.add_abbreviations(["M.D.", "CEO", "CFO"])

# Tokenize
sentences = tokenizer.tokenize(text)

# Get spans
for start, end in tokenizer.span_tokenize(text):
    sentence = text[start:end]
    print(f"[{start}:{end}] {sentence}")

# Save modified model
tokenizer.save("custom_model.bin")
```

### PunktParagraphTokenizer

Tokenizer for paragraph segmentation.

#### Constructor
```python
PunktParagraphTokenizer(sentence_tokenizer: PunktSentenceTokenizer)
```

#### Methods

##### tokenize
```python
tokenize(text: str) -> Iterator[str]
```
Tokenize text into paragraphs.

##### span_tokenize
```python
span_tokenize(text: str) -> Iterator[Tuple[int, int]]
```
Return paragraph spans.

##### tokenize_with_spans
```python
tokenize_with_spans(text: str) -> Iterator[Tuple[str, Tuple[int, int]]]
```
Return paragraphs with their spans.

**Example:**
```python
# Create paragraph tokenizer
sent_tokenizer = nupunkt.load("default")
para_tokenizer = PunktParagraphTokenizer(sent_tokenizer)

# Tokenize
paragraphs = list(para_tokenizer.tokenize(text))
```

### PunktTrainer

Trainer for learning Punkt parameters from text.

#### Constructor
```python
PunktTrainer(
    train_text: str = None,
    verbose: bool = False,
    lang_vars: PunktLanguageVars = None,
    token_cls: Type[PunktToken] = PunktToken,
)
```

#### Methods

##### get_params
```python
get_params() -> PunktParameters
```
Get the learned parameters.

##### save
```python
save(file_path: Union[str, Path]) -> None
```
Save the trained model.

**Example:**
```python
# Train a model
trainer = PunktTrainer(training_text, verbose=True)
params = trainer.get_params()

# Create tokenizer with trained parameters
tokenizer = PunktSentenceTokenizer(params)

# Save the model
trainer.save("my_model.bin")
```

### PunktParameters

Container for Punkt algorithm parameters.

#### Attributes
- `abbrev_types`: Set of abbreviations
- `collocations`: Set of word pairs (collocations)
- `sent_starters`: Set of sentence starting words
- `ortho_context`: Dictionary of orthographic contexts

**Example:**
```python
# Access parameters
tokenizer = nupunkt.load("default")
params = tokenizer._params

print(f"Abbreviations: {len(params.abbrev_types)}")
print(f"Sample: {sorted(params.abbrev_types)[:10]}")
```

## Training Functions

### train_model

```python
from nupunkt.training import train_model

train_model(
    training_texts: Union[str, List[str], List[Path]],
    abbreviations: List[str] = None,
    abbreviation_files: List[Path] = None,
    output_path: Path = None,
    max_samples: int = None,
    format_type: str = "binary",
    hyperparameters: Union[str, Dict, PunktHyperparameters] = None,
    # ... many more options
) -> PunktTrainer
```

Train a new Punkt model. See [Training Guide](training-guide.md) for details.

## Hyperparameters

### PunktHyperparameters

```python
from nupunkt.training.hyperparameters import PunktHyperparameters

# Presets
conservative = PunktHyperparameters.conservative()
balanced = PunktHyperparameters.balanced()
aggressive = PunktHyperparameters.aggressive()

# Custom
params = PunktHyperparameters(
    abbrev_threshold=0.1,
    sent_starter_threshold=20.0,
    collocation_threshold=8.0
)
```

## Evaluation Functions

### evaluate_model

```python
from nupunkt.evaluation import evaluate_model

evaluation = evaluate_model(
    model_path="model.bin",
    dataset_path="test.jsonl",
    output_path="results.json"
)

print(f"F1 Score: {evaluation.metrics.f1:.2%}")
```

### compare_models

```python
from nupunkt.evaluation import compare_models

results = compare_models(
    models=["model1.bin", "model2.bin"],
    dataset_path="test.jsonl"
)
```

## Utility Functions

### Model Information

```python
from nupunkt.training.optimizer import get_model_info

info = get_model_info("model.bin")
print(f"Format: {info['format']}")
print(f"Size: {info['size_mb']:.2f} MB")
print(f"Abbreviations: {info['parameters']['abbreviations']}")
```

### Model Conversion

```python
from nupunkt.training import convert_model_format

convert_model_format(
    input_path="model.json",
    output_path="model.bin",
    format_type="binary"
)
```

## Constants

### Orthographic Context Flags

```python
from nupunkt.core.constants import (
    ORTHO_BEG_UC,  # Uppercase at sentence beginning
    ORTHO_MID_UC,  # Uppercase mid-sentence
    ORTHO_UNK_UC,  # Unknown position uppercase
    ORTHO_BEG_LC,  # Lowercase at sentence beginning
    # ... etc
)
```

## Error Handling

nupunkt raises standard Python exceptions:

```python
try:
    tokenizer = nupunkt.load("nonexistent_model")
except FileNotFoundError as e:
    print(f"Model not found: {e}")

try:
    sentences = sent_tokenize(None)
except TypeError as e:
    print(f"Invalid input: {e}")
```

## Performance Considerations

1. **Model Caching**: The `load()` function caches up to 8 models
2. **Tokenizer Reuse**: Create once, use many times
3. **Batch Processing**: Process multiple texts with same tokenizer
4. **Memory**: Models use ~50-100MB RAM when loaded

## Thread Safety

PunktSentenceTokenizer instances are thread-safe for reading (tokenization) but not for writing (adding/removing abbreviations).

```python
# Safe: Multiple threads tokenizing
tokenizer = nupunkt.load("default")
# Each thread can call tokenizer.tokenize(text)

# Unsafe: Modifying while tokenizing
# Don't call add_abbreviation() while other threads tokenize
```