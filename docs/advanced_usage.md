# Advanced Usage

This guide covers advanced usage patterns and customization options for nupunkt.

## Memory-Efficient Training

NUpunkt now includes several optimizations for training on very large text collections with manageable memory usage. These optimizations allow you to train on much larger corpora than was previously possible.

### Memory Optimization Techniques

1. **Early Pruning**: Discard low-frequency items during training
2. **Streaming Processing**: Process text without storing complete token lists
3. **Batch Training**: Process text in manageable chunks
4. **Memory Configuration**: Fine-tune memory usage via parameters

### Basic Memory-Efficient Training

```python
from nupunkt.trainers.base_trainer import PunktTrainer

# Create a memory-efficient trainer
trainer = PunktTrainer(memory_efficient=True, verbose=True)

# Train with streaming mode (avoids storing all tokens at once)
trainer.train(text)

# Get the trained parameters
params = trainer.get_params()
```

### Batch Training for Very Large Corpora

For extremely large text collections, you can use batch training:

```python
from nupunkt.trainers.base_trainer import PunktTrainer

# Create a trainer
trainer = PunktTrainer(verbose=True)

# Split text into batches
batches = PunktTrainer.text_to_batches(huge_text, batch_size=1000000)

# Train in batches
trainer.train_batches(batches, verbose=True)
```

### Memory Configuration Parameters

You can fine-tune the memory usage with these parameters:

```python
trainer = PunktTrainer(memory_efficient=True)

# Configure memory usage
trainer.TYPE_FDIST_MIN_FREQ = 2      # Minimum frequency to keep a type
trainer.COLLOC_FDIST_MIN_FREQ = 3    # Minimum frequency for collocations
trainer.SENT_STARTER_MIN_FREQ = 2    # Minimum frequency for sentence starters
trainer.PRUNE_INTERVAL = 10000       # How often to prune (token count)
trainer.CHUNK_SIZE = 10000           # Size of token chunks for processing
```

### Command-Line Usage

When using the default training script, you can enable memory optimizations:

```bash
python -m scripts.train_default_model \
  --memory-efficient \
  --min-type-freq 2 \
  --prune-freq 10000 \
  --use-batches \
  --batch-size 1000000
```

### Memory Impact

The memory optimizations can significantly reduce memory usage:

| Optimization | Memory Reduction | Impact on Model Quality |
|--------------|------------------|-------------------------|
| Early Pruning | ~30-50% | Minimal (removes rare items) |
| Streaming Processing | ~40-60% | None (same algorithm) |
| Batch Training | Scales to any size | None (same algorithm) |

## Model Compression

Models in nupunkt are saved with LZMA compression by default, which significantly reduces file size while maintaining fast loading times. You can control compression settings when saving models:

```python
from nupunkt import PunktTrainer

# Train a model
trainer = PunktTrainer(training_text)

# Save with default compression (level 1 - fast compression)
trainer.save("my_model.json")  # Creates my_model.json.xz

# Save with higher compression (smaller file, slower compression)
trainer.save("my_model_high_compression.json", compression_level=6)

# Save without compression
trainer.save("my_model_uncompressed.json", compress=False)
```

Loading compressed models is transparent - the library automatically detects and handles compressed files:

```python
from nupunkt import PunktSentenceTokenizer

# Both of these will work regardless of whether the model is compressed
tokenizer1 = PunktSentenceTokenizer.load("my_model.json")
tokenizer2 = PunktSentenceTokenizer.load("my_model.json.xz")
```

Compressing an existing uncompressed model:

```python
from nupunkt.models import compress_default_model

# Compress the default model
compressed_path = compress_default_model()
print(f"Compressed model saved to: {compressed_path}")

# Compress with custom settings
custom_path = compress_default_model("custom_model.json.xz", compression_level=3)
```

## Custom Language Variables

For handling different languages or specific text domains, you can customize the language variables:

```python
from nupunkt import PunktLanguageVars, PunktSentenceTokenizer

class FrenchLanguageVars(PunktLanguageVars):
    # French uses colons as sentence endings
    sent_end_chars = (".", "?", "!", ":")
    
    # Customize internal punctuation
    internal_punctuation = ",:;«»"
    
    # Customize word tokenization pattern if needed
    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,«»]"

# Create a tokenizer with the custom language variables
french_vars = FrenchLanguageVars()
tokenizer = PunktSentenceTokenizer(lang_vars=french_vars)

# Tokenize French text
french_text = "Bonjour! Comment allez-vous? Très bien, merci."
sentences = tokenizer.tokenize(french_text)
```

## Custom Token Class

You can extend the `PunktToken` class to add additional functionality:

```python
from nupunkt import PunktToken, PunktSentenceTokenizer
from typing import Optional

class EnhancedToken(PunktToken):
    def __init__(self, tok, parastart=False, linestart=False, 
                 sentbreak=False, abbr=False, ellipsis=False,
                 pos_tag: Optional[str] = None):
        super().__init__(tok, parastart, linestart, sentbreak, abbr, ellipsis)
        self.pos_tag = pos_tag
    
    @property
    def is_noun(self) -> bool:
        return self.pos_tag == "NOUN" if self.pos_tag else False

# Create a tokenizer with the custom token class
tokenizer = PunktSentenceTokenizer(token_cls=EnhancedToken)
```

## Working with Spans

For applications that need character-level positions:

```python
from nupunkt import load_default_model

tokenizer = load_default_model()
text = "Hello world. This is a test."

# Get character-level spans for each sentence
spans = list(tokenizer.span_tokenize(text))
print(spans)  # [(0, 12), (13, 27)]

# Useful for highlighting or extracting sentences
for start, end in spans:
    print(f"Sentence: {text[start:end]}")
```

## Customizing Boundary Realignment

By default, nupunkt realigns sentence boundaries to handle trailing punctuation like quotes. You can disable this:

```python
sentences = tokenizer.tokenize(text, realign_boundaries=False)
```

## Reconfiguring Tokenizers

You can update tokenizer settings without retraining:

```python
from nupunkt import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer.load("my_model.json")

# Update configuration
config = {
    "parameters": {
        "abbrev_types": ["Dr", "Mr", "Mrs", "Ms", "Prof", "Inc", "Co"],
        # Add custom abbreviations
    }
}

tokenizer.reconfigure(config)
```

## Sentence Internal Spans

To get spans of sentences excluding certain punctuation:

```python
import re
from nupunkt import load_default_model

tokenizer = load_default_model()
text = "\"Hello,\" he said. \"How are you?\""

# Get spans for each sentence
spans = list(tokenizer.span_tokenize(text))

# Clean up internal spans by removing quotation marks
for start, end in spans:
    sentence = text[start:end]
    # Remove leading/trailing quotes and whitespace
    cleaned = re.sub(r'^[\s"\']+|[\s"\']+$', '', sentence)
    print(f"Original: {sentence}")
    print(f"Cleaned: {cleaned}")
```

## Parallelizing Tokenization

For processing large volumes of text, you can parallelize the work:

```python
from nupunkt import load_default_model
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def tokenize_text(text):
    tokenizer = load_default_model()
    return tokenizer.tokenize(text)

# Break large text into chunks
def chunk_text(text, chunk_size=100000):
    # Simple chunking by character count
    # More sophisticated chunking could preserve paragraph/document boundaries
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Process in parallel
def parallel_tokenize(text, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    chunks = chunk_text(text)
    all_sentences = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(tokenize_text, chunks))
        
    # Flatten results
    for sentences in chunk_results:
        all_sentences.extend(sentences)
    
    return all_sentences
```

## Customizing JSON Serialization

If you need to customize how models are saved and loaded:

```python
from nupunkt import PunktParameters, PunktTrainer
import json

# Extended parameters with custom metadata
class ExtendedParameters(PunktParameters):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.metadata = metadata or {}
    
    def to_json(self):
        data = super().to_json()
        data["metadata"] = self.metadata
        return data
    
    @classmethod
    def from_json(cls, data):
        params = super().from_json(data)
        params.metadata = data.get("metadata", {})
        return params

# Create parameters with metadata
params = ExtendedParameters(metadata={"domain": "legal", "source": "case_law"})

# Add some data
params.abbrev_types.add("etc")

# Save with pretty formatting
with open("custom_params.json", "w") as f:
    json.dump(params.to_json(), f, indent=2)
```

## Debug and Visualization Tools

To understand how nupunkt is making decisions:

```python
from nupunkt import PunktSentenceTokenizer, PunktToken

# Create a debugging tokenizer
class DebugTokenizer(PunktSentenceTokenizer):
    def tokenize_with_debug(self, text):
        # Tokenize and collect debugging info
        tokens = list(self._tokenize_words(text))
        first_pass = list(self._annotate_first_pass(tokens))
        
        # Second pass with decision tracking
        decisions = []
        for token1, token2 in self._pair_iter(first_pass):
            if token1.period_final or token1.tok in self._lang_vars.sent_end_chars:
                decision = self._second_pass_annotation(token1, token2)
                decisions.append((token1, decision))
        
        # Return the debug info along with sentences
        return {
            "sentences": self.tokenize(text),
            "tokens": tokens,
            "first_pass": first_pass,
            "decisions": decisions
        }

# Use the debug tokenizer
debug_tokenizer = DebugTokenizer()
results = debug_tokenizer.tokenize_with_debug("Dr. Smith went to Washington D.C. yesterday.")

# Print debug info
for token, decision in results["decisions"]:
    print(f"Token: {token.tok}, Abbr: {token.abbr}, SentBreak: {token.sentbreak}, Decision: {decision}")
```

These advanced usage patterns should help you customize nupunkt for specific needs and troubleshoot any issues that arise.