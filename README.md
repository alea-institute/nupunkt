# nupunkt

> **Looking for Maximum Speed?** Check out [nupunkt-rs](https://github.com/alea-institute/nupunkt-rs) for a high-performance Rust implementation that's **3x faster** (30M+ characters/second) with the same accuracy. It provides a Python API via PyO3 bindings.

A high-precision, high-throughput sentence boundary detection library optimized for legal text processing, with zero runtime dependencies.

> **0.7.0**: deterministic output, a 25 KB bundled model that loads in milliseconds, a 4-5x faster tokenizer, and a standard segmentation interface for words, sentences and paragraphs. See the [changelog](CHANGELOG.md) for details and migration notes.

[![PyPI version](https://badge.fury.io/py/nupunkt.svg)](https://badge.fury.io/py/nupunkt)
[![Python Version](https://img.shields.io/pypi/pyversions/nupunkt.svg)](https://pypi.org/project/nupunkt/)
[![License](https://img.shields.io/github/license/alea-institute/nupunkt.svg)](https://github.com/alea-institute/nupunkt/blob/main/LICENSE)

## Overview

nupunkt is a next-generation implementation of the Punkt algorithm specifically optimized for legal text processing. It accurately detects sentence boundaries in complex legal documents where periods are used for abbreviations, citations, and other non-sentence-ending contexts.

Key features:
- **Zero dependencies**: Pure Python 3.11+ (tqdm optional for progress bars)
- **Adaptive mode**: an optional confidence-based variant with a tunable precision/recall threshold
- **High precision**: 91.1% precision on legal text benchmarks
- **High performance**: tens of millions of characters per second on standard CPU hardware, with a 25 KB model
- **Pre-trained model**: Ready to use with legal-optimized abbreviations
- **Trainable**: Can be trained on domain-specific text
- **Paragraph detection**: Split text into both sentences and paragraphs
- **CLI tools**: Complete command-line interface for training and evaluation

## Paper

For the research behind this implementation, see:
> **Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary**  
> Michael J Bommarito, Daniel Martin Katz, Jillian Bommarito  
> arXiv:2504.04131 [cs.CL]  
> https://arxiv.org/abs/2504.04131

Interactive demo available at: https://sentences.aleainstitute.ai/

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
Employment Act (29 U.S.C. § 621 et. seq.). This release does not include Employee's right to indemnification, 
and related insurance coverage, under Sec. 7.1.4 or Ex. 1-1 of the Employment Agreement.
"""

# Tokenize into sentences
sentences = sent_tokenize(text)

for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}\n")
```

## Segmentation Interface

Words, sentences and paragraphs share one interface. For each level there are
three list functions and a reusable segmenter object with generator forms:

| Level     | Strings           | Spans                   | Both                       |
|-----------|-------------------|-------------------------|----------------------------|
| word      | `words(text)`     | `word_spans(text)`      | `word_segments(text)`      |
| sentence  | `sentences(text)` | `sentence_spans(text)`  | `sentence_segments(text)`  |
| paragraph | `paragraphs(text)`| `paragraph_spans(text)` | `paragraph_segments(text)` |

```python
from nupunkt import sentences, sentence_spans, sentence_segments, segmenter, contiguous

text = "Dr. Smith arrived.  He left at 5 p.m.\n\nThe end."

sentences(text)         # ['Dr. Smith arrived.', 'He left at 5 p.m.', 'The end.']
sentence_spans(text)    # [(0, 18), (20, 37), (39, 47)]
sentence_segments(text) # [Segment(text='Dr. Smith arrived.', start=0, end=18), ...]

# Reusable object with generators: iter_segments / iter_texts / iter_spans
seg = segmenter("paragraph")
for para in seg.iter_segments(text):
    print(para.start, para.end, para.text)
```

`Segment` is a named tuple `(text, start, end)` with `.span` for `(start, end)`.
Spans are tight: `text[start:end]` is exactly the segment with no surrounding
whitespace, segments never overlap, and whitespace stays in the gaps. When you
need gap-free coverage of the whole input, use `contiguous(segments, text)`.

Sentence functions accept `model=` and `adaptive=`; `segmenter("sentence", adaptive=True)`
returns the adaptive tokenizer. The older `sent_tokenize`, `sent_spans`, `para_tokenize`
family remains available unchanged.

### All levels in one pass

`segment(text)` runs sentence segmentation once and returns a `Document` tree
(paragraphs, then sentences, then words). Paragraphs come from the same sentence
boundaries, and words are only computed for a sentence when you read them:

```python
from nupunkt import segment

doc = segment(text)
for para in doc.paragraphs:
    for sent in para.sentences:
        print(sent.start, sent.end, [w.text for w in sent.words])

doc.sentences   # flat list, == sentence_segments(text)
doc.words       # flat list, == word_segments(text)
doc.to_dict()   # JSON-ready nested dict; to_dict(words=False) omits words
```

Every node is a `Segment` (it unpacks as `(text, start, end)` and has `.span`),
and each word lies inside its sentence, which lies inside its paragraph. For
paragraphs plus sentences, `segment()` costs one sentence pass where
`paragraph_segments()` + `sentence_segments()` cost two.

## Adaptive Tokenization (New in v0.6.0)

Adaptive mode dynamically discovers abbreviation patterns and improves sentence boundary detection:

```python
from nupunkt import sent_tokenize_adaptive

text = """Dr. Smith graduated from M.I.T. in 2020. She works at N.A.S.A. now.
Her colleague Mr. Johnson has a Ph.D. from U.C.L.A. and collaborates with researchers
at C.E.R.N. on quantum physics."""

# Use adaptive mode with abbreviation pattern detection
sentences = sent_tokenize_adaptive(text)

# Adjust confidence threshold (default: 0.7)
sentences = sent_tokenize_adaptive(text, threshold=0.8)

# Get confidence scores for each decision
sentences_with_scores = sent_tokenize_adaptive(text, return_confidence=True)
for sentence, confidence in sentences_with_scores:
    print(f"[{confidence:.2f}] {sentence}")
```

The adaptive tokenizer:
- Automatically detects abbreviation patterns (M.I.T., Ph.D., etc.)
- Uses context clues to make better decisions
- Provides confidence scores for each boundary decision
- Falls back to the robust base algorithm when uncertain

## Sentence and Paragraph Spans

Get character-level spans for sentences and paragraphs:

```python
from nupunkt import sent_spans, sent_spans_with_text, para_spans, para_spans_with_text

# Get sentence spans (start, end positions)
sentence_spans = sent_spans(text)

# Get sentences with their spans
sentences_with_spans = sent_spans_with_text(text)
for sentence, (start, end) in sentences_with_spans:
    print(f"[{start}:{end}] {sentence}")

# Same for paragraphs
paragraph_spans = para_spans(text)
paragraphs_with_spans = para_spans_with_text(text)
```

### Adaptive Spans

Get spans using the adaptive algorithm for better abbreviation handling:

```python
from nupunkt import sent_spans_adaptive, sent_spans_with_text_adaptive

# Get adaptive sentence spans
text = "Dr. Smith studied at M.I.T. in Cambridge."
spans = sent_spans_adaptive(text)
# Returns: [(0, 41)] - single sentence preserved

# Get sentences with spans
results = sent_spans_with_text_adaptive(text)
for sentence, (start, end) in results:
    print(f"[{start}:{end}] {sentence}")

# With confidence scores
results = sent_spans_with_text_adaptive(text, return_confidence=True)
for sentence, (start, end), confidence in results:
    print(f"[{confidence:.2f}] [{start}:{end}] {sentence}")
```

All span functions guarantee:
- Contiguous spans with no gaps
- Full coverage of the input text
- Preservation of all whitespace

## Paragraph Detection

```python
from nupunkt import para_tokenize

# Get paragraph text
paragraphs = para_tokenize(text)
```

## Command-line Interface

### Basic usage
```bash
# Using Python directly
echo "Hello world. How are you?" | python -c "import sys; from nupunkt import sent_tokenize; print('\n'.join(sent_tokenize(sys.stdin.read())))"

# Or create a simple script
python -c "from nupunkt import sent_tokenize; import sys; [print(s) for s in sent_tokenize(sys.stdin.read())]"
```

### Training models
```bash
# Train from text files
nupunkt train corpus.txt --output model.bin

# Train from HuggingFace datasets
nupunkt train hf:alea-institute/kl3m-data-usc -o legal_model.bin

# Memory-efficient training for large datasets
nupunkt train huge_corpus.txt --batch-size 1000000 --min-type-freq 5
```

### Evaluating models
```bash
# Evaluate a model
nupunkt evaluate test_data.jsonl -m my_model.bin

# Compare multiple models
nupunkt evaluate test_data.jsonl --compare --models baseline.bin custom.bin
```

### Model management
```bash
# Convert between formats
nupunkt convert model.json model.bin

# Get model information
nupunkt info model.bin

# Optimize hyperparameters
nupunkt optimize-params train.jsonl test.jsonl -o best_model.bin
```

## Performance

nupunkt is designed for high-precision, high-throughput processing with a tiny
footprint:

- **Bundled model: 25 KB**, loads in about 2 ms, about 20 MB resident memory after load
- **Deterministic**: the same input always produces the same output, in any order
- **Fast path processing** for texts without sentence boundaries
- **Memoized token properties** and a lean per-boundary decision path

Typical throughput on legal text is 15-25 million characters per second per
core on a modern machine (measured on a 4 MB corpus with the default model;
numbers vary with hardware and text). A reusable benchmark lives in
`scripts/profiling/`.

## Documentation

- [Getting Started Guide](docs/getting-started.md) - Detailed usage examples
- [Training Guide](docs/training-guide.md) - Train custom models
- [Algorithm Overview](docs/algorithm.md) - How nupunkt works
- [API Reference](docs/api-reference.md) - Complete API documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use nupunkt in your research, please cite:

```bibtex
@article{bommarito2025precise,
  title={Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary},
  author={Bommarito, Michael J and Katz, Daniel Martin and Bommarito, Jillian},
  journal={arXiv preprint arXiv:2504.04131},
  year={2025}
}
```

## Acknowledgments

nupunkt is based on the Punkt algorithm originally developed by Tibor Kiss and Jan Strunk.