# nupunkt

> **Looking for Maximum Speed?** Check out [nupunkt-rs](https://github.com/alea-institute/nupunkt-rs) for a high-performance Rust implementation that's **3x faster** (30M+ characters/second) with the same accuracy. It provides a Python API via PyO3 bindings.

A high-precision, high-throughput sentence boundary detection library optimized for legal text processing, with zero runtime dependencies.

> **0.8.0**: layout-aware segmentation (headings, list items and page breaks in scanned documents), a public comparison against other splitters with recommendations by use case, and fixes for publicly reported cases. **0.7.0** brought deterministic output, a 25 KB model that loads in about 14 ms, twice the throughput of 0.6.0, and the standard segmentation interface. See the [changelog](CHANGELOG.md) for details and migration notes.

[![PyPI version](https://badge.fury.io/py/nupunkt.svg)](https://badge.fury.io/py/nupunkt)
[![Python Version](https://img.shields.io/pypi/pyversions/nupunkt.svg)](https://pypi.org/project/nupunkt/)
[![License](https://img.shields.io/github/license/alea-institute/nupunkt.svg)](https://github.com/alea-institute/nupunkt/blob/main/LICENSE)

## Overview

nupunkt is a next-generation implementation of the Punkt algorithm specifically optimized for legal text processing. It accurately detects sentence boundaries in complex legal documents where periods are used for abbreviations, citations, and other non-sentence-ending contexts.

Key features:
- **Zero dependencies, pure Python**: one 140 KB wheel, no compiled code, nothing to
  download at runtime; Python 3.11+ (tqdm optional for training progress bars)
- **Accurate on legal text**: boundary precision 0.90 on a 38,527-document legal gold set,
  and within 0.01 F1 of the best library on general English
  ([benchmarks](docs/benchmarks/README.md))
- **Layout-aware**: headings, list items and paragraph breaks are boundaries; sentences
  continue across page breaks in scanned documents ([layout rules](docs/layout.md))
- **One interface for words, sentences and paragraphs**: strings, tight character spans,
  or both, as lists or generators, plus a one-pass `Document` tree
- **Deterministic and fast**: same output in any order; about 23 million characters per
  second per core on unseen text; 25 KB model, 14 ms cold start, 20 MB resident
- **Trainable**: unsupervised training on your own text, plus per-abbreviation break
  rates from sentence-annotated text
- **Adaptive mode**: an optional confidence-based variant with a tunable threshold
- **CLI tools**: training, evaluation and model management

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
from nupunkt import sentences, segment

text = """RELEASE OF CLAIMS

Employee also specifically and forever releases the Acme Inc. (Company) and the Company
Parties from any claims based on unlawful employment discrimination, including, but not
limited to, the Federal Age Discrimination in Employment Act (29 U.S.C. § 621 et seq.).
This release does not include Employee's right to indemnification under Sec. 7.1.4 or
Ex. 1-1 of the Employment Agreement.
"""

for i, sentence in enumerate(sentences(text), 1):
    print(f"Sentence {i}: {sentence}")
# Sentence 1: RELEASE OF CLAIMS
# Sentence 2: Employee also specifically and forever releases the Acme Inc. (Company) ... (29 U.S.C. § 621 et seq.).
# Sentence 3: This release does not include ... of the Employment Agreement.

doc = segment(text)           # paragraphs -> sentences -> words, with character spans
doc.paragraphs[0].text        # 'RELEASE OF CLAIMS'
doc.sentences[1].span         # (19, 282)
```

`sent_tokenize(text)` (the original API) is still available and unchanged; it applies
Punkt only, without the layout rules, so the heading above would be merged into the
first sentence.

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

**Layout.** The segmentation interface is layout-aware by default: a blank line is a
hard sentence boundary (so headings and list items are units of their own), unless the
sentence plainly continues across it as at a page break in a scanned document. Pass
`paragraph_breaks=False` for Punkt-only behaviour, `line_breaks=True` to also cut at
heading and list-item line breaks, and use `blank_page_furniture(text)` to blank page
numbers without moving any offsets. See [docs/layout.md](docs/layout.md) for the rules,
their measured effect, and what happens at page breaks.

### All levels in one pass

`segment(text)` runs sentence segmentation once and returns a `Document` tree
(paragraphs, then sentences, then words). Paragraphs are the blocks between blank
lines, sentences never cross them, and words are only computed for a sentence when
you read them:

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

## Adaptive Tokenization

Adaptive mode dynamically discovers abbreviation patterns and improves sentence boundary detection:

```python
from nupunkt import sent_tokenize_adaptive

text = """Dr. Smith graduated from M.I.T. in 2020. She works at N.A.S.A. now.
Her colleague Mr. Johnson has a Ph.D. from U.C.L.A. and collaborates with researchers
at C.E.R.N. on quantum physics."""

# Use adaptive mode with abbreviation pattern detection
sentences = sent_tokenize_adaptive(text)

# Adjust confidence threshold (default: 0.7); higher = more breaks, lower = fewer
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

## Legacy Span Functions

The original span functions remain available and unchanged. Unlike the segmentation
interface above, their spans are *contiguous*: each carries the whitespace up to the
next span, and no layout rules apply.

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

These legacy span functions guarantee contiguous spans with no gaps, full coverage of
the input text, and preservation of all whitespace. For tight spans that index the
source exactly, use `sentence_spans` / `sentence_segments` or `segment()`.

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

Measured on a 12th-gen Intel core, one process, unseen text, fresh interpreter for
cold start (full methodology and comparisons in
[docs/benchmarks/performance.md](docs/benchmarks/performance.md)):

| | nupunkt 0.5.1 | nupunkt 0.6.0 | nupunkt 0.7.0+ |
|---|--:|--:|--:|
| Import + first call | 541 ms | 1,284 ms | 14 ms |
| Resident memory after load | 143 MB | 260 MB | 20 MB |
| Throughput, unseen legal text | 24.7 Mchar/s | 11.0 Mchar/s | 24.1 Mchar/s |
| Wheel / model size | 5.6 MB | 9.1 MB | 0.14 MB / 26 KB |

Repeated input runs faster (43-68 Mchar/s) because boundary decisions are memoized.
Compiled splitters are faster on raw throughput: sentencex by about 5x and blingfire by
1.3x. For maximum speed with the same algorithm, see
[nupunkt-rs](https://github.com/alea-institute/nupunkt-rs), a Rust implementation with
Python bindings. Benchmark scripts live in `scripts/benchmarks/`.

## Choosing nupunkt

nupunkt is one of two pure-Python, zero-dependency sentence splitters on PyPI and the
only one that is competitive on accuracy and speed. It leads on legal text and on
documents with layout, is within 0.01 F1 of the best library on general English, and
is slower than compiled splitters (sentencex, blingfire) on raw throughput. The full
comparison, with recommendations by use case and every case where another library is
the better choice, is in [docs/benchmarks/comparison.md](docs/benchmarks/comparison.md).

## Documentation

- [Getting Started Guide](docs/getting-started.md) - Detailed usage examples
- [Training Guide](docs/training-guide.md) - Train custom models
- [Algorithm Overview](docs/algorithm.md) - How nupunkt works
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Layout Rules](docs/layout.md) - Blank lines, headings, lists and page breaks
- [Benchmarks and Comparison](docs/benchmarks/README.md) - Accuracy, speed, footprint, and when to use something else

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