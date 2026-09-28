# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-09-28

Layout-aware segmentation, a public comparison against other sentence splitters, and
fixes for publicly reported cases. All 0.7.0 and 0.6.0 function and method names keep
working; the segmentation interface introduced in 0.7.0 changes its default where blank
lines occur inside what Punkt considered one sentence (see Migration notes).

### Highlights

| | 0.7.0 | 0.8.0 |
|---|---|---|
| Legal gold set with paragraph markers, F1 (P / R) | 0.714 (0.93 / 0.58) | 0.932 (0.95 / 0.92) |
| UD English GUM with paragraphs, F1 | 0.927 | 0.972 |
| UD English EWT with paragraphs, F1 | 0.869 | 0.919 |
| Legal gold set, Punkt-only, F1 | 0.782 | 0.784 |
| yasbd 92-string golden rules, passes | 67 | 72 |
| Publicly reported failure cases fixed (of 41) | 27 pass | 29 pass, 0 regressed |

### Added
- **Layout-aware segmentation** (`nupunkt.layout`, [docs/layout.md](docs/layout.md)).
  The segmentation interface (`sentences`, `sentence_spans`, `sentence_segments`,
  `segmenter`, `segment`, `iter_segments`) now treats a blank line as a hard sentence
  boundary by default (`paragraph_breaks=True`), so headings, captions and list items
  become units of their own. Measured on gold sets that keep paragraph structure, recall
  rises 9-35 points at unchanged precision (legal with paragraph markers F1 0.716 -> 0.932,
  UD EWT 0.869 -> 0.919, UD GUM 0.928 -> 0.972). A sentence still continues across a blank
  line when the preceding line has no terminal punctuation and either ends with a hyphen
  or the next block starts lowercase, which is the page-break shape of OCR'd documents.
  `line_breaks=True` (opt-in) also cuts at heading and list-item line breaks.
  `blank_page_furniture(text)` blanks page-number lines and form feeds while preserving
  offsets. The legacy `sent_tokenize` family is unchanged.
- `PunktSentenceTokenizer.abbreviations` (read-only snapshot) and `.parameters` (the live
  `PunktParameters`), so integrations no longer need to read `_params`.
- `PunktSentenceTokenizer.EXCL_QUEST_LOWERCASE_CONTINUES` (opt-in, off): treat `!` or `?`
  followed by a lowercase word as sentence-internal (`Yahoo! in 1995`). Off by default
  because it costs a point of F1 on informal web text.
- `docs/benchmarks/`: reproducible accuracy, performance, Golden Rules and known-cases
  evidence, with scripts under `scripts/benchmarks/`, and `comparison.md` with a Pareto
  analysis against other splitters (including dependency footprint) and recommendations
  by use case.

### Changed
- Closing punctuation tokens (`"`, `)`, `”` ...) are transparent when pairing tokens, so an
  abbreviation followed by a closing quote sees the next word (`'Do not follow me.' Then`
  now splits; `"U.S." He` too).
- A spaced or Unicode ellipsis followed by the pronoun `I` no longer ends the sentence.
- Runs of terminal punctuation separated by spaces (`Hello ! ! !`) are one terminator.
- `term` removed from the bundled abbreviation set (it broke `2. Term.` list items).
- The adaptive threshold documentation had its direction backwards: a higher
  `confidence_threshold` produces more sentence breaks, a lower one fewer.
- Segmentation-interface methods (`segments`, `texts`, `spans` and `iter_*`) accept keyword
  options and pass them to `iter_segments`; unknown options raise `TypeError`.

### Corrected
- The 0.7.0 release notes overstated throughput: "60+ MB/s" was measured on repeated
  input, where boundary decisions are memoized. On unseen text 0.7.0 runs at 23-26
  Mchar/s per core, twice 0.6.0 and about equal to 0.5.1. The 0.7.0 table below has been
  corrected, and the full measurement is in `docs/benchmarks/performance.md`.

### Migration notes
- Output of the segmentation interface changes where blank lines occur inside what Punkt
  considered one sentence. Pass `paragraph_breaks=False` to keep the 0.7.0 behaviour, or set
  `tokenizer.paragraph_breaks = False` on a tokenizer object.
- `paragraph_segments` / `paragraphs` now return one paragraph per blank-line block;
  `para_tokenize` and `para_spans` are unchanged.

## [0.7.0] - 2026-09-27

A correctness and performance release. Tokenization is deterministic again, the bundled
model is 25 KB instead of 9.2 MB, the tokenizer is twice as fast as 0.6.0 on unseen text, and there is a
single segmentation interface for words, sentences and paragraphs. All 0.6 function and
method names keep working with their previous behaviour.

### Highlights

| | 0.6.0 | 0.7.0 |
|---|---|---|
| Legal gold set boundary F1 (precision) | 0.722 (0.776) | 0.782 (0.902) |
| Bundled model file | 9.2 MB | 25 KB |
| Import plus first call, fresh process | 1.3 s | 14 ms |
| Resident memory after load | ~320 MB | ~19 MB |
| Throughput on unseen legal text, one core | 11 Mchar/s | 24 Mchar/s |
| Throughput on repeated text (memoized decisions) | 13 Mchar/s | 43-68 Mchar/s |

### Added
- **Standard segmentation interface** (`nupunkt.segmentation`). Words, sentences and
  paragraphs share one shape: `words` / `word_spans` / `word_segments`, `sentences` /
  `sentence_spans` / `sentence_segments`, `paragraphs` / `paragraph_spans` /
  `paragraph_segments`, plus `segmenter(level, ...)` for a reusable object with generator
  forms (`iter_segments`, `iter_texts`, `iter_spans`). `Segment` is a `(text, start, end)`
  named tuple with a `.span` property. Spans are tight (`text[start:end]` is exactly the
  segment, no surrounding whitespace, never overlapping); `contiguous(segments, text)`
  derives gap-free coverage. `SegmenterMixin` derives the six methods from a single
  `iter_segments`, and `PunktSentenceTokenizer`, `AdaptiveTokenizer`,
  `PunktParagraphTokenizer` and the new `WordSegmenter` all implement it.
- **One-pass hierarchical segmentation**: `nupunkt.segment(text, model="default",
  adaptive=False)` returns a `Document` (`paragraphs` -> `sentences` -> `words`) from a
  single sentence pass; words are computed lazily per sentence. `Paragraph` and `Sentence`
  are `Segment` subclasses, the flat `Document.sentences` / `.words` equal the per-level
  functions, and `Document.to_dict()` gives a JSON-ready tree.
- **Per-abbreviation break rates.** `PunktParameters.abbrev_break_rates` records, per
  abbreviation, how often it precedes a capitalized word and how often that position is a
  sentence boundary. Learned from sentence-annotated text (`<|sentence|>` markers) by
  `PunktTrainer.train()` or `PunktTrainer.learn_break_rates(texts)`;
  `PunktTrainer.strip_sentence_markers()` converts annotated text to plain text plus
  offsets. The tokenizer breaks after an abbreviation seen at least `BREAK_RATE_MIN_COUNT`
  (20) times before a capitalized word when its rate is at least `BREAK_RATE_HIGH` (0.8);
  the `BREAK_RATE_LOW` veto is off by default. The bundled model ships rates learned from
  the Universal Dependencies English treebanks. Models without the field load and
  tokenize exactly as before.
- Deterministic boundary heuristics, each validated on the legal gold set and on general
  English (UD EWT, UD GUM, Brown):
  - Unicode closing quotes and guillemets (`” ’ »`) and markdown `*` act like ASCII closers.
  - `…` and spaced `. . .` ellipses can end a sentence before a capitalized word.
  - A terminator followed by closing punctuation and a lowercase word on the same line does
    not end the sentence (`"Is it?" he asked.`); runs like `?!` and `!!!` are one terminator.
  - Line-start list enumerators (`1.`, `(a).`, `IV.`) are never sentences on their own.
  - Prenominal titles (`Dr.`, `Mr.`, `Gen.`, ...) before a capitalized word never end a sentence.
  - A period-final token followed by a paragraph break and a capitalized word is a boundary.
  - Abbreviation candidates may contain apostrophes and `&` (`aff'd.`, `gov't.`).
  - The word after a candidate boundary is no longer truncated at its first period, so
    `App. No. 5` sees `No.` as an abbreviation (264 fewer false positives on the gold set).
  - Adaptive mode no longer treats a capitalized word before a closing quote as an abbreviation.
- `PunktParameters.compact_ortho_context()` and `drop_ortho_context()`, and
  `scripts/compact_default_model.py` (with `--drop-ortho`, `--remove`, `--drop-malformed`)
  to export inference models.
- `PunktSentenceTokenizer.clear_decision_cache()` for in-place parameter edits that do not
  change any collection's size.
- 168 new tests (`test_determinism`, `test_heuristics`, `test_segmentation`,
  `test_document`, `test_decision_engine`, `test_break_rates`).

### Changed
- **Bundled model no longer ships an orthographic context.** Ablation showed the 891k
  entries (97% of the file) change fewer than 0.1% of boundaries on the legal gold set and
  on three general-English corpora (F1 within 0.0003 everywhere). Abbreviations, sentence
  starters and collocations are unchanged. Models you train yourself keep their
  orthographic context unless you drop it. Model format version is now `1.1.0`.
- **Abbreviation data curated**: the plain words `Court.`, `Case.`, `Cases.`, `Trial.`,
  `Judge.`, `Law.` and `Child.` were removed from `legal_abbreviations.json` and the bundled
  model, along with 73 malformed entries.
- **String-level decision engine.** Boundary decisions are computed from token strings and
  memoized per context instead of building `PunktToken` objects for every candidate.
  Output is identical; subclasses that override annotation hooks (such as
  `AdaptiveTokenizer`) transparently keep the token-based path.
- Hot-path cleanups with identical output: no per-call frozenset rebuilds, no unused
  ellipsis scans, no string-keyed LRU caches, no per-sentence copies during realignment.
- Default abbreviation lists are bundled in `nupunkt/data/` so
  `train_model(use_default_abbreviations=True)` works from installed wheels.
- Type annotations use PEP 604/585 syntax; `ty` replaces `mypy` and `ruff format`
  replaces `black` in the dev dependencies (`mypy.ini` removed). Package declares
  `Typing :: Typed`.
- The `ortho_cache_size`, `sent_starter_cache_size` and `whitespace_cache_size` arguments of
  `PunktSentenceTokenizer` are accepted but ignored.
- `nupunkt --version` reports the package version from `nupunkt.__version__`.

### Fixed
- **Tokenization was not deterministic.** `PunktToken` instances were cached and shared
  between positions and calls while the annotation passes mutated them, so the same input
  could tokenize differently depending on what the process had seen before. Tokens are
  now always fresh; only immutable per-string fields are memoized.
- `add_abbreviation()` / `remove_abbreviation()` had no effect on models loaded from disk.
- Tokenizers built from in-process parameters were about 12x slower than loaded models.
- `PunktSentenceTokenizer(training_text)` raised `OSError: File name too long` for text
  with a run of more than 255 characters without a path separator.
- Custom `token_cls` subclasses were silently ignored.
- `PunktTrainer`: abbreviation reclassification was quadratic in vocabulary size;
  memory-efficient training pruned counts mid-pass and corrupted frequencies;
  `math domain error` from the Dunning log-likelihood on inconsistent counts is guarded.
- `scripts/profiling/profile_sent_tokenize_adaptive.py` referenced an undefined variable.

### Migration notes
- No code changes are required. `sent_tokenize`, `sent_spans`, `sent_spans_with_text`,
  `para_tokenize`, `para_spans`, `para_spans_with_text`, the `_adaptive` variants and the
  `tokenize` / `span_tokenize` / `tokenize_with_spans` methods keep their exact previous
  semantics, including contiguous spans that carry surrounding whitespace. New code should
  prefer the segmentation interface, whose spans are tight.
- Output changes: results that depended on process history are now stable, and the
  heuristics above change some boundaries (see the Highlights table). If you pinned
  expected output, re-generate it.
- The bundled model cannot be used as a starting point for continued training, because
  it no longer carries the orthographic context. Train from text, or from a model you
  saved yourself.
- Models saved by 0.6.0 load unchanged. Models saved by 0.7.0 (format `1.1.0`) load in
  0.6.0 as well; the `abbrev_break_rates` field is ignored there.

## [0.6.0] - 2025-08-04

### Added
- **AdaptiveTokenizer** in hybrid module for enhanced sentence boundary detection
  - Dynamic abbreviation pattern detection (M.I.T., Ph.D., B.B.C., etc.)
  - Context-aware boundary decisions considering continuation words
  - Confidence-based adaptive refinement that preserves base Punkt accuracy
  - Debug mode with detailed decision explanations
  - Handles abbreviations not present in training data
- **New API functions for adaptive tokenization:**
  - `sent_tokenize_adaptive()` - Uses confidence scoring with dynamic abbreviations
  - `sent_tokenize(adaptive=True)` - Enable adaptive mode in standard API
- **Cross-platform model loading and management:**
  - Platform-specific paths: Linux (~/.local/share), macOS (~/Library/Application Support), Windows (%LOCALAPPDATA%)
  - XDG base directory specification support
  - Model migration from legacy locations
  - CLI model management commands (list, info, install, migrate)
- **Model version tracking:**
  - Version metadata in all serialized models
  - Compatibility warnings for version mismatches
  - Graceful handling of models from older versions
- Comprehensive test suite for hybrid tokenizers
- Documentation for hybrid tokenizer usage and development
- Comprehensive test coverage for `SentenceTokenizer`, `ParagraphTokenizer`, and `PunktTrainer`
- New `nupunkt.load()` function for flexible model loading
- CLI entry point `nupunkt` for training and model management
- `nupunkt/training/` module with refactored training logic
- Hybrid sentence boundary detection experiments in `nupunkt/hybrid/`
- `ConfidenceSentenceTokenizer` with confidence scoring approach
- Research documentation for hybrid approaches
- Support for loading models by name or path in `sent_tokenize()`
- Model discovery in package and user directories

### Changed
- **BREAKING**: Renamed `PunktSentenceTokenizer.__init__` parameter from `train_text` to `model_or_text`
- **BREAKING**: Default model format changed from binary to gzipped JSON (.json.gz)
- `PunktSentenceTokenizer` now accepts file paths to model files in `__init__`
- `sent_tokenize()` now accepts optional `model` parameter
- Refactored training scripts from `scripts/` into `nupunkt.training` module
- Pinned development dependencies for reproducible environments
- Updated CLI from click to argparse (maintaining zero dependencies)
- Model serialization now uses gzipped JSON exclusively for better maintainability
- Improved CLI commands to display results properly

### Fixed
- Original ConfidenceSentenceTokenizer was too aggressive in splitting sentences
- Hybrid tokenizers now properly integrate with base Punkt algorithm
- Model loading API now properly handles `.json.xz` files
- CLI evaluate and optimize commands now display results instead of silently exiting
- Fixed attribute names in hyperparameter optimization (ABBREV vs ABBREV_THRESHOLD)
- Resolved numerous type annotation issues throughout the codebase
- Fixed unused variable warnings and improved code quality

## [0.5.1] - 2025-04-05

### Changed
- Documentation improvements
- Internal code quality enhancements

## [0.5.0] - 2025-04-05

### Added
- **Paragraph detection functionality:**
  - New `PunktParagraphTokenizer` for paragraph boundary detection
  - Paragraph breaks identified at sentence boundaries with multiple newlines
  - API for paragraph tokenization with span information
- **Sentence and paragraph span extraction:**
  - Contiguous spans that preserve all whitespace
  - Spans guaranteed to cover entire text without gaps
  - API for getting spans with text content
- **Extended public API with new functions:**
  - `sent_spans()` and `sent_spans_with_text()` for sentence spans
  - `para_tokenize()`, `para_spans()`, and `para_spans_with_text()` for paragraphs
- Singleton pattern for efficient model loading
- **Memory-efficient training for large text corpora:**
  - Early frequency pruning to discard rare items during training
  - Streaming processing mode to avoid storing complete token lists
  - Batch training for processing very large text collections
  - Configurable memory usage parameters
- Memory benchmarking tools in `.benchmark` directory
- Documentation for memory-efficient training

### Changed
- Updated default training script with memory optimization options

### Performance
- Optimized model loading with caching mechanisms
- Single model instance shared across multiple operations
- Efficient memory usage for repeated sentence/paragraph tokenization
- Improved memory usage during training (up to 60% reduction)
- Support for training on very large text collections
- Pruning of low-frequency tokens, collocations, and sentence starters
- Configurable frequency thresholds and pruning intervals

## [0.4.0] - 2025-03-19

### Added
- Binary model format (`.bin`) for faster loading and smaller file sizes
- Support for multiple compression methods (zlib, lzma, gzip)
- Model optimization tools for reducing storage size
- Format conversion utilities

### Changed
- Default model now uses binary format instead of JSON
- Improved model loading performance (10x faster)

### Performance
- Binary models load ~10x faster than compressed JSON
- Binary format reduces storage size by 40-60%
- Support for selective compression based on size/speed tradeoffs

## [0.3.0] - 2025-02-14

### Added
- Support for custom abbreviation lists during training
- Dynamic abbreviation management (add/remove at runtime)
- Improved handling of domain-specific abbreviations

### Changed
- Training API now accepts abbreviation files
- Better handling of edge cases in abbreviation detection

## [0.2.0] - 2025-01-10

### Added
- Basic paragraph tokenization support
- Span extraction for sentences
- Improved documentation

### Fixed
- Edge cases in sentence boundary detection
- Unicode handling improvements

## [0.1.0] - 2024-12-15

### Added
- Initial release
- Core Punkt algorithm implementation
- Basic sentence tokenization
- Pre-trained English model
- Training capabilities for custom models

[Unreleased]: https://github.com/alea-institute/nupunkt/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/alea-institute/nupunkt/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/alea-institute/nupunkt/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/alea-institute/nupunkt/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/alea-institute/nupunkt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/alea-institute/nupunkt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/alea-institute/nupunkt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alea-institute/nupunkt/releases/tag/v0.1.0