# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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