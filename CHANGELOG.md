# Changelog

## 0.5.1 (2025-04-05)

### Minor Updates

- Bump version to 0.5.1
- Documentation improvements
- Internal code quality enhancements

## 0.5.0 (2025-04-05)

### New Features

- **Added paragraph detection functionality:**
  - New `PunktParagraphTokenizer` for paragraph boundary detection
  - Paragraph breaks identified at sentence boundaries with multiple newlines
  - API for paragraph tokenization with span information
- **Added sentence and paragraph span extraction:**
  - Contiguous spans that preserve all whitespace
  - Spans guaranteed to cover entire text without gaps
  - API for getting spans with text content
- **Extended public API with new functions:**
  - `sent_spans()` and `sent_spans_with_text()` for sentence spans
  - `para_tokenize()`, `para_spans()`, and `para_spans_with_text()` for paragraphs
- Implemented singleton pattern for efficient model loading

### Performance Improvements

- Optimized model loading with caching mechanisms
- Single model instance shared across multiple operations
- Efficient memory usage for repeated sentence/paragraph tokenization

### Memory Optimizations

- **Added memory-efficient training for large text corpora:**
  - Early frequency pruning to discard rare items during training
  - Streaming processing mode to avoid storing complete token lists
  - Batch training for processing very large text collections
  - Configurable memory usage parameters
- Added memory benchmarking tools in `.benchmark` directory
- Added documentation for memory-efficient training
- Updated default training script with memory optimization options

### Performance Improvements

- Improved memory usage during training (up to 60% reduction)
- Support for training on very large text collections
- Pruning of low-frequency tokens, collocations, and sentence starters
- Configurable frequency thresholds and pruning intervals

## 0.4.0 (2025-03-31)

### Performance Improvements

- **Major tokenization performance optimization:**
  - Normal text processing: 31M chars/sec (9% faster)
  - Text without sentence endings: 1.4B chars/sec (383% faster)
  - Overall tokenization time reduced by 11%
  - Function call count reduced by 22%
- PunktToken initialization optimized with token caching and pre-computed properties
- Added fast path optimizations for texts without sentence boundaries
- Improved string handling and regex operations in hot spots
- Added profiling tools for performance analysis and optimization

## 0.3.0 (2025-03-31)

### New Features

- Implemented optimized binary model storage format with multiple compression options
- Added utility scripts for working with models (model_info.py, convert_model.py, optimize_model.py)
- Added check_abbreviation.py tool to check if a token is in the model's abbreviation list
- Added general_abbreviations.json file with common English abbreviations
- Updated training process to use both legal and general abbreviation lists
- Improved testing tools with test_tokenizer.py
- Added benchmarking utilities to compare model loading and tokenization performance
- Added profiling tools for performance analysis and optimization

### Performance Improvements

- Reduced default model size by 32% using binary LZMA format (1.5MB vs 2.2MB)
- Better memory usage during model loading
- Automatic format selection prioritizing the most efficient format
- **Major tokenization performance optimization:**
  - Normal text processing: 31M chars/sec (9% faster)
  - Text without sentence endings: 1.4B chars/sec (383% faster)
  - Overall tokenization time reduced by 11%
  - Function call count reduced by 22%
- PunktToken initialization optimized with token caching and pre-computed properties
- Added fast path optimizations for texts without sentence boundaries
- Improved string handling and regex operations in hot spots

## 0.2.0 (2025-03-30)

### New Features

- Initial release of nupunkt (renamed from punkt2)
- Added compression support for model files using LZMA
- Improved documentation