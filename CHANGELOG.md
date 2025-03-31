# Changelog

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