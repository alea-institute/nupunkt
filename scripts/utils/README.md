# Nupunkt Utility Scripts

Scripts to manage, optimize, and analyze nupunkt models.

## Utility Scripts

### Model Management

- `benchmark_load_times.py`: Compare load times of different model formats
- `convert_model.py`: Convert between different model storage formats
- `model_info.py`: Display information about a model file
- `optimize_model.py`: Optimize a model file for size and loading performance

### Tokenization and Analysis

- `check_abbreviation.py`: Check if a token is in the model's abbreviation list
- `test_tokenizer.py`: Test the tokenizer on sample text
- `explain_tokenization.py`: Show detailed explanation for tokenization decisions

## Usage Examples

### Check Abbreviation Tool

Check if a specific token is recognized as an abbreviation in the model:

```bash
# Check a specific token
python check_abbreviation.py "Dr."

# List all abbreviations in the model
python check_abbreviation.py --list

# Count total abbreviations
python check_abbreviation.py --count

# Find abbreviations starting with a specific prefix
python check_abbreviation.py --startswith "u.s"

# Check in a custom model
python check_abbreviation.py "Dr." --model /path/to/custom_model.bin
```
