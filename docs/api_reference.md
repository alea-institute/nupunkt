# API Reference

This document provides a detailed reference of the nupunkt API.

## Main Functions

### sent_tokenize

```python
def sent_tokenize(text: str) -> list[str]:
```

Tokenize text into sentences using the default pre-trained model.

**Parameters:**
- `text` (str): The text to tokenize

**Returns:**
- `list[str]`: A list of sentences

**Example:**
```python
from nupunkt import sent_tokenize

sentences = sent_tokenize("Hello world. This is a test.")
```

### load_default_model

```python
def load_default_model() -> PunktSentenceTokenizer:
```

Load the default pre-trained model.

**Returns:**
- `PunktSentenceTokenizer`: A tokenizer initialized with the default model

**Example:**
```python
from nupunkt.models import load_default_model

tokenizer = load_default_model()
```

## Core Classes

### PunktSentenceTokenizer

```python
class PunktSentenceTokenizer(PunktBase):
```

Sentence tokenizer using the Punkt algorithm.

**Constructor:**
```python
def __init__(
    self,
    train_text: Optional[Any] = None,
    verbose: bool = False,
    lang_vars: Optional[PunktLanguageVars] = None,
    token_cls: Type[PunktToken] = PunktToken,
    include_common_abbrevs: bool = True,
) -> None:
```

**Parameters:**
- `train_text` (Optional[Any]): Training text or pre-trained parameters
- `verbose` (bool): Whether to show verbose training information
- `lang_vars` (Optional[PunktLanguageVars]): Language-specific variables
- `token_cls` (Type[PunktToken]): The token class to use
- `include_common_abbrevs` (bool): Whether to include common abbreviations

**Methods:**

```python
def tokenize(self, text: str, realign_boundaries: bool = True) -> List[str]:
```
Tokenize text into sentences.

**Parameters:**
- `text` (str): The text to tokenize
- `realign_boundaries` (bool): Whether to realign sentence boundaries

**Returns:**
- `List[str]`: A list of sentences

```python
def span_tokenize(self, text: str, realign_boundaries: bool = True) -> Iterator[Tuple[int, int]]:
```
Tokenize text into sentence spans.

**Parameters:**
- `text` (str): The text to tokenize
- `realign_boundaries` (bool): Whether to realign sentence boundaries

**Returns:**
- `Iterator[Tuple[int, int]]`: An iterator of (start, end) character offsets

```python
def sentences_from_text(self, text: str, realign_boundaries: bool = True) -> List[str]:
```
Extract sentences from text.

**Parameters:**
- `text` (str): The text to tokenize
- `realign_boundaries` (bool): Whether to realign sentence boundaries

**Returns:**
- `List[str]`: A list of sentences

```python
def to_json(self) -> Dict[str, Any]:
```
Convert the tokenizer to a JSON-serializable dictionary.

**Returns:**
- `Dict[str, Any]`: A JSON-serializable dictionary

```python
@classmethod
def from_json(cls, data: Dict[str, Any], lang_vars: Optional[PunktLanguageVars] = None,
             token_cls: Optional[Type[PunktToken]] = None) -> "PunktSentenceTokenizer":
```
Create a PunktSentenceTokenizer from a JSON dictionary.

**Parameters:**
- `data` (Dict[str, Any]): The JSON dictionary
- `lang_vars` (Optional[PunktLanguageVars]): Optional language variables
- `token_cls` (Optional[Type[PunktToken]]): Optional token class

**Returns:**
- `PunktSentenceTokenizer`: A new PunktSentenceTokenizer instance

```python
def save(self, file_path: str) -> None:
```
Save the tokenizer to a JSON file.

**Parameters:**
- `file_path` (str): The path to save the file to

```python
@classmethod
def load(cls, file_path: str, lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Optional[Type[PunktToken]] = None) -> "PunktSentenceTokenizer":
```
Load a PunktSentenceTokenizer from a JSON file.

**Parameters:**
- `file_path` (str): The path to load the file from
- `lang_vars` (Optional[PunktLanguageVars]): Optional language variables
- `token_cls` (Optional[Type[PunktToken]]): Optional token class

**Returns:**
- `PunktSentenceTokenizer`: A new PunktSentenceTokenizer instance

```python
def reconfigure(self, config: Dict[str, Any]) -> None:
```
Reconfigure the tokenizer with new settings.

**Parameters:**
- `config` (Dict[str, Any]): A dictionary with configuration settings

### PunktTrainer

```python
class PunktTrainer(PunktBase):
```

Trainer for Punkt sentence boundary detection parameters.

**Constructor:**
```python
def __init__(
    self,
    train_text: Optional[str] = None,
    verbose: bool = False,
    lang_vars: Optional[PunktLanguageVars] = None,
    token_cls: Type[PunktToken] = PunktToken,
    include_common_abbrevs: bool = True,
) -> None:
```

**Parameters:**
- `train_text` (Optional[str]): Optional training text to immediately train on
- `verbose` (bool): Whether to show verbose training information
- `lang_vars` (Optional[PunktLanguageVars]): Language-specific variables
- `token_cls` (Type[PunktToken]): The token class to use
- `include_common_abbrevs` (bool): Whether to include common abbreviations by default

**Class Attributes:**

- `ABBREV` (float): Threshold for identifying abbreviations (default: 0.1)
- `ABBREV_BACKOFF` (int): Frequency threshold for rare abbreviations (default: 10)
- `COLLOCATION` (float): Threshold for identifying collocations (default: 5)
- `SENT_STARTER` (float): Threshold for identifying sentence starters (default: 25.0)
- `INCLUDE_ALL_COLLOCS` (bool): Whether to include all collocations (default: False)
- `INCLUDE_ABBREV_COLLOCS` (bool): Whether to include abbreviation collocations (default: False)
- `MIN_COLLOC_FREQ` (int): Minimum frequency for collocations (default: 5)
- `MAX_ABBREV_LENGTH` (int): Maximum length for abbreviation detection (default: 9)
- `ABBREV_CONSISTENCY` (float): Consistency threshold for abbreviations (default: 0.25)
- `PERSIST_ABBREVS` (bool): Whether to persist abbreviations between training runs (default: True)
- `COMMON_ABBREVS` (List[str]): Common abbreviations that should always be detected

**Methods:**

```python
def get_params(self) -> PunktParameters:
```
Get the trained parameters.

**Returns:**
- `PunktParameters`: The trained Punkt parameters

```python
def train(self, text: str, verbose: bool = False, finalize: bool = True, 
          preserve_abbrevs: bool = None) -> None:
```
Train the model on the given text.

**Parameters:**
- `text` (str): The training text
- `verbose` (bool): Whether to display progress information
- `finalize` (bool): Whether to finalize training after this run
- `preserve_abbrevs` (bool): Whether to preserve existing abbreviations

```python
def finalize_training(self, verbose: bool = False, preserve_common_abbrevs: bool = True) -> None:
```
Finalize the training by identifying sentence starters and collocations.

**Parameters:**
- `verbose` (bool): Whether to display progress information
- `preserve_common_abbrevs` (bool): Whether to preserve common abbreviations

```python
def to_json(self) -> Dict[str, Any]:
```
Convert trainer configuration and parameters to a JSON-serializable dictionary.

**Returns:**
- `Dict[str, Any]`: A JSON-serializable dictionary with trainer config and parameters

```python
@classmethod
def from_json(cls, data: Dict[str, Any], lang_vars: Optional[PunktLanguageVars] = None,
             token_cls: Optional[Type[PunktToken]] = None) -> "PunktTrainer":
```
Create a PunktTrainer instance from a JSON dictionary.

**Parameters:**
- `data` (Dict[str, Any]): The JSON dictionary
- `lang_vars` (Optional[PunktLanguageVars]): Optional language variables
- `token_cls` (Optional[Type[PunktToken]]): Optional token class

**Returns:**
- `PunktTrainer`: A new PunktTrainer instance

```python
def save(self, file_path: str) -> None:
```
Save trainer configuration and parameters to a JSON file.

**Parameters:**
- `file_path` (str): The path to save the file to

```python
@classmethod
def load(cls, file_path: str, lang_vars: Optional[PunktLanguageVars] = None,
        token_cls: Optional[Type[PunktToken]] = None) -> "PunktTrainer":
```
Load trainer configuration and parameters from a JSON file.

**Parameters:**
- `file_path` (str): The path to load the file from
- `lang_vars` (Optional[PunktLanguageVars]): Optional language variables
- `token_cls` (Optional[Type[PunktToken]]): Optional token class

**Returns:**
- `PunktTrainer`: A new PunktTrainer instance

### PunktParameters

```python
@dataclass
class PunktParameters:
```

Stores the parameters that Punkt uses for sentence boundary detection.

**Attributes:**
- `abbrev_types` (Set[str]): Set of known abbreviation types
- `collocations` (Set[Tuple[str, str]]): Set of word pairs that occur across sentence boundaries
- `sent_starters` (Set[str]): Set of words that start sentences
- `ortho_context` (Dict[str, int]): Dictionary tracking capitalization patterns

**Methods:**

```python
def add_ortho_context(self, typ: str, flag: int) -> None:
```
Add an orthographic context flag to a token type.

**Parameters:**
- `typ` (str): The token type
- `flag` (int): The orthographic context flag

```python
def to_json(self) -> Dict[str, Any]:
```
Convert parameters to a JSON-serializable dictionary.

**Returns:**
- `Dict[str, Any]`: A JSON-serializable dictionary

```python
@classmethod
def from_json(cls, data: Dict[str, Any]) -> "PunktParameters":
```
Create a PunktParameters instance from a JSON dictionary.

**Parameters:**
- `data` (Dict[str, Any]): The JSON dictionary

**Returns:**
- `PunktParameters`: A new PunktParameters instance

```python
def save(self, file_path: str) -> None:
```
Save parameters to a JSON file.

**Parameters:**
- `file_path` (str): The path to save the file to

```python
@classmethod
def load(cls, file_path: str) -> "PunktParameters":
```
Load parameters from a JSON file.

**Parameters:**
- `file_path` (str): The path to load the file from

**Returns:**
- `PunktParameters`: A new PunktParameters instance

### PunktLanguageVars

```python
class PunktLanguageVars:
```

Contains language-specific variables for Punkt sentence boundary detection.

**Attributes:**
- `sent_end_chars` (Tuple[str, ...]): Characters that can end sentences (default: ".", "?", "!")
- `internal_punctuation` (str): Characters considered internal punctuation (default: ",:;")
- `re_boundary_realignment` (re.Pattern): Pattern for realigning boundaries
- `_re_word_start` (str): Pattern for identifying word starts
- `_re_multi_char_punct` (str): Pattern for multi-character punctuation

**Methods:**

```python
def word_tokenize(self, text: str) -> list[str]:
```
Tokenize text into words using the word_tokenize_pattern.

**Parameters:**
- `text` (str): The text to tokenize

**Returns:**
- `list[str]`: A list of word tokens

### PunktToken

```python
@dataclass
class PunktToken:
```

Represents a token in the Punkt algorithm.

**Attributes:**
- `tok` (str): The token string
- `parastart` (bool): Whether the token starts a paragraph
- `linestart` (bool): Whether the token starts a line
- `sentbreak` (bool): Whether the token ends a sentence
- `abbr` (bool): Whether the token is an abbreviation
- `ellipsis` (bool): Whether the token is an ellipsis
- `period_final` (bool): Whether the token ends with a period
- `type` (str): The normalized token type
- `valid_abbrev_candidate` (bool): Whether the token could be a valid abbreviation

**Properties:**

```python
@property
def type_no_period(self) -> str:
```
Get the token type without a trailing period.

**Returns:**
- `str`: The token type without a period

```python
@property
def type_no_sentperiod(self) -> str:
```
Get the token type without a sentence-final period.

**Returns:**
- `str`: The token type without a sentence period

```python
@property
def first_upper(self) -> bool:
```
Check if the first character of the token is uppercase.

**Returns:**
- `bool`: True if the first character is uppercase

```python
@property
def first_lower(self) -> bool:
```
Check if the first character of the token is lowercase.

**Returns:**
- `bool`: True if the first character is lowercase

```python
@property
def first_case(self) -> str:
```
Get the case of the first character of the token.

**Returns:**
- `str`: "upper", "lower", or "none"

```python
@property
def is_ellipsis(self) -> bool:
```
Check if the token is an ellipsis.

**Returns:**
- `bool`: True if the token is an ellipsis

```python
@property
def is_number(self) -> bool:
```
Check if the token is a number.

**Returns:**
- `bool`: True if the token is a number

```python
@property
def is_initial(self) -> bool:
```
Check if the token is an initial (single letter followed by a period).

**Returns:**
- `bool`: True if the token is an initial

```python
@property
def is_alpha(self) -> bool:
```
Check if the token is alphabetic (contains only letters).

**Returns:**
- `bool`: True if the token is alphabetic

```python
@property
def is_non_punct(self) -> bool:
```
Check if the token contains non-punctuation characters.

**Returns:**
- `bool`: True if the token contains non-punctuation characters