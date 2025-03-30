#!/usr/bin/env python3
"""
Script to train the default model for nupunkt.

This script:
1. Trains a general-purpose English sentence tokenizer using local data files
2. Loads abbreviations from data/legal_abbreviations.json
3. Saves the model to the models directory as the package's default model

Requires data/train.jsonl.gz, data/train2.jsonl.gz, and data/legal_abbreviations.json to be present.
"""

import sys
import os
from pathlib import Path
import gzip
import json

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.trainers.base_trainer import PunktTrainer
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

def load_abbreviations(file_path: Path) -> list:
    """Load abbreviations from a JSON file."""
    print(f"Loading abbreviations from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        abbreviations = json.load(f)
    
    # Convert abbreviations to lowercase and remove trailing periods
    cleaned_abbrevs = []
    for abbr in abbreviations:
        # Remove trailing period if present
        if abbr.endswith('.'):
            abbr = abbr[:-1]
        # Convert to lowercase
        cleaned_abbrevs.append(abbr.lower())
    
    print(f"Loaded {len(cleaned_abbrevs)} abbreviations")
    return cleaned_abbrevs

def load_jsonl_text(file_path: Path, max_samples: int = None) -> str:
    """Load text from a compressed JSONL file."""
    print(f"Loading text from {file_path}...")
    combined_text = ""
    sample_count = 0
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "text" in data:
                    combined_text += data["text"] + "\n\n"
                    sample_count += 1
                    if max_samples and sample_count >= max_samples:
                        break
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {file_path}")
                continue
    
    print(f"Loaded {sample_count} text samples, total size: {len(combined_text)} characters")
    return combined_text

def print_training_details(trainer: PunktTrainer, initial_abbrevs: set) -> None:
    """Print detailed information about the trained model."""
    print("\n=== Training Details ===")
    
    # Print hyperparameters
    print("\nHyperparameters:")
    print(f"  - Abbreviation threshold:      {trainer.ABBREV}")
    print(f"  - Abbreviation backoff:        {trainer.ABBREV_BACKOFF}")
    print(f"  - Collocation threshold:       {trainer.COLLOCATION}")
    print(f"  - Sentence starter threshold:  {trainer.SENT_STARTER}")
    print(f"  - Minimum collocation freq:    {trainer.MIN_COLLOC_FREQ}")
    print(f"  - Maximum abbreviation length: {trainer.MAX_ABBREV_LENGTH}")
    
    # Print abbreviation stats
    params = trainer.get_params()
    loaded_abbrev_count = len(initial_abbrevs)
    current_abbrevs = set(params.abbrev_types)
    learned_abbrevs = current_abbrevs - initial_abbrevs
    learned_abbrev_count = len(learned_abbrevs)
    
    # Print abbreviation stats
    print(f"\nAbbreviations Stats:")
    print(f"  - Total abbreviations:         {len(params.abbrev_types)}")
    print(f"  - Loaded from file:            {loaded_abbrev_count}")
    print(f"  - Learned during training:     {learned_abbrev_count}")
    
    # Print some learned abbreviations (if any)
    if learned_abbrevs:
        print("\nSample of learned abbreviations (up to 20):")
        for abbr in sorted(list(learned_abbrevs))[:20]:
            print(f"  - {abbr}")
    
    # Print collocations stats
    print(f"\nCollocations Stats:")
    print(f"  - Total collocations:          {len(params.collocations)}")
    
    # Print some sample collocations
    if params.collocations:
        print("\nSample of collocations (up to 20):")
        sorted_collocs = sorted(params.collocations)
        for w1, w2 in sorted_collocs[:20]:
            print(f"  - {w1} {w2}")
    
    # Print sentence starters stats
    print(f"\nSentence Starters Stats:")
    print(f"  - Total sentence starters:     {len(params.sent_starters)}")
    
    # Print some sample sentence starters
    if params.sent_starters:
        print("\nSample of sentence starters (up to 20):")
        sorted_starters = sorted(params.sent_starters)
        for starter in sorted_starters[:20]:
            print(f"  - {starter}")
    
    print("\n" + "="*60)

def main():
    """Train a default model and save it to the package models directory."""
    # Set paths
    data_dir = root_dir / "data"
    models_dir = root_dir / "nupunkt" / "models"
    model_path = models_dir / "default_model.json.xz"
    
    # Make sure the models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if we have required files
    train_path = data_dir / "train.jsonl.gz"
    train2_path = data_dir / "train2.jsonl.gz"
    abbrev_path = data_dir / "legal_abbreviations.json"
    
    # Throw exception if required files are not present
    if not train_path.exists():
        raise FileNotFoundError(f"Required training file not found: {train_path}")
    
    if not train2_path.exists():
        raise FileNotFoundError(f"Required training file not found: {train2_path}")
    
    if not abbrev_path.exists():
        raise FileNotFoundError(f"Required abbreviations file not found: {abbrev_path}")
    
    # Load legal abbreviations
    abbreviations = load_abbreviations(abbrev_path)
    
    # Load training data
    train_text1 = load_jsonl_text(train_path, max_samples=5000)
    train_text2 = load_jsonl_text(train2_path, max_samples=5000)
    
    combined_text = train_text1 + "\n\n" + train_text2
    
    # Train model
    print("\n=== Training Default Model ===")
    trainer = PunktTrainer(verbose=True)
    
    # Add abbreviations from legal_abbreviations.json file only
    print(f"\nAdding {len(abbreviations)} abbreviations from {abbrev_path}")
    initial_abbrevs = set()
    for abbr in abbreviations:
        trainer._params.abbrev_types.add(abbr.lower())
        initial_abbrevs.add(abbr.lower())
    
    # Train on the combined text
    trainer.train(combined_text)
    
    # Print detailed statistics about the trained model
    print_training_details(trainer, initial_abbrevs)
    
    # Save model
    print(f"\n=== Saving Model to {model_path} ===")
    trainer.save(model_path)
    print(f"Model saved successfully to {model_path}")
    
    # Test the model
    print("\n=== Testing Default Model ===")
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    
    test_cases = [
        "Dr. Smith went to Washington, D.C. He was very excited about the trip.",
        "The company (Ltd.) was founded in 1997. It has grown significantly since then.",
        "This text contains an ellipsis... And this is a new sentence.",
        "Let me give you an example, e.g. this one. Did you understand it?",
        "The meeting is at 3 p.m. Don't be late!",
        "Under 18 U.S.C. 12, this is a legal citation. The next sentence begins here."
    ]
    
    print("\nTokenizing sample texts:")
    for test in test_cases:
        print("\nText: ", test)
        sentences = tokenizer.tokenize(test)
        for i, sentence in enumerate(sentences, 1):
            print(f"  Sentence {i}: {sentence.strip()}")
    
    print("\nDefault model created successfully!")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()