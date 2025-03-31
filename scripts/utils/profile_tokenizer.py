#!/usr/bin/env python3
"""
Profiling script for nupunkt sentence tokenizer.

This script profiles the performance of the sentence tokenizer
to identify bottlenecks and optimization opportunities.
"""

import cProfile
import io
import os
import pstats
import sys
from pathlib import Path

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent.parent
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.models import load_default_model
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer

# Import test data loading function
sys.path.append(str(script_dir))
from test_default_model import load_test_data

def profile_tokenization():
    """Profile the sentence tokenization process."""
    print("Loading default model...")
    tokenizer = load_default_model()
    
    # Load test data
    test_path = root_dir / "data" / "test.jsonl.gz"
    if not test_path.exists():
        print(f"Error: Test data file not found: {test_path}")
        return

    test_texts = load_test_data(test_path)
    print(f"Loaded {len(test_texts)} test documents.")
    
    # Create a sample with first N characters to avoid extremely long profiles
    sample_size = min(100000, len(test_texts[0]))
    sample_text = test_texts[0][:sample_size]
    
    print(f"Profiling tokenization of a {sample_size} character sample...")
    
    # Function to profile
    def tokenize_sample():
        # Let's tokenize the sample multiple times to get more data
        for _ in range(5):
            sentences = tokenizer.tokenize(sample_text)
            # Force evaluation of generator
            sentence_count = len(sentences)
            
    # Function to profile specific methods with caching
    def profile_cached_methods():
        print("\nProfiling LRU-cached methods specifically...")
        
        # Create text samples that will trigger the cached methods
        sample_tokens = list(tokenizer._tokenize_words(sample_text))[:100]
        
        # For orthographic heuristic profiling
        token_samples = sample_tokens[:20]  # Get a few sample tokens
        
        # Profile orthographic heuristic
        profile_ortho = cProfile.Profile()
        profile_ortho.enable()
        for token in token_samples:
            for _ in range(100):  # Repeat multiple times
                tokenizer._ortho_heuristic(token)
        profile_ortho.disable()
        
        # Print orthographic heuristic stats
        s = io.StringIO()
        ps = pstats.Stats(profile_ortho, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print("\nOrthographic Heuristic Profile:")
        print(s.getvalue())
        
        # Get some token types for sentence starter lookup
        token_types = [token.type_no_sentperiod for token in token_samples]
        
        # Profile sentence starter lookup
        profile_sent_starter = cProfile.Profile()
        profile_sent_starter.enable()
        for typ in token_types:
            for _ in range(100):  # Repeat multiple times
                tokenizer._is_sent_starter(typ)
        profile_sent_starter.disable()
        
        # Print sentence starter lookup stats
        s = io.StringIO()
        ps = pstats.Stats(profile_sent_starter, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print("\nSentence Starter Lookup Profile:")
        print(s.getvalue())
        
        # Get some candidate abbreviations
        abbrev_candidates = ["mr", "dr", "inc", "ltd", "co", "corp", "prof", "jan", "feb", "mar"]
        
        # Profile abbreviation lookup
        profile_abbrev = cProfile.Profile()
        profile_abbrev.enable()
        for abbr in abbrev_candidates:
            for _ in range(50):  # Repeat multiple times
                # We need to access the base method directly from the tokenizer instance
                tokenizer._is_abbreviation(abbr)
        profile_abbrev.disable()
        
        # Print abbreviation lookup stats
        s = io.StringIO()
        ps = pstats.Stats(profile_abbrev, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print("\nAbbreviation Lookup Profile:")
        print(s.getvalue())
    
    # Run the main profiler
    profile = cProfile.Profile()
    profile.enable()
    tokenize_sample()
    profile.disable()
    
    # Print sorted stats
    s = io.StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 items
    print(s.getvalue())
    
    # Profile specific cached methods
    profile_cached_methods()
    
    # Save profile results to a file for later analysis
    output_dir = script_dir / "utils" / "profiles"
    os.makedirs(output_dir, exist_ok=True)
    profile_path = output_dir / "tokenizer_profile.prof"
    ps.dump_stats(profile_path)
    print(f"Saved profile results to {profile_path}")
    
    # Also print stats sorted by time
    s = io.StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('time')
    ps.print_stats(30)  # Print top 30 items
    print("\nStats sorted by time:")
    print(s.getvalue())
    
    return profile_path

def main():
    """Run the profiling."""
    profile_path = profile_tokenization()
    
    # Print instructions for viewing the profile with snakeviz
    if profile_path:
        print("\nTo visualize the profile with snakeviz, run:")
        print(f"  pip install snakeviz")
        print(f"  snakeviz {profile_path}")

if __name__ == "__main__":
    main()