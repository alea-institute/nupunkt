"""
Challenge set for testing sentence boundary detection improvements.

Each entry is a tuple of (text, expected_sentence_count, description).
These are cases where the standard Punkt algorithm may struggle.
"""

CHALLENGE_SET = [
    # Complex abbreviations
    ("Dr. Smith studied at M.I.T. in Cambridge.", 1, "Multiple abbreviations in one sentence"),
    (
        "The company's revenue was $5.2M. Their profit margin increased.",
        2,
        "Currency abbreviation at sentence end",
    ),
    ("She has a Ph.D. in Computer Science.", 1, "Academic degree abbreviation"),
    # Legal text challenges
    ("See 42 U.S.C. § 1983. The statute provides relief.", 2, "Legal code citation"),
    ("Per Section 3.2.1. employees must comply.", 1, "Section number that looks like sentence end"),
    ("The contract expires on Jan. 1, 2025.", 1, "Date abbreviation"),
    # Quotes and dialog
    ('She said "Hello." Then she left.', 2, "Period inside quotes"),
    ('"Is this correct?" he asked.', 1, "Question mark inside quotes"),
    ('He shouted "Stop!" The car halted.', 2, "Exclamation inside quotes"),
    # Lists and enumerations
    ("Requirements: 1. Valid ID. 2. Proof of residence.", 1, "Numbered list items"),
    ("Steps: a) Download the file. b) Extract contents.", 1, "Lettered list items"),
    ("Options: (i) Accept. (ii) Decline. (iii) Postpone.", 1, "Roman numeral list"),
    # Time and web
    ("The meeting is at 3:30 p.m. Tomorrow works better.", 2, "Time abbreviation"),
    ("Visit our website at www.example.com. Contact us for more info.", 2, "Website URL"),
    ("Email us at info@example.com. We'll respond quickly.", 2, "Email address"),
    # Edge cases with numbers
    ("The temperature was 98.6. This is normal.", 2, "Decimal number at sentence end"),
    ("Chapter 3. The Beginning.", 2, "Chapter number vs. sentence"),
    ("See Figure 3.2. for details.", 1, "Figure reference with decimal"),
    # Parentheses and special punctuation
    ("The results (see Table 1.) were significant.", 1, "Period inside parentheses"),
    ("The company (founded in 2020) grew rapidly. Sales doubled.", 2, "Parenthetical information"),
    (
        "Key features: speed, accuracy, and reliability. All are important.",
        2,
        "Colon followed by list",
    ),
    # Ellipsis handling
    ("She paused... Then continued speaking.", 2, "Ellipsis as sentence boundary"),
    ("The options were: stay, leave, or... wait.", 1, "Ellipsis within sentence"),
    ("He said he would... Actually, never mind.", 2, "Ellipsis with interruption"),
    # Mixed challenges
    ("Dr. Johnson et al. published in Nature.", 1, "Et al. abbreviation"),
    ("The study (Smith et al., 2023) found significant results.", 1, "Citation within parentheses"),
    ("Inc., LLC., and Corp. are common suffixes.", 1, "Multiple business abbreviations"),
]


def evaluate_tokenizer(tokenizer, verbose=False):
    """
    Evaluate a tokenizer on the challenge set.

    Args:
        tokenizer: A tokenizer with a tokenize() method
        verbose: Whether to print detailed results

    Returns:
        tuple: (correct_count, total_count, errors)
    """
    correct = 0
    total = len(CHALLENGE_SET)
    errors = []

    for text, expected_count, description in CHALLENGE_SET:
        sentences = tokenizer.tokenize(text)
        actual_count = len(sentences)

        if actual_count == expected_count:
            correct += 1
            if verbose:
                print(f"✓ {description}")
        else:
            errors.append(
                {
                    "text": text,
                    "expected": expected_count,
                    "actual": actual_count,
                    "description": description,
                    "sentences": sentences,
                }
            )
            if verbose:
                print(f"✗ {description}")
                print(f"  Expected: {expected_count}, Got: {actual_count}")
                print(f"  Sentences: {sentences}")

    accuracy = correct / total * 100

    if verbose:
        print(f"\nResults: {correct}/{total} ({accuracy:.1f}%)")

    return correct, total, errors


def print_detailed_errors(errors):
    """Print detailed error analysis."""
    print(f"\nDetailed Error Analysis ({len(errors)} errors):")
    print("=" * 80)

    for i, error in enumerate(errors, 1):
        print(f"\n{i}. {error['description']}")
        print(f"   Text: {error['text']}")
        print(f"   Expected {error['expected']} sentences, got {error['actual']}")
        print("   Tokenized as:")
        for j, sent in enumerate(error["sentences"], 1):
            print(f"     [{j}] {sent}")


if __name__ == "__main__":
    # Test with the default tokenizer
    from nupunkt import PunktSentenceTokenizer

    print("Evaluating default PunktSentenceTokenizer on challenge set...")
    print("=" * 80)

    tokenizer = PunktSentenceTokenizer()
    correct, total, errors = evaluate_tokenizer(tokenizer, verbose=True)

    if errors:
        print_detailed_errors(errors)
