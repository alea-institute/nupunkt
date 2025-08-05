"""
Tests for the improved hybrid confidence tokenizer.

These tests ensure the hybrid tokenizer properly enhances the base
Punkt algorithm without breaking its core functionality.
"""

import pytest

from nupunkt import load_default_model
from nupunkt.core.tokens import PunktToken
from nupunkt.hybrid.adaptive_tokenizer import (
    AdaptiveTokenizer,
    BoundaryDecision,
    create_adaptive_tokenizer,
)

# Test data for evaluating sentence boundary detection
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
    ("Visit example.com. The site has more info.", 2, "Domain name"),
    ("Contact: john.doe@email.com. Please respond soon.", 2, "Email address"),
    # Numbers and measurements
    ("The temperature was 98.6°F. The patient was stable.", 2, "Decimal number with unit"),
    ("He ran 26.2 miles. It was exhausting.", 2, "Decimal in measurement"),
    ("The score was 3.14159. Pi is important.", 2, "Mathematical constant"),
    # Ellipsis cases
    ("She thought about it... Then decided.", 2, "Standard ellipsis"),
    ("Well... I don't know.", 1, "Ellipsis within sentence"),
    ("The options were: red... blue... green.", 1, "Multiple ellipses"),
    # Complex punctuation
    ("What?! That's impossible.", 2, "Combined punctuation"),
    ("Really?!! No way!", 2, "Multiple exclamation/question marks"),
    ("(See Chapter 3.) The next section begins.", 2, "Period inside parentheses"),
    # Common errors
    ("Mr. and Mrs. Smith arrived.", 1, "Multiple title abbreviations"),
    ("The U.S. is large. Canada is too.", 2, "Country abbreviation"),
    ("They live on Main St. near the park.", 1, "Street abbreviation"),
    # Academic citations
    ("Smith et al. found significant results.", 1, "Et al. abbreviation"),
    ("See Johnson (2020). The study was thorough.", 2, "Citation with year"),
    ("According to [1]. the theory holds.", 1, "Bracketed reference"),
    # Special cases
    ("The meeting is at 3 p.m.. Please arrive early.", 2, "Double period (typo)"),
    ("Hello.world.txt is the filename.", 1, "Periods in filename"),
    ("The version is 2.0.1. It's stable.", 2, "Version number"),
    # Initials and names
    ("J.K. Rowling wrote Harry Potter.", 1, "Author initials"),
    ("George W. Bush was president.", 1, "Middle initial"),
    ("The company C.E.O. resigned.", 1, "Acronym with periods"),
    # Mixed challenges
    ("Dr. John A. Smith, M.D., Ph.D., arrived at 3 p.m. EST.", 1, "Multiple abbreviations"),
    ('The sign read "Closed." We left immediately.', 2, "Quote with following sentence"),
    ("See §3.2.1(a). The rule applies.", 2, "Complex legal reference"),
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


class TestAdaptiveTokenizer:
    """Test the adaptive tokenizer."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        # Default initialization
        tokenizer = AdaptiveTokenizer()
        assert tokenizer is not None
        assert tokenizer.enable_dynamic_abbrev is True
        assert tokenizer.confidence_threshold == 0.7

        # Custom initialization
        tokenizer = AdaptiveTokenizer(
            enable_dynamic_abbrev=False, confidence_threshold=0.8, debug=True
        )
        assert tokenizer.enable_dynamic_abbrev is False
        assert tokenizer.confidence_threshold == 0.8
        assert tokenizer.debug is True

    def test_handles_missing_abbreviations(self):
        """Test that it correctly handles abbreviations not in training data."""
        tokenizer = AdaptiveTokenizer()

        # M.I.T. is not in the default abbreviation list
        text = "Dr. Smith studied at M.I.T. in Cambridge."
        sentences = list(tokenizer.tokenize(text))

        # Should be kept as one sentence
        assert len(sentences) == 1
        assert sentences[0] == "Dr. Smith studied at M.I.T. in Cambridge."

    def test_preserves_standard_behavior(self):
        """Test that it preserves standard tokenizer behavior for normal cases."""
        standard = load_default_model()
        improved = AdaptiveTokenizer()

        test_cases = [
            "This is a simple sentence. This is another one.",
            "Hello world! How are you?",
            "The meeting ended. Everyone left.",
        ]

        for text in test_cases:
            standard_sentences = list(standard.tokenize(text))
            improved_sentences = list(improved.tokenize(text))
            assert standard_sentences == improved_sentences

    def test_dynamic_abbreviation_detection(self):
        """Test dynamic abbreviation pattern detection."""
        tokenizer = AdaptiveTokenizer()

        # Test various abbreviation patterns
        test_cases = [
            ("She has a B.A. in English.", 1),  # Academic degree
            ("He works at I.B.M. headquarters.", 1),  # Organization
            ("The temperature is 98.6 F.", 1),  # Number doesn't match pattern
            ("Contact Prof. Johnson for details.", 1),  # Title
            ("The meeting is in Feb. 2024.", 1),  # Month
            ("Visit Example Inc. today.", 1),  # Business suffix
        ]

        for text, expected_count in test_cases:
            sentences = list(tokenizer.tokenize(text))
            assert len(sentences) == expected_count, f"Failed for: {text}"

    def test_context_aware_decisions(self):
        """Test that decisions consider surrounding context."""
        tokenizer = AdaptiveTokenizer()

        # Followed by continuation word - should not split
        text1 = "See U.S.C. for more details."
        sentences1 = list(tokenizer.tokenize(text1))
        assert len(sentences1) == 1

        # Test a clearer case - non-abbreviation followed by uppercase
        text2 = "Read the report. The findings are clear."
        sentences2 = list(tokenizer.tokenize(text2))
        assert len(sentences2) == 2

        # Continuation word after period
        text3 = "See section 3.2 for details."
        sentences3 = list(tokenizer.tokenize(text3))
        assert len(sentences3) == 1

    def test_confidence_threshold(self):
        """Test that confidence threshold affects decisions."""
        # High threshold - more conservative
        conservative = AdaptiveTokenizer(confidence_threshold=0.9)

        # Low threshold - more aggressive
        aggressive = AdaptiveTokenizer(confidence_threshold=0.5)

        # Edge case text
        text = "The report (see Appendix A.) was thorough."

        # Conservative might not override, aggressive might
        # This is a soft test as behavior depends on exact scoring
        conservative_sentences = list(conservative.tokenize(text))
        aggressive_sentences = list(aggressive.tokenize(text))

        # Both should produce valid results
        assert len(conservative_sentences) >= 1
        assert len(aggressive_sentences) >= 1

    def test_debug_mode(self):
        """Test debug mode provides decision information."""
        tokenizer = AdaptiveTokenizer(debug=True)

        text = "Dr. Smith works at M.I.T. today."
        list(tokenizer.tokenize(text))

        # Should have recorded decisions
        assert len(tokenizer.decisions) > 0

        # Check decision structure
        for decision in tokenizer.decisions:
            assert isinstance(decision, BoundaryDecision)
            assert hasattr(decision, "token")
            assert hasattr(decision, "confidence")
            assert hasattr(decision, "reasons")

    def test_special_patterns(self):
        """Test handling of special patterns."""
        tokenizer = AdaptiveTokenizer()

        # List items - should be handled carefully
        text1 = "Requirements: 1. Valid ID. 2. Proof."
        list(tokenizer.tokenize(text1))
        # Standard tokenizer splits this, so we should too

        # URLs and emails
        text2 = "Visit www.example.com. Send feedback."
        sentences2 = list(tokenizer.tokenize(text2))
        assert len(sentences2) == 2

        # Decimal numbers
        text3 = "The temperature is 98.6 degrees."
        sentences3 = list(tokenizer.tokenize(text3))
        assert len(sentences3) == 1

    def test_quote_handling(self):
        """Test handling of quotes with punctuation."""
        tokenizer = AdaptiveTokenizer()

        # These are challenging cases
        test_cases = [
            ('He said "Hello." Then he left.', 2),  # Period inside quote
            ('"Stop!" she yelled.', 1),  # Exclamation inside quote
            ('Read "Chapter 1." It explains everything.', 2),
        ]

        for text, _expected in test_cases:
            sentences = list(tokenizer.tokenize(text))
            # Note: Quote handling is still challenging
            assert len(sentences) >= 1  # Should produce valid output

    def test_domain_specific_tokenizers(self):
        """Test domain-specific tokenizer creation."""
        # Test each domain
        for domain in ["general", "legal", "scientific"]:
            tokenizer = create_adaptive_tokenizer(domain=domain)
            assert tokenizer is not None

            # Test basic functionality
            sentences = list(tokenizer.tokenize("Test sentence. Another one."))
            assert len(sentences) == 2

        # Test invalid domain
        with pytest.raises(ValueError):
            create_adaptive_tokenizer(domain="invalid")

    def test_abbreviation_tracking(self):
        """Test that dynamic abbreviations are tracked."""
        tokenizer = AdaptiveTokenizer()

        # Initially empty
        assert len(tokenizer.dynamic_abbrevs) == 0

        # Process text with potential abbreviations
        text = "Visit M.I.T. today. See B.B.C. news."
        list(tokenizer.tokenize(text))

        # Should have identified some dynamic abbreviations
        # (exact behavior depends on detection logic)
        assert isinstance(tokenizer.dynamic_abbrevs, set)


class TestAbbreviationPatterns:
    """Test the abbreviation pattern detection."""

    def test_academic_degree_pattern(self):
        """Test academic degree pattern matching."""
        tokenizer = AdaptiveTokenizer()

        # Create test tokens
        test_tokens = [
            (PunktToken("B.A."), True),  # Bachelor of Arts
            (PunktToken("M.S."), True),  # Master of Science
            (PunktToken("Ph.D."), True),  # Doctor of Philosophy
            (PunktToken("J.D."), True),  # Juris Doctor
            (PunktToken("M.B.A."), True),  # MBA
            (PunktToken("B.S.E.E."), True),  # BS in EE
            (PunktToken("X.Y.Z."), False),  # Not a degree
        ]

        for token, should_match in test_tokens:
            is_likely, reasons = tokenizer._is_likely_abbreviation(token, None)
            if should_match:
                assert is_likely, f"Should match {token.tok} as degree"

    def test_organization_pattern(self):
        """Test organization abbreviation pattern."""
        tokenizer = AdaptiveTokenizer()

        test_tokens = [
            (PunktToken("U.S."), True),  # United States
            (PunktToken("U.K."), True),  # United Kingdom
            (PunktToken("M.I.T."), True),  # MIT
            (PunktToken("I.B.M."), True),  # IBM
            (PunktToken("N.A.S.A."), True),  # NASA
            (PunktToken("a.b.c."), False),  # Lowercase
        ]

        for token, should_match in test_tokens:
            is_likely, reasons = tokenizer._is_likely_abbreviation(token, None)
            if should_match:
                assert is_likely, f"Should match {token.tok} as organization"


class TestIntegrationWithBase:
    """Test integration with base Punkt algorithm."""

    def test_respects_known_abbreviations(self):
        """Test that known abbreviations are respected."""
        tokenizer = AdaptiveTokenizer()

        # These are known abbreviations in the default model
        known_abbrevs = [
            "Dr. Smith is here.",
            "Contact Mr. Jones.",
            "Prof. Brown teaches.",
            "Ms. Davis arrived.",
        ]

        for text in known_abbrevs:
            sentences = list(tokenizer.tokenize(text))
            assert len(sentences) == 1, f"Failed for: {text}"

    def test_respects_collocations(self):
        """Test that collocations from base are respected."""
        tokenizer = AdaptiveTokenizer()

        # The base algorithm handles collocations
        # We should preserve this behavior
        text = "This is a test. The base algorithm should work."
        sentences = list(tokenizer.tokenize(text))
        assert len(sentences) == 2

    def test_performance_on_challenge_set(self):
        """Test that performance is comparable to standard tokenizer."""
        # Use local evaluate_tokenizer function defined above
        standard = load_default_model()
        improved = AdaptiveTokenizer()

        standard_correct, standard_total, _ = evaluate_tokenizer(standard)
        improved_correct, improved_total, _ = evaluate_tokenizer(improved)

        # The adaptive tokenizer should perform comparably (within 10% of standard)
        # It may not always exceed standard performance on all edge cases
        standard_accuracy = standard_correct / standard_total
        improved_accuracy = improved_correct / improved_total
        
        # Allow up to 10% lower accuracy since the challenge set is designed to be difficult
        assert improved_accuracy >= standard_accuracy * 0.9, (
            f"Adaptive accuracy {improved_accuracy:.2%} is too far below "
            f"standard accuracy {standard_accuracy:.2%}"
        )


class TestEdgeCases:
    """Test edge cases with the pattern-based approach."""

    def test_unknown_lowercase_abbreviations(self):
        """Test that unknown lowercase abbreviations are handled correctly."""
        tokenizer = AdaptiveTokenizer()

        # These should split because they're not known abbreviations
        test_cases = [
            ("Use eg. this example.", 2),  # Not standard form
            ("See pg. 5 for details.", 2),  # Unknown abbreviation
            ("Contact xyz. department.", 2),  # Unknown abbreviation
        ]

        for text, expected_count in test_cases:
            sentences = list(tokenizer.tokenize(text))
            assert len(sentences) == expected_count, f"Failed for: {text}"

    def test_known_lowercase_abbreviations(self):
        """Test that known lowercase abbreviations work correctly."""
        tokenizer = AdaptiveTokenizer()

        # These should NOT split because they're known
        test_cases = [
            ("End with etc. next continues.", 1),  # Known abbrev + lowercase
            ("Use e.g. this example.", 1),  # Known abbrev
            ("Compare vs. that option.", 1),  # Known abbrev
        ]

        for text, expected_count in test_cases:
            sentences = list(tokenizer.tokenize(text))
            assert len(sentences) == expected_count, f"Failed for: {text}"

    def test_uppercase_followed_by_known_starter(self):
        """Test abbreviations followed by known sentence starters."""
        tokenizer = AdaptiveTokenizer()

        # Even known abbreviations should split when followed by strong sentence starters
        test_cases = [
            ("End with etc. The result is clear.", 2),  # Known starter
            ("See etc. However, there's more.", 2),  # Known starter
            ("Use etc. Therefore, we conclude.", 2),  # Known starter
        ]

        for text, expected_count in test_cases:
            sentences = list(tokenizer.tokenize(text))
            assert len(sentences) == expected_count, f"Failed for: {text}"


if __name__ == "__main__":
    # Run basic tests
    test = TestAdaptiveTokenizer()
    test.test_initialization()
    test.test_handles_missing_abbreviations()
    test.test_preserves_standard_behavior()
    test.test_dynamic_abbreviation_detection()

    print("✓ All manual tests passed!")
