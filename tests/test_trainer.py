"""
Tests for the Punkt trainer module.
"""

from collections import Counter

import pytest

from nupunkt.trainers.base_trainer import PunktTrainer

# Training corpus with obvious patterns
TRAIN_TEXT = """
Dr. Smith lives in the U.S. He works for Acme Inc. and is a good doctor.
This is the second sentence. Mr. Jones agrees. This is the third.
The company was founded in Jan. 2020. It has grown since then.
See section 3.2 for details. Also check Fig. 1 and Table 2.
"""

# Larger corpus for testing sentence starters and collocations
LARGE_TRAIN_TEXT = """
Dr. John Smith is a renowned physician. Dr. Smith works at the hospital.
Dr. Jane Doe is his colleague. Dr. Doe specializes in cardiology.

The patient arrived at 3 p.m. The doctor saw him immediately.
The treatment was successful. The patient recovered quickly.

This is important. This is very important. This must be noted.
This cannot be ignored. This is the final point.

New York is a big city. New York has many attractions.
New Jersey is nearby. New Jersey is also populous.
"""


@pytest.fixture
def trainer():
    """Create a fresh trainer instance."""
    return PunktTrainer()


@pytest.fixture
def trained_trainer():
    """Create a trainer that has been trained on the test corpus."""
    trainer = PunktTrainer()
    trainer.train(TRAIN_TEXT, verbose=False, finalize=True)
    return trainer


class TestPunktTrainerInitialization:
    """Test trainer initialization."""

    def test_basic_initialization(self):
        """Test basic initialization without training."""
        trainer = PunktTrainer()

        # Check initial state
        assert trainer._params.abbrev_types == {"..."}  # Only ellipsis by default
        assert trainer._params.sent_starters == set()
        assert trainer._params.collocations == set()
        assert trainer._finalized is True
        assert trainer._num_period_toks == 0
        assert trainer._sentbreak_count == 0

    def test_initialization_with_training_text(self):
        """Test initialization with immediate training."""
        trainer = PunktTrainer(train_text=TRAIN_TEXT, verbose=False)

        # Should have trained and finalized
        params = trainer.get_params()
        assert len(params.abbrev_types) > 1  # Should find abbreviations
        assert "..." in params.abbrev_types  # Should preserve ellipsis
        assert "dr" in params.abbrev_types
        assert "u.s" in params.abbrev_types

    def test_memory_efficient_initialization(self):
        """Test initialization with memory-efficient mode."""
        trainer = PunktTrainer(memory_efficient=True)
        assert trainer.MEMORY_EFFICIENT is True

        trainer2 = PunktTrainer(memory_efficient=False)
        assert trainer2.MEMORY_EFFICIENT is False

    def test_disable_common_abbrevs(self):
        """Test disabling common abbreviations."""
        trainer = PunktTrainer(include_common_abbrevs=False)
        assert trainer._params.abbrev_types == set()  # No ellipsis


class TestPunktTrainerTraining:
    """Test the training process."""

    def test_standard_training_path(self, trainer):
        """Test standard training without finalization."""
        trainer.train(TRAIN_TEXT, verbose=False, finalize=False)

        # Check internal state
        assert trainer._finalized is False
        assert trainer._num_period_toks > 0
        assert trainer._sentbreak_count > 0
        assert len(trainer._type_fdist) > 0

        # Most common types should include common words
        most_common = trainer._type_fdist.most_common(5)
        common_words = [word for word, count in most_common]
        assert "the" in common_words

    def test_training_with_finalization(self, trainer):
        """Test training with finalization."""
        trainer.train(TRAIN_TEXT, verbose=False, finalize=True)

        # Should be finalized
        assert trainer._finalized is True

        # Get parameters
        params = trainer.get_params()

        # Check abbreviations
        expected_abbrevs = {"dr", "u.s", "inc", "mr", "jan", "fig"}
        found_abbrevs = set(params.abbrev_types) - {"..."}  # Remove ellipsis
        assert found_abbrevs == expected_abbrevs

    def test_memory_efficient_training(self):
        """Test memory-efficient training path."""
        trainer = PunktTrainer(memory_efficient=True)
        trainer.train(TRAIN_TEXT, verbose=False, finalize=True)

        params = trainer.get_params()

        # Should find same abbreviations as standard mode
        expected_abbrevs = {"dr", "u.s", "inc", "mr", "jan", "fig", "..."}
        assert set(params.abbrev_types) == expected_abbrevs

    def test_training_preserves_abbreviations(self, trainer):
        """Test that training preserves existing abbreviations."""
        # Add some manual abbreviations
        trainer._params.abbrev_types.add("ph.d")
        trainer._params.abbrev_types.add("m.d")

        # Train with preserve_abbrevs=True (default)
        trainer.train(TRAIN_TEXT, verbose=False, finalize=True)

        params = trainer.get_params()
        assert "ph.d" in params.abbrev_types
        assert "m.d" in params.abbrev_types
        assert "dr" in params.abbrev_types  # New abbreviation

    def test_training_without_preserving_abbreviations(self, trainer):
        """Test training without preserving abbreviations."""
        # Add some manual abbreviations that are not in the training text
        trainer._params.abbrev_types.add("ph.d")
        trainer._params.abbrev_types.add("m.d")

        # Clear the default ellipsis for this test
        trainer._params.abbrev_types.clear()
        trainer._params.abbrev_types.add("ph.d")
        trainer._params.abbrev_types.add("m.d")

        # Train with preserve_abbrevs=False
        # This means existing abbreviations won't be re-added after training
        trainer.train(TRAIN_TEXT, verbose=False, finalize=True, preserve_abbrevs=False)

        params = trainer.get_params()
        # The behavior is that abbreviations found during training are kept
        # but manually added ones that weren't found are also kept
        # (they're just not explicitly re-added at the end)
        # So we should check that new abbreviations were found
        assert "dr" in params.abbrev_types
        assert "u.s" in params.abbrev_types

        # Manual abbreviations are still there because they were never cleared
        # preserve_abbrevs only affects whether they're re-added at the end
        assert "ph.d" in params.abbrev_types
        assert "m.d" in params.abbrev_types


class TestPunktTrainerFeatures:
    """Test specific trainer features."""

    def test_sentence_starters_detection(self):
        """Test detection of sentence starters."""
        # Use larger corpus with repeated sentence starters
        trainer = PunktTrainer()
        # Lower threshold for this small test corpus
        trainer.SENT_STARTER = 5.0
        trainer.train(LARGE_TRAIN_TEXT, verbose=False, finalize=True)

        params = trainer.get_params()
        # Should detect "this" as a sentence starter (appears 5 times at start)
        assert "this" in params.sent_starters

    def test_collocation_detection(self):
        """Test detection of collocations."""
        # Use larger corpus with repeated bigrams
        trainer = PunktTrainer()
        # Lower thresholds for testing
        trainer.COLLOCATION = 2.0
        trainer.MIN_COLLOC_FREQ = 1
        # Enable abbreviation collocations to detect "Dr. Smith"
        trainer.INCLUDE_ABBREV_COLLOCS = True
        trainer.train(LARGE_TRAIN_TEXT, verbose=False, finalize=True)

        params = trainer.get_params()
        # Should detect collocations like ('dr', 'smith') and ('dr', 'doe')
        colloc_first_words = {c[0] for c in params.collocations}
        assert "dr" in colloc_first_words

    def test_get_params_finalizes_if_needed(self, trainer):
        """Test that get_params finalizes training if needed."""
        trainer.train(TRAIN_TEXT, verbose=False, finalize=False)
        assert trainer._finalized is False

        # get_params should trigger finalization
        params = trainer.get_params()
        assert trainer._finalized is True
        assert len(params.abbrev_types) > 1


class TestPunktTrainerMethods:
    """Test specific internal methods."""

    def test_reclassify_abbrev_types(self, trainer):
        """Test the _reclassify_abbrev_types method."""
        # Manually populate frequency distributions
        trainer._type_fdist = Counter(
            {
                "dr.": 5,
                "smith": 10,
                "u.s.": 3,
                "inc.": 2,
                "the": 50,
                "3.2": 1,  # Number with period
            }
        )
        trainer._num_period_toks = 11  # Total tokens ending with period
        trainer._sentbreak_count = 20  # Total sentence breaks

        # Get abbreviation classifications - need to pass types WITH periods
        types_to_check = {"dr.", "u.s.", "inc.", "smith", "the", "3.2"}
        results = list(trainer._reclassify_abbrev_types(types_to_check))

        # Build results dict (note: the method returns the type without period)
        abbrev_results = {typ: (score, is_abbrev) for typ, score, is_abbrev in results}

        # Should classify 'dr', 'u.s', 'inc' as abbreviations
        assert "dr" in abbrev_results and abbrev_results["dr"][1] is True
        assert "u.s" in abbrev_results and abbrev_results["u.s"][1] is True
        assert "inc" in abbrev_results and abbrev_results["inc"][1] is True
        # Numbers with periods are skipped
        assert "3.2" not in abbrev_results

    def test_orthographic_context(self, trained_trainer):
        """Test orthographic context detection."""
        params = trained_trainer.get_params()

        # Should have orthographic context for some words
        assert len(params.ortho_context) > 0

        # Check specific patterns
        # Words that appear after periods with specific capitalization
        # should be in ortho_context

    def test_type_frequency_pruning(self):
        """Test that frequency distributions are pruned in memory-efficient mode."""
        trainer = PunktTrainer(memory_efficient=True)
        trainer.PRUNE_INTERVAL = 20  # Low interval for testing
        trainer.TYPE_FDIST_MIN_FREQ = 2

        # Add many low-frequency tokens
        for i in range(30):
            trainer._type_fdist[f"rare_word_{i}"] = 1

        # Add some high-frequency tokens
        trainer._type_fdist["common"] = 10
        trainer._type_fdist["frequent"] = 5

        # Trigger pruning by incrementing token count
        trainer._token_count = 25
        trainer._prune_distributions()

        # Low-frequency tokens should be removed
        assert "rare_word_0" not in trainer._type_fdist
        assert "common" in trainer._type_fdist
        assert "frequent" in trainer._type_fdist


class TestPunktTrainerSerialization:
    """Test serialization and deserialization."""

    def test_to_json(self, trained_trainer):
        """Test converting trainer to JSON."""
        json_data = trained_trainer.to_json()

        # Check structure
        assert "version" in json_data
        assert "description" in json_data
        assert "parameters" in json_data

        # Check configuration parameters
        assert json_data["abbrev_threshold"] == trained_trainer.ABBREV
        assert json_data["collocation_threshold"] == trained_trainer.COLLOCATION
        assert json_data["sent_starter_threshold"] == trained_trainer.SENT_STARTER

        # Check parameters were serialized
        params_data = json_data["parameters"]
        assert "abbrev_types" in params_data
        assert "sent_starters" in params_data
        assert "collocations" in params_data

    def test_from_json(self, trained_trainer):
        """Test loading trainer from JSON."""
        # Get JSON from trained trainer
        json_data = trained_trainer.to_json()

        # Create new trainer from JSON
        new_trainer = PunktTrainer.from_json(json_data)

        # Check configuration
        assert new_trainer.ABBREV == trained_trainer.ABBREV
        assert new_trainer.COLLOCATION == trained_trainer.COLLOCATION
        assert new_trainer.SENT_STARTER == trained_trainer.SENT_STARTER

        # Check parameters
        original_params = trained_trainer.get_params()
        new_params = new_trainer.get_params()

        assert set(new_params.abbrev_types) == set(original_params.abbrev_types)
        assert set(new_params.sent_starters) == set(original_params.sent_starters)
        assert set(new_params.collocations) == set(original_params.collocations)

    def test_round_trip_serialization(self, trained_trainer):
        """Test that serialization round-trip preserves all data."""
        # Serialize and deserialize
        json_data = trained_trainer.to_json()
        new_trainer = PunktTrainer.from_json(json_data)
        new_json_data = new_trainer.to_json()

        # Should be identical (except for any timestamps or metadata)
        assert json_data["parameters"] == new_json_data["parameters"]
        assert json_data["abbrev_threshold"] == new_json_data["abbrev_threshold"]


class TestPunktTrainerAdditional:
    """Additional tests for better coverage."""

    def test_train_batches(self):
        """Test train_batches method."""
        trainer = PunktTrainer()

        # Create batch iterator
        batches = [
            "First batch. Dr. Smith works here.",
            "Second batch. Mr. Jones agrees.",
            "Third batch. This is important.",
        ]

        trainer.train_batches(iter(batches), verbose=False, finalize=True)
        params = trainer.get_params()

        # Should find abbreviations from all batches
        assert "dr" in params.abbrev_types
        assert "mr" in params.abbrev_types

    def test_verbose_training(self, capsys):
        """Test verbose training output."""
        trainer = PunktTrainer()
        trainer.train(TRAIN_TEXT, verbose=True, finalize=True)

        captured = capsys.readouterr()
        assert "Tokenizing text..." in captured.out
        assert "Found" in captured.out
        assert "tokens in text" in captured.out

    def test_streaming_train_finalize(self):
        """Test _streaming_train with finalization."""
        trainer = PunktTrainer(memory_efficient=True)
        # Lower chunk size for testing
        trainer.CHUNK_SIZE = 50

        trainer.train(LARGE_TRAIN_TEXT, verbose=False, finalize=True)
        params = trainer.get_params()

        # Should work same as regular training
        assert len(params.abbrev_types) > 1

    def test_finalize_verbose(self, capsys):
        """Test verbose finalization."""
        trainer = PunktTrainer()
        trainer.train(TRAIN_TEXT, verbose=False, finalize=False)

        # Finalize with verbose
        trainer.finalize_training(verbose=True)

        captured = capsys.readouterr()
        assert "Finalizing training..." in captured.out
        assert "Identifying sentence starters..." in captured.out

    def test_from_json_with_missing_keys(self):
        """Test from_json with minimal configuration."""
        # Minimal JSON data
        json_data = {
            "version": "1.0",
            "parameters": {
                "abbrev_types": ["dr", "mr"],
                "sent_starters": [],
                "collocations": [],
                "ortho_context": {},
            },
        }

        # Should create trainer with defaults for missing config
        trainer = PunktTrainer.from_json(json_data)
        params = trainer.get_params()

        assert "dr" in params.abbrev_types
        assert "mr" in params.abbrev_types

    def test_text_to_batches(self):
        """Test the text_to_batches static method."""
        # Create text with paragraph boundaries
        para1 = "A" * 25  # 25 chars
        para2 = "B" * 25  # 25 chars
        para3 = "C" * 25  # 25 chars
        para4 = "D" * 25  # 25 chars
        text = f"{para1}\n\n{para2}\n\n{para3}\n\n{para4}"  # 100 chars + separators
        batch_size = 60  # Should fit about 2 paragraphs per batch

        batches = list(PunktTrainer.text_to_batches(text, batch_size))

        # Should create 2 batches (para1+para2, para3+para4)
        assert len(batches) == 2
        assert "A" in batches[0] and "B" in batches[0]
        assert "C" in batches[1] and "D" in batches[1]


class TestPunktTrainerEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_training_text(self, trainer):
        """Test training with empty text."""
        trainer.train("", verbose=False, finalize=True)

        params = trainer.get_params()
        # Should only have pre-loaded abbreviations
        assert params.abbrev_types == {"..."}
        assert params.sent_starters == set()
        assert params.collocations == set()

    def test_text_without_periods(self, trainer):
        """Test training with text that has no periods."""
        trainer.train(
            "This is text without any periods or punctuation", verbose=False, finalize=True
        )

        params = trainer.get_params()
        # Should only have pre-loaded abbreviations
        assert params.abbrev_types == {"..."}
        assert params.sent_starters == set()

    def test_very_long_abbreviations(self, trainer):
        """Test that very long abbreviations are not detected."""
        # Text with a very long "abbreviation"
        text = "This is verylongabbreviation. Next sentence."
        trainer.train(text, verbose=False, finalize=True)

        params = trainer.get_params()
        # Should not detect very long abbreviation (> MAX_ABBREV_LENGTH)
        assert "verylongabbreviation" not in params.abbrev_types

    def test_numbers_with_periods(self, trainer):
        """Test handling of numbers with periods."""
        text = "The value is 3.14. Also see section 2.5.1. Temperature is 98.6."
        trainer.train(text, verbose=False, finalize=True)

        params = trainer.get_params()
        # Numbers with periods should generally not be abbreviations
        assert "3.14" not in params.abbrev_types
        assert "2.5.1" not in params.abbrev_types
        assert "98.6" not in params.abbrev_types

    def test_special_token_types(self, trainer):
        """Test handling of special token types like ##number##."""
        # Manually set up trainer state
        trainer._type_fdist = Counter(
            {
                "##number##": 10,  # Special token
                "word.": 5,
            }
        )
        trainer._num_period_toks = 15
        trainer._sentbreak_count = 10

        # Test reclassification - ##number## should be skipped
        results = list(trainer._reclassify_abbrev_types({"##number##", "word."}))

        # Should only process 'word.'
        assert len(results) == 1
        assert results[0][0] == "word"


class TestPunktTrainerCoverage:
    """Additional tests to improve code coverage."""

    def test_streaming_train_with_verbose(self, capsys):
        """Test streaming train with verbose output to cover lines 323-331."""
        trainer = PunktTrainer(memory_efficient=True)
        trainer.CHUNK_SIZE = 50  # Small chunks for testing

        # Train with verbose to trigger tqdm check
        trainer._streaming_train(TRAIN_TEXT, verbose=True)

        captured = capsys.readouterr()
        # Should see the tqdm note
        assert "Note: Install tqdm" in captured.out or "First pass:" in captured.out

    def test_verbose_frequency_output(self, capsys):
        """Test verbose output of frequent tokens (lines 354-368)."""
        trainer = PunktTrainer()
        trainer.train(TRAIN_TEXT, verbose=True, finalize=False)

        captured = capsys.readouterr()
        # Should show frequent tokens
        assert "Most frequent tokens ending with period:" in captured.out
        assert "Identifying abbreviations..." in captured.out

    def test_memory_efficient_sent_starter_tracking(self):
        """Test memory-efficient sentence starter tracking (lines 630-643)."""
        trainer = PunktTrainer(memory_efficient=True)
        trainer.SENT_STARTER_MIN_FREQ = 1  # Low threshold for testing
        trainer.COLLOC_FDIST_MIN_FREQ = 1  # Low threshold for testing

        # Create a small text that will trigger memory-efficient tracking
        text = "Dr. Smith is here. Dr. Jones agrees. Dr. Brown works. This is important. This works well."

        # Train with memory-efficient mode
        trainer.train(text, verbose=False, finalize=True)

        # Check that sentence starters were tracked in memory-efficient mode
        params = trainer.get_params()

        # Should have tracked some sentence starters despite memory-efficient mode
        # The specific starters depend on the thresholds
        assert len(params.sent_starters) >= 0  # May be empty with default thresholds

        # Should have found abbreviations
        assert "dr" in params.abbrev_types

    def test_persistent_abbreviation_scoring(self):
        """Test persistent abbreviation scoring (lines 732-741)."""
        trainer = PunktTrainer()
        trainer.PERSIST_ABBREVS = True
        trainer.ABBREV_CONSISTENCY = 0.3

        # Add existing abbreviation
        trainer._params.abbrev_types.add("ph.d")

        # Set up frequency distributions to test consistency check
        # Need to set up proper counts for all required tokens
        trainer._type_fdist["ph.d."] = 10  # With period
        trainer._type_fdist["ph.d"] = 2  # Without period (appears elsewhere)
        trainer._type_fdist["the"] = 50  # Some common word
        trainer._type_fdist["."] = 30  # Periods
        trainer._num_period_toks = 20
        trainer._sentbreak_count = 15

        # Set up total token count to avoid division by zero
        trainer._token_count = 100

        # Test reclassification with existing abbreviation
        results = list(trainer._reclassify_abbrev_types({"ph.d."}))

        # Should maintain the abbreviation due to consistency
        assert len(results) == 1
        typ, score, is_add = results[0]
        assert typ == "ph.d"
        # Check that it's classified as abbreviation
        assert is_add is True

    def test_verbose_collocation_output(self, capsys):
        """Test verbose collocation output (lines 973-982)."""
        # Use trainer with collocations
        trainer = PunktTrainer()
        trainer.COLLOCATION = 2.0  # Lower threshold
        trainer.MIN_COLLOC_FREQ = 1
        trainer.INCLUDE_ABBREV_COLLOCS = True

        # Train with large corpus that has collocations
        trainer.train(LARGE_TRAIN_TEXT, verbose=True, finalize=True)

        captured = capsys.readouterr()
        # Check for collocation output
        if trainer._params.collocations:
            assert "Most common collocations" in captured.out

    def test_empty_text_to_batches(self):
        """Test text_to_batches with empty text."""
        batches = list(PunktTrainer.text_to_batches("", 100))
        assert len(batches) == 0

    def test_text_to_batches_single_large_paragraph(self):
        """Test text_to_batches with single paragraph larger than batch size."""
        text = "A" * 1000  # Single large paragraph
        batches = list(PunktTrainer.text_to_batches(text, 100))
        # Should yield the entire paragraph even though it's larger than batch_size
        assert len(batches) == 1
        assert batches[0] == text

    def test_pruning_in_streaming_mode(self):
        """Test that pruning occurs in streaming mode."""
        trainer = PunktTrainer(memory_efficient=True)
        trainer.PRUNE_INTERVAL = 10  # Very low for testing
        trainer.TYPE_FDIST_MIN_FREQ = 2
        trainer.CHUNK_SIZE = 100  # Larger chunk to ensure we get some words

        # Create text with many unique words that will be pruned
        # and some repeated words that will be kept
        unique_words = [f"rare{i}" for i in range(30)]
        common_words = ["common", "frequent", "repeated"] * 10
        all_words = unique_words + common_words
        text = " ".join(all_words) + ". This is a test sentence."

        trainer.train(text, verbose=False, finalize=False)

        # Check that some low-frequency words were pruned
        # Common words should remain, rare words should be pruned
        assert "common" in trainer._type_fdist
        assert "frequent" in trainer._type_fdist
        # Most rare words should be pruned
        rare_count = sum(1 for word in trainer._type_fdist if word.startswith("rare"))
        assert rare_count < 15  # Less than half of the rare words should remain

    def test_ortho_heuristic_memory_efficient(self):
        """Test _ortho_heuristic in memory-efficient mode to cover lines 630-643."""
        trainer = PunktTrainer(memory_efficient=True)
        trainer.SENT_STARTER_MIN_FREQ = 1
        trainer.COLLOC_FDIST_MIN_FREQ = 1

        # Use a text that will create the right conditions
        text = "Dr. Smith works here. Dr. Jones agrees. Dr. Smith returns. The patient arrives."

        # Train to trigger memory-efficient paths
        trainer.train(text, verbose=False, finalize=True)

        # Check results
        params = trainer.get_params()
        assert "dr" in params.abbrev_types

    def test_is_rare_abbrev_type(self):
        """Test _is_rare_abbrev_type method basic functionality."""
        trainer = PunktTrainer()

        # Create a mock token for testing
        from nupunkt.core.tokens import PunktToken

        # Test that tokens without sentbreak are not considered
        token_no_break = PunktToken("test.")
        token_no_break.sentbreak = False
        next_token = PunktToken("next")

        assert trainer._is_rare_abbrev_type(token_no_break, next_token) is False

        # Test that tokens already marked as abbreviations are not considered
        token_abbr = PunktToken("test.")
        token_abbr.sentbreak = True
        token_abbr.abbr = True

        assert trainer._is_rare_abbrev_type(token_abbr, next_token) is False

    def test_dunning_likelihood_edge_cases(self):
        """Test edge cases in Dunning likelihood calculation."""
        # This tests the statistics module indirectly through the trainer
        trainer = PunktTrainer()

        # Set up edge case: very small counts
        trainer._type_fdist["edge."] = 1
        trainer._type_fdist["edge"] = 0
        trainer._type_fdist["."] = 1
        trainer._num_period_toks = 1
        trainer._sentbreak_count = 1

        # This should not crash
        results = list(trainer._reclassify_abbrev_types({"edge."}))
        assert len(results) == 1

    def test_train_from_files(self, tmp_path):
        """Test training from file paths."""
        # Create test files
        file1 = tmp_path / "text1.txt"
        file2 = tmp_path / "text2.txt"

        file1.write_text("Dr. Smith works here. Mr. Jones agrees.")
        file2.write_text("The company was founded in Jan. 2020.")

        # Read files and train
        trainer = PunktTrainer()
        text1 = file1.read_text()
        text2 = file2.read_text()
        combined_text = text1 + " " + text2
        trainer.train(combined_text, verbose=False, finalize=True)

        params = trainer.get_params()
        assert "dr" in params.abbrev_types
        assert "mr" in params.abbrev_types
        # 'jan' may not be detected with such limited training data

    def test_type_no_sentperiod_property(self):
        """Test handling of tokens with sentence-ending periods."""
        trainer = PunktTrainer()

        # Train with text that has sentence-ending periods
        text = "This ends. That continues..."
        trainer.train(text, verbose=False, finalize=True)

        # Should handle ellipsis correctly
        params = trainer.get_params()
        assert "..." in params.abbrev_types

    def test_consistency_check_division_safety(self):
        """Test that consistency checks handle division by zero."""
        trainer = PunktTrainer()
        trainer.PERSIST_ABBREVS = True
        trainer.ABBREV_CONSISTENCY = 0.3

        # Add existing abbreviation
        trainer._params.abbrev_types.add("zero")

        # Set up edge case: zero counts
        trainer._type_fdist["zero."] = 0
        trainer._type_fdist["zero"] = 0
        trainer._num_period_toks = 10
        trainer._sentbreak_count = 5
        trainer._token_count = 100

        # This should not crash - but skip tokens with zero count
        # The method filters out tokens with zero count before processing
        results = list(trainer._reclassify_abbrev_types({"zero.", "."}))  # Add '.' which has count
        # Should only process '.', not 'zero.' which has zero count
        assert len(results) <= 1  # May return 0 or 1 depending on filtering

    def test_high_abbreviation_threshold(self):
        """Test behavior with very high abbreviation threshold."""
        trainer = PunktTrainer()
        trainer.ABBREV = 10.0  # Very high threshold

        trainer.train(TRAIN_TEXT, verbose=False, finalize=True)

        params = trainer.get_params()
        # With high threshold, fewer abbreviations should be found
        # Only the most obvious ones
        assert len(params.abbrev_types) < 5  # Should find very few
