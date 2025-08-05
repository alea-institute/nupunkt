#!/usr/bin/env python3
"""
Integration tests for the full training workflow.

These tests ensure that the complete user journey works:
train -> save -> load -> use
"""

import tempfile
from pathlib import Path

import pytest

from nupunkt.tokenizers.sentence_tokenizer import PunktParameters, PunktSentenceTokenizer
from nupunkt.training import train_model


def test_basic_training_workflow():
    """Test the most basic workflow a user would follow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "my_model.bin"

        # Step 1: Train a model
        training_text = """
        Dr. Smith went to Washington. He met with Mr. Jones.
        The meeting was at 3 p.m. on January 15th.
        They discussed the new project. It was very exciting.
        """

        train_model(training_texts=[training_text], output_path=model_path)

        # Verify model was saved
        assert model_path.exists(), "Model file was not created"

        # Step 2: Load the model (as a user would)
        params = PunktParameters.load(model_path)
        tokenizer = PunktSentenceTokenizer(params)

        # Step 3: Use the model
        test_text = "Dr. Johnson arrived at 2 p.m. for the meeting. Mr. Williams was already there."
        sentences = list(tokenizer.tokenize(test_text))

        # Basic sanity checks
        assert len(sentences) == 2
        assert sentences[0].strip() == "Dr. Johnson arrived at 2 p.m. for the meeting."
        assert sentences[1].strip() == "Mr. Williams was already there."

        # Verify learned abbreviations are present
        assert "dr" in params.abbrev_types
        assert "mr" in params.abbrev_types
        assert "p.m" in params.abbrev_types


def test_training_with_abbreviations_workflow():
    """Test training with custom abbreviations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "abbrev_model.bin"

        # Create custom abbreviations file
        import json

        abbrev_file = Path(tmpdir) / "custom_abbrevs.json"
        with open(abbrev_file, "w") as f:
            json.dump(["Ph.D", "M.D", "B.S", "U.S.A", "Dr", "Mr", "Ms"], f)

        # Train with abbreviations
        training_text = "She has a Ph.D. in Computer Science. He got his B.S. from MIT."

        train_model(
            training_texts=[training_text],
            abbreviation_files=[abbrev_file],
            output_path=model_path,
            use_default_abbreviations=False,  # Only use our custom ones
        )

        # Load and test
        params = PunktParameters.load(model_path)
        tokenizer = PunktSentenceTokenizer(params)

        test_text = (
            "Dr. Lee has both a Ph.D. and an M.D. degree. She studied in the U.S.A. for years."
        )
        sentences = list(tokenizer.tokenize(test_text))

        # Should handle Ph.D. and M.D. correctly
        assert len(sentences) == 2
        assert "ph.d" in params.abbrev_types
        assert "m.d" in params.abbrev_types
        assert "u.s.a" in params.abbrev_types


def test_all_format_workflows():
    """Test that all save formats work in the full workflow."""
    formats = [("binary", ".bin"), ("json", ".json"), ("json_xz", ".json.xz")]

    training_text = "This is a test. It has multiple sentences. Dr. Smith approves."

    for format_type, extension in formats:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / f"model{extension}"

            # Train and save in specific format
            train_model(
                training_texts=[training_text], output_path=model_path, format_type=format_type
            )

            # Verify file exists and has reasonable size
            assert model_path.exists()
            assert model_path.stat().st_size > 0

            # Load and use
            params = PunktParameters.load(model_path)
            tokenizer = PunktSentenceTokenizer(params)

            sentences = list(tokenizer.tokenize("Test text. Another sentence."))
            assert len(sentences) == 2


def test_memory_efficient_training_workflow():
    """Test memory-efficient training mode works in full workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "efficient_model.bin"

        # Create a larger training text
        training_text = " ".join([f"Sentence {i}. Dr. Smith wrote this." for i in range(1000)])

        # Train with memory-efficient mode
        # Pass as string, not list, to avoid file path checking
        train_model(
            training_texts=training_text,
            output_path=model_path,
            memory_efficient=True,
            batch_size=50000,  # Small batch size to test batching
        )

        # Load and verify
        params = PunktParameters.load(model_path)
        assert "dr" in params.abbrev_types

        # Use the model
        tokenizer = PunktSentenceTokenizer(params)
        sentences = list(tokenizer.tokenize("Dr. Jones arrived. Ms. Smith left."))
        assert len(sentences) == 2


def test_model_compatibility_error_handling():
    """Test that we get useful errors for incompatible model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file that's not a valid model
        bad_model = Path(tmpdir) / "bad_model.bin"
        with open(bad_model, "wb") as f:
            f.write(b"This is not a valid model file")

        # Should raise a clear error
        with pytest.raises(ValueError, match="Invalid file format|not a nupunkt"):
            PunktParameters.load(bad_model)


def test_incremental_training_workflow():
    """Test training on multiple texts and combining them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "combined_model.bin"

        # Multiple training sources
        texts = [
            "Legal text with U.S.C. citations. See 15 U.S.C. § 78j(b).",
            "Medical text about Dr. Smith, M.D. who works at the hospital.",
            "Business text about Apple Inc. and Microsoft Corp. earnings.",
        ]

        # Train on all
        train_model(training_texts=texts, output_path=model_path)

        # Load and verify all abbreviations were learned
        params = PunktParameters.load(model_path)

        # Should have abbreviations from all domains
        assert "u.s.c" in params.abbrev_types
        assert "m.d" in params.abbrev_types
        assert "inc" in params.abbrev_types
        assert "corp" in params.abbrev_types


if __name__ == "__main__":
    # Run basic tests if executed directly
    test_basic_training_workflow()
    test_training_with_abbreviations_workflow()
    test_all_format_workflows()
    test_memory_efficient_training_workflow()
    test_incremental_training_workflow()
    print("✅ All integration tests passed!")
