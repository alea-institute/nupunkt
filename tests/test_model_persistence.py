#!/usr/bin/env python3
"""
Test model persistence - ensure save/load cycle preserves all parameters.
"""

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from nupunkt.tokenizers.sentence_tokenizer import PunktParameters


def test_model_persistence():
    """Test that all parameters are preserved through save/load cycle."""
    print("Testing Model Persistence")
    print("=" * 60)

    # Create parameters with all types of data
    original = PunktParameters()

    # Add abbreviations
    original.abbrev_types = {"dr", "mr", "mrs", "u.s.c", "inc", "ltd", "ph.d"}

    # Add sentence starters
    original.sent_starters = {"The", "However", "Therefore", "Moreover"}

    # Add collocations
    original.collocations = {
        ("New", "York"),
        ("Los", "Angeles"),
        ("United", "States"),
        ("et", "al"),
    }

    # Add orthographic contexts
    original.ortho_context = {
        "Apple": 6,  # BEG_UC | MID_UC
        "Microsoft": 6,
        "however": 16,  # BEG_LC
        "the": 48,  # MID_LC | UNK_LC
    }

    # Test all formats
    formats = ["binary", "json", "json_xz"]

    for format_type in formats:
        print(f"\nTesting {format_type} format:")

        # Choose appropriate suffix
        if format_type == "binary":
            suffix = ".bin"
        elif format_type == "json_xz":
            suffix = ".json.xz"
        else:
            suffix = ".json"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Save
            original.save(tmp_path, format_type=format_type)
            print(f"  Saved to: {tmp_path} ({tmp_path.stat().st_size} bytes)")

            # Load using classmethod (correct way)
            loaded = PunktParameters.load(tmp_path)

            # Verify abbreviations
            assert loaded.abbrev_types == original.abbrev_types, (
                f"Abbreviations mismatch: {loaded.abbrev_types} != {original.abbrev_types}"
            )
            print(f"  ✓ Abbreviations: {len(loaded.abbrev_types)} preserved")

            # Verify sentence starters
            assert loaded.sent_starters == original.sent_starters, (
                f"Sentence starters mismatch: {loaded.sent_starters} != {original.sent_starters}"
            )
            print(f"  ✓ Sentence starters: {len(loaded.sent_starters)} preserved")

            # Verify collocations
            assert loaded.collocations == original.collocations, (
                f"Collocations mismatch: {loaded.collocations} != {original.collocations}"
            )
            print(f"  ✓ Collocations: {len(loaded.collocations)} preserved")

            # Verify orthographic context
            assert dict(loaded.ortho_context) == dict(original.ortho_context), (
                f"Ortho context mismatch: {dict(loaded.ortho_context)} != {dict(original.ortho_context)}"
            )
            print(f"  ✓ Orthographic contexts: {len(loaded.ortho_context)} preserved")

            # Clean up
            tmp_path.unlink()

    print("\n✓ All persistence tests passed!")


def test_edge_cases():
    """Test edge cases in persistence."""
    print("\n\nTesting Edge Cases")
    print("=" * 60)

    # Empty parameters
    empty = PunktParameters()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        empty.save(tmp_path, format_type="binary")
        loaded = PunktParameters.load(tmp_path)

        assert len(loaded.abbrev_types) == 0
        assert len(loaded.sent_starters) == 0
        assert len(loaded.collocations) == 0
        assert len(loaded.ortho_context) == 0
        print("✓ Empty parameters preserved correctly")

        tmp_path.unlink()

    # Large model (simulate realistic scenario)
    large = PunktParameters()

    # Add many abbreviations
    for i in range(1000):
        large.abbrev_types.add(f"abbr{i}")

    # Add many ortho contexts
    for i in range(5000):
        large.ortho_context[f"word{i}"] = i % 64

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        large.save(tmp_path, format_type="binary", compression_method="lzma", compression_level=9)
        loaded = PunktParameters.load(tmp_path)

        assert len(loaded.abbrev_types) == 1000
        assert len(loaded.ortho_context) == 5000
        print(
            f"✓ Large model preserved correctly (file size: {tmp_path.stat().st_size / 1024:.2f} KB)"
        )

        tmp_path.unlink()


def test_real_model():
    """Test loading the actual trained USC model."""
    print("\n\nTesting Real USC Model")
    print("=" * 60)

    model_path = Path("kl3m-usc-punkt-model.bin")
    if model_path.exists():
        params = PunktParameters.load(model_path)

        print("✓ Successfully loaded USC model")
        print(f"  - Abbreviations: {len(params.abbrev_types)}")
        print(f"  - Sentence starters: {len(params.sent_starters)}")
        print(f"  - Collocations: {len(params.collocations)}")
        print(f"  - Orthographic contexts: {len(params.ortho_context)}")

        # Test that key abbreviations are present
        expected_abbrevs = {"u.s", "u.s.c", "inc", "corp", "dr", "mr"}
        found = expected_abbrevs & params.abbrev_types
        print(f"  - Key abbreviations found: {found}")

        assert len(found) == len(expected_abbrevs), (
            f"Missing abbreviations: {expected_abbrevs - found}"
        )
    else:
        print("⚠ USC model not found, skipping real model test")


if __name__ == "__main__":
    test_model_persistence()
    test_edge_cases()
    test_real_model()
    print("\n✅ All tests passed!")
