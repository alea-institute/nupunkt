"""Tests for model loading with cross-platform search paths."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nupunkt import load
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_default_model(self):
        """Test loading the default model."""
        tokenizer = load("default")
        assert isinstance(tokenizer, PunktSentenceTokenizer)

        # Should be able to tokenize
        sentences = list(tokenizer.tokenize("Hello world. How are you?"))
        assert len(sentences) == 2

    def test_load_by_absolute_path(self):
        """Test loading a model by absolute file path."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
            # Create a minimal model file
            model_data = {
                "parameters": {
                    "abbrev_types": {"Dr": 1, "Mr": 1},
                    "collocations": {},
                    "sent_starters": {},
                    "ortho_context": {},
                }
            }
            tmpfile.write(json.dumps(model_data).encode())
            tmpfile.flush()

            # Load by absolute path
            tokenizer = load(tmpfile.name)
            assert isinstance(tokenizer, PunktSentenceTokenizer)

            # Clean up
            Path(tmpfile.name).unlink()

    def test_load_by_name_from_package(self):
        """Test loading a model by name from package directory."""
        # This test uses the actual default model in the package
        # We'll mock the search to only look in package directory
        with patch("nupunkt.utils.paths.get_model_search_paths") as mock_search:
            package_dir = Path(__file__).parent.parent / "nupunkt" / "models"
            mock_search.return_value = [package_dir]

            # Should find default_model.bin or default_model.json.xz
            tokenizer = load("default_model")
            assert isinstance(tokenizer, PunktSentenceTokenizer)

    def test_load_by_name_from_user_dir(self):
        """Test loading a model by name from user directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a models directory
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create a test model
            model_path = models_dir / "custom_model.json"
            model_data = {
                "parameters": {
                    "abbrev_types": {"Ph.D": 1},
                    "collocations": {},
                    "sent_starters": {},
                    "ortho_context": {},
                },
                "nupunkt_version": "0.6.0",
            }
            model_path.write_text(json.dumps(model_data))

            # Mock search paths to include our temp directory
            with patch("nupunkt.utils.paths.get_model_search_paths") as mock_search:
                mock_search.return_value = [models_dir]

                tokenizer = load("custom_model")
                assert isinstance(tokenizer, PunktSentenceTokenizer)
                assert "Ph.D" in tokenizer._params.abbrev_types

    def test_load_model_not_found(self):
        """Test error when model is not found."""
        with patch("nupunkt.utils.paths.get_model_search_paths") as mock_search:
            mock_search.return_value = [Path("/fake/path1"), Path("/fake/path2")]

            with pytest.raises(FileNotFoundError) as exc_info:
                load("nonexistent_model")

            error_msg = str(exc_info.value)
            assert "nonexistent_model" in error_msg
            assert "/fake/path1" in error_msg
            assert "/fake/path2" in error_msg

    def test_load_with_version_warning(self):
        """Test that loading a model with different version shows warning."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
            # Create a model with different version
            model_data = {
                "parameters": {
                    "abbrev_types": {},
                    "collocations": {},
                    "sent_starters": {},
                    "ortho_context": {},
                },
                "nupunkt_version": "0.1.0",  # Old version
            }
            tmpfile.write(json.dumps(model_data).encode())
            tmpfile.flush()

            # Load and check for warning
            with pytest.warns(UserWarning, match="Model was created with nupunkt 0.1.0"):
                tokenizer = load(tmpfile.name)

            assert isinstance(tokenizer, PunktSentenceTokenizer)

            # Clean up
            Path(tmpfile.name).unlink()

    def test_load_caching(self):
        """Test that models are cached when loaded multiple times."""
        # Load default model twice
        tokenizer1 = load("default")
        tokenizer2 = load("default")

        # Should be the same object due to caching
        assert tokenizer1 is tokenizer2


class TestSearchPathPriority:
    """Test that search paths are checked in correct priority order."""

    def test_package_dir_takes_precedence(self):
        """Test that package directory is searched before user directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two directories with same model name
            package_dir = Path(tmpdir) / "package" / "models"
            user_dir = Path(tmpdir) / "user" / "models"
            package_dir.mkdir(parents=True)
            user_dir.mkdir(parents=True)

            # Create models with different content
            package_model = package_dir / "test.json"
            user_model = user_dir / "test.json"

            package_data = {
                "parameters": {
                    "abbrev_types": {"PACKAGE": 1},
                    "collocations": {},
                    "sent_starters": {},
                    "ortho_context": {},
                }
            }
            user_data = {
                "parameters": {
                    "abbrev_types": {"USER": 1},
                    "collocations": {},
                    "sent_starters": {},
                    "ortho_context": {},
                }
            }

            package_model.write_text(json.dumps(package_data))
            user_model.write_text(json.dumps(user_data))

            # Mock search paths with package dir first
            with patch("nupunkt.utils.paths.get_model_search_paths") as mock_search:
                mock_search.return_value = [package_dir, user_dir]

                tokenizer = load("test")
                # Should load from package directory
                assert "PACKAGE" in tokenizer._params.abbrev_types
                assert "USER" not in tokenizer._params.abbrev_types
