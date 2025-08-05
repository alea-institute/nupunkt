"""Tests for cross-platform path utilities."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from nupunkt.utils.paths import (
    ensure_user_directories,
    get_legacy_user_dir,
    get_model_search_paths,
    get_user_cache_dir,
    get_user_data_dir,
    migrate_legacy_models,
)


class TestPlatformPaths:
    """Test platform-specific directory detection."""

    @patch("sys.platform", "linux")
    def test_linux_data_dir_with_xdg(self):
        """Test Linux data directory with XDG_DATA_HOME set."""
        with patch.dict(os.environ, {"XDG_DATA_HOME": "/custom/data"}):
            assert get_user_data_dir() == Path("/custom/data/nupunkt")

    @patch("sys.platform", "linux")
    def test_linux_data_dir_without_xdg(self):
        """Test Linux data directory without XDG_DATA_HOME."""
        with patch.dict(os.environ, {}, clear=True):
            expected = Path.home() / ".local" / "share" / "nupunkt"
            assert get_user_data_dir() == expected

    @patch("sys.platform", "linux")
    def test_linux_cache_dir_with_xdg(self):
        """Test Linux cache directory with XDG_CACHE_HOME set."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/custom/cache"}):
            assert get_user_cache_dir() == Path("/custom/cache/nupunkt")

    @patch("sys.platform", "linux")
    def test_linux_cache_dir_without_xdg(self):
        """Test Linux cache directory without XDG_CACHE_HOME."""
        with patch.dict(os.environ, {}, clear=True):
            expected = Path.home() / ".cache" / "nupunkt"
            assert get_user_cache_dir() == expected

    @patch("sys.platform", "darwin")
    def test_macos_data_dir(self):
        """Test macOS data directory."""
        expected = Path.home() / "Library" / "Application Support" / "nupunkt"
        assert get_user_data_dir() == expected

    @patch("sys.platform", "darwin")
    def test_macos_cache_dir(self):
        """Test macOS cache directory."""
        expected = Path.home() / "Library" / "Caches" / "nupunkt"
        assert get_user_cache_dir() == expected

    @patch("sys.platform", "win32")
    def test_windows_data_dir_with_localappdata(self):
        """Test Windows data directory with LOCALAPPDATA set."""
        with patch.dict(os.environ, {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}):
            result = get_user_data_dir()
            # Check the path components instead of exact match due to platform differences
            assert str(result).replace("\\", "/").endswith("Local/nupunkt")

    @patch("sys.platform", "win32")
    def test_windows_data_dir_with_appdata(self):
        """Test Windows data directory with only APPDATA set."""
        with patch.dict(os.environ, {"APPDATA": "C:\\Users\\Test\\AppData\\Roaming"}, clear=True):
            result = get_user_data_dir()
            assert str(result).replace("\\", "/").endswith("Roaming/nupunkt")

    @patch("sys.platform", "win32")
    def test_windows_data_dir_fallback(self):
        """Test Windows data directory fallback to home."""
        with patch.dict(os.environ, {}, clear=True):
            expected = Path.home() / "nupunkt"
            assert get_user_data_dir() == expected

    @patch("sys.platform", "win32")
    def test_windows_cache_dir_with_localappdata(self):
        """Test Windows cache directory with LOCALAPPDATA set."""
        with patch.dict(os.environ, {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}):
            result = get_user_cache_dir()
            assert str(result).replace("\\", "/").endswith("Local/nupunkt/Cache")

    @patch("sys.platform", "win32")
    def test_windows_cache_dir_with_temp(self):
        """Test Windows cache directory fallback to TEMP."""
        with patch.dict(os.environ, {"TEMP": "C:\\Temp"}, clear=True):
            result = get_user_cache_dir()
            assert str(result).replace("\\", "/").endswith("Temp/nupunkt")


class TestLegacySupport:
    """Test legacy directory support."""

    def test_legacy_dir(self):
        """Test legacy directory path."""
        expected = Path.home() / ".nupunkt"
        assert get_legacy_user_dir() == expected


class TestModelSearchPaths:
    """Test model search path functionality."""

    def test_search_paths_order(self):
        """Test that search paths are returned in correct order."""
        paths = get_model_search_paths()

        # Should have at least package directory
        assert len(paths) >= 1

        # First should be package models directory
        assert paths[0].name == "models"
        assert paths[0].parent.name == "nupunkt"

    def test_search_paths_include_existing_only(self):
        """Test that only existing directories are included."""
        paths = get_model_search_paths()

        # All returned paths should exist
        for path in paths:
            assert path.exists(), f"Path {path} should exist"


class TestDirectoryCreation:
    """Test directory creation functionality."""

    def test_ensure_user_directories(self):
        """Test creating user directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the directory functions to use temp directory
            test_data_dir = Path(tmpdir) / "data" / "nupunkt"
            test_cache_dir = Path(tmpdir) / "cache" / "nupunkt"

            with (
                patch("nupunkt.utils.paths.get_user_data_dir", return_value=test_data_dir),
                patch("nupunkt.utils.paths.get_user_cache_dir", return_value=test_cache_dir),
            ):
                ensure_user_directories()

                # Check directories were created
                assert test_data_dir.exists()
                assert (test_data_dir / "models").exists()
                assert test_cache_dir.exists()
                assert (test_cache_dir / "models").exists()

                # Check permissions on Unix-like systems
                if sys.platform != "win32":
                    assert oct(test_data_dir.stat().st_mode)[-3:] == "700"
                    assert oct(test_cache_dir.stat().st_mode)[-3:] == "700"


class TestModelMigration:
    """Test model migration functionality."""

    def test_migrate_legacy_models(self):
        """Test migrating models from legacy to new location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up legacy and new directories
            legacy_dir = Path(tmpdir) / ".nupunkt" / "models"
            new_dir = Path(tmpdir) / "data" / "nupunkt" / "models"

            # Create legacy directory with a model
            legacy_dir.mkdir(parents=True)
            test_model = legacy_dir / "test_model.bin"
            test_model.write_text("test model data")

            # Mock the directory functions
            with (
                patch("nupunkt.utils.paths.get_legacy_user_dir", return_value=legacy_dir.parent),
                patch("nupunkt.utils.paths.get_user_data_dir", return_value=new_dir.parent),
            ):
                migrate_legacy_models()

                # Check model was copied
                new_model = new_dir / "test_model.bin"
                assert new_model.exists()
                assert new_model.read_text() == "test model data"

                # Original should still exist
                assert test_model.exists()

    def test_migrate_legacy_models_no_overwrite(self):
        """Test that migration doesn't overwrite existing models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up legacy and new directories
            legacy_dir = Path(tmpdir) / ".nupunkt" / "models"
            new_dir = Path(tmpdir) / "data" / "nupunkt" / "models"

            # Create both directories with models
            legacy_dir.mkdir(parents=True)
            new_dir.mkdir(parents=True)

            # Create models with different content
            legacy_model = legacy_dir / "test_model.bin"
            legacy_model.write_text("old model data")

            new_model = new_dir / "test_model.bin"
            new_model.write_text("new model data")

            # Mock the directory functions
            with (
                patch("nupunkt.utils.paths.get_legacy_user_dir", return_value=legacy_dir.parent),
                patch("nupunkt.utils.paths.get_user_data_dir", return_value=new_dir.parent),
            ):
                migrate_legacy_models()

                # New model should not be overwritten
                assert new_model.read_text() == "new model data"
