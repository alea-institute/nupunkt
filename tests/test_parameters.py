"""Unit tests for nupunkt parameters module."""

import pytest
import tempfile
from pathlib import Path
import json
import os

from nupunkt.core.parameters import PunktParameters
from nupunkt.utils.compression import load_compressed_json


def test_punkt_parameters_basic():
    """Test basic functionality of PunktParameters."""
    params = PunktParameters()
    
    # Should start empty
    assert len(params.abbrev_types) == 0
    assert len(params.collocations) == 0
    assert len(params.sent_starters) == 0
    
    # Test adding abbreviations
    params.abbrev_types.add("dr")
    assert "dr" in params.abbrev_types
    
    # Test adding collocations
    params.collocations.add(("new", "york"))
    assert ("new", "york") in params.collocations
    
    # Test adding sentence starters
    params.sent_starters.add("however")
    assert "however" in params.sent_starters


def test_punkt_parameters_ortho_context():
    """Test orthographic context in PunktParameters."""
    params = PunktParameters()
    
    # Test adding orthographic context
    params.add_ortho_context("word", 1)
    assert params.ortho_context["word"] == 1
    
    # Test adding to existing context
    params.add_ortho_context("word", 2)
    assert params.ortho_context["word"] == 3  # 1 | 2 = 3


def test_punkt_parameters_save_load():
    """Test saving and loading parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parameters
        params = PunktParameters()
        params.abbrev_types.update(["dr", "mr", "prof"])
        params.collocations.add(("new", "york"))
        params.sent_starters.add("however")
        params.add_ortho_context("word", 1)
        
        # Save uncompressed JSON
        uncompressed_path = Path(tmpdir) / "params.json"
        params.save(uncompressed_path, format_type="json")
        
        # Verify it's saved as expected
        assert uncompressed_path.exists()
        
        # Load back
        loaded_params = PunktParameters.load(uncompressed_path)
        
        # Verify it's the same
        assert "dr" in loaded_params.abbrev_types
        assert "mr" in loaded_params.abbrev_types
        assert "prof" in loaded_params.abbrev_types
        assert ("new", "york") in loaded_params.collocations
        assert "however" in loaded_params.sent_starters
        assert loaded_params.ortho_context["word"] == 1
        
        # Save compressed JSON
        compressed_path = Path(tmpdir) / "params.json.xz"
        params.save(compressed_path, format_type="json_xz")
        
        # Verify it's saved as expected
        assert compressed_path.exists()
        
        # Load back
        loaded_params = PunktParameters.load(compressed_path)
        
        # Verify it's the same
        assert "dr" in loaded_params.abbrev_types
        assert "mr" in loaded_params.abbrev_types
        assert "prof" in loaded_params.abbrev_types
        assert ("new", "york") in loaded_params.collocations
        assert "however" in loaded_params.sent_starters
        assert loaded_params.ortho_context["word"] == 1
        
        # Save binary format
        binary_path = Path(tmpdir) / "params.bin"
        params.save(binary_path, format_type="binary")
        
        # Verify it's saved as expected
        assert binary_path.exists()
        
        # Load back
        loaded_params = PunktParameters.load(binary_path)
        
        # Verify it's the same
        assert "dr" in loaded_params.abbrev_types
        assert "mr" in loaded_params.abbrev_types
        assert "prof" in loaded_params.abbrev_types
        assert ("new", "york") in loaded_params.collocations
        assert "however" in loaded_params.sent_starters
        assert loaded_params.ortho_context["word"] == 1


def test_punkt_parameters_json_methods():
    """Test to_json and from_json methods."""
    # Create parameters
    params = PunktParameters()
    params.abbrev_types.update(["dr", "mr", "prof"])
    params.collocations.add(("new", "york"))
    params.sent_starters.add("however")
    params.add_ortho_context("word", 1)
    
    # Convert to JSON
    json_data = params.to_json()
    
    # Verify it's a dict
    assert isinstance(json_data, dict)
    assert "abbrev_types" in json_data
    assert "collocations" in json_data
    assert "sent_starters" in json_data
    assert "ortho_context" in json_data
    
    # Convert back from JSON
    new_params = PunktParameters.from_json(json_data)
    
    # Verify it's the same
    assert set(new_params.abbrev_types) == set(params.abbrev_types)
    assert set(new_params.collocations) == set(params.collocations)
    assert set(new_params.sent_starters) == set(params.sent_starters)
    assert new_params.ortho_context == params.ortho_context


@pytest.mark.benchmark(group="parameters")
def test_parameters_save_benchmark(benchmark):
    """Benchmark parameter saving with/without compression."""
    # Create parameters with substantial data
    params = PunktParameters()
    
    # Add a significant number of abbreviations
    params.abbrev_types.update([f"abbrev{i}" for i in range(500)])
    
    # Add collocations
    for i in range(200):
        params.collocations.add((f"word{i}", f"word{i+1}"))
    
    # Add sentence starters
    params.sent_starters.update([f"starter{i}" for i in range(100)])
    
    # Add orthographic contexts
    for i in range(1000):
        params.add_ortho_context(f"word{i}", i % 4)
    
    def save_compressed():
        with tempfile.NamedTemporaryFile(suffix=".json.xz", delete=True) as tmp:
            params.save(tmp.name, format_type="json_xz", compression_level=1)
            # Get file size
            size = os.path.getsize(tmp.name)
            # Return size to see in benchmark results
            return size
    
    # Run the benchmark
    file_size = benchmark(save_compressed)
    
    # Simple verification that something was saved
    assert file_size > 0


@pytest.mark.benchmark(group="parameters")
def test_parameters_load_benchmark(benchmark):
    """Benchmark parameter loading with/without compression."""
    # Create parameters with substantial data
    params = PunktParameters()
    
    # Add a significant number of abbreviations
    params.abbrev_types.update([f"abbrev{i}" for i in range(500)])
    
    # Add collocations
    for i in range(200):
        params.collocations.add((f"word{i}", f"word{i+1}"))
    
    # Add sentence starters
    params.sent_starters.update([f"starter{i}" for i in range(100)])
    
    # Add orthographic contexts
    for i in range(1000):
        params.add_ortho_context(f"word{i}", i % 4)
    
    # Save to a temporary file first
    with tempfile.NamedTemporaryFile(suffix=".json.xz", delete=False) as tmp:
        params.save(tmp.name, format_type="json_xz", compression_level=1)
        temp_path = tmp.name
    
    def load_compressed():
        loaded_params = PunktParameters.load(temp_path)
        return loaded_params
    
    # Run the benchmark
    loaded_params = benchmark(load_compressed)
    
    # Cleanup
    os.unlink(temp_path)
    
    # Simple verification that it loaded correctly
    assert len(loaded_params.abbrev_types) == 500
    assert len(loaded_params.collocations) == 200
    assert len(loaded_params.sent_starters) == 100
    assert len(loaded_params.ortho_context) == 1000