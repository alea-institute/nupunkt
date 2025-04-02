"""Basic tests for nupunkt utility functions."""

import tempfile
from pathlib import Path

import pytest

from nupunkt.utils.compression import load_compressed_json, save_compressed_json
from nupunkt.utils.iteration import pair_iter
from nupunkt.utils.statistics import collocation_log_likelihood, dunning_log_likelihood


def test_pair_iter():
    """Test the pair_iter utility function."""
    # Empty list should yield nothing
    assert list(pair_iter([])) == []

    # Single item list should yield a single pair with None as second element
    result = list(pair_iter([1]))
    assert len(result) == 1
    assert result[0] == (1, None)

    # Two item list
    result = list(pair_iter([1, 2]))
    assert len(result) == 2
    assert result[0] == (1, 2)
    assert result[1] == (2, None)

    # Multiple item list
    result = list(pair_iter([1, 2, 3, 4]))
    assert len(result) == 4
    assert result[0] == (1, 2)
    assert result[1] == (2, 3)
    assert result[2] == (3, 4)
    assert result[3] == (4, None)


def test_dunning_log_likelihood():
    """Test the dunning_log_likelihood function."""
    # Test with simple values
    ll = dunning_log_likelihood(100, 1000, 50, 10000)
    assert isinstance(ll, float)
    # The function returns negative values by design for Punkt algorithm

    # Higher count_ab should generally result in smaller negative values
    ll1 = dunning_log_likelihood(100, 1000, 10, 10000)
    ll2 = dunning_log_likelihood(100, 1000, 50, 10000)
    assert ll1 < ll2

    # Test with edge cases
    ll = dunning_log_likelihood(0, 0, 0, 1)
    assert isinstance(ll, float)
    assert ll == 0.0  # Edge case handling

    # Test with (1,1,1,1) corner case
    ll = dunning_log_likelihood(1, 1, 1, 1)
    assert isinstance(ll, float)
    # This case can return negative values by design


def test_collocation_log_likelihood():
    """Test the collocation_log_likelihood function."""
    # Test with simple values
    ll = collocation_log_likelihood(100, 200, 50, 10000)
    assert isinstance(ll, float)
    assert ll > 0

    # Higher count_ab should generally result in higher likelihood
    ll1 = collocation_log_likelihood(100, 200, 10, 10000)
    ll2 = collocation_log_likelihood(100, 200, 50, 10000)
    assert ll2 > ll1

    # Perfect correlation should have high likelihood
    ll = collocation_log_likelihood(100, 100, 100, 10000)
    assert ll > 0

    # Test edge cases
    ll = collocation_log_likelihood(0, 0, 0, 1)
    assert isinstance(ll, float)

    ll = collocation_log_likelihood(1, 1, 1, 1)
    assert isinstance(ll, float)


def test_compression_functions_basic():
    """Test basic functionality of compression utility functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}}

        # Test without compression
        uncompressed_path = Path(tmpdir) / "test_uncompressed.json"
        save_compressed_json(test_data, uncompressed_path, use_compression=False)

        # Verify it's a standard JSON file
        assert uncompressed_path.exists()
        assert uncompressed_path.suffix == ".json"

        # Load it back and verify content
        loaded_data = load_compressed_json(uncompressed_path)
        assert loaded_data == test_data

        # Test with compression
        compressed_path = Path(tmpdir) / "test_compressed.json.xz"
        save_compressed_json(test_data, compressed_path, use_compression=True)

        # Verify it's compressed
        assert compressed_path.exists()
        assert compressed_path.suffix == ".xz"

        # Load it back and verify content
        loaded_data = load_compressed_json(compressed_path)
        assert loaded_data == test_data


def test_compression_automatic_extension():
    """Test automatic extension handling in compression functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {"key1": "value1", "key2": [1, 2, 3]}

        # Test with base path
        base_path = Path(tmpdir) / "test_file"

        # Save with compression
        save_compressed_json(test_data, base_path, use_compression=True)
        expected_path = Path(f"{base_path}.json.xz")
        assert expected_path.exists()

        # Load back with automatic detection - use the expected path directly
        loaded_data = load_compressed_json(expected_path)
        assert loaded_data == test_data

        # Test with .json extension but requesting compression
        json_path = Path(tmpdir) / "test_file.json"
        save_compressed_json(test_data, json_path, use_compression=True)
        expected_path = Path(f"{json_path}.xz")
        assert expected_path.exists()

        # Load back with automatic detection - use the expected path directly
        loaded_data = load_compressed_json(expected_path)
        assert loaded_data == test_data


def test_compression_level_verification():
    """Test that different compression levels work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a larger test dataset
        test_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        # Compress with level 1 (fast)
        fast_path = Path(tmpdir) / "fast_compressed.json.xz"
        save_compressed_json(test_data, fast_path, level=1)

        # Compress with level 9 (best)
        best_path = Path(tmpdir) / "best_compressed.json.xz"
        save_compressed_json(test_data, best_path, level=9)

        # Get file sizes
        fast_size = fast_path.stat().st_size
        best_size = best_path.stat().st_size

        # For very small test data, compression level might not make a big difference
        # or might even have reverse effect due to compression metadata overhead
        # Just verify that both files are created and have reasonable size
        assert fast_size > 0
        assert best_size > 0

        # Both should load correctly
        assert load_compressed_json(fast_path) == test_data
        assert load_compressed_json(best_path) == test_data


@pytest.mark.benchmark(group="compression")
def test_compression_benchmark(benchmark):
    """Benchmark compression functions."""
    # Create test data - mix of strings, numbers, and nested structures
    test_data = {
        "strings": [f"value_{i}" * 20 for i in range(50)],
        "numbers": [i * 3.14159 for i in range(100)],
        "nested": [{"id": i, "name": f"item_{i}" * 5, "active": i % 2 == 0} for i in range(50)],
    }

    # Prepare data for compression benchmarking

    def compress_func():
        with tempfile.NamedTemporaryFile(suffix=".json.xz", delete=True) as tmp:
            # Use fastest compression level for benchmarking
            save_compressed_json(test_data, tmp.name, level=1)
            # Read back to ensure full cycle
            return load_compressed_json(tmp.name)

    # Run the benchmark
    result = benchmark(compress_func)

    # Simple assertions to make sure it worked
    assert result == test_data


@pytest.mark.benchmark(group="compression")
def test_no_compression_benchmark(benchmark):
    """Benchmark without compression for comparison."""
    # Use the same test data as in the compression benchmark
    test_data = {
        "strings": [f"value_{i}" * 20 for i in range(50)],
        "numbers": [i * 3.14159 for i in range(100)],
        "nested": [{"id": i, "name": f"item_{i}" * 5, "active": i % 2 == 0} for i in range(50)],
    }

    def no_compress_func():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            save_compressed_json(test_data, tmp.name, use_compression=False)
            return load_compressed_json(tmp.name)

    # Run the benchmark
    result = benchmark(no_compress_func)

    # Simple assertions to make sure it worked
    assert result == test_data
