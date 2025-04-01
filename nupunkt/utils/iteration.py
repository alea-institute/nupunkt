"""
Iteration utilities for nupunkt.

This module provides utility functions for iterating through sequences
with specialized behaviors needed for the Punkt algorithm.
"""

from typing import Any, Iterator, Optional, Tuple


def pair_iter(iterable: Iterator[Any]) -> Iterator[Tuple[Any, Optional[Any]]]:
    """
    Iterate through pairs of items from an iterable, where the second item
    can be None for the last item.
    
    Args:
        iterable: The input iterator
        
    Yields:
        Pairs of (current_item, next_item) where next_item is None for the last item
    """
    # Check if iterable is already a list or other sequence with O(1) random access
    # This is a significant optimization for repeated calls
    if hasattr(iterable, '__getitem__') and hasattr(iterable, '__len__'):
        # Use indexed access which is faster than iteration
        sequence = iterable
        length = len(sequence)
        if length == 0:
            return
        for i in range(length - 1):
            yield sequence[i], sequence[i + 1]
        yield sequence[length - 1], None
    else:
        # Fall back to iterator-based approach for non-sequence iterables
        it = iter(iterable)
        prev = next(it, None)
        if prev is None:
            return
        for current in it:
            yield prev, current
            prev = current
        yield prev, None