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
    it = iter(iterable)
    prev = next(it, None)
    if prev is None:
        return
    for current in it:
        yield prev, current
        prev = current
    yield prev, None