"""
Paragraph segmentation functionality.
"""

import re
from typing import List, Tuple, Optional, TYPE_CHECKING

from charboundary.segmenters.spans import SpanHandler

if TYPE_CHECKING:
    from charboundary.segmenters.base import TextSegmenter

# Precompiled regex pattern for two or more consecutive newlines
# This will efficiently detect paragraph breaks
PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n+")


class ParagraphSegmenter:
    """
    Handles segmenting text into paragraphs.
    """

    @classmethod
    def segment_to_paragraphs(
        cls,
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,  # pylint: disable=unused-argument
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Segment text into a list of paragraphs.

        This method identifies paragraph breaks ONLY at sentence boundaries that are
        immediately followed by two or more newlines.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            streaming (bool, optional): Ignored. Included for API compatibility only.
                                       Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[str]: List of paragraphs
        """
        # Quick return for empty text
        if not text:
            return []

        # We need to handle the case where the sentence segmentation
        # includes the newlines in the next sentence
        paragraph_spans = cls.get_paragraph_spans_with_text(
            segmenter, text, streaming, threshold
        )
        return [para_text for para_text, _ in paragraph_spans]

    @classmethod
    def get_paragraph_spans(
        cls,
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,  # pylint: disable=unused-argument
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get the character spans for each paragraph in the text.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            streaming (bool, optional): Ignored. Included for API compatibility only.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[tuple[int, int]]: List of character spans (start_index, end_index)
        """
        paragraphs_with_spans = cls.get_paragraph_spans_with_text(
            segmenter, text, streaming, threshold
        )
        return [span for _, span in paragraphs_with_spans]

    @classmethod
    # pylint: disable=too-many-locals,too-many-branches
    def get_paragraph_spans_with_text(
        cls,
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,  # pylint: disable=unused-argument
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Segment text into a list of paragraphs with their character spans.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            streaming (bool, optional): Ignored. Included for API compatibility only.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: List of tuples containing 
                (paragraph, (start_index, end_index))
        """
        # Quick return for empty text
        if not text:
            return []

        # Direct search for multiple newlines after sentence boundaries

        # Get all sentence boundary positions
        boundary_positions = SpanHandler.find_boundary_positions(
            segmenter, text, threshold
        )

        # If no boundaries, treat the whole text as one paragraph
        if not boundary_positions:
            return [(text.strip(), (0, len(text)))]

        # Find paragraph boundaries (sentence boundaries followed by 2+ newlines)
        paragraph_boundaries = []

        for pos in boundary_positions:
            # Look for 2+ newlines right after this boundary
            window_end = min(
                pos + 10, len(text)
            )  # 10 char window is enough for newlines

            if pos < len(text):
                # Get the text slice to check
                window = text[pos:window_end]

                # Search for 2+ newlines in the window using the precompiled pattern
                match = PARAGRAPH_BREAK_PATTERN.search(window)

                # Only consider it a match if the newlines appear near the start of the window
                if (
                    match and match.start() <= 3
                ):  # Allow for a few whitespace chars after sentence
                    paragraph_boundaries.append(pos)

        # Always include the end of text as a paragraph boundary
        if not paragraph_boundaries or paragraph_boundaries[-1] != len(text):
            paragraph_boundaries.append(len(text))

        # Create paragraph spans from boundaries
        result = []
        start_idx = 0

        for end_idx in paragraph_boundaries:
            # Get the paragraph text (trim whitespace)
            paragraph_text = text[start_idx:end_idx].strip()

            # Only include non-empty paragraphs
            if paragraph_text:
                # For the span, include leading/trailing whitespace to ensure contiguity
                result.append((paragraph_text, (start_idx, end_idx)))

            start_idx = end_idx

        # Fix span coverage and ensure contiguity
        if result:
            # Make sure the first span starts at 0
            if result[0][1][0] > 0:
                paragraph_text, (_, end) = result[0]
                result[0] = (paragraph_text, (0, end))

            # Make sure the last span ends at the text length
            if result[-1][1][1] < len(text):
                paragraph_text, (start, _) = result[-1]
                result[-1] = (paragraph_text, (start, len(text)))

            # Ensure contiguity between spans
            for i in range(len(result) - 1):
                if result[i][1][1] != result[i + 1][1][0]:
                    paragraph_text, (start, _) = result[i]
                    result[i] = (paragraph_text, (start, result[i + 1][1][0]))

        return result