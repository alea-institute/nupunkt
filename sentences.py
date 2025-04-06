"""
Sentence segmentation functionality.
"""

from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from charboundary.segmenters.base import TextSegmenter


class SentenceSegmenter:
    """
    Handles segmenting text into sentences.
    """

    @staticmethod
    def segment_to_sentences(
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Segment text into a list of sentences.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                       Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[str]: List of sentences
        """
        # Quick return for empty text
        if not text:
            return []

        # Use optimized segmentation based on text size
        if streaming and len(text) > 10000:
            # For large texts, use streaming segmentation
            # Note: streaming mode doesn't currently support custom threshold
            segmented_parts = list(segmenter.segment_text_streaming(text))
            segmented_text = "".join(segmented_parts)
        else:
            # For smaller texts, use regular segmentation
            segmented_text = segmenter.segment_text(text, threshold=threshold)

        # Fast path: if no sentence tags were added, return the whole text as one sentence
        if segmenter.SENTENCE_TAG not in segmented_text:
            return [text] if text else []

        # More efficient string splitting and processing
        # Pre-compute tag lengths for performance
        para_tag_len = len(segmenter.PARAGRAPH_TAG)

        # Split by sentence tag, but handle paragraph tags properly
        sentences = []
        segments = segmented_text.split(segmenter.SENTENCE_TAG)

        # First segment is always before any sentence tag
        if segments[0]:
            sentences.append(segments[0])

        # Process remaining segments (each starts after a sentence tag)
        for segment in segments[1:]:
            # Remove any paragraph tags at the beginning of the segment
            if segment.startswith(segmenter.PARAGRAPH_TAG):
                segment = segment[para_tag_len:]

            # Remove any paragraph tags in the segment
            segment = segment.replace(segmenter.PARAGRAPH_TAG, "")

            if segment:
                sentences.append(segment)

        # Post-processing to fix incorrectly segmented quotation marks
        # This handles edge cases where the model fails to correctly process quotes
        i = 0
        while i < len(sentences) - 1:
            # Handle case where a sentence ends with a quote and next "sentence" is just a quote
            if (sentences[i].endswith('"') or sentences[i].endswith('"')) and sentences[
                i + 1
            ].strip() == '"':
                # Merge the quote with the following sentence
                if i + 2 < len(sentences):
                    sentences[i + 2] = '" ' + sentences[i + 2]
                    sentences.pop(i + 1)  # Remove the standalone quote
                    continue
            # Handle case where a "sentence" is just a quote that should connect to the next sentence
            if sentences[i].strip() == '"' and i + 1 < len(sentences):
                # Join with the next sentence
                sentences[i + 1] = '" ' + sentences[i + 1]
                sentences.pop(i)  # Remove the standalone quote
                continue
            i += 1

        return sentences

    @classmethod
    def get_sentence_spans_with_text(
        cls,
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Segment text into a list of sentences with their character spans.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[tuple[str, tuple[int, int]]]: List of tuples containing (sentence, (start_index, end_index))
        """
        from charboundary.segmenters.spans import SpanHandler

        # Quick return for empty text
        if not text:
            return []

        # Find boundary positions
        boundary_positions = SpanHandler.find_boundary_positions(
            segmenter, text, threshold=threshold
        )

        # If no boundaries found, return the whole text as one sentence
        if not boundary_positions:
            return [(text, (0, len(text)))]

        # Create spans from boundary positions
        result = []
        start_idx = 0

        # Build spans from boundaries
        for end_idx in boundary_positions:
            sentence = text[start_idx:end_idx]
            # Include all spans, even if only whitespace
            result.append((sentence, (start_idx, end_idx)))
            start_idx = end_idx

        # Add final segment if needed (for text after the last boundary)
        if start_idx < len(text):
            sentence = text[start_idx:]
            result.append((sentence, (start_idx, len(text))))

        return result

    @classmethod
    def get_sentence_spans(
        cls,
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get the character spans for each sentence in the text.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[tuple[int, int]]: List of character spans (start_index, end_index)
        """
        segments_with_spans = cls.get_sentence_spans_with_text(
            segmenter, text, streaming=streaming, threshold=threshold
        )
        return [span for _, span in segments_with_spans]