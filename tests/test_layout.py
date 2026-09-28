"""Tests for layout-aware segmentation (nupunkt.layout)."""

import pytest

import nupunkt
from nupunkt import PunktParagraphTokenizer, Segment, blank_page_furniture
from nupunkt.layout import continues_across, is_heading_line, iter_blocks, layout_segments

HEADING_DOC = "INTRODUCTION\n\nThe court held for the plaintiff. It awarded fees.\n\nII. Facts\n\nThe parties met."


def _tight(segments, text):
    for s in segments:
        assert text[s.start : s.end] == s.text == s.text.strip()


class TestBlankLines:
    def test_heading_is_its_own_sentence_by_default(self):
        assert nupunkt.sentences(HEADING_DOC) == [
            "INTRODUCTION",
            "The court held for the plaintiff.",
            "It awarded fees.",
            "II. Facts",
            "The parties met.",
        ]

    def test_punkt_only_when_disabled(self):
        assert nupunkt.sentences(HEADING_DOC, paragraph_breaks=False) == [
            "INTRODUCTION\n\nThe court held for the plaintiff.",
            "It awarded fees.",
            "II. Facts\n\nThe parties met.",
        ]

    def test_disabled_matches_legacy_functions(self):
        legacy = [s.strip() for s in nupunkt.sent_tokenize(HEADING_DOC)]
        assert nupunkt.sentences(HEADING_DOC, paragraph_breaks=False) == legacy

    def test_legacy_functions_are_unchanged(self):
        # sent_tokenize never applies layout rules
        assert (
            nupunkt.sent_tokenize(HEADING_DOC)[0]
            == "INTRODUCTION\n\nThe court held for the plaintiff."
        )
        assert nupunkt.sent_spans(HEADING_DOC)[0] == (0, 48)

    def test_spans_are_tight_and_nested(self):
        segs = nupunkt.sentence_segments(HEADING_DOC)
        _tight(segs, HEADING_DOC)
        paras = nupunkt.paragraph_segments(HEADING_DOC)
        _tight(paras, HEADING_DOC)
        assert [p.text for p in paras] == [
            "INTRODUCTION",
            "The court held for the plaintiff. It awarded fees.",
            "II. Facts",
            "The parties met.",
        ]
        assert all(any(p.start <= s.start and s.end <= p.end for p in paras) for s in segs)

    def test_whitespace_in_blank_lines(self):
        text = "A heading  \n \t \nBody text here."
        assert nupunkt.sentences(text) == ["A heading", "Body text here."]

    def test_windows_newlines(self):
        text = "Heading\r\n\r\nBody text here. More."
        assert nupunkt.sentences(text) == ["Heading", "Body text here.", "More."]


class TestPageBreaks:
    @pytest.mark.parametrize(
        "text",
        [
            "The court held that the\n\ndefendant had waived the claim. Next one.",
            "The court considered the estab-\n\nlishment clause. Next one.",
            "The court held that the\n\n\n\ndefendant had waived the claim. Next one.",
        ],
    )
    def test_sentence_continues_across_page_break(self, text):
        sentences = nupunkt.sentences(text)
        assert len(sentences) == 2
        assert sentences[1] == "Next one."
        assert "defendant" in sentences[0] or "lishment" in sentences[0]

    def test_page_number_between_halves(self):
        text = "The court held that the\n\n12\n\ndefendant had waived the claim. Next one."
        # Without cleanup the page number is its own segment and the sentence is cut
        assert nupunkt.sentences(text) == [
            "The court held that the",
            "12",
            "defendant had waived the claim.",
            "Next one.",
        ]
        cleaned = blank_page_furniture(text)
        assert len(cleaned) == len(text)
        sentences = nupunkt.sentence_segments(cleaned)
        assert [s.text.split() for s in sentences] == [
            ["The", "court", "held", "that", "the", "defendant", "had", "waived", "the", "claim."],
            ["Next", "one."],
        ]
        # spans index the original text
        assert text[sentences[1].start : sentences[1].end] == "Next one."

    @pytest.mark.parametrize(
        "furniture", ["12", "- 12 -", "[12]", "Page 3", "Page 3 of 10", "p. 12", "\f", "(7)"]
    )
    def test_blank_page_furniture_shapes(self, furniture):
        text = f"Held that the\n\n{furniture}\n\nclaim was waived."
        cleaned = blank_page_furniture(text)
        assert len(cleaned) == len(text)
        assert furniture.strip() not in cleaned or furniture.strip() == ""
        assert nupunkt.sentences(cleaned)[0].split() == [
            "Held",
            "that",
            "the",
            "claim",
            "was",
            "waived.",
        ]

    def test_words_are_not_furniture(self):
        text = "Held that the\n\nPage header words here\n\nclaim was waived."
        assert blank_page_furniture(text) == text

    def test_uppercase_after_gap_is_a_boundary(self):
        text = "The court held that the defendant\n\nThe appeal followed."
        assert nupunkt.sentences(text) == [
            "The court held that the defendant",
            "The appeal followed.",
        ]

    def test_punctuation_before_gap_is_a_boundary(self):
        assert continues_across("He left.\n\nthen came back.", 8, 10) is False
        assert continues_across("He left the\n\nroom quietly.", 11, 13) is True


class TestLineBreaks:
    def test_off_by_default(self):
        text = "Section 1\nThe court held. Second."
        assert nupunkt.sentences(text) == ["Section 1\nThe court held.", "Second."]

    def test_headings_and_lists(self):
        text = "Section 1\nThe court held. Second.\nItems:\n- first thing\n- second thing\nDone."
        assert nupunkt.sentences(text, line_breaks=True) == [
            "Section 1",
            "The court held.",
            "Second.",
            "Items:",
            "- first thing",
            "- second thing",
            "Done.",
        ]

    def test_wrapped_prose_is_not_cut(self):
        text = "wrapped prose line here that\ncontinues on the next line. Done."
        assert nupunkt.sentences(text, line_breaks=True) == [
            "wrapped prose line here that\ncontinues on the next line.",
            "Done.",
        ]

    def test_heading_detection(self):
        assert is_heading_line("INTRODUCTION")
        assert is_heading_line("II. Facts of the Case")
        assert not is_heading_line("The court held.")
        assert not is_heading_line(
            "a very long line of ordinary prose that keeps going without any punctuation at all"
        )
        assert not is_heading_line("   ")


class TestBlocks:
    def test_iter_blocks_modes(self):
        text = "A\n\nB b.\nC\n\n\n"
        assert list(iter_blocks(text, False, False)) == [(0, len(text))]
        assert [text[s:e] for s, e in iter_blocks(text, True, False)] == ["A", "B b.\nC"]
        assert [text[s:e] for s, e in iter_blocks(text, True, True)] == ["A", "B b.\nC"]
        text2 = "Body.\nII. Facts\nMore."
        # "II. Facts" starts with a list-style marker, so it is a unit of its own
        assert [text2[s:e] for s, e in iter_blocks(text2, True, True)] == [
            "Body.",
            "II. Facts",
            "More.",
        ]
        assert list(iter_blocks("  \n\n ", True, True)) == []

    def test_layout_segments_with_custom_splitter(self):
        def split_words(block):
            pos = 0
            for w in block.split(" "):
                yield (pos, pos + len(w))
                pos += len(w) + 1

        text = "ab cd\n\nEf"
        segs = list(layout_segments(text, split_words))
        assert segs == [Segment("ab", 0, 2), Segment("cd", 3, 5), Segment("Ef", 7, 9)]


class TestObjects:
    def test_tokenizer_attributes_are_defaults(self):
        tok = nupunkt.load("default")
        assert tok.paragraph_breaks is True and tok.line_breaks is False
        assert tok.texts(HEADING_DOC)[0] == "INTRODUCTION"
        assert tok.texts(HEADING_DOC, paragraph_breaks=False)[0].startswith("INTRODUCTION\n\n")

    def test_unknown_option_raises(self):
        with pytest.raises(TypeError):
            nupunkt.load("default").texts("A.", nonsense=True)
        with pytest.raises(TypeError):
            nupunkt.segmenter("word").texts("A.", paragraph_breaks=True)

    def test_paragraph_tokenizer_modes(self):
        para = PunktParagraphTokenizer(nupunkt.load("default"))
        assert para.texts(HEADING_DOC)[0] == "INTRODUCTION"
        legacy = para.texts(HEADING_DOC, paragraph_breaks=False)
        assert legacy[0].startswith("INTRODUCTION\n\nThe court")
        assert para.tokenize(HEADING_DOC)[0].startswith("INTRODUCTION\n\nThe court")

    def test_document_modes(self):
        doc = nupunkt.segment(HEADING_DOC)
        assert [p.text for p in doc.paragraphs] == nupunkt.paragraphs(HEADING_DOC)
        assert [s.text for s in doc.sentences] == nupunkt.sentences(HEADING_DOC)
        assert all(
            text_ok for text_ok in (HEADING_DOC[s.start : s.end] == s.text for s in doc.sentences)
        )
        legacy = nupunkt.segment(HEADING_DOC, paragraph_breaks=False)
        assert [s.text for s in legacy.sentences] == nupunkt.sentences(
            HEADING_DOC, paragraph_breaks=False
        )
        lines = nupunkt.segment("Section 1\nThe court held.", line_breaks=True)
        assert [s.text for s in lines.sentences] == ["Section 1", "The court held."]

    def test_adaptive_gets_layout_too(self):
        assert nupunkt.sentences(HEADING_DOC, adaptive=True)[0] == "INTRODUCTION"
