"""Tests for the standard segmentation interface (words, sentences, paragraphs)."""

import pytest

import nupunkt
from nupunkt import Segment, Segmenter, WordSegmenter, contiguous
from nupunkt.segmentation import SegmenterMixin, tight

TEXT = (
    '  Dr. Smith\'s "quote," here.  Second one!\n\n'
    "New para here. It cites 42 U.S.C. § 1983. Done.\n"
)

LEVELS = ["word", "sentence", "paragraph"]


def _check_invariants(segments: list[Segment], source: str) -> None:
    """Tight, ascending, non-overlapping, and text == source[start:end]."""
    prev_end = 0
    for seg in segments:
        assert isinstance(seg, Segment)
        assert 0 <= seg.start < seg.end <= len(source)
        assert seg.start >= prev_end
        assert source[seg.start : seg.end] == seg.text
        assert seg.text == seg.text.strip()
        prev_end = seg.end


class TestSegment:
    def test_unpacks_as_triple(self):
        seg = Segment("abc", 2, 5)
        text, start, end = seg
        assert (text, start, end) == ("abc", 2, 5)
        assert seg.span == (2, 5)
        assert len(seg) == 3
        assert seg == ("abc", 2, 5)


class TestInvariants:
    @pytest.mark.parametrize("level", LEVELS)
    def test_tight_ascending_spans(self, level):
        seg = nupunkt.segmenter(level)
        _check_invariants(seg.segments(TEXT), TEXT)

    @pytest.mark.parametrize("level", LEVELS)
    def test_three_forms_agree(self, level):
        seg = nupunkt.segmenter(level)
        segments = seg.segments(TEXT)
        assert seg.texts(TEXT) == [s.text for s in segments]
        assert seg.spans(TEXT) == [s.span for s in segments]

    @pytest.mark.parametrize("level", LEVELS)
    def test_generators_match_lists(self, level):
        seg = nupunkt.segmenter(level)
        assert list(seg.iter_segments(TEXT)) == seg.segments(TEXT)
        assert list(seg.iter_texts(TEXT)) == seg.texts(TEXT)
        assert list(seg.iter_spans(TEXT)) == seg.spans(TEXT)
        # they really are generators
        gen = seg.iter_segments(TEXT)
        assert next(gen) == seg.segments(TEXT)[0]

    @pytest.mark.parametrize("level", LEVELS)
    def test_segmenter_protocol(self, level):
        assert isinstance(nupunkt.segmenter(level), Segmenter)

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
    def test_empty_and_whitespace_only(self, level, text):
        seg = nupunkt.segmenter(level)
        assert seg.segments(text) == []
        assert seg.texts(text) == []
        assert seg.spans(text) == []

    def test_unknown_level(self):
        with pytest.raises(ValueError):
            nupunkt.segmenter("phrase")  # type: ignore[arg-type]


class TestModuleFunctions:
    def test_words(self):
        assert nupunkt.words('Dr. Smith\'s "quote," here.') == [
            "Dr.",
            "Smith's",
            '"',
            "quote",
            ",",
            '"',
            "here.",
        ]
        assert nupunkt.word_spans("ab cd") == [(0, 2), (3, 5)]
        assert nupunkt.word_segments("ab cd") == [Segment("ab", 0, 2), Segment("cd", 3, 5)]

    def test_words_across_lines_keep_offsets(self):
        text = "one two\nthree\n\nfour"
        segs = nupunkt.word_segments(text)
        assert [s.text for s in segs] == ["one", "two", "three", "four"]
        _check_invariants(segs, text)

    def test_sentences(self):
        assert nupunkt.sentences(TEXT) == [
            'Dr. Smith\'s "quote," here.',
            "Second one!",
            "New para here.",
            "It cites 42 U.S.C. § 1983.",
            "Done.",
        ]
        assert nupunkt.sentence_spans(TEXT)[0] == (2, 28)
        assert nupunkt.sentence_segments(TEXT)[1] == Segment("Second one!", 30, 41)

    def test_sentences_adaptive(self):
        plain = nupunkt.sentences(TEXT, adaptive=True)
        assert plain == nupunkt.sentences(TEXT)
        _check_invariants(nupunkt.sentence_segments(TEXT, adaptive=True), TEXT)
        assert isinstance(nupunkt.segmenter("sentence", adaptive=True), Segmenter)

    def test_paragraphs(self):
        assert nupunkt.paragraphs(TEXT) == [
            'Dr. Smith\'s "quote," here.  Second one!',
            "New para here. It cites 42 U.S.C. § 1983. Done.",
        ]
        assert nupunkt.paragraph_spans(TEXT) == [(2, 41), (43, 90)]
        assert nupunkt.paragraph_segments(TEXT)[0].text == nupunkt.paragraphs(TEXT)[0]

    def test_levels_nest(self):
        """Every sentence lies within a paragraph and every word within a sentence."""
        paras = nupunkt.paragraph_spans(TEXT)
        sents = nupunkt.sentence_spans(TEXT)
        wordspans = nupunkt.word_spans(TEXT)
        assert all(any(ps <= s and e <= pe for ps, pe in paras) for s, e in sents)
        assert all(any(ss <= s and e <= se for ss, se in sents) for s, e in wordspans)

    def test_model_argument(self):
        path = str(nupunkt.models.get_default_model_path())
        assert nupunkt.sentences(TEXT, model=path) == nupunkt.sentences(TEXT)
        assert nupunkt.paragraphs(TEXT, model=path) == nupunkt.paragraphs(TEXT)


class TestContiguous:
    @pytest.mark.parametrize("level", LEVELS)
    def test_covers_source_without_gaps(self, level):
        segs = contiguous(nupunkt.segmenter(level).segments(TEXT), TEXT)
        assert "".join(s.text for s in segs) == TEXT
        assert segs[0].start == 0 and segs[-1].end == len(TEXT)
        assert all(a.end == b.start for a, b in zip(segs, segs[1:]))

    def test_empty(self):
        assert contiguous([], "   ") == []

    def test_tight_round_trip(self):
        segs = nupunkt.sentence_segments(TEXT)
        assert list(tight(contiguous(segs, TEXT), TEXT)) == segs


class TestTokenizerObjects:
    def test_sentence_tokenizer_has_interface(self):
        tok = nupunkt.load("default")
        assert isinstance(tok, SegmenterMixin)
        assert tok.texts(TEXT) == nupunkt.sentences(TEXT)
        assert tok.spans(TEXT) == nupunkt.sentence_spans(TEXT)

    def test_paragraph_tokenizer_has_interface(self):
        tok = nupunkt.PunktParagraphTokenizer(nupunkt.load("default"))
        assert tok.texts(TEXT) == nupunkt.paragraphs(TEXT)

    def test_word_segmenter_custom_lang_vars(self):
        seg = WordSegmenter(nupunkt.PunktLanguageVars())
        assert seg.texts("a b.") == ["a", "b."]

    def test_custom_segmenter_from_mixin(self):
        class Lines(SegmenterMixin):
            def iter_segments(self, text):
                pos = 0
                for line in text.split("\n"):
                    if line.strip():
                        yield Segment(line, pos, pos + len(line))
                    pos += len(line) + 1

        seg = Lines()
        assert seg.texts("a\n\nb") == ["a", "b"]
        assert seg.spans("a\n\nb") == [(0, 1), (3, 4)]
        assert list(seg.iter_texts("a\n\nb")) == ["a", "b"]
        assert isinstance(seg, Segmenter)


class TestLegacyCompatibility:
    """The old names keep their exact behaviour."""

    def test_legacy_sentence_functions(self):
        assert nupunkt.sent_tokenize(TEXT)[0] == '  Dr. Smith\'s "quote," here.'
        assert nupunkt.sent_spans(TEXT)[0] == (0, 30)
        assert nupunkt.sent_spans_with_text(TEXT)[0] == ('  Dr. Smith\'s "quote," here.  ', (0, 30))
        assert nupunkt.sent_spans_adaptive(TEXT) == nupunkt.sent_spans(TEXT)
        assert [t for t, _ in nupunkt.sent_spans_with_text_adaptive(TEXT)] == [
            t for t, _ in nupunkt.sent_spans_with_text(TEXT)
        ]

    def test_legacy_paragraph_functions(self):
        assert nupunkt.para_spans(TEXT) == [(0, 41), (41, len(TEXT))]
        assert nupunkt.para_tokenize(TEXT)[1].startswith("\n\nNew para")
        assert nupunkt.para_spans_with_text(TEXT)[0][1] == (0, 41)

    def test_legacy_tokenizer_methods(self):
        tok = nupunkt.load("default")
        assert tok.tokenize(TEXT) == tok.sentences_from_text(TEXT) == nupunkt.sent_tokenize(TEXT)
        assert list(tok.span_tokenize(TEXT))[0] == (0, 28)
        assert tok.tokenize_with_spans(TEXT) == nupunkt.sent_spans_with_text(TEXT)
        para = nupunkt.PunktParagraphTokenizer(tok)
        assert para.tokenize(TEXT) == nupunkt.para_tokenize(TEXT)
        assert para.span_tokenize(TEXT) == nupunkt.para_spans(TEXT)
        assert para.tokenize_with_spans(TEXT) == nupunkt.para_spans_with_text(TEXT)
        assert tok._lang_vars.word_tokenize("a b.") == ["a", "b."]

    def test_legacy_names_exported(self):
        for name in [
            "sent_tokenize",
            "sent_tokenize_adaptive",
            "sent_spans",
            "sent_spans_with_text",
            "sent_spans_adaptive",
            "sent_spans_with_text_adaptive",
            "para_tokenize",
            "para_spans",
            "para_spans_with_text",
        ]:
            assert name in nupunkt.__all__
            assert callable(getattr(nupunkt, name))
