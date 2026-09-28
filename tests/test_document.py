"""Tests for one-pass hierarchical segmentation (nupunkt.segment / Document)."""

import json
import pickle

import pytest

import nupunkt
from nupunkt import Document, Paragraph, Segment, Sentence
from nupunkt.document import is_paragraph_break

TEXT = (
    '  Dr. Smith\'s "quote," here.  Second one!\n\n'
    "New para here. It cites 42 U.S.C. § 1983. Done.\n"
)

# Texts that exercise the paragraph-break window and whitespace handling
EDGE_TEXTS = [
    "",
    "   ",
    "\n\n\t\n",
    "a",
    "No period at all",
    "A.\n\n\n\nB.",
    "A.  \n \n B.",
    "A.\n         \nB.",
    "A.    \n\nB.",
    "A. \n\nB.\n\n",
    "x\n\ny",
    "Hi.\r\n\r\nThere.",
    "One. Two.\nStill para one.\n\nPara two. Yes!\n\n\n  Para three?  ",
    'He left. . . Then she came.\n\nIt ended."Next one." Ok.',
    TEXT,
]


def _check_tree(doc: Document) -> None:
    """Tight spans, ascending, non-overlapping, and each level nests in its parent."""
    text = doc.text
    prev_p = prev_s = prev_w = 0
    for para in doc.paragraphs:
        assert isinstance(para, Paragraph)
        assert text[para.start : para.end] == para.text == para.text.strip()
        assert para.start >= prev_p
        prev_p = para.end
        for sent in para.sentences:
            assert isinstance(sent, Sentence)
            assert para.start <= sent.start < sent.end <= para.end
            assert text[sent.start : sent.end] == sent.text == sent.text.strip()
            assert sent.start >= prev_s
            prev_s = sent.end
            for word in sent.words:
                assert sent.start <= word.start < word.end <= sent.end
                assert text[word.start : word.end] == word.text
                assert word.start >= prev_w
                prev_w = word.end


class TestMatchesPerLevelFunctions:
    @pytest.mark.parametrize("text", EDGE_TEXTS)
    def test_flat_levels_equal_separate_calls(self, text):
        doc = nupunkt.segment(text)
        assert doc.paragraphs == nupunkt.paragraph_segments(text)
        assert doc.sentences == nupunkt.sentence_segments(text)
        assert doc.words == nupunkt.word_segments(text)
        _check_tree(doc)

    def test_matches_paragraph_tokenizer_object(self):
        tok = nupunkt.PunktParagraphTokenizer(nupunkt.load("default"))
        for text in EDGE_TEXTS:
            assert nupunkt.segment(text).paragraphs == tok.segments(text)

    def test_tokenizer_instance_as_model(self):
        tok = nupunkt.load("default")
        assert nupunkt.segment(TEXT, model=tok).sentences == tok.segments(TEXT)

    def test_model_path(self):
        path = str(nupunkt.models.get_default_model_path())
        assert nupunkt.segment(TEXT, model=path).paragraphs == nupunkt.paragraph_segments(TEXT)


class TestTree:
    def test_structure(self):
        doc = nupunkt.segment(TEXT)
        assert doc.text == TEXT
        assert len(doc) == 2
        assert [len(p.sentences) for p in doc.paragraphs] == [2, 3]
        assert doc.paragraphs[0].sentences[1] == Segment("Second one!", 30, 41)
        assert [w.text for w in doc.paragraphs[1].sentences[0].words] == [
            "New",
            "para",
            "here.",
        ]
        assert "paragraphs=2" in repr(doc)

    def test_nodes_are_segments(self):
        doc = nupunkt.segment(TEXT)
        para = doc.paragraphs[0]
        sent = para.sentences[0]
        assert isinstance(para, Segment) and isinstance(sent, Segment)
        text, start, end = sent
        assert (text, start, end) == ('Dr. Smith\'s "quote," here.', 2, 28)
        assert sent.span == (2, 28)
        assert len(sent) == 26

    def test_iterators_match_lists(self):
        doc = nupunkt.segment(TEXT)
        assert list(doc) == doc.paragraphs
        assert list(doc.iter_paragraphs()) == doc.paragraphs
        assert list(doc.iter_sentences()) == doc.sentences
        assert list(doc.iter_words()) == doc.words
        para = doc.paragraphs[1]
        assert list(para.iter_sentences()) == para.sentences
        assert list(para.iter_words()) == para.words
        sent = para.sentences[0]
        assert list(sent.iter_words()) == sent.words

    def test_paragraph_words(self):
        doc = nupunkt.segment(TEXT)
        assert doc.paragraphs[0].words + doc.paragraphs[1].words == doc.words

    def test_words_are_lazy_and_cached(self):
        doc = nupunkt.segment(TEXT)
        sent = doc.sentences[0]
        assert sent._words is None
        first = sent.words
        assert sent._words is first
        assert sent.words is first
        # other sentences untouched
        assert doc.sentences[1]._words is None

    def test_iter_words_is_incremental(self):
        doc = nupunkt.segment(TEXT)
        gen = doc.iter_words()
        assert next(gen).text == "Dr."
        assert doc.sentences[-1]._words is None

    def test_empty_document(self):
        doc = nupunkt.segment("")
        assert doc.paragraphs == doc.sentences == doc.words == []
        assert doc.to_dict() == {"text": "", "paragraphs": []}

    def test_plain_constructed_nodes_have_empty_children(self):
        assert Paragraph("a", 0, 1).sentences == []
        assert Paragraph("a", 0, 1).words == []
        assert Sentence("a b", 3, 6).words == [Segment("a", 3, 4), Segment("b", 5, 6)]

    def test_pickle_round_trip(self):
        doc = nupunkt.segment(TEXT)
        sent = doc.sentences[0]
        sent.words  # noqa: B018 - populate cache
        restored = pickle.loads(pickle.dumps(sent))
        assert restored == sent
        assert restored.words == sent.words


class TestToDict:
    def test_json_round_trip(self):
        doc = nupunkt.segment(TEXT)
        data = json.loads(json.dumps(doc.to_dict()))
        assert data["text"] == TEXT
        assert [p["text"] for p in data["paragraphs"]] == [p.text for p in doc.paragraphs]
        first = data["paragraphs"][0]["sentences"][0]
        assert (first["text"], first["start"], first["end"]) == tuple(doc.sentences[0])
        assert first["words"][0] == {"text": "Dr.", "start": 2, "end": 5}

    def test_without_words(self):
        doc = nupunkt.segment(TEXT)
        data = doc.to_dict(words=False)
        assert "words" not in data["paragraphs"][0]["sentences"][0]
        assert all(s._words is None for s in doc.sentences)


class TestParagraphBreakRule:
    @pytest.mark.parametrize(
        ("text", "pos", "expected"),
        [
            ("A.\n\nB.", 2, True),
            ("A.   \n\nB.", 2, True),
            ("A.    \n\nB.", 2, False),  # blank line starts too far away
            ("A.\n         \nB.", 2, False),  # second newline outside the window
            ("A.\nB.", 2, False),
            ("A.", 2, False),  # end of text
            ("A. B.", 2, False),
        ],
    )
    def test_rule(self, text, pos, expected):
        assert is_paragraph_break(text, pos) is expected


def test_segment_adaptive_matches_adaptive_sentences():
    text = "She studied at M.I.T. in Cambridge. Dr. Smith agreed.\n\nThe end."
    doc = nupunkt.segment(text, adaptive=True)
    assert [s.text for s in doc.sentences] == nupunkt.sentences(text, adaptive=True)
    assert doc.paragraphs[-1].text == "The end."
    assert all(text[s.start : s.end] == s.text for s in doc.sentences)
