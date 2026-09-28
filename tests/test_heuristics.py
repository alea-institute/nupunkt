"""Tests for deterministic boundary heuristics added on top of the Punkt core."""

import pytest

import nupunkt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Closing punctuation, ASCII and Unicode
        ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
        ("He said “Stop.” Then he left.", ["He said “Stop.”", "Then he left."]),
        ("He said ‘Stop.’ Then he left.", ["He said ‘Stop.’", "Then he left."]),
        ("Il a dit «Stop.» Puis il est parti.", ["Il a dit «Stop.»", "Puis il est parti."]),
        # Quotation continuing the sentence
        ('"Is it?" he asked.', ['"Is it?" he asked.']),
        ('"Go!" she shouted. He went.', ['"Go!" she shouted.', "He went."]),
        # Unicode ellipsis
        ("Wait… Then he left.", ["Wait…", "Then he left."]),
        ("He paused… and continued.", ["He paused… and continued."]),
        # Prenominal titles before a capitalized name
        (
            "Dr. Smith Jr. met Consultant Davis at Acme Inc. yesterday.",
            ["Dr. Smith Jr. met Consultant Davis at Acme Inc. yesterday."],
        ),
        (
            "I saw Gen. Smith. Gen. The general left.",
            ["I saw Gen. Smith.", "Gen. The general left."],
        ),
        # Apostrophes in abbreviations
        ("The ruling was aff'd. by the court.", ["The ruling was aff'd. by the court."]),
        ("Nature’s law applies. It is old.", ["Nature’s law applies.", "It is old."]),
        # Regression checks for classic behaviour
        ("The U.S. Then it rained.", ["The U.S.", "Then it rained."]),
        ("He arrived at 5 p.m. and left.", ["He arrived at 5 p.m. and left."]),
        ("See 42 U.S.C. § 1983. Next sentence.", ["See 42 U.S.C. § 1983.", "Next sentence."]),
    ],
)
def test_sentence_heuristics(text, expected):
    assert nupunkt.sent_tokenize(text) == expected


def test_line_start_enumerators_are_not_sentences():
    text = "The agreement has these parts:\n1. Definitions.\n2. Scope.\n(a) Notice.\nIV. Payment."
    sentences = nupunkt.sent_tokenize(text)
    assert "1." not in sentences
    assert "2." not in sentences
    assert "(a)" not in sentences
    assert "IV." not in sentences
    # "1." is joined to the preceding line because ":" is not a boundary either
    assert sentences == [
        "The agreement has these parts:\n1. Definitions.",
        "2. Scope.",
        "(a) Notice.",
        "IV. Payment.",
    ]


def test_paragraph_break_after_abbreviation_is_a_boundary():
    text = "The motion was denied by the Court of App.\n\nThe appeal followed."
    assert nupunkt.sent_tokenize(text) == [
        "The motion was denied by the Court of App.",
        "The appeal followed.",
    ]


def test_plain_words_are_not_abbreviations():
    # "court", "judge", "law", "case", "trial" used to be in the legal abbreviation list
    text = "The motion was denied by the court. The judge agreed. That is the law. It was a hard case. It went to trial. All done."
    assert nupunkt.sent_tokenize(text) == [
        "The motion was denied by the court.",
        "The judge agreed.",
        "That is the law.",
        "It was a hard case.",
        "It went to trial.",
        "All done.",
    ]


def test_spans_stay_consistent_with_heuristics():
    text = (
        "He said “Stop.” Then he left… Later, Dr. Smith Jr. arrived.\n\n1. Definitions.\n2. Term."
    )
    spans = nupunkt.sent_spans(text)
    sentences = nupunkt.sent_tokenize(text)
    # spans are contiguous and include trailing whitespace
    assert [text[s:e].strip() for s, e in spans] == sentences
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
