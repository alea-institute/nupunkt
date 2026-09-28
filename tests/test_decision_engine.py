"""Tests for the string-level boundary decision engine of PunktSentenceTokenizer."""

import pytest

import nupunkt
from nupunkt.core.constants import ORTHO_MID_LC
from nupunkt.core.parameters import PunktParameters
from nupunkt.core.tokens import PunktToken
from nupunkt.hybrid.adaptive_tokenizer import AdaptiveTokenizer
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer


class TokenPathTokenizer(PunktSentenceTokenizer):
    """Reference tokenizer that always decides on PunktToken objects."""

    def _uses_decision_engine(self) -> bool:
        return False


SAMPLES = [
    "Dr. Smith went to Washington. He arrived on Jan. 5. The court agreed.",
    "See 42 U.S.C. § 1983. The statute applies. Cf. Smith v. Jones, 123 F.3d 456.",
    'He said "Stop." Then he left. Is it? he asked. What?! No!! Yes.',
    "Wait for it. . . The end... and more… Then it stopped… later.",
    "1. First item.\n2. Second item.\n\nA. Heading. The text.\n(iv). Roman.",
    "J. R. R. Tolkien wrote books. The U.S. Army won. A. B. Smith, Esq. agreed.",
    "The price was $5.00. It rose to 3.5%. In 1999. Then 2000.",
    "Mr. \nMenendez, Committee on Foreign Relations. Reported by Mr. Smith.",
    "Etc.). Next one (Fig. 3). Also [see id.]. Done.\n\n\nNew paragraph. yes.",
    "a?b. c!d. ?! . . . .. ... x.y.z. Next",
    "",
    "   ",
    "No terminal punctuation here",
    "Mr. Smith. Ok",
    "J. , and 3. ; x. v. Wade. Inc. The",
    "Gen. Grant spoke. Corp. Is next. Ph.D. Candidates. e.g. Apples.",
]


@pytest.fixture(scope="module")
def tokenizers():
    fast = nupunkt.load("default")
    reference = TokenPathTokenizer(fast._params, include_common_abbrevs=False)
    return fast, reference


def test_engine_enabled_for_stock_tokenizer(tokenizers):
    fast, reference = tokenizers
    assert fast._uses_decision_engine()
    assert not reference._uses_decision_engine()


@pytest.mark.parametrize("text", SAMPLES)
def test_engine_matches_token_path(tokenizers, text):
    fast, reference = tokenizers
    assert list(fast.span_tokenize(text)) == list(reference.span_tokenize(text))
    assert fast.text_contains_sentbreak(text) == reference.text_contains_sentbreak(text)
    for _, context in fast._match_potential_end_contexts(text):
        assert fast._decision_state()[3](context) == reference._context_contains_sentbreak(context)


def test_engine_matches_token_path_on_combinations(tokenizers):
    fast, reference = tokenizers
    pieces = ["Mr.", "No.", "v.", "J.", "3.", "iv.", "etc.", "...", "…", "?", "!", '"', ")"]
    words = ["The", "the", "Smith", "and", "\n\nNew"]
    for piece in pieces:
        for word in words:
            text = f"Before {piece} {word} after. End"
            assert list(fast.span_tokenize(text)) == list(reference.span_tokenize(text)), text


def test_adaptive_tokenizer_uses_token_path():
    tok = AdaptiveTokenizer()
    assert not tok._uses_decision_engine()
    assert tok.tokenize("Dr. Smith arrived. He sat down.") == [
        "Dr. Smith arrived.",
        "He sat down.",
    ]


def test_custom_token_class_uses_token_path():
    class MyToken(PunktToken):
        pass

    tok = PunktSentenceTokenizer(nupunkt.load("default")._params, token_cls=MyToken)
    assert not tok._uses_decision_engine()
    assert tok.tokenize("Mr. Smith left. He was late.") == ["Mr. Smith left.", "He was late."]


def _fresh():
    params = PunktParameters.from_json(nupunkt.load("default")._params.to_json())
    return PunktSentenceTokenizer(params, include_common_abbrevs=False)


TEXT = "See Zzq. then more. Also Yyq. next one."


def test_memo_invalidated_by_abbreviation_changes():
    tok = _fresh()
    assert tok.tokenize(TEXT) == ["See Zzq.", "then more.", "Also Yyq.", "next one."]
    tok.add_abbreviation("Zzq.")
    assert tok.tokenize(TEXT) == ["See Zzq. then more.", "Also Yyq.", "next one."]
    tok.add_abbreviations(["yyq"])
    tok.remove_abbreviation("zzq")
    assert tok.tokenize(TEXT) == ["See Zzq.", "then more.", "Also Yyq. next one."]


def test_memo_shared_parameters_are_invalidated():
    first = _fresh()
    second = PunktSentenceTokenizer(first._params, include_common_abbrevs=False)
    assert second.tokenize(TEXT)[0] == "See Zzq."
    first.add_abbreviation("zzq")
    first.remove_abbreviation("not-an-abbreviation")
    assert second.tokenize(TEXT)[0] == "See Zzq. then more."


def test_memo_invalidated_by_direct_parameter_changes():
    tok = _fresh()
    assert tok.tokenize(TEXT)[0] == "See Zzq."
    tok._params.abbrev_types.add("zzq")
    assert tok.tokenize(TEXT)[0] == "See Zzq. then more."
    tok._params = PunktParameters()
    assert tok.tokenize(TEXT)[0] == "See Zzq."


def test_clear_decision_cache_after_in_place_edit():
    tok = _fresh()
    tok.add_abbreviation("zzq")
    text = "See Zzq. Zzqword now. End."
    tok._params.ortho_context["zzqword"] = 0
    assert tok.tokenize(text) == ["See Zzq. Zzqword now.", "End."]
    # Same size, different value (seen in lowercase): only an explicit clear shows it
    tok._params.ortho_context["zzqword"] = ORTHO_MID_LC
    tok.clear_decision_cache()
    reference = TokenPathTokenizer(tok._params, include_common_abbrevs=False)
    assert tok.tokenize(text) == reference.tokenize(text) == ["See Zzq.", "Zzqword now.", "End."]


def test_memo_is_bounded(monkeypatch):
    tok = _fresh()
    monkeypatch.setattr(PunktSentenceTokenizer, "_DECISION_MEMO_SIZE", 4)
    text = " ".join(f"Word{i}. Next{i}" for i in range(20))
    reference = TokenPathTokenizer(tok._params, include_common_abbrevs=False)
    assert tok.tokenize(text) == reference.tokenize(text)
    _, first_memo, context_memo, _ = tok._decision_state()
    assert len(context_memo) <= 4
    assert len(first_memo) <= 4
