"""Regression tests for determinism and runtime parameter changes."""

import subprocess
import sys

import pytest

import nupunkt
from nupunkt import PunktParameters, PunktSentenceTokenizer, PunktTrainer
from nupunkt.core.tokens import PunktToken, create_punkt_token

# Texts whose tokenization previously depended on what had been tokenized before,
# because annotated PunktToken instances were shared between positions and calls.
ORDER_SENSITIVE_TEXTS = [
    "He works at Acme Inc. The company is large.",
    "See Acme Inc. for details.",
    "I met Gen. Smith. Gen. The general left.",
    "Send it to No. 5. No. That is wrong.",
    "He arrived at 5 p.m. Then he left.",
    "The U.S. Then it rained.",
]


def _tokenize_all(order):
    return {i: nupunkt.sent_tokenize(ORDER_SENSITIVE_TEXTS[i]) for i in order}


def test_output_independent_of_call_order():
    forward = _tokenize_all(range(len(ORDER_SENSITIVE_TEXTS)))
    backward = _tokenize_all(reversed(range(len(ORDER_SENSITIVE_TEXTS))))
    assert forward == backward


def test_repeated_calls_are_stable():
    text = "He founded Acme Inc. The firm grew. Acme Inc. is growing fast."
    first = nupunkt.sent_tokenize(text)
    for _ in range(3):
        assert nupunkt.sent_tokenize(text) == first
    assert first == ["He founded Acme Inc.", "The firm grew.", "Acme Inc. is growing fast."]


def test_output_matches_fresh_process():
    """A fresh interpreter must produce the same result as a warm one."""
    text = "See Acme Inc. for details."
    code = (
        "import nupunkt, json; "
        "nupunkt.sent_tokenize('He works at Acme Inc. The company is large.'); "
        f"print(json.dumps(nupunkt.sent_tokenize({text!r})))"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, check=True
    )
    import json

    assert json.loads(out.stdout) == nupunkt.sent_tokenize(text) == [text]


def test_tokens_are_never_shared():
    a = create_punkt_token("Inc.")
    b = create_punkt_token("Inc.")
    assert a is not b
    a.sentbreak = True
    assert b.sentbreak is False
    # derived, immutable fields are still equal
    assert (a.type, a.period_final, a.valid_abbrev_candidate) == (
        b.type,
        b.period_final,
        b.valid_abbrev_candidate,
    )


def test_type_no_sentperiod_tracks_sentbreak():
    token = PunktToken("Inc.")
    assert token.type_no_sentperiod == "inc."
    token.sentbreak = True
    assert token.type_no_sentperiod == "inc"
    token.sentbreak = False
    assert token.type_no_sentperiod == "inc."


class TestRuntimeAbbreviations:
    def test_add_abbreviation_takes_effect_on_loaded_model(self):
        tokenizer = PunktSentenceTokenizer.load(nupunkt.models.get_default_model_path())
        text = "He met Zzq. smith there. Zzq. Smith left."
        assert tokenizer.tokenize(text)[0] == "He met Zzq."
        tokenizer.add_abbreviation("Zzq.")
        assert tokenizer.tokenize(text)[0] == "He met Zzq. smith there."
        tokenizer.remove_abbreviation("zzq")
        assert tokenizer.tokenize(text)[0] == "He met Zzq."

    def test_remove_abbreviation_takes_effect(self):
        tokenizer = PunktSentenceTokenizer.load(nupunkt.models.get_default_model_path())
        text = "Dr. Smith arrived."
        assert tokenizer.tokenize(text) == [text]
        tokenizer.remove_abbreviation("Dr.")
        assert tokenizer.tokenize(text) == ["Dr.", "Smith arrived."]

    def test_in_process_parameters_are_used_directly(self):
        params = PunktParameters()
        params.abbrev_types.add("zzq")
        tokenizer = PunktSentenceTokenizer(params, include_common_abbrevs=False)
        assert tokenizer.tokenize("He met Zzq. smith there.") == ["He met Zzq. smith there."]


class TestConstruction:
    def test_long_training_text_is_not_treated_as_path(self):
        # A run of >255 characters without a separator used to raise OSError
        # (file name too long) from Path.is_file().
        text = "word " * 100 + "x" * 300 + ". Another sentence here."
        tokenizer = PunktSentenceTokenizer(text, include_common_abbrevs=False)
        assert tokenizer.tokenize("First one. Second one.") == ["First one.", "Second one."]

    def test_custom_token_class_is_honoured(self):
        class MyToken(PunktToken):
            __slots__ = ()

        tokenizer = PunktSentenceTokenizer(PunktParameters(), token_cls=MyToken)
        tokens = list(tokenizer._tokenize_words("One two. Three"))
        assert tokens and all(type(t) is MyToken for t in tokens)


class TestTrainerRobustness:
    def test_dunning_log_likelihood_clamps_inconsistent_counts(self):
        from nupunkt.utils.statistics import collocation_log_likelihood, dunning_log_likelihood

        # count_b > N used to raise "math domain error"
        assert dunning_log_likelihood(5, 20, 3, 10) == pytest.approx(
            dunning_log_likelihood(5, 10, 3, 10)
        )
        assert isinstance(collocation_log_likelihood(5, 20, 3, 10), float)

    def test_training_after_training_is_independent(self):
        text_a = "Dr. Smith went home. Dr. Jones stayed. Dr. Lee left. Mr. Roe came."
        text_b = "Hello world. It is Jan. 5 today. Jan. was cold. See Fig. 3 now."
        fresh = PunktTrainer(text_b, include_common_abbrevs=False).get_params()
        PunktTrainer(text_a, include_common_abbrevs=False)
        after = PunktTrainer(text_b, include_common_abbrevs=False).get_params()
        assert fresh.abbrev_types == after.abbrev_types
        assert fresh.sent_starters == after.sent_starters
        assert fresh.collocations == after.collocations


class TestOrthoCompaction:
    def test_compact_ortho_context_preserves_output(self):
        from nupunkt.core.constants import ORTHO_LC, ORTHO_MID_UC

        tokenizer = PunktSentenceTokenizer.load(nupunkt.models.get_default_model_path())
        params = tokenizer._params
        texts = [
            "The U.S. Then it rained. Dr. Smith arrived at 5 p.m. He left.",
            "See Brown v. Board of Educ., 347 U.S. 483 (1954). The Court agreed.",
        ]
        before = [tokenizer.tokenize(t) for t in texts]
        params.ortho_context["zzq"] = ORTHO_MID_UC
        assert params.compact_ortho_context() > 0
        assert "zzq" not in params.ortho_context
        assert all(v & ORTHO_LC for v in params.ortho_context.values())
        assert [tokenizer.tokenize(t) for t in texts] == before

    def test_drop_ortho_context(self):
        params = PunktParameters()
        params.ortho_context["the"] = 112
        params.ortho_context["foo"] = 14
        assert params.drop_ortho_context() == 2
        assert len(params.ortho_context) == 0
        assert params.ortho_context.default_factory is int
        assert params.drop_ortho_context() == 0

    def test_tokenizer_without_ortho_context(self):
        params = PunktParameters.from_json(
            PunktParameters.load(nupunkt.models.get_default_model_path()).to_json()
        )
        params.drop_ortho_context()
        restored = PunktParameters.from_json(params.to_json())
        assert len(restored.ortho_context) == 0
        tokenizer = PunktSentenceTokenizer(restored)
        assert tokenizer.tokenize("Dr. Smith arrived at 5 p.m. He left. It was late.") == [
            "Dr. Smith arrived at 5 p.m.",
            "He left.",
            "It was late.",
        ]
        assert tokenizer.tokenize("See Fig. 3 for details. The end.") == [
            "See Fig. 3 for details.",
            "The end.",
        ]
        assert tokenizer.tokenize("Use e.g. apples. They are good.") == [
            "Use e.g. apples.",
            "They are good.",
        ]


class TestPublicAccessors:
    def test_abbreviations_snapshot(self):
        tokenizer = PunktSentenceTokenizer.load(nupunkt.models.get_default_model_path())
        abbrevs = tokenizer.abbreviations
        assert isinstance(abbrevs, frozenset)
        assert "dr" in abbrevs and "u.s.c" in abbrevs
        tokenizer.add_abbreviation("Zzq.")
        assert "zzq" not in abbrevs  # snapshot
        assert "zzq" in tokenizer.abbreviations

    def test_parameters_is_live(self):
        params = PunktParameters()
        tokenizer = PunktSentenceTokenizer(params, include_common_abbrevs=False)
        assert tokenizer.parameters is params
        tokenizer.parameters.abbrev_types.add("zzq")
        assert tokenizer.tokenize("Met Zzq. smith.") == ["Met Zzq. smith."]
