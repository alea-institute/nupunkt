"""Tests for per-abbreviation break rates learned from sentence-annotated text."""

import pytest

from nupunkt.core.parameters import PunktParameters
from nupunkt.tokenizers.sentence_tokenizer import PunktSentenceTokenizer
from nupunkt.trainers.base_trainer import PunktTrainer

S = PunktTrainer.SENTENCE_MARKER
P = PunktTrainer.PARAGRAPH_MARKER


def _params(abbrevs=("ltd", "u.s", "etc", "pp"), rates=None, starters=()):
    params = PunktParameters()
    params.abbrev_types = set(abbrevs)
    params.sent_starters = set(starters)
    if rates:
        params.abbrev_break_rates = dict(rates)
    return params


def _trainer(abbrevs=("ltd", "u.s", "etc", "pp")):
    trainer = PunktTrainer(include_common_abbrevs=False)
    trainer._params = _params(abbrevs)
    return trainer


class TestStripSentenceMarkers:
    def test_boundaries_ignore_trailing_whitespace(self):
        text, boundaries = PunktTrainer.strip_sentence_markers(f"One.{S} Two.{S}  Three.")
        assert text == "One. Two.  Three."
        assert boundaries == {4, 9}

    def test_final_marker_and_paragraphs(self):
        text, boundaries = PunktTrainer.strip_sentence_markers(f"One.{S}{P}Two.{S}")
        assert text == "One.\n\nTwo."
        assert boundaries == {4}

    def test_no_markers(self):
        assert PunktTrainer.strip_sentence_markers("Plain text.") == ("Plain text.", set())

    def test_whitespace_only_segment(self):
        _, boundaries = PunktTrainer.strip_sentence_markers(f"One.{S}   {S}Two.")
        assert boundaries == {4}


class TestLearnBreakRates:
    def test_counts_capitalized_followers(self):
        trainer = _trainer()
        learned = trainer.learn_break_rates(
            [
                f"We sell to Acme Ltd.{S} The deal closed.",
                "We sell to Acme Ltd. Group members.",
                f"Paper, pens, etc.{S} Other items.",
            ]
        )
        assert learned == {"ltd": (2, 1), "etc": (1, 1)}
        assert trainer.get_params().abbrev_break_rates == {"ltd": (2, 1), "etc": (1, 1)}

    def test_accumulates_across_calls(self):
        trainer = _trainer()
        trainer.learn_break_rates(f"Acme Ltd.{S} The end.")
        trainer.learn_break_rates("Acme Ltd. Group.")
        assert trainer._params.abbrev_break_rates == {"ltd": (2, 1)}

    def test_ignores_lowercase_digit_paragraph_and_initials(self):
        trainer = _trainer(abbrevs=("ltd", "pp", "j"))
        learned = trainer.learn_break_rates(
            [
                "Acme Ltd. and others.",
                "See pp. 12 of the brief.",
                f"Acme Ltd.{S}{P}New paragraph.",
                "Signed by J. Smith today.",
                "No periods here",
            ]
        )
        assert learned == {}

    def test_follower_on_next_line_counts(self):
        trainer = _trainer()
        learned = trainer.learn_break_rates(f"Acme Ltd.{S}\nThe end.")
        assert learned == {"ltd": (1, 1)}

    def test_train_learns_from_marked_text(self):
        trainer = PunktTrainer(include_common_abbrevs=False)
        trainer._params.abbrev_types.add("etc")
        text = " ".join(f"We bought paper, pens, etc.{S} The store closed." for _ in range(5))
        trainer.train(text, preserve_abbrevs=True)
        params = trainer.get_params()
        assert params.abbrev_break_rates["etc"] == (5, 5)
        # Markers never reach the learned vocabulary
        assert not any("<|" in typ for typ in params.ortho_context)
        assert not any("<|" in typ for typ in params.sent_starters)

    def test_train_verbose_reports_break_counts(self, capsys):
        trainer = PunktTrainer(include_common_abbrevs=False)
        trainer._params.abbrev_types.add("etc")
        trainer.train(f"Paper, pens, etc.{S} The store closed.", verbose=True)
        assert "Learned break counts for 1 abbreviations" in capsys.readouterr().out

    def test_train_without_markers_learns_nothing(self):
        trainer = PunktTrainer(include_common_abbrevs=False)
        trainer.train("We bought paper, pens, etc. The store closed. " * 5)
        assert trainer.get_params().abbrev_break_rates == {}


class TestSerialization:
    def test_round_trip(self):
        params = _params(rates={"ltd": (10, 9), "u.s": (40, 0)})
        data = params.to_json()
        assert data["abbrev_break_rates"] == {"ltd": [10, 9], "u.s": [40, 0]}
        assert PunktParameters.from_json(data).abbrev_break_rates == {
            "ltd": (10, 9),
            "u.s": (40, 0),
        }

    def test_omitted_when_empty(self):
        assert "abbrev_break_rates" not in _params().to_json()

    def test_old_models_load(self):
        data = _params().to_json()
        assert PunktParameters.from_json(data).abbrev_break_rates == {}

    def test_update_accumulates(self):
        params = _params(rates={"ltd": (1, 1)})
        params.update_abbrev_break_rates({"ltd": (2, 0), "etc": (3, 3)})
        assert params.abbrev_break_rates == {"ltd": (3, 1), "etc": (3, 3)}

    def test_tokenizer_json_round_trip(self):
        tok = PunktSentenceTokenizer(_params(rates={"ltd": (30, 27)}))
        loaded = PunktSentenceTokenizer.from_json(tok.to_json())
        assert loaded._params.abbrev_break_rates == {"ltd": (30, 27)}


class TestTokenizerRule:
    TEXT = "The supplier is Smith & Sons Ltd. Their headquarters moved."

    def test_without_rates_abbreviation_does_not_break(self):
        tok = PunktSentenceTokenizer(_params())
        assert tok.tokenize(self.TEXT) == [self.TEXT]

    def test_high_rate_breaks(self):
        tok = PunktSentenceTokenizer(_params(rates={"ltd": (30, 27)}))
        assert tok.tokenize(self.TEXT) == [
            "The supplier is Smith & Sons Ltd.",
            "Their headquarters moved.",
        ]

    @pytest.mark.parametrize("counts", [(10, 10), (30, 21)])
    def test_rare_or_moderate_rate_falls_back(self, counts):
        tok = PunktSentenceTokenizer(_params(rates={"ltd": counts}))
        assert tok.tokenize(self.TEXT) == [self.TEXT]

    def test_thresholds_are_configurable(self):
        tok = PunktSentenceTokenizer(_params(rates={"ltd": (4, 3)}))
        tok.BREAK_RATE_MIN_COUNT = 4
        tok.BREAK_RATE_HIGH = 0.75
        assert len(tok.tokenize(self.TEXT)) == 2

    def test_lowercase_follower_unaffected(self):
        tok = PunktSentenceTokenizer(_params(rates={"ltd": (30, 30)}))
        text = "Smith & Sons Ltd. and others agreed."
        assert tok.tokenize(text) == [text]

    def test_low_rate_is_opt_in(self):
        text = "He was born in the U.S. He later moved."
        rates = {"u.s": (100, 1)}
        tok = PunktSentenceTokenizer(_params(rates=rates, starters=("he",)))
        assert tok.tokenize(text) == ["He was born in the U.S.", "He later moved."]
        tok.BREAK_RATE_LOW = 0.2
        assert tok.tokenize(text) == [text]

    def test_paragraph_break_still_wins(self):
        tok = PunktSentenceTokenizer(_params(rates={"ltd": (100, 0)}))
        tok.BREAK_RATE_LOW = 0.2
        assert len(tok.tokenize("Acme Ltd.\n\nThe next paragraph.")) == 2

    def test_deterministic(self):
        tok = PunktSentenceTokenizer(_params(rates={"ltd": (30, 27)}))
        assert tok.tokenize(self.TEXT) == tok.tokenize(self.TEXT)


def test_learned_rates_drive_tokenizer():
    trainer = _trainer()
    trainer.learn_break_rates(
        [f"The vendor is Acme Ltd.{S} It delivers on time." for _ in range(25)]
    )
    tok = PunktSentenceTokenizer(trainer.get_params())
    assert tok.tokenize("We hired Beta Ltd. Their work was good.") == [
        "We hired Beta Ltd.",
        "Their work was good.",
    ]


class _TokenPath(PunktSentenceTokenizer):
    """Overrides a hook so decisions run on PunktToken objects, not the string engine."""

    def _is_sent_starter(self, token_type):
        return super()._is_sent_starter(token_type)


@pytest.mark.parametrize("low", [None, 0.2])
def test_string_engine_matches_token_path(low):
    params = _params(rates={"ltd": (10, 9), "u.s": (100, 1), "etc": (10, 5)}, starters=("he",))
    text = (
        "Acme Ltd. Their office is closed. He was born in the U.S. He moved. "
        "Pens, etc. Other things. Acme Ltd. and Beta Ltd.\n\nNew paragraph."
    )
    engine, token_path = PunktSentenceTokenizer(params), _TokenPath(params)
    for tok in (engine, token_path):
        tok.BREAK_RATE_LOW = low
    assert engine.tokenize(text) == token_path.tokenize(text)


def test_threshold_change_invalidates_decision_cache():
    tok = PunktSentenceTokenizer(_params(rates={"ltd": (30, 27)}))
    text = "The supplier is Smith & Sons Ltd. Their headquarters moved."
    assert len(tok.tokenize(text)) == 2
    tok.BREAK_RATE_HIGH = 0.95
    assert len(tok.tokenize(text)) == 1
