#!/usr/bin/env python3
"""Re-run publicly reported nupunkt failure cases and print Markdown tables.

Every case is an input that someone reported publicly as a nupunkt failure or
complaint (or a close construction, marked as such), with the expected output
and a link to the source. Each (case, library, mode) runs in its own fresh
Python process, because nupunkt 0.6.0 is history-dependent: the same input can
split differently depending on what the process tokenized before.

Usage::

    # 0.7.0 only (whatever `nupunkt` the running interpreter imports)
    python scripts/benchmarks/known_cases.py

    # add a 0.6.0 column (a venv with `pip install nupunkt==0.6.0`)
    python scripts/benchmarks/known_cases.py --compare-06 /path/to/venv06/bin/python

    # add NLTK punkt_tab and pysbd columns (a venv with nltk + pysbd; set
    # NLTK_DATA if punkt_tab is not in a default location)
    python scripts/benchmarks/known_cases.py --compare-others /path/to/venvcmp/bin/python

Standard library only; the libraries under test are imported in child processes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT = str(Path(__file__).resolve())

# --------------------------------------------------------------------------- sources

SOURCES: dict[str, tuple[str, str]] = {
    "scout": (
        "abraham-jacob/scout#30 (nupunkt output posted by speedyk-005)",
        "https://github.com/abraham-jacob/scout/issues/30#issuecomment-5287859994",
    ),
    "scout-yasbd": (
        "abraham-jacob/scout#30 (reported against yasbd, same thread)",
        "https://github.com/abraham-jacob/scout/issues/30#issuecomment-5262960213",
    ),
    "scout-yasbd2": (
        "abraham-jacob/scout#30 (speedyk-005, cases yasbd must still split)",
        "https://github.com/abraham-jacob/scout/issues/30#issuecomment-5267951914",
    ),
    "scout-yasbd3": (
        "abraham-jacob/scout#30 (abraham-jacob sanity check)",
        "https://github.com/abraham-jacob/scout/issues/30#issuecomment-5285710323",
    ),
    "yasbd": (
        "yasbd-lib benchmarks README (2026-09-25)",
        "https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/README.md",
    ),
    "yasbd-golden": (
        "yasbd-lib EN_GOLDEN_DATA.py",
        "https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/EN_GOLDEN_DATA.py",
    ),
    "yasbd-pt": (
        "speedyk-005/yasbd-lib#30 (Portuguese comparison)",
        "https://github.com/speedyk-005/yasbd-lib/issues/30#issuecomment-4639723570",
    ),
    "brief": ("maintainer complaint list (verbatim source not located, see notes)", ""),
    "nltk2082": ("nltk/nltk#2082", "https://github.com/nltk/nltk/issues/2082"),
    "nltk3370": ("nltk/nltk#3370", "https://github.com/nltk/nltk/issues/3370"),
    "nltk2892": ("nltk/nltk#2892", "https://github.com/nltk/nltk/issues/2892"),
    "nltk2154": ("nltk/nltk#2154", "https://github.com/nltk/nltk/issues/2154"),
    "nltk60": ("nltk/nltk#60", "https://github.com/nltk/nltk/issues/60"),
    "knowseams": (
        "KnowSeams README (@SteveJSteiner)",
        "https://github.com/KnowSeams/KnowSeams/blob/b0f2d29597aec4025d1b5a35829fe6c6878c24e8/README.md",
    ),
    "constructed": ("constructed for this page", ""),
    "research": ("nupunkt 0.7.0 development notes", ""),
}

# --------------------------------------------------------------------------- cases


@dataclass(frozen=True)
class Case:
    cid: str
    text: str
    expected: tuple[str, ...]
    source: str
    note: str = ""
    # Other segmentations that are also acceptable (convention differences).
    accept: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


SCOUT_BLOCK = """
Acme Inc. USA is expanding its engineering team this quarter.
Beta Corp. North America leads this hiring initiative.
Martin Luther King Jr. Day is a paid holiday at this company.
John Doe Sr. VP of Engineering will be your hiring manager.
Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday alongside
Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements,
considering e.g. U.S. policies, U.K. guidelines, etc. Johnson outlined the roadmap.
"""

WITNESS = (
    "The witness testified: \"He said — and I quote — 'I will not comply.' "
    "Then he turned around and left. I couldn't believe it.\""
)

KNOWSEAMS = (
    '"Well, you can see him easily enough," said Mr. Hoad. "He\'s staying in\n'
    "your village, I believe. He's a nephew of Squire Broderick's.\"\n\n"
    '"What! Captain Forrester?" cried I.'
)

MUFFIN = (
    "Do you like the muffin? No. Are you sure you don't? But this is call the neutrino. "
    "Not an inferno. Perhaps a neutri no. Maybe an infer no. Conseula likes to say no. "
    "And Jim carrey is the yes man."
)

THEMES: list[tuple[str, str, list[Case]]] = [
    (
        "abbrev",
        "Abbreviations before a capitalized word",
        [
            Case(
                "A1",
                SCOUT_BLOCK,
                (
                    "Acme Inc. USA is expanding its engineering team this quarter.",
                    "Beta Corp. North America leads this hiring initiative.",
                    "Martin Luther King Jr. Day is a paid holiday at this company.",
                    "John Doe Sr. VP of Engineering will be your hiring manager.",
                    "Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday alongside "
                    "Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree "
                    "standards vs. Ph.D. requirements, considering e.g. U.S. policies, U.K. "
                    "guidelines, etc.",
                    "Johnson outlined the roadmap.",
                ),
                "scout",
                "Full passage from the thread. The posted nupunkt output split at `Sr.` and "
                "missed `etc. Johnson`.",
            ),
            Case(
                "A2",
                "Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday.",
                ("Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday.",),
                "scout",
                "Shortened from A1.",
            ),
            Case(
                "A3",
                "John Doe Sr. VP of Engineering will be your hiring manager.",
                ("John Doe Sr. VP of Engineering will be your hiring manager.",),
                "scout",
            ),
            Case(
                "A4",
                "This role requires U.S. Government security clearance.",
                ("This role requires U.S. Government security clearance.",),
                "scout-yasbd",
            ),
            Case(
                "A5",
                "This job is open to U.S. Persons only per export law.",
                ("This job is open to U.S. Persons only per export law.",),
                "scout-yasbd",
            ),
            Case(
                "A6",
                "This policy was adopted in the U.S. Next week we will review it.",
                ("This policy was adopted in the U.S.", "Next week we will review it."),
                "scout-yasbd2",
            ),
            Case(
                "A7",
                "I live in the E.U. How about you?",
                ("I live in the E.U.", "How about you?"),
                "scout-yasbd2",
            ),
            Case(
                "A8",
                "We are Acme Inc. Our team is fully remote.",
                ("We are Acme Inc.", "Our team is fully remote."),
                "scout-yasbd3",
            ),
            Case(
                "A9",
                "John saw Mary in the U.S.A.  John saw Mary in the U.S.A. today.",
                ("John saw Mary in the U.S.A.", "John saw Mary in the U.S.A. today."),
                "nltk60",
            ),
            Case(
                "A10",
                "He was a Ph.D.: that meant studying a lot.",
                ("He was a Ph.D.: that meant studying a lot.",),
                "nltk3370",
            ),
            Case(
                "A11",
                "She is an M.D.: an oncologist. Please assist me.",
                ("She is an M.D.: an oncologist.", "Please assist me."),
                "nltk3370",
            ),
            Case(
                "A12",
                "It was the Ph.D.'s responsibility to perform research.",
                ("It was the Ph.D.'s responsibility to perform research.",),
                "nltk3370",
            ),
            Case(
                "A13",
                "We use a sentence tokenizer, i.e. a function to transform a string with one or "
                "more sentences into a list of individual sentences, etc. and it often succeeds.",
                (
                    "We use a sentence tokenizer, i.e. a function to transform a string with one "
                    "or more sentences into a list of individual sentences, etc. and it often "
                    "succeeds.",
                ),
                "nltk2892",
                "From a comment on the issue.",
            ),
            Case(
                "A14",
                "Jersey no. 5 is the best. But I don't like no. 6.",
                ("Jersey no. 5 is the best.", "But I don't like no. 6."),
                "nltk2154",
            ),
            Case(
                "A15",
                MUFFIN,
                (
                    "Do you like the muffin?",
                    "No.",
                    "Are you sure you don't?",
                    "But this is call the neutrino.",
                    "Not an inferno.",
                    "Perhaps a neutri no.",
                    "Maybe an infer no.",
                    "Conseula likes to say no.",
                    "And Jim carrey is the yes man.",
                ),
                "nltk2154",
                "The counter-example from the issue: `no.` that really ends a sentence.",
            ),
            Case(
                "A16",
                "The appeal is App. No. 5 in this court.",
                ("The appeal is App. No. 5 in this court.",),
                "research",
            ),
            Case(
                "A17",
                "She returned to the U.S. Then it rained.",
                ("She returned to the U.S.", "Then it rained."),
                "research",
            ),
            Case(
                "A18",
                "The contract was signed by Global Ltd. Their lawyers reviewed it.",
                ("The contract was signed by Global Ltd.", "Their lawyers reviewed it."),
                "research",
            ),
        ],
    ),
    (
        "numbers",
        "Numbers, enumerators and years",
        [
            Case(
                "N1",
                "This Agreement contains the following sections. 1. Definitions. 2. Term. "
                "3. Termination.",
                (
                    "This Agreement contains the following sections.",
                    "1. Definitions.",
                    "2. Term.",
                    "3. Termination.",
                ),
                "brief",
            ),
            Case(
                "N2",
                "1. Submit form. 2. Pay fee. 3. Wait for approval.",
                ("1. Submit form.", "2. Pay fee.", "3. Wait for approval."),
                "brief",
            ),
            Case(
                "N3",
                "1. Definitions.\n2. Term.\n3. Termination.",
                ("1. Definitions.", "2. Term.", "3. Termination."),
                "constructed",
                "N1 as a line-start list.",
            ),
            Case(
                "N4",
                "1. The first item. 2. The second item.",
                ("1. The first item.", "2. The second item."),
                "yasbd-golden",
            ),
            Case(
                "N5",
                "Did you remove num 2. Put it back.",
                ("Did you remove num 2.", "Put it back."),
                "yasbd-golden",
            ),
            Case(
                "N6",
                "The company was founded in 2013. It has grown every year since.",
                ("The company was founded in 2013.", "It has grown every year since."),
                "nltk2892",
                "The issue describes `2013.` read as an ordinal; the sentence is ours.",
            ),
        ],
    ),
    (
        "dialog",
        "Quotes, dialog and terminator runs",
        [
            Case(
                "D1",
                KNOWSEAMS,
                (
                    '"Well, you can see him easily enough," said Mr. Hoad.',
                    "\"He's staying in your village, I believe.",
                    "He's a nephew of Squire Broderick's.\"",
                    '"What! Captain Forrester?" cried I.',
                ),
                "knowseams",
                'Reported: nupunkt splits `"What!` / `Captain Forrester?"` / `cried I.`. '
                "SEAMS' own 3-unit output (quote kept whole) is also accepted.",
                accept=(
                    (
                        '"Well, you can see him easily enough," said Mr. Hoad.',
                        "\"He's staying in your village, I believe. He's a nephew of Squire "
                        "Broderick's.\"",
                        '"What! Captain Forrester?" cried I.',
                    ),
                ),
            ),
            Case(
                "D2",
                WITNESS,
                (
                    "The witness testified: \"He said — and I quote — 'I will not comply.'",
                    "Then he turned around and left.",
                    "I couldn't believe it.\"",
                ),
                "yasbd",
                "Convention: yasbd prefers the whole quotation as one unit; both accepted.",
                accept=((WITNESS,),),
            ),
            Case(
                "D3",
                '"Is it?" he asked. "Yes," she said.',
                ('"Is it?" he asked.', '"Yes," she said.'),
                "research",
            ),
            Case(
                "D4",
                '"Really?!" asked Tom. "Yes."',
                ('"Really?!" asked Tom.', '"Yes."'),
                "constructed",
                "Capitalized attribution after `?!` (same pattern as D1).",
            ),
            Case(
                "D5",
                '"I am leaving." He closed the door.',
                ('"I am leaving."', "He closed the door."),
                "constructed",
            ),
            Case(
                "D6",
                "She said, \"He told me 'Run!' and I did.\" Then silence.",
                ("She said, \"He told me 'Run!' and I did.\"", "Then silence."),
                "constructed",
                "Nested quotes.",
            ),
            Case(
                "D7",
                '"Well—I suppose—yes," he said. She nodded.',
                ('"Well—I suppose—yes," he said.', "She nodded."),
                "constructed",
                "Em-dash interruptions.",
            ),
            Case(
                "D8",
                'He said—and I quote—"No." Then he left.',
                ('He said—and I quote—"No."', "Then he left."),
                "constructed",
            ),
            Case(
                "D9",
                "“Stop.” Then he left.",
                ("“Stop.”", "Then he left."),
                "research",
                "Unicode closing quote.",
            ),
            Case(
                "D10",
                "“Is it?” he asked.",
                ("“Is it?” he asked.",),
                "research",
            ),
            Case(
                "D11",
                "I waited… Then it happened.",
                ("I waited…", "Then it happened."),
                "research",
                "Unicode ellipsis.",
            ),
            Case(
                "D12",
                "No way!!! I can't believe it.",
                ("No way!!!", "I can't believe it."),
                "research",
            ),
            Case(
                "D13",
                "what even is this. broh !! \nthat is so sad.",
                ("what even is this.", "broh !!", "that is so sad."),
                "yasbd",
                "README: nupunkt is 'over-aggressive on double exclamation marks "
                "(`broh !`, `!`)'. Excerpt of the chat-log input.",
            ),
        ],
    ),
    (
        "scope",
        "Outside Punkt's input model: no space after the period, CJK, other languages",
        [
            Case(
                "S1",
                "Mary had little lamb.Mary had a little lamb",
                ("Mary had little lamb.", "Mary had a little lamb"),
                "nltk2082",
            ),
            Case(
                "S2",
                "今日はいい天気ですね。明日から雨が降るそうです。"
                "外出するなら傘を持って行ったほうがいいでしょう。\n"
                "「すみません、駅はどちらですか？」と観光客が聞いた。",
                (
                    "今日はいい天気ですね。",
                    "明日から雨が降るそうです。",
                    "外出するなら傘を持って行ったほうがいいでしょう。",
                    "「すみません、駅はどちらですか？」と観光客が聞いた。",
                ),
                "yasbd",
                "First four sentences of the README's 26-sentence Japanese passage.",
            ),
            Case(
                "S3",
                "Le manuel, c.-à-d. la version complète, a été publié après une longue m.-à-j. "
                "du système interne. Le rendez-vous, noté R.-V. dans le dossier administratif, "
                "a été déplacé à 14 h. après une d.-h. d'attente.",
                (
                    "Le manuel, c.-à-d. la version complète, a été publié après une longue "
                    "m.-à-j. du système interne.",
                    "Le rendez-vous, noté R.-V. dans le dossier administratif, a été déplacé à "
                    "14 h. après une d.-h. d'attente.",
                ),
                "yasbd",
                "First line of the README's French passage.",
            ),
            Case(
                "S4",
                "Segundo o acórdão do tribunal (vid. capítulo II, parágrafo 3º, item B), a "
                "diretoria executiva da Beta S.A. descumpriu deliberadamente o of. n.º 89/2026 "
                "emitido pela presidência da assoc. comercial.",
                (
                    "Segundo o acórdão do tribunal (vid. capítulo II, parágrafo 3º, item B), a "
                    "diretoria executiva da Beta S.A. descumpriu deliberadamente o of. n.º "
                    "89/2026 emitido pela presidência da assoc. comercial.",
                ),
                "yasbd-pt",
                "One sentence of the Portuguese comparison.",
            ),
        ],
    ),
]

ALL_CASES = [c for _, _, cases in THEMES for c in cases]

# Layout cases: (id, text, expected, apply blank_page_furniture first, note)
LAYOUT_CASES: list[tuple[str, str, tuple[str, ...], bool, str]] = [
    (
        "L1",
        "INTRODUCTION\n\nThe court held for the plaintiff. It awarded fees.",
        ("INTRODUCTION", "The court held for the plaintiff.", "It awarded fees."),
        False,
        "Heading, blank line, sentence.",
    ),
    (
        "L2",
        "The court held that the\n\ndefendant had waived the claim. It ruled.",
        ("The court held that the defendant had waived the claim.", "It ruled."),
        False,
        "Sentence split across a blank line; the next block starts lowercase.",
    ),
    (
        "L3",
        "The defendant waived the claim under the doc-\n\ntrine of laches. The court agreed.",
        ("The defendant waived the claim under the doc- trine of laches.", "The court agreed."),
        False,
        "Hyphenated word across a page break (no de-hyphenation is done).",
    ),
    (
        "L4",
        "The court held that the\n\n12\n\ndefendant had waived the claim.",
        ("The court held that the defendant had waived the claim.",),
        False,
        "Page number between the halves of a sentence.",
    ),
    (
        "L5",
        "The court held that the\n\n12\n\ndefendant had waived the claim.",
        ("The court held that the defendant had waived the claim.",),
        True,
        "L4 after `blank_page_furniture(text)` (same length, page number blanked).",
    ),
    (
        "L6",
        "The parties agree:\n1. Payment is due monthly\n2. Notice must be written\n"
        "3. Disputes go to arbitration",
        (
            "The parties agree:",
            "1. Payment is due monthly",
            "2. Notice must be written",
            "3. Disputes go to arbitration",
        ),
        False,
        "Numbered list, one item per line, no terminal punctuation (items as units).",
    ),
]

# --------------------------------------------------------------------------- worker side


def _split(lib: str, text: str) -> list[str]:
    if lib == "nupunkt":
        import nupunkt

        return list(nupunkt.sent_tokenize(text))
    if lib == "nupunkt-adaptive":
        import nupunkt

        return list(nupunkt.sent_tokenize(text, adaptive=True))
    if lib == "nltk":
        from nltk.tokenize import sent_tokenize

        return list(sent_tokenize(text))
    if lib == "pysbd":
        import pysbd

        return list(pysbd.Segmenter(language="en", clean=False).segment(text))
    raise ValueError(lib)


def _version(lib: str) -> str:
    try:
        if lib.startswith("nupunkt"):
            import nupunkt

            return f"nupunkt {nupunkt.__version__}"
        if lib == "nltk":
            import nltk

            return f"nltk {nltk.__version__} punkt_tab"
        if lib == "pysbd":
            import pysbd

            return f"pysbd {pysbd.__version__}"
    except Exception as e:  # pragma: no cover - reported, not raised
        return f"{lib} ({type(e).__name__})"
    return lib


def _demo(name: str) -> None:
    """Print demo output. Everything printed here is reproduced verbatim in the doc."""
    import nupunkt

    def show(label: str, fn) -> None:
        try:
            print(f"{label} -> {fn()!r}")
        except Exception as e:
            print(f"{label} -> {type(e).__name__}: {e}")

    print(f"# nupunkt {nupunkt.__version__}")
    if name == "api":
        t = "First sentence.  Second one.\n\nNew paragraph here."
        show("sent_tokenize(t)", lambda: nupunkt.sent_tokenize(t))
        show(
            "sent_tokenize(t, return_confidence=True)",
            lambda: nupunkt.sent_tokenize(t, return_confidence=True),
        )
        show(
            "sent_tokenize(t, adaptive=True, return_confidence=True)",
            lambda: nupunkt.sent_tokenize(t, adaptive=True, return_confidence=True),
        )
        show("para_tokenize(t)", lambda: nupunkt.para_tokenize(t))
        show("paragraphs(t)", lambda: nupunkt.paragraphs(t))
        show(
            "[[s.text for s in p.sentences] for p in segment(t).paragraphs]",
            lambda: [[s.text for s in p.sentences] for p in nupunkt.segment(t).paragraphs],
        )
    elif name == "spans":
        t = "  First sentence.  Second one.\n\nThird. "
        show("sent_spans(t)", lambda: nupunkt.sent_spans(t))
        show(
            "[t[a:b] for a, b in sent_spans(t)]", lambda: [t[a:b] for a, b in nupunkt.sent_spans(t)]
        )
        show("sentence_spans(t)", lambda: nupunkt.sentence_spans(t))
        show(
            "[t[a:b] for a, b in sentence_spans(t)]",
            lambda: [t[a:b] for a, b in nupunkt.sentence_spans(t)],
        )
    elif name == "train":
        import itertools
        import string

        from nupunkt.training import train_model

        # Same call shape as institutional-books-enriched-text-pipeline:
        # train_model(book_text, abbreviations=<base model abbrevs>, output_path=None).
        # Degenerate "book": 10,050 one-word sentences, each word seen once, so the
        # memory-efficient trainer (default) prunes them at its 10,000-token interval.
        names = ("".join(p) for p in itertools.product(string.ascii_lowercase, repeat=3))
        corpus = " ".join(f"W{n}." for n in itertools.islice(names, 10_050)) + " Yes. Yes. Yes."
        abbrevs = sorted(nupunkt.load("default")._params.abbrev_types)
        show(
            "len(train_model(corpus, abbreviations=abbrevs, output_path=None)"
            ".get_params().abbrev_types)",
            lambda: len(
                train_model(corpus, abbreviations=abbrevs, output_path=None)
                .get_params()
                .abbrev_types
            ),
        )
    elif name == "abbrevs":
        tok = nupunkt.load("default")
        x = "The shipment went to Acme Wdg. Holdings last week."
        show("tok.tokenize(x)", lambda: tok.tokenize(x))
        show(
            "tok.add_abbreviation('Wdg.'); tok.tokenize(x)",
            lambda: (tok.add_abbreviation("Wdg."), tok.tokenize(x))[1],
        )
        show("'wdg' in tok.abbreviations", lambda: "wdg" in tok.abbreviations)
        show(
            "'wdg' in tok._params.abbrev_types  # private",
            lambda: "wdg" in tok._params.abbrev_types,
        )
        show(
            "tok.remove_abbreviation('Wdg'); tok.tokenize(x)",
            lambda: (tok.remove_abbreviation("Wdg"), tok.tokenize(x))[1],
        )
    elif name == "probes":
        # Why some still-open cases fail: vary one factor at a time.
        params = nupunkt.load("default")._params
        for w in ("term", "sr", "ltd", "no"):
            show(f"{w!r} in abbrev_types", lambda w=w: w in params.abbrev_types)
        show("'consultant' in sent_starters", lambda: "consultant" in params.sent_starters)
        for t in (
            "1. Definitions.\n2. Term.\n3. Termination.",
            "1. Definitions.\n2. Scope.\n3. Termination.",
            "He met Sr. Consultant Davis.",
            "He met Dr. Consultant Davis.",
            'He said "No." Then he left.',
            'He said "Yes." Then he left.',
            "'Do not follow me.' Then he left.",
            'He moved to the "U.S." He stayed.',
            "I waited… I left.",
            "Hello ! ! ! ! How are you?",
        ):
            show(f"sent_tokenize({t!r})", lambda t=t: nupunkt.sent_tokenize(t))
    else:
        raise ValueError(name)


def worker() -> None:
    # Import the libraries under test from the interpreter's environment, never from cwd.
    sys.path[:] = [x for x in sys.path if x not in ("", ".")]
    req = json.load(sys.stdin)
    op = req["op"]
    buf = io.StringIO()
    result: dict = {}
    if op == "demo":
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            _demo(req["name"])
        print(json.dumps({"text": buf.getvalue()}))
        return
    with contextlib.redirect_stdout(buf):
        lib = req["lib"]
        result["version"] = _version(lib)
        try:
            if op == "split":
                out = _split(lib, req["text"])
                result["out"] = out
                result["idempotent"] = all(
                    [" ".join(s.split())] == [" ".join(x.split()) for x in _split(lib, s)]
                    for s in out
                )
            elif op == "layout":
                import nupunkt

                t = req["text"]
                if req["furniture"]:
                    t = nupunkt.blank_page_furniture(t)
                calls = {
                    "sent_tokenize": lambda: nupunkt.sent_tokenize(t),
                    "sentences": lambda: nupunkt.sentences(t),
                    "sentences_lb": lambda: nupunkt.sentences(t, line_breaks=True),
                }
                result["outs"] = {}
                for k, fn in calls.items():
                    try:
                        result["outs"][k] = {"out": list(fn())}
                    except Exception as e:
                        result["outs"][k] = {"error": f"{type(e).__name__}: {e}"}
            elif op == "seq":  # several inputs in one process, in order
                result["outs"] = [_split(lib, t) for t in req["texts"]]
            else:
                raise ValueError(op)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(result))


# --------------------------------------------------------------------------- driver side


def run(python: str, req: dict) -> dict:
    p = subprocess.run(
        [python, SCRIPT, "--worker"],
        input=json.dumps(req),
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0 or not p.stdout.strip():
        return {"error": f"worker failed: {p.stderr.strip().splitlines()[-1:]}"}
    return json.loads(p.stdout.strip().splitlines()[-1])


def norm(sents) -> tuple[str, ...]:
    return tuple(" ".join(s.split()) for s in sents if s.strip())


def passes(case: Case, res: dict) -> bool | None:
    if "out" not in res:
        return None
    got = norm(res["out"])
    return any(got == norm(e) for e in (case.expected, *case.accept))


def cell(sents) -> str:
    if sents is None:
        return "—"
    parts = [" ".join(s.split()).replace("|", "\\|") for s in sents if s.strip()]
    return "<br>".join(f"`{p}`" for p in parts) or "(empty)"


def mark(ok: bool | None) -> str:
    return {True: "ok", False: "**FAIL**", None: "error"}[ok]


def verdict(p06: bool | None, p07: bool | None) -> str:
    if p06 is None:
        return "pass" if p07 else "**still fails**"
    return {
        (False, True): "**fixed**",
        (True, True): "unchanged (pass)",
        (False, False): "**still fails**",
        (True, False): "**REGRESSED**",
    }.get((bool(p06), bool(p07)), "error")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--python07", default=sys.executable, help="interpreter for nupunkt 0.7.x")
    ap.add_argument("--compare-06", metavar="VENV_PYTHON", help="interpreter with nupunkt 0.6.0")
    ap.add_argument(
        "--compare-main",
        metavar="PYTHON",
        help="interpreter whose nupunkt is the unreleased main tree (e.g. .venv/bin/python)",
    )
    ap.add_argument("--compare-others", metavar="VENV_PYTHON", help="interpreter with nltk+pysbd")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.worker:
        worker()
        return

    cols: list[tuple[str, str, str]] = []  # (key, python, lib)
    if args.compare_06:
        cols.append(("06", args.compare_06, "nupunkt"))
    cols += [("07", args.python07, "nupunkt"), ("07a", args.python07, "nupunkt-adaptive")]
    if args.compare_main:
        cols.append(("main", args.compare_main, "nupunkt"))
    if args.compare_others:
        cols += [("nltk", args.compare_others, "nltk"), ("pysbd", args.compare_others, "pysbd")]

    jobs = [(c, key, py, lib) for c in ALL_CASES for key, py, lib in cols]
    with ThreadPoolExecutor(args.jobs) as ex:
        outs = list(
            ex.map(lambda j: run(j[2], {"op": "split", "lib": j[3], "text": j[0].text}), jobs)
        )
    res = {(c.cid, key): r for (c, key, _, _), r in zip(jobs, outs, strict=True)}
    versions = {key: res[(ALL_CASES[0].cid, key)].get("version", "?") for key, _, _ in cols}
    headers = {
        "06": f"{versions.get('06', '')}",
        "07": f"{versions['07']} default",
        "07a": f"{versions['07a']} adaptive",
        "main": "nupunkt main (unreleased) default",
        "nltk": versions.get("nltk", ""),
        "pysbd": versions.get("pysbd", ""),
    }

    p = print
    p("<!-- generated by scripts/benchmarks/known_cases.py -->")
    p("Versions: " + "; ".join(f"`{headers[k]}`" for k, _, _ in cols) + ".\n")
    tally: dict[str, int] = {}
    tally_main: dict[str, int] = {}
    fails: list[str] = []
    fails_main: list[str] = []
    has_main = any(k == "main" for k, _, _ in cols)
    for _, title, cases in THEMES:
        p(f"### {title}\n")
        keys = [k for k, _, _ in cols]
        extra = " | 0.7.0 to main |" if has_main else " |"
        p("| # | Input | Expected | " + " | ".join(headers[k] for k in keys) + " | Verdict" + extra)
        p("|---" * (len(keys) + 4 + has_main) + "|")
        notes = []
        for c in cases:
            src_label, url = SOURCES[c.source]
            src = f"[{c.cid}]({url})" if url else c.cid
            row = [src, cell([c.text]), cell(c.expected)]
            for k in keys:
                r = res[(c.cid, k)]
                ok = passes(c, r)
                body = cell(r.get("out")) if "out" in r else f"`{r.get('error')}`"
                row.append(f"{mark(ok)}<br>{body}" if k in ("06", "07", "07a", "main") else body)
            v = verdict(
                passes(c, res[(c.cid, "06")]) if "06" in keys else None,
                passes(c, res[(c.cid, "07")]),
            )
            if (
                "06" in keys
                and v == "**still fails**"
                and norm(res[(c.cid, "06")].get("out", []))
                != norm(res[(c.cid, "07")].get("out", []))
            ):
                v += " (output changed)"
            if passes(c, res[(c.cid, "07a")]) and not passes(c, res[(c.cid, "07")]):
                v += " (adaptive ok)"
            tkey = v.replace(" (adaptive ok)", "").replace("*", "")
            tally[tkey] = tally.get(tkey, 0) + 1
            if not passes(c, res[(c.cid, "07")]):
                fails.append(c.cid)
            row.append(v)
            if has_main:
                p07, pm = passes(c, res[(c.cid, "07")]), passes(c, res[(c.cid, "main")])
                vm = {
                    (False, True): "**fixed on main**",
                    (True, True): "pass",
                    (True, False): "**REGRESSED on main**",
                }.get((bool(p07), bool(pm)), "still fails")
                if vm == "still fails" and norm(res[(c.cid, "07")].get("out", [])) != norm(
                    res[(c.cid, "main")].get("out", [])
                ):
                    vm += " (output changed)"
                tally_main[vm.replace("*", "")] = tally_main.get(vm.replace("*", ""), 0) + 1
                if not pm:
                    fails_main.append(c.cid)
                row.append(vm)
            p("| " + " | ".join(row) + " |")
            notes.append(f"- **{c.cid}** — {src_label}." + (f" {c.note}" if c.note else ""))
        p("\n" + "\n".join(notes) + "\n")

    p("### Summary\n")
    p("| Verdict (0.6.0 to 0.7.0 default) | Cases |\n|---|---|")
    for k, n in sorted(tally.items()):
        p(f"| {k} | {n} |")
    p(f"\n0.7.0 default does not match the expected output on: {', '.join(fails) or 'none'}.\n")
    if has_main:
        p("| Verdict (0.7.0 to main) | Cases |\n|---|---|")
        for k, n in sorted(tally_main.items()):
            p(f"| {k} | {n} |")
        p(f"\nmain does not match the expected output on: {', '.join(fails_main) or 'none'}.\n")
    idem = {
        k: [c.cid for c in ALL_CASES if res[(c.cid, k)].get("idempotent") is False]
        for k, _, _ in cols
    }
    p(
        "Idempotency: cases where re-splitting one of the output sentences splits it again: "
        + "; ".join(f"{headers[k]}: {', '.join(v) or 'none'}" for k, v in idem.items())
        + ".\n"
    )

    lay_cols = [(k, py) for k, py, lib in cols if k in ("07", "main")]
    if lay_cols:
        p("### Layout and page breaks\n")
        p(
            "`sent_tokenize` is Punkt only. `sentences()` uses the segmentation interface "
            "(layout-aware by default on main, see `docs/layout.md`).\n"
        )
        lcols = [("main", "sent_tokenize")] if has_main else [("07", "sent_tokenize")]
        lcols += [(k, "sentences") for k, _ in lay_cols]
        if has_main:
            lcols.append(("main", "sentences_lb"))
        label = {
            "sent_tokenize": "`sent_tokenize`",
            "sentences": "`sentences()`",
            "sentences_lb": "`sentences(line_breaks=True)`",
        }
        short = {"07": versions["07"], "main": "main"}
        p(
            "| # | Input | Expected | "
            + " | ".join(f"{short[k]} {label[f]}" for k, f in lcols)
            + " |"
        )
        p("|---" * (len(lcols) + 3) + "|")
        py_of = dict(lay_cols)
        lnotes = []
        for cid, text, exp, furn, note in LAYOUT_CASES:
            got = {
                k: run(
                    py_of[k], {"op": "layout", "lib": "nupunkt", "text": text, "furniture": furn}
                )
                for k in py_of
            }
            row = [cid, cell([repr(text)[1:-1]]), cell(exp)]
            for k, f in lcols:
                r = got[k].get("outs", {}).get(f, got[k])
                if "out" in r:
                    ok = norm(r["out"]) == norm(exp)
                    row.append(f"{mark(ok)}<br>{cell([repr(x)[1:-1] for x in r['out']])}")
                else:
                    row.append(f"`{r.get('error')}`")
            p("| " + " | ".join(row) + " |")
            lnotes.append(f"- **{cid}** — constructed. {note}")
        p("\n" + "\n".join(lnotes) + "\n")
        p("Layout cells show `\\n` literally; the pass check collapses whitespace.\n")

    # Determinism: same input, fresh process vs after one earlier input.
    a = "He works at Acme Inc. The company is large."
    b = "See Acme Inc. for details."
    p("### Determinism: output depends on earlier calls\n")
    p(f"Input A: `{a}`  Input B: `{b}`. Each line is a fresh process.\n")
    p("```")
    for key, py, lib in cols:
        if lib != "nupunkt":
            continue
        fresh = run(py, {"op": "seq", "lib": lib, "texts": [b]})
        after = run(py, {"op": "seq", "lib": lib, "texts": [a, b]})
        p(f"{headers[key]}: B alone      -> {fresh.get('outs', fresh)[0]!r}")
        p(f"{headers[key]}: A, then B    -> {after.get('outs', after)[1]!r}")
    p("```\n")
    # Whole-suite history check: all cases in one process vs one process per case.
    for key, py, lib in cols:
        if not lib.startswith("nupunkt"):
            continue
        batch = run(py, {"op": "seq", "lib": lib, "texts": [c.text for c in ALL_CASES]})
        diff = [
            c.cid
            for c, o in zip(ALL_CASES, batch.get("outs", []), strict=False)
            if norm(o) != norm(res[(c.cid, key)].get("out", []))
        ]
        p(
            f"- {headers[key]}: all {len(ALL_CASES)} cases "
            f"in one process vs one process per case differ on {len(diff)} "
            f"case(s){': ' + ', '.join(diff) if diff else ''}."
        )
    p("")

    for name, title in (
        ("api", "API: return types and paragraphs"),
        ("spans", "Spans: legacy contiguous vs tight"),
        ("train", "Training on a degenerate corpus (math domain error)"),
        ("abbrevs", "Public abbreviation access"),
        ("probes", "Diagnostic probes for still-open cases"),
    ):
        p(f"### {title}\n\n```")
        prev: tuple[str, str] | None = None
        for key, py, lib in cols:
            if lib == "nupunkt":
                text = run(py, {"op": "demo", "name": name}).get("text", "").rstrip()
                _head, _, rest = text.partition("\n")
                label = headers[key].removesuffix(" default")
                if prev and prev[1] == rest:
                    p(f"# {label}: output identical to {prev[0]}")
                else:
                    p(f"# {label}\n{rest}")
                    prev = (label, rest)
        p("```\n")


if __name__ == "__main__":
    main()
