#!/usr/bin/env python3
"""Golden Rule Set (GRS) benchmark for nupunkt and optional comparators.

Runs two English edge-case sets stored under ``scripts/benchmarks/data/``:

* ``golden_rules_pysbd_en.json`` - the 48-case GRS shipped in pySBD's
  ``benchmarks/english_golden_rules.py`` (MIT).
* ``golden_rules_yasbd_en.json`` - the 92-case expanded set from yasbd-lib's
  ``benchmarks/EN_GOLDEN_DATA.py`` (MPL-2.0).

Scoring
-------
* **Exact pass**: ``[s.strip() for s in output if s.strip()] == expected``.
* **Boundary P/R/F1 (offset)**: internal sentence boundaries (text end
  excluded) as offsets counted in non-whitespace characters, micro-averaged
  over all cases.
* **Boundary F1 (yasbd scorer)**: re-implementation of yasbd-lib's
  ``benchmarks/scorer.py`` (word-level 0/1 arrays, right-aligned, text end
  included), so the numbers can be compared with the published table.

Rows
----
nupunkt: ``sent_tokenize`` (default and ``adaptive=True``),
``nupunkt.sentences`` (the segmentation interface; layout-aware where the
installed version supports it) and a fresh default tokenizer with the opt-in
``EXCL_QUEST_LOWERCASE_CONTINUES = True``. Rows whose API is missing in the
installed version are skipped.

Timing
------
* **Warm**: one untimed pass over the set, then ``--repeats`` timed passes;
  the median pass time is reported.
* **Cold**: ``--cold-runs`` fresh interpreter processes, each timing
  ``import`` + segmenter construction + the first call on case 1 with
  ``time.perf_counter`` inside the process (interpreter startup excluded);
  the median is reported.

Usage
-----
    python scripts/benchmarks/golden_rules.py                    # nupunkt only
    python scripts/benchmarks/golden_rules.py --with-comparators # + others
    python scripts/benchmarks/golden_rules.py --json out.json --failures

Only the standard library and nupunkt are required. Comparators (nltk with
punkt_tab, pysbd, sentencex, blingfire, spacy, yasbd-lib) are used only when
``--with-comparators`` is given and they import successfully.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CASE_SETS = {
    "pysbd-48": DATA_DIR / "golden_rules_pysbd_en.json",
    "yasbd-92": DATA_DIR / "golden_rules_yasbd_en.json",
}

Segmenter = Callable[[str], list[str]]


# --------------------------------------------------------------------------
# Segmenter factories. Each returns (segment_fn, version_string). Imports are
# local so that cold-start timing includes them.
# --------------------------------------------------------------------------


def _nupunkt_default() -> tuple[Segmenter, str]:
    import nupunkt

    return (lambda t: list(nupunkt.sent_tokenize(t))), nupunkt.__version__


def _nupunkt_adaptive() -> tuple[Segmenter, str]:
    import nupunkt

    return (lambda t: list(nupunkt.sent_tokenize(t, adaptive=True))), nupunkt.__version__


def _nupunkt_sentences() -> tuple[Segmenter, str]:
    import nupunkt

    if not hasattr(nupunkt, "sentences"):
        raise ImportError("nupunkt.sentences() not available in this version")
    return (lambda t: list(nupunkt.sentences(t))), nupunkt.__version__


def _nupunkt_excl_quest() -> tuple[Segmenter, str]:
    import nupunkt
    from nupunkt.models import load_default_model

    tok = load_default_model()  # independent instance; nupunkt.load() is shared
    if not hasattr(tok, "EXCL_QUEST_LOWERCASE_CONTINUES"):
        raise ImportError("EXCL_QUEST_LOWERCASE_CONTINUES not available in this version")
    tok.EXCL_QUEST_LOWERCASE_CONTINUES = True
    return (lambda t: list(tok.tokenize(t))), nupunkt.__version__


def _nltk() -> tuple[Segmenter, str]:
    import nltk

    nltk.data.find("tokenizers/punkt_tab/english/")
    return (lambda t: nltk.sent_tokenize(t)), nltk.__version__


def _pysbd() -> tuple[Segmenter, str]:
    import importlib.metadata

    import pysbd

    seg = pysbd.Segmenter(language="en", clean=False)
    return (lambda t: seg.segment(t)), importlib.metadata.version("pysbd")


def _sentencex() -> tuple[Segmenter, str]:
    import importlib.metadata

    import sentencex

    return (lambda t: list(sentencex.segment("en", t))), importlib.metadata.version("sentencex")


def _blingfire() -> tuple[Segmenter, str]:
    import importlib.metadata

    import blingfire

    return (lambda t: blingfire.text_to_sentences(t).split("\n")), importlib.metadata.version(
        "blingfire"
    )


def _spacy_sentencizer() -> tuple[Segmenter, str]:
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return (lambda t: [s.text for s in nlp(t).sents]), spacy.__version__


def _yasbd() -> tuple[Segmenter, str]:
    import importlib.metadata

    from yasbd.boundary_detector import BoundaryDetector

    det = BoundaryDetector(lang="en")
    return (lambda t: list(det.segment(t))), importlib.metadata.version("yasbd-lib")


NUPUNKT = {
    "nupunkt default": _nupunkt_default,
    "nupunkt adaptive": _nupunkt_adaptive,
    "nupunkt sentences()": _nupunkt_sentences,
    "nupunkt EXCL_QUEST_LOWERCASE_CONTINUES": _nupunkt_excl_quest,
}
COMPARATORS = {
    "nltk punkt_tab": _nltk,
    "pysbd": _pysbd,
    "sentencex": _sentencex,
    "blingfire": _blingfire,
    "spacy sentencizer": _spacy_sentencizer,
    "yasbd": _yasbd,
}
ALL = {**NUPUNKT, **COMPARATORS}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def normalize(sentences: list[str]) -> list[str]:
    """Strip each sentence and drop empty strings."""
    return [s.strip() for s in sentences if s.strip()]


def offset_boundaries(sentences: list[str]) -> set[int]:
    """Internal boundaries as cumulative non-whitespace character counts."""
    out: set[int] = set()
    pos = 0
    for s in sentences:
        pos += sum(1 for c in s if not c.isspace())
        out.add(pos)
    out.discard(pos)  # the end of the text is not a decision
    return out


def yasbd_counts(pred: list[str], gold: list[str]) -> tuple[int, int, int]:
    """TP/FP/FN as computed by yasbd-lib benchmarks/scorer.py."""

    def to_bin(sents: list[str]) -> list[int]:
        b: list[int] = []
        for s in sents:
            n = len(s.split())
            if n:
                b.extend([0] * (n - 1) + [1])
        return b

    g, p = to_bin(gold), to_bin(pred)
    d = len(g) - len(p)
    if d > 0:
        p = [0] * d + p
    elif d < 0:
        g = [0] * (-d) + g
    tp = sum(1 for a, b in zip(g, p, strict=True) if a and b)
    fp = sum(1 for a, b in zip(g, p, strict=True) if not a and b)
    fn = sum(1 for a, b in zip(g, p, strict=True) if a and not b)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(seg: Segmenter, cases: list[dict], repeats: int) -> dict:
    passed = 0
    otp = ofp = ofn = ytp = yfp = yfn = 0
    failures = []
    for case in cases:
        try:
            got = normalize(seg(case["text"]))
        except Exception as exc:  # report, do not crash the run
            got = [f"<ERROR: {exc!r}>"]
        exp = case["expected"]
        if got == exp:
            passed += 1
        else:
            failures.append({"id": case["id"], "text": case["text"], "expected": exp, "got": got})
        gb, pb = offset_boundaries(exp), offset_boundaries(got)
        otp += len(gb & pb)
        ofp += len(pb - gb)
        ofn += len(gb - pb)
        a, b, c = yasbd_counts(got, exp)
        ytp, yfp, yfn = ytp + a, yfp + b, yfn + c

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for case in cases:
            seg(case["text"])
        times.append(time.perf_counter() - t0)

    op, orc, of = prf(otp, ofp, ofn)
    return {
        "passed": passed,
        "total": len(cases),
        "offset_p": op,
        "offset_r": orc,
        "offset_f1": of,
        "offset_counts": [otp, ofp, ofn],
        "yasbd_f1": prf(ytp, yfp, yfn)[2],
        "warm_ms_median": statistics.median(times) * 1000 if times else 0.0,
        "failures": failures,
    }


def cold_start(name: str, runs: int) -> float | None:
    """Median seconds for import + construction + first call in a fresh process."""
    vals = []
    for _ in range(runs):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_cold", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        vals.append(float(proc.stdout.strip().splitlines()[-1]))
    return statistics.median(vals)


def _cold_child(name: str) -> None:
    text = json.loads(CASE_SETS["pysbd-48"].read_text(encoding="utf-8"))["cases"][0]["text"]
    t0 = time.perf_counter()
    seg, _ = ALL[name]()
    seg(text)
    print(time.perf_counter() - t0)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--with-comparators", action="store_true")
    ap.add_argument("--repeats", type=int, default=20, help="timed warm passes per set")
    ap.add_argument("--cold-runs", type=int, default=5)
    ap.add_argument("--no-cold", action="store_true", help="skip cold-start timing")
    ap.add_argument("--failures", action="store_true", help="print every failing case")
    ap.add_argument("--json", type=Path, help="write full results to this path")
    ap.add_argument(
        "--nupunkt-label",
        help="version label for nupunkt rows (e.g. 'main (unreleased)'); default: __version__",
    )
    ap.add_argument("--_cold", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._cold:
        _cold_child(args._cold)
        return 0

    sets = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in CASE_SETS.items()}
    libs = dict(NUPUNKT)
    if args.with_comparators:
        libs.update(COMPARATORS)

    results: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "libraries": {},
    }
    for name, factory in libs.items():
        try:
            seg, version = factory()
        except Exception as exc:
            print(f"# skipped {name}: {exc!r}", file=sys.stderr)
            continue
        if name in NUPUNKT and args.nupunkt_label:
            version = args.nupunkt_label
        label = f"{name} {version}"
        entry: dict = {"version": version, "sets": {}}
        for set_name, data in sets.items():
            evaluate(seg, data["cases"][:3], 0)  # warm-up (caches, lazy loads)
            entry["sets"][set_name] = evaluate(seg, data["cases"], args.repeats)
        entry["cold_s_median"] = None if args.no_cold else cold_start(name, args.cold_runs)
        results["libraries"][label] = entry

    print(f"python {results['python']} on {results['platform']}")
    for set_name in sets:
        print(f"\n## {set_name}")
        print("| library | pass | % | offset F1 (P/R) | yasbd F1 | warm ms | cold ms |")
        print("|---|---|---|---|---|---|---|")
        for label, e in results["libraries"].items():
            r = e["sets"][set_name]
            cold = "-" if e["cold_s_median"] is None else f"{e['cold_s_median'] * 1000:.0f}"
            print(
                f"| {label} | {r['passed']}/{r['total']} | {100 * r['passed'] / r['total']:.1f} "
                f"| {r['offset_f1']:.3f} ({r['offset_p']:.3f}/{r['offset_r']:.3f}) "
                f"| {r['yasbd_f1']:.3f} | {r['warm_ms_median']:.2f} | {cold} |"
            )
    if args.failures:
        for label, e in results["libraries"].items():
            for set_name, r in e["sets"].items():
                for f in r["failures"]:
                    print(f"\n[{label} / {set_name} #{f['id']}]")
                    print(f"  input:    {f['text']!r}")
                    print(f"  expected: {f['expected']!r}")
                    print(f"  got:      {f['got']!r}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
