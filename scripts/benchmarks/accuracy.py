#!/usr/bin/env python3
"""Sentence-boundary accuracy benchmark for nupunkt (and, optionally, other libraries).

Scores every system on the same gold sets with the same boundary definition and writes
Markdown tables plus a JSON file of all counts.

Gold sets
  legal      data/test.jsonl.gz in this repo: 38,527 legal/government text excerpts with
             ``<|sentence|>`` markers. ``<|paragraph|>`` becomes "\\n\\n".
  legal_pm   the same texts, but a ``<|paragraph|>`` marker also counts as a boundary.
  ewt_*, gum_*, brown_flat
             built by build_gold_sets.py (UD English EWT/GUM, Brown); "para" keeps
             paragraph breaks as "\\n\\n", "flat" joins everything with single spaces.

Boundary definition (all systems, all sets): the character offset just after the last
non-whitespace character of a sentence. The end of each document is never scored.
Systems that return strings instead of offsets are aligned by counting non-whitespace
characters, so whitespace normalization cannot shift a boundary.

Usage
  python scripts/benchmarks/build_gold_sets.py --download
  python scripts/benchmarks/accuracy.py                      # nupunkt in this interpreter
  python scripts/benchmarks/accuracy.py --with-layout        # + layout-aware iter_spans configs
  python scripts/benchmarks/accuracy.py --with-comparators   # + nupunkt 0.7.0/0.6.0/0.5.1 from
                                                             #   PyPI, nltk, pysbd, sentencex,
                                                             #   blingfire, spaCy (uv venvs)

Rows labelled "nupunkt main (unreleased)" (change with ``--current-label``) use whatever nupunkt
this interpreter imports, e.g. the working tree under ``uv run``.

Only the standard library is needed to run this script; comparators are installed into
separate virtual environments with ``uv`` and run as subprocesses (``--worker`` mode).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGAL = REPO / "data" / "test.jsonl.gz"
UD_SETS = ["ewt_flat", "ewt_para", "gum_flat", "gum_para", "brown_flat"]
ALL_SETS = ["legal", "legal_pm", *UD_SETS]
CLOSERS = "\"')]}’”»"
LOW_THRESHOLD = 0.5  # a lower adaptive threshold (fewer breaks than the default 0.7)
LAYOUT = {  # system name -> iter_spans options (only with --with-layout)
    "layout-punkt": {"paragraph_breaks": False},
    "layout-default": {},
    "layout-lines": {"line_breaks": True},
}

# System name -> (environment key, human label). Environment "self" is this interpreter;
# "{cur}" is replaced by --current-label and "{v}" by the installed version.
SYSTEMS = {
    "nupunkt": ("self", "{cur} span_tokenize"),
    "layout-punkt": ("self", "{cur} iter_spans Punkt-only"),
    "layout-default": ("self", "{cur} iter_spans layout default"),
    "layout-lines": ("self", "{cur} iter_spans layout + line_breaks"),
    "nupunkt-adaptive-0.7": ("self", "{cur} adaptive t=0.7"),
    f"nupunkt-adaptive-{LOW_THRESHOLD}": ("self", f"{{cur}} adaptive t={LOW_THRESHOLD}"),
    "nupunkt@0.7.0": ("nupunkt070", "nupunkt {v} default"),
    "nupunkt@0.6.0": ("nupunkt060", "nupunkt {v} default"),
    "nupunkt@0.5.1": ("nupunkt051", "nupunkt {v} default"),
    "nltk": ("comparators", "nltk {v} punkt_tab"),
    "pysbd": ("comparators", "pysbd {v}"),
    "sentencex": ("comparators", "sentencex {v}"),
    "blingfire": ("comparators", "blingfire {v}"),
    "spacy": ("comparators", "spaCy {v} sentencizer"),
}
ENVS = {
    "nupunkt070": ["nupunkt==0.7.0"],
    "nupunkt060": ["nupunkt==0.6.0"],
    "nupunkt051": ["nupunkt==0.5.1"],
    "comparators": ["nltk", "pysbd", "sentencex", "blingfire", "spacy"],
}


# --------------------------------------------------------------------------- gold sets
def load_legal(para_markers: bool) -> list[dict]:
    rows = []
    with gzip.open(LEGAL, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            raw = json.loads(line)["text"]
            if para_markers:
                raw = raw.replace("<|paragraph|>", "<|sentence|>\n\n")
            raw = raw.replace("<|paragraph|>", "\n\n")
            text, gold = "", set()
            for part in raw.split("<|sentence|>"):
                text += part
                end = len(text.rstrip())
                if end > 0:
                    gold.add(end)
            gold.discard(len(text.rstrip()))
            rows.append({"id": str(i), "genre": "legal", "text": text, "bounds": sorted(gold)})
    return rows


def load_set(name: str, data_dir: Path) -> list[dict]:
    if name in ("legal", "legal_pm"):
        return load_legal(name == "legal_pm")
    with gzip.open(data_dir / f"gold_{name}.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    for r in rows:  # normalize to the same definition as predictions
        t = r["text"]
        r["bounds"] = sorted({len(t[:b].rstrip()) for b in r["bounds"]} - {0, len(t.rstrip())})
    return rows


# --------------------------------------------------------------------------- worker side
def ends_from_spans(text: str, spans) -> list[int]:
    return [len(text[:e].rstrip()) for _, e in spans]


def ends_from_strings(text: str, sents) -> tuple[list[int], bool]:
    """Map sentence strings back to offsets by counting non-whitespace characters."""
    pos = [i for i, c in enumerate(text) if not c.isspace()]
    ends, k = [], 0
    for s in sents:
        n = sum(1 for c in s if not c.isspace())
        if n == 0:
            continue
        k += n
        if k > len(pos):
            return ends, False
        ends.append(pos[k - 1] + 1)
    return ends, k == len(pos)


def make_system(name: str):
    """Return (version, fn) where fn(text) -> (ends, aligned_ok)."""
    if name.startswith("nupunkt"):
        import nupunkt

        v = nupunkt.__version__
        if name.startswith("nupunkt-adaptive-"):
            th = float(name.rsplit("-", 1)[1])
            return v, lambda t: (
                ends_from_spans(t, nupunkt.sent_spans_adaptive(t, threshold=th)),
                True,
            )
        if hasattr(nupunkt, "load"):
            tok = nupunkt.load("default")

            def run(t):
                return ends_from_spans(t, tok.span_tokenize(t)), True
        else:  # 0.5.x has no load(); use the public function

            def run(t):
                return ends_from_strings(t, nupunkt.sent_tokenize(t))

        return v, run
    if name in LAYOUT:
        import nupunkt

        tok, opts = nupunkt.load("default"), LAYOUT[name]
        return nupunkt.__version__, lambda t: (ends_from_spans(t, tok.iter_spans(t, **opts)), True)
    from importlib.metadata import version

    if name == "nltk":
        import nltk

        nltk.download("punkt_tab", quiet=True)
        tok = nltk.tokenize.PunktTokenizer("english")
        return version("nltk"), lambda t: (ends_from_spans(t, tok.span_tokenize(t)), True)
    if name == "pysbd":
        import pysbd

        seg = pysbd.Segmenter(language="en", clean=False)
        return version("pysbd"), lambda t: ends_from_strings(t, seg.segment(t))
    if name == "sentencex":
        import sentencex

        return version("sentencex"), lambda t: ends_from_strings(t, sentencex.segment("en", t))
    if name == "blingfire":
        import blingfire

        def bf(t):
            _, offs = blingfire.text_to_sentences_and_offsets(t)
            return ends_from_spans(t, offs), True

        return version("blingfire"), bf
    if name == "spacy":
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.max_length = 10**8
        return version("spacy"), lambda t: (
            ends_from_spans(t, [(s.start_char, s.end_char) for s in nlp(t).sents]),
            True,
        )
    raise SystemExit(f"unknown system {name}")


DET_SYSTEMS = ("nupunkt", "layout-default", "nupunkt@0.7.0", "nupunkt@0.6.0", "nupunkt@0.5.1")


def worker(a: argparse.Namespace) -> None:
    out = {}
    for name in a.systems.split(","):
        v, fn = make_system(name)
        res = {"version": v, "sets": {}}
        for s in a.sets.split(","):
            rows = load_set(s, a.data_dir)
            order = range(len(rows) - 1, -1, -1) if a.order == "reverse" else range(len(rows))
            preds: list = [None] * len(rows)
            bad = 0
            t0 = time.perf_counter()
            for i in order:
                ends, ok = fn(rows[i]["text"])
                preds[i] = sorted(set(ends))
                bad += not ok
            res["sets"][s] = {"preds": preds, "align_fail": bad, "sec": time.perf_counter() - t0}
            print(f"  {name} {v} {s}: {time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)
        out[name] = res
    with gzip.open(a.out, "wt") as fh:
        json.dump(out, fh)


# --------------------------------------------------------------------------- driver side
def ensure_env(key: str, venv_dir: Path) -> Path | None:
    py = venv_dir / key / "bin" / "python"
    if py.exists():
        return py
    try:
        subprocess.run(["uv", "venv", "-q", "-p", "3.13", str(venv_dir / key)], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(py), *ENVS[key]], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"!! could not build env {key}: {e}", file=sys.stderr)
        return None
    return py


def run_worker(py: Path, system: str, sets: list[str], a, order: str = "forward") -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as tmp:
        out = Path(tmp.name)
    env = dict(os.environ, NLTK_DATA=str(a.venv_dir / "nltk_data"), PYTHONWARNINGS="ignore")
    cmd = [str(py), __file__, "--worker", "--systems", system, "--sets", ",".join(sets)]
    cmd += ["--data-dir", str(a.data_dir), "--out", str(out), "--order", order]
    r = subprocess.run(cmd, env=env)
    if r.returncode:
        print(f"!! {system} failed (exit {r.returncode}); skipped", file=sys.stderr)
        return {}
    with gzip.open(out, "rt") as fh:
        res = json.load(fh)
    out.unlink()
    return res


def is_term(text: str, e: int) -> bool:
    i = e
    while i > 0 and text[i - 1] in CLOSERS:
        i -= 1
    return i > 0 and text[i - 1] in ".!?"


LIST_START = re.compile(r"(\(?[A-Za-z0-9]{1,4}[.)]\s|[•\-*·]\s)")


def bucket(text: str, e: int) -> str:
    if is_term(text, e):
        return "terminal .!?"
    nxt = text[e : e + 20].lstrip()
    if LIST_START.match(nxt):
        return "list item follows"
    if text[e - 1] in ";:,":
        return f"'{text[e - 1]}'"
    after = text[e : e + 20]
    if "\n" in after[: len(after) - len(after.lstrip())]:
        return "no punct., line break"
    return "no punct., inline"


def f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(rows: list[dict], preds: list[list[int]]) -> dict:
    c = Counter()
    genre = defaultdict(Counter)
    for row, pred in zip(rows, preds, strict=True):
        t = row["text"]
        g = set(row["bounds"])
        p = set(pred) - {0, len(t.rstrip())}
        tp, fp, fn = len(p & g), len(p - g), len(g - p)
        c.update(tp=tp, fp=fp, fn=fn, pred_sents=len(p) + 1, gold_sents=len(g) + 1)
        c["words"] += len(t.split())
        gt, pt = {e for e in g if is_term(t, e)}, {e for e in p if is_term(t, e)}
        c.update(ttp=len(pt & gt), tfp=len(pt - gt), tfn=len(gt - pt), gold_term=len(gt))
        genre[row["genre"]].update(tp=tp, fp=fp, fn=fn, gold=len(g))
    P, R, F = f1(c["tp"], c["fp"], c["fn"])
    return {
        "docs": len(rows),
        "P": P,
        "R": R,
        "F1": F,
        "F1_term": f1(c["ttp"], c["tfp"], c["tfn"])[2],
        "R_term": f1(c["ttp"], c["tfp"], c["tfn"])[1],
        "gold_term_share": c["gold_term"] / max(1, c["tp"] + c["fn"]),
        "wps": c["words"] / c["pred_sents"],
        "gold_wps": c["words"] / c["gold_sents"],
        "counts": dict(c),
        "genre": {
            k: {"F1": f1(v["tp"], v["fp"], v["fn"])[2], "gold": v["gold"]} for k, v in genre.items()
        },
    }


def buckets(rows: list[dict], preds: list[list[int]]) -> dict:
    out = defaultdict(Counter)
    for row, pred in zip(rows, preds, strict=True):
        t, g, p = row["text"], set(row["bounds"]), set(pred) - {0, len(row["text"].rstrip())}
        for e in g:
            out[bucket(t, e)].update(gold=1, tp=e in p)
        for e in p - g:
            out[bucket(t, e)]["fp"] += 1
    return {k: dict(v) for k, v in out.items()}


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def report(results: dict, labels: dict[str, str], a) -> str:
    sets = [s for s in ALL_SETS if any(s in results[k] for k in results)]
    systems = list(results)
    out = []

    def cell(k, s, fmt):
        return fmt(results[k][s]) if s in results[k] else "-"

    docs = {s: next(results[k][s]["docs"] for k in systems if s in results[k]) for s in sets}
    out.append("### Documents per gold set\n")
    out.append(md_table(sets, [[f"{docs[s]:,}" for s in sets]]))
    for title, fmt in (
        ("Boundary F1", lambda r: f"{r['F1']:.3f}"),
        ("Precision / recall", lambda r: f"{r['P']:.3f} / {r['R']:.3f}"),
        (
            "F1 on boundaries preceded by .!? (plus closing quotes/brackets)",
            lambda r: f"{r['F1_term']:.3f}",
        ),
        ("Mean words per sentence (whitespace tokens)", lambda r: f"{r['wps']:.1f}"),
    ):
        out.append(f"\n### {title}\n")
        rows = [[labels[k]] + [cell(k, s, fmt) for s in sets] for k in systems]
        if title.startswith("Mean words"):
            rows.insert(
                0,
                ["**gold**"]
                + [
                    f"{next(results[k][s]['gold_wps'] for k in systems if s in results[k]):.1f}"
                    for s in sets
                ],
            )
        if title.startswith("F1 on"):
            rows.insert(
                0,
                ["share of gold boundaries preceded by .!?"]
                + [
                    f"{next(results[k][s]['gold_term_share'] for k in systems if s in results[k]):.3f}"
                    for s in sets
                ],
            )
        out.append(md_table(["system", *sets], rows))
    ref = systems[0]
    for s in ("gum_para", "brown_flat"):
        if s not in results[ref]:
            continue
        g = results[ref][s]["genre"]
        order = sorted(g, key=lambda k: -g[k]["F1"])
        pick = order[:5] + ["..."] + order[-5:] if len(order) > 10 else order
        skip = ("adaptive", "0.5.1", "layout-punkt", "layout-lines")
        cols = [k for k in systems if s in results[k] and not any(x in k for x in skip)]
        rows = []
        for gn in pick:
            if gn == "...":
                rows.append(["..."] + [""] * (len(cols) + 1))
                continue
            rows.append(
                [gn, str(g[gn]["gold"])] + [f"{results[k][s]['genre'][gn]['F1']:.3f}" for k in cols]
            )
        out.append(f"\n### Per-genre F1, {s} (top and bottom 5 for {labels[ref]})\n")
        out.append(md_table(["genre", "gold bounds", *[labels[k] for k in cols]], rows))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=None, help="gold sets from build_gold_sets.py")
    ap.add_argument("--with-comparators", action="store_true")
    ap.add_argument("--with-layout", action="store_true", help="score iter_spans layout options")
    ap.add_argument("--current-label", default="nupunkt main (unreleased)")
    ap.add_argument("--venv-dir", type=Path, default=None, help="where comparator venvs live")
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--systems", default=None, help="comma-separated subset of system names")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--no-determinism", action="store_true", help="skip the order test")
    # worker mode (internal)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--order", default="forward", help=argparse.SUPPRESS)
    a = ap.parse_args()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nupunkt-benchmarks"
    a.data_dir = a.data_dir or cache
    a.venv_dir = a.venv_dir or cache / "venvs"
    if a.worker:
        worker(a)
        return

    sets = a.sets.split(",")
    names = a.systems.split(",") if a.systems else list(SYSTEMS)
    if not a.with_comparators:
        names = [n for n in names if SYSTEMS[n][0] == "self"]
    if not a.with_layout and not a.systems:
        names = [n for n in names if n not in LAYOUT]
    pythons: dict[str, Path | None] = {"self": Path(sys.executable)}
    for n in names:
        key = SYSTEMS[n][0]
        if key not in pythons:
            a.venv_dir.mkdir(parents=True, exist_ok=True)
            pythons[key] = ensure_env(key, a.venv_dir)
    if pythons.get("comparators"):
        (a.venv_dir / "nltk_data").mkdir(mode=0o700, exist_ok=True)
    tasks = [(n, pythons[SYSTEMS[n][0]]) for n in names if pythons[SYSTEMS[n][0]]]
    for n in names:
        if not pythons[SYSTEMS[n][0]]:
            print(f"!! skipping {n}: environment unavailable", file=sys.stderr)

    with ThreadPoolExecutor(a.jobs) as ex:
        raw = dict(
            zip(
                [n for n, _ in tasks],
                ex.map(lambda t: run_worker(t[1], t[0], sets, a), tasks),
                strict=True,
            )
        )
    gold = {s: load_set(s, a.data_dir) for s in sets}
    results, labels, meta = {}, {}, {}
    for n, r in raw.items():
        if n not in r:
            continue
        v = r[n]["version"]
        labels[n] = SYSTEMS[n][1].format(v=v, cur=a.current_label)
        results[n] = {s: score(gold[s], r[n]["sets"][s]["preds"]) for s in sets}
        meta[n] = {s: {k: r[n]["sets"][s][k] for k in ("align_fail", "sec")} for s in sets}
    print(report(results, labels, a))

    extra: dict = {"labels": labels, "meta": meta}
    if "legal" in sets:
        keys = [
            k
            for k in results
            if k
            in (
                "nupunkt",
                "layout-default",
                "layout-lines",
                f"nupunkt-adaptive-{LOW_THRESHOLD}",
                "nupunkt@0.7.0",
                "nupunkt@0.6.0",
                "nltk",
                "pysbd",
                "blingfire",
                "spacy",
            )
        ]
        bk = {k: buckets(gold["legal"], raw[k][k]["sets"]["legal"]["preds"]) for k in keys}
        extra["legal_buckets"] = bk
        ref = bk[keys[0]]
        total_fn = sum(v["gold"] - v.get("tp", 0) for v in ref.values())
        rows = []
        for b in sorted(ref, key=lambda b: -ref[b]["gold"]):
            fn = ref[b]["gold"] - ref[b].get("tp", 0)
            rows.append(
                [
                    b,
                    f"{ref[b]['gold']:,}",
                    f"{fn:,} ({fn / total_fn:.0%})",
                    f"{ref[b].get('fp', 0):,}",
                ]
                + [f"{bk[k].get(b, {}).get('tp', 0) / ref[b]['gold']:.3f}" for k in keys]
            )
        print(
            f"\n### Legal gold boundaries by context ({labels[keys[0]]} FN/FP; recall per system)\n"
        )
        print(
            md_table(
                ["context before boundary", "gold", "missed", "FP", *[labels[k] for k in keys]],
                rows,
            )
        )

    if not a.no_determinism and "legal" in sets:
        print(
            "\n### Determinism: legal set in forward vs reverse document order (fresh process each)\n"
        )
        det_rows = []
        for n in [n for n in DET_SYSTEMS if n in raw]:
            py = pythons[SYSTEMS[n][0]]
            fwd = run_worker(py, n, ["legal"], a, "forward")[n]["sets"]["legal"]["preds"]
            rev = run_worker(py, n, ["legal"], a, "reverse")[n]["sets"]["legal"]["preds"]
            diff = sum(x != y for x, y in zip(fwd, rev, strict=True))
            det_rows.append([labels[n], f"{len(fwd):,}", f"{diff:,}"])
            extra.setdefault("determinism", {})[n] = diff
        print(md_table(["system", "docs", "docs with different boundaries"], det_rows))

    if a.json_out:
        a.json_out.write_text(json.dumps({"results": results, **extra}, indent=1))


if __name__ == "__main__":
    main()
