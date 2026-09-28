#!/usr/bin/env python3
"""Build general-English sentence-boundary gold sets for nupunkt evaluation (stdlib only).

Sources (downloaded with ``curl`` into ``--data-dir`` when ``--download`` is given):

* UD English-EWT, release r2.18 (web text: weblogs, newsgroups, email, reviews, answers).
  Silveira et al. 2014; https://github.com/UniversalDependencies/UD_English-EWT
  License: CC BY-SA 4.0.
* UD English-GUM, release r2.18 (academic, bio, fiction, interview, news, voyage, whow,
  reddit, speech, vlog, textbook, ...). Zeldes 2017;
  https://github.com/UniversalDependencies/UD_English-GUM
  License: CC BY-SA 4.0 for the annotations; some source texts carry their own CC licenses
  (see the GUM README). The Reddit subset is not included in the UD release; any
  document whose text is masked with underscores would be skipped.
* Brown corpus (Francis & Kucera 1961/1979), from the NLTK data package ``brown.zip``
  (https://github.com/nltk/nltk_data). Distributed by NLTK "for non-commercial use";
  see the README inside the zip.

All train/dev/test splits are used: nothing here is trained on these corpora, and more
documents give tighter estimates.

Gold offsets
------------
UD: each sentence is taken verbatim from its ``# text =`` comment (the untokenized
sentence). Sentences are joined with a single space; paragraphs (``# newpar``) are joined
with ``"\\n\\n"`` in the "para" variant and with a single space in the "flat" variant.
Brown: the corpus ships only tokens, so each sentence is detokenized with the simple rules
in :func:`detok_brown` and sentences are joined with a single space (no paragraph info).

A gold boundary is the character offset just after the last character of each sentence.
The end of each document is not a scored boundary.

Output: ``<data-dir>/gold_<name>.jsonl.gz``, one JSON object per document::

    {"id": ..., "genre": ..., "text": ..., "bounds": [end offsets, excluding end of doc]}

Ported from an exploratory script written during the nupunkt 0.7.0 evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import zipfile
from collections.abc import Iterator
from pathlib import Path

UD_TAG = "r2.18"
UD = {
    "ewt": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{tag}/en_ewt-ud-{split}.conllu",
    "gum": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/{tag}/en_gum-ud-{split}.conllu",
}
BROWN = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
SPLITS = ("train", "dev", "test")


def default_data_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "nupunkt-benchmarks"


def download(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    for url in UD.values():
        for split in SPLITS:
            u = url.format(tag=UD_TAG, split=split)
            dst = data / u.rsplit("/", 1)[1]
            if not dst.exists():
                subprocess.run(["curl", "-sfL", "-o", str(dst), u], check=True)
    if not (data / "brown").exists():
        subprocess.run(["curl", "-sfL", "-o", str(data / "brown.zip"), BROWN], check=True)
        with zipfile.ZipFile(data / "brown.zip") as z:
            z.extractall(data)


def read_ud(data: Path, name: str) -> Iterator[tuple[str, list[list[str]]]]:
    """Yield (doc_id, paragraphs), each paragraph a list of raw sentence strings."""
    for split in SPLITS:
        path = data / f"en_{name}-ud-{split}.conllu"
        doc_id: str | None = None
        paras: list[list[str]] = []
        newpar = True
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("# newdoc"):
                    if doc_id and paras:
                        yield doc_id, paras
                    doc_id = line.split("=", 1)[1].strip() if "=" in line else f"{name}-{split}"
                    paras, newpar = [], True
                elif line.startswith("# newpar"):
                    newpar = True
                elif line.startswith("# text ="):
                    s = line.split("=", 1)[1].strip()
                    if newpar or not paras:
                        paras.append([])
                        newpar = False
                    paras[-1].append(s)
        if doc_id and paras:
            yield doc_id, paras


def genre_of(name: str, doc_id: str) -> str:
    return doc_id.split("_")[1] if name == "gum" else doc_id.split("-")[0]


_NOSPACE_BEFORE = {",", ".", ";", ":", "?", "!", ")", "]", "}", "''", "'", "%"}
_NOSPACE_BEFORE |= {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m"}
_NOSPACE_AFTER = {"(", "[", "{", "``", "`", "$"}


def detok_brown(tokens: list[str]) -> str:
    """Rule-based detokenizer for Brown tokens.

    ````/'' `` become ``"``; closing punctuation attaches to the left, opening brackets and
    opening quotes to the right; a sentence-final ``.`` after a token that already ends in
    ``.`` (e.g. ``Jr.``) is dropped, because Brown splits it off as its own token; a
    sentence-final ``;``, ``?``, ``!`` or ``:`` that repeats the previous token is dropped
    (the Brown files end ~6,000 sentences with a doubled token such as ``;/. ;/.``, which
    would otherwise appear in the text as ``;;`` or ``??``).
    Limitations: single-quote direction is ambiguous, spacing around ``--`` is kept as
    tokenized, and the original spacing of hyphens and ellipses is unknown.
    """
    out: list[str] = []
    glue = True
    for i, t in enumerate(tokens):
        last = i == len(tokens) - 1
        if last and out and t == "." and out[-1].endswith("."):
            continue
        if last and i and t in ";?!:" and tokens[i - 1] == t:
            continue
        tt = '"' if t in ("``", "''") else t
        if out and not glue and t not in _NOSPACE_BEFORE:
            out.append(" ")
        out.append(tt)
        glue = t in _NOSPACE_AFTER
    return re.sub(r"\s+", " ", "".join(out)).strip()


def read_brown(data: Path) -> Iterator[tuple[str, str, list[list[str]]]]:
    cats = {}
    with (data / "brown" / "cats.txt").open() as fh:
        for line in fh:
            f, c = line.split()
            cats[f] = c
    for f in sorted(cats):
        sents = []
        with (data / "brown" / f).open(encoding="latin-1") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = detok_brown([w.rsplit("/", 1)[0] for w in line.split()])
                if s:
                    sents.append(s)
        yield f, cats[f], [sents]


def assemble(paras: list[list[str]], para_sep: str) -> tuple[str, list[int]]:
    text, bounds = "", []
    for pi, sents in enumerate(paras):
        if pi:
            text += para_sep
        for si, s in enumerate(sents):
            if si:
                text += " "
            text += s
            bounds.append(len(text))
    bounds.pop()  # the end of the document is not scored
    return text, bounds


def write(data: Path, name: str, rows: list[dict]) -> None:
    path = data / f"gold_{name}.jsonl.gz"
    n_s = sum(len(r["bounds"]) + 1 for r in rows)
    n_c = sum(len(r["text"]) for r in rows)
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        for r in rows:
            raw.write((json.dumps(r) + "\n").encode("utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(f"{path.name}: {len(rows)} docs, {n_s} sentences, {n_c} chars, sha256 {digest}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--download", action="store_true", help="fetch sources with curl first")
    a = ap.parse_args()
    data: Path = a.data_dir
    if a.download:
        download(data)
    for name in UD:
        docs = [
            (d, p)
            for d, p in read_ud(data, name)
            # GUM reddit text is not distributed in UD (tokens are "_")
            if not all(set(s) <= {"_", " "} for para in p for s in para)
        ]
        for mode, sep in (("para", "\n\n"), ("flat", " ")):
            rows = []
            for doc_id, paras in docs:
                text, b = assemble(paras, sep)
                rows.append(
                    {"id": doc_id, "genre": genre_of(name, doc_id), "text": text, "bounds": b}
                )
            write(data, f"{name}_{mode}", rows)
    rows = []
    for f, cat, paras in read_brown(data):
        text, b = assemble(paras, " ")
        rows.append({"id": f, "genre": cat, "text": text, "bounds": b})
    write(data, "brown_flat", rows)


if __name__ == "__main__":
    main()
