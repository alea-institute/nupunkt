# Performance: cold start, footprint and throughput

This page measures nupunkt 0.7.0 against nupunkt 0.6.0, nupunkt 0.5.1, six other sentence
segmenters and a one-line regex splitter, and sets each public performance criticism next to
what we measured. It lists every case where nupunkt is slower. All numbers come from
`scripts/benchmarks/performance.py` (stdlib only); see [Reproduction](#reproduction).

## Summary

| | nupunkt 0.5.1 | nupunkt 0.6.0 | **nupunkt 0.7.0** |
|---|--:|--:|--:|
| Wheel size | 5.58 MB | 9.08 MB | **0.14 MB** |
| Bundled model file | 5.53 MB | 9.16 MB | **0.026 MB** |
| Fresh process: `import nupunkt` + first `sent_tokenize` of 1 KB | 541 ms | 1,284 ms | **13.8 ms** |
| RSS after loading the model | 143 MB | 260 MB | **20 MB** |
| Peak RSS after tokenizing 5 MB | 172 MB | 332 MB | **59 MB** |
| Unseen legal text, 14.5M chars, one call (Mchar/s), interleaved runs | 24.7 | 11.0 | **24.1** |
| 38,527 legal documents, one call each (Mchar/s, warm) | 21.3 | 10.9 | **21.7** |

In short:

* 0.7.0 fixes cold start and memory. Model load no longer dominates.
* Throughput on text it has not seen before is about 2.2x that of 0.6.0, and **about the
  same as 0.5.1** (0.5.1 is 2% faster in the interleaved run and faster on inputs under
  1 MB). 0.6.0 was a regression that 0.7.0 recovers. 0.7.0 does not go beyond 0.5.1.
* 0.7.0 memoizes boundary decisions, so **tokenizing the same text again** runs at 43–68
  Mchar/s. Treat that as a best case, not a typical speed. The CHANGELOG figure "60+ MB/s,
  warm, 4 MB legal corpus" is this repeated-input number. On unseen legal text 0.7.0 runs at
  about 23–26 MB/s, and on the two novels at 12–19 MB/s.
* sentencex (Rust), blingfire (C++) and a regex split are faster than nupunkt 0.7.0. NLTK
  Punkt is about the same speed. pysbd, spaCy's sentencizer and syntok are slower.

## Method

* **Machine:** 12th Gen Intel Core i7-12700K, 91 GB RAM, Linux 6.18 (glibc 2.43), `powersave`
  governor. Every measurement process was pinned with `taskset -c 2` (a P-core; the sibling
  hyperthread 3 was not reserved). This was a shared workstation. The 1-minute load average
  was 0.9–2.3 while the reported numbers were taken. An earlier pass taken at load average
  33 is used below only to show sensitivity to load.
* **Software:** CPython 3.13.15 in a separate `uv` venv per library. nupunkt 0.7.0 came from
  the release wheel (`dist/nupunkt-0.7.0-py3-none-any.whl`); 0.6.0 and 0.5.1 came from PyPI.
  The other libraries were nltk 3.10.3 (`punkt_tab`, `sent_tokenize`), pysbd 0.3.4
  (`clean=False`), sentencex 1.0.31, blingfire 0.1.8 (with numpy 2.5.3, which it imports
  without declaring it), spaCy 3.8.16 (`spacy.blank("en")` + `sentencizer`) and
  syntok 1.4.4. The floor is `re.split(r'(?<=[.!?])\s+', text)` on the stdlib.
* **Statistic:** each value is the median of 5 repetitions. Every (library, task) pair runs
  in a fresh interpreter. "First call" means the first call on that corpus in a fresh
  process, made after a 1 KB warm-up call that loads the model. For nupunkt 0.7.0 this is
  also the unseen-text case. "Warm" is the median of 5 more calls on the same input.
  "Cold start" is the time from before `import` to the end of the first `sent_tokenize` of
  a 1 KB legal excerpt, measured inside a fresh process. The process wall time includes
  about 14 ms of bare interpreter start-up. One untimed run first populates `.pyc` files
  and the OS page cache, so these numbers are not cold-disk numbers.
* **Memory:** `VmRSS` and `VmHWM` are read from `/proc/self/status` in a fresh process
  after the 1 KB call, and again after tokenizing a 5.0M-char legal text (`VmHWM` = peak).
* **Corpora:**
  * Project Gutenberg #11 (*Alice's Adventures in Wonderland*, 144,600 chars) and #1661
    (*The Adventures of Sherlock Holmes*, 562,203 chars), fetched from
    `gutenberg.org/cache/epub/<id>/pg<id>.txt`. Everything outside the `*** START/END ***`
    markers is stripped. The yasbd-lib figures below use 148,208 and 593,911 chars, which
    suggests their header handling differs.
  * `legal_4mb`: 4.0M chars, 10,572 docs from `data/test.jsonl.gz`, shuffled with seed 1234.
  * `legal_all`: all 38,527 docs, 14.5M chars, no repeats.
  * `legal_50mb`: `legal_all` repeated to 50.0M chars.
  * The 5 MB memory text is a prefix of `legal_50mb`.
* **Time budget:** a (library, corpus) pair is skipped if a single call is estimated at more
  than 60 s (power-law extrapolation). pysbd was the only library skipped.

## Public criticisms and the 0.7.0 measurement

| Source | What was reported (version) | Measured with 0.7.0 |
|---|---|---|
| [houfu/redlines #48](https://github.com/houfu/redlines/issues/48#issuecomment-3401466733) | `NupunktProcessor` "2.1x to 6.5x" (avg **5.3x**) slower than the regex paragraph splitter; "~1.6M chars/second" (nupunkt>=0.6.0; end-to-end redlines diffing, not segmentation alone) | **Still true in ratio terms.** Segmentation alone: 0.7.0 runs 4.5x slower than a regex split on unseen legal text (23 vs 105 Mchar/s) and 5.6x slower per document (17.3 vs 3.1 µs per doc). The absolute speed is about 23 Mchar/s, but redlines' 1.6M figure includes their diffing, so the two numbers are not directly comparable. |
| [speedyk-005/yasbd-lib](https://github.com/speedyk-005/yasbd-lib) README | "nupunkt has an 11-second cold start" (0.6.x) | We did not reproduce 11 s with any version. 0.6.0: 1.28 s at load 1; 2.20 s at load 33. **0.7.0: 13.8 ms** (11.5 ms import + 2.3 ms first call). |
| [yasbd-lib benchmarks](https://github.com/speedyk-005/yasbd-lib/tree/main/benchmarks) (2026-09-25) | Alice warm 47.3 ms vs sentencex 3.8 ms, blingfire 9.7 ms; Sherlock warm 240.7 ms vs 11.2 / 42.9 ms; "Highly Accurate, but Speed Liability" | Alice first call 11.0 ms (sentencex 1.8, blingfire 4.9); Sherlock first call 45.6 ms (sentencex 5.2, blingfire 18.0). **nupunkt is still 6–9x slower than sentencex and 2.3–2.5x slower than blingfire on these books** (on unseen text). |
| [GriffynHancock/homogenous-cluster](https://github.com/GriffynHancock/homogenous-cluster) `missing_link/sentences.py` | "first sentence_spans call pays ~2.1 s to load" (0.6.0) | 2.3 ms for the first call after import; 13.8 ms including import. |
| [mohamed-benoughidene/openreview-cli](https://github.com/mohamed-benoughidene/openreview-cli) `docs/BENCHMARKS.md` | "cold 3.1-3.2 s \| one-time nupunkt model load per process" | Same as above: 13.8 ms for import plus the first call. |
| [cognizant-ai-lab/neuro-san-studio #735](https://github.com/cognizant-ai-lab/neuro-san-studio/pull/735) | nupunkt "~9 MB" (0.6.0) | Wheel 0.14 MB; installed package 0.7 MB; model 26 KB. |
| [craigtrim/fast-sentence-segment #16](https://github.com/craigtrim/fast-sentence-segment/issues/16) (quoting [arXiv 2504.04131](https://arxiv.org/abs/2504.04131)) | "Speed: 10M chars/sec", "Memory: 432 MB" | 20 MB RSS after load and 59 MB peak on 5 MB of input. 19–26 Mchar/s on unseen legal text and 12–19 on the novels. The 10M figure holds; 432 MB no longer applies. |
| [KnowSeams](https://github.com/knowseams/knowseams/blob/ba9fb10a5fb3cb363119e28e5fc7043fcc852dba/benchmarks/performance-results.md) | nupunkt 0.5.1 at "18.0" and "19.7 MB/s" sentence-detection throughput | 0.7.0 on unseen legal text runs at 23–26 MB/s, the same as 0.5.1 on this machine (24–28 MB/s). There is **no speed-up to report against 0.5.1**. (KnowSeams times `PunktSentenceTokenizer()` without the default model, so their setup differs from ours.) |

## Cold start and footprint

| library | import + first 1 KB call | process wall | RSS after load | peak RSS, 5 MB input | installed size* |
|---|--:|--:|--:|--:|--:|
| **nupunkt 0.7.0** | **13.8 ms** | 31 ms | **20 MB** | 59 MB | 0.7 MB |
| nupunkt 0.6.0 | 1,284 ms | 1,345 ms | 260 MB | 332 MB | 9.7 MB |
| nupunkt 0.5.1 | 541 ms | 579 ms | 143 MB | 172 MB | 6.1 MB |
| nltk (punkt_tab) | 95 ms | 123 ms | 43 MB | 67 MB | 15.9 MB + data |
| pysbd | 11 ms | 26 ms | 16 MB | not run (too slow) | 0.4 MB |
| sentencex | 1.0 ms | 15 ms | 18 MB | 45 MB | 4.2 MB |
| blingfire | 33 ms | 52 ms | 33 MB | 170 MB | 160 MB |
| spaCy sentencizer | 338 ms | 415 ms | 96 MB | 360 MB | 260 MB |
| syntok | 8 ms | 23 ms | 18 MB | 59 MB | 3.1 MB |
| regex split | 0.0 ms | 16 ms | 15 MB | 39 MB | stdlib |

\* Size of the venv's `site-packages`, dependencies included; blingfire bundles about 99 MB
of tokenizer models. The bare interpreter starts in 14 ms and uses 14 MB RSS. For comparison,
the same cold-start run taken at load average 33 measured 0.6.0 at 2,204 ms and 0.7.0 at
16.2 ms.

## Throughput

Cells show **first call / warm**, in Mchar/s. The legal text is almost all ASCII, so MB/s
(UTF-8) is 1.00–1.01x these values; for the novels it is 1.02–1.04x. The script prints both.

| library | Alice 0.14M | Sherlock 0.56M | legal 4.0M | legal_all 14.5M (unseen) | legal 50M | 38.5k docs, one call each |
|---|--:|--:|--:|--:|--:|--:|
| **nupunkt 0.7.0** | 13.2 / 51.1 | 12.3 / 42.8 | 23.7 / 67.8 | 22.9 / 25.5 | 23.3 / 23.8 | 19.3 / 21.7 |
| nupunkt 0.7.0, memo cleared before each warm run | 18.6 | 14.9 | 26.1 | 25.8 | 23.6 | 20.6 |
| nupunkt 0.7.0 `adaptive=True` | 8.8 / 11.6 | 7.2 / 8.9 | 12.8 / 13.9 | 12.6 / 13.5 | 12.4 / 12.5 | 10.8 / 11.2 |
| nupunkt 0.6.0 | 6.0 / 19.5 | 7.0 / 10.6 | 10.6 / 13.6 | 11.1 / 11.9 | 11.5 / 11.5 | 9.8 / 10.9 |
| nupunkt 0.6.0 `adaptive=True` | 7.3 / 16.2 | 6.3 / 9.0 | 9.4 / 11.8 | 9.7 / 10.7 | 10.2 / 10.3 | 8.1 / 8.9 |
| nupunkt 0.5.1 | 10.5 / 37.8 | 16.0 / 26.4 | 25.1 / 27.5 | 25.0 / 25.8 | 23.9 / 24.0 | 20.8 / 21.3 |
| nltk punkt_tab | 14.2 / 14.7 | 13.0 / 13.4 | 20.0 / 20.0 | 20.0 / 19.8 | 20.5 / 20.0 | 18.3 / 18.3 |
| pysbd | 0.34 / 0.34 | 0.18 / 0.18 | >780 s, stopped | skipped | skipped | 0.43 / 0.43 |
| sentencex | 78.9 / 117.1 | 108.6 / 137.9 | 112.3 / 131.1 | 112.7 / 127.2 | 112.9 / 127.9 | 114.2 / 120.8 |
| blingfire | 29.8 / 33.3 | 31.2 / 33.0 | 30.3 / 31.3 | 30.9 / 31.6 | 29.8 / 30.2 | 25.7 / 25.8 |
| spaCy sentencizer | 1.15 / 1.43 | 1.24 / 1.47 | 1.28 / 1.46 | 1.29 / 1.43 | 1.41 / 1.45 | 2.16 / 2.57 |
| syntok | 2.13 / 2.17 | 2.21 / 2.22 | 2.63 / 2.62 | 2.62 / 2.62 | 2.59 / 2.60 | 2.57 / 2.59 |
| regex split (floor) | 100.9 / 112.8 | 100.7 / 108.0 | 103.6 / 108.5 | 104.9 / 107.7 | 104.8 / 106.5 | 118.9 / 119.4 |

To check version-to-version differences under the same machine load, `legal_all` was run
first-call only, rotating 0.7.0 → 0.6.0 → 0.5.1, 5 rounds, median. The results were **24.1 /
11.0 / 24.7 Mchar/s**. An independent earlier pass gave 24.0 / 11.0 / 24.8.

**The memo effect.** From 0.7.0 on, nupunkt caches each context-string → boundary decision,
up to 32,768 entries per tokenizer. Repeating the same input, as a "warm" benchmark does,
hits that cache. For a 0.14M-char novel the hits turn 13–19 Mchar/s into 51 Mchar/s. Across
the 14.5M chars of distinct legal documents the cache adds little: 22.9 on the first call
against 25.5 warm. For real workloads, use the first-call, memo-cleared and `legal_all`
columns, not the warm column.

**Outputs differ.** On the 5 MB text, nupunkt 0.7.0 returns 17,726 sentences, NLTK 25,265,
blingfire 19,235, spaCy 24,117, syntok 54,859 and sentencex 99,325; sentencex also breaks at
line breaks. Speed says nothing about accuracy; see [accuracy.md](accuracy.md).

## Scaling (time per call vs input size)

The inputs are prefixes of `legal_50mb`, and each value is the median time per call in
nanoseconds per character. For nupunkt 0.7.0 the memo is cleared before every call. A
constant value means linear scaling.

| library | 10 K | 100 K | 1 M | 10 M | 50 M |
|---|--:|--:|--:|--:|--:|
| **nupunkt 0.7.0** | 40 | 39 | 40 | 39 | 42 |
| nupunkt 0.7.0 adaptive | 62 | 64 | 68 | 74 | 80 |
| nupunkt 0.6.0 | 67 | 64 | 70 | 82 | 87 |
| nupunkt 0.5.1 | 28 | 28 | 32 | 38 | 41 |
| nltk | 48 | 49 | 48 | 50 | 50 |
| sentencex | 4 | 7 | 7 | 8 | 8 |
| blingfire | 21 | 26 | 29 | 32 | 33 |
| spaCy sentencizer | 596 | 645 | 672 | 690 | 694 |
| syntok | 368 | 382 | 379 | 381 | 383 |
| pysbd | 1,576 | 2,724 | 13,198 | skipped | skipped |
| regex split | 9 | 9 | 9 | 9 | 10 |

nupunkt 0.7.0 is linear from 10 K to 50 M chars: 2.1 s for 50 M chars, with no per-call
overhead that matters at 10 K. The adaptive mode, 0.6.0, and 0.5.1 grow slightly per
character with size (0.5.1 by 28 → 41 ns). pysbd is strongly superlinear on this text.

## Where nupunkt 0.7.0 is still slower

* **sentencex** (Rust extension): 5–9x faster on unseen text (4.9x on `legal_all`, 6.0x on
  Alice, 8.8x on Sherlock) and 5.6x faster per document.
* **blingfire** (C++ finite-state machine): 1.3x faster on legal text and 2.3–2.5x on the
  novels. On the novels, only nupunkt's repeated-input warm number is faster than blingfire.
* **A regex split:** 4.5x faster on long legal text and 5.6x per document. It handles no
  abbreviations, which is how the redlines #48 ratio arises.
* **nupunkt 0.5.1:** equal on long unseen text and 1.3–1.4x faster below 1 MB (28 vs 40
  ns/char). 0.7.0's gains over 0.5.1 are in load time, memory and package size, not
  throughput.
* **NLTK Punkt:** first call 6–8% faster than nupunkt on the two novels. nupunkt is
  about 15% faster on legal text.
* **`adaptive=True`:** runs at about half the speed of the default mode (12–14 vs 23–26
  Mchar/s on legal text).

Why: nupunkt is a pure-Python implementation of Punkt with zero runtime dependencies (see
`CLAUDE.md`). Each candidate boundary goes through Python-level token classification and
orthographic/collocation lookups. sentencex and blingfire do comparable work in compiled
code. Staying pure Python and dependency-free is a deliberate trade-off. What 0.7.0 changed
is the fixed costs: import plus first call in 14 ms, 20 MB resident and a 26 KB model. It
did not change the per-character cost relative to 0.5.1.

nupunkt 0.7.0 is faster than pysbd (40–330x), syntok (6–9x), spaCy's blank-model
sentencizer (11–18x) and nupunkt 0.6.0 (2.2x), and roughly matches NLTK.

## Reproduction

```bash
# nupunkt-only (whatever `import nupunkt` resolves to) plus the regex floor, ~1 minute
uv run python scripts/benchmarks/performance.py

# everything on this page (needs `uv` and network; ~60 minutes, mostly pysbd/spaCy/syntok)
python scripts/benchmarks/performance.py --with-comparators --json perf.json
```

Options: `--runs` (default 5), `--cpu` (pinned core, default 2; `-1` disables pinning),
`--work-dir` (corpora and venv cache), `--wheel` (nupunkt 0.7.0 wheel; defaults to
`dist/*.whl`, otherwise PyPI), `--only nupunkt,sentencex,...`, `--max-run-seconds`, and
`--resume` (reuse the libraries already in `--json`). The Gutenberg files are downloaded on
the first run. The SHA-256 of each raw file is stored in the JSON; for the runs above they
were `01b38ea4…725e` (#11) and `922e2a12…7cd0` (#1661).
