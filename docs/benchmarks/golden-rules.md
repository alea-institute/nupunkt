# Golden Rule Set (GRS) benchmark

## Purpose

Two public comparisons rank nupunkt low on English Golden Rule Set cases:

- speedyk-005/yasbd-lib, `benchmarks/README.md`: nupunkt 59/92 (64.1%), 7th of 8 libraries,
  on a 92-case set expanded from pySBD's GRS.
- craigtrim/fast-sentence-segment issue #16 cites pySBD's own benchmark: pySBD 97.92%,
  NLTK 56.25% on the 48-case GRS.

This page reproduces both sets for nupunkt 0.7.0 (release), the current unreleased main
tree, nupunkt 0.6.0 and six other libraries, and lists every case nupunkt gets wrong.

## Case sets

| Set | Cases | Source | License |
|---|---|---|---|
| `pysbd-48` | 48 | [pySBD `benchmarks/english_golden_rules.py`](https://github.com/nipunsadvilkar/pySBD/blob/5905f13be4fc95f407b98392e0ec303617a33d86/benchmarks/english_golden_rules.py) at commit `5905f13` | MIT |
| `yasbd-92` | 92 | [yasbd-lib `benchmarks/EN_GOLDEN_DATA.py`](https://github.com/speedyk-005/yasbd-lib/blob/0d2f0f4db8afcd9e6a5fc647fed7960b8a37f3b1/benchmarks/EN_GOLDEN_DATA.py) at commit `0d2f0f4` | MPL-2.0 |

Both sets derive from the Golden Rules Set written by the TM-Town team for Pragmatic Segmenter.
The cases are stored unmodified as JSON in `scripts/benchmarks/data/`
(`golden_rules_pysbd_en.json`, `golden_rules_yasbd_en.json`); each file carries its source
URL, commit and license. 43 of the 92 yasbd inputs appear verbatim in the pySBD set (field
`in_pysbd`). yasbd-lib removed or rewrote some pySBD cases (e.g. the `⁃9.` list and the
mid-ellipsis split) and added cases for `!` inside brand names, spaced terminator runs,
quotations, parentheticals and mixed CJK/English.

The yasbd table was published in commit `7128d43` (2026-06-13). The only later change to
`EN_GOLDEN_DATA.py` is a comment edit, so the cases are the same ones that produced 59/92.

## Methodology

Script: `scripts/benchmarks/golden_rules.py` (standard library only).

- **Exact pass**: `[s.strip() for s in output if s.strip()] == expected`. This is the rule
  used by yasbd-lib's `run_golden.py` (it strips but does not drop empty strings; no library
  here returned an empty string, so the two rules give identical counts). pySBD's own script
  compares unstripped output.
- **Offset boundary P/R/F1**: each sentence boundary is an offset counted in
  non-whitespace characters; the end of the text is excluded (it is not a decision).
  TP/FP/FN are summed over all cases (micro-average).
- **yasbd F1**: a re-implementation of yasbd-lib's `benchmarks/scorer.py` (word-level 0/1
  arrays, right-aligned, end of text counted as a boundary). Included so the numbers can be
  checked against the published table. Counting the final boundary inflates this metric.
- **Warm time**: one untimed warm-up, then 20 timed passes over the whole set in one
  process; median pass time in milliseconds.
- **Cold start**: 5 fresh interpreter processes, each timing `import` + segmenter
  construction + the first call on case 1 with `time.perf_counter`; median. Interpreter
  startup itself is excluded.

Library calls:

| Label | Call |
|---|---|
| nupunkt default | `nupunkt.sent_tokenize(text)` |
| nupunkt adaptive | `nupunkt.sent_tokenize(text, adaptive=True)` |
| nupunkt sentences() | `nupunkt.sentences(text)` (segmentation interface; on main it is layout-aware by default, blank lines are hard boundaries, see `docs/layout.md`) |
| nupunkt opt-in `!?` rule | fresh `nupunkt.models.load_default_model()` with `EXCL_QUEST_LOWERCASE_CONTINUES = True` (main only; off by default) |
| nltk punkt_tab | `nltk.sent_tokenize(text)` |
| pysbd | `pysbd.Segmenter(language="en", clean=False).segment(text)` |
| sentencex | `sentencex.segment("en", text)` |
| blingfire | `blingfire.text_to_sentences(text).split("\n")` |
| spacy sentencizer | `spacy.blank("en")` + `add_pipe("sentencizer")`, `[s.text for s in nlp(text).sents]` |
| yasbd | `yasbd.boundary_detector.BoundaryDetector(lang="en").segment(text)` |

## Environment

- CPU: 12th Gen Intel Core i7-12700K; OS: Linux 6.18.0-9-generic x86_64 (glibc 2.43)
- Python 3.13.15 (uv-managed virtualenvs)
- nupunkt 0.7.0 (release wheel `dist/nupunkt-0.7.0-py3-none-any.whl`); nupunkt main
  (unreleased: commit `7befa44` plus uncommitted working-tree changes, run with the
  project's `.venv`, labelled "main (unreleased)"); nupunkt 0.6.0 (PyPI, separate venv); nltk 3.10.3 with `punkt_tab`, pysbd 0.3.4, sentencex 1.0.31,
  blingfire 0.1.8, spacy 3.8.16, yasbd-lib 1.0.1 (PyPI)
- sentsplit and sentence-splitter, which appear in the yasbd table, were not installed.

Timing caveat: other benchmark processes were running on the same machine. The full run was
done twice for released versions; timing columns show `run 1 / run 2`, and timings moved
by up to 2x between runs. The main-tree rows are from a single later run. Accuracy numbers
were identical across all runs.

## Results

### pysbd-48

| Library | Pass | % | Offset F1 (P / R) | yasbd F1 | Warm ms | Cold ms |
|---|---|---|---|---|---|---|
| pysbd 0.3.4 | 47/48 | 97.9 | 0.985 (0.970 / 1.000) | 0.994 | 5.86 / 5.69 | 23 / 40 |
| yasbd 1.0.1 | 45/48 | 93.8 | 0.909 (0.882 / 0.938) | 0.963 | 2.61 / 5.51 | 56 / 95 |
| sentencex 1.0.31 | 43/48 | 89.6 | 0.818 (0.794 / 0.844) | 0.926 | 0.04 / 0.08 | 17 / 30 |
| blingfire 0.1.8 | 36/48 | 75.0 | 0.611 (0.550 / 0.688) | 0.833 | 0.13 / 0.20 | 51 / 82 |
| nupunkt main, opt-in `!?` rule | 36/48 | 75.0 | 0.688 (0.688 / 0.688) | 0.875 | 0.22 | 15 |
| **nupunkt main default** | 35/48 | 72.9 | 0.677 (0.667 / 0.688) | 0.870 | 0.22 | 15 |
| nupunkt main adaptive | 35/48 | 72.9 | 0.677 (0.667 / 0.688) | 0.870 | 0.76 | 16 |
| nupunkt main sentences() | 35/48 | 72.9 | 0.677 (0.667 / 0.688) | 0.870 | 0.36 | 15 |
| **nupunkt 0.7.0 default** | 34/48 | 70.8 | 0.667 (0.647 / 0.688) | 0.864 | 0.24 / 0.21 | 14 / 14 |
| nupunkt 0.7.0 adaptive | 34/48 | 70.8 | 0.667 (0.647 / 0.688) | 0.864 | 0.69 / 0.71 | 15 / 15 |
| nupunkt 0.6.0 adaptive | 29/48 | 60.4 | 0.575 (0.512 / 0.656) | 0.817 | 0.82 / 0.81 | 1351 / 1546 |
| nupunkt 0.6.0 default | 27/48 * | 56.2 | 0.571 (0.489 / 0.688) | 0.809 | 0.65 / 0.69 | 1388 / 1529 |
| nltk punkt_tab 3.10.3 | 27/48 | 56.2 | 0.463 (0.349 / 0.688) | 0.733 | 0.55 / 0.50 | 134 / 160 |
| spacy sentencizer 3.8.16 | 25/48 | 52.1 | 0.479 (0.436 / 0.531) | 0.778 | 0.76 / 0.81 | 357 / 641 |

### yasbd-92

| Library | Pass | % | Offset F1 (P / R) | yasbd F1 | Warm ms | Cold ms |
|---|---|---|---|---|---|---|
| yasbd 1.0.1 | 91/92 | 98.9 | 0.991 (1.000 / 0.982) | 0.997 | 5.09 / 10.31 | 56 / 95 |
| pysbd 0.3.4 | 77/92 | 83.7 | 0.843 (0.773 / 0.927) | 0.938 | 11.51 / 11.32 | 23 / 40 |
| sentencex 1.0.31 | 77/92 | 83.7 | 0.797 (0.746 / 0.855) | 0.921 | 0.08 / 0.16 | 17 / 30 |
| blingfire 0.1.8 | 75/92 | 81.5 | 0.744 (0.682 / 0.818) | 0.898 | 0.26 / 0.58 | 51 / 82 |
| nupunkt main, opt-in `!?` rule | 74/92 | 80.4 | 0.793 (0.786 / 0.800) | 0.922 | 0.42 | 15 |
| **nupunkt main default** | 72/92 | 78.3 | 0.759 (0.721 / 0.800) | 0.907 | 0.45 | 15 |
| nupunkt main adaptive | 72/92 | 78.3 | 0.759 (0.721 / 0.800) | 0.907 | 1.50 | 16 |
| nupunkt main sentences() | 72/92 | 78.3 | 0.759 (0.721 / 0.800) | 0.907 | 0.70 | 15 |
| **nupunkt 0.7.0 default** | 67/92 | 72.8 | 0.705 (0.642 / 0.782) | 0.882 | 0.44 / 0.43 | 14 / 14 |
| nupunkt 0.7.0 adaptive | 67/92 | 72.8 | 0.705 (0.642 / 0.782) | 0.882 | 1.41 / 1.46 | 15 / 15 |
| nupunkt 0.6.0 adaptive | 61/92 | 66.3 | 0.620 (0.541 / 0.727) | 0.843 | 1.56 / 1.57 | 1351 / 1546 |
| nupunkt 0.6.0 default | 57/92 * | 62.0 | 0.613 (0.512 / 0.764) | 0.835 | 1.24 / 1.30 | 1388 / 1529 |
| nltk punkt_tab 3.10.3 | 51/92 | 55.4 | 0.503 (0.367 / 0.800) | 0.758 | 1.00 / 0.98 | 134 / 160 |
| spacy sentencizer 3.8.16 | 51/92 | 55.4 | 0.559 (0.469 / 0.691) | 0.819 | 1.60 / 1.74 | 357 / 641 |

Reproduction check: the published yasbd-lib pass counts and F1 values for yasbd, pysbd,
sentencex, blingfire and spacy match these to the stated precision. pySBD's published
97.92% and NLTK's 56.25% on the 48-case set also match.

\* **nupunkt 0.6.0 results depend on what was tokenized earlier in the process.** Scoring
each case in a fresh process gives 29/48 and 61/92. Running the 92-case set first in a fresh
process gives 59/92 (the published figure). Running it after the 48-case set gives 57/92:
tokenizing `Let's ask Jane and co. They should know.` changes the later output for
`Were Jane and co. at the party?` from one sentence to two. Over 10 shuffled orderings in
one process, 0.6.0 produced 2 distinct outputs; 0.7.0 and main produced 1.

Other observations:

- Adaptive mode and `sentences()` produce exactly the same output as `sent_tokenize` on
  every case, for both 0.7.0 and main. No case contains a blank line, so the layout layer
  in main's `sentences()` never applies here. Adaptive costs about 3x the warm time.
- The opt-in `EXCL_QUEST_LOWERCASE_CONTINUES` fixes 3 cases (P41, Y40, Y74) and breaks none
  on these sets. It is off by default because it was measured to cost about 1 point of
  boundary F1 on UD English EWT (informal web text, where sentences often start lowercase);
  that measurement is not part of this benchmark.
- Ranking: nupunkt 0.7.0 and main default are 5th of 8 installed libraries on both sets.
  pysbd, yasbd, sentencex and blingfire score higher on both. With the opt-in rule, main ties
  blingfire on pysbd-48 (36/48) and is one case behind it on yasbd-92.
- yasbd-lib's README describes a ~2.6 s nupunkt cold start. That matches 0.6.0 (1.3-1.5 s
  here, plus interpreter startup). 0.7.0 and main measured 14-16 ms.

### Regressions in 0.7.0 and their status on main

Relative to 0.6.0 default (as run by the script), 0.7.0 fixed 8 pySBD and 13 yasbd cases
and regressed 4 case ids (3 distinct inputs; P47 and Y46 are the same text). All reproduce
with the 0.7.0 wheel:

| Case | 0.7.0 output | main |
|---|---|---|
| P47 / Y46 `... the thing is . . . I didn’t mean it.` | split after `. . .` | fixed (pronoun `I` after an ellipsis no longer breaks) |
| Y73 `What I'm saying, the thing is . . . I didn't mean it.` | split after `. . .` | fixed (same rule) |
| Y91 `The meeting is at 2 p.m. 请别迟到。` | one sentence | **still fails** (CJK, class H) |

## nupunkt failures

Case ids are `P` = pysbd-48, `Y` = yasbd-92. Default, adaptive and `sentences()` fail the
same cases. Counts are default mode; the opt-in `!?` rule changes class D only.

| Class | P 0.7.0 | P main | Y 0.7.0 | Y main | Y main + opt-in |
|---|---|---|---|---|---|
| A. Inline list items | 8 | 8 | 6 | 6 | 6 |
| B. Abbreviation before a capitalized sentence start | 2 | 2 | 3 | 3 | 3 |
| C. Abbreviation list gap or collision | 1 | 1 | 2 | 1 | 1 |
| D. `!` / `?` inside a sentence | 1 | 1 | 5 | 5 | 3 |
| E. Spaced terminator runs | 0 | 0 | 2 | 0 | 0 |
| F. Ellipsis before a capitalized word | 2 | 1 | 2 | 0 | 0 |
| G. Multi-sentence quotation or parenthetical kept whole | 0 | 0 | 2 | 2 | 2 |
| H. CJK text | 0 | 0 | 3 | 3 | 3 |
| **Total** | **14** | **13** | **25** | **20** | **18** |

Fixed on main since 0.7.0: P47, Y46, Y73 (ellipsis + `I`), Y69, Y70 (spaced `!`/`?` runs are
one terminator run), Y83 (closing quote after `me.`; closing punctuation is now transparent
when pairing). No case that passes with 0.7.0 fails on main.

**A. Inline list items** (P31-33, P35-39; Y32, Y33, Y35-38). Example P35:
`1. The first item 2. The second item` -> expected `[1. The first item | 2. The second item]`,
got `[1. The first item 2. | The second item]`. P36 gives `[1. The first item. | 2. | The second item.]`;
P37 gives `[• 9. | The first item • 10. | The second item]`; P39 (`a. ... b. ... c. ...`)
gives one sentence. Items without terminal punctuation give Punkt no boundary signal, and
single letters are read as initials: design limitation. The standalone `2.` / `2.)` / `• 9.`
fragments are fixable: nupunkt already refuses enumerator-only sentences at line start, and
the rule could cover enumerators after a terminator mid-line.

**B. Abbreviation followed by a capitalized sentence start** (P18/Y19, P42/Y41, Y53).
`He left the bank at 6 P.M. Mr. Smith then went to the store.` -> no split after `P.M.`;
`We make a good team, you and I. Did you see ...` -> no split after `I.`;
`Our office is at 1600 Pennsylvania Ave. Hours are ...` -> no split after `Ave.` (yasbd
also fails this one). A known abbreviation followed by a capitalized word is the core Punkt
ambiguity. Narrow deterministic rules are possible (time abbreviation followed by a title;
`I` after a conjunction), but each can add false splits in legal text. Partly fixable.

**C. Abbreviation list gap or collision.** P40/Y39 (still failing):
`You can find it at N°. 1026.253.553. That is where ...` -> `[... N°. | 1026.253.553. | That is ...]`;
`N°` is not a known abbreviation (gap, fixable). Y83 (fixed on main): with 0.7.0,
`... warning: 'Do not follow me.' Then she left.` stayed one sentence because `me` is in the
bundled abbreviation list (Maine) and the closing quote let that reading win.

**D. `!` / `?` inside a sentence** (P41/Y40, Y74, Y75, Y80, Y81).
`She works at Yahoo! in the accounting department.` -> `[... Yahoo! | in the accounting department.]`.
Punkt treats `!` and `?` as unconditional terminators. P41, Y40 and Y74 pass with the opt-in
lowercase-continuation rule, which is off by default (see above). Not fixed by the opt-in:
`... Where's Wally? series of books (published in the US as Where's Waldo?)?` still leaves a
`)?` fragment; `... in 1985 [or 1984?], but he wasn't ...` splits before `],` in default
mode. Those fragments are bugs and fixable. `Yum! Brands` (Y75, capitalized next word) is not
fixable without a brand lexicon.

**E. Spaced terminator runs** (Y69, Y70). 0.7.0 produced punctuation-only sentences
(`[Hello ! | ! | ! | !]`); main passes both.

**F. Ellipsis before a capitalized word** (P47/Y46, Y73 fixed on main; P48 still failing).
P48 expects `. . .` to begin the next sentence
(`[... compounds. | . . . The practice was not abandoned. . . .]`); nupunkt attaches it to
the previous sentence. yasbd-lib removed that expectation as unrealistic. Other capitalized
words after a spaced ellipsis still end the sentence on main (`. . . Then it ended.`); these
sets do not test that.

**G. Quotation or parenthetical with several sentences** (Y77, Y84).
`(See Fig. 4. This outlines the memory layout.) The engine executes next.` and
`He said: "First sentence. Second sentence." Then done.` -> nupunkt splits inside the
brackets/quotes; the set expects the bracketed span kept whole. This is a convention choice.
Bracket-depth tracking is deterministic, but it would change output on legal block quotes.

**H. CJK** (Y90, Y91, Y92). `我喜欢AI。 It is useful` -> one sentence. `。` is not a terminator
in nupunkt's language variables, and case-based heuristics do not apply to uncased scripts
(Y91: `2 p.m. 请别迟到。`, which 0.6.0 passed). Adding CJK terminators is deterministic and
fixable, but it is outside the English/legal scope of the bundled model.

Summary for the 20 main-tree failures on yasbd-92: 6 look fixable with deterministic rules
inside Punkt (A fragments Y33/Y36/Y37, C `N°`, D fragments Y80/Y81); 2 are fixed by the
existing opt-in (Y40, Y74); 9 are ambiguous or convention-dependent (unpunctuated lists
Y32/Y35/Y38, class B, `Yum! Brands`, class G) and would need structure beyond Punkt or would
trade against legal-text precision; 3 are CJK.

## What this benchmark does and does not measure

- It measures hand-picked edge cases (48 or 92 short strings), not a corpus sample; pass
  rates do not estimate error rates on running text.
- The expected outputs encode one set of conventions (e.g. inline lists split per item,
  quoted sentences kept whole, `...` never splits before `I`). Other gold standards,
  including legal gold sets, make different choices.
- The yasbd-92 set was assembled by the author of one of the compared libraries, which scores
  91/92 on it; the pySBD set was assembled by pySBD's author. That does not make the cases
  wrong, but case selection is not neutral.
- nupunkt's bundled model and heuristics target legal and formal English. It has no CJK
  support and no brand-name lexicon.
- Warm times are for tiny inputs (per-call overhead), not throughput on large documents.
- Exact match fails a whole case for one wrong boundary; boundary F1 is the per-boundary view.

## Reproduction

nupunkt only, with the project environment (the main-tree rows above):

```bash
.venv/bin/python scripts/benchmarks/golden_rules.py --failures --nupunkt-label "main (unreleased)"
```

With comparators (throwaway venv; run the script file rather than `python -c` from the repo
root, so the installed nupunkt is imported rather than the source tree):

```bash
uv venv -p 3.13 /tmp/grs && uv pip install -p /tmp/grs/bin/python \
  nupunkt==0.7.0 nltk pysbd sentencex blingfire spacy yasbd-lib \
  && /tmp/grs/bin/python -c "import nltk; nltk.download('punkt_tab')" \
  && /tmp/grs/bin/python scripts/benchmarks/golden_rules.py --with-comparators --failures --json grs.json
```

Libraries that fail to import are skipped with a message on stderr.
