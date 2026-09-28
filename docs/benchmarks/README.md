# Benchmarks and evidence

Reproducible measurements behind the claims nupunkt makes, and honest answers
to the criticisms it has received in public. Each page states its
methodology, hardware and library versions, shows where nupunkt loses, and ends
with a one-command reproduction using the scripts in `scripts/benchmarks/`.

| Page | Question it answers | Script |
|---|---|---|
| [comparison.md](comparison.md) | Pareto analysis against six other splitters on accuracy, speed, startup and dependency footprint, with recommendations by use case | all of the below |
| [accuracy.md](accuracy.md) | Boundary precision, recall and F1 on a legal gold set (38.5k documents) and on general English (UD EWT, UD GUM, Brown), against NLTK, pysbd, sentencex, blingfire and spaCy; where recall is lost; determinism | `accuracy.py`, `build_gold_sets.py` |
| [performance.md](performance.md) | Cold start, model size, memory and throughput versus earlier nupunkt versions, other libraries and a regex baseline | `performance.py` |
| [golden-rules.md](golden-rules.md) | Pass rates on pySBD's Golden Rules and on yasbd-lib's 92-case set, with every failure classified | `golden_rules.py` |
| [known-cases.md](known-cases.md) | Every input reported publicly as a nupunkt failure, run through 0.6.0, 0.7.0 and the current tree, with what is still open | `known_cases.py` |
| [../layout.md](../layout.md) | What happens at blank lines, headings, list items and page breaks in scanned documents, and what other libraries do | `accuracy.py --with-layout` |

## Criticisms addressed

| Public claim | Source | Status |
|---|---|---|
| Output depends on what was tokenized earlier | found during the 0.7.0 audit | fixed in 0.7.0; verified by tokenizing the gold set in two orders |
| "11-second" / "2.6 s" / "2.1 s" / "3.1 s" cold start | yasbd-lib, homogenous-cluster, openreview-cli | 0.7.0: import plus first call 14 ms in a fresh process; see performance.md |
| "5.3x slower than a regex splitter", "20-60x slower than sentencex/blingfire", "Speed Liability" | redlines #48, yasbd-lib | still partly true: on unseen text a regex split is 4.5x faster, sentencex 5-9x, blingfire 1.3x; NLTK is about equal; 0.7.0 is 2x 0.6.0 and equal to 0.5.1; see performance.md |
| "432 MB" memory, "9 MB" model | fast-sentence-segment #16, neuro-san-studio | 25 KB model, about 19 MB resident; see performance.md |
| 59/92 on the yasbd golden set | yasbd-lib benchmarks | 0.6.0 reproduces at 57-61 depending on run order; 0.7.0 and the current tree score higher; see golden-rules.md |
| "Prioritizes precision over recall" | eyecite #249 | accurate for Punkt-only mode; 94% of misses have no terminal punctuation. The layout default recovers most of them; see accuracy.md and layout.md |
| Numbered clauses, `Sr.`, nested quotes, dialog | scout #30, yasbd-lib, KnowSeams | partly fixed; remaining cases listed in known-cases.md |
| `math domain error` during training, private `_params` access | institutional-books pipeline | error guarded in 0.7.0; public `abbreviations` and `parameters` accessors added |
| CJK, no-space-after-period | yasbd-lib, NLTK #2082 | not supported; stated in known-cases.md |

## Reproduction

```bash
python scripts/benchmarks/build_gold_sets.py --download      # UD EWT, UD GUM, Brown
python scripts/benchmarks/accuracy.py --with-comparators --with-layout
python scripts/benchmarks/performance.py --with-comparators
python scripts/benchmarks/golden_rules.py --with-comparators
python scripts/benchmarks/known_cases.py
```

Comparator libraries are installed into throwaway virtual environments by the
scripts' `--with-comparators` mode; nupunkt itself has no runtime dependencies.
