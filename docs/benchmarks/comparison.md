# Choosing a sentence splitter: nupunkt against the alternatives

This page pulls the numbers from [accuracy.md](accuracy.md),
[performance.md](performance.md) and [golden-rules.md](golden-rules.md) into one
comparison, adds the dependency footprint of each library, states where nupunkt
is and is not the best choice, and gives recommendations by use case. It is
written to be linked from discussions, so it errs on the side of saying where
nupunkt loses.

Versions: nupunkt (current tree, layout default; 0.7.0 where noted), NLTK 3.10.3
with `punkt_tab`, pysbd 0.3.4, sentencex 1.0.31, blingfire 0.1.8, spaCy 3.8.16
`sentencizer`, syntok, yasbd-lib 1.0.1. CPython 3.13 on Linux x86-64, one core.

## The axes

### Accuracy (boundary F1)

| Gold set | nupunkt | sentencex | pysbd | blingfire | NLTK | spaCy |
|---|--:|--:|--:|--:|--:|--:|
| Legal, paragraph markers counted | **0.932** | 0.893 | 0.875 | 0.657 | 0.650 | 0.621 |
| Legal, sentence markers only | **0.822** | 0.783 | 0.767 | 0.719 | 0.705 | 0.674 |
| UD EWT (web), paragraphs kept | **0.919** | 0.904 | 0.898 | 0.834 | 0.867 | 0.848 |
| UD GUM (12 genres), paragraphs kept | 0.972 | **0.979** | 0.967 | 0.935 | 0.927 | 0.919 |
| UD EWT, flat | **0.867** | 0.822 | 0.821 | 0.834 | **0.867** | 0.845 |
| UD GUM, flat | 0.924 | 0.922 | 0.898 | **0.930** | 0.927 | 0.901 |
| Brown (15 genres), flat | 0.944 | 0.881 | 0.847 | **0.948** | 0.945 | 0.867 |
| yasbd 92 golden-rule strings (passes) | 72 | 77 | 77 | 75 | 51 | 51 |

yasbd-lib passes 91 of the 92 golden-rule strings, which its author assembled;
it was not run on the corpora.

### Speed and startup

| | nupunkt 0.7.0 | sentencex | pysbd | blingfire | NLTK | spaCy | syntok |
|---|--:|--:|--:|--:|--:|--:|--:|
| Unseen legal text, Mchar/s | 23 | **113** | 0.4 | 31 | 20 | 1.3 | 2.6 |
| Import + first call, fresh process | **14 ms** | 17-30 ms | 23-40 ms | 51-82 ms | 134-160 ms | 357-641 ms | n/a |
| Resident memory after load | **20 MB** | not measured | not measured | not measured | not measured | not measured | not measured |

A one-line regex split (`(?<=[.!?])\s+`) runs at 105 Mchar/s. On repeated
input nupunkt memoizes decisions and reaches 43-68 Mchar/s; the table shows the
unseen-text figure because that is what a pipeline sees.

### Dependency footprint

Measured by resolving each library alone for CPython 3.13 on Linux x86-64:
the full transitive set of wheels that `pip download` fetches, the depth of the
dependency tree (`uv pip tree`), and what a fresh install contains.

| | Transitive packages | Tree depth | Download (all wheels) | Installed | Compiled code | Runtime download |
|---|--:|--:|--:|--:|---|---|
| **nupunkt 0.7.0** | **1** | **1** | **0.14 MB** | **1 MB** | **none, pure Python** | **none** |
| pysbd 0.3.4 | 1 | 1 | 0.07 MB | 1 MB | none, pure Python | none |
| sentencex 1.0.31 | 1 | 1 | 1.2 MB | 5 MB | C extension (`.so`) | none |
| syntok | 2 | 2 | 0.8 MB | 4 MB | `regex` C extension | none |
| yasbd-lib 1.0.1 | 5 | 2 | 2.2 MB | 9 MB | `regex` C extension | none |
| NLTK 3.10.3 | 7 | 3 | 3.1 MB | 13 MB | `regex` C extension | `punkt_tab` via `nltk.download` |
| blingfire 0.1.8 | 1 (+ `numpy`, imported but undeclared) | 1 | 41 MB | 99 MB | C++ shared library, 18 bundled model binaries | none |
| spaCy 3.8.16 | 44 | 5 | 75 MB | 258 MB | 90 compiled files | none for `sentencizer` |

"Pure Python, zero dependencies" is a property with consequences beyond size:
one wheel serves every platform and interpreter, there is nothing to build on
an unsupported architecture, no transitive package can introduce a CVE or a
license question, an air-gapped or serverless deployment needs no download
step, and the whole library can be read and audited in an afternoon. Two of the
libraries above have it: nupunkt and pysbd.

### Other properties

| | nupunkt | sentencex | pysbd | blingfire | NLTK |
|---|---|---|---|---|---|
| Character spans | yes, tight; contiguous on request | not checked | yes (`char_span`) | yes (offsets API) | yes (`span_tokenize`) |
| Deterministic across calls | yes (verified on 38.5k documents, two orders) | not checked | not checked | not checked | yes |
| Trainable on your own text | yes, unsupervised | no | no | no | yes |
| Languages | English models bundled; trainable | many | many | many | many via `punkt_tab` |
| Layout-aware (blank lines, page breaks) | yes, configurable | newline splitting | newline splitting | no | no |

## Pareto analysis

Taking accuracy on each gold set, throughput, cold start, and dependency
footprint as the axes:

- **nupunkt is not dominated by any library.** Every library that beats it on
  one axis loses to it on several others.
- **nupunkt dominates NLTK, pysbd, spaCy's sentencizer and syntok**: equal or
  better accuracy on every set (the one exception is Brown, where NLTK is
  0.001 higher), faster or equal throughput, faster startup, and an equal or
  smaller footprint. Against pysbd, the only other pure-Python zero-dependency
  splitter, nupunkt is better on every set and about 50x faster.
- **sentencex is not dominated by nupunkt**: it is 5x faster, 0.007 higher on
  GUM with paragraphs, and multilingual. nupunkt beats it by 0.04 on legal,
  0.015 on EWT, 0.06 on Brown, and starts faster; sentencex ships a compiled
  extension.
- **blingfire is not dominated by nupunkt**: 1.3x faster, and 0.004 to 0.006
  higher on Brown and flat GUM. nupunkt beats it by 0.04 to 0.28 on every
  paragraph-bearing set, starts 4x faster, and installs in 1 MB against 99 MB
  of C++ and bundled models.
- **Among pure-Python, zero-dependency splitters, nupunkt is Pareto dominant.**

The accuracy gaps where nupunkt loses are all below 0.01 and inside the
uncertainty of Brown's approximate detokenization. The gaps where it wins are
0.03 to 0.28. The one axis with a real deficit is raw throughput against
compiled code, and pure Python will not close a 5x gap; the Rust port
[nupunkt-rs](https://github.com/alea-institute/nupunkt-rs) exists for that.

## Recommendations by use case

| Use case | Recommendation | Why |
|---|---|---|
| Legal, regulatory or financial documents (opinions, filings, contracts, OCR'd PDFs) | **nupunkt** | Highest accuracy on legal text by a wide margin; abbreviation and citation handling; layout rules handle headings, numbered clauses and page breaks; trainable on your corpus |
| Chunking structured documents for retrieval (RAG) | **nupunkt** with the default layout rules, `segment()` for the paragraph/sentence/word tree | Headings and list items become their own units; tight spans index the source; one pass per document; 20 MB resident, 14 ms start |
| Deployment where dependencies matter: serverless, air-gapped, security-reviewed, many platforms, or a library that must not drag in packages | **nupunkt** | The only splitter here that is pure Python, zero dependency, and competitive on accuracy and speed. pysbd shares the footprint but is 50x slower and less accurate |
| General English prose with paragraph structure (web, news, essays) | **nupunkt** or sentencex | Within 0.01 of each other on GUM; nupunkt ahead on EWT and Brown; pick sentencex if you also need its languages or its speed |
| Maximum throughput on large plain-text corpora, English | sentencex or blingfire, or nupunkt-rs | 5x and 1.3x faster than pure-Python nupunkt; accept their compiled code and, for blingfire, 99 MB |
| Many languages in one pipeline | sentencex, pysbd or blingfire | nupunkt ships English models only; training your own per language is possible but is work |
| Fiction, dialogue, informal short strings, punctuation edge cases | yasbd-lib or pysbd, or nupunkt with `adaptive=True` checked on your data | The golden-rule sets favour rule engines built for them; nupunkt passes 72 of 92 and its remaining failures are listed in known-cases.md |
| Hard-wrapped plain text (email, old text files) | nupunkt with `paragraph_breaks=True, line_breaks=False` (the default), or NLTK | Keep the line rule off; the blank-line rule still helps |
| Text with no layout and no sentence-final punctuation (transcripts, chat) | none of these; use a model-based splitter such as wtpsplit/SaT | Every library here needs punctuation or layout to find boundaries |
| You already depend on NLTK or spaCy for other reasons | switching to nupunkt still pays on legal text and startup; on general prose the accuracy gain over NLTK is small | NLTK ties nupunkt on flat EWT and Brown; the difference is legal text, layout, startup and the `punkt_tab` download |

## Caveats

- The legal gold set is nupunkt's home domain, and its conventions were used
  during development. The general-English sets were used to validate the 0.7.0
  heuristics and the layout default, so they are not fully held out either.
- Comparators were run with their defaults and English settings. Someone tuning
  sentencex or pysbd for a domain may do better than these tables show.
- Brown is detokenized approximately; differences under 0.01 there are noise.
- Speed figures are single-core, one machine, medians of five; see
  performance.md for the load conditions and the repeated-input caveat.
- yasbd-lib was only run on the golden-rule strings, not on the corpora.

Reproduce with the scripts named in [README.md](README.md).
