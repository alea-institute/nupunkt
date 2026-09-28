# Sentence-boundary accuracy on gold corpora

This page reports boundary precision, recall and F1 for nupunkt and for five other sentence
splitters on one legal gold set and three general-English gold sets. It is meant to be
checked, not believed: every number comes from `scripts/benchmarks/accuracy.py` (one
command, see [Reproduction](#reproduction)). The hand-built edge-case benchmark is in
[golden-rules.md](golden-rules.md).

Rows labelled **nupunkt main (unreleased)** come from the current working tree, which
includes the layout-aware segmentation interface ([docs/layout.md](../layout.md)) and the
Unreleased changes in the CHANGELOG. Rows labelled **nupunkt 0.7.0** come from the release
on PyPI. Run on 2026-09-27, Python 3.13, Linux. No timing is reported here.

## Summary

- **Punkt favours precision over recall.** On the legal set, `span_tokenize` (0.7.0 and main
  alike) has precision 0.90 and recall 0.69. 94% of its misses follow no sentence-final
  punctuation: headings and other unpunctuated lines (62%), list items (12%), semicolons
  (10%), colons (4%). Where the gold boundary follows `.!?`, recall is 0.975.
- **The layout default in main recovers most of that recall.** `sentences()` /
  `iter_spans()` now treat blank lines as boundaries. Recall rises by 8–33 points on every
  set that has blank lines, and precision stays the same or improves on every set except
  `legal`. Its F1 is 0.932 on legal with paragraph markers (was 0.716), 0.919 on EWT with
  paragraphs (was 0.869) and 0.972 on GUM with paragraphs (was 0.928). The exception is
  `legal` scored on sentence markers only: that gold treats headings as paragraph ends
  rather than sentence ends, so precision falls from 0.901 to 0.757. F1 still rises there,
  from 0.784 to 0.822.
- **Where layout marks sentences, other libraries still win narrowly.** On GUM with
  paragraphs sentencex scores 0.979 against 0.972. pysbd has the highest recall on `legal`
  and `legal_pm`.
- **On flat, well-punctuated English, nupunkt is roughly tied with NLTK Punkt and
  blingfire.** NLTK is within 0.003 F1 of Punkt-only nupunkt on EWT, GUM and Brown. blingfire is ahead on GUM and Brown
  (0.930 / 0.935 / 0.948 against 0.924 / 0.928 / 0.944). nupunkt's clear lead at `.!?`
  boundaries is on the legal set, which is its training domain.
- **The adaptive threshold only buys precision.** A higher threshold gives more breaks, as
  the corrected docstring now says. Even at t=0.9, adaptive mode only matches the plain
  tokenizer, and every lower threshold trades recall for precision.
- **nupunkt's sentences run longer than gold, not shorter.** Punkt-only nupunkt averages
  16.5–19.6 words per sentence against 13.3–17.7 in the gold. This bears on the
  Institutional Books comparison below.
- **Output is order-independent.** Tokenizing the legal set forward and in reverse gives
  identical boundaries for main (both modes), 0.7.0 and 0.5.1. 0.6.0 differs on 4,958 of
  38,527 documents.

## Criticisms this page responds to

- freelawproject/eyecite#249: nupunkt "prioritizes precision over recall"
  ([comment](https://github.com/freelawproject/eyecite/issues/249#issuecomment-2840480639)),
  and it "wasn't a good fit" for semantic search
  ([comment](https://github.com/freelawproject/eyecite/issues/249#issuecomment-2836856486)).
  For the Punkt tokenizer the data agree. See
  [Where recall is lost](#where-recall-is-lost-legal-set) for what the layout default changes.
- Institutional Books – Enriched Text, [arXiv 2608.19026](https://arxiv.org/abs/2608.19026),
  §4.8.2, Table 9: 23.3 words per sentence for nupunkt against 36.7 for SaT, with 113.2
  against 119.5 characters per sentence. "Whether this reflects language conventions or
  segmenter differences is unclear." See [Sentence length](#sentence-length-and-the-sat-comparison).
- yasbd-lib [benchmarks README](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/README.md)
  calls nupunkt "over-aggressive" in some tests and measures 77.9% boundary precision on its
  92-case English golden set. It also quotes the nupunkt README's "91.1% precision". See
  [The 91.1% precision figure](#the-911-precision-figure).

## Method

**Boundary.** A boundary is the character offset just after the last non-whitespace
character of a sentence. The end of each document is never scored. Precision, recall and F1
are micro-averaged over all boundaries in a set. Systems that return strings are aligned to
the input by counting non-whitespace characters, so whitespace normalisation cannot move a
boundary. There were no alignment failures.

**Gold sets.**

| set | source | docs | text and offsets |
|---|---|---|---|
| `legal` | `data/test.jsonl.gz` (this repo) | 38,527 | `<|paragraph|>` becomes `"\n\n"`. Each `<|sentence|>` marks a boundary at the right-stripped end of the preceding text. Documents are excerpts that start and end mid-sentence |
| `legal_pm` | same texts | 38,527 | As `legal`, but `<|paragraph|>` also counts as a boundary. 25,545 of the 81,548 paragraph markers have no sentence marker (headings, captions, navigation, table cells) |
| `ewt_flat` / `ewt_para` | UD English-EWT r2.18, all splits | 1,174 | Verbatim `# text =` sentences joined by one space. `para` joins paragraphs (`# newpar`) with `"\n\n"`, `flat` with one space |
| `gum_flat` / `gum_para` | UD English-GUM r2.18, all splits | 257 | As EWT. 15 genres. The Reddit subset is not in the UD release |
| `brown_flat` | Brown corpus (NLTK `brown.zip`) | 500 | Tokens detokenized by rule, sentences joined by one space. No paragraph information |

`flat` sets contain no layout cues. The **Brown detokenization** is approximate: `` ``/'' ``
become `"`, punctuation is attached by rule, and single-quote direction and the original
spacing around `--` and ellipses are lost. About 6,000 Brown sentences end in a doubled token
(`;/. ;/.`), which `build_gold_sets.py` collapses to one. Brown treats at least 2,783
semicolons as sentence ends, which Punkt-style splitters do not reproduce.

**Systems.**

- **main:** `load("default").span_tokenize` (Punkt only, the `sent_tokenize` family).
  `iter_spans(text, paragraph_breaks=False)` gave identical output on every set, so it
  appears once below as "Punkt-only".
- **main, layout default:** `iter_spans(text)`. A blank line is a hard boundary, except
  where the sentence continues across it: no terminal punctuation before the gap, and a
  lowercase letter or a hyphenated word break after it (the page-break rule).
- **main, layout + lines:** `iter_spans(text, line_breaks=True)`, which also cuts after
  short heading-like lines and before list-marker lines.
- **main, adaptive:** `sent_spans_adaptive(threshold=0.7 | 0.5)`.
- **Other releases:** nupunkt 0.7.0 and 0.6.0 via `load("default").span_tokenize`, and
  0.5.1 via `sent_tokenize` (0.5.1 has no `load`).
- **Other libraries:** NLTK 3.10.3 `PunktTokenizer("english")` (punkt_tab), pysbd 0.3.4
  (`clean=False`), sentencex 1.0.31, blingfire 0.1.8 (`text_to_sentences_and_offsets`) and
  spaCy 3.8.16 (`blank("en")` plus the `sentencizer`).

Every comparator installed and none was skipped. Each version runs in its own `uv` venv.
SaT/wtpsplit was not tested because it needs a neural-network runtime.

"F1 at `.!?`" restricts gold and predicted boundaries to those whose preceding character,
after skipping closers (`" ' ) ] } ’ ” »`), is `.`, `!` or `?`. The "share" row is the
fraction of gold boundaries that qualify, which is roughly the recall ceiling for a splitter
that only splits at `.!?`.

## Results

### Boundary F1 and precision / recall

| system | legal | legal_pm | ewt_flat | ewt_para | gum_flat | gum_para | brown_flat |
|---|---|---|---|---|---|---|---|
| main Punkt-only | 0.784 | 0.716 | **0.867** | 0.869 | 0.924 | 0.928 | 0.944 |
| main layout default | 0.822 | 0.932 | **0.867** | 0.919 | 0.924 | 0.972 | 0.944 |
| main layout + line_breaks | **0.829** | **0.936** | **0.867** | **0.919** | 0.924 | 0.972 | 0.944 |
| main adaptive t=0.7 | 0.772 | 0.700 | 0.859 | 0.859 | 0.920 | 0.921 | 0.941 |
| main adaptive t=0.5 | 0.733 | 0.656 | 0.823 | 0.823 | 0.889 | 0.890 | 0.923 |
| nupunkt 0.7.0 | 0.782 | 0.714 | **0.867** | 0.869 | 0.924 | 0.927 | 0.943 |
| nupunkt 0.6.0 | 0.722 | 0.653 | 0.860 | 0.860 | 0.916 | 0.916 | 0.928 |
| nupunkt 0.5.1 | 0.734 | 0.664 | 0.845 | 0.845 | 0.896 | 0.896 | 0.934 |
| nltk 3.10.3 punkt_tab | 0.705 | 0.650 | **0.867** | 0.867 | 0.927 | 0.927 | 0.945 |
| pysbd 0.3.4 | 0.767 | 0.875 | 0.821 | 0.898 | 0.898 | 0.967 | 0.847 |
| sentencex 1.0.31 | 0.783 | 0.893 | 0.822 | 0.904 | 0.922 | **0.979** | 0.881 |
| blingfire 0.1.8 | 0.719 | 0.657 | 0.834 | 0.834 | **0.930** | 0.935 | **0.948** |
| spaCy 3.8.16 sentencizer | 0.674 | 0.621 | 0.845 | 0.848 | 0.901 | 0.919 | 0.867 |

| system (P / R) | legal | legal_pm | ewt_flat | ewt_para | gum_flat | gum_para | brown_flat |
|---|---|---|---|---|---|---|---|
| main Punkt-only | 0.901 / 0.694 | 0.932 / 0.582 | 0.984 / 0.775 | 0.985 / 0.777 | 0.975 / 0.879 | 0.977 / 0.883 | 0.993 / 0.900 |
| main layout default | 0.757 / 0.901 | 0.949 / 0.915 | 0.984 / 0.775 | 0.986 / 0.861 | 0.975 / 0.879 | 0.978 / 0.965 | 0.993 / 0.900 |
| main layout + line_breaks | 0.752 / 0.924 | 0.939 / 0.934 | 0.984 / 0.775 | 0.986 / 0.861 | 0.975 / 0.879 | 0.978 / 0.965 | 0.993 / 0.900 |
| main adaptive t=0.7 | 0.914 / 0.668 | 0.940 / 0.558 | 0.985 / 0.761 | 0.986 / 0.761 | 0.978 / 0.869 | 0.980 / 0.869 | 0.994 / 0.892 |
| main adaptive t=0.5 | 0.926 / 0.607 | 0.945 / 0.502 | 0.986 / 0.706 | 0.987 / 0.706 | 0.979 / 0.815 | 0.981 / 0.815 | 0.996 / 0.861 |
| nupunkt 0.7.0 | 0.902 / 0.691 | 0.932 / 0.578 | 0.983 / 0.776 | 0.984 / 0.778 | 0.975 / 0.878 | 0.977 / 0.882 | 0.993 / 0.898 |
| nupunkt 0.6.0 | 0.776 / 0.675 | 0.772 / 0.566 | 0.954 / 0.782 | 0.954 / 0.782 | 0.962 / 0.875 | 0.961 / 0.875 | 0.951 / 0.905 |
| nupunkt 0.5.1 | 0.874 / 0.632 | 0.898 / 0.526 | 0.975 / 0.746 | 0.975 / 0.746 | 0.975 / 0.829 | 0.975 / 0.829 | 0.997 / 0.878 |
| nltk 3.10.3 punkt_tab | 0.716 / 0.694 | 0.738 / 0.580 | 0.968 / 0.786 | 0.968 / 0.786 | 0.969 / 0.888 | 0.969 / 0.888 | 0.983 / 0.910 |
| pysbd 0.3.4 | 0.651 / 0.934 | 0.814 / 0.946 | 0.953 / 0.720 | 0.960 / 0.844 | 0.974 / 0.833 | 0.977 / 0.957 | 0.989 / 0.740 |
| sentencex 1.0.31 | 0.681 / 0.919 | 0.855 / 0.934 | 0.974 / 0.711 | 0.978 / 0.841 | 0.986 / 0.865 | 0.989 / 0.968 | 0.986 / 0.796 |
| blingfire 0.1.8 | 0.794 / 0.656 | 0.819 / 0.549 | 0.982 / 0.725 | 0.982 / 0.725 | 0.987 / 0.880 | 0.986 / 0.888 | 0.991 / 0.908 |
| spaCy 3.8.16 sentencizer | 0.694 / 0.655 | 0.717 / 0.548 | 0.953 / 0.760 | 0.956 / 0.762 | 0.944 / 0.862 | 0.963 / 0.879 | 0.905 / 0.832 |

### F1 at boundaries preceded by `.!?`

| system | legal | legal_pm | ewt_flat | ewt_para | gum_flat | gum_para | brown_flat |
|---|---|---|---|---|---|---|---|
| *share of gold boundaries preceded by .!?* | 0.704 | 0.590 | 0.794 | 0.794 | 0.891 | 0.891 | 0.915 |
| main Punkt-only | 0.939 | 0.953 | 0.978 | 0.981 | 0.980 | 0.984 | 0.989 |
| main layout default (and + line_breaks) | **0.942** | **0.959** | 0.978 | **0.982** | 0.980 | 0.984 | 0.989 |
| main adaptive t=0.7 | 0.929 | 0.939 | 0.970 | 0.971 | 0.976 | 0.977 | 0.985 |
| main adaptive t=0.5 | 0.891 | 0.893 | 0.934 | 0.934 | 0.945 | 0.946 | 0.968 |
| nupunkt 0.7.0 | 0.938 | 0.952 | **0.979** | 0.980 | 0.980 | 0.983 | 0.988 |
| nupunkt 0.6.0 | 0.858 | 0.856 | 0.969 | 0.969 | 0.972 | 0.971 | 0.970 |
| nupunkt 0.5.1 | 0.886 | 0.895 | 0.957 | 0.957 | 0.952 | 0.952 | 0.978 |
| nltk 3.10.3 punkt_tab | 0.829 | 0.844 | 0.978 | 0.978 | **0.982** | 0.982 | 0.989 |
| pysbd 0.3.4 | 0.841 | 0.857 | 0.930 | 0.953 | 0.954 | 0.977 | 0.891 |
| sentencex 1.0.31 | 0.869 | 0.885 | 0.934 | 0.961 | 0.975 | **0.986** | 0.929 |
| blingfire 0.1.8 | 0.856 | 0.868 | 0.946 | 0.946 | **0.982** | **0.986** | **0.992** |
| spaCy 3.8.16 sentencizer | 0.839 | 0.851 | 0.965 | 0.968 | 0.976 | **0.986** | 0.947 |

At `.!?` boundaries, nupunkt leads the other libraries on the legal set by about 0.07 F1 and
is within 0.006 of the best system on general English. Punkt-only F1 falls short of this
`.!?` F1 by roughly the share of gold boundaries without terminal punctuation. Main differs
from 0.7.0 by at most 0.002 F1 in Punkt-only mode.

### Per-genre F1 (top and bottom five genres for main Punkt-only)

`gum_para`:

| genre | gold | main Punkt | main layout | 0.7.0 | 0.6.0 | nltk | pysbd | sentencex | blingfire | spaCy |
|---|---|---|---|---|---|---|---|---|---|---|
| vlog | 1169 | 0.987 | 0.991 | 0.987 | 0.987 | 0.988 | 0.993 | 0.993 | 0.988 | 0.987 |
| speech | 737 | 0.979 | 0.996 | 0.979 | 0.965 | 0.979 | 0.989 | 0.992 | 0.978 | 0.979 |
| podcast | 935 | 0.977 | 0.978 | 0.977 | 0.972 | 0.978 | 0.980 | 0.993 | 0.971 | 0.978 |
| court | 863 | 0.974 | 0.985 | 0.973 | 0.959 | 0.973 | 0.963 | 0.991 | 0.972 | 0.970 |
| conversation | 2001 | 0.952 | 0.985 | 0.952 | 0.952 | 0.955 | 0.990 | 0.987 | 0.950 | 0.953 |
| whow | 1069 | 0.901 | 0.971 | 0.901 | 0.900 | 0.905 | 0.976 | 0.987 | 0.920 | 0.903 |
| textbook | 781 | 0.897 | 0.965 | 0.897 | 0.883 | 0.892 | 0.965 | 0.962 | 0.893 | 0.873 |
| letter | 766 | 0.892 | 0.974 | 0.892 | 0.862 | 0.882 | 0.971 | 0.976 | 0.891 | 0.873 |
| academic | 615 | 0.891 | 0.954 | 0.891 | 0.879 | 0.880 | 0.947 | 0.957 | 0.901 | 0.871 |
| bio | 751 | 0.719 | 0.855 | 0.718 | 0.703 | 0.715 | 0.849 | 0.976 | 0.871 | 0.713 |

`brown_flat` (no blank lines, so layout equals Punkt-only):

| genre | gold | main Punkt | 0.7.0 | 0.6.0 | nltk | pysbd | sentencex | blingfire | spaCy |
|---|---|---|---|---|---|---|---|---|---|
| adventure | 4608 | 0.974 | 0.973 | 0.971 | 0.980 | 0.801 | 0.876 | 0.979 | 0.799 |
| fiction | 4220 | 0.970 | 0.969 | 0.963 | 0.976 | 0.869 | 0.928 | 0.972 | 0.836 |
| romance | 4402 | 0.970 | 0.968 | 0.963 | 0.975 | 0.776 | 0.877 | 0.975 | 0.779 |
| mystery | 3862 | 0.966 | 0.965 | 0.956 | 0.972 | 0.767 | 0.850 | 0.972 | 0.814 |
| humor | 1044 | 0.963 | 0.960 | 0.944 | 0.965 | 0.760 | 0.902 | 0.963 | 0.803 |
| learned | 7654 | 0.923 | 0.923 | 0.918 | 0.919 | 0.889 | 0.867 | 0.934 | 0.905 |
| news | 4579 | 0.922 | 0.921 | 0.860 | 0.922 | 0.840 | 0.862 | 0.918 | 0.858 |
| hobbies | 4157 | 0.921 | 0.921 | 0.911 | 0.915 | 0.835 | 0.868 | 0.929 | 0.900 |
| religion | 1699 | 0.910 | 0.910 | 0.908 | 0.921 | 0.828 | 0.777 | 0.913 | 0.896 |
| government | 3002 | 0.892 | 0.892 | 0.873 | 0.884 | 0.831 | 0.830 | 0.892 | 0.854 |

In both corpora the first five rows are the top five genres and the last five the bottom
five. The worst genre is GUM `bio`: 0.719 for Punkt-only and 0.855 with layout, against 0.976
for sentencex. It consists of Wikipedia biographies whose gold keeps reference markers inside
the sentence (`…in 1821. [22] His…`), while nupunkt splits before the marker. In 0.7.0 that
pattern caused 126 of the 136 false positives and 122 of the 254 misses in the genre. On
Brown, NLTK beats nupunkt in 6 of the 10 genres shown. pysbd's low Brown recall looks tied to
document length: it returned 61 sentences for one 187-sentence document but segmented a
600-character window of that document correctly. We did not investigate further.

## Where recall is lost (legal set)

Each gold boundary goes into the first bucket that matches, checked in this order:

1. preceded by `.!?` (plus closers)
2. next segment starts with a list enumerator (`(a) `, `1. `, `iv) `, `- `, `• `)
3. preceded by `;`, `:` or `,`
4. no punctuation, followed by a line break
5. no punctuation, inline

"Missed" and "FP" are for main Punkt-only. The remaining columns give recall.

| context before boundary | gold | missed (share) | FP | main Punkt | main layout | layout + lines | 0.7.0 | 0.6.0 | nltk | pysbd | blingfire | spaCy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| terminal `.!?` | 76,241 | 1,944 (6%) | 7,693 | 0.975 | 0.984 | 0.984 | 0.972 | 0.958 | 0.985 | 0.974 | 0.923 | 0.918 |
| no punct., line break (headings, captions) | 21,169 | 20,488 (62%) | 253 | 0.032 | 0.860 | 0.929 | 0.027 | 0.000 | 0.000 | 1.000 | 0.022 | 0.032 |
| list item follows | 3,954 | 3,909 (12%) | 85 | 0.011 | 0.645 | 0.916 | 0.011 | 0.000 | 0.000 | 0.929 | 0.013 | 0.013 |
| `;` | 3,347 | 3,347 (10%) | 0 | 0.000 | 0.134 | 0.134 | 0.000 | 0.000 | 0.000 | 0.160 | 0.001 | 0.003 |
| no punct., inline | 1,984 | 1,900 (6%) | 179 | 0.042 | 0.045 | 0.047 | 0.032 | 0.002 | 0.005 | 0.084 | 0.042 | 0.044 |
| `:` | 1,440 | 1,440 (4%) | 0 | 0.000 | 0.792 | 0.792 | 0.000 | 0.000 | 0.000 | 0.832 | 0.012 | 0.001 |
| `,` | 101 | 101 (0%) | 0 | 0.000 | 0.436 | 0.436 | 0.000 | 0.000 | 0.000 | 0.515 | 0.000 | 0.030 |

**The Punkt tokenizer.** The "precision over recall" point is accurate. Punkt decides only
at `.`, `!` and `?`. On this set, 30% of gold boundaries have no terminal punctuation, and
they account for 94% of Punkt's misses. On its own, `sent_tokenize` / `span_tokenize` is a
poor fit for chunking where headings and list items must be separate units.

**The layout default.** Blank-line boundaries recover headings (recall 0.032 → 0.860) and
most colon-before-block boundaries (0 → 0.792). With `line_breaks=True`, list items also go
from 0.011 to 0.916. Effect per set, precision / recall, Punkt-only → layout default:

| set | precision | recall | F1 |
|---|---|---|---|
| `legal` | 0.901 → 0.757 | 0.694 → 0.901 | 0.784 → 0.822 |
| `legal_pm` | 0.932 → 0.949 | 0.582 → 0.915 | 0.716 → 0.932 |
| `ewt_para` | 0.985 → 0.986 | 0.777 → 0.861 | 0.869 → 0.919 |
| `gum_para` | 0.977 → 0.978 | 0.883 → 0.965 | 0.928 → 0.972 |
| `*_flat`, `brown_flat` | unchanged (no blank lines) | unchanged | unchanged |

Precision holds or improves wherever the gold treats blank-line blocks as units. On `legal`
it drops because that gold does not end a sentence at most headings (see
[Limitations](#limitations)). The layout default recovers slightly less recall than the naive
cut at every blank line tested in an earlier revision of this page (legal 0.901 vs 0.911, GUM
0.965 vs 0.970). The gap is the page-break continuation rule: a block that starts lowercase
continues the previous sentence. Semicolon boundaries (10% of misses) remain unrecovered by
any nupunkt mode.

**Precision.** A random sample of 40 of 0.7.0's 8,103 legal false positives (seed 7) was
judged by hand. About 15 were header or caption lines ending in a period that the gold marks
inconsistently (`Supreme Court of Alabama.`). About 8 were a citation after a complete
sentence. About 9 were real abbreviation errors (`2d.`, `1st.`, `Wag. Stats.`). The rest were
footnote fragments. This is one person's judgment on a small sample.

### The adaptive threshold

`AdaptiveTokenizer` removes a Punkt boundary when its confidence is below `1 − t` and adds
one when confidence is above `t`. A higher `t` therefore gives more breaks and a lower `t`
fewer, which the corrected docstrings now state. On every set, lowering `t` from 0.7 to 0.5
lowers recall and raises precision. On a 1-in-5 sample of the legal set, t=0.9 gave F1 0.788
against 0.787 for the plain tokenizer, and t=0.3 gave 0.693. So the threshold is a precision
knob. Adaptive mode never exceeded the plain tokenizer's recall, and it cannot recover
layout boundaries.

## Sentence length and the SaT comparison

Mean words per sentence (whitespace tokens divided by sentence count; the first and last
sentences of legal excerpts are partial):

| system | legal | legal_pm | ewt_flat | ewt_para | gum_flat | gum_para | brown_flat |
|---|---|---|---|---|---|---|---|
| **gold** | 13.9 | 11.9 | 13.3 | 13.3 | 15.1 | 15.1 | 17.7 |
| main Punkt-only (= 0.7.0) | 16.8 | 16.8 | 16.5 | 16.5 | 16.8 | 16.7 | 19.6 |
| main layout default | 12.2 | 12.2 | 16.5 | 15.0 | 16.8 | 15.3 | 19.6 |
| main layout + line_breaks | 11.9 | 11.9 | 16.5 | 15.0 | 16.8 | 15.3 | 19.6 |
| nupunkt 0.6.0 | 15.4 | 15.0 | 15.9 | 15.9 | 16.6 | 16.6 | 18.6 |
| nltk 3.10.3 punkt_tab | 14.2 | 14.2 | 16.1 | 16.1 | 16.5 | 16.5 | 19.2 |
| pysbd 0.3.4 | 10.6 | 10.6 | 17.2 | 14.9 | 17.6 | 15.5 | 23.6 |
| sentencex 1.0.31 | 11.1 | 11.1 | 17.7 | 15.3 | 17.2 | 15.5 | 21.9 |
| blingfire 0.1.8 | 16.0 | 16.0 | 17.5 | 17.5 | 16.9 | 16.8 | 19.4 |
| spaCy 3.8.16 sentencizer | 14.5 | 14.5 | 16.3 | 16.3 | 16.6 | 16.6 | 19.3 |

On English, Punkt-only sentences are 11–24% longer than gold, as high precision with lower
recall predicts. The layout default moves the `para` sets closer to gold (EWT 15.0 vs 13.3,
GUM 15.3 vs 15.1).

We cannot reproduce the paper's comparison, for three reasons:

- Per its §4.7–4.8, SaT was used only for the 112 languages not treated as
  Nupunkt-compatible (2.1% of books), so the two averages come from different languages.
- nupunkt ran with per-language models trained on the corpus, not the bundled English model,
  and through `sent_tokenize`, which applies no layout rule.
- Words were counted with polyglot, whose tokenization differs by script.

Characters per sentence (113.2 vs 119.5) differ much less than words per sentence. That fits
a language or word-counting effect better than a segmenter effect, but these data cannot
settle it.

## The 91.1% precision figure

The README's "91.1% precision" comes from the nupunkt paper
([arXiv 2504.04131](https://arxiv.org/abs/2504.04131)), measured on "five diverse legal
datasets" with 197,000 boundaries, a different evaluation from this one. On this repository's
legal set, Punkt precision is 0.902 (0.7.0) and 0.901 (main). It was 0.776 for 0.6.0 and
0.874 for 0.5.1. External benchmarks published before 0.7.0, yasbd's among them, measured an
older release. yasbd's 77.9% comes from short hand-built cases with a word-level scorer; see
[golden-rules.md](golden-rules.md) for those cases.

## Determinism

The legal set was tokenized in forward and in reverse document order, each in a fresh process:

| system | docs | docs with different boundaries |
|---|---|---|
| main `span_tokenize` | 38,527 | 0 |
| main `iter_spans` layout default | 38,527 | 0 |
| nupunkt 0.7.0 | 38,527 | 0 |
| nupunkt 0.6.0 | 38,527 | 4,958 |
| nupunkt 0.5.1 | 38,527 | 0 |

The 0.6.0 differences come from processing order, not hash randomisation: two forward runs
with different `PYTHONHASHSEED` values agree exactly. The 0.7.0 changelog gives the cause:
token objects were cached across calls and mutated during annotation.

## Limitations

- **The legal set favours nupunkt.** The bundled model was trained on legal text with the
  same conventions. The 0.7.0 and main heuristics, including the layout rules, were
  checked against these legal, EWT, GUM and Brown sets during development, so none of them
  is held out for nupunkt. The comparators saw none of them.
- **The `legal` sentence-marker set penalises the layout default.** In that gold, a heading
  followed by a blank line ends a paragraph but usually not a sentence, so every heading the
  layout rule separates counts as a false positive. That is why precision falls to 0.757
  there while it rises on `legal_pm`. Neither convention is "right". Pass
  `paragraph_breaks=False` if your gold standard matches `legal`.
- The legal gold also marks some semicolons as boundaries and annotates caption lines
  inconsistently.
- **UD and Brown are out of domain for a legal model.** The `flat` variants have no layout
  cues, the `para` variants only blank-line paragraph breaks, and Brown is detokenized
  approximately. None of them exercises `line_breaks` or the page-break rule.
- Boundaries are compared by exact offset, so a split one character away counts as both a
  false positive and a miss.
- One run on one machine. The nupunkt configurations reproduced exactly in the order test;
  the comparators were not rerun.
- English only.

## Reproduction

```bash
python scripts/benchmarks/build_gold_sets.py --download      # UD r2.18 + Brown, via curl
uv run python scripts/benchmarks/accuracy.py --with-layout --with-comparators --json-out accuracy.json
```

The scripts need only the standard library. The "main" rows use whatever nupunkt the
interpreter imports; `uv run` in the repository imports the working tree. Rename those rows
with `--current-label`. `--with-layout` adds the three `iter_spans` configurations.
`--with-comparators` builds throwaway `uv venv -p 3.13` environments (nupunkt 0.7.0, 0.6.0 and
0.5.1 from PyPI, and nltk + pysbd + sentencex + blingfire + spacy) and runs each system in a
subprocess, skipping with a message any environment that fails to build. Gold files and
venvs default to `~/.cache/nupunkt-benchmarks` (`--data-dir`, `--venv-dir`). The script
prints all of these tables, with longer row labels and a few extra columns.
