# Layout-aware segmentation

Punkt decides sentence boundaries at sentence-ending punctuation. Documents
also mark boundaries with layout: a blank line between paragraphs, a heading on
a line of its own, one list item per line. None of these carry a period, so a
purely punctuation-driven splitter merges a heading into the sentence after it.
On nupunkt's legal gold set, 94% of the boundaries Punkt misses have no
terminal punctuation at all: headings and captions (62%), list items (12%),
semicolons (10%) and colons (4%).

nupunkt therefore applies two deterministic layout rules *around* Punkt, not
inside it. They live in `nupunkt.layout` and are used by the segmentation
interface (`sentences`, `sentence_spans`, `sentence_segments`, `segmenter`,
`segment`, and the `iter_segments` family on tokenizer objects). The legacy
functions (`sent_tokenize`, `sent_spans`, `para_tokenize`, ...) never apply
them, so their output is unchanged.

| Option | Default | Rule |
|---|---|---|
| `paragraph_breaks` | `True` | A blank line is a hard sentence boundary, unless the sentence plainly continues across it (see page breaks below). Paragraphs are the blocks between blank lines. |
| `line_breaks` | `False` | A single newline is a boundary after a heading-like line (short, no terminal punctuation) or before a list-marker line. Opt-in. |

```python
from nupunkt import sentences, segment

text = "INTRODUCTION\n\nThe court held for the plaintiff. It awarded fees."

sentences(text)
# ['INTRODUCTION', 'The court held for the plaintiff.', 'It awarded fees.']

sentences(text, paragraph_breaks=False)   # Punkt only, as sent_tokenize()
# ['INTRODUCTION\n\nThe court held for the plaintiff.', 'It awarded fees.']

doc = segment(text)
[p.text for p in doc.paragraphs]
# ['INTRODUCTION', 'The court held for the plaintiff. It awarded fees.']
```

## Measured effect

Boundary F1 of the default model, Punkt only versus the layout default. On the
three sets that keep paragraph structure, precision is unchanged and recall is
what the layout rule recovers; the fourth row shows the one convention under
which the rule costs precision.

| Gold set | Punkt only P / R / F1 | With `paragraph_breaks` P / R / F1 |
|---|---|---|
| Legal, paragraph markers counted | 0.93 / 0.58 / 0.716 | 0.95 / 0.92 / 0.932 |
| UD English EWT, paragraphs kept | 0.98 / 0.78 / 0.869 | 0.99 / 0.86 / 0.919 |
| UD English GUM, paragraphs kept | 0.98 / 0.88 / 0.928 | 0.98 / 0.97 / 0.972 |
| Legal, sentence markers only | 0.90 / 0.69 / 0.784 | 0.76 / 0.90 / 0.822 |

The last row is the same legal texts scored against a convention under which a
heading followed by a blank line ends a paragraph but not a sentence. It shows
what the rule costs when your gold standard does not want headings as units.
Turn the rule off with `paragraph_breaks=False` in that case.

`line_breaks=True` adds 0.004 to 0.007 F1 on the legal sets and nothing on UD,
whose texts have no single-newline headings. It is off by default because on
hard-wrapped plain text (email, old plain-text files) a line can end
mid-sentence without punctuation.

Reproduce with `python scripts/benchmarks/accuracy.py --with-layout` after
building the gold sets with `python scripts/benchmarks/build_gold_sets.py`.

## Page breaks in scanned documents

A page break in OCR'd or PDF-extracted text usually appears as a blank line,
often with a page number between the pages, in the middle of a sentence:

```
... the court held that the

12

defendant had waived the claim.
```

Four questions arise: what should happen, what does happen, what is
documented, and what other libraries do.

**What should happen.** The sentence is one sentence. The page number is not
part of it. A splitter that returns one span per sentence cannot express "one
sentence with a hole in it", so the page number has to be removed, or blanked,
before segmentation, and the two halves joined.

**What nupunkt does.**

| Input | `sentences()` result |
|---|---|
| Blank line, next page starts lowercase | one sentence across the gap: `continues_across` sees no terminal punctuation before the gap and a lowercase letter after it |
| Word split with a hyphen at the page end (`estab-` / `lishment`) | one sentence |
| Blank line, next page starts uppercase | two units, since "heading, then sentence" and "sentence, then sentence" look identical |
| Page number on its own line between the halves | three units: the first half, the number, the second half |
| Same, after `blank_page_furniture(text)` | one sentence; spans still index the original text |

`blank_page_furniture` replaces page-number lines (`12`, `- 12 -`, `[12]`,
`Page 3`, `Page 3 of 10`, `p. 12`) and form feeds with the same number of
spaces, so every character offset is preserved. Running headers and footers
that contain words are not recognised; blank them with a document-specific
pattern in the same offset-preserving way. With `paragraph_breaks=False` the
page break is invisible to Punkt and the page number is absorbed into the
sentence text, which is the 0.7.0 behaviour and NLTK's.

```python
from nupunkt import blank_page_furniture, sentence_segments

raw = "The court held that the\n\n12\n\ndefendant had waived the claim. Next."
clean = blank_page_furniture(raw)          # same length as raw
for s in sentence_segments(clean):
    print(raw[s.start:s.end])              # offsets are valid for raw
```

**What is documented.** This page, the `nupunkt.layout` docstrings and
`tests/test_layout.py` (class `TestPageBreaks`) define the behaviour above.
Before 0.8.0 nothing was documented, and the behaviour was "whatever Punkt
does", which merged headings into sentences and page numbers into sentences.

**What other libraries do.** As far as we can tell from their documentation
and from running them on the gold sets, no sentence splitter in this class
models page breaks. NLTK's Punkt uses a blank line only as orthographic
context and never as a boundary. pysbd and sentencex treat newlines as
boundaries (they beat Punkt-only nupunkt on every gold set that keeps
paragraphs), so they cut a page-break sentence in two and emit the page number
as a sentence. blingfire and spaCy's rule sentencizer are punctuation-driven
like Punkt. Layout handling is normally left to the document parser
(PDF and OCR toolkits offer their own page-join heuristics). nupunkt's
position is the same, with two additions: the lowercase and hyphen
continuation rule, which repairs the common case without any cleanup, and
`blank_page_furniture`, which handles the page-number case while keeping
offsets stable.

## Limits

- A heading followed by a paragraph that starts lowercase is merged with it.
- Two paragraphs where the first ends without punctuation and the second starts
  lowercase are merged.
- Running headers or footers with words are not detected as furniture.
- `line_breaks=True` cuts after any short unpunctuated line whose successor
  starts with a capital, which is wrong for hard-wrapped prose; keep it off for
  such text.
