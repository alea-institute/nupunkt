# Publicly reported nupunkt cases: 0.6.0, 0.7.0 and unreleased main

This page collects inputs that people have reported in public as nupunkt failures or
complaints, plus a few from our own 0.7.0 development notes. Each one is run through nupunkt 0.6.0,
0.7.0 (default and adaptive mode) and the unreleased main tree. NLTK Punkt and pysbd are included
for reference. The page shows the failures too. Of the 41 cases, 0.7.0 fixes 8 relative to 0.6.0,
14 still fail, and none regressed. The unreleased main tree fixes 2 more (N3, D8), leaving 12, and
regresses none. A separate [layout](#layout-and-page-breaks) table covers blank lines and page
breaks. The [still open](#still-open) section lists what is still wrong and says why.

## Methodology

- **Versions.** `nupunkt 0.7.0` from the release wheel `dist/nupunkt-0.7.0-py3-none-any.whl`
  (same source as commit `7befa44`), `nupunkt 0.6.0` from PyPI, `nltk 3.10.3` with `punkt_tab`, and
  `pysbd 0.3.4`. Each library was installed in its own throwaway `uv venv -p 3.13` (CPython
  3.13.15, Linux 6.18). **nupunkt main (unreleased)** is the repository working tree on
  2026-09-27 (commit `7befa44` plus uncommitted changes), imported by the project `.venv`. Its
  `__version__` still reads `0.7.0`. Those columns describe code that has not been released and may
  still change.
- **Fresh process per case.** Every (case, library, mode) runs in a new Python process. This
  matters because 0.6.0 is history-dependent: its output for an input can change depending on
  what the process tokenized earlier (see [Determinism](#determinism-output-depends-on-earlier-calls)).
- **Calls.** nupunkt: `sent_tokenize(text)` and `sent_tokenize(text, adaptive=True)`. NLTK:
  `nltk.tokenize.sent_tokenize(text)`. pysbd: `Segmenter(language="en", clean=False)`.
- **Scoring.** Whitespace inside each sentence is collapsed before the output is compared with the
  expected output. A case passes only when every boundary matches. When the expected answer
  depends on convention (D1, D2), each accepted segmentation is listed in the notes. The verdict
  compares 0.6.0 with 0.7.0 default. "Output changed" means both versions fail but produce
  different output.
- **Expected outputs** follow the reporter where the reporter gave one. Otherwise they follow
  ordinary English sentence conventions. Cases marked *constructed* are ours, built to probe a
  reported pattern.
- **Regenerate** (tables pasted unedited from this command; prose is hand-written):

```bash
uv venv -p 3.13 /tmp/v07 && uv pip install -p /tmp/v07/bin/python dist/nupunkt-0.7.0-py3-none-any.whl
uv venv -p 3.13 /tmp/v06 && uv pip install -p /tmp/v06/bin/python nupunkt==0.6.0
uv venv -p 3.13 /tmp/vc && uv pip install -p /tmp/vc/bin/python nltk pysbd
/tmp/vc/bin/python -c "import nltk; nltk.download('punkt_tab', download_dir='/tmp/nltk_data')"
NLTK_DATA=/tmp/nltk_data /tmp/v07/bin/python scripts/benchmarks/known_cases.py \
    --compare-06 /tmp/v06/bin/python --compare-main .venv/bin/python \
    --compare-others /tmp/vc/bin/python
```

With no flags, the script tests only the `nupunkt` that the running interpreter imports.

## Sources and what they reported

- **abraham-jacob/scout#30**
  ([comment](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5287859994)).
  speedyk-005 posted nupunkt output that includes `[5]: 'Dr. Smith Jr. met Sr.'` and
  `[6]: 'Consultant Davis at Acme Inc. ... etc. Johnson outlined the roadmap.'`. Cases A4–A8 come
  from the same thread but were reported against yasbd. They are included as related
  abbreviation checks.
- **yasbd-lib [benchmarks README](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/README.md)**
  (last updated 2026-09-25). It calls nupunkt "over-aggressive on double exclamation marks
  (`broh !`, `!`)" and says it "Shreds `c.-à-d.` and `m.-à-j.`". For Japanese it gives "Total
  Failure. No support for CJK punctuation." It splits the witness quotation into 3 pieces, which the
  README scores below yasbd's single unit. Its
  [`bench_utils.py`](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/bench_utils.py)
  calls `nupunkt.sent_tokenize(text)`, which uses the English model, for every language. The
  92-case golden set is covered on a sibling page. N4 and N5 are taken from it here.
- **Numbered clauses (N1, N2).** These came from our internal complaint list, which attributes them
  to the yasbd README. We could not find either string in the current README, in four older
  revisions of it, or in `EN_GOLDEN_DATA.py`. N4 is the closest verbatim golden-set case.
- **NLTK issues** [#2082](https://github.com/nltk/nltk/issues/2082),
  [#3370](https://github.com/nltk/nltk/issues/3370), [#2892](https://github.com/nltk/nltk/issues/2892),
  [#2154](https://github.com/nltk/nltk/issues/2154) and [#60](https://github.com/nltk/nltk/issues/60)
  were filed against NLTK's Punkt, not nupunkt. They are included because nupunkt is a Punkt
  derivative. #2892 gives no concrete input about the year, so N6 is constructed. A13 is taken from
  a comment on that issue. #2154 gives `no. 5` and the counter-example A15.
- **KnowSeams [README](https://github.com/KnowSeams/KnowSeams/blob/b0f2d29597aec4025d1b5a35829fe6c6878c24e8/README.md)**
  (@SteveJSteiner) says: "**nupunkt**: Splits attribution (`"What!"` + `Captain Forrester?"` +
  `cried I.`)". Its benchmark table names nupunkt 0.5.1. D3–D8 are constructed dialog cases.
- **houfu/redlines** [#48](https://github.com/houfu/redlines/issues/48): "Sentence and paragraph
  boundaries like /n remain some of our thorniest issues". [#113](https://github.com/houfu/redlines/pull/113):
  sentence mode "destroyed the input's real paragraph boundaries". That happened because redlines
  joined every sentence with `¶`, not because nupunkt dropped text. The code also carried
  `# sent_tokenize can return either strings or tuples (text, score)`. See
  [API](#api-return-types-and-paragraphs).
- **GriffynHancock/homogenous-cluster**
  [`sentences.py`](https://github.com/GriffynHancock/homogenous-cluster/blob/d1e872ac6530b36aa7dcf009c47e6ff1ba9250a6/missing-link/missing_link/sentences.py#L168-L183)
  has to trim nupunkt's spans: "`nupunkt.sent_spans` returns contiguous spans that include the
  whitespace between sentences". See [Spans](#spans-legacy-contiguous-vs-tight).
- **institutional/institutional-books-enriched-text-pipeline**
  [`nupunkt_segmenter.py`](https://github.com/institutional/institutional-books-enriched-text-pipeline/blob/cd21359741cd519900a555b174f6ecb6901aa480/library/segment/nupunkt_segmenter.py)
  catches `if "math domain error" in str(e):` and falls back to the base model. It notes: "This is
  fragile and involves accessing a model's _params directly." See
  [Training](#training-on-a-degenerate-corpus-math-domain-error).

## Results

<!-- generated by scripts/benchmarks/known_cases.py -->
Versions: `nupunkt 0.6.0`; `nupunkt 0.7.0 default`; `nupunkt 0.7.0 adaptive`; `nupunkt main (unreleased) default`; `nltk 3.10.3 punkt_tab`; `pysbd 0.3.4`.

### Abbreviations before a capitalized word

| # | Input | Expected | nupunkt 0.6.0 | nupunkt 0.7.0 default | nupunkt 0.7.0 adaptive | nupunkt main (unreleased) default | nltk 3.10.3 punkt_tab | pysbd 0.3.4 | Verdict | 0.7.0 to main |
|---|---|---|---|---|---|---|---|---|---|---|
| [A1](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5287859994) | `Acme Inc. USA is expanding its engineering team this quarter. Beta Corp. North America leads this hiring initiative. Martin Luther King Jr. Day is a paid holiday at this company. John Doe Sr. VP of Engineering will be your hiring manager. Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday alongside Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements, considering e.g. U.S. policies, U.K. guidelines, etc. Johnson outlined the roadmap.` | `Acme Inc. USA is expanding its engineering team this quarter.`<br>`Beta Corp. North America leads this hiring initiative.`<br>`Martin Luther King Jr. Day is a paid holiday at this company.`<br>`John Doe Sr. VP of Engineering will be your hiring manager.`<br>`Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday alongside Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements, considering e.g. U.S. policies, U.K. guidelines, etc.`<br>`Johnson outlined the roadmap.` | **FAIL**<br>`Acme Inc. USA is expanding its engineering team this quarter.`<br>`Beta Corp. North America leads this hiring initiative.`<br>`Martin Luther King Jr. Day is a paid holiday at this company.`<br>`John Doe Sr. VP of Engineering will be your hiring manager.`<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday alongside Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements, considering e.g. U.S. policies, U.K. guidelines, etc. Johnson outlined the roadmap.` | **FAIL**<br>`Acme Inc. USA is expanding its engineering team this quarter.`<br>`Beta Corp. North America leads this hiring initiative.`<br>`Martin Luther King Jr. Day is a paid holiday at this company.`<br>`John Doe Sr. VP of Engineering will be your hiring manager.`<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday alongside Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements, considering e.g. U.S. policies, U.K. guidelines, etc.`<br>`Johnson outlined the roadmap.` | **FAIL**<br>`Acme Inc. USA is expanding its engineering team this quarter.`<br>`Beta Corp. North America leads this hiring initiative.`<br>`Martin Luther King Jr. Day is a paid holiday at this company.`<br>`John Doe Sr. VP of Engineering will be your hiring manager.`<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday alongside Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements, considering e.g. U.S. policies, U.K. guidelines, etc. Johnson outlined the roadmap.` | **FAIL**<br>`Acme Inc. USA is expanding its engineering team this quarter.`<br>`Beta Corp. North America leads this hiring initiative.`<br>`Martin Luther King Jr. Day is a paid holiday at this company.`<br>`John Doe Sr. VP of Engineering will be your hiring manager.`<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday alongside Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements, considering e.g. U.S. policies, U.K. guidelines, etc.`<br>`Johnson outlined the roadmap.` | `Acme Inc. USA is expanding its engineering team this quarter.`<br>`Beta Corp. North America leads this hiring initiative.`<br>`Martin Luther King Jr. Day is a paid holiday at this company.`<br>`John Doe Sr. VP of Engineering will be your hiring manager.`<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday alongside Representatives Corp. and Global Ltd. to discuss B.S.`<br>`and M.S.`<br>`degree standards vs. Ph.D. requirements, considering e.g.`<br>`U.S. policies, U.K. guidelines, etc.`<br>`Johnson outlined the roadmap.` | `Acme Inc.`<br>`USA is expanding its engineering team this quarter.`<br>`Beta Corp.`<br>`North America leads this hiring initiative.`<br>`Martin Luther King Jr.`<br>`Day is a paid holiday at this company.`<br>`John Doe Sr.`<br>`VP of Engineering will be your hiring manager.`<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday alongside`<br>`Representatives Corp. and Global Ltd. to discuss B.S. and M.S. degree standards vs. Ph.D. requirements,`<br>`considering e.g. U.S. policies, U.K. guidelines, etc.`<br>`Johnson outlined the roadmap.` | **still fails** (output changed) | still fails |
| [A2](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5287859994) | `Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday.` | `Dr. Smith Jr. met Sr. Consultant Davis at Acme Inc. yesterday.` | **FAIL**<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday.` | **FAIL**<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday.` | **FAIL**<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday.` | **FAIL**<br>`Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday.` | `Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday.` | `Dr. Smith Jr. met Sr.`<br>`Consultant Davis at Acme Inc. yesterday.` | **still fails** | still fails |
| [A3](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5287859994) | `John Doe Sr. VP of Engineering will be your hiring manager.` | `John Doe Sr. VP of Engineering will be your hiring manager.` | ok<br>`John Doe Sr. VP of Engineering will be your hiring manager.` | ok<br>`John Doe Sr. VP of Engineering will be your hiring manager.` | ok<br>`John Doe Sr. VP of Engineering will be your hiring manager.` | ok<br>`John Doe Sr. VP of Engineering will be your hiring manager.` | `John Doe Sr. VP of Engineering will be your hiring manager.` | `John Doe Sr.`<br>`VP of Engineering will be your hiring manager.` | unchanged (pass) | pass |
| [A4](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5262960213) | `This role requires U.S. Government security clearance.` | `This role requires U.S. Government security clearance.` | ok<br>`This role requires U.S. Government security clearance.` | ok<br>`This role requires U.S. Government security clearance.` | ok<br>`This role requires U.S. Government security clearance.` | ok<br>`This role requires U.S. Government security clearance.` | `This role requires U.S. Government security clearance.` | `This role requires U.S. Government security clearance.` | unchanged (pass) | pass |
| [A5](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5262960213) | `This job is open to U.S. Persons only per export law.` | `This job is open to U.S. Persons only per export law.` | ok<br>`This job is open to U.S. Persons only per export law.` | ok<br>`This job is open to U.S. Persons only per export law.` | ok<br>`This job is open to U.S. Persons only per export law.` | ok<br>`This job is open to U.S. Persons only per export law.` | `This job is open to U.S.`<br>`Persons only per export law.` | `This job is open to U.S. Persons only per export law.` | unchanged (pass) | pass |
| [A6](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5267951914) | `This policy was adopted in the U.S. Next week we will review it.` | `This policy was adopted in the U.S.`<br>`Next week we will review it.` | ok<br>`This policy was adopted in the U.S.`<br>`Next week we will review it.` | ok<br>`This policy was adopted in the U.S.`<br>`Next week we will review it.` | ok<br>`This policy was adopted in the U.S.`<br>`Next week we will review it.` | ok<br>`This policy was adopted in the U.S.`<br>`Next week we will review it.` | `This policy was adopted in the U.S. Next week we will review it.` | `This policy was adopted in the U.S. Next week we will review it.` | unchanged (pass) | pass |
| [A7](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5267951914) | `I live in the E.U. How about you?` | `I live in the E.U.`<br>`How about you?` | ok<br>`I live in the E.U.`<br>`How about you?` | ok<br>`I live in the E.U.`<br>`How about you?` | ok<br>`I live in the E.U.`<br>`How about you?` | ok<br>`I live in the E.U.`<br>`How about you?` | `I live in the E.U.`<br>`How about you?` | `I live in the E.U.`<br>`How about you?` | unchanged (pass) | pass |
| [A8](https://github.com/abraham-jacob/scout/issues/30#issuecomment-5285710323) | `We are Acme Inc. Our team is fully remote.` | `We are Acme Inc.`<br>`Our team is fully remote.` | ok<br>`We are Acme Inc.`<br>`Our team is fully remote.` | ok<br>`We are Acme Inc.`<br>`Our team is fully remote.` | ok<br>`We are Acme Inc.`<br>`Our team is fully remote.` | ok<br>`We are Acme Inc.`<br>`Our team is fully remote.` | `We are Acme Inc. Our team is fully remote.` | `We are Acme Inc.`<br>`Our team is fully remote.` | unchanged (pass) | pass |
| [A9](https://github.com/nltk/nltk/issues/60) | `John saw Mary in the U.S.A. John saw Mary in the U.S.A. today.` | `John saw Mary in the U.S.A.`<br>`John saw Mary in the U.S.A. today.` | **FAIL**<br>`John saw Mary in the U.S.A.`<br>`John saw Mary in the U.S.A.`<br>`today.` | ok<br>`John saw Mary in the U.S.A.`<br>`John saw Mary in the U.S.A. today.` | ok<br>`John saw Mary in the U.S.A.`<br>`John saw Mary in the U.S.A. today.` | ok<br>`John saw Mary in the U.S.A.`<br>`John saw Mary in the U.S.A. today.` | `John saw Mary in the U.S.A. John saw Mary in the U.S.A. today.` | `John saw Mary in the U.S.A. John saw Mary in the U.S.A. today.` | **fixed** | pass |
| [A10](https://github.com/nltk/nltk/issues/3370) | `He was a Ph.D.: that meant studying a lot.` | `He was a Ph.D.: that meant studying a lot.` | ok<br>`He was a Ph.D.: that meant studying a lot.` | ok<br>`He was a Ph.D.: that meant studying a lot.` | ok<br>`He was a Ph.D.: that meant studying a lot.` | ok<br>`He was a Ph.D.: that meant studying a lot.` | `He was a Ph.D.: that meant studying a lot.` | `He was a Ph.D.: that meant studying a lot.` | unchanged (pass) | pass |
| [A11](https://github.com/nltk/nltk/issues/3370) | `She is an M.D.: an oncologist. Please assist me.` | `She is an M.D.: an oncologist.`<br>`Please assist me.` | ok<br>`She is an M.D.: an oncologist.`<br>`Please assist me.` | ok<br>`She is an M.D.: an oncologist.`<br>`Please assist me.` | ok<br>`She is an M.D.: an oncologist.`<br>`Please assist me.` | ok<br>`She is an M.D.: an oncologist.`<br>`Please assist me.` | `She is an M.D.`<br>`: an oncologist.`<br>`Please assist me.` | `She is an M.D.: an oncologist.`<br>`Please assist me.` | unchanged (pass) | pass |
| [A12](https://github.com/nltk/nltk/issues/3370) | `It was the Ph.D.'s responsibility to perform research.` | `It was the Ph.D.'s responsibility to perform research.` | ok<br>`It was the Ph.D.'s responsibility to perform research.` | ok<br>`It was the Ph.D.'s responsibility to perform research.` | ok<br>`It was the Ph.D.'s responsibility to perform research.` | ok<br>`It was the Ph.D.'s responsibility to perform research.` | `It was the Ph.D.'s responsibility to perform research.` | `It was the Ph.D.'s responsibility to perform research.` | unchanged (pass) | pass |
| [A13](https://github.com/nltk/nltk/issues/2892) | `We use a sentence tokenizer, i.e. a function to transform a string with one or more sentences into a list of individual sentences, etc. and it often succeeds.` | `We use a sentence tokenizer, i.e. a function to transform a string with one or more sentences into a list of individual sentences, etc. and it often succeeds.` | ok<br>`We use a sentence tokenizer, i.e. a function to transform a string with one or more sentences into a list of individual sentences, etc. and it often succeeds.` | ok<br>`We use a sentence tokenizer, i.e. a function to transform a string with one or more sentences into a list of individual sentences, etc. and it often succeeds.` | ok<br>`We use a sentence tokenizer, i.e. a function to transform a string with one or more sentences into a list of individual sentences, etc. and it often succeeds.` | ok<br>`We use a sentence tokenizer, i.e. a function to transform a string with one or more sentences into a list of individual sentences, etc. and it often succeeds.` | `We use a sentence tokenizer, i.e.`<br>`a function to transform a string with one or more sentences into a list of individual sentences, etc.`<br>`and it often succeeds.` | `We use a sentence tokenizer, i.e. a function to transform a string with one or more sentences into a list of individual sentences, etc. and it often succeeds.` | unchanged (pass) | pass |
| [A14](https://github.com/nltk/nltk/issues/2154) | `Jersey no. 5 is the best. But I don't like no. 6.` | `Jersey no. 5 is the best.`<br>`But I don't like no. 6.` | ok<br>`Jersey no. 5 is the best.`<br>`But I don't like no. 6.` | ok<br>`Jersey no. 5 is the best.`<br>`But I don't like no. 6.` | ok<br>`Jersey no. 5 is the best.`<br>`But I don't like no. 6.` | ok<br>`Jersey no. 5 is the best.`<br>`But I don't like no. 6.` | `Jersey no.`<br>`5 is the best.`<br>`But I don't like no.`<br>`6.` | `Jersey no. 5 is the best.`<br>`But I don't like no. 6.` | unchanged (pass) | pass |
| [A15](https://github.com/nltk/nltk/issues/2154) | `Do you like the muffin? No. Are you sure you don't? But this is call the neutrino. Not an inferno. Perhaps a neutri no. Maybe an infer no. Conseula likes to say no. And Jim carrey is the yes man.` | `Do you like the muffin?`<br>`No.`<br>`Are you sure you don't?`<br>`But this is call the neutrino.`<br>`Not an inferno.`<br>`Perhaps a neutri no.`<br>`Maybe an infer no.`<br>`Conseula likes to say no.`<br>`And Jim carrey is the yes man.` | **FAIL**<br>`Do you like the muffin?`<br>`No. Are you sure you don't?`<br>`But this is call the neutrino.`<br>`Not an inferno.`<br>`Perhaps a neutri no. Maybe an infer no. Conseula likes to say no. And Jim carrey is the yes man.` | **FAIL**<br>`Do you like the muffin?`<br>`No. Are you sure you don't?`<br>`But this is call the neutrino.`<br>`Not an inferno.`<br>`Perhaps a neutri no. Maybe an infer no. Conseula likes to say no. And Jim carrey is the yes man.` | **FAIL**<br>`Do you like the muffin?`<br>`No. Are you sure you don't?`<br>`But this is call the neutrino.`<br>`Not an inferno.`<br>`Perhaps a neutri no. Maybe an infer no. Conseula likes to say no. And Jim carrey is the yes man.` | **FAIL**<br>`Do you like the muffin?`<br>`No. Are you sure you don't?`<br>`But this is call the neutrino.`<br>`Not an inferno.`<br>`Perhaps a neutri no. Maybe an infer no. Conseula likes to say no. And Jim carrey is the yes man.` | `Do you like the muffin?`<br>`No.`<br>`Are you sure you don't?`<br>`But this is call the neutrino.`<br>`Not an inferno.`<br>`Perhaps a neutri no.`<br>`Maybe an infer no.`<br>`Conseula likes to say no.`<br>`And Jim carrey is the yes man.` | `Do you like the muffin?`<br>`No.`<br>`Are you sure you don't?`<br>`But this is call the neutrino.`<br>`Not an inferno.`<br>`Perhaps a neutri no.`<br>`Maybe an infer no.`<br>`Conseula likes to say no.`<br>`And Jim carrey is the yes man.` | **still fails** | still fails |
| A16 | `The appeal is App. No. 5 in this court.` | `The appeal is App. No. 5 in this court.` | **FAIL**<br>`The appeal is App.`<br>`No. 5 in this court.` | ok<br>`The appeal is App. No. 5 in this court.` | ok<br>`The appeal is App. No. 5 in this court.` | ok<br>`The appeal is App. No. 5 in this court.` | `The appeal is App.`<br>`No.`<br>`5 in this court.` | `The appeal is App.`<br>`No. 5 in this court.` | **fixed** | pass |
| A17 | `She returned to the U.S. Then it rained.` | `She returned to the U.S.`<br>`Then it rained.` | ok<br>`She returned to the U.S.`<br>`Then it rained.` | ok<br>`She returned to the U.S.`<br>`Then it rained.` | ok<br>`She returned to the U.S.`<br>`Then it rained.` | ok<br>`She returned to the U.S.`<br>`Then it rained.` | `She returned to the U.S. Then it rained.` | `She returned to the U.S. Then it rained.` | unchanged (pass) | pass |
| A18 | `The contract was signed by Global Ltd. Their lawyers reviewed it.` | `The contract was signed by Global Ltd.`<br>`Their lawyers reviewed it.` | **FAIL**<br>`The contract was signed by Global Ltd. Their lawyers reviewed it.` | **FAIL**<br>`The contract was signed by Global Ltd. Their lawyers reviewed it.` | **FAIL**<br>`The contract was signed by Global Ltd. Their lawyers reviewed it.` | **FAIL**<br>`The contract was signed by Global Ltd. Their lawyers reviewed it.` | `The contract was signed by Global Ltd. Their lawyers reviewed it.` | `The contract was signed by Global Ltd.`<br>`Their lawyers reviewed it.` | **still fails** | still fails |

- **A1** — abraham-jacob/scout#30 (nupunkt output posted by speedyk-005). Full passage from the thread. The posted nupunkt output split at `Sr.` and missed `etc. Johnson`.
- **A2** — abraham-jacob/scout#30 (nupunkt output posted by speedyk-005). Shortened from A1.
- **A3** — abraham-jacob/scout#30 (nupunkt output posted by speedyk-005).
- **A4** — abraham-jacob/scout#30 (reported against yasbd, same thread).
- **A5** — abraham-jacob/scout#30 (reported against yasbd, same thread).
- **A6** — abraham-jacob/scout#30 (speedyk-005, cases yasbd must still split).
- **A7** — abraham-jacob/scout#30 (speedyk-005, cases yasbd must still split).
- **A8** — abraham-jacob/scout#30 (abraham-jacob sanity check).
- **A9** — nltk/nltk#60.
- **A10** — nltk/nltk#3370.
- **A11** — nltk/nltk#3370.
- **A12** — nltk/nltk#3370.
- **A13** — nltk/nltk#2892. From a comment on the issue.
- **A14** — nltk/nltk#2154.
- **A15** — nltk/nltk#2154. The counter-example from the issue: `no.` that really ends a sentence.
- **A16** — nupunkt 0.7.0 development notes.
- **A17** — nupunkt 0.7.0 development notes.
- **A18** — nupunkt 0.7.0 development notes.

### Numbers, enumerators and years

| # | Input | Expected | nupunkt 0.6.0 | nupunkt 0.7.0 default | nupunkt 0.7.0 adaptive | nupunkt main (unreleased) default | nltk 3.10.3 punkt_tab | pysbd 0.3.4 | Verdict | 0.7.0 to main |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 | `This Agreement contains the following sections. 1. Definitions. 2. Term. 3. Termination.` | `This Agreement contains the following sections.`<br>`1. Definitions.`<br>`2. Term.`<br>`3. Termination.` | **FAIL**<br>`This Agreement contains the following sections.`<br>`1. Definitions.`<br>`2.`<br>`Term. 3.`<br>`Termination.` | **FAIL**<br>`This Agreement contains the following sections.`<br>`1. Definitions.`<br>`2.`<br>`Term. 3.`<br>`Termination.` | **FAIL**<br>`This Agreement contains the following sections.`<br>`1. Definitions. 2.`<br>`Term. 3.`<br>`Termination.` | **FAIL**<br>`This Agreement contains the following sections.`<br>`1. Definitions.`<br>`2.`<br>`Term.`<br>`3.`<br>`Termination.` | `This Agreement contains the following sections.`<br>`1.`<br>`Definitions.`<br>`2.`<br>`Term.`<br>`3.`<br>`Termination.` | `This Agreement contains the following sections.`<br>`1. Definitions.`<br>`2. Term.`<br>`3. Termination.` | **still fails** | still fails (output changed) |
| N2 | `1. Submit form. 2. Pay fee. 3. Wait for approval.` | `1. Submit form.`<br>`2. Pay fee.`<br>`3. Wait for approval.` | **FAIL**<br>`1.`<br>`Submit form.`<br>`2.`<br>`Pay fee.`<br>`3.`<br>`Wait for approval.` | **FAIL**<br>`1. Submit form.`<br>`2.`<br>`Pay fee.`<br>`3.`<br>`Wait for approval.` | **FAIL**<br>`1. Submit form.`<br>`2.`<br>`Pay fee.`<br>`3.`<br>`Wait for approval.` | **FAIL**<br>`1. Submit form.`<br>`2.`<br>`Pay fee.`<br>`3.`<br>`Wait for approval.` | `1.`<br>`Submit form.`<br>`2.`<br>`Pay fee.`<br>`3.`<br>`Wait for approval.` | `1. Submit form.`<br>`2. Pay fee.`<br>`3. Wait for approval.` | **still fails** (output changed) | still fails |
| N3 | `1. Definitions. 2. Term. 3. Termination.` | `1. Definitions.`<br>`2. Term.`<br>`3. Termination.` | **FAIL**<br>`1. Definitions.`<br>`2.`<br>`Term. 3.`<br>`Termination.` | **FAIL**<br>`1. Definitions.`<br>`2. Term. 3. Termination.` | **FAIL**<br>`1. Definitions. 2. Term. 3. Termination.` | ok<br>`1. Definitions.`<br>`2. Term.`<br>`3. Termination.` | `1.`<br>`Definitions.`<br>`2.`<br>`Term.`<br>`3.`<br>`Termination.` | `1. Definitions.`<br>`2. Term.`<br>`3. Termination.` | **still fails** (output changed) | **fixed on main** |
| [N4](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/EN_GOLDEN_DATA.py) | `1. The first item. 2. The second item.` | `1. The first item.`<br>`2. The second item.` | **FAIL**<br>`1.`<br>`The first item.`<br>`2.`<br>`The second item.` | **FAIL**<br>`1. The first item.`<br>`2.`<br>`The second item.` | **FAIL**<br>`1. The first item.`<br>`2.`<br>`The second item.` | **FAIL**<br>`1. The first item.`<br>`2.`<br>`The second item.` | `1.`<br>`The first item.`<br>`2.`<br>`The second item.` | `1. The first item.`<br>`2. The second item.` | **still fails** (output changed) | still fails |
| [N5](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/EN_GOLDEN_DATA.py) | `Did you remove num 2. Put it back.` | `Did you remove num 2.`<br>`Put it back.` | ok<br>`Did you remove num 2.`<br>`Put it back.` | ok<br>`Did you remove num 2.`<br>`Put it back.` | ok<br>`Did you remove num 2.`<br>`Put it back.` | ok<br>`Did you remove num 2.`<br>`Put it back.` | `Did you remove num 2.`<br>`Put it back.` | `Did you remove num 2.`<br>`Put it back.` | unchanged (pass) | pass |
| [N6](https://github.com/nltk/nltk/issues/2892) | `The company was founded in 2013. It has grown every year since.` | `The company was founded in 2013.`<br>`It has grown every year since.` | ok<br>`The company was founded in 2013.`<br>`It has grown every year since.` | ok<br>`The company was founded in 2013.`<br>`It has grown every year since.` | ok<br>`The company was founded in 2013.`<br>`It has grown every year since.` | ok<br>`The company was founded in 2013.`<br>`It has grown every year since.` | `The company was founded in 2013.`<br>`It has grown every year since.` | `The company was founded in 2013.`<br>`It has grown every year since.` | unchanged (pass) | pass |

- **N1** — maintainer complaint list (verbatim source not located, see notes).
- **N2** — maintainer complaint list (verbatim source not located, see notes).
- **N3** — constructed for this page. N1 as a line-start list.
- **N4** — yasbd-lib EN_GOLDEN_DATA.py.
- **N5** — yasbd-lib EN_GOLDEN_DATA.py.
- **N6** — nltk/nltk#2892. The issue describes `2013.` read as an ordinal; the sentence is ours.

### Quotes, dialog and terminator runs

| # | Input | Expected | nupunkt 0.6.0 | nupunkt 0.7.0 default | nupunkt 0.7.0 adaptive | nupunkt main (unreleased) default | nltk 3.10.3 punkt_tab | pysbd 0.3.4 | Verdict | 0.7.0 to main |
|---|---|---|---|---|---|---|---|---|---|---|
| [D1](https://github.com/KnowSeams/KnowSeams/blob/b0f2d29597aec4025d1b5a35829fe6c6878c24e8/README.md) | `"Well, you can see him easily enough," said Mr. Hoad. "He's staying in your village, I believe. He's a nephew of Squire Broderick's." "What! Captain Forrester?" cried I.` | `"Well, you can see him easily enough," said Mr. Hoad.`<br>`"He's staying in your village, I believe.`<br>`He's a nephew of Squire Broderick's."`<br>`"What! Captain Forrester?" cried I.` | **FAIL**<br>`"Well, you can see him easily enough," said Mr. Hoad.`<br>`"He's staying in your village, I believe.`<br>`He's a nephew of Squire Broderick's."`<br>`"What!`<br>`Captain Forrester?"`<br>`cried I.` | **FAIL**<br>`"Well, you can see him easily enough," said Mr. Hoad.`<br>`"He's staying in your village, I believe.`<br>`He's a nephew of Squire Broderick's."`<br>`"What!`<br>`Captain Forrester?" cried I.` | **FAIL**<br>`"Well, you can see him easily enough," said Mr. Hoad.`<br>`"He's staying in your village, I believe.`<br>`He's a nephew of Squire Broderick's."`<br>`"What!`<br>`Captain Forrester?" cried I.` | **FAIL**<br>`"Well, you can see him easily enough," said Mr. Hoad.`<br>`"He's staying in your village, I believe.`<br>`He's a nephew of Squire Broderick's."`<br>`"What!`<br>`Captain Forrester?" cried I.` | `"Well, you can see him easily enough," said Mr. Hoad.`<br>`"He's staying in your village, I believe.`<br>`He's a nephew of Squire Broderick's."`<br>`"What!`<br>`Captain Forrester?"`<br>`cried I.` | `"Well, you can see him easily enough," said Mr. Hoad.`<br>`"He's staying in`<br>`your village, I believe.`<br>`He's a nephew of Squire Broderick's.`<br>`"`<br>`"What! Captain Forrester?" cried I.` | **still fails** (output changed) | still fails |
| [D2](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/README.md) | `The witness testified: "He said — and I quote — 'I will not comply.' Then he turned around and left. I couldn't believe it."` | `The witness testified: "He said — and I quote — 'I will not comply.'`<br>`Then he turned around and left.`<br>`I couldn't believe it."` | ok<br>`The witness testified: "He said — and I quote — 'I will not comply.'`<br>`Then he turned around and left.`<br>`I couldn't believe it."` | ok<br>`The witness testified: "He said — and I quote — 'I will not comply.'`<br>`Then he turned around and left.`<br>`I couldn't believe it."` | ok<br>`The witness testified: "He said — and I quote — 'I will not comply.'`<br>`Then he turned around and left.`<br>`I couldn't believe it."` | ok<br>`The witness testified: "He said — and I quote — 'I will not comply.'`<br>`Then he turned around and left.`<br>`I couldn't believe it."` | `The witness testified: "He said — and I quote — 'I will not comply.'`<br>`Then he turned around and left.`<br>`I couldn't believe it."` | `The witness testified: "He said — and I quote — 'I will not comply.' Then he turned around and left. I couldn't believe it."` | unchanged (pass) | pass |
| D3 | `"Is it?" he asked. "Yes," she said.` | `"Is it?" he asked.`<br>`"Yes," she said.` | **FAIL**<br>`"Is it?"`<br>`he asked.`<br>`"Yes," she said.` | ok<br>`"Is it?" he asked.`<br>`"Yes," she said.` | ok<br>`"Is it?" he asked.`<br>`"Yes," she said.` | ok<br>`"Is it?" he asked.`<br>`"Yes," she said.` | `"Is it?"`<br>`he asked.`<br>`"Yes," she said.` | `"Is it?" he asked.`<br>`"Yes," she said.` | **fixed** | pass |
| D4 | `"Really?!" asked Tom. "Yes."` | `"Really?!" asked Tom.`<br>`"Yes."` | **FAIL**<br>`"Really?!"`<br>`asked Tom.`<br>`"Yes."` | ok<br>`"Really?!" asked Tom.`<br>`"Yes."` | ok<br>`"Really?!" asked Tom.`<br>`"Yes."` | ok<br>`"Really?!" asked Tom.`<br>`"Yes."` | `"Really?!"`<br>`asked Tom.`<br>`"Yes."` | `"Really?!" asked Tom.`<br>`"Yes."` | **fixed** | pass |
| D5 | `"I am leaving." He closed the door.` | `"I am leaving."`<br>`He closed the door.` | ok<br>`"I am leaving."`<br>`He closed the door.` | ok<br>`"I am leaving."`<br>`He closed the door.` | ok<br>`"I am leaving."`<br>`He closed the door.` | ok<br>`"I am leaving."`<br>`He closed the door.` | `"I am leaving."`<br>`He closed the door.` | `"I am leaving."`<br>`He closed the door.` | unchanged (pass) | pass |
| D6 | `She said, "He told me 'Run!' and I did." Then silence.` | `She said, "He told me 'Run!' and I did."`<br>`Then silence.` | **FAIL**<br>`She said, "He told me 'Run!'`<br>`and I did."`<br>`Then silence.` | ok<br>`She said, "He told me 'Run!' and I did."`<br>`Then silence.` | ok<br>`She said, "He told me 'Run!' and I did."`<br>`Then silence.` | ok<br>`She said, "He told me 'Run!' and I did."`<br>`Then silence.` | `She said, "He told me 'Run!'`<br>`and I did."`<br>`Then silence.` | `She said, "He told me 'Run!' and I did."`<br>`Then silence.` | **fixed** | pass |
| D7 | `"Well—I suppose—yes," he said. She nodded.` | `"Well—I suppose—yes," he said.`<br>`She nodded.` | ok<br>`"Well—I suppose—yes," he said.`<br>`She nodded.` | ok<br>`"Well—I suppose—yes," he said.`<br>`She nodded.` | ok<br>`"Well—I suppose—yes," he said.`<br>`She nodded.` | ok<br>`"Well—I suppose—yes," he said.`<br>`She nodded.` | `"Well—I suppose—yes," he said.`<br>`She nodded.` | `"Well—I suppose—yes," he said.`<br>`She nodded.` | unchanged (pass) | pass |
| D8 | `He said—and I quote—"No." Then he left.` | `He said—and I quote—"No."`<br>`Then he left.` | **FAIL**<br>`He said—and I quote—"No." Then he left.` | **FAIL**<br>`He said—and I quote—"No." Then he left.` | **FAIL**<br>`He said—and I quote—"No." Then he left.` | ok<br>`He said—and I quote—"No."`<br>`Then he left.` | `He said—and I quote—"No."`<br>`Then he left.` | `He said—and I quote—"No."`<br>`Then he left.` | **still fails** | **fixed on main** |
| D9 | `“Stop.” Then he left.` | `“Stop.”`<br>`Then he left.` | **FAIL**<br>`“Stop.” Then he left.` | ok<br>`“Stop.”`<br>`Then he left.` | ok<br>`“Stop.”`<br>`Then he left.` | ok<br>`“Stop.”`<br>`Then he left.` | `“Stop.”`<br>`Then he left.` | `“Stop.”`<br>`Then he left.` | **fixed** | pass |
| D10 | `“Is it?” he asked.` | `“Is it?” he asked.` | ok<br>`“Is it?” he asked.` | ok<br>`“Is it?” he asked.` | ok<br>`“Is it?” he asked.` | ok<br>`“Is it?” he asked.` | `“Is it?”`<br>`he asked.` | `“Is it?” he asked.` | unchanged (pass) | pass |
| D11 | `I waited… Then it happened.` | `I waited…`<br>`Then it happened.` | **FAIL**<br>`I waited… Then it happened.` | ok<br>`I waited…`<br>`Then it happened.` | ok<br>`I waited…`<br>`Then it happened.` | ok<br>`I waited…`<br>`Then it happened.` | `I waited… Then it happened.` | `I waited… Then it happened.` | **fixed** | pass |
| D12 | `No way!!! I can't believe it.` | `No way!!!`<br>`I can't believe it.` | ok<br>`No way!!!`<br>`I can't believe it.` | ok<br>`No way!!!`<br>`I can't believe it.` | ok<br>`No way!!!`<br>`I can't believe it.` | ok<br>`No way!!!`<br>`I can't believe it.` | `No way!!!`<br>`I can't believe it.` | `No way!!! I can't believe it.` | unchanged (pass) | pass |
| [D13](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/README.md) | `what even is this. broh !! that is so sad.` | `what even is this.`<br>`broh !!`<br>`that is so sad.` | **FAIL**<br>`what even is this.`<br>`broh !`<br>`!`<br>`that is so sad.` | ok<br>`what even is this.`<br>`broh !!`<br>`that is so sad.` | ok<br>`what even is this.`<br>`broh !!`<br>`that is so sad.` | ok<br>`what even is this.`<br>`broh !!`<br>`that is so sad.` | `what even is this.`<br>`broh !`<br>`!`<br>`that is so sad.` | `what even is this.`<br>`broh !!`<br>`that is so sad.` | **fixed** | pass |

- **D1** — KnowSeams README (@SteveJSteiner). Reported: nupunkt splits `"What!` / `Captain Forrester?"` / `cried I.`. SEAMS' own 3-unit output (quote kept whole) is also accepted.
- **D2** — yasbd-lib benchmarks README (2026-09-25). Convention: yasbd prefers the whole quotation as one unit; both accepted.
- **D3** — nupunkt 0.7.0 development notes.
- **D4** — constructed for this page. Capitalized attribution after `?!` (same pattern as D1).
- **D5** — constructed for this page.
- **D6** — constructed for this page. Nested quotes.
- **D7** — constructed for this page. Em-dash interruptions.
- **D8** — constructed for this page.
- **D9** — nupunkt 0.7.0 development notes. Unicode closing quote.
- **D10** — nupunkt 0.7.0 development notes.
- **D11** — nupunkt 0.7.0 development notes. Unicode ellipsis.
- **D12** — nupunkt 0.7.0 development notes.
- **D13** — yasbd-lib benchmarks README (2026-09-25). README: nupunkt is 'over-aggressive on double exclamation marks (`broh !`, `!`)'. Excerpt of the chat-log input.

### Outside Punkt's input model: no space after the period, CJK, other languages

| # | Input | Expected | nupunkt 0.6.0 | nupunkt 0.7.0 default | nupunkt 0.7.0 adaptive | nupunkt main (unreleased) default | nltk 3.10.3 punkt_tab | pysbd 0.3.4 | Verdict | 0.7.0 to main |
|---|---|---|---|---|---|---|---|---|---|---|
| [S1](https://github.com/nltk/nltk/issues/2082) | `Mary had little lamb.Mary had a little lamb` | `Mary had little lamb.`<br>`Mary had a little lamb` | **FAIL**<br>`Mary had little lamb.Mary had a little lamb` | **FAIL**<br>`Mary had little lamb.Mary had a little lamb` | **FAIL**<br>`Mary had little lamb.Mary had a little lamb` | **FAIL**<br>`Mary had little lamb.Mary had a little lamb` | `Mary had little lamb.Mary had a little lamb` | `Mary had little lamb.Mary had a little lamb` | **still fails** | still fails |
| [S2](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/README.md) | `今日はいい天気ですね。明日から雨が降るそうです。外出するなら傘を持って行ったほうがいいでしょう。 「すみません、駅はどちらですか？」と観光客が聞いた。` | `今日はいい天気ですね。`<br>`明日から雨が降るそうです。`<br>`外出するなら傘を持って行ったほうがいいでしょう。`<br>`「すみません、駅はどちらですか？」と観光客が聞いた。` | **FAIL**<br>`今日はいい天気ですね。明日から雨が降るそうです。外出するなら傘を持って行ったほうがいいでしょう。 「すみません、駅はどちらですか？」と観光客が聞いた。` | **FAIL**<br>`今日はいい天気ですね。明日から雨が降るそうです。外出するなら傘を持って行ったほうがいいでしょう。 「すみません、駅はどちらですか？」と観光客が聞いた。` | **FAIL**<br>`今日はいい天気ですね。明日から雨が降るそうです。外出するなら傘を持って行ったほうがいいでしょう。 「すみません、駅はどちらですか？」と観光客が聞いた。` | **FAIL**<br>`今日はいい天気ですね。明日から雨が降るそうです。外出するなら傘を持って行ったほうがいいでしょう。 「すみません、駅はどちらですか？」と観光客が聞いた。` | `今日はいい天気ですね。明日から雨が降るそうです。外出するなら傘を持って行ったほうがいいでしょう。 「すみません、駅はどちらですか？」と観光客が聞いた。` | `今日はいい天気ですね。`<br>`明日から雨が降るそうです。`<br>`外出するなら傘を持って行ったほうがいいでしょう。`<br>`「すみません、駅はどちらですか？`<br>`」と観光客が聞いた。` | **still fails** | still fails |
| [S3](https://github.com/speedyk-005/yasbd-lib/blob/main/benchmarks/README.md) | `Le manuel, c.-à-d. la version complète, a été publié après une longue m.-à-j. du système interne. Le rendez-vous, noté R.-V. dans le dossier administratif, a été déplacé à 14 h. après une d.-h. d'attente.` | `Le manuel, c.-à-d. la version complète, a été publié après une longue m.-à-j. du système interne.`<br>`Le rendez-vous, noté R.-V. dans le dossier administratif, a été déplacé à 14 h. après une d.-h. d'attente.` | **FAIL**<br>`Le manuel, c.-à-d.`<br>`la version complète, a été publié après une longue m.-à-j.`<br>`du système interne.`<br>`Le rendez-vous, noté R.-V.`<br>`dans le dossier administratif, a été déplacé à 14 h. après une d.-h.`<br>`d'attente.` | **FAIL**<br>`Le manuel, c.-à-d.`<br>`la version complète, a été publié après une longue m.-à-j.`<br>`du système interne.`<br>`Le rendez-vous, noté R.-V.`<br>`dans le dossier administratif, a été déplacé à 14 h. après une d.-h.`<br>`d'attente.` | ok<br>`Le manuel, c.-à-d. la version complète, a été publié après une longue m.-à-j. du système interne.`<br>`Le rendez-vous, noté R.-V. dans le dossier administratif, a été déplacé à 14 h. après une d.-h. d'attente.` | **FAIL**<br>`Le manuel, c.-à-d.`<br>`la version complète, a été publié après une longue m.-à-j.`<br>`du système interne.`<br>`Le rendez-vous, noté R.-V.`<br>`dans le dossier administratif, a été déplacé à 14 h. après une d.-h.`<br>`d'attente.` | `Le manuel, c.-à-d. la version complète, a été publié après une longue m.-à-j.`<br>`du système interne.`<br>`Le rendez-vous, noté R.-V. dans le dossier administratif, a été déplacé à 14 h. après une d.-h. d'attente.` | `Le manuel,`<br>`c.-à-d.`<br>`la version complète, a été publié après une longue m.`<br>`-à-j.`<br>`du système interne.`<br>`Le rendez-vous, noté R.`<br>`-V.`<br>`dans le dossier administratif, a été déplacé à 14 h.`<br>`après une d.`<br>`-h.`<br>`d'attente.` | **still fails** (adaptive ok) | still fails |
| [S4](https://github.com/speedyk-005/yasbd-lib/issues/30#issuecomment-4639723570) | `Segundo o acórdão do tribunal (vid. capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of. n.º 89/2026 emitido pela presidência da assoc. comercial.` | `Segundo o acórdão do tribunal (vid. capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of. n.º 89/2026 emitido pela presidência da assoc. comercial.` | **FAIL**<br>`Segundo o acórdão do tribunal (vid.`<br>`capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of.`<br>`n.º 89/2026 emitido pela presidência da assoc. comercial.` | **FAIL**<br>`Segundo o acórdão do tribunal (vid.`<br>`capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of.`<br>`n.º 89/2026 emitido pela presidência da assoc. comercial.` | **FAIL**<br>`Segundo o acórdão do tribunal (vid.`<br>`capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of.`<br>`n.º 89/2026 emitido pela presidência da assoc. comercial.` | **FAIL**<br>`Segundo o acórdão do tribunal (vid.`<br>`capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of.`<br>`n.º 89/2026 emitido pela presidência da assoc. comercial.` | `Segundo o acórdão do tribunal (vid.`<br>`capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of.`<br>`n.º 89/2026 emitido pela presidência da assoc.`<br>`comercial.` | `Segundo o acórdão do tribunal (vid. capítulo II, parágrafo 3º, item B), a diretoria executiva da Beta S.A. descumpriu deliberadamente o of.`<br>`n.`<br>`º 89/2026 emitido pela presidência da assoc.`<br>`comercial.` | **still fails** | still fails |

- **S1** — nltk/nltk#2082.
- **S2** — yasbd-lib benchmarks README (2026-09-25). First four sentences of the README's 26-sentence Japanese passage.
- **S3** — yasbd-lib benchmarks README (2026-09-25). First line of the README's French passage.
- **S4** — speedyk-005/yasbd-lib#30 (Portuguese comparison). One sentence of the Portuguese comparison.

### Summary

| Verdict (0.6.0 to 0.7.0 default) | Cases |
|---|---|
| fixed | 8 |
| still fails | 9 |
| still fails (output changed) | 5 |
| unchanged (pass) | 19 |

0.7.0 default does not match the expected output on: A1, A2, A15, A18, N1, N2, N3, N4, D1, D8, S1, S2, S3, S4.

| Verdict (0.7.0 to main) | Cases |
|---|---|
| fixed on main | 2 |
| pass | 27 |
| still fails | 11 |
| still fails (output changed) | 1 |

main does not match the expected output on: A1, A2, A15, A18, N1, N2, N4, D1, S1, S2, S3, S4.

Idempotency: cases where re-splitting one of the output sentences splits it again: nupunkt 0.6.0: A1, D12; nupunkt 0.7.0 default: none; nupunkt 0.7.0 adaptive: none; nupunkt main (unreleased) default: none; nltk 3.10.3 punkt_tab: D12; pysbd 0.3.4: S3.

### Layout and page breaks

`sent_tokenize` is Punkt only. `sentences()` uses the segmentation interface (layout-aware by default on main, see `docs/layout.md`).

| # | Input | Expected | main `sent_tokenize` | nupunkt 0.7.0 `sentences()` | main `sentences()` | main `sentences(line_breaks=True)` |
|---|---|---|---|---|---|---|
| L1 | `INTRODUCTION\n\nThe court held for the plaintiff. It awarded fees.` | `INTRODUCTION`<br>`The court held for the plaintiff.`<br>`It awarded fees.` | **FAIL**<br>`INTRODUCTION\n\nThe court held for the plaintiff.`<br>`It awarded fees.` | **FAIL**<br>`INTRODUCTION\n\nThe court held for the plaintiff.`<br>`It awarded fees.` | ok<br>`INTRODUCTION`<br>`The court held for the plaintiff.`<br>`It awarded fees.` | ok<br>`INTRODUCTION`<br>`The court held for the plaintiff.`<br>`It awarded fees.` |
| L2 | `The court held that the\n\ndefendant had waived the claim. It ruled.` | `The court held that the defendant had waived the claim.`<br>`It ruled.` | ok<br>`The court held that the\n\ndefendant had waived the claim.`<br>`It ruled.` | ok<br>`The court held that the\n\ndefendant had waived the claim.`<br>`It ruled.` | ok<br>`The court held that the\n\ndefendant had waived the claim.`<br>`It ruled.` | ok<br>`The court held that the\n\ndefendant had waived the claim.`<br>`It ruled.` |
| L3 | `The defendant waived the claim under the doc-\n\ntrine of laches. The court agreed.` | `The defendant waived the claim under the doc- trine of laches.`<br>`The court agreed.` | ok<br>`The defendant waived the claim under the doc-\n\ntrine of laches.`<br>`The court agreed.` | ok<br>`The defendant waived the claim under the doc-\n\ntrine of laches.`<br>`The court agreed.` | ok<br>`The defendant waived the claim under the doc-\n\ntrine of laches.`<br>`The court agreed.` | ok<br>`The defendant waived the claim under the doc-\n\ntrine of laches.`<br>`The court agreed.` |
| L4 | `The court held that the\n\n12\n\ndefendant had waived the claim.` | `The court held that the defendant had waived the claim.` | **FAIL**<br>`The court held that the\n\n12\n\ndefendant had waived the claim.` | **FAIL**<br>`The court held that the\n\n12\n\ndefendant had waived the claim.` | **FAIL**<br>`The court held that the`<br>`12`<br>`defendant had waived the claim.` | **FAIL**<br>`The court held that the`<br>`12`<br>`defendant had waived the claim.` |
| L5 | `The court held that the\n\n12\n\ndefendant had waived the claim.` | `The court held that the defendant had waived the claim.` | ok<br>`The court held that the\n \ndefendant had waived the claim.` | `AttributeError: module 'nupunkt' has no attribute 'blank_page_furniture'` | ok<br>`The court held that the\n \ndefendant had waived the claim.` | ok<br>`The court held that the\n \ndefendant had waived the claim.` |
| L6 | `The parties agree:\n1. Payment is due monthly\n2. Notice must be written\n3. Disputes go to arbitration` | `The parties agree:`<br>`1. Payment is due monthly`<br>`2. Notice must be written`<br>`3. Disputes go to arbitration` | **FAIL**<br>`The parties agree:\n1. Payment is due monthly\n2. Notice must be written\n3. Disputes go to arbitration` | **FAIL**<br>`The parties agree:\n1. Payment is due monthly\n2. Notice must be written\n3. Disputes go to arbitration` | **FAIL**<br>`The parties agree:\n1. Payment is due monthly\n2. Notice must be written\n3. Disputes go to arbitration` | ok<br>`The parties agree:`<br>`1. Payment is due monthly`<br>`2. Notice must be written`<br>`3. Disputes go to arbitration` |

- **L1** — constructed. Heading, blank line, sentence.
- **L2** — constructed. Sentence split across a blank line; the next block starts lowercase.
- **L3** — constructed. Hyphenated word across a page break (no de-hyphenation is done).
- **L4** — constructed. Page number between the halves of a sentence.
- **L5** — constructed. L4 after `blank_page_furniture(text)` (same length, page number blanked).
- **L6** — constructed. Numbered list, one item per line, no terminal punctuation (items as units).

Layout cells show `\n` literally; the pass check collapses whitespace.

### Determinism: output depends on earlier calls

Input A: `He works at Acme Inc. The company is large.`  Input B: `See Acme Inc. for details.`. Each line is a fresh process.

```
nupunkt 0.6.0: B alone      -> ['See Acme Inc. for details.']
nupunkt 0.6.0: A, then B    -> ['See Acme Inc.', 'for details.']
nupunkt 0.7.0 default: B alone      -> ['See Acme Inc. for details.']
nupunkt 0.7.0 default: A, then B    -> ['See Acme Inc. for details.']
nupunkt main (unreleased) default: B alone      -> ['See Acme Inc. for details.']
nupunkt main (unreleased) default: A, then B    -> ['See Acme Inc. for details.']
```

- nupunkt 0.6.0: all 41 cases in one process vs one process per case differ on 1 case(s): A3.
- nupunkt 0.7.0 default: all 41 cases in one process vs one process per case differ on 0 case(s).
- nupunkt 0.7.0 adaptive: all 41 cases in one process vs one process per case differ on 0 case(s).
- nupunkt main (unreleased) default: all 41 cases in one process vs one process per case differ on 0 case(s).

### API: return types and paragraphs

```
# nupunkt 0.6.0
sent_tokenize(t) -> ['First sentence.', 'Second one.', 'New paragraph here.']
sent_tokenize(t, return_confidence=True) -> ValueError: return_confidence is only available in adaptive mode
sent_tokenize(t, adaptive=True, return_confidence=True) -> [('First sentence.', 0.9), ('Second one.', 0.7), ('New paragraph here.', 1.0)]
para_tokenize(t) -> ['First sentence.  Second one.', '\n\nNew paragraph here.']
paragraphs(t) -> AttributeError: module 'nupunkt' has no attribute 'paragraphs'
[[s.text for s in p.sentences] for p in segment(t).paragraphs] -> AttributeError: module 'nupunkt' has no attribute 'segment'
# nupunkt 0.7.0
sent_tokenize(t) -> ['First sentence.', 'Second one.', 'New paragraph here.']
sent_tokenize(t, return_confidence=True) -> ValueError: return_confidence is only available in adaptive mode
sent_tokenize(t, adaptive=True, return_confidence=True) -> [('First sentence.', 0.9), ('Second one.', 0.7), ('New paragraph here.', 1.0)]
para_tokenize(t) -> ['First sentence.  Second one.', '\n\nNew paragraph here.']
paragraphs(t) -> ['First sentence.  Second one.', 'New paragraph here.']
[[s.text for s in p.sentences] for p in segment(t).paragraphs] -> [['First sentence.', 'Second one.'], ['New paragraph here.']]
# nupunkt main (unreleased): output identical to nupunkt 0.7.0
```

### Spans: legacy contiguous vs tight

```
# nupunkt 0.6.0
sent_spans(t) -> [(0, 19), (19, 32), (32, 39)]
[t[a:b] for a, b in sent_spans(t)] -> ['  First sentence.  ', 'Second one.\n\n', 'Third. ']
sentence_spans(t) -> AttributeError: module 'nupunkt' has no attribute 'sentence_spans'
[t[a:b] for a, b in sentence_spans(t)] -> AttributeError: module 'nupunkt' has no attribute 'sentence_spans'
# nupunkt 0.7.0
sent_spans(t) -> [(0, 19), (19, 32), (32, 39)]
[t[a:b] for a, b in sent_spans(t)] -> ['  First sentence.  ', 'Second one.\n\n', 'Third. ']
sentence_spans(t) -> [(2, 17), (19, 30), (32, 38)]
[t[a:b] for a, b in sentence_spans(t)] -> ['First sentence.', 'Second one.', 'Third.']
# nupunkt main (unreleased): output identical to nupunkt 0.7.0
```

### Training on a degenerate corpus (math domain error)

```
# nupunkt 0.6.0
len(train_model(corpus, abbreviations=abbrevs, output_path=None).get_params().abbrev_types) -> ValueError: math domain error
# nupunkt 0.7.0
len(train_model(corpus, abbreviations=abbrevs, output_path=None).get_params().abbrev_types) -> 5935
# nupunkt main (unreleased): output identical to nupunkt 0.7.0
```

### Public abbreviation access

```
# nupunkt 0.6.0
tok.tokenize(x) -> ['The shipment went to Acme Wdg.', 'Holdings last week.']
tok.add_abbreviation('Wdg.'); tok.tokenize(x) -> ['The shipment went to Acme Wdg.', 'Holdings last week.']
'wdg' in tok.abbreviations -> AttributeError: 'PunktSentenceTokenizer' object has no attribute 'abbreviations'
'wdg' in tok._params.abbrev_types  # private -> True
tok.remove_abbreviation('Wdg'); tok.tokenize(x) -> ['The shipment went to Acme Wdg.', 'Holdings last week.']
# nupunkt 0.7.0
tok.tokenize(x) -> ['The shipment went to Acme Wdg.', 'Holdings last week.']
tok.add_abbreviation('Wdg.'); tok.tokenize(x) -> ['The shipment went to Acme Wdg. Holdings last week.']
'wdg' in tok.abbreviations -> AttributeError: 'PunktSentenceTokenizer' object has no attribute 'abbreviations'
'wdg' in tok._params.abbrev_types  # private -> True
tok.remove_abbreviation('Wdg'); tok.tokenize(x) -> ['The shipment went to Acme Wdg.', 'Holdings last week.']
# nupunkt main (unreleased)
tok.tokenize(x) -> ['The shipment went to Acme Wdg.', 'Holdings last week.']
tok.add_abbreviation('Wdg.'); tok.tokenize(x) -> ['The shipment went to Acme Wdg. Holdings last week.']
'wdg' in tok.abbreviations -> True
'wdg' in tok._params.abbrev_types  # private -> True
tok.remove_abbreviation('Wdg'); tok.tokenize(x) -> ['The shipment went to Acme Wdg.', 'Holdings last week.']
```

### Diagnostic probes for still-open cases

```
# nupunkt 0.6.0
'term' in abbrev_types -> True
'sr' in abbrev_types -> True
'ltd' in abbrev_types -> True
'no' in abbrev_types -> True
'consultant' in sent_starters -> True
sent_tokenize('1. Definitions.\n2. Term.\n3. Termination.') -> ['1. Definitions.', '2.', 'Term.\n3.', 'Termination.']
sent_tokenize('1. Definitions.\n2. Scope.\n3. Termination.') -> ['1. Definitions.', '2.', 'Scope.', '3.', 'Termination.']
sent_tokenize('He met Sr. Consultant Davis.') -> ['He met Sr.', 'Consultant Davis.']
sent_tokenize('He met Dr. Consultant Davis.') -> ['He met Dr.', 'Consultant Davis.']
sent_tokenize('He said "No." Then he left.') -> ['He said "No." Then he left.']
sent_tokenize('He said "Yes." Then he left.') -> ['He said "Yes."', 'Then he left.']
sent_tokenize("'Do not follow me.' Then he left.") -> ["'Do not follow me.' Then he left."]
sent_tokenize('He moved to the "U.S." He stayed.') -> ['He moved to the "U.S." He stayed.']
sent_tokenize('I waited… I left.') -> ['I waited… I left.']
sent_tokenize('Hello ! ! ! ! How are you?') -> ['Hello !', '!', '!', '!', 'How are you?']
# nupunkt 0.7.0
'term' in abbrev_types -> True
'sr' in abbrev_types -> True
'ltd' in abbrev_types -> True
'no' in abbrev_types -> True
'consultant' in sent_starters -> True
sent_tokenize('1. Definitions.\n2. Term.\n3. Termination.') -> ['1. Definitions.', '2. Term.\n3. Termination.']
sent_tokenize('1. Definitions.\n2. Scope.\n3. Termination.') -> ['1. Definitions.', '2. Scope.', '3. Termination.']
sent_tokenize('He met Sr. Consultant Davis.') -> ['He met Sr.', 'Consultant Davis.']
sent_tokenize('He met Dr. Consultant Davis.') -> ['He met Dr. Consultant Davis.']
sent_tokenize('He said "No." Then he left.') -> ['He said "No." Then he left.']
sent_tokenize('He said "Yes." Then he left.') -> ['He said "Yes."', 'Then he left.']
sent_tokenize("'Do not follow me.' Then he left.") -> ["'Do not follow me.' Then he left."]
sent_tokenize('He moved to the "U.S." He stayed.') -> ['He moved to the "U.S." He stayed.']
sent_tokenize('I waited… I left.') -> ['I waited…', 'I left.']
sent_tokenize('Hello ! ! ! ! How are you?') -> ['Hello !', '!', '!', '!', 'How are you?']
# nupunkt main (unreleased)
'term' in abbrev_types -> False
'sr' in abbrev_types -> True
'ltd' in abbrev_types -> True
'no' in abbrev_types -> True
'consultant' in sent_starters -> True
sent_tokenize('1. Definitions.\n2. Term.\n3. Termination.') -> ['1. Definitions.', '2. Term.', '3. Termination.']
sent_tokenize('1. Definitions.\n2. Scope.\n3. Termination.') -> ['1. Definitions.', '2. Scope.', '3. Termination.']
sent_tokenize('He met Sr. Consultant Davis.') -> ['He met Sr.', 'Consultant Davis.']
sent_tokenize('He met Dr. Consultant Davis.') -> ['He met Dr. Consultant Davis.']
sent_tokenize('He said "No." Then he left.') -> ['He said "No."', 'Then he left.']
sent_tokenize('He said "Yes." Then he left.') -> ['He said "Yes."', 'Then he left.']
sent_tokenize("'Do not follow me.' Then he left.") -> ["'Do not follow me.'", 'Then he left.']
sent_tokenize('He moved to the "U.S." He stayed.') -> ['He moved to the "U.S."', 'He stayed.']
sent_tokenize('I waited… I left.') -> ['I waited… I left.']
sent_tokenize('Hello ! ! ! ! How are you?') -> ['Hello ! ! ! !', 'How are you?']
```

## Notes on the demos

- **API.** `sent_tokenize` returns `list[str]` in both versions. It returns `(text, score)` tuples
  only with `adaptive=True, return_confidence=True`. Without `adaptive=True`, `return_confidence`
  raises `ValueError`, so the `isinstance(sentence, tuple)` branch in redlines never runs for its
  call. `para_tokenize` exists in 0.6.0, and its legacy output keeps the `\n\n` separator on the
  next paragraph. 0.7.0 adds `paragraphs()` and the one-pass `segment()` tree. Both return trimmed
  text.
- **Spans.** `sent_spans` keeps its contiguous 0.6.0 semantics in 0.7.0, so existing offsets do
  not move. The new `sentence_spans` returns tight spans, which is what homogenous-cluster's
  `_trim` does by hand.
- **Training.** The 0.6.0 repro is a degenerate corpus we constructed, not the pipeline's data. It
  uses the pipeline's call shape (`train_model(..., abbreviations=<base abbrevs>, output_path=None)`).
  In 0.6.0 the memory-efficient trainer prunes the type counts every 10,000 tokens while still
  counting every period, so `count_b > N` in the Dunning log-likelihood. 0.7.0 no longer prunes
  mid-pass and clamps the inputs.
- **Abbreviations.** `add_abbreviation` has no effect on a loaded model in 0.6.0 and works in
  0.7.0. **The released 0.7.0 has no public read accessor.** `tok.abbreviations` raises
  `AttributeError`, so reading the set still needs `tok._params.abbrev_types`. Main adds the
  read-only `abbreviations` property (shown above) and `parameters`.
- **Layout.** `sent_tokenize` is unchanged on main: it merges a heading into the next sentence
  (L1). The layout-aware `sentences()` on main splits the heading off. It keeps continuations
  across a blank line when the next block starts lowercase (L2) or the line ends with a hyphen
  (L3). L2 and L3 pass everywhere because Punkt never splits without terminal punctuation; the
  point is that the new blank-line rule does not break them. With a page number between the
  halves (L4), `sentences()` on main returns three pieces including `12`. After
  `blank_page_furniture` (L5), it returns one sentence. Line-per-item lists (L6) need
  `line_breaks=True`. Hyphens are not rejoined: `doc-` / `trine` stays as written.

## Still open

These are the 12 cases where main's default output still does not match the expected output (0.7.0 also fails N3 and D8), plus layout case L4. The
diagnoses rely on the probes above.

| Case(s) | What 0.7.0 does | Why | Kind |
|---|---|---|---|
| A1, A2 | Splits `Sr.` / `Consultant Davis`. The `etc. Johnson` half of A1 is fixed in 0.7.0. | `consultant` is a learned sentence starter, and `Sr.` is not covered by the 0.7.0 prenominal-title rule. `Dr. Consultant` stays joined. | Future fix (title list / sentence-starter curation) |
| A15 | Keeps sentence-final `no.` joined (`No. Are you…`, `neutri no. Maybe…`) | `no` is an abbreviation, which A14 (`no. 5`) needs. The bundled break-rate data for `no` has 19 observations, one short of the 20 minimum. | Punkt trade-off |
| A18 | `Global Ltd. Their lawyers…` stays one sentence | Abbreviation followed by a capitalized word that is not a sentence starter. The bundled model has no orthographic context and no break-rate data for `ltd`. | Punkt design limit, partly data |
| N1, N2, N4 | Inline enumerators split as `2.` / `Pay fee.` | The enumerator rule applies only at line start. Inline `2.` stays ambiguous with a sentence-final number (N5 `num 2. Put it back.` must still split). On main, N1 now splits `Term.` from `3.` (6 pieces instead of 5), because `term` is no longer an abbreviation. | Future heuristic, not attempted |
| D1 | `"What!` / `Captain Forrester?" cried I.` | The reported attribution split (`cried I.`) is fixed in 0.7.0. The break after `"What!` before a capitalized word remains, and SEAMS-style dialog coalescing is out of scope. | Punkt design limit |
| S1 | `lamb.Mary` not split | Punkt only considers a period followed by whitespace. Splitting inside `x.Y` would also break `index.html`-like tokens. | Design limit |
| S2 | Japanese returned as one sentence | `。？！` are not sentence terminators, and there is no Japanese model. | Not supported |
| S3, S4 | French/Portuguese abbreviations split | The bundled model is English. Adaptive 0.7.0 happens to fix S3. The Punkt answer is a model trained on the language. | Out of scope for the English model |
| L4 | `sentences()` makes the page number `12` its own sentence and cuts the sentence around it | This is by design: `blank_page_furniture(text)` must be called first (L5). Nothing removes page numbers automatically. | Opt-in step |

Fixed on main relative to 0.7.0: **N3**, because `term` was removed from the model's abbreviations,
and **D8** (`…"No." Then he left.`), because closing quotes are now transparent when pairing. The
probes above show the same change for `'Do not follow me.' Then` and `"U.S." He`. Changed but still
failing on main: **N1** (see the table). On main, `I waited… I left.` is deliberately not split (see
the probes), and D11 (`I waited… Then it happened.`) still splits.

These 41 cases were picked because someone complained about them, so they are not a representative sample. For aggregate accuracy and speed, see the sibling pages in `docs/benchmarks/`.
