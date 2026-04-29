# prereview

AI-assisted pre-submission review of academic preprints, with **verification of every cited reference** as the defining feature.

`prereview` reads a draft PDF, parses every in-text citation and its bibliography entry, resolves each entry against Crossref / Semantic Scholar / arXiv / OpenAlex, judges whether the resolved paper actually supports the surrounding claim, and writes one structured Markdown review next to the PDF.

It is built on top of [PaperQA2](https://github.com/Future-House/paper-qa) for retrieval and grounding. The contribution of this project is the citation-verification post-processor and the review-generation prompt.

## What it catches

- **Ghost citations** — bibliography entries that resolve to no canonical record anywhere.
- **Misattributed citations** — citations whose resolved target does not actually support the surrounding claim.
- **Abstract-only verifications** of load-bearing claims, surfaced honestly so you can read the cited paper yourself.

## What it does not do

See [ROADMAP.md](./ROADMAP.md). It does not look for *missing* prior work, does not critique figures or proofs, and does not run any plagiarism or stat-audit checks.

## Quickstart

```bash
git clone https://github.com/echarris/prereview.git
cd prereview
uv venv --python 3.12 .venv
uv pip install -e .
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
prereview path/to/draft.pdf
```

Output: `path/to/draft.review.md`.

The CLI auto-loads `.env` from the project root, the input file's directory, and the current working directory — set keys there once and forget about them.

If a Semantic Scholar API key is available, set `S2_API_KEY` for higher rate limits. Set `PREREVIEW_MAILTO` (or pass `--mailto`) to a real email address to join the polite pools at Crossref / OpenAlex / Unpaywall. Neither is required.

### Two input modes

`prereview` accepts either a PDF or a TeX source:

```bash
prereview draft.pdf                       # heuristic PDF parse + LLM bibliography parse
prereview manuscript.tex                  # native .tex/.bib parse — more reliable
prereview manuscript.tex --bib refs.bib   # explicit bibliography path
```

The TeX path is preferred when available: it parses `\cite{}` commands and the BibTeX file directly, so the citation-to-bib linkage is exact rather than heuristic. Bibliography is auto-discovered from `\bibliography{}`, `\addbibresource{}`, or sibling `references.bib`.

## CLI

```
prereview <paper.pdf> [options]

  --out PATH               Output Markdown path. Default: <paper-stem>.review.md
                           next to the input PDF.
  --model NAME             Model for retrieval/extraction passes.
                           Default: anthropic/claude-sonnet-4-6
  --synthesis-model NAME   Model for the final review-writing pass.
                           Default: anthropic/claude-opus-4-7
  --no-fetch-cited         Skip downloading PDFs of cited works. Faster but
                           verifications fall back to abstract-only.
  --cache-dir PATH         Where to cache resolved references and fetched PDFs.
                           Default: ~/.prereview/cache/
  --verbose                Log every retrieval and verification step to stderr.
```

## Pipeline

1. **Ingest.** Parse the PDF with PaperQA2; extract title, abstract, sections, and the bibliography. For every in-text citation, record the surrounding sentence(s) and the bibliography entry it points to.
2. **Resolve.** For each bibliography entry, query Crossref → Semantic Scholar → arXiv → OpenAlex, in that priority order, until a canonical record is returned. A citation that resolves nowhere is flagged.
3. **Verify.** For each citation that did resolve, ask the LLM whether the resolved paper supports the surrounding claim. Verdicts: *supports*, *partially supports*, *does not support*, *abstract too thin to tell*, *target unavailable*. The LLM never invents metadata; it only judges retrieved text.
4. **Synthesize.** Hand the ingest + verification table to Opus and write a 7-section review: summary, strengths, weaknesses, **citation issues**, questions, suggested rating with confidence range, and methodology / limits of this review.
5. **Write.** Drop the Markdown next to the input PDF. If a previous review exists, back it up to `<name>.review.md.bak.<timestamp>` rather than overwriting silently.

## Design notes

- The LLM is allowed to abstain. *Abstract too thin to tell* is a first-class verdict, not papered over.
- Canonical citation metadata (title, authors, year, DOI) only ever comes from Crossref / Semantic Scholar / arXiv / OpenAlex. The LLM is only allowed to judge whether retrieved text supports a claim.
- The rating is a range with a one-sentence justification, explicitly labeled as an LLM rating. No conformal calibration.
- Anthropic only. Single API key.

## License

Apache-2.0, matching PaperQA2.
