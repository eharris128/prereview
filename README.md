# prereview

AI-assisted pre-submission review of academic preprints, with **verification of every cited reference** as the defining feature — plus a set of deterministic desk-reject guards that run on the `.tex` source.

`prereview` reads a draft (PDF, or `.tex` + `.bib`), parses every in-text citation and its bibliography entry, resolves each entry against Crossref / Semantic Scholar / arXiv / OpenAlex, judges whether the resolved paper actually supports the surrounding claim, runs a battery of mechanical source-level checks, and writes one structured Markdown review next to the input.

It is built on top of [PaperQA2](https://github.com/Future-House/paper-qa) for PDF retrieval and grounding. The contribution of this project is the citation-verification post-processor, the deterministic checks, and the review-generation prompts.

## What it catches

**Citations (the core)**

- **Ghost citations** — bibliography entries that resolve to no canonical record anywhere.
- **Misattributed citations** — the resolved target does not support the surrounding claim.
- **Retracted citations** — OpenAlex's `is_retracted` flag (mirrors Retraction Watch).
- **Abstract-only verifications** of load-bearing claims, surfaced honestly so you can read the cited paper yourself.

**Source hygiene** (`.tex` input)

- Broken `\ref` / `\cref` / `\eqref` targets, and `.bib` entries that are never `\cite`-d.
- Unreachable `\url{}` / `\href{}` / `.bib` `url =` links.

**Desk-reject guards** (`.tex` input; deterministic, no LLM, advisory-framed)

- **Reproducibility-checklist linter** (AAAI-27 kit format) — unanswered items, answers outside the option set, gate-vs-subitem contradictions, and "yes" answers with no supporting evidence in the paper.
- **Double-blind anonymization audit** — residual `\author` / `\thanks` / `\email` blocks, "in our previous work [X]", identity-revealing GitHub / homepage URLs, acknowledgments in a submission build, and an opt-in `--authors` surname grep.
- **Submission readiness** — per-venue rules for page limits (PDF only), placeholder or empty title/abstract, color-coded result tables, and checklist completeness. Per-venue facts are data (`venue_rules.VENUE_RULES`); the table is **currently empty** — the AAAI-27 entry was removed after that deadline passed — so `--venue` has nothing to select until the next venue is added. The `--abstract-baseline` diff is venue-independent and always available.
- **ML numerical-sanity pack** — bounded metrics over their ceiling, train/val/test splits that don't sum to the stated total, mean ± std ranges that escape the metric's bounds, prose-vs-table hyperparameter drift, and abstract headline deltas no results table backs up. Precision over recall: every detector skips when unsure.

**Opt-in network enrichments**

- `--artifacts` — do the claimed Hugging Face models/datasets and GitHub repos actually exist?
- `--openreview` — accept/reject decision and rating distribution for cited papers that are on OpenReview.

## What it does not do

See [ROADMAP.md](./ROADMAP.md). It does not look for *missing* prior work, does not critique figures or proofs, does not run plagiarism or statcheck-style p-value audits, and is a single-user local CLI by design.

## Quickstart

```bash
git clone https://github.com/echarris/prereview.git
cd prereview
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"          # add ",openreview" to the extras for --openreview
export ANTHROPIC_API_KEY=...        # or put it in a .env file (see Environment)
prereview manuscript.tex
```

Output: `manuscript.review.md` next to the input. A previous review is backed up to `<name>.review.md.bak.<timestamp>` rather than overwritten.

### Environment

The CLI auto-loads these keys from a `.env` in the current directory, the input file's directory, or the project root. Existing environment values always win.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | **Required** unless you use `--auth oauth`. Metered Anthropic API credits. |
| `CLAUDE_CODE_OAUTH_TOKEN` | Alternative to the above: routes spend to a Claude subscription. Needs the `[oauth]` extra and the `claude` CLI. |
| `S2_API_KEY` | Optional. Higher Semantic Scholar rate limits. |
| `OPENALEX_API_KEY` | Optional. Keyed OpenAlex access. |
| `PREREVIEW_MAILTO` | Optional. Joins the Crossref / OpenAlex / Unpaywall polite pools (`--mailto` overrides). |
| `HF_TOKEN` | Optional. Lets `--artifacts` probe gated or private Hugging Face artifacts. |
| `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` | Required for `--openreview`. |

#### Two credential paths

`--auth api-key` (the Anthropic API) and `--auth oauth` (the Claude Code CLI, via the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk)) are interchangeable. The
OAuth path needs the optional extra and the CLI on `PATH`:

```bash
uv pip install -e ".[oauth]"
claude --version            # must resolve
prereview draft.tex --auth oauth
```

`--auth oauth` does not require the token variable at all — the CLI authenticates from
`$CLAUDE_CODE_OAUTH_TOKEN` *or* an interactive `claude login`, so a machine already
logged into Claude Code needs no token minted. `--auth auto` (the default) is stricter:
it picks the OAuth path only when `$CLAUDE_CODE_OAUTH_TOKEN` is set *and* the path is
usable, otherwise falling back to `ANTHROPIC_API_KEY` — so the backend never flips on
the strength of a stored login you forgot about. Pin one explicitly when both are set,
which is common under secret-injection wrappers.

Spend on the OAuth path counts against Claude Code subscription limits: a review is one
model call per citation plus a few, so a 50-reference paper is 50+ calls.

## Two input modes

```bash
prereview draft.pdf                       # heuristic PDF parse + LLM bibliography parse
prereview manuscript.tex                  # native .tex/.bib parse — more reliable
prereview manuscript.tex --bib refs.bib   # explicit bibliography path
```

The TeX path is preferred when available: it parses `\cite{}` commands and the BibTeX file directly, so the citation-to-bib linkage is exact rather than heuristic. The bibliography is auto-discovered from `\bibliography{}`, `\addbibresource{}`, or a sibling `references.bib`.

All source-level guards (hygiene, link health, checklist, anonymization, numerical sanity, and the TeX side of submission readiness) need `.tex` input. On PDF input, `prereview` runs citation verification (including retraction detection) and — once a venue is configured — the page-length and placeholder-title checks.

## CLI

```
prereview INPUT [options]

Input
  --bib PATH               (.tex) Explicit .bib path. Default: auto-detect.
  --checklist PATH         (.tex) Explicit reproducibility checklist .tex (AAAI-27 kit
                           format). Default: auto-detect via \input{} or a sibling
                           ReproducibilityChecklist.tex.

Checks (on by default for .tex unless noted)
  --no-checklist           Skip the reproducibility-checklist linter.
  --no-anonymize           Skip the double-blind anonymization audit.
  --authors "A,B"          Extra anonymization check: grep the body for these surnames.
  --no-numeric             Skip the ML numerical-sanity pack.
  --venue ID               Venue whose submission rules to check (desk-reject guard).
                           Default: none — venue-specific checks are skipped. No venue
                           is currently configured.
  --abstract-baseline PATH (.tex) Snapshot the abstract on first run; flag later runs
                           whose abstract changed substantially. Works without --venue.
  --gate                   Exit 4 if a hard desk-reject blocker is found (residual
                           author identity, changed abstract, or a --venue blocker).
  --artifacts              Opt-in: verify claimed HF models/datasets and GitHub repos.
  --openreview             Opt-in: annotate cited papers with OpenReview decisions.

Review
  --no-reviewer2           Skip the adversarial Reviewer-2 pass (one extra Opus call).
  --show-rating            Include the LLM 1–10 rating, with a generosity caveat.
                           Default: off — the review leads with severity buckets.

Models & plumbing
  --auth {auto,api-key,oauth}
                           Credential path. 'oauth' drives the Claude Code CLI
                           ($CLAUDE_CODE_OAUTH_TOKEN or `claude login`); 'api-key'
                           calls the Anthropic API with $ANTHROPIC_API_KEY.
                           Default: auto (oauth only when the token var is set
                           and usable).
  --model NAME             Retrieval/extraction model. Default: anthropic/claude-sonnet-4-6
  --synthesis-model NAME   Review-writing model.       Default: anthropic/claude-opus-4-7
  --no-fetch-cited         Don't download cited works' full text (abstract-only verification).
  --cache-dir PATH         Resolved references + fetched PDFs. Default: ~/.prereview/cache/
  --mailto EMAIL           Polite-pool contact for Crossref / OpenAlex ($PREREVIEW_MAILTO).
  --out PATH               Output path. Default: <input-stem>.review.md next to the input.
  --verbose                Log every stage to stderr.
  --version
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Ran clean and complete. Honest negative verdicts (does-not-support, genuine ghosts) still exit 0. |
| 1 | Failed — the review was not written or is incomplete. |
| 2 | Usage error (bad flag, unknown `--venue`, missing input). |
| 3 | Completed with coverage gaps — a citation could not be checked because of infrastructure, or the prose pass degraded. See *Review coverage & reliability* in the output. |
| 4 | `--gate` tripped on a hard desk-reject blocker (outranks 3). |

## Pipeline

1. **Ingest.** Parse the PDF (PaperQA2 + heuristics + one LLM bibliography pass) or the `.tex`/`.bib` natively. Record every in-text citation with its surrounding sentence(s). On `.tex`, run the deterministic guards: hygiene, checklist, anonymization, submission readiness, numerical sanity.
2. **Probe.** Check link health; optionally check artifact existence (`--artifacts`).
3. **Resolve.** For each bibliography entry, query Crossref → Semantic Scholar → arXiv → OpenAlex until a canonical record is returned, then a follow-up OpenAlex retraction lookup. Outcomes are three-state (resolved / terminal miss / degraded) with retries and a per-source circuit breaker, so an API outage is disclosed as degradation rather than reported as a ghost citation.
4. **Verify.** Classify each citation's role, then ask the LLM whether the resolved paper supports the surrounding claim. Verdicts: *supports*, *partially supports*, *does not support*, *abstract too thin to tell*, *target unavailable*, and a distinct *verification unavailable* for infrastructure failure. Optionally enrich with OpenReview decisions (`--openreview`).
5. **Synthesize.** Opus writes Summary, Strengths, Weaknesses, Questions for the author, and (by default) an adversarial Reviewer-2 pass. Everything else — Overall assessment, Citation issues, every guard section, Coverage & reliability, Methodology — is rendered deterministically from the data, so nothing flagged can be dropped and no metadata can be invented.
6. **Write.** Markdown next to the input; a previous review is backed up first.

Review sections, in order: Review coverage & reliability (only when something degraded — it leads so you see it first) · Anonymization audit · Submission readiness · Overall assessment (critical / major / minor) · Summary · Strengths · Weaknesses · Reviewer 2 (adversarial) · Citation issues · Hygiene checks · Reproducibility checklist · Numerical sanity · Artifact availability · OpenReview decisions · Questions for the author · Methodology and limits of this review. Guard and enrichment sections appear only when their check ran.

## Design notes

- The LLM is allowed to abstain. *Abstract too thin to tell* is a first-class verdict, not papered over — and infrastructure failure is a separate verdict, never conflated with abstention.
- Canonical citation metadata (title, authors, year, DOI) only ever comes from Crossref / Semantic Scholar / arXiv / OpenAlex. The LLM only judges whether retrieved text supports a claim.
- Every deterministic check is precision-over-recall and advisory-framed ("verify this", never "you did this"). Only `BLOCKER`-severity findings affect the exit code, and only under `--gate`.
- The review leads with severity buckets, not a number. LLM reviewers skew generous (>95% accept rates in Keuper 2025), so the 1–10 rating sits behind `--show-rating` and is labeled as secondary.
- Per-venue facts are data, detectors are generic: adding a venue is a `VenueRules` entry, not code.
- Anthropic only, but either credential: a metered API key or a Claude Code OAuth
  token. The OAuth backend pins `setting_sources=[]` and an empty tool set so the
  Claude Code harness (CLAUDE.md, project settings, MCP servers, tools) can never
  leak into a verification prompt and steer a verdict.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q      # all network calls mocked with respx
```

No linter is configured; `python -m py_compile` is the quick syntax check.

## License

Apache-2.0, matching PaperQA2.
