# ROADMAP

The MVP focuses narrowly on **verifying citations the author already made**. The following are deliberate omissions for Phase 1 and may be picked up later. If you find yourself wanting to build any of these, add a note here first instead of starting the implementation.

## Phase 2 — explicitly out of scope for the MVP

- **Baseline-scout / missing prior work.** No automatic search for relevant work the author *failed* to cite. Verification is only on what is already cited.
- **Figure and table critique.** No layout, axis, or caption sanity checking.
- **Formal-proof / Lean integration.** No machine-checked proof verification.
- **Statcheck / GRIM / stat-audit.** No automatic re-derivation of reported statistics.
- **Plagiarism / overlap detection.** No n-gram or embedding-based overlap checks against existing literature.
- **Web UI, OpenReview integration, auth, rate limiting, multi-tenant infrastructure.** This is a single-user local CLI by design.
- **Ensembles across multiple LLM providers.** Anthropic only, single API key.
- **Conformal-prediction calibration on the rating.** The output is an honest range with a one-sentence justification, no calibration step.

## Possible Phase 1.x improvements

These are smaller, in-scope-spirit additions that could land on top of the current pipeline without expanding the mission:

- Resolver: add Unpaywall as a fifth fallback for OA PDF discovery.
- Verify: cache the Sonnet judgement keyed on `(claim_hash, target_doi)` so re-runs of the same paper don't re-spend tokens.
- Ingest: a `--bibtex PATH` option to consume an explicit BibTeX file when the PDF's bibliography is hard to parse.
- Output: an optional sidecar JSON of the verification table for programmatic post-processing (still a Phase-2 question — keep the Markdown as the only artifact unless asked).
