# ROADMAP

The MVP focuses narrowly on **verifying citations the author already made**. The following are deliberate omissions for Phase 1 and may be picked up later. If you find yourself wanting to build any of these, add a note here first instead of starting the implementation.

## Shipped beyond the MVP

Deterministic checks and metadata enrichments rendered alongside Citation Issues. Each extends verification without expanding the tool's mission.

- **Hygiene checks** (`bd54a6e`) — broken `\ref` / `\cref` / `\eqref` cross-references, and bibliography entries that are never `\cite`-d.
- **Retraction detection** (`3dba07d`) — OpenAlex's `is_retracted` flag (mirrors Retraction Watch). Runs as a follow-up DOI lookup for Crossref / Semantic Scholar / arXiv primary hits, so retraction surfaces regardless of which source resolved the reference.
- **Link health** (`92b9bde`) — probes `\url{}`, `\href{}{}`, and `.bib` `url =` fields for reachability (HEAD with GET fallback, async-bounded).

## Queued — informed by 2026-04-29 deep-research synthesis

In rough priority order. Compass (deep-research artifact, see `~/Downloads/compass_artifact_wf-32c7bf15-...md`) ranked these against 2024–2026 prior art; we accepted them after triage.

1. **OpenReview citation enrichment** (1–2 days). When a cited paper is on OpenReview (ICLR 2017+, NeurIPS 2022+, COLM, TMLR, RLC, AISTATS), fetch its accept/reject decision and rating distribution via `openreview-py`. Catches papers cited as foundational that were actually rejected and never re-submitted. Resolve via DOI / arXiv ID; fall back silently when no OpenReview ID is found.

2. **Hugging Face / Papers With Code existence checks** (1–2 days). When the paper claims a model (HF `model_info`), dataset (`mlcroissant.Dataset(jsonld=...)`), or code repo (PWC's arXiv-ID-to-repo mapping), verify the artifact exists and surface license drift / missing dataset cards. CS/ML-native ghost-reference catching that the existing resolver pipeline can't do.

3. **Anonymization audit for double-blind venues** (single LLM pass, half-day). Detect (i) author surname leaks against the body text, (ii) self-revealing phrases like "in our previous work [X]" where X cites a paper sharing the submission's title bigrams, (iii) embedded GitHub repo URLs that reveal author identity, (iv) acknowledgments / author footnotes that shouldn't appear in the submission PDF.

4. **ML numerical sanity pack** (~1 week; replaces the original "statcheck-style" plan). Statcheck's Python port inherits Böschen 2024's documented false-negative problem on non-APA reporting; GRIM/SPRITE are psychology-shaped and irrelevant to accuracy/F1/BLEU. Build CS/ML-native checks instead:
   - Bounded metrics: accuracy / F1 ≤ 100 (or 1.0); flag values that exceed the theoretical maximum.
   - Train/val/test split arithmetic sums to the stated dataset cardinality.
   - Mean ± std impossibility (e.g., `99.5 ± 1.2` — the lower σ tail crosses 100%).
   - Hyperparameter table-vs-text consistency (extract `lr=`, batch size, epochs from prose, fuzzy-match against Table 2; only flag when magnitudes differ by >2×).
   - Result-table-vs-abstract delta consistency.

## Roadmap (deferred but in-scope-spirit)

- **Adversarial Reviewer-2 mode.** A separate synthesis pass with mean-but-defensible framing, written as a distinct artifact rather than diluting the balanced review.
- **Diff mode for revision rounds.** Take a v1 review and a v2 paper; report which v1-flagged issues are still present and what the revision changed.

## Considered and declined

- **PDF prompt-injection sanitizer** (white-text / off-page / font-size-0 detection in PDFs sent to the LLM). Compass rated this "very high confidence, mandatory" against attacks documented in Lin (arXiv:2507.06185) and Keuper (arXiv:2509.10248). Declined for prereview's deployment context: this is a single-user local CLI run on the user's own draft, not a service that ingests PDFs from third parties. The threat model doesn't justify the engineering cost. Reconsider if prereview ever runs against PDFs the user didn't write.

## Possible Phase 1.x improvements

Smaller, in-scope-spirit additions that could land on top of the current pipeline without expanding the mission:

- Resolver: add Unpaywall as a fifth fallback for OA PDF discovery.
- Verify: cache the Sonnet judgement keyed on `(claim_hash, target_doi)` so re-runs of the same paper don't re-spend tokens.
- Ingest: a `--bibtex PATH` option to consume an explicit BibTeX file when the PDF's bibliography is hard to parse.
- Output: an optional sidecar JSON of the verification table for programmatic post-processing (still a Phase-2 question — keep the Markdown as the only artifact unless asked).

## Open decisions (not features — design questions)

- **Suggested-rating bias.** Keuper 2025 (arXiv:2509.10248), running 1,000 ICLR 2024 papers, found ">95% acceptance rates in many models" without any prompt injection. The synthesized 1–10 rating is at material risk of being systematically generous. Compass argues for replacing the numeric range with a calibrated severity-bucketed summary ("issues found at severity {high, medium, low}"). Worth deciding before adding more LLM-judged outputs to the review.

## Phase 2 — explicitly out of scope for the MVP

- **Baseline-scout / missing prior work.** No automatic search for relevant work the author *failed* to cite. Verification is only on what is already cited. (Compass notes this is a credible 1+ week project via Semantic Scholar Recommendations + Claude verification, but precision ceiling is ~50–70% even with discipline — surface only as suggestions if ever built.)
- **Figure and table critique.** No layout, axis, or caption sanity checking. (Compass notes a 1–2 day VLM-based axis/legend check is plausible per Bachhofner et al. 2025; not currently planned.)
- **Formal-proof / Lean integration.** No machine-checked proof verification.
- **Statcheck / GRIM / stat-audit.** No automatic re-derivation of reported statistics in APA form. (See queue item #4 for the CS/ML-native replacement.)
- **Plagiarism / overlap detection.** No n-gram or embedding-based overlap checks against existing literature.
- **Web UI, OpenReview integration, auth, rate limiting, multi-tenant infrastructure.** This is a single-user local CLI by design.
- **Ensembles across multiple LLM providers.** Anthropic only, single API key.
- **Conformal-prediction calibration on the rating.** The output is an honest range with a one-sentence justification, no calibration step. (See "Open decisions" above — the rating itself may be ablated.)
