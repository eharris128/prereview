# ROADMAP

`prereview`'s mission is narrow: **verify the citations the author already made**, and catch the mechanically-detectable problems that get a paper desk-rejected before anyone reads it. Everything below either extends that without widening it, or is explicitly out of scope. If you find yourself wanting to build one of the deferred items, add a note here first.

_Last reconciled with the code: 2026-08-28. Comparable tools: [docs/competitive-analysis.md](./docs/competitive-analysis.md)._

## Shipped

Hashes point at the landing commit; later fixes may have touched the same area.

### Citation verification (the core)

- Four-source resolution (Crossref → Semantic Scholar → arXiv → OpenAlex), role-aware support judgement (`5b0f3a6`), issues grouped by bibkey (`fb796ba`).
- **Retraction detection** (`3dba07d`) — OpenAlex `is_retracted` follow-up on every resolved DOI, whichever source resolved it.
- **Resolver wrong-match guard** (2026-08-28) — identifier lookups (DOI / arXiv ID) across all four sources before any title search; title-search hits year-gated (±2, `_SourceConfig.year_tolerance`) against the `.bib` entry; a title+author match with the wrong year is accepted only as a last resort and rendered everywhere as a *weak match*; rejected near-misses are named in the ghost rationale; DataCite `10.48550/arXiv.*` DOIs yield the arXiv ID; arXiv's HTTP-200 "Error" entry for an unknown ID is a terminal miss, not a hit. Motivated by a real case found in the cache: *Attention Is All You Need* (no DOI in the `.bib`) had resolved via Crossref to a 2025 mirror-journal DOI (`10.65215/r5bs2d54`), which beat the real paper because Crossref was queried first and nothing checked the year.
- **Outdated-arXiv-citation check** (2026-08-28) — for entries that cite an arXiv preprint, a `published_version` on the canonical record from Semantic Scholar's merged record, arXiv's registered journal DOI, a strong Crossref / OpenAlex venue hit, a DBLP search (curated, keyless, and the only source that knows the DOI-less NeurIPS / ICLR / ICML versions — this is DBLP's first use in prereview, as a confirmer only, not yet a resolver source per C1), or (when S2 is unavailable) a tightly gated Crossref title search. Crossref's relevance ranking cannot surface well-known titles at all (*BERT* is buried under hundreds of "X-BERT: Pre-training…" derivatives under every query form tried), which is why DBLP is the primary confirmer. Cached with the record like the retraction flag; records cached before the field existed are upgraded on the next run; transient failures leave it un-checked and are disclosed in Coverage (not an exit-3 gap). Rendered under Hygiene with the venue / DOI to switch to. This was the venue-independent half of C5.
- **Verify cache** — stage-3 verdicts are cached under `~/.prereview/cache/verify/`, keyed on (model, bibkey, sentence, evidence mode, prompt version), so re-runs re-spend tokens only on changed citations. Listed under *Next up* until 2026-08-28; it had already shipped.
- **Unpaywall OA-PDF route** — `verify._fetch_full_text` tries the record's OA URL, the `.bib` URL, Unpaywall by DOI, then OpenAlex by DOI. Same stale *Next up* entry; Unpaywall is DOI-only so it never made sense as a *resolver* source.
- **Robustness hardening** (`9d88bd2` … `74145bc`) — three-state resolution outcomes with tenacity retries and a per-source circuit breaker; `VERIFICATION_UNAVAILABLE` kept distinct from honest abstention; LLM-call retries; the *Review coverage & reliability* section; CLI exit codes 0 / 1 / 3 / 4 with clean error messages.
- **OpenReview decision enrichment** (`aa9b24e`; `--openreview`, opt-in) — accept/reject decision and rating distribution for cited papers on OpenReview. Needs credentials and the `[openreview]` extra; degrades cleanly without either.

### Source-level checks (`.tex` input; deterministic, advisory)

- **Hygiene** (`bd54a6e`) — broken `\ref` / `\cref` / `\eqref`, never-cited bib entries.
- **Link health** (`92b9bde`) — `\url{}`, `\href{}`, `.bib` `url =` reachability (HEAD with GET fallback, async-bounded).
- **Reproducibility-checklist linter** (`bc708c7` … `43b13c8`; `--checklist` / `--no-checklist`) — AAAI-27 kit format; self-consistency (unanswered, invalid option, gate-vs-subitem) plus claim-vs-paper presence checks. Parser is structural, so other years/venues are mostly data.
- **Manuscript-structure extraction** (`2730d44`) — title, abstract, author block, acknowledgments, section titles, feeding the guards below.
- **Anonymization audit** (`c49d724`; `--no-anonymize`, `--authors`) — residual identity blocks, self-revealing phrases, identity URLs, acknowledgments in a submission build, dual-submission tells.
- **Submission-readiness / desk-reject guard** (`99b1e3e`; `--venue`, `--gate`, `--abstract-baseline`) — generic detectors (page length, placeholder/empty title or abstract, color result tables, checklist completeness) over a per-venue data table. **The AAAI-27 entry was removed on 2026-08-28 after that deadline passed, so `VENUE_RULES` is currently empty:** `--venue` has nothing to select and only the venue-independent `--abstract-baseline` diff runs. `git show 99b1e3e:src/prereview/venue_rules.py` has the old entry as a template.
- **ML numerical-sanity pack** (`c02d339`; `--no-numeric`) — bounded metrics, split arithmetic, mean ± std range, prose-vs-table hyperparameters, abstract-vs-table deltas. Replaces the original statcheck/GRIM idea, which is psychology-shaped and false-negative-prone on ML reporting (Böschen 2024).
- **Artifact existence checks** (`0e5cbbe`; `--artifacts`, opt-in) — Hugging Face models/datasets and GitHub repos via public REST. Papers With Code is excluded (its API wound down mid-2025).

### Review synthesis

- **Adversarial Reviewer-2 pass** (`b54848b`; `--no-reviewer2`) — a separate rubric-anchored pass rendered as its own section, so it doesn't dilute the balanced review.
- **Severity-bucketed overall assessment, numeric rating off by default** (`eb0bf13`; `--show-rating`) — resolves the long-standing "suggested-rating bias" question. Keuper 2025 (arXiv:2509.10248) found >95% LLM accept rates on ICLR 2024 papers, so the review now leads with critical / major / minor buckets; the 1–10 rating is opt-in and caveated.

## Next up

In rough priority order.

1. **Add the next target venue to `VENUE_RULES`.** Data only — a `VenueRules` entry with the page limit, placeholder markers, color-table policy, and whether a checklist is mandatory; the detectors need no changes. Until this lands the desk-reject guard is effectively just the abstract-baseline diff.
2. **Other venues' reproducibility checklists** (NeurIPS, ACL-ARR). The parser is structural (question / options / response / gate), so this should mostly be discovery rules plus option-set data — but each kit needs a golden fixture like the AAAI-27 one.
3. **Diff mode for revision rounds** — take a v1 review and a v2 paper; report which v1-flagged issues remain and what the revision changed.
4. **Sidecar JSON of the verification table** for programmatic post-processing. Still a Phase-2 question — the Markdown stays the only artifact unless asked.
5. **LLM-assisted anonymization pass** for the semantic self-revelations the deterministic vectors can't see (`anonymize.py` defers this explicitly).

## Candidates — stubs, not yet triaged

Surfaced by the 2026-08-28 competitive analysis. Each is a placeholder: enough to start a plan doc, not a commitment. Promote into *Next up* with a number, or move to *Considered and declined* with a reason.

### C1. DBLP and ACL Anthology as resolver sources
- **What:** two more fallbacks after OpenAlex in `resolve.py`, same three-state `Resolution` contract.
- **Why:** CS-native coverage the current four miss — workshop papers, older ACL, venue-only records without a DOI. RefChecker and Hallucinator both query them.
- **From:** RefChecker, Hallucinator.
- **Size:** ~1 day each; DBLP has a public search API (already wired as a published-version confirmer on 2026-08-28 — `Resolver._dblp_published_version` has the gate, breaker source name, and response parsing a resolver source would reuse), ACL Anthology needs its bib dump or a title-search shim.
- **Open:** does the circuit breaker's per-source budget still hold with six sources on a 100-reference bibliography?

### C2. Corrected-BibTeX export
- **What:** `--export-bib PATH` writing a `.bib` whose resolved entries carry canonical title / authors / year / DOI, unresolved ones passed through untouched and commented.
- **Why:** prereview already holds canonical metadata for every resolved entry; this is rendering, and it's the one output RefChecker users say they want.
- **From:** RefChecker.
- **Size:** half a day.
- **Open:** never silently rewrite — diff-style report alongside, or only write when `--export-bib` is explicit?

### C3. Local verifier option for stage 3
- **What:** a `--verify-model ollama/...` path using SemanticCite's fine-tuned Qwen3-4B (or similar) as the support judge, keeping Opus for synthesis.
- **Why:** stage 3 is the token-heavy stage (one call per citation site); a local judge makes re-runs free and offline.
- **From:** SemanticCite (~84% weighted accuracy reported for the 4B model).
- **Size:** 2–3 days including a calibration run against the current Sonnet verdicts.
- **Open:** conflicts with the "Anthropic only, single key" design note. Decide whether that note is about *synthesis* or the whole pipeline before building.

### C4. Resolver evaluation set
- **What:** an offline harness that runs `resolve.py` against a labelled set of real and perturbed citations and reports ghost-detection precision / recall per source.
- **Why:** the resolver has never been measured; every change to gating or fallback order is currently judged by feel.
- **From:** CiteAudit (human-validated set with a hallucination taxonomy).
- **Size:** 1–2 days; respx fixtures for the API responses so it runs in CI.
- **Open:** licence / availability of the CiteAudit data; otherwise build a small in-house set from the golden fixtures.
- **Evidence (2026-08-28):** the first real wrong match (see *Resolver wrong-match guard*) was found by reading the cache, not by a harness. An in-house set can start from the two in-flight papers' bibliographies plus perturbed copies (wrong year, swapped author, mirror-journal duplicate).

### C5. ACL venue entry
- **What:** `VENUE_RULES["acl-arr"]` — page limit, anonymity expectations, outdated-arXiv-citation rule — mirroring what aclpubcheck enforces.
- **Why:** it's the first concrete candidate for *Next up* #1, and aclpubcheck is an authoritative spec to copy from rather than guess.
- **From:** aclpubcheck.
- **Size:** hours for the data entry. The outdated-arXiv-citation rule shipped on 2026-08-28 as a venue-independent Hygiene check; only the `VenueRules` data remains.
- **Open:** ARR's rolling deadlines mean no "deadline passed" retirement — decide whether venue entries carry an expiry at all.

### C6. Checklist parser for a second kit
- **What:** NeurIPS (or ACL-ARR) checklist support alongside the AAAI-27 kit format.
- **Why:** the linter is the only guard still hard-wired to one venue's artifact; it needs a second kit to prove the "structural parser, venue data" claim.
- **From:** NeurIPS Checklist Assistant (as the LLM contrast case, not a source).
- **Size:** 1–2 days plus a golden fixture.
- **Open:** same as *Next up* #2 — this stub exists to keep it visible next to C5.

### C7. Duplicate bibliography entries
- **What:** a Hygiene finding for two bibkeys with the same normalized title, or the same DOI / arXiv ID.
- **Why:** it is a cheap deterministic check with no false-positive surface, and the AAAI author kit's sample `.bib` ships a literal "Duplicate Entry" that survives into real drafts.
- **From:** the 2026-08-28 look at the in-flight drafts.
- **Size:** hours.
- **Open:** whether to also flag near-duplicates (same title, different year) or leave those to the resolver's weak-match note.

## Considered and declined

- **PDF prompt-injection sanitizer** (white-text / off-page / font-size-0 detection in PDFs sent to the LLM). Rated "mandatory" by the 2026-04-29 deep-research pass against Lin (arXiv:2507.06185) and Keuper (arXiv:2509.10248). Declined for prereview's deployment context: a single-user local CLI run on the user's own draft, not a service ingesting third-party PDFs. Reconsider if that ever changes — Ai-Review ships a reference implementation.
- **Statcheck / GRIM / SPRITE ports.** See the numerical-sanity pack above for why.

## Out of scope

- **Baseline-scout / missing prior work.** Verification is only on what is already cited. A Semantic Scholar Recommendations + Claude pass is credible (~1 week) but its precision ceiling is ~50–70%; if ever built, surface as suggestions only.
- **Figure and table critique.** No layout, axis, or caption checking (a VLM axis/legend check is plausible per Bachhofner et al. 2025; not planned). Ai-Review covers figures; loupe covers proof-level triage.
- **Formal-proof / Lean integration.** No machine-checked proof verification.
- **Plagiarism / overlap detection.** No n-gram or embedding-based overlap checks.
- **Web UI, auth, rate limiting, multi-tenant infrastructure.** Single-user local CLI by design.
- **Ensembles across LLM providers.** Anthropic only, single API key. ai-peer-review exists for anyone who wants the six-model meta-review.
- **Conformal-prediction calibration on the rating.** Moot now that the rating is opt-in and secondary.
