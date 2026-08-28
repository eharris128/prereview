# Competitive analysis — open-source tools comparable to `prereview`

_Snapshot taken 2026-08-28 via web search plus a read of each project's README. Star counts and activity are approximate as of that date; re-check before relying on them._

## Summary

`prereview` sits at the intersection of three tool families that mostly do not overlap in the wild:

1. **Reference-existence checkers** — is the cited paper real? (≈ prereview's Resolve stage)
2. **Claim-support verifiers** — does the cited paper say what the author says it says? (≈ Verify stage)
3. **LLM review generators** — write a reviewer-style critique. (≈ Synthesize stage)

plus a fourth, adjacent family: **deterministic desk-reject / format checkers** (≈ the hygiene and guard sections).

No open-source tool found does what prereview's pipeline does end-to-end: parse a whole manuscript, resolve every reference, fetch the cited source, judge whether it supports the surrounding claim, and fold that into a review. The nearest pieces are **RefChecker** (best resolver) and **SemanticCite** (best judge), and neither is a pipeline.

## Comparison matrix

| Tool | Inputs | Existence / metadata | Claim support | Retraction | LLM review | Deterministic checks | License | Activity (2026-08) |
|---|---|---|---|---|---|---|---|---|
| **prereview** | PDF, `.tex`+`.bib` | Crossref, S2, arXiv, OpenAlex | Yes — 6 verdicts incl. abstain and infra-fail | OpenAlex | Opus + adversarial Reviewer-2 | hygiene, anonymization, checklist, venue rules, numeric sanity, links, artifacts | Apache-2.0 | — |
| [RefChecker](https://github.com/markrussinovich/refchecker) | arXiv ID, PDF, `.tex`, `.bib`/`.bbl`, text | S2, OpenAlex, Crossref, **DBLP, ACL Anthology** + LLM web search | No | — | No | metadata correction, corrected BibTeX/RIS export | MIT | ~480★, active |
| [Hallucinator](https://github.com/gianlucasb/hallucinator) | PDF | 9 DBs incl. PubMed, Europe PMC, Open Library; offline mode | No | Crossref | No | — | AGPL-3.0 | ~350★, active |
| [SemanticCite](https://github.com/sebhaan/SemanticCite) | claim + reference document pairs | No | **Yes** — supported / partial / unsupported / uncertain | — | No | — | MIT | ~25★ |
| [OpenReviewer](https://github.com/maxidl/openreviewer) | PDF | No | No | — | fine-tuned Llama-3.1-8B | — | unspecified | low |
| [DeepReviewer / CycleReviewer](https://github.com/zhu-minjun/Researcher) | PDF | No | No | — | fine-tuned 14B / 70B, retrieval-augmented | — | research | research |
| [ai-peer-review](https://github.com/poldrack/ai-peer-review) | PDF | No | No | — | 6-model ensemble + meta-review | — | MIT | ~150★ |
| [Ai-Review](https://github.com/NeuroDong/Ai-Review) | `.tex`, PDF, `.docx` | No | No | — | LLM + VLM (figures, layout), venue ranking | prompt-injection detection | MIT | ~600★, active |
| [loupe](https://github.com/AIScientists-Dev/loupe) | PDF | No | No | — | proof-level triage, 6 dimensions | vision-verified finding locations | Apache-2.0 | ~150★ |
| [Rigorous](https://github.com/Agentic-Systems-Lab/rigorous) | PDF | No | No | — | multi-agent, PDF report | — | see repo | active |
| [aclpubcheck](https://github.com/acl-org/aclpubcheck) | PDF | outdated-arXiv-citation check | No | — | No | page limit, margins, fonts, anonymity (ACL venues) | see repo | maintained by ACL |
| [NeurIPS Checklist Assistant](https://github.com/ihsaan-ullah/neurips-checklist-assistant) | PDF + checklist | No | No | — | GPT-4-turbo judges checklist answers | — | see repo | 2024 experiment |

## 1. Reference-existence checkers

This category grew quickly once LLM-hallucinated bibliographies became common. They answer "is this citation real, and is its metadata right?" — prereview's ghost-citation check — and generally go wider on sources than prereview does.

- **RefChecker** (Mark Russinovich). The most polished. Accepts arXiv IDs, PDFs, `.tex`, `.bib`/`.bbl`, and plain text; batch mode up to 50 files. Cross-references Semantic Scholar, OpenAlex, Crossref, DBLP, and ACL Anthology, then uses an LLM with web search (OpenAI / Anthropic / Google / Azure) to flag likely fabrications. Reports metadata mismatches (title, authors, year, venue, DOI, arXiv ID, URL) and exports *corrected* BibTeX and RIS. Existence and metadata only; no claim support. Desktop app, web UI, CLI, and API.
- **Hallucinator**. PDF-only. Queries nine databases in parallel (Crossref, arXiv, DBLP, Semantic Scholar, ACL Anthology, Europe PMC, PubMed, OpenAlex, Open Library); five can be mirrored locally for offline use. Flags retractions via Crossref's retraction metadata (which now carries the Retraction Watch database). Verdicts: Verified / Author Mismatch / Not Found. Existence only.
- Lighter tools: [BibTeX Verifier](https://merfanian.github.io/Bibtex-Verifier/) (web, S2 + Crossref + OpenAlex fallback), [CheckIfExist](https://arxiv.org/html/2602.15871v1) (paper + web tool; multi-source author-consistency scoring), a [Claude Code citation-checker skill](https://github.com/PHY041/claude-skill-citation-checker) (`.bib` against Crossref / S2 / OpenAlex, no keys).
- **[CiteAudit](https://arxiv.org/abs/2602.23452)** (Feb 2026) is a benchmark rather than a tool: human-validated real citations from OpenReview / Google Scholar plus controlled perturbations (title, author, metadata). Useful as an evaluation set for prereview's resolver.

**Relative to prereview:** RefChecker is a superset of the resolver on sources (DBLP, ACL Anthology) and on outputs (corrected BibTeX). prereview's distinct advantages are three-state outcomes with retries / circuit breaking so an outage is disclosed rather than reported as a ghost, and the fact that resolution feeds a downstream support judgement.

## 2. Claim-support verifiers

The defining feature of prereview, and the thinnest category.

- **SemanticCite** ([arXiv 2511.16198](https://arxiv.org/abs/2511.16198)). The same four-class taxonomy prereview uses (supported / partially supported / unsupported / uncertain), with confidence scores and evidence snippets. Retrieval is hybrid BM25 + dense vectors with FlashRank reranking over the *full reference document*. Two deployment modes: cloud LLMs via LiteLLM (OpenAI / Claude / Gemini), or **fine-tuned Qwen3 1.7B / 4B models run locally through Ollama**, which the paper reports at ~84% weighted accuracy. Input is individual claim + reference-document pairs — it does not parse a manuscript or fetch sources. A component, not a pipeline.
- Research systems without a runnable tool located: [CiteCheck](https://arxiv.org/html/2605.27700v1) (retrieval-grounded hallucination detection, structured LLM verifier), [CiteGuard](https://arxiv.org/html/2510.17853v4) (retrieval-augmented attribution validation), [CiteLLM](https://arxiv.org/pdf/2602.23075) (two-layer existence + semantic-alignment check). Commercial: scite.

**Relative to prereview:** SemanticCite's judge is comparable in taxonomy and arguably better instrumented (evidence snippets, confidence). prereview's advantages are the manuscript-level context (citation role classification, surrounding sentences, bibkey grouping), source fetching, and the explicit distinction between "abstract too thin" and "verification unavailable". SemanticCite's small local model is the most interesting thing to borrow: a cheaper stage-3 judge.

## 3. LLM review generators

Crowded, and mostly orthogonal to verification.

- **OpenReviewer** (NAACL 2025). Llama-3.1-8B fully fine-tuned on 79k reviews from top ML conferences; PDF → Marker → review in a conference template (ICLR 2025 default). No verification of any kind; light maintenance.
- **DeepReviewer-14B / CycleReviewer-8B/70B**. Research models. DeepReview adds structured analysis, literature retrieval, and evidence-based argumentation; CycleReviewer is the stricter one.
- **ai-peer-review** (Russ Poldrack). Independent reviews from six models (GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Pro, DeepSeek R1, Llama 4 Maverick, GPT-4o-mini), then a meta-review with a concerns table. This is exactly the multi-provider ensemble prereview's ROADMAP declines.
- **Ai-Review**. Accepts `.tex`, PDF, and `.docx`; LLM review with CoT / few-shot modes, VLM review of figures and layout, relative ranking against papers from the target venue, and **prompt-injection detection for tampered PDFs** — the check prereview considered and declined. Updated August 2026; no citation checks.
- **loupe**. Proof-level triage — arithmetic slips, flipped inequalities, unstated assumptions, wrong constants, quantifier confusion — with findings pinned to PDF bounding boxes and vision-verified (parser mismatches are dropped rather than fabricated). Cloud or local models. Covers the proof / figure critique prereview scopes out.
- **Rigorous**. Multi-agent manuscript review producing a PDF report; hosted at rigorous.review with prompts promised open-source; an "outlet fit" agent in development.

**Relative to prereview:** these produce prose reviews from the paper alone. prereview's prose is anchored by a deterministic citation table and guard findings, leads with severity buckets rather than a rating (LLM reviewers skew generous — see [LLM-as-a-Reviewer](https://arxiv.org/pdf/2605.25415)), and renders the Citation Issues / Coverage / Methodology sections without an LLM so nothing flagged can be dropped.

## 4. Deterministic desk-reject and format checkers

- **aclpubcheck**. The canonical example: page limits, margins, fonts, author formatting, anonymity, and outdated arXiv citations for papers using the ACL style file. Run by ACL publication chairs; recommended pre-submission; also hosted as a Hugging Face Space and a Colab. PDF-based. This is the direct analogue of prereview's venue-rules guard and the obvious template for an ACL entry in `VENUE_RULES`.
- **NeurIPS Checklist Assistant** ([arXiv 2411.03417](https://arxiv.org/pdf/2411.03417)). The 2024 experiment: GPT-4-turbo evaluates authors' answers to the 15-item checklist against the paper text. Reports both false positives and false negatives, and does not check figures, tables, or external links. prereview's checklist linter is the deterministic counterpart (and currently AAAI-27-kit-shaped).
- statcheck / GRIM / SPRITE — psychology-shaped statistical audits; see ROADMAP for why prereview built an ML-native numeric pack instead.

**Relative to prereview:** aclpubcheck is more authoritative for its venue but PDF-only and single-venue; prereview's `.tex`-native guards (anonymization from source, `\ref` hygiene, numeric sanity on tables) have no open-source equivalent found.

## Where prereview is distinct

1. **End-to-end claim verification.** Manuscript → resolve → fetch source → judge support → review. No other open-source tool found does this as a pipeline.
2. **`.tex`-native source checks.** Anonymization audit, cross-reference hygiene, numeric sanity on result tables, reproducibility-checklist linting — all from the source, deterministic, advisory. Nobody else found does numeric sanity; aclpubcheck works on PDFs.
3. **Honest degradation.** Coverage reporting, "couldn't check" kept distinct from "doesn't support", and scriptable exit codes. Every other tool here is a web UI that returns an answer regardless of what failed.

## Candidates to borrow

Triaged into ROADMAP stubs; listed here with provenance.

| Idea | From | Notes |
|---|---|---|
| DBLP and ACL Anthology as resolver sources | RefChecker, Hallucinator | CS-native coverage the current four sources miss (workshop papers, older ACL). |
| Corrected-BibTeX export | RefChecker | prereview already has canonical metadata for every resolved entry; emitting a cleaned `.bib` is mostly rendering. |
| Local verifier option | SemanticCite | Fine-tuned Qwen3-4B via Ollama as an alternative stage-3 judge. Tension with the "Anthropic only, single key" design note. |
| Resolver evaluation set | CiteAudit | Precision / recall of ghost detection against a labelled set, run offline. |
| ACL venue rules | aclpubcheck | Page limit, anonymity, outdated-arXiv-citation rules as a `VenueRules` entry; PDF-only checks map onto the existing `audit_submission_pdf`. |
| Prompt-injection detection | Ai-Review | Still declined for a single-user local CLI; noted as a reference implementation if the deployment context changes. |

## Sources

- RefChecker — https://github.com/markrussinovich/refchecker
- Hallucinator — https://github.com/gianlucasb/hallucinator
- BibTeX Verifier — https://merfanian.github.io/Bibtex-Verifier/
- CheckIfExist — https://arxiv.org/html/2602.15871v1
- claude-skill-citation-checker — https://github.com/PHY041/claude-skill-citation-checker
- CiteAudit — https://arxiv.org/abs/2602.23452
- SemanticCite — https://github.com/sebhaan/SemanticCite · https://arxiv.org/abs/2511.16198
- CiteCheck — https://arxiv.org/html/2605.27700v1
- CiteGuard — https://arxiv.org/html/2510.17853v4
- CiteLLM — https://arxiv.org/pdf/2602.23075
- OpenReviewer — https://github.com/maxidl/openreviewer · https://arxiv.org/html/2412.11948v1
- DeepReview — https://arxiv.org/abs/2503.08569
- CycleResearcher / CycleReviewer — https://github.com/zhu-minjun/Researcher
- ai-peer-review — https://github.com/poldrack/ai-peer-review
- Ai-Review — https://github.com/NeuroDong/Ai-Review
- loupe — https://github.com/AIScientists-Dev/loupe
- Rigorous — https://github.com/Agentic-Systems-Lab/rigorous
- aclpubcheck — https://github.com/acl-org/aclpubcheck
- NeurIPS Checklist Assistant — https://github.com/ihsaan-ullah/neurips-checklist-assistant · https://arxiv.org/pdf/2411.03417
- LLM-as-a-Reviewer benchmark — https://arxiv.org/pdf/2605.25415
- AI-assisted peer review across communities — https://arxiv.org/html/2608.03581v1
