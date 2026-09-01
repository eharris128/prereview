# Contributing to prereview

Thanks for considering a contribution. `prereview` has a deliberately narrow mission:
**verify the citations the author already made, and catch the mechanically-detectable
problems that get a paper desk-rejected**. The guidance below exists mostly to keep
it that way. Read [ROADMAP.md](./ROADMAP.md) first: it lists what has shipped, what is
next, what was considered and declined, and what is out of scope. A change that lands in
the *out of scope* list needs a ROADMAP discussion before code.

## Development setup

```bash
git clone https://github.com/eharris128/prereview.git
cd prereview
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"          # add ",oauth" / ",openreview" for those extras
.venv/bin/python -m pytest -q
```

The suite is fully offline: every HTTP call is mocked with `respx`, and every LLM call is
stubbed. A test that needs the network or a real API key is a bug in the test. Tests run
in a few seconds; please run them before every push.

There is no linter configured yet. `python -m py_compile` is the quick syntax check.

### Credentials

Never commit an API key, a `.env` file, or a real manuscript. `.env` and
`tests/fixtures/local/` are gitignored for exactly this reason. If you need a real draft
to reproduce an issue, stage it under `tests/fixtures/local/` and describe the shape of
the problem in the PR rather than attaching the paper.

## Design principles to respect

Every part of the codebase leans on a few rules. A PR that quietly breaks one of them will
be sent back even if it works.

- **Canonical metadata only ever comes from the resolvers** (Crossref, Semantic Scholar,
  arXiv, OpenAlex, and DBLP as a confirmer). The LLM judges whether retrieved text
  supports a claim; it never invents or "corrects" a title, author, year, or DOI.
- **Identifiers beat searches, and searches are year-gated.** A title hit with the wrong
  year is a *weak match* and is labelled as such everywhere it is shown. Never silently
  accept one.
- **Three-state outcomes.** A resolver or LLM call is *resolved*, a *terminal miss*, or
  *degraded*. Infrastructure failure must never be reported as a ghost citation, and
  `VERIFICATION_UNAVAILABLE` is never conflated with the LLM honestly abstaining.
- **Precision over recall, advisory framing.** Deterministic checks skip when unsure and
  say "verify this", never "you did this". Only `BLOCKER` findings affect the exit code,
  and only under `--gate`.
- **Per-venue facts are data, detectors are generic.** Adding a venue is a
  `VenueRules` entry, not a new code path.
- **The review leads with severity buckets, not a number.** The 1–10 rating stays behind
  `--show-rating`.

## Common contributions

**Adding a venue.** Add a `VenueRules` entry to `src/prereview/venue_rules.py` with the
page limit, placeholder markers, color-table policy, and whether a checklist is
mandatory. `git show 99b1e3e:src/prereview/venue_rules.py` has the retired AAAI-27 entry
as a template. Add a test in `tests/test_venue_rules.py` that exercises each rule you set,
and note the venue's deadline or rolling status in the entry so it can be retired cleanly.

**Adding a reproducibility-checklist kit.** The parser in `checklist.py` is structural
(question / options / response / gate), so a new kit is mostly discovery rules plus option
data. Every kit needs a golden fixture like `tests/fixtures/ReproducibilityChecklist.tex`.

**Adding a resolver source.** Follow the existing `_Outcome` contract in `resolve.py`,
register the source with the circuit breaker, and mock its responses with `respx` in
`tests/test_resolve.py`. Check that the per-source breaker budget still holds on a
100-reference bibliography.

**Adding a deterministic check.** Put it in the module that owns its stage, make it
advisory, give it a way to abstain, and add both a positive and a false-positive test.

## Pull requests

- Keep one logical change per PR. Small PRs get reviewed; large ones get postponed.
- Add or update tests for every behaviour change. Mock the network.
- Update `README.md` if a flag, environment variable, or exit code changed, and
  reconcile `ROADMAP.md` if you shipped or declined something it lists.
- Commit messages follow the existing `type(scope): summary` style, e.g.
  `feat(resolve): ...`, `fix(verify): ...`, `docs(readme): ...`, `chore(venue): ...`.

## Reporting bugs

Open an issue with the command you ran, the exit code, the *Review coverage &
reliability* section of the generated review, and a minimal `.tex` + `.bib` that
reproduces the problem. Do not attach an unpublished manuscript; a two-entry
bibliography that shows the same behaviour is far more useful.

## License

By contributing you agree that your contributions are licensed under the project's
[MIT License](./LICENSE).
