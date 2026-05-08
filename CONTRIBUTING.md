# Contributing to UltraCompress

Thanks for considering a contribution. This guide covers how to file issues, propose changes, and submit pull requests against `github.com/sipsalabs/ultracompress`.

---

## TL;DR

- **Bug reports:** open an issue with reproduction steps. Templates available.
- **Feature requests:** open a discussion in GitHub Discussions first to scope.
- **Pull requests:** small + focused + tested. CLA required for non-trivial changes.
- **Security vulnerabilities:** see `SECURITY.md`. Do NOT open public issues.

---

## Code of conduct

Be technical, be calibrated, be honest. No hype, no punching down on adjacent open-source projects, no off-topic discussions in issues. Disagreements should be technical and reproducible.

Sipsa Labs reserves the right to remove comments or close issues that don't meet this standard.

---

## How to file an issue

### Bug report

Use the bug-report issue template. Include:

1. **Reproduction steps:** exact commands you ran. Include the output (paste, don't paraphrase).
2. **Expected behavior:** what you thought would happen.
3. **Actual behavior:** what did happen.
4. **Environment:** OS (Windows / Linux / Mac), Python version, `pip show ultracompress` version, GPU model if relevant.
5. **Reproducer:** if possible, a minimal script or model that triggers the bug.

We reply within 1 business day for the first 30 days post-launch (2026-05-05 onward) and within 5 business days thereafter.

### Feature request

Open a GitHub Discussion in the "Ideas" category before opening an issue. Discussion lets us scope whether the feature fits the project direction; issues are for confirmed work.

For features that touch the closed-source production pipeline (post-quantization correction overlay training, composed-stack mechanisms), the open-source CLI is NOT the right place — those are commercial-license features. Open a Discussion or email founder@sipsalabs.com.

### Question

Use GitHub Discussions in the "Q&A" category. For commercial / pilot questions, email founder@sipsalabs.com directly.

### Security vulnerability

DO NOT file a public issue. Follow the procedure in `SECURITY.md`:
- Email security@sipsalabs.com (PGP-encrypted preferred), OR
- Open a private security advisory at github.com/sipsalabs/ultracompress/security/advisories/new.

---

## How to open a pull request

### Before you start

For non-trivial changes (more than a typo or 1-line bug fix):
1. Open an issue or Discussion FIRST to scope the work.
2. Wait for maintainer acknowledgment before investing significant time.
3. We don't want you to write a 1,000-line PR that we then ask you to rewrite — let's align on direction first.

For trivial changes (typo, doc clarification, dependency bump):
- Just open the PR.

### Setting up your dev environment

```bash
git clone https://github.com/sipsalabs/ultracompress
cd ultracompress
pip install -e ".[dev]"  # installs ultracompress + dev dependencies (pytest, ruff, mypy)
pytest  # baseline test run; should pass on main
```

### Making your changes

- **Code style:** follow existing patterns. We use `ruff` for linting and formatting; `mypy` for type checking. Run `ruff check . && ruff format . && mypy src/` before submitting.
- **Type hints:** add them to all new functions. The existing codebase aspires to full type coverage.
- **Docstrings:** Google-style, on all public functions / classes.
- **Tests:** add `pytest` tests for any new functionality. Aim for 70%+ coverage on new code; existing code has gaps that we're filling over time.
- **Commit messages:** [Conventional Commits](https://www.conventionalcommits.org/) format. Examples:
  - `fix(cli): correct HF_ORG capitalization`
  - `feat(bench): add hellaswag-100 quick-eval`
  - `docs(readme): clarify install command for v0.4.1+`

### CLA (Contributor License Agreement)

For PRs larger than 10 lines (excluding whitespace and trivial doc changes), you'll need to sign the Sipsa Labs Contributor License Agreement. This grants Sipsa Labs the right to incorporate your contribution into the open-source CLI AND the closed-source production pipeline, while you retain copyright.

The CLA is short (~250 words) and standard. We'll comment on your PR with a link when needed.

### PR checklist

- [ ] PR description explains what + why (not just what).
- [ ] Tests added or modified for the change.
- [ ] `ruff check`, `ruff format`, `mypy src/` all pass.
- [ ] `pytest` passes locally.
- [ ] CHANGELOG.md updated if user-facing change.
- [ ] CLA signed (we'll prompt if needed).
- [ ] PR is focused (one logical change per PR).

### Review process

- Maintainer review within 5 business days of PR open (often faster in the first 30 days post-launch).
- Comments are technical, not stylistic preferences. We may ask for changes; please don't take them personally.
- Once approved, we'll squash-merge to main with a clean commit message.

---

## What we WILL accept

- Bug fixes (with tests).
- Documentation improvements.
- New CLI commands that fit the existing surface (`uc <verb> <object>`).
- New evaluation metrics for `uc bench`.
- Performance improvements with benchmarks.
- Test coverage improvements.
- Type-hint additions.
- Compatibility fixes for new Python / PyTorch / huggingface_hub versions.

## What we WON'T accept

- Reimplementations of patented mechanisms (correction overlay training, shared-block dispatch, etc.). The closed-source production pipeline is commercial.
- Major architectural rewrites of the CLI surface (we have a stable API contract for v0.x; major rewrites land in v1.0+).
- Features that add heavy dependencies (we keep the dependency footprint minimal).
- Changes that conflict with patented mechanism specifics — we'll flag and ask for redirection.
- Cosmetic changes (renaming variables, restructuring without behavioral change) without prior discussion.
- AI-generated PRs without human review (we accept human-augmented contributions but a PR that's clearly a copy-paste from an LLM without testing or review will be closed).

---

## Open-source vs commercial — what's open-source?

The Apache 2.0 CLI:
- `src/ultracompress_cli/` — all CLI commands, model loading, eval helpers.
- `tests/` — test suite.
- `docs/` — public documentation.
- `pyproject.toml` — packaging metadata.

The closed-source production pipeline (commercial license):
- The calibration-fitted correction overlay training loop.
- The patented composition mechanisms (shared-block dispatch, etc.).
- The production CUDA kernels (post-cofounder hire).

Contributions to the Apache 2.0 CLI are welcome under the CLA. Contributions to the closed-source pipeline are not accepted via PR — that's commercial-collaboration territory; email founder@sipsalabs.com if you have specific expertise to offer.

---

## Patent considerations

USPTO 64/049,511 (calibration-fitted correction overlay) and 64/049,517 (shared-block parameter dispatch with content-routed retrieval) plus continuations cover the production composition mechanisms. The Apache 2.0 LICENSE on the CLI does NOT grant patent licenses to these mechanisms.

If you're contributing functionality that COULD touch the patented mechanisms (e.g., adding a quantization technique that depends on per-Linear correction overlay training), let's discuss BEFORE you write the code. We'll either (a) confirm it doesn't infringe and accept the PR, or (b) redirect you to a non-infringing approach, or (c) propose commercial collaboration if the contribution is substantively in the protected space.

This is not a hostile patent posture — it's structural clarity so contributors don't waste time on PRs we can't merge.

---

## Recognition

Contributors are recognized in:
- `AUTHORS.md` (alphabetical, all contributors with merged PRs).
- Release notes (significant contributors per release).
- CHANGELOG.md feature entries (mentioned where applicable).
- Annual report (post-incorporation; 2026 for the inaugural year).

We don't currently offer paid bug bounties (pre-funding constraint). Roadmap for paid bounty post-funding.

---

## Maintainer responsiveness

The maintainer (Sip) is solo until cofounder hire (target end of Q3 2026). Response times:

- Bug reports: 1 business day for first 30 days post-launch (2026-05-05 onward); 5 business days thereafter.
- Feature requests: 5 business days for initial scoping.
- PRs: 5 business days for initial review.
- Security: same-day acknowledgment.

If you don't get a response, reply on the thread — sometimes notifications get lost. After a second silent week, escalate to founder@sipsalabs.com.

---

## What's NOT in scope for the Apache 2.0 repo

- **Customer-specific compressed model artifacts.** Those are commercial license territory.
- **Training data or evaluation prompts** beyond the public reproducibility examples.
- **Customer engagement contracts** (NDA, SOW, license templates).
- **Investor materials** (pitch memo, deck, DD pack).
- **Internal LAB-NOTEBOOK with raw competitive analysis.**

The Apache 2.0 repo is the public surface of the open-source CLI. Internal company artifacts live in private repos.

---

## Community

- **GitHub Discussions** at github.com/sipsalabs/ultracompress/discussions for technical Q&A.
- **HuggingFace Discord** (#compression-quant-distillation channel) — Sipsa is occasionally active here.
- **Twitter / X:** @SipsaLabs (company) and Sip's personal (active during launch + Wednesday morning office hours informally).
- **Quarterly newsletter:** sign up at sipsalabs.com (single field, no marketing blah).

---

## Thanks

Open-source compression infrastructure is a multi-year project. Every contribution — bug report, doc fix, feature idea, PR, conference shout-out — is appreciated. We're building this for the long term.

— Missipssa Ounnar, Founder
founder@sipsalabs.com
sipsalabs.com
