# Release process

UltraCompress follows [Semantic Versioning](https://semver.org/) and uses tagged GitHub releases for both the GitHub release page and the PyPI package upload (via Trusted Publishing).

This page documents the release process for maintainers.

## Release cadence

- **Patch** (0.1.x): on demand for bugfixes; typically every 1-4 weeks during active development
- **Minor** (0.x.0): roughly every 2-3 months as new features land
- **Major** (x.0.0): only on a breaking change; typically every 12+ months

## Pre-release checklist

Before tagging a release:

- [ ] All CI checks green on `main`
- [ ] CHANGELOG.md has a populated section for the new version
- [ ] Version bumped in `src/ultracompress_cli/__init__.py` (`__version__`)
- [ ] Version bumped in `pyproject.toml`
- [ ] No secrets or credentials in the diff (gitleaks pre-commit catches this)
- [ ] Manual smoke test: `pip install -e . && uc --version && uc list && uc info <local-artifact>`
- [ ] Documentation site builds: `mkdocs build --strict`

## Cutting a release

```bash
# 1. Make sure you're on main and up-to-date
git checkout main
git pull origin main

# 2. Verify CHANGELOG has an entry for the new version
head -30 CHANGELOG.md

# 3. Tag the release (annotated tag with message)
git tag -a v0.1.1 -m "v0.1.1 — bugfix release for issue #N"
git push origin v0.1.1
```

## What happens automatically on the tag push

The `release.yml` GitHub Actions workflow:

1. Extracts the changelog section for the new version
2. Creates a GitHub Release page with that content as the release notes

The `ci.yml` workflow's publish job:

1. Triggers on the tag (since `if: startsWith(github.ref, 'refs/tags/v')`)
2. Builds the wheel + sdist
3. Publishes to PyPI via Trusted Publishing (no token required)

Both should complete within a few minutes. Monitor the Actions tab to verify.

## Verification after release

- [ ] Visit https://pypi.org/project/ultracompress/ — confirm new version is listed
- [ ] Visit https://github.com/sipsalabs/ultracompress/releases — confirm Release page is rendered
- [ ] Run `pip install --upgrade ultracompress` in a fresh venv; confirm version matches
- [ ] Run `uc --version` — confirm version matches
- [ ] Smoke test the published package end-to-end (`uc list`, `uc pull`, `uc info`)

## Rollback

If something is wrong with a published release:

1. **Don't republish to the same version** — PyPI doesn't allow rewriting an already-published version.
2. **Yank the broken version** on PyPI (Manage Project → Releases → yank). This prevents new installs from picking it up.
3. **Cut a new patch release** with the fix.

For the GitHub release page:
- If the release notes are wrong, edit them on the GitHub UI
- If the tag is wrong, you can delete the tag (locally + remotely) and recreate it, BUT this only works if you haven't published to PyPI yet

```bash
# delete tag locally
git tag -d v0.1.1
# delete tag remotely
git push --delete origin v0.1.1
# (then re-tag and re-push)
```

## Hotfix protocol

For a critical bug (security vulnerability, broken core functionality):

1. **Don't merge new features** into `main` until the hotfix is shipped.
2. Branch from the latest stable tag, not from `main`:
   ```bash
   git checkout v0.1.0
   git checkout -b hotfix/security-fix
   ```
3. Make the minimal fix.
4. Bump the patch version (v0.1.0 → v0.1.1).
5. Tag from the hotfix branch (NOT from `main`).
6. After the tag is published, fast-forward `main` from the hotfix branch.

## Pre-release versions (alpha, beta, rc)

For pre-release versions, use the `aN`, `bN`, `rcN` suffix:

```
v0.2.0a1   # alpha 1
v0.2.0b1   # beta 1
v0.2.0rc1  # release candidate 1
v0.2.0     # final
```

Tag-pattern in `release.yml` includes a `prerelease: contains(tag, '-')` flag (currently using `-` for pre-release detection; pre-release tags should use `v0.2.0-alpha1` style).

## Things that should never happen

- Releasing without a corresponding CHANGELOG entry
- Releasing with secrets in the diff
- Force-pushing to `main` after a release
- Releasing a version that's behind PyPI's published version
- Releasing on a Friday afternoon (any post-release issues land in your weekend)

## Post-release follow-up

After every release:

- [ ] Tweet / post the release notes (concise; link to changelog)
- [ ] Update docs.sipsalabs.com if any docs need to follow
- [ ] Update relevant social profiles' bio (when version is on the bio)
- [ ] Notify active design partners + customers via their dedicated Slack/Discord channels
- [ ] Open the next milestone on GitHub for the upcoming release
