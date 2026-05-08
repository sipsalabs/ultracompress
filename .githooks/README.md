# Git Hooks for Identity + Attribution Enforcement

**Purpose**: prevent accidental re-introduction of `copilot <ai@local>` authorship or AI co-authorship trailers. Patent-defense measure.

**Location**: `.githooks/pre-commit` + `.githooks/commit-msg`

## Install in this repo (one-time)

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/commit-msg
```

## Install in other repos (ultracompress-cli, portfolio)

Copy `pre-commit` and `commit-msg` from this repo's `.githooks/` to the other repo's `.githooks/` (or `.git/hooks/`). Then same install command.

## What the hooks do

### pre-commit

- Verifies `git config user.email` = `micipsa.ounner@gmail.com`
- Verifies `git config user.name` = `Missipssa Ounnar`
- Rejects commit if either mismatches

### commit-msg

- Scans commit message for `Co-Authored-By: Claude|Copilot|GPT-|Anthropic|OpenAI|ChatGPT|Cursor|Aider|Windsurf`
- Scans for `Generated with/by Claude|Copilot`
- Rejects commit if found

## Why this matters

Patent inventorship requires a natural-person inventor. Any AI attribution in git history can be weaponized by:

- Competitors during non-provisional prosecution (challenging inventorship)
- USPTO examiners (questioning prior-art or inventorship status)
- Defendants in infringement lawsuits (challenging patent validity)

## Bypass (strongly discouraged)

If you really need to bypass:

```bash
git commit --no-verify
```

But only do this if you've verified the commit doesn't introduce AI attribution. Every bypassed commit should be logged.

## Recovery (if hooks fire on a commit you actually want)

If a commit is blocked:

```bash
# For identity mismatch:
git config user.email micipsa.ounner@gmail.com
git config user.name "Missipssa Ounnar"
git commit --amend --no-edit

# For commit message with Co-Authored-By:
git commit --amend  # Edit the message to remove the trailer
```

---

**Last updated**: 2026-04-25
