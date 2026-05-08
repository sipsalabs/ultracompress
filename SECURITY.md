# Security Policy — UltraCompress

**Project:** github.com/sipsalabs/ultracompress
**Maintainer:** Sipsa Labs (sole-proprietor of record: Missipssa Ounnar)
**Last updated:** 2026-05-04

---

## Reporting a vulnerability

We take security seriously. If you discover a security vulnerability in `ultracompress`, please report it responsibly via one of the following channels:

### Preferred: encrypted email

Send a PGP-encrypted message to **security@sipsalabs.com**.

PGP key fingerprint: (to be published when Sipsa Labs Inc. incorporation completes)

In the meantime, plain email to security@sipsalabs.com is acceptable for the initial contact; we will provide a temporary PGP-encrypted exchange channel for sensitive details.

### Alternative: GitHub Security Advisories

Create a private security advisory at https://github.com/sipsalabs/ultracompress/security/advisories/new. This is the GitHub-native mechanism for confidential vulnerability reporting.

### What NOT to do

- Do NOT open a public GitHub issue for security vulnerabilities. Public disclosure before a patch is available puts users at risk.
- Do NOT post to social media, mailing lists, or forums about the vulnerability before coordinated disclosure.
- Do NOT share details with third parties without our coordination.

---

## What to include in your report

To help us triage and address the issue quickly, please include:

1. **Description of the vulnerability**, including the affected component (CLI, compression pipeline, model card metadata, etc.).
2. **Reproduction steps**, ideally with a minimal proof-of-concept.
3. **Impact assessment**: who is affected, how severe, what an attacker could achieve.
4. **Affected versions**: which `ultracompress` releases are vulnerable.
5. **Suggested fix or mitigation** (optional but appreciated).
6. **Your contact information** for follow-up coordination.

---

## Response timeline

| Phase | SLA |
|---|---|
| Acknowledgment of report | within 5 business days |
| Initial triage / severity assessment | within 10 business days |
| Patch development for critical vulnerabilities | within 30 days |
| Patch development for non-critical vulnerabilities | within 90 days |
| Coordinated public disclosure | per agreement with reporter, typically 7-30 days after patch release |

---

## Severity classification

We use the following severity levels:

### Critical
- Remote code execution via `ultracompress` CLI processing untrusted input.
- Authentication bypass affecting customer artifact access.
- Data exfiltration of customer model weights or evaluation data.
- Patch within 30 days.

### High
- Privilege escalation in customer environment via Sipsa-delivered scripts.
- Cryptographic weakness in artifact integrity verification.
- Patch within 60 days.

### Medium
- Information disclosure via error messages or logs.
- Denial of service via malformed input.
- Patch within 90 days.

### Low
- Hardening opportunities, defense-in-depth improvements.
- Dependency CVEs that don't affect `ultracompress` directly but should be updated.
- Patch in next regular release.

---

## Coordinated disclosure

For critical and high severity vulnerabilities:
1. Reporter and maintainer agree on disclosure timeline (typically 30-90 days).
2. Maintainer develops and tests the patch.
3. Maintainer notifies major downstream users (commercial license customers) under embargo.
4. Patch released as a new `ultracompress` version with an advisory.
5. Public advisory at github.com/sipsalabs/ultracompress/security/advisories.

For medium and low severity vulnerabilities, we may release the patch with a CHANGELOG entry without a separate advisory.

---

## Recognition

We do not currently offer a paid bug bounty program (pre-funding constraint). We DO offer:

- **Public acknowledgment** in the security advisory (with reporter's permission).
- **Reference letter** if the reporter is a security researcher seeking professional validation.
- **Early access to future paid bug bounty program** when one is established post-funding.

If you are a security researcher who needs paid compensation for your work, please indicate that in your report so we can discuss arrangements.

---

## Scope

In scope:
- The `ultracompress` Python package (PyPI).
- The `uc` CLI commands (`pull`, `list`, `info`, `bench`, `demo`, `version`).
- The Python source code in `github.com/sipsalabs/ultracompress`.
- The HuggingFace Hub artifacts under `huggingface.co/SipsaLabs/`.
- The closed-source production pipeline (under commercial license).

Out of scope:
- Vulnerabilities in third-party dependencies (PyTorch, huggingface_hub, etc.) — please report those upstream.
- Vulnerabilities in customer's downstream USE of compressed models (we cannot remediate customer-side deployment issues).
- Social engineering of Sipsa Labs employees / contractors (we have only one — Missipssa Ounnar — at this time, but we still ask reporters not to use social engineering).
- Physical security of Sipsa Labs hardware (home office).
- Denial-of-service attacks against sipsalabs.com web infrastructure (handled separately by hosting provider).

---

## Security best practices for users

When using `ultracompress`:

1. **Verify package integrity.** Compare the SHA256 of installed `ultracompress` against the published hash on PyPI. Use `pip install ultracompress --require-hashes` for hash-verified installs.
2. **Verify model artifact integrity.** When pulling compressed models from HuggingFace, the `uc pull` command verifies hashes by default. Do not bypass this verification.
3. **Sandbox compressed model loading.** Treat compressed model artifacts as you would treat any third-party data file: load them in a sandboxed environment first if you have not verified their provenance.
4. **Update regularly.** `pip install --upgrade ultracompress` to get the latest security patches.
5. **Review the source.** The Apache 2.0 CLI is open-source. Review the source if you are deploying in a security-sensitive environment.
6. **Report issues.** If you observe unexpected behavior that could be a security issue, report it. False positives are welcome; we'd rather investigate a non-issue than miss a real one.

---

## Known issues

### As of 2026-05-04 (v0.4.0)

- **CLI version banner stale at 0.1.3** despite package version 0.4.0. Cosmetic only. Patched in v0.4.1 (in progress).
- **`uc list` returns no models** due to HF_ORG case mismatch. Fixed in v0.4.1.
- **`uc pull` crashes on Windows** after successful download due to Rich Braille char + cp1252 codec mismatch. Fixed in v0.4.1.

These are documented as functional bugs in `docs/PRELAUNCH_BUGS_v0_4_0_FIXES.md` (private) and will be addressed in the v0.4.1 patch release.

### Historical

(None yet — pre-launch.)

---

## Compliance and certifications

Sipsa Labs does NOT currently hold:
- SOC 2 Type 1 or Type 2.
- ISO 27001.
- FedRAMP authorization.
- HIPAA Business Associate Agreement readiness.
- ITAR registration.

Roadmap for these is documented in `docs/SECURITY_FAQ_AND_POSTURE_2026_05_04.md` (available on customer request).

The structural mitigation: `ultracompress` is a code-delivery model. Compressed models are produced inside the customer's environment OR Sipsa-controlled hardware that doesn't process customer-protected data beyond the engagement window. The certification surface is minimized by the architecture, not by certifications.

---

## Pre-incorporation note

Sipsa Labs Inc. is in the process of incorporation (Delaware C-corp, post-YC June 5 OR upon first revenue). Until incorporation completes:
- Security commitments are made by sole-proprietor-of-record Missipssa Ounnar.
- Vulnerability reports are tracked in private issue tracker; will migrate to corporate issue tracker post-incorporation.
- The maintainer email security@sipsalabs.com remains the canonical contact through the entity transition.

---

## Contact

- **Vulnerability disclosure:** security@sipsalabs.com
- **General security questions:** founder@sipsalabs.com
- **Compliance / customer security review:** founder@sipsalabs.com
- **Press / media security inquiries:** press@sipsalabs.com

---

*This security policy follows the OWASP Vulnerability Disclosure Cheat Sheet recommendations. Updates to this policy are tracked in CHANGELOG.md.*
