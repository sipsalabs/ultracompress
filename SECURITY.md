# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in the UltraCompress CLI, **please do not open a public issue**. Instead, email **security@sipsalabs.com** with:

- A description of the issue
- Steps to reproduce
- Affected version(s)
- Your name and (optionally) a way to credit you in the advisory

We will acknowledge receipt within **2 business days** and aim to provide a remediation plan within **7 business days** for high-severity issues.

## Scope

In scope:

- The published `ultracompress` Python package on PyPI
- The `uc` / `ultracompress` CLI commands
- Anything in this repository (source, CI workflows, packaging configuration)

Out of scope:

- Pre-compressed models hosted on Hugging Face Hub (please report to **legal@sipsalabs.com**)
- Issues in upstream dependencies (please report to those projects directly; we will track them via Dependabot / pip-audit)
- Vulnerabilities affecting only end-of-life Python versions (we support 3.10+)

## Disclosure policy

We follow **coordinated disclosure**. Once a fix is available we will:

1. Publish a patched release on PyPI
2. Open a GitHub Security Advisory
3. Credit reporters who request it

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |
| < 0.1   | no        |

We commit to security patches for the latest minor release line and the immediately previous one. Older releases will be marked end-of-life when a new minor line ships.

## Best practices for users

- Always install from PyPI (`pip install ultracompress`) — never from untrusted forks
- Verify model artifacts using the SHA-256 manifests we publish on Hugging Face Hub
- Run `pip-audit` periodically against your environment
- Pin versions in production (`ultracompress==0.1.0`)
