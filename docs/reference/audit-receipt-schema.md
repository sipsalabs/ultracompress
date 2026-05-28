# Audit receipt schema (`uc audit`)

`uc audit <pack_dir>` emits a JSON receipt that a customer can hand to their
compliance, procurement, or security team. The receipt records what was
checked, the byte-fingerprint of every file in the pack, and the host where
the check was performed. It is **not** a reconstruction proof and it is not
a license. It is an audit trail.

This document is the stable contract for that JSON. Field names and field
semantics here are versioned by `schema_version`; new fields may be added
without bumping the version, but existing fields will not be renamed or
re-typed.

---

## Top-level shape

```json
{
  "schema_version": "1.0",
  "tool": {"name": "ultracompress", "version": "0.6.23"},
  "audited_at_utc": "2026-05-28T03:14:09Z",
  "pack": { ... },
  "host": { ... },
  "checks": { ... },
  "files": [ ... ],
  "pack_fingerprint_sha256": "..."
}
```

A receipt with `checks.structure_ok == true` and `checks.no_zero_byte_files
== true` and `checks.declared_layer_count_matches == true` is a passing
audit. Any `false` is a failing audit; the receipt is still emitted so the
failure is auditable.

---

## Fields

### `schema_version` (string, required)

Semver-style. `"1.0"` is the initial schema. Major bumps change field
meaning; minor bumps add new optional fields only.

### `tool` (object, required)

| field | type | description |
|---|---|---|
| `name` | string | Always `"ultracompress"`. |
| `version` | string | Version of the public `ultracompress` package that produced the receipt. |

### `audited_at_utc` (string, required)

ISO 8601 UTC timestamp with `Z` suffix. Always UTC; the auditor's local
timezone is intentionally not recorded.

### `pack` (object, required)

What was audited.

| field | type | description |
|---|---|---|
| `path` | string | Absolute path to the pack directory at audit time. |
| `manifest` | object | The pack's `manifest.json`, parsed (forwarded as-is). |
| `declared_layer_count` | integer or null | `manifest.n_layers` if declared. |
| `observed_layer_file_count` | integer | Count of `layer_*.uc` files on disk. |
| `bytes_on_disk` | integer | Sum of file sizes (manifest + layers + aux). |

The full manifest is included verbatim so the receipt is self-contained:
a third party can re-verify the receipt without access to the pack itself.
The manifest contains no compression methodology; see
`manifest-schema.md` for its public contract.

### `host` (object, required)

Where the check ran. Designed to identify the *class* of host without any
PII.

| field | type | description |
|---|---|---|
| `os` | string | `platform.system()` (`"Linux"`, `"Windows"`, `"Darwin"`). |
| `os_release` | string | Coarse release identifier (e.g. `"11"`, `"22.04"`). PII-free. |
| `python_version` | string | `sys.version.split()[0]` (e.g. `"3.12.10"`). |
| `cpu_arch` | string | `platform.machine()` (`"AMD64"`, `"x86_64"`, `"arm64"`). |
| `cpu_count` | integer | `os.cpu_count()`. |
| `host_fingerprint` | string | SHA-256 hex of `os + os_release + arch + cpu_count`. Lets the customer prove "this is the same machine class" across two audits without recording the actual hostname, username, or MAC. |

Deliberately **NOT** recorded: hostname, username, home directory, MAC
address, IP address, GPU vendor/serial, drive serial, BIOS UUID. The
receipt is safe to hand to an external compliance team.

### `checks` (object, required)

The structural assertions. Boolean per check.

| field | type | description |
|---|---|---|
| `manifest_present` | bool | `manifest.json` exists and parses as JSON. |
| `structure_ok` | bool | All structural checks passed. |
| `declared_layer_count_matches` | bool | `manifest.n_layers == observed_layer_file_count`, or no count was declared. |
| `no_zero_byte_files` | bool | Every layer file is non-empty. |

A passing audit is `manifest_present && structure_ok && no_zero_byte_files
&& declared_layer_count_matches`, all `true`.

### `files` (array of objects, required)

Per-file fingerprint. Sorted by `name` ascending. The receipt records
every layer file plus `manifest.json` plus any `aux_file` declared by the
manifest.

| field | type | description |
|---|---|---|
| `name` | string | Basename only (e.g. `"layer_017.uc"`). No directory components. |
| `bytes` | integer | File size in bytes. |
| `sha256` | string | SHA-256 hex digest of the file contents. |

### `pack_fingerprint_sha256` (string, required)

Stable pack fingerprint: SHA-256 of `"\n".join(sorted("name:sha256"))`
across every entry in `files`. Two parties holding byte-identical packs
produce identical fingerprints. The same algorithm `uc verify` uses, so an
audit receipt and a `verify --full` printout agree.

---

## Reconstruction status

The receipt deliberately does **not** assert "this pack reconstructs to
bit-identical weights." That assertion is provided by Sipsa Labs under
engagement; the public package contains no reconstruction methodology and
therefore cannot make the claim from a customer's machine. The audit
receipt is a *structural* and *integrity* artifact: it proves the customer
holds the bytes they think they hold, on the host they think they're
holding them on.

To upgrade an audit receipt to a reconstruction proof, contact
founder@sipsalabs.com — the engagement runs the reference reconstruction
against the same `pack_fingerprint_sha256` and returns a counter-signed
certificate that incorporates the customer's receipt by fingerprint.

---

## Verification of a receipt

A receipt is self-verifying. Given a receipt and the pack directory it
references:

1. Recompute each `files[*].sha256` from disk. Must match.
2. Recompute `pack_fingerprint_sha256` from the `files` array. Must match.
3. Recompute the SHA-256 of `os|os_release|cpu_arch|cpu_count` and compare
   to `host.host_fingerprint`. Must match (if same machine class).
4. Confirm `checks.structure_ok && checks.no_zero_byte_files &&
   checks.declared_layer_count_matches`.

Steps 1-2 prove the receipt describes the pack on disk; step 3 confirms
machine class hasn't changed; step 4 confirms the audit passed.

---

## Signing (optional, opt-in)

The base receipt is unsigned: the customer's own SHA-256 of the file is
the proof anyone needs. Customers who want a counter-signed receipt
contact Sipsa Labs and submit the receipt JSON; we return a detached
signature over the canonical JSON form. The signature format is recorded
separately (`docs/reference/audit-signature.md`, future).

---

## Backward-compatibility policy

- Field renames or type changes bump `schema_version` major.
- Adding new optional fields does not bump the version.
- Removing a field bumps the version major.
- The receipt parser ignores unknown fields, so an older verifier reading
  a newer receipt sees a superset of what it understands.
