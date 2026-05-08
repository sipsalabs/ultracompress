# Outreach 2026-05-08 — Send List

**Sender:** `sipsalabs@gmail.com` (Sip pastes manually — no auto-send)
**Drafts location:** `docs/OUTREACH_2026_05_08/`
**Send order recommendation:** 01 → 02 → 03 → 04 (Lambda + CoreWeave separate sends) → 05

---

| # | Recipient (To:) | Fallback / Cc | Subject line | Draft file |
|---|---|---|---|---|
| 01 | `tri@tridao.me` | `tridao@princeton.edu`, `tri@together.ai` | First lossless 5-bit Mamba result — PPL_r 1.0119 on mamba-2.8b-hf, runnable now | `01_tri_dao.md` |
| 02 | `albert@cartesia.ai` | `hello@cartesia.ai`, `agu@andrew.cmu.edu` | Lossless 5-bit Mamba-2.8B pack — PPL_r 1.0119x, useful for Cartesia edge? | `02_albert_gu_cartesia.md` |
| 03 | `yi@reka.ai` | `yitayml@gmail.com`, LinkedIn DM `/in/yitay` | 12-arch lossless 5-bit pack including SSM — fit for Reka serving costs? | `03_yi_tay_ai21.md` |
| 04a | `partnerships@lambdalabs.com` | warm AE if any | 5x more rentable models per H100 — UltraCompress v0.5.2 just shipped, 30-second eval | `04_lambda_coreweave.md` (Lambda variant) |
| 04b | `partnerships@coreweave.com` | Brian Venturo (CTO) LinkedIn `/in/brianventuro`; Rosie Zhao on LinkedIn | 5x more rentable models per H100 — UltraCompress v0.5.2 just shipped, 30-second eval | `04_lambda_coreweave.md` (CoreWeave variant) |
| 05 | `sbir-sttr@nasa.gov` | `sbir@nasa.gov`; warm NASA HPSC project office intro via LinkedIn | Heads-up — Sipsa Labs filing Phase I imminently against ENABLE.2.S26B (HPSC), STREAM-HPSC project | `05_nasa_hpsc.md` |

---

## Pre-send checklist (Sip)

- [ ] Verify each `To:` address resolves to a real person / mailbox before clicking Send (check website footer + recent LinkedIn activity).
- [ ] For #04, send TWO separate emails — Lambda and CoreWeave. Do not BCC.
- [ ] For #05, NASA program office may take 24-72 hr to acknowledge; resend to `sbir@nasa.gov` if no read receipt by EOD Saturday.
- [ ] After each send: log the send in `docs/CUSTOMER_PIPELINE.md` with date + recipient + thread URL placeholder.

## Verification-link sanity check (for any recipient that asks)

```
pip install ultracompress
hf download SipsaLabs/qwen3-1.7b-uc-v3-bpw5 --local-dir ./pack
uc verify ./pack
```

Both `SipsaLabs/qwen3-1.7b-uc-v3-bpw5` and `SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5` were `uc verify` PASS as of 2026-05-08 EOD.

## Anti-patterns to avoid (from Sip's outreach playbook)

- Don't BCC. One email per recipient, personalized salutation.
- Don't lead with the patent stack. Patents come up at licensing-conversation stage.
- Don't oversell 405B until Hermes-3 finishes uploading (currently in flight). Frame it as "mid-compression as I write this" — accurate and honest.
- Don't reference YC for #05 (NASA). YC is irrelevant or slightly negative for federal SBIR reviewers.
- For #02 / #03, do NOT cite specific compression method internals or sub-5-bpw numbers — patents 3–7 file tomorrow; pre-filing disclosure rule applies.

---

**Total send time estimate:** ~10 min (5 emails × ~2 min each, manual paste from Gmail Drafts in `sipsalabs@gmail.com`).
