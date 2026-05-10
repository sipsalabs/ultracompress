# Notice — Licensing Change for ultracompress v0.6.0+

**Date:** 2026-05-10
**TL;DR:** v0.5.x stays Apache 2.0 forever. v0.6+ ships under BUSL 1.1 with a generous Additional Use Grant. We followed the Sentry pattern, not the MongoDB pattern.

---

## What changed

- **v0.5.x and earlier** remain under [Apache License 2.0](./LICENSE.apache). This is permanent and cannot be revoked. The `legacy/0.5.x` branch is the canonical pin.
- **v0.6.0 and later** are licensed under the [Business Source License 1.1](./LICENSE), with an Additional Use Grant that keeps the Licensed Work freely usable for:
  - Non-production use
  - Internal research
  - Individual use
  - Any commercial entity with annual gross revenue under $1,000,000 USD
- **Four years after each v0.6+ release**, that release auto-converts to Apache 2.0 (the BUSL Change License clause).

If you fall in any of the categories above, nothing changes for you. Keep building.

If your company is over $1M ARR and you're running v0.6+ in commercial production: `founder@sipsalabs.com`. We're not trying to nickel-and-dime anyone — we're trying to keep an independent compression research lab funded.

## Why

Sipsa Labs is one solo founder, building a frontier-scale compression codec. The 1.0066x PPL ratio on a 405-billion-parameter transformer reproducible on a single 32 GB consumer GPU took real compute to produce. The patent corpus and the reproducibility infrastructure took real time to build.

Apache 2.0 on the entire surface meant we were giving away the diamonds along with the recipe. That's not a sustainable model for a research lab funded by a single person.

The BUSL transition lets us keep the recipe public (and the smaller verifier artifacts free, on Hugging Face), while charging the companies that build commercial products on top of the codec.

## Who is unaffected

- Researchers, students, and academics → unaffected (fits Additional Use Grant)
- Individual hobbyists → unaffected
- Companies under $1M ARR → unaffected
- Anyone who downloaded v0.5.x before this change → unaffected forever

## Who pays

- Companies above $1M ARR running v0.6+ in commercial production → commercial license required
- Hosted/managed services that compete with Sipsa Labs' commercial offerings → talk to us

Pricing tiers will be published at `sipsalabs.com/pricing` when the v0.6.0 release ships. Until then, contact `founder@sipsalabs.com` for commercial inquiries.

## The four-year clock

Each v0.6+ release publishes with a Change Date four years out. After that date, that exact release auto-converts to Apache 2.0. So a Q3 2026 release becomes Apache 2.0 in Q3 2030. This is a feature, not a footnote — it ensures the codebase eventually returns to fully open source.

## References

- Business Source License 1.1 canonical text: https://mariadb.com/bsl11/
- The transition pattern we followed (Sentry, 2019): https://blog.sentry.io/lets-talk-about-open-source/
- HashiCorp's BUSL adoption (2023): https://www.hashicorp.com/en/blog/hashicorp-adopts-business-source-license

## Questions

- Commercial license: `founder@sipsalabs.com`
- Security: `security@sipsalabs.com`
- General: `founder@sipsalabs.com`
