# Patent Notice

The compression methods implemented in this repository — including but not limited to:
- Per-row scalar quantization at 5 bits per weight with a learned per-module codebook
- A low-rank residual correction adapter trained via KL divergence distillation against a teacher
- The bit-identical reconstruction contract from persisted codec state plus the correction adapter
- The streaming per-layer compression pipeline that bounds peak VRAM to a single transformer layer

— are covered under United States Patent and Trademark Office provisional applications **64/049,511** and **64/049,517** (filed April 2026), with additional continuations and supplements pending.

## License grant (Apache 2.0)

This codebase is published under the Apache License 2.0. The Apache 2.0 license includes an explicit patent grant for the use, reproduction, and distribution of the **as-published source code** for any purpose, including commercial use of the open-source software itself.

## Commercial productization

If you are integrating these compression methods into a commercial product, service, or paid offering — particularly if you are reimplementing the methods in a different language or runtime, or building a derivative product whose core value depends on these methods — we ask that you contact us at **founder@sipsalabs.com** to discuss a commercial license.

We are not seeking to restrict honest engineers running this code on their own infrastructure. We are seeking to maintain a clear conversation with companies productizing the underlying invention.

## Research use

Academic research, internal benchmarking, paper reproduction, and non-commercial experimentation are explicitly welcomed. If you publish a paper that uses or compares against this work, please cite:

```bibtex
@software{sipsa_ultracompress_2026,
  author = {{Sipsa Labs, Inc.}},
  title  = {UltraCompress: Lossless 5-bit Transformer Compression},
  year   = {2026},
  url    = {https://github.com/sipsalabs/ultracompress}
}
```

## Contact

- Commercial licensing: **founder@sipsalabs.com**
- Security disclosures: **security@sipsalabs.com**
- Press: **press@sipsalabs.com**
