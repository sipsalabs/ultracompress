# LinkedIn Long-Form Post — 7 Stars + 3 New Sub-1.01x Ratios (2026-05-08)

**Audience:** ML researchers, infra engineers, AI-investor crowd.
**Tone:** confident-but-honest, technical. No hype words. No "first" / "lossless" without qualifier. No personal info per global policy.
**Brand:** Sipsa Labs.
**Link policy:** one URL — `huggingface.co/SipsaLabs`. Homepage is secondary; do not include sipsalabs.com in the body.
**IP discipline:** numbers + customer flow only. No algorithm specifics.
**Status:** DRAFT — Sip copy/pastes manually. Do not auto-post.
**Length:** ~610 words (LinkedIn long-form sweet spot).

---

## Post body (paste this block into LinkedIn)

```
Two small things compounded for Sipsa Labs this week, and the second one only matters because of the first.

The first is that github.com/sipsalabs/ultracompress crossed seven stars. Seven is a small number. It's also the first non-zero one we've ever had. A week ago Sipsa Labs was a name nobody had heard of; this week seven engineers found the repo on their own, read what we were claiming, and chose to mark it. That is the smallest credible community signal in open-source infra and it is the one I cared most about getting to.

The second thing is that we owe those seven engineers — and anyone else who arrives next — concrete numbers in return for the click. Today we measured three new compressed checkpoints end-to-end on full perplexity evaluation, and all three came in under a 1.01× ratio against their fp16 baselines:

  Qwen3-0.6B           baseline PPL 21.4792 → compressed 21.6274 → ratio 1.0069×
  OLMo-2-0425-1B       baseline PPL 12.9933 → compressed 13.0879 → ratio 1.0073×
  SmolLM2-1.7B         baseline PPL  9.1389 → compressed  9.2168 → ratio 1.0085×

The Qwen3-0.6B 1.0069× is the tightest dense-decoder ratio at 5 bits per weight that we have measured on any architecture to date. The OLMo-2 result is the first sub-1.01× ratio we have on an Allen Institute model, and the SmolLM2 result is the first on a HuggingFaceTB model. Three different model families, three different tokenizers, three independent runs — all under the same envelope.

The format under all of these is the v3 binary, which stores grid + codes + per-block absmax for each compressed Linear. That gives mathematically lossless reconstruction of W_base — the dequantized weight matrix the customer's runtime consumes is bitwise the one our trainer measured during the perplexity evaluation. Whatever drift you see in the ratio is from the quantization step itself; it is not from format ambiguity, hidden calibration metadata, or reload-path noise.

The cumulative coverage on the SipsaLabs HuggingFace organization now spans twelve architectures validated end-to-end: SmolLM2, OLMo-2, Qwen3-0.6B/1.7B/8B/14B, Mistral-7B, Llama-3.1-8B, Mixtral-8x7B and 8x22B, Phi-3.5-MoE, Mamba-2.8B, and Qwen3-235B-A22B. A Hermes-3-Llama-3.1-405B compression run is in flight tonight, sitting at 53 of 126 layers as I write this.

Two HF repos already pass the customer-side `uc verify` flow end-to-end on a stranger's GPU — Qwen3-1.7B and Mistral-7B-v0.3. Four more public artifacts are uploading right now: the four small architectures that produced the new ratios above. Each repo ships its PPL JSON file in the repo root, so anyone can read the baseline number, the compressed number, and the ratio without having to take our word for it.

What we are not claiming:

— We are not claiming "lossless compression" of model behavior. We are claiming mathematically lossless reconstruction of W_base. Those are different statements and the difference matters.
— We are not claiming this is the first 5-bpw work on these architectures. There is real prior art in 4 / 5-bit weight quantization, and the technical writeup will name it specifically.
— We are not publishing the algorithm details on a public surface yet. The provisional patents on the underlying methods are filed; the supplement is in flight.

If you want to reproduce, break, or stress-test any of these checkpoints:

    huggingface.co/SipsaLabs

The PPL JSON files are linked off each model card via `uc verify`. Issues, reproduction failures, and adversarial benchmarks all welcome. The whole point is for the result to be checkable.

If you're one of the seven who starred the repo this week — thank you. We see it.

— Sipsa Labs
```

---

## Notes for Sip (do not paste)

- The post deliberately leads with the 7-star beat because that is the social signal that earns the reader's attention long enough to get to the perplexity numbers. Without that opening, the body reads as another quant-numbers post.
- The 1.0069× on Qwen3-0.6B is the load-bearing technical claim — it is the tightest dense-decoder ratio at 5 bpw we have measured on any architecture to date. Keep that exact qualifier ("dense-decoder ratio at 5 bits per weight") in case someone produces a tighter number on an MoE or encoder model.
- "Mathematically lossless reconstruction of W_base" stays intact — do not shorten to "lossless" in any LinkedIn edits.
- The "what we are not claiming" section is the part that buys credibility with researchers who scan for hype before reading. Keep it unedited.
- One link only: `huggingface.co/SipsaLabs`. The website is intentionally absent — for this post, HF is the proof surface, not the marketing surface. The PPL JSON files reachable via `uc verify` are the falsifiability layer.
- Comment-prep: in the first comment, paste the model-card link to the Qwen3-0.6B repo specifically and the eval seed/config. That blunts "show me the eval setup" replies before they happen.
- Hermes-3-405B is mentioned because it is the right size to be the "next milestone" for any reader scanning for momentum. If the run has finished and verified by the time Sip posts this, edit the line to past tense with the final ratio. If it has failed, drop the sentence rather than over-claim.
- Three model families, three tokenizers angle in paragraph 4 is the cross-architecture generalization beat — an investor reading this should land on that sentence.
- Do not tag specific people in the body. If you want amplification, do it via DMs after the post is live.
- Post timing: weekday 7:30-9:00 AM PT for the US infra crowd. If posting on a Friday risks weekend-drift, hold to Monday morning.
```
