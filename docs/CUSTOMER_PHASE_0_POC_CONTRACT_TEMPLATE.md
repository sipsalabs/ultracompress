# Phase 0 Proof-of-Concept Engagement Agreement — Template

> **TEMPLATE ONLY — review with counsel before customer execution.**
> This document is a fillable Markdown working copy. Replace every bracketed `[PLACEHOLDER]` before sending. Render to PDF (e.g., Pandoc, Typora, or paste into Google Docs and download as PDF) and counter-sign once Customer signs. This is a starting point intended to be reviewable by Customer's counsel; redlines on most non-IP, non-refund clauses are expected and acceptable.

---

**Document title:** Phase 0 Proof-of-Concept Engagement Agreement
**Effective date:** [EFFECTIVE_DATE]
**Reference:** Sipsa Labs SOW-P0-[YYYYMM]-[NNN]

---

## 1. Parties

This Phase 0 Proof-of-Concept Engagement Agreement (this "**Agreement**") is entered into as of the Effective Date set forth above by and between:

**Sipsa Labs, Inc.** ("**Sipsa**"), a Delaware corporation with EIN [PENDING — to be supplied upon issuance by IRS], with principal contact Missipssa Ounnar, Founder, founder@sipsalabs.com, and notice address [SIPSA_NOTICE_ADDRESS]; and

**[CUSTOMER LEGAL ENTITY NAME]** ("**Customer**"), a [STATE_OR_COUNTRY_OF_FORMATION] [ENTITY_TYPE — e.g., corporation / LLC], with principal contact [CUSTOMER_CONTACT_NAME], [CUSTOMER_CONTACT_TITLE], [CUSTOMER_CONTACT_EMAIL], and notice address [CUSTOMER_NOTICE_ADDRESS].

Sipsa and Customer are each a "**Party**" and collectively the "**Parties**."

---

## 2. Engagement scope (Phase 0)

### 2.1 Work performed

Sipsa shall apply its production v3 lossless-format compression pipeline to one (1) Customer-specified transformer-architecture large language model (the "**Model**") and produce a compressed artifact targeted at five (5) bits-per-weight ("**BPW**") on the Model's quantizable Linear layers, in accordance with the v3 binary format documented in `pip install ultracompress` and `github.com/sipsalabs/ultracompress`.

For the avoidance of doubt, the v3 lossless format refers to the structural pack format (deterministic, hash-verifiable, reconstructable via `parse_uc_layer_v3`); compression below the bf16 teacher is achieved at the quantization layer and is not bit-exact to the teacher. "Lossless" in this Section refers to the pack format's deterministic reconstruction, not to bit-exactness against the teacher checkpoint.

### 2.2 Deliverables

On or before the delivery deadline set forth in Section 2.3, Sipsa shall deliver the following to Customer's designated technical contact via Customer's designated secure-transfer destination:

(a) the packed `.uc` artifact (per-layer files plus `manifest.json`) for the Model, hereinafter the "**Compressed Artifact**";

(b) a `uc verify` PASS report demonstrating that the Compressed Artifact is structurally sound, all layer files are present, SHA256 hashes are stable, and `parse_uc_layer_v3` reconstructs every quantized Linear without NaN or Inf;

(c) a benchmark JSON file containing the perplexity ("**PPL**") ratio of the Compressed Artifact against the bf16 teacher Model on the evaluation set specified in Section 4, plus throughput and peak-VRAM measurements; and

(d) a one-page deployment guide describing how to load the Compressed Artifact into Customer's inference stack and any Model-specific notes (per-layer compression profile, calibration parameters, reproducibility receipt).

### 2.3 Timeline

The engagement runs five (5) business days commencing on the first business day after Sipsa's receipt of (i) Customer's executed copy of this Agreement, (ii) the kickoff payment described in Section 3, and (iii) all Customer-side prerequisites described in Section 2.5 (the "**Engagement Start**"). Deliverables shall be transmitted to Customer no later than the close of business on the fifth business day after the Engagement Start (the "**Delivery Deadline**").

Reference architecture matrix and current verified-PASS scope: `docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md` in the Sipsa public repository, which documents the architectures Sipsa has compressed end-to-end with `uc verify` PASS as of the document date.

### 2.4 Sipsa-side delays

If Sipsa fails to deliver by the Delivery Deadline due solely to delays attributable to Sipsa (and not to Customer-side delays under Section 2.5 or Force Majeure under Section 9.2), Sipsa shall not charge Customer any additional fee for the extended timeline. Customer's exclusive remedies for Sipsa-side delivery delay are set forth in Sections 3.4, 9.1, and 9.3.

### 2.5 Customer-side prerequisites and delays

Customer shall provide the following on or before the Engagement Start:

(a) identification of the Model (HuggingFace `model_id` for a publicly available checkpoint, or, for a Customer-fine-tuned variant, the checkpoint files transferred to Sipsa's secure-transfer destination);

(b) Customer's evaluation set, or written election of the FineWeb-edu held-out tail default per Section 4.1;

(c) target deployment hardware identification (per Appendix A);

(d) calibration-data preference (per Appendix A); and

(e) any acceptance-threshold variation from the Section 4.2 default.

If Customer fails to provide any item required under this Section 2.5 by the Engagement Start, the Delivery Deadline extends day-for-day with no additional fee charged by Sipsa. Such delay shall not be deemed a breach by Sipsa of Section 2.3.

---

## 3. Pricing and payment

### 3.1 Fee

The total engagement fee is **five thousand U.S. dollars ($5,000 USD)** (the "**Fee**"), inclusive of all Sipsa labor, compute, software, and standard delivery. Fee is exclusive of any Customer-side hardware, Customer-side cloud egress, and any taxes that may be imposed on Customer.

### 3.2 Payment schedule

The Fee is payable as follows:

(a) **fifty percent (50%) — $2,500 USD** ("**Kickoff Payment**") due upon execution of this Agreement, payable within ten (10) business days of Customer's receipt of Sipsa's invoice; and

(b) **fifty percent (50%) — $2,500 USD** ("**Delivery Payment**") due upon Sipsa's transmission of the deliverables enumerated in Section 2.2, payable within thirty (30) days of Customer's receipt of Sipsa's invoice (Net 30).

### 3.3 Invoicing

Sipsa shall invoice Customer at the addresses specified for billing under Section 11. Invoices shall reference SOW number, line items, and Sipsa's Delaware EIN once issued.

### 3.4 Sipsa-side delay credit

If Sipsa fails to deliver by the Delivery Deadline for reasons solely attributable to Sipsa (excluding Customer-side delays under Section 2.5 and Force Majeure under Section 9.2) and Sipsa does not deliver within five (5) additional business days, Customer may at its option (i) extend the Delivery Deadline by mutual written agreement at no additional charge, or (ii) terminate under Section 9 and receive a pro-rata refund of any portion of the Fee paid for which deliverables have not been transmitted.

### 3.5 Late payment

Undisputed amounts not paid by the due date accrue interest at the lesser of one percent (1.0%) per month or the maximum rate permitted by applicable law, compounded monthly until paid.

---

## 4. Acceptance criteria

### 4.1 Evaluation set

Acceptance is measured against the evaluation set specified by Customer under Section 2.5(b). If Customer does not specify an evaluation set, the default is the FineWeb-edu held-out tail (30 prompts, sequence length 1024, real tokenization, computed against the bf16 teacher Model on Sipsa-controlled hardware), consistent with Sipsa's published verification methodology in `docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md`.

### 4.2 Acceptance gates

The Compressed Artifact is "**Accepted**" if and only if all of the following gates are satisfied:

(a) the `uc verify` report delivered under Section 2.2(b) reports PASS for the Compressed Artifact;

(b) the PPL ratio (compressed PPL ÷ bf16 teacher PPL) measured on the evaluation set under Section 4.1 is no greater than **1.015** (i.e., within one and one-half percent (1.5%) of baseline), or such other threshold as Customer specifies in writing on or before the Engagement Start under Section 2.5(e); and

(c) the Compressed Artifact, the `uc verify` report, the benchmark JSON, and the deployment guide have all been transmitted under Section 2.2.

### 4.3 Acceptance window

Customer shall notify Sipsa in writing of acceptance or rejection within five (5) business days after Sipsa's transmission of the deliverables (the "**Acceptance Window**"). A rejection notice shall identify the specific gate(s) under Section 4.2 that Customer asserts have not been satisfied, with reasonably particular evidence (such as Customer's reproduction logs).

### 4.4 Deemed acceptance

If Customer does not deliver a written rejection notice satisfying Section 4.3 before the close of the Acceptance Window, the deliverables are deemed Accepted. Deemed acceptance triggers the Delivery Payment under Section 3.2(b).

### 4.5 Cure period

If Customer delivers a timely rejection notice under Section 4.3, Sipsa has ten (10) business days to remediate the identified gate failure(s) and re-deliver. The Acceptance Window restarts upon re-delivery. If Sipsa does not cure within the cure period, Customer's exclusive remedies are set forth in Section 9.

---

## 5. Intellectual property

### 5.1 Customer materials

Customer retains all right, title, and interest in and to the Model (as input to this engagement), Customer's calibration data (if any), Customer's evaluation set, and any Customer-supplied materials (collectively, "**Customer Materials**"). Nothing in this Agreement transfers ownership of any Customer Materials to Sipsa.

### 5.2 License to Sipsa for performance

Customer grants Sipsa a non-exclusive, royalty-free, worldwide license to use, copy, transmit, store, and operate on the Customer Materials solely as necessary to perform Sipsa's obligations under this Agreement. This license terminates on the later of (i) Customer's acceptance under Section 4 and (ii) Sipsa's destruction of Customer Materials under Section 6.5.

### 5.3 Sipsa retained rights

Sipsa retains all right, title, and interest in and to (a) the compression algorithm and any derivative methods; (b) the v3 binary format specification and the `parse_uc_layer_v3` reconstruction logic; (c) the trainer code, calibration code, and pipeline code; (d) any patents, patent applications, trade secrets, and know-how owned or applied for by Sipsa, including without limitation U.S. provisional patent applications 64/049,511 and 64/049,517 and any continuations, divisionals, supplements, or non-provisional applications based thereon; and (e) all improvements, modifications, and derivative works of any of the foregoing made by Sipsa before, during, or after the Engagement.

### 5.4 License to Customer for Compressed Artifact

Subject to Customer's payment of the Fee in full and Customer's continuing compliance with this Agreement, Sipsa grants Customer a perpetual, worldwide, non-transferable, non-sublicensable, non-exclusive license to use the Compressed Artifact for Customer's own internal inference and internal evaluation. Production or external use of the Compressed Artifact beyond Customer's internal scope requires a separate Phase 1 license under Section 7.

### 5.5 Customer restrictions

Customer shall NOT, directly or indirectly:

(a) redistribute, sublicense, sell, lease, lend, publish, or otherwise make available the Compressed Artifact to any third party;

(b) reverse-engineer, decompile, disassemble, or otherwise attempt to derive the v3 binary format specification, the compression algorithm, the calibration parameters, or any related Sipsa trade secrets from the Compressed Artifact, the deployment guide, or the benchmark JSON;

(c) remove, obscure, or alter any proprietary notices, file headers, or `manifest.json` provenance fields contained in the Compressed Artifact; or

(d) use the Compressed Artifact in violation of applicable law, including export-control laws and regulations.

The restrictions in this Section 5.5 survive termination and expire only upon Sipsa's written release.

### 5.6 Aggregate publication

Sipsa may publish anonymized aggregate metrics (e.g., "Sipsa has compressed N transformer architectures with PPL ratio ≤ X across the verification matrix") that do not identify Customer, the Model, or Customer's specific measurement results. Sipsa shall NOT publish (i) Customer's name or identity, (ii) the specific Model identification, or (iii) Customer's specific measurement results, without Customer's prior written consent.

**Customer opt-out (initial here to opt out): _______** — If initialed, Sipsa shall not include this engagement (even in anonymized aggregate) in any external Sipsa publication, investor materials, or public statement.

---

## 6. Confidentiality

### 6.1 Mutual confidentiality

Each Party (as "**Recipient**") shall hold in confidence all non-public information disclosed by the other Party (as "**Discloser**") that is identified as confidential at disclosure or that a reasonable person would understand to be confidential under the circumstances ("**Confidential Information**").

### 6.2 Customer Confidential Information

Customer's Confidential Information includes, without limitation, the Model and its weights, Customer's calibration data, Customer's evaluation set, Customer's identity as a Sipsa customer, the existence and substance of this Agreement, and Customer's deployment plans.

### 6.3 Sipsa Confidential Information

Sipsa's Confidential Information includes, without limitation, the v3 binary format specification, the compression algorithm internals, the trainer and calibration code, the unpublished portions of the patent applications referenced in Section 5.3(d), Sipsa's pricing and customer pipeline, and Sipsa's unannounced product plans.

### 6.4 Permitted use and disclosure

Recipient shall (a) use Discloser's Confidential Information solely for performance under this Agreement and the Phase 1 evaluation contemplated by Section 7; (b) protect Discloser's Confidential Information with at least the same degree of care Recipient uses to protect its own confidential information of similar nature, and in no event less than reasonable care; and (c) limit access to Recipient's employees, contractors, and advisors who have a need to know for the foregoing purposes and are bound by written confidentiality obligations at least as protective as those in this Section 6.

### 6.5 Exclusions

Confidential Information does not include information that (a) is or becomes publicly available without breach by Recipient; (b) was rightfully in Recipient's possession before disclosure, as evidenced by written records; (c) is rightfully received from a third party without restriction; (d) is independently developed by Recipient without use of or reference to Discloser's Confidential Information, as evidenced by written records; or (e) is required to be disclosed by law or court order, provided Recipient gives Discloser prompt written notice (when legally permissible) and reasonable opportunity to object.

### 6.6 Return or destruction

Upon Discloser's written request following termination or completion of this Agreement, Recipient shall return or destroy all copies of Discloser's Confidential Information in Recipient's possession or control and certify completion in writing within thirty (30) days. Sipsa shall destroy Customer Materials within thirty (30) days after the later of Acceptance under Section 4 or termination under Section 9, except that Sipsa may retain (i) one archival copy of the deliverables under Section 2.2 for Sipsa's audit and reproducibility records, and (ii) routine system backups containing Customer Materials that are deleted in the ordinary course of backup retention.

### 6.7 Term

The confidentiality obligations under this Section 6 survive for a period of three (3) years from the date of disclosure of the relevant Confidential Information.

### 6.8 Equitable relief

Each Party acknowledges that unauthorized disclosure or use of the other Party's Confidential Information may cause irreparable harm for which monetary damages would be inadequate. Each Party shall be entitled to seek injunctive and other equitable relief, in addition to any other remedies available at law or in equity, without the necessity of posting a bond.

### 6.9 Standalone NDA

If the Parties have executed a separate mutual non-disclosure agreement covering the subject matter of this Agreement, the more protective of (i) that NDA and (ii) this Section 6 shall govern, and the two instruments shall be read consistently to the extent possible.

---

## 7. Path to Phase 1 (optional, no commitment)

### 7.1 Availability

If the deliverables are Accepted under Section 4 and Customer wishes to expand the engagement, Sipsa offers a Phase 1 engagement at a fee of **fifty thousand U.S. dollars ($50,000 USD)** ("**Phase 1**"), subject to a separate written statement of work executed by both Parties.

### 7.2 Phase 1 scope (informational, not binding)

Phase 1 expands the engagement to:

(a) up to three (3) Customer-specified transformer model architectures (additional architectures priced separately);

(b) Customer-specific calibration data integration, including ingestion of Customer's domain-specific calibration corpus and per-layer calibration tuning;

(c) deployment integration support for Customer's chosen inference framework (e.g., vLLM, TensorRT-LLM, llama.cpp, custom); and

(d) thirty (30) days of post-deployment support, including bug-fix turnaround, deployment-issue diagnosis, and reproducibility assistance.

### 7.3 No commitment

Neither Party is obligated by this Section 7 to enter into Phase 1. Phase 0 is a self-contained measurement engagement; Phase 1 exists at Customer's option only and is contingent on a separately negotiated SOW. Pricing in Section 7.1 is held open for sixty (60) days after Acceptance and may be renegotiated thereafter.

---

## 8. Limitation of liability and warranties

### 8.1 Cap

EXCEPT FOR (i) BREACH OF SECTION 5 (INTELLECTUAL PROPERTY), (ii) BREACH OF SECTION 6 (CONFIDENTIALITY), AND (iii) A PARTY'S INDEMNIFICATION OBLIGATIONS UNDER SECTION 8.4, EACH PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATING TO THIS AGREEMENT, WHETHER IN CONTRACT, TORT (INCLUDING NEGLIGENCE), OR OTHERWISE, IS CAPPED AT THE TOTAL FEE ACTUALLY PAID BY CUSTOMER TO SIPSA UNDER THIS AGREEMENT (I.E., UP TO $5,000 USD).

### 8.2 No consequential damages

NEITHER PARTY SHALL BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES, INCLUDING WITHOUT LIMITATION LOST PROFITS, LOST REVENUE, LOST DATA, OR BUSINESS INTERRUPTION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

### 8.3 "AS IS" warranty disclaimer

THE COMPRESSED ARTIFACT, BENCHMARK JSON, DEPLOYMENT GUIDE, AND ALL OTHER DELIVERABLES ARE PROVIDED "**AS IS**" AND "**AS AVAILABLE**." EXCEPT FOR THE PPL RATIO ACCEPTANCE GATE EXPRESSLY SET FORTH IN SECTION 4.2(b), SIPSA MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ACCURACY ON ANY DOWNSTREAM TASK, OR PERFORMANCE ON ANY EVALUATION SUITE OTHER THAN THE ONE DESIGNATED UNDER SECTION 4.1. SIPSA DOES NOT WARRANT THAT THE COMPRESSED ARTIFACT IS FIT FOR PRODUCTION DEPLOYMENT IN CUSTOMER'S SPECIFIC ENVIRONMENT, AND CUSTOMER ASSUMES SOLE RESPONSIBILITY FOR ITS USE AND DEPLOYMENT DECISIONS.

### 8.4 Mutual indemnification (IP)

(a) Sipsa shall indemnify, defend, and hold Customer harmless from any third-party claim alleging that Customer's authorized use of the Compressed Artifact under Section 5.4 (other than as combined with Customer Materials or third-party materials not supplied by Sipsa) infringes a U.S. patent, copyright, or trade secret of such third party. Sipsa's indemnification under this Section 8.4(a) is capped at the Fee paid by Customer.

(b) Customer shall indemnify, defend, and hold Sipsa harmless from any third-party claim arising out of or relating to (i) Customer's Model or other Customer Materials infringing or misappropriating any third-party intellectual property right, or (ii) Customer's use of the Compressed Artifact in violation of Sections 5.4 or 5.5.

### 8.5 Allocation

The Parties acknowledge that the Fee reflects the allocation of risk set forth in this Section 8 and that the limitations and disclaimers in this Section 8 are an essential basis of the bargain.

---

## 9. Term and termination

### 9.1 Term

This Agreement commences on the Effective Date and continues until the later of (i) Acceptance under Section 4, (ii) full payment of the Fee under Section 3, and (iii) Sipsa's destruction of Customer Materials under Section 6.6. Sections 5 (IP), 6 (Confidentiality), 8 (Liability), 10 (Governing law), and 11 (General) survive termination.

### 9.2 Termination for cause

Either Party may terminate this Agreement for cause upon five (5) business days' prior written notice if the other Party materially breaches this Agreement and fails to cure the breach within the notice period. Termination for cause does not waive any other remedies the non-breaching Party may have.

### 9.3 Effect of termination — pro-rata refund

If this Agreement is terminated by Customer for cause under Section 9.2 due to Sipsa's failure to deliver, or terminated by either Party because of Sipsa's inability to perform (including for the architectural-incompatibility reason described in Section 9.5), Sipsa shall refund any portion of the Fee that has been paid for which Sipsa has not delivered corresponding work product. Refund is calculated pro-rata based on the percentage of deliverables under Section 2.2 that have been transmitted to Customer at the time of termination.

### 9.4 Force majeure

Neither Party shall be liable for any delay or failure to perform under this Agreement (other than payment obligations) caused by events beyond its reasonable control, including without limitation natural disasters, governmental action, regulatory change, war, terrorism, civil unrest, labor disputes, internet or cloud-provider outages, or pandemic-related disruption. The affected Party shall promptly notify the other Party and use commercially reasonable efforts to resume performance.

### 9.5 Honest-walk option

If, during the engagement, Sipsa determines in good faith that the compression mechanism cannot transfer to the Model for fundamental architectural reasons (e.g., the Model uses non-standard activations, operator types, or layer structures that the v3 pipeline does not support), Sipsa shall notify Customer in writing within forty-eight (48) hours of discovery, deliver a written diagnostic explaining the failure mode in lieu of the standard deliverables, and refund Customer pro-rata under Section 9.3. This option exists in lieu of forcing a deliverable that does not meet acceptance gates.

---

## 10. Governing law, jurisdiction, jury waiver

### 10.1 Governing law

This Agreement is governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict-of-laws principles.

### 10.2 Exclusive jurisdiction

The Parties consent to the exclusive jurisdiction of the Court of Chancery of the State of Delaware (or, where the Court of Chancery does not have subject-matter jurisdiction, the federal or state courts located in New Castle County, Delaware) for any dispute arising out of or relating to this Agreement.

### 10.3 Jury trial waiver

EACH PARTY KNOWINGLY, VOLUNTARILY, AND INTENTIONALLY WAIVES ANY RIGHT TO A TRIAL BY JURY IN ANY ACTION ARISING OUT OF OR RELATING TO THIS AGREEMENT.

### 10.4 Export controls

Each Party shall comply with all applicable U.S. and foreign export-control and sanctions laws and regulations in its handling of the other Party's materials, Confidential Information, and the Compressed Artifact. Sipsa Labs is not, as of the Effective Date, an ITAR-registered entity; if Customer's Model or evaluation data is subject to ITAR or other export-control regimes that require an ITAR-registered vendor, Customer shall notify Sipsa in writing before the Engagement Start, and the Parties shall in good faith consider whether to scope an alternative no-model-sharing engagement structure.

---

## 11. General

### 11.1 Notices

All notices required under this Agreement shall be in writing and shall be delivered to the principal contact addresses set forth in Section 1, by email with confirmation of receipt or by overnight courier with tracking. Either Party may update its notice address by providing written notice to the other Party.

### 11.2 Entire agreement

This Agreement, together with any standalone NDA referenced in Section 6.9 and any executed Phase 1 SOW under Section 7, constitutes the entire agreement between the Parties with respect to its subject matter and supersedes all prior or contemporaneous agreements, communications, and understandings, whether oral or written. Any inconsistency between this Agreement and a Customer purchase order or similar instrument shall be resolved in favor of this Agreement, and the Parties expressly disclaim any pre-printed terms on a Customer purchase order.

### 11.3 Amendments

This Agreement may be amended only by a written instrument signed by both Parties.

### 11.4 Assignment

Neither Party may assign this Agreement without the other Party's prior written consent, except that either Party may assign this Agreement, in connection with a merger, acquisition, reorganization, or sale of all or substantially all of its assets, to its successor entity, provided that the successor entity assumes in writing all obligations under this Agreement.

### 11.5 Independent contractors

The Parties are independent contractors. Nothing in this Agreement creates an agency, partnership, joint venture, or employment relationship between the Parties.

### 11.6 Severability

If any provision of this Agreement is held invalid or unenforceable, the remaining provisions shall remain in full force and effect, and the invalid or unenforceable provision shall be reformed to the minimum extent necessary to make it valid and enforceable.

### 11.7 No waiver

No waiver of any provision of this Agreement is effective unless in writing and signed by the waiving Party. Failure to enforce any right under this Agreement does not constitute a waiver of that right or any other right.

### 11.8 Counterparts and electronic signature

This Agreement may be executed in counterparts, each of which shall be deemed an original but which together constitute one and the same instrument. Electronic signatures (DocuSign, HelloSign, Adobe Sign, or PGP-signed PDFs) and scanned wet-ink signatures shall be deemed valid and binding.

---

## 12. Signatures

By signing below, each Party's authorized signatory acknowledges that they have read and understood this Agreement and that they have authority to bind their respective entity.

**SIPSA LABS, INC.**

Signature: ___________________________________

Name: Missipssa Ounnar

Title: Founder

Date: ___________________________________

---

**[CUSTOMER LEGAL ENTITY NAME]**

Signature: ___________________________________

Name: [CUSTOMER_AUTHORIZED_SIGNATORY_NAME]

Title: [CUSTOMER_AUTHORIZED_SIGNATORY_TITLE]

Date: ___________________________________

---

# Appendix A — Customer info checklist (pre-engagement)

To be completed by Customer and returned to Sipsa on or before the Engagement Start. All fields are required unless marked optional.

| # | Item | Customer response |
|---|---|---|
| A.1 | HuggingFace `model_id` (for public Model) OR local checkpoint path (for fine-tuned Model) | _______________________ |
| A.2 | Model architecture family (e.g., Llama-3, Qwen3, Mistral, Mixtral, Phi, Gemma, custom) | _______________________ |
| A.3 | Approximate parameter count and layer count | _______________________ |
| A.4 | Customer evaluation task — choose ONE: <br> ☐ FineWeb-edu held-out PPL (default) <br> ☐ HumanEval <br> ☐ MMLU <br> ☐ Custom (describe below) | Custom description: _______________________ |
| A.5 | Target deployment hardware — choose ONE: <br> ☐ Consumer GPU (e.g., RTX 4090 / 5090) <br> ☐ Data-center GPU (e.g., A100, H100, B200) <br> ☐ Edge / embedded <br> ☐ Other (describe) | Description: _______________________ |
| A.6 | Calibration-data preference — choose ONE: <br> ☐ FineWeb-edu (default) <br> ☐ Customer-supplied corpus (will ship to Sipsa secure-transfer destination) | If customer-supplied, brief description: _______________________ |
| A.7 | Acceptance threshold — PPL ratio (compressed ÷ teacher): <br> ☐ ≤ 1.015 (default — within 1.5% of baseline) <br> ☐ Customer-specified value: _______ | _______________________ |
| A.8 | Designated technical contact (name, email, role) | _______________________ |
| A.9 | Designated business contact (name, email, role) | _______________________ |
| A.10 | Secure-transfer destination for deliverables (S3 bucket, GCS bucket, Azure container, SFTP, or alternative; include access credentials handoff plan) | _______________________ |
| A.11 | Billing contact (name, email, billing address, PO number if required) | _______________________ |
| A.12 (optional) | Any export-control or regulatory flags Sipsa should know (ITAR, EAR, HIPAA, SOC2, FedRAMP) | _______________________ |

---

# Appendix B — Mutual reference checks

The Parties agree to provide reasonable references during the Phase 0 engagement to support each Party's diligence on the other.

**B.1 Customer reference to Sipsa.** Customer agrees, on Sipsa's request, to provide one (1) named reference contact who can speak to Customer's evaluation of Sipsa's deliverables. The reference may be conducted verbally or in writing, at Customer's election, and may be provided anonymously (e.g., "VP of Inference Infrastructure at a major cloud provider") if Customer prefers. Customer's reference obligation is contingent on Acceptance under Section 4 and may be deferred until the close of the Acceptance Window.

**B.2 Sipsa reference to Customer.** Sipsa agrees to provide Customer, on Customer's request before the Engagement Start, with:

(a) the current public artifact verification dashboard at `docs/PUBLIC_VERIFICATION_DASHBOARD_2026_05_08.md` (or its then-current successor), listing the architectures Sipsa has compressed end-to-end with `uc verify` PASS;

(b) public links to the corresponding HuggingFace `SipsaLabs/*-uc-v3-bpw5` artifacts so Customer can independently reproduce the published `uc verify` PASS and PPL ratio numbers; and

(c) on reasonable request, an introduction to one (1) prior or current Sipsa engagement contact, subject to that contact's consent and any applicable confidentiality obligations.

---

*End of template. Replace all `[PLACEHOLDERS]` and remove this footer line before sending. Counsel review recommended before first use against any Customer.*
