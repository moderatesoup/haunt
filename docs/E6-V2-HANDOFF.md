# E6 v2 — abstention handoff

**Not implemented in the v0.3 cleanup PR, deliberately.** This is what a v2
attempt needs to know before it starts, written while the surrounding code was
being changed so that the changes are on the record rather than discovered
later.

## Where E6 v1 actually stands

E6 v1 is **blocked on evidence, not on engineering**. The harness works; the
pinned hybrid cohort cannot satisfy the predeclared gates.

Frozen separately from E0:

| | |
|---|---|
| dataset hash | `8119f4508d3582bc665a5a0117940c6eeca593de56f33999563ddf188846264c` |
| split hash | `d57b1a02e30d48087065528bfc01575d67735295003de26b1087bb348d05414d` |
| size | 40 records, 160 cases |
| overlap with E0 | none — no logical-ID, canonical-record or query-hash overlap; sealed E0 paths diff byte-for-byte empty |

**FTS-only closes.** Fit cohort separable at `0.875`; held-out Recall@5,
conditional retention and negative abstention all `1.0`.

**Hybrid does not.** Under local `BAAI/bge-m3` ONNX (1024-d) with genuine
sqlite-vec native cosine execution, the fit-only boundary that gives 100%
negative abstention retains **6/20 conditional positives (0.30)** against a
required `0.95`. Held-out pre-abstention Recall@5 is `15/20` (`0.75`); at that
same unchanged boundary all 20 negatives abstain but only **6/15** conditional
positives survive (`0.40`).

**The mechanism, which is the part that matters for v2:** close
absent-attribute negatives dominate real semantic positives on *both* evidence
strength and the diagnostic top-two distance margin. No threshold over the
approved raw-evidence feature set separates them, because the two populations
are not separated in that feature space at all. A v2 that only re-tunes a
threshold will reproduce this result.

## The three honest unblock paths

1. Predeclare and review a **new raw retrieval feature** in a new, separately
   versioned E6 evidence set. The current feature set is what fails; adding to
   it is legitimate if it is declared before fitting.
2. Explicitly **amend the contract to permit a reader or cross-encoder**. This
   is a scope decision, not an implementation detail — the current contract
   forbids it, and the E6 feature definition names `reader_model` and
   `cross_encoder` as excluded.
3. **Leave hybrid abstention unshipped** and ship FTS-only abstention alone, or
   nothing.

### Not valid, and each has been considered and rejected

Removing or relabeling the hard negatives. Padding the positive set with
lexical positives. Thresholding `rrf_score` as though a fusion rank were a
confidence. Fitting on held-out labels. Silently falling back to FTS-only when
hybrid fails to calibrate. A mutation test exists specifically to fail the
`rrf_score` variant.

## What the v0.3 cleanup changed underneath E6

Four things a v2 implementer will otherwise trip over.

**The manifest verifier moved.** `verify_local_hybrid_cache`, the artifact
constants, canonical hashing and the JSON/audit helpers now live in
`src/haunt/model_manifest.py`. `abstention_eval` re-exports every one of them as
the same object, so existing E6 code and scripts are unaffected. It moved
because `haunt.embed` needs it at runtime on every model load and was reaching
it through a function-local import that dragged the whole E6 harness — plus
`recall`, `store`, `rerank` and the `urllib`/`ssl`/`email` stack — into every
process that embeds anything.

**`model_manifest.py` is now in `E6_EVIDENCE_PATHS`,** which restores the
isolation gate over the enforcement code. That imposes a **commit-composition
constraint**: no single commit may touch both an E6 evidence path and a public
runtime path, or `evaluate_profile` refuses with *"E6 blocker evidence must not
ship public runtime policy"*. This is not new machinery — it was already true
of `abstention_eval.py` — but the set of triggering files is wider now. Split
your commits along that line.

**`SCHEMA_VERSION` was split.** `model_manifest.MANIFEST_SCHEMA_VERSION`
versions the manifest *document*; `abstention_eval.SCHEMA_VERSION` versions the
E6 fixture set and its value is baked into `DATASET_MANIFEST_SHA256`. They were
one shared `1`, so bumping the E6 fixture schema would have silently rejected
every installed model cache, and a manifest v2 would have silently invalidated
the E6 fixtures. Both are still `1`. **Bump only the one you mean.**

**Retrieval tie ordering changed** — candidate and fused ties now break on
`content_hash` before `memory_id`, because `memory_id` is a fresh uuid4 per
write and re-randomized the order on every ingest. This does **not** move the
committed E6 results: `tests/test_abstention_feasibility.py` asserts the frozen
report and passes on all four gate matrices. Recorded because a v2 regenerating
evidence should know the ordering is now reproducible across ingests, which it
was not when v1's evidence was produced.

## Before regenerating any E6 evidence

- Commit the tree first. `_e6_attributed_diff` folds the working-tree diff of
  the public runtime surface in whenever an E6 evidence path is uncommitted, so
  an unrelated dirty `recall.py` will make `evaluate_profile` refuse.
- Reproduction needs the verified local cache. Hybrid leaves `HAUNT_OFFLINE`
  unset, requires an explicit verified cache, denies socket/DNS/HTTP, proves
  zero attempts, and fails on any non-native vector arm.
- The manifest preflight accepts only `haunt-bge-m3-onnx-split-f8425123-v1`:
  exact relative paths, sizes and hashes for config, non-quantized ONNX, the
  required external-data sidecar, tokenizer and tokenizer config.
  Missing/extra/zero-byte/wrong-size/wrong-hash/missing-sidecar/variant
  mismatches all fail *before* embedding initialization; quantized and
  root-level variants are explicitly forbidden. That behaviour is now pinned by
  a 24-state differential harness and is byte-identical to the pre-move
  implementation.
