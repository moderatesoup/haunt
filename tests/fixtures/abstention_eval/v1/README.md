# E6 abstention feasibility evidence v1

This dataset is independent of the frozen E0 retrieval regression lock. Its
logical IDs, canonical query hashes, and canonical record hashes are checked
against E0 before any fit analysis runs. `records.json`, unlabeled
`queries.json`, and `split.json` load first. Fit labels live only in
`fit-labels.json`. The physically separate `held-labels.json` is not opened,
parsed, hashed, or validated until fit observations, analysis, metrics, and the
fit-only boundary are complete. `dataset-manifest.json` is opened and verified
only after held-out scoring; joining both label files back onto query order
preserves the predeclared composite dataset hash.

The labels measure whether this small synthetic corpus contains an intended
answer, not whether a statement is true. The policy controls retrieval-evidence
sufficiency and never uses provenance, trust, RRF, source reputation, or a
reader model. Every change requires a new version and new reviewed hashes.

All forty hybrid answerable queries are manually authored semantic
paraphrases. Their relevant rows are required not to appear in the complete FTS
candidate set during reproduction, so all twenty held cases exercise vector
retrieval rather than a lexical or stemming shortcut. Both splits also contain
fifteen close absent-attribute or multi-record unanswerable cases, not only
far-topic negatives.

The FTS-only cohort is separable at the fit-only `0.875` threshold and passes
both held-out gates. The pinned `BAAI/bge-m3` ONNX, 1024-dimensional,
native-cosine cohort is not separable: the minimum boundary that abstains on all
fit negatives retains only 6/20 fit positives, and it retains 6/15 held
positives conditional on pre-abstention Recall@5. The report is therefore a
successful, reproducible `status=blocked` result—not a calibration artifact.
No runtime policy or public API wiring ships from this evidence.

Artifacts:

- `reports/fts.json`: deterministic FTS-only feasibility report under strict
  `HAUNT_OFFLINE=1` and network denial.
- `reports/hybrid-blocked.json`: deterministic pinned-hybrid blocker report
  using a hashed local cache with `HAUNT_OFFLINE` unset inside network denial.
- `hybrid-model-manifest.json`: the only accepted BGE-M3 artifact identity. It
  locks all five selected ONNX-directory files by relative path, size, and
  SHA-256, requires the external-data sidecar, and rejects quantized, root-level,
  missing, zero-byte, wrong-size, wrong-hash, and extra selected variants before
  embedding initialization.
- `reports/latency.json`: non-gating 1k/10k/100k timing observation. Its
  deterministic gate is one batched coverage statement for the fixed top five;
  absolute timings are machine-specific.
- `scripts/reproduce_abstention_eval.py`: regenerates either feasibility report
  and exits zero for the scientifically valid blocked result.
- `scripts/benchmark_abstention_evidence.py`: regenerates the latency evidence.
