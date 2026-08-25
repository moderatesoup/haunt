# Frozen retrieval evaluation

This is Haunt's CI retrieval-regression lock. Its public evaluator creates a
temporary `HAUNT_HOME` and forces `HAUNT_FTS_ONLY=1`, then restores the
caller's environment unchanged. It uses only the checked-in corpus and never
loads, downloads, or scores an embedding model. The harness seeds real `Store`
instances, performs supersession through `Store.contradict` with frozen clocks,
and calls `recall` or `planned_recall` through their namespace/open-existing
paths as declared by each case.

The corpus covers lexical ordering, Unicode FTS, current and as-of
supersession, a compiled temporal window, procedural-tier recall, namespace
isolation, and an out-of-distribution negative query. Vector/semantic quality
is intentionally not locked: locally available models and floating-point ANN
results are not a portable deterministic CI contract.

`baseline.json` is a lock, not an automatically updated artifact. Change the
corpus or retrieval contract first, review the per-case output and metric
delta, then explicitly relock and commit both files in the same change:

```sh
HAUNT_ALLOW_FROZEN_RETRIEVAL_RELOCK=1 PYTHONPATH=src python -m haunt.frozen_retrieval_eval --write-baseline
pytest tests/test_frozen_retrieval_eval.py
```

Do not relock for incidental host, model, or dependency differences. The test
checks the corpus/config hashes, each ranked fixture result, and the aggregate
metrics, so an unnoticed corpus/config/baseline mismatch fails CI. Exact
ranked-output locking is intentionally stronger than the metric floors: a
regression can return a distractor above the right memory, or change a negative
query into a non-empty result, while a rounded macro Recall@K or MRR still
looks acceptable. Relocking therefore requires reviewing each case, not merely
accepting a passing aggregate score.
