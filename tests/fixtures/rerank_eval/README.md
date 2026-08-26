# Rerank measurement fixture (C12)

This is the corpus `haunt.rerank_eval` measures against. Unlike
`tests/fixtures/retrieval_eval/` (the E0 frozen FTS regression lock), this is
not a CI gate and has no `baseline.json` — see `src/haunt/rerank_eval.py`'s
module docstring for why. It is a re-runnable measurement fixture: the same
corpus, evaluated with and without `haunt.rerank.mmr_rerank`, to answer
whether the lexical MMR diversity pass helps, hurts, or does nothing.

Twenty records, eight cases, split into two query classes the C12 backlog
entry names:

- **`clear_top1`** (4 cases) — one unambiguous right answer, no near-duplicate
  content anywhere nearby. These exist to prove reranking does not cost
  anything on easy queries.
- **`ambiguous`** (4 cases) — a cluster of three near-identical restatements
  of the same fact (a `rate_limit`, `retro`, `postmortem`, or `budget`
  scenario) plus one lexically distinct, genuinely relevant second fact that
  shares only partial vocabulary with the query. At `k=3`, plain top-k
  truncation tends to fill the window with two or three near-duplicates of
  the same fact and miss the second one; MMR's redundancy penalty can make
  room for it instead. `relevant` for each ambiguous case is
  `[<canonical duplicate>, <distinct fact>]` — one representative of the
  cluster plus the distinct fact, not all three duplicates individually,
  since recall@k against a ground truth that over-counted near-duplicates
  would not respond to a diversity pass at all (reordering *within* an
  already-fully-relevant cluster does not change recall@k or MRR).

Content lengths within each near-duplicate cluster are deliberately kept
distinct (not just reworded) so that SQLite FTS5's bm25 never produces an
exact tie between two records — see the corpus edit history for why: an
earlier draft had two records tie in raw bm25 score, and recall.py's tie
break for equal ranks falls back to the FTS rowid, which traces back to each
record's freshly-generated `memories.id` — a new random value every time
`_seed()` reseeds a fresh temporary store. An exact bm25 tie is therefore the
one way this fixture could stop being deterministic across repeated
`evaluate()` calls; `tests/test_rerank_eval.py::test_evaluate_is_deterministic`
is the regression gate for that, and `haunt.rerank_eval` requires
`HAUNT_FTS_ONLY=1` for the same portability reason
`tests/fixtures/retrieval_eval/README.md` gives for its own corpus: no model,
no vectors, nothing host- or install-dependent.
