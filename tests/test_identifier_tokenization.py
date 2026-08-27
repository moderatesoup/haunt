"""C9: does the FTS tokenizer let people find code identifiers?

Investigation outcome: the tokenizer is UNCHANGED. This file locks in why.

`memories_fts` uses ``tokenize='porter unicode61'`` (src/haunt/store.py's
`_init_namespace_schema`). unicode61 already treats `_` and `.` as
non-word separators, so `content_hash`, `_ensure_namespace_schema`, and
`haunt.store.observe` already tokenize into their word parts (["content",
"hash"], ["ensure", "namespace", "schema"], ["haunt", "store", "observe"])
and are already findable by a whole-identifier query (recall wraps each
extracted query token in an FTS5 phrase, which re-tokenizes it the same
way, so "content_hash" -> phrase(content, hash) matches an adjacent
[content, hash] token run) OR by any single word part. That part of the
originally reported problem does not actually reproduce end to end.

The real, confirmed gap is camelCase/PascalCase: unicode61 has no notion of
a case transition, so `getUserName` and `HTTPResponse` each tokenize as one
opaque blob (`getusername`, `httpresponse`). A query for the identifier
itself (any case) still matches; a query for its "natural words" (`get
user name`, `username`, `http response`) does not, because there is no
[get, user, name] token run in the index to match against -- there is one
token, and none of the query words equal it.

Two candidate fixes were evaluated against real evidence, not assumption:

  - `unicode61 tokenchars='_.'`: does not touch camelCase at all (case
    transitions are not something `tokenchars` can express -- it only adds
    characters to the "this joins a token" set, and camelCase has no
    non-letter characters to add), and it actively regresses the
    already-working snake_case/dotted-path word-part search (folding
    `content_hash` into one token means a bare `hash` query stops
    matching it). Net loss on both axes; rejected without needing a full
    E0 baseline regeneration cycle -- it doesn't touch the actual defect.

  - `trigram`: does fix camelCase decomposition (substring matching finds
    `Name` inside `getUserName`), and technically passed E0's frozen gate
    byte-for-byte (all 10 cases, all metrics, unchanged) when actually
    run through `frozen_retrieval_eval.evaluate()`. But it is a wholesale
    tokenizer replacement that drops Porter's real morphological stemming
    for every query in the system, not just identifier queries -- and
    independent verification through the real `haunt.recall.recall()`
    pipeline (not a proxy) found a genuine, ordinary-English regression
    E0's ten cases don't happen to cover: a query for "use" no longer
    finds content containing "using" (porter unicode61 does; trigram's
    substring matching only survives when the query word happens to be a
    literal prefix of the content word, e.g. porter_stemming's own locked
    "synchronize"/"synchronizes" pair, or "argue"/"argued",
    "cache"/"cached", "extract"/"extracts" -- "use"/"using" is not a
    prefix relationship and fails). A tiny frozen corpus passing is not
    proof of no regression; this is a real, reproduced one. Given the
    corpus this ships against is prose-and-code coding sessions, not
    identifiers alone, trading working ordinary-word stemming for
    identifier substring matching is a net loss, not a net win. Rejected.

No *custom tokenizer* is implementable here: FTS5 tokenizers are a C-level
extension point and Python's stdlib `sqlite3` exposes no hook to register
one. That rules out replacing the tokenizer, which is narrower than ruling
out a fix. At least two paths remain unexplored and are deliberately
deferred rather than dismissed:

  - Expand only the *indexed* text at insert time -- split camelCase with
    a regex so `getUserName` also indexes `get user name` -- leaving both
    the tokenizer and the verbatim `memories.content` untouched. An
    adversarial review prototyped this and confirmed camelCase becomes
    searchable with `use`/`using` stemming still intact.
  - Maintain a second, identifier-specific FTS table alongside this one
    and fuse it at query time.

Both change what is indexed and therefore move BM25 ranks, so either needs
its own frozen-eval cycle and ranking work; that is why C9 ships nothing
*now*, not because no fix exists. This file documents and locks the
current, unchanged behavior across all three identifier shapes so a future
silent tokenizer change is caught here rather than in production search.

Separately: E0's frozen gate passed trigram byte-for-byte despite that
regression, because its only stemming case ("synchronize"/"synchronizes")
is a literal prefix pair that trigram matches by substring coincidence. A
non-prefix inflection case would close that blind spot. That is a gap in
the gate itself, independent of C9.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def identifier_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "haunt"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_EMBED_MODEL", "off")
    monkeypatch.setenv("HAUNT_NAMESPACE", "identifiers")
    from haunt import embed
    from haunt.paths import ensure_layout
    from haunt.store import init_registry, register_namespace

    embed.reset()
    ensure_layout()
    init_registry()
    register_namespace("identifiers")
    yield tmp_path / "haunt"
    embed.reset()


def _found(store, query, memory_id):
    from haunt.recall import recall

    hits = recall(query, namespace="identifiers", k=5, use_vectors=False, store=store)
    return memory_id in [h.memory_id for h in hits]


# ---------------------------------------------------------------------------
# snake_case and dotted.path: already decompose on `_`/`.`, already findable
# by any word part -- this is the "not actually broken" half of C9.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "matching_queries"),
    [
        ("content_hash", ["content_hash", "content hash", "hash", "content"]),
        (
            "_ensure_namespace_schema",
            ["_ensure_namespace_schema", "ensure_namespace_schema", "namespace", "schema"],
        ),
        (
            "snake_case_with_many_parts",
            ["snake_case_with_many_parts", "snake", "parts", "many"],
        ),
    ],
)
def test_snake_case_identifiers_are_findable_by_whole_id_or_any_part(
    identifier_env, content, matching_queries
):
    from haunt.store import Store

    with Store("identifiers") as store:
        result = store.observe(content, defer_embedding=True)
        for query in matching_queries:
            assert _found(store, query, result.memory_id), (
                f"expected {query!r} to find {content!r}"
            )


def test_dotted_path_identifier_is_findable_by_whole_path_or_any_segment(identifier_env):
    from haunt.store import Store

    with Store("identifiers") as store:
        result = store.observe("haunt.store.observe", defer_embedding=True)
        for query in ("haunt.store.observe", "haunt store observe", "observe", "store", "haunt"):
            assert _found(store, query, result.memory_id), (
                f"expected {query!r} to find the dotted path"
            )


# ---------------------------------------------------------------------------
# camelCase / PascalCase: the confirmed, real, and (per C9's evidence-based
# decision above) currently-unfixed gap. Locked both ways: whole-identifier
# search still works; decomposed "natural word" search still does not.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "id_queries", "decomposed_queries"),
    [
        (
            "getUserName",
            ["getUserName", "getusername"],
            ["get user name", "username", "user", "name"],
        ),
        (
            "HTTPResponse",
            ["HTTPResponse", "httpresponse"],
            ["http response", "response", "http"],
        ),
    ],
)
def test_camel_case_identifiers_match_whole_but_not_decomposed_words(
    identifier_env, content, id_queries, decomposed_queries
):
    from haunt.store import Store

    with Store("identifiers") as store:
        result = store.observe(content, defer_embedding=True)

        for query in id_queries:
            assert _found(store, query, result.memory_id), (
                f"expected the whole identifier query {query!r} to find {content!r}"
            )
        for query in decomposed_queries:
            assert not _found(store, query, result.memory_id), (
                f"expected {query!r} to still NOT find {content!r} -- "
                "if this now passes, the tokenizer changed and C9's "
                "evidence/decision in this file's docstring needs revisiting"
            )


def test_porter_stemming_still_works_alongside_unchanged_identifier_behavior(identifier_env):
    """A minimal in-repo canary for the exact regression that ruled out the
    trigram candidate (see this module's docstring): an ordinary English
    inflection that is NOT a prefix relationship. If this ever starts
    failing, something changed the tokenizer's stemming behavior."""
    from haunt.store import Store

    with Store("identifiers") as store:
        result = store.observe(
            "The team is using the new deploy script today.", defer_embedding=True
        )
        assert _found(store, "use", result.memory_id)
