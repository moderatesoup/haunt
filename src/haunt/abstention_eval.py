"""Deterministic E6 abstention-feasibility evidence.

This module is deliberately evaluation-only.  It records the raw retrieval
signals that E6 permits, proves the fit/held-out split is isolated from E0,
and reports whether a fit-only threshold can satisfy the predeclared gates.
It does not install a runtime policy: the pinned hybrid cohort currently
cannot satisfy those gates without a feature outside the approved contract.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socket
import subprocess
import unicodedata
import urllib.request
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from haunt import embed
from haunt.recall import Hit, recall
from haunt.store import ReadOnlyStore, Store, open_existing_readonly

SCHEMA_VERSION = 1
FEATURE_IMPLEMENTATION = "haunt-abstention-feasibility-v1"
FTS_PROFILE_ID = "fts5-porter-unicode61-top40-k5-v1"
HYBRID_PROFILE_ID = "bge-m3-onnx-1024-native-cosine-fts5-rrf60-top40-k5-v1"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "abstention_eval" / "v1"
DEFAULT_E0_CORPUS = ROOT / "tests" / "fixtures" / "retrieval_eval" / "corpus.json"
E0_BASE_REVISION = "ed806b2"
E0_SEALED_PATHS = (
    "src/haunt/frozen_retrieval_eval.py",
    "tests/fixtures/retrieval_eval",
    "tests/test_frozen_retrieval_eval.py",
)
PUBLIC_RUNTIME_PATHS = (
    "pyproject.toml",
    "src/haunt/__init__.py",
    "src/haunt/recall.py",
    "src/haunt/planner.py",
    "src/haunt/cli.py",
    "src/haunt/mcp_server.py",
    "src/haunt/dashboard.py",
)

_FTS_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

FEATURE_DEFINITION: dict[str, Any] = {
    "schema_version": 1,
    "implementation": FEATURE_IMPLEMENTATION,
    "candidate_scope": "fixed_pre_abstention_top_5",
    "raw_inputs": {
        "fts": [
            "candidate_membership",
            "component_rank",
            "fts5_bm25_raw",
            "deduplicated_query_token_count",
            "matched_query_token_count",
            "porter_unicode61_query_coverage",
        ],
        "vector": [
            "candidate_membership",
            "component_rank",
            "native_cosine_distance",
            "native_cosine_top_two_distance_margin",
        ],
    },
    "derived_inputs": {
        "vector_closeness": "clamp(1 - native_cosine_distance, 0, 1)",
    },
    "candidate_membership_rule": {
        "fts": "coverage contributes only when fts_rank is not null",
        "vector": (
            "closeness contributes only when vec_rank is not null and "
            "vec_metric is native cosine_distance"
        ),
    },
    "decision_feature": (
        "max over the fixed top five candidates of eligible FTS coverage or "
        "eligible native-cosine closeness"
    ),
    "diagnostic_only": [
        "fts5_bm25_raw",
        "component_ranks",
        "candidate_membership",
        "native_cosine_top_two_distance_margin",
    ],
    "forbidden_inputs": [
        "fused_rrf_score",
        "renormalized_rrf_score",
        "fact_truth",
        "source_reputation",
        "provenance_fidelity",
        "trust_label",
        "reader_model",
        "cross_encoder",
    ],
    "semantics": "retrieval_evidence_strength_not_probability_or_truth_confidence",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RetrievalObservation:
    case_id: str
    returned: tuple[str, ...]
    evidence: dict[str, Any]
    vector_profile: dict[str, Any]
    lexical_returned: tuple[str, ...] | None = None

    @property
    def strength(self) -> float:
        return float(self.evidence["strength"])

    @property
    def vector_margin(self) -> float | None:
        value = self.evidence["diagnostics"].get(
            "native_cosine_top_two_distance_margin"
        )
        return None if value is None else float(value)


@dataclass(frozen=True)
class FitLabel:
    label: str
    relevant: tuple[str, ...]


@dataclass(frozen=True)
class FitLabels:
    """Capability object containing fit labels and nothing from held-out."""

    by_case: Mapping[str, FitLabel]

    def __post_init__(self) -> None:
        if not self.by_case:
            raise ValueError("fit labels cannot be empty")
        if any(
            value.label not in {"answerable", "unanswerable"}
            for value in self.by_case.values()
        ):
            raise ValueError("invalid fit label")


class NetworkDeny(AbstractContextManager["NetworkDeny"]):
    """Fail closed on common Python DNS/socket/HTTP network surfaces."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._originals: dict[str, Any] = {}

    def _deny(self, *args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs
        self.attempts.append(repr(target))
        raise RuntimeError("E6 network-deny harness blocked network access")

    def __enter__(self) -> "NetworkDeny":
        original_socket = socket.socket
        self._originals = {
            "socket": original_socket,
            "connect": original_socket.connect,
            "connect_ex": original_socket.connect_ex,
            "send": original_socket.send,
            "sendall": original_socket.sendall,
            "sendto": original_socket.sendto,
            "create_connection": socket.create_connection,
            "create_server": socket.create_server,
            "getaddrinfo": socket.getaddrinfo,
            "gethostbyname": socket.gethostbyname,
            "gethostbyname_ex": socket.gethostbyname_ex,
            "gethostbyaddr": socket.gethostbyaddr,
            "getnameinfo": socket.getnameinfo,
            "urlopen": urllib.request.urlopen,
            "http_connect": http.client.HTTPConnection.connect,
            "https_connect": http.client.HTTPSConnection.connect,
        }
        original_socket.connect = self._deny  # type: ignore[method-assign]
        original_socket.connect_ex = self._deny  # type: ignore[method-assign]
        original_socket.send = self._deny  # type: ignore[method-assign]
        original_socket.sendall = self._deny  # type: ignore[method-assign]
        original_socket.sendto = self._deny  # type: ignore[method-assign]
        socket.create_connection = self._deny  # type: ignore[assignment]
        socket.create_server = self._deny  # type: ignore[assignment]
        socket.getaddrinfo = self._deny  # type: ignore[assignment]
        socket.gethostbyname = self._deny  # type: ignore[assignment]
        socket.gethostbyname_ex = self._deny  # type: ignore[assignment]
        socket.gethostbyaddr = self._deny  # type: ignore[assignment]
        socket.getnameinfo = self._deny  # type: ignore[assignment]
        urllib.request.urlopen = self._deny  # type: ignore[assignment]
        http.client.HTTPConnection.connect = self._deny  # type: ignore[method-assign]
        http.client.HTTPSConnection.connect = self._deny  # type: ignore[method-assign]
        # Constructor patch comes last, after the original type's methods are
        # fenced. This catches TCP and UDP creation as well as helper paths.
        socket.socket = self._deny  # type: ignore[assignment,misc]
        return self

    def __exit__(self, *args: Any) -> None:
        original_socket = self._originals["socket"]
        socket.socket = original_socket  # type: ignore[assignment,misc]
        original_socket.connect = self._originals["connect"]
        original_socket.connect_ex = self._originals["connect_ex"]
        original_socket.send = self._originals["send"]
        original_socket.sendall = self._originals["sendall"]
        original_socket.sendto = self._originals["sendto"]
        socket.create_connection = self._originals["create_connection"]
        socket.create_server = self._originals["create_server"]
        socket.getaddrinfo = self._originals["getaddrinfo"]
        socket.gethostbyname = self._originals["gethostbyname"]
        socket.gethostbyname_ex = self._originals["gethostbyname_ex"]
        socket.gethostbyaddr = self._originals["gethostbyaddr"]
        socket.getnameinfo = self._originals["getnameinfo"]
        urllib.request.urlopen = self._originals["urlopen"]
        http.client.HTTPConnection.connect = self._originals["http_connect"]
        http.client.HTTPSConnection.connect = self._originals["https_connect"]


class _DeterministicFixtureIds(AbstractContextManager["_DeterministicFixtureIds"]):
    """Give generated fixture rows stable UUIDs so tie breaks reproduce."""

    def __init__(self) -> None:
        self.counter = 0
        self._originals: dict[str, Any] = {}

    def _new_id(self) -> str:
        self.counter += 1
        return str(uuid.UUID(int=self.counter))

    def __enter__(self) -> "_DeterministicFixtureIds":
        from haunt import graph, store

        self._originals = {"graph": graph.new_id, "store": store.new_id}
        graph.new_id = self._new_id
        store.new_id = self._new_id
        return self

    def __exit__(self, *args: Any) -> None:
        from haunt import graph, store

        graph.new_id = self._originals["graph"]
        store.new_id = self._originals["store"]


def load_inputs(
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = json.loads((fixture_dir / "dataset.json").read_text("utf-8"))
    split = json.loads((fixture_dir / "split.json").read_text("utf-8"))
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E6 dataset schema")
    if split.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E6 split schema")
    records = dataset.get("records")
    cases = dataset.get("cases")
    if not isinstance(records, list) or not isinstance(cases, list):
        raise ValueError("E6 dataset requires records and cases")
    record_ids = [row.get("id") for row in records]
    case_ids = [row.get("id") for row in cases]
    if len(record_ids) != len(set(record_ids)) or not all(
        isinstance(value, str) for value in record_ids
    ):
        raise ValueError("E6 record IDs must be unique strings")
    if len(case_ids) != len(set(case_ids)) or not all(
        isinstance(value, str) for value in case_ids
    ):
        raise ValueError("E6 case IDs must be unique strings")
    memberships = [
        case_id
        for phase in ("fit", "held_out")
        for profile in ("fts", "hybrid")
        for case_id in split.get(phase, {}).get(profile, [])
    ]
    if len(memberships) != len(set(memberships)) or set(memberships) != set(case_ids):
        raise ValueError("E6 split must partition every case exactly once")
    by_case = {row["id"]: row for row in cases}
    for phase in ("fit", "held_out"):
        for profile in ("fts", "hybrid"):
            selected = [by_case[case_id] for case_id in split[phase][profile]]
            positives = [row for row in selected if row["label"] == "answerable"]
            negatives = [row for row in selected if row["label"] == "unanswerable"]
            if len(positives) < 20 or len(negatives) < 20:
                raise ValueError(f"{phase}/{profile} requires 20+ cases per label")
            if any(row["profile"] != profile for row in selected):
                raise ValueError("profile/split mismatch")
            if profile == "hybrid" and any(
                row.get("semantic_paraphrase") is not True
                or row.get("required_lexical_overlap") is not False
                for row in positives
            ):
                raise ValueError("every hybrid answerable must be a pure paraphrase")
    return dataset, split


def separation_evidence(
    dataset: dict[str, Any], *, e0_path: Path = DEFAULT_E0_CORPUS
) -> dict[str, Any]:
    """Prove structural disjointness without altering sealed E0 artifacts."""
    e0 = json.loads(e0_path.read_text("utf-8"))
    e6_ids = {str(row["id"]) for row in dataset["records"]}
    e0_ids = {str(row["id"]) for row in e0["records"]}
    e6_query_hashes = {text_hash(str(row["query"])) for row in dataset["cases"]}
    e0_query_hashes = {text_hash(str(row["query"])) for row in e0["cases"]}
    e6_record_hashes = {canonical_hash(row) for row in dataset["records"]}
    e0_record_hashes = {canonical_hash(row) for row in e0["records"]}
    overlaps = {
        "logical_ids": sorted(e6_ids & e0_ids),
        "query_sha256": sorted(e6_query_hashes & e0_query_hashes),
        "record_sha256": sorted(e6_record_hashes & e0_record_hashes),
    }
    if any(overlaps.values()):
        raise ValueError(f"E6 evidence overlaps sealed E0: {overlaps}")
    return {
        "e0_canonical_dataset_sha256": canonical_hash(e0),
        "e6_logical_id_count": len(e6_ids),
        "e6_query_hash_count": len(e6_query_hashes),
        "e6_record_hash_count": len(e6_record_hashes),
        "overlaps": overlaps,
        "manual_semantic_audit_required": True,
    }


def sealed_e0_evidence() -> dict[str, Any]:
    command = ["git", "diff", "--exit-code", E0_BASE_REVISION, "--", *E0_SEALED_PATHS]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    tracked = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            E0_BASE_REVISION,
            "--",
            *E0_SEALED_PATHS,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    byte_mismatches: list[str] = []
    byte_sha256: dict[str, str] = {}
    for relative in tracked:
        base_bytes = subprocess.run(
            ["git", "show", f"{E0_BASE_REVISION}:{relative}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        working_bytes = (ROOT / relative).read_bytes()
        if working_bytes != base_bytes:
            byte_mismatches.append(relative)
        byte_sha256[relative] = hashlib.sha256(working_bytes).hexdigest()
    return {
        "base_revision": E0_BASE_REVISION,
        "paths": list(E0_SEALED_PATHS),
        "diff_empty": completed.returncode == 0,
        "diff": completed.stdout,
        "tracked_file_count": len(tracked),
        "byte_mismatches": byte_mismatches,
        "working_tree_byte_sha256": byte_sha256,
    }


def public_runtime_evidence() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "diff", "--exit-code", E0_BASE_REVISION, "--", *PUBLIC_RUNTIME_PATHS],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "base_revision": E0_BASE_REVISION,
        "paths": list(PUBLIC_RUNTIME_PATHS),
        "diff_empty": completed.returncode == 0,
        "diff": completed.stdout,
    }


def verify_local_hybrid_cache(cache_root: Path) -> dict[str, Any]:
    """Hash exact local ONNX inputs before entering the network-deny guard."""
    model_root = cache_root / "BAAI-bge-m3"
    onnx_candidates = [
        model_root / "onnx" / "model.onnx",
        model_root / "onnx" / "model_quantized.onnx",
        model_root / "model.onnx",
        model_root / "model_quantized.onnx",
    ]
    tokenizer_candidates = [
        model_root / "onnx" / "tokenizer.json",
        model_root / "tokenizer.json",
    ]
    onnx = next((path for path in onnx_candidates if path.is_file()), None)
    tokenizer = next((path for path in tokenizer_candidates if path.is_file()), None)
    if onnx is None or tokenizer is None:
        raise RuntimeError(
            "hybrid reproduction requires an explicit verified HAUNT_MODEL_CACHE "
            "containing BAAI-bge-m3 ONNX and tokenizer files"
        )
    files = [onnx, tokenizer]
    sidecar = onnx.with_name("model.onnx_data")
    if sidecar.is_file():
        files.append(sidecar)
    return {
        "cache_root_supplied": True,
        "model_root": "BAAI-bge-m3",
        "files": [
            {
                "relative_path": str(path.relative_to(cache_root)),
                "size": path.stat().st_size,
                "sha256": _file_hash(path),
            }
            for path in files
        ],
    }


def classify_vector_profile(
    *,
    requested: bool,
    sqlite_vec_available: bool,
    native_table: bool,
    persisted_embedding_rows: int,
    candidate_metrics: Sequence[str],
) -> dict[str, Any]:
    """Classify execution evidence without treating empty fallback as native."""
    metrics = set(candidate_metrics)
    if not requested:
        state = "not_requested"
        match = False
    elif metrics == {"cosine_distance"} and sqlite_vec_available and native_table:
        state = "native_cosine_candidate"
        match = True
    elif "l2_distance" in metrics and "cosine_distance" not in metrics:
        state = "persisted_l2_candidate"
        match = False
    elif metrics:
        state = "mixed_or_unknown_candidate"
        match = False
    elif sqlite_vec_available and native_table:
        state = "native_cosine_empty"
        match = True
    elif persisted_embedding_rows:
        state = "persisted_l2_empty"
        match = False
    elif not sqlite_vec_available:
        state = "sqlite_vec_unavailable_empty"
        match = False
    else:
        state = "native_table_absent_empty"
        match = False
    return {
        "state": state,
        "requested": requested,
        "sqlite_vec_available": sqlite_vec_available,
        "native_table": native_table,
        "persisted_embedding_rows": persisted_embedding_rows,
        "candidate_metrics": sorted(metrics),
        "matches_pinned_native_cosine": match,
    }


def _observed_vector_profile(
    store: ReadOnlyStore, hits: Sequence[Hit], *, requested: bool
) -> dict[str, Any]:
    sqlite_vec_available = bool(store.vec_ok())
    native_table = bool(
        store.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_memories'"
        ).fetchone()
    )
    persisted = int(
        store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        ).fetchone()[0]
    )
    metrics = [hit.vec_metric for hit in hits if hit.vec_rank is not None]
    if any(metric is None for metric in metrics):
        metrics = [str(metric) for metric in metrics]
    return classify_vector_profile(
        requested=requested,
        sqlite_vec_available=sqlite_vec_available,
        native_table=native_table,
        persisted_embedding_rows=persisted,
        candidate_metrics=[str(metric) for metric in metrics],
    )


def _unique_query_tokens(query: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    return tuple(dict.fromkeys(_FTS_TOKEN.findall(normalized)))[:24]


def _match_token(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def _coverage_many(
    store: ReadOnlyStore, query: str, memory_ids: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Measure top-hit token coverage with batched SQL, never hit×token N+1."""
    unique_ids = tuple(dict.fromkeys(memory_ids))
    tokens = _unique_query_tokens(query)
    matched: dict[str, set[int]] = {memory_id: set() for memory_id in unique_ids}
    if not unique_ids or not tokens:
        return (
            {
                memory_id: {
                    "query_token_count": len(tokens),
                    "matched_query_token_count": 0,
                    "coverage": 0.0,
                }
                for memory_id in unique_ids
            },
            {
                "deduplicated_query_tokens": list(tokens),
                "sql_statement_count": 0,
                "batch_size": 0,
            },
        )
    # SQLite's conventional parameter ceiling is 999. Each UNION arm carries
    # its token index, MATCH query, and the same small fixed top-five ID list.
    parameters_per_term = 2 + len(unique_ids)
    terms_per_statement = max(1, 999 // parameters_per_term)
    statement_count = 0
    for start in range(0, len(tokens), terms_per_statement):
        batch = tokens[start : start + terms_per_statement]
        arms: list[str] = []
        params: list[Any] = []
        id_placeholders = ",".join("?" for _ in unique_ids)
        for offset, token in enumerate(batch, start=start):
            arms.append(
                "SELECT ? AS token_index, f.id AS memory_id "
                "FROM memories_fts f "
                f"WHERE memories_fts MATCH ? AND f.id IN ({id_placeholders})"
            )
            params.extend([offset, _match_token(token), *unique_ids])
        rows = store.conn.execute(" UNION ALL ".join(arms), params).fetchall()
        statement_count += 1
        for row in rows:
            matched[str(row["memory_id"])].add(int(row["token_index"]))
    return (
        {
            memory_id: {
                "query_token_count": len(tokens),
                "matched_query_token_count": len(matched[memory_id]),
                "coverage": len(matched[memory_id]) / len(tokens),
            }
            for memory_id in unique_ids
        },
        {
            "deduplicated_query_tokens": list(tokens),
            "sql_statement_count": statement_count,
            "batch_size": terms_per_statement,
        },
    )


def _evidence(store: ReadOnlyStore, query: str, hits: Sequence[Hit]) -> dict[str, Any]:
    fixed_hits = list(hits[:5])
    coverage, coverage_diag = _coverage_many(
        store, query, [hit.memory_id for hit in fixed_hits]
    )
    candidates: list[dict[str, Any]] = []
    strengths: list[float] = []
    native_distances: list[float] = []
    for hit in fixed_hits:
        fts_candidate = hit.fts_rank is not None
        vector_candidate = (
            hit.vec_rank is not None
            and hit.vec_metric == "cosine_distance"
            and hit.vec_distance is not None
        )
        coverage_row = coverage[hit.memory_id]
        closeness = (
            max(0.0, min(1.0, 1.0 - float(hit.vec_distance)))
            if vector_candidate
            else None
        )
        eligible: list[float] = []
        if fts_candidate:
            eligible.append(float(coverage_row["coverage"]))
        if closeness is not None:
            eligible.append(closeness)
            native_distances.append(float(hit.vec_distance))
        candidate_strength = max(eligible, default=0.0)
        strengths.append(candidate_strength)
        candidates.append(
            {
                "memory_id": hit.memory_id,
                "final_rank": hit.final_rank,
                "fts": {
                    "candidate": fts_candidate,
                    "rank": hit.fts_rank,
                    "bm25_raw": hit.fts_rank_raw,
                    **coverage_row,
                    "eligible_for_strength": fts_candidate,
                },
                "vector": {
                    "candidate": hit.vec_rank is not None,
                    "rank": hit.vec_rank,
                    "distance": hit.vec_distance,
                    "metric": hit.vec_metric,
                    "closeness": closeness,
                    "eligible_for_strength": vector_candidate,
                },
                "candidate_strength": candidate_strength,
            }
        )
    native_distances.sort()
    margin = (
        native_distances[1] - native_distances[0]
        if len(native_distances) >= 2
        else None
    )
    return {
        "feature_implementation": FEATURE_IMPLEMENTATION,
        "candidate_count": len(fixed_hits),
        "strength": max(strengths, default=0.0),
        "candidates": candidates,
        "diagnostics": {
            "native_cosine_top_two_distance_margin": margin,
            "coverage_sql_statement_count": coverage_diag["sql_statement_count"],
            "coverage_batch_size": coverage_diag["batch_size"],
            "deduplicated_query_tokens": coverage_diag["deduplicated_query_tokens"],
            "rrf_score_read_by_feature": False,
        },
    }


def _fit_labels(
    dataset: dict[str, Any], split: dict[str, Any], profile: str
) -> FitLabels:
    by_case = {row["id"]: row for row in dataset["cases"]}
    return FitLabels(
        {
            case_id: FitLabel(
                label=str(by_case[case_id]["label"]),
                relevant=tuple(str(value) for value in by_case[case_id]["relevant"]),
            )
            for case_id in split["fit"][profile]
        }
    )


def _labels_for_ids(
    dataset: dict[str, Any], case_ids: Sequence[str]
) -> dict[str, FitLabel]:
    by_case = {row["id"]: row for row in dataset["cases"]}
    return {
        case_id: FitLabel(
            label=str(by_case[case_id]["label"]),
            relevant=tuple(str(value) for value in by_case[case_id]["relevant"]),
        )
        for case_id in case_ids
    }


def _observe_case(
    case: Mapping[str, Any],
    memory_ids: Mapping[str, str],
    *,
    profile: str,
    store: ReadOnlyStore,
) -> RetrievalObservation:
    use_vectors = profile == "hybrid"
    hits = recall(str(case["query"]), k=5, store=store, use_vectors=use_vectors)
    reverse_ids = {physical: logical for logical, physical in memory_ids.items()}
    returned = tuple(reverse_ids[hit.memory_id] for hit in hits)
    vector_profile = _observed_vector_profile(store, hits, requested=use_vectors)
    if use_vectors and not vector_profile["matches_pinned_native_cosine"]:
        raise RuntimeError(f"pinned hybrid execution mismatch: {vector_profile}")
    lexical_returned: tuple[str, ...] | None = None
    if profile == "hybrid" and case.get("semantic_paraphrase"):
        lexical = recall(str(case["query"]), k=40, store=store, use_vectors=False)
        lexical_returned = tuple(reverse_ids[hit.memory_id] for hit in lexical)
        relevant = set(str(value) for value in case["relevant"])
        if relevant & set(lexical_returned):
            raise ValueError(
                f"semantic case {case['id']} has a lexical/stemming answer path"
            )
    evidence = _evidence(store, str(case["query"]), hits)
    for candidate in evidence["candidates"]:
        candidate["logical_id"] = reverse_ids[candidate.pop("memory_id")]
    return RetrievalObservation(
        case_id=str(case["id"]),
        returned=returned,
        evidence=evidence,
        vector_profile=vector_profile,
        lexical_returned=lexical_returned,
    )


def math_ceil_95(count: int) -> int:
    return (95 * count + 99) // 100


def _score(
    observations: Sequence[RetrievalObservation],
    labels: Mapping[str, FitLabel],
    threshold: float,
) -> dict[str, Any]:
    by_id = {row.case_id: row for row in observations}
    if set(by_id) != set(labels):
        raise ValueError("score observations and labels must match exactly")
    answerable = [key for key, label in labels.items() if label.label == "answerable"]
    unanswerable = [
        key for key, label in labels.items() if label.label == "unanswerable"
    ]
    recalled = [
        key
        for key in answerable
        if set(labels[key].relevant) & set(by_id[key].returned[:5])
    ]
    retained = [key for key in recalled if by_id[key].strength >= threshold]
    abstained_negative = [
        key for key in unanswerable if by_id[key].strength < threshold
    ]
    return {
        "cases": len(labels),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "pre_abstention_recall_at_5": len(recalled) / len(answerable),
        "conditional_answerable_count": len(recalled),
        "conditional_answerable_retained": len(retained),
        "conditional_answerable_retention": (
            len(retained) / len(recalled) if recalled else 0.0
        ),
        "unanswerable_abstained": len(abstained_negative),
        "unanswerable_abstention_rate": (
            len(abstained_negative) / len(unanswerable) if unanswerable else None
        ),
        "gate_100_percent_unanswerable_abstained": (
            len(abstained_negative) == len(unanswerable) if unanswerable else None
        ),
        "gate_95_percent_conditional_answerable_retained": (
            bool(recalled) and len(retained) / len(recalled) >= 0.95
        ),
    }


def _dominating_pairs(
    negatives: Sequence[RetrievalObservation],
    positives: Sequence[RetrievalObservation],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for negative in negatives:
        for positive in positives:
            negative_margin = negative.vector_margin
            positive_margin = positive.vector_margin
            if negative.strength < positive.strength:
                continue
            if (
                negative_margin is not None
                and positive_margin is not None
                and negative_margin < positive_margin
            ):
                continue
            pairs.append(
                {
                    "negative_case_id": negative.case_id,
                    "negative_strength": negative.strength,
                    "negative_vector_margin": negative_margin,
                    "positive_case_id": positive.case_id,
                    "positive_strength": positive.strength,
                    "positive_vector_margin": positive_margin,
                }
            )
    pairs.sort(
        key=lambda row: (
            -(row["negative_strength"] - row["positive_strength"]),
            row["negative_case_id"],
            row["positive_case_id"],
        )
    )
    return pairs


def _overlap_evidence(
    observations: Sequence[RetrievalObservation], labels: Mapping[str, FitLabel]
) -> dict[str, Any]:
    by_id = {row.case_id: row for row in observations}
    negatives = [
        by_id[key] for key, label in labels.items() if label.label == "unanswerable"
    ]
    positives = [
        by_id[key]
        for key, label in labels.items()
        if label.label == "answerable"
        and set(label.relevant) & set(by_id[key].returned[:5])
    ]
    return {
        "conditional_positive_count": len(positives),
        "negative_count": len(negatives),
        "dominating_negative_positive_pairs": _dominating_pairs(negatives, positives),
    }


def analyze_fit(
    observations: Sequence[RetrievalObservation], labels: FitLabels
) -> dict[str, Any]:
    """Analyze feasibility using a fit-only label capability."""
    by_id = {row.case_id: row for row in observations}
    if set(by_id) != set(labels.by_case):
        raise ValueError("fit observations and fit labels must match exactly")
    negatives = [
        by_id[key]
        for key, label in labels.by_case.items()
        if label.label == "unanswerable"
    ]
    conditional_positives = [
        by_id[key]
        for key, label in labels.by_case.items()
        if label.label == "answerable"
        and set(label.relevant) & set(by_id[key].returned[:5])
    ]
    if not negatives or not conditional_positives:
        raise ValueError("fit requires negative and recalled-positive evidence")
    max_negative = max(row.strength for row in negatives)
    minimum_boundary = math.nextafter(max_negative, math.inf)
    required = math_ceil_95(len(conditional_positives))
    eligible = [
        row for row in conditional_positives if row.strength >= minimum_boundary
    ]
    possible = len(eligible) >= required
    selected_threshold: float | None = None
    if possible:
        retained_strengths = sorted(
            (row.strength for row in conditional_positives), reverse=True
        )[:required]
        selected_threshold = (max_negative + min(retained_strengths)) / 2.0
    domination = _dominating_pairs(negatives, conditional_positives)
    return {
        "possible_under_feature_definition": possible,
        "selected_threshold": selected_threshold,
        "minimum_threshold_for_100pct_fit_negative_abstention": minimum_boundary,
        "max_fit_negative_strength": max_negative,
        "conditional_fit_positive_count": len(conditional_positives),
        "required_fit_positive_retained_for_95pct": required,
        "fit_positive_retained_at_minimum_negative_boundary": len(eligible),
        "fit_positive_retention_at_minimum_negative_boundary": (
            len(eligible) / len(conditional_positives)
        ),
        "positive_strength_range": [
            min(row.strength for row in conditional_positives),
            max(row.strength for row in conditional_positives),
        ],
        "negative_strength_range": [
            min(row.strength for row in negatives),
            max_negative,
        ],
        "dominating_negative_positive_pairs": domination,
    }


def _case_report(
    row: RetrievalObservation, label: FitLabel, threshold: float
) -> dict[str, Any]:
    relevant_top5 = bool(set(label.relevant) & set(row.returned[:5]))
    result: dict[str, Any] = {
        "id": row.case_id,
        "label": label.label,
        "relevant": list(label.relevant),
        "returned_top_5": list(row.returned[:5]),
        "relevant_in_top_5": relevant_top5,
        "strength": row.strength,
        "retained_at_fit_boundary": row.strength >= threshold,
        "evidence": row.evidence,
        "vector_profile": row.vector_profile,
    }
    if row.lexical_returned is not None:
        result["fts_only_returned"] = list(row.lexical_returned)
    return result


def _profile_identity(profile: str) -> dict[str, Any]:
    return {
        "profile_id": FTS_PROFILE_ID if profile == "fts" else HYBRID_PROFILE_ID,
        "mode": "fts_only" if profile == "fts" else "hybrid",
        "model_id": None if profile == "fts" else "BAAI/bge-m3",
        "dimension": 0 if profile == "fts" else 1024,
        "vector_backend": None if profile == "fts" else "sqlite_vec_native",
        "vector_metric": None if profile == "fts" else "cosine_distance",
        "fts_backend": "sqlite_fts5_porter_unicode61",
        "candidate_limit": 40,
        "result_k": 5,
        "rrf_k": 60,
    }


def evaluate_profile(
    profile: str,
    *,
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    if profile not in {"fts", "hybrid"}:
        raise ValueError("profile must be fts or hybrid")
    dataset, split = load_inputs(fixture_dir)
    separation = separation_evidence(dataset)
    e0_evidence = sealed_e0_evidence()
    if not e0_evidence["diff_empty"]:
        raise RuntimeError("sealed E0 artifacts changed during E6")
    runtime_evidence = public_runtime_evidence()
    if not runtime_evidence["diff_empty"]:
        raise RuntimeError("E6 blocker evidence must not ship public runtime policy")
    by_case = {row["id"]: row for row in dataset["cases"]}
    saved = {
        key: os.environ.get(key)
        for key in (
            "HAUNT_HOME",
            "HAUNT_FTS_ONLY",
            "HAUNT_OFFLINE",
            "HAUNT_EMBED_MODEL",
            "HAUNT_MODEL_CACHE",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        )
    }
    cache_evidence: dict[str, Any] | None = None
    if profile == "hybrid":
        if model_cache is None:
            raise RuntimeError("hybrid reproduction requires --model-cache")
        cache_evidence = verify_local_hybrid_cache(model_cache)

    network_attempts: list[str] = []
    with TemporaryDirectory(prefix=f"haunt-e6-{profile}-") as home:
        try:
            os.environ["HAUNT_HOME"] = home
            os.environ["OPENAI_API_KEY"] = "e6-ambient-fake-key"
            os.environ["ANTHROPIC_API_KEY"] = "e6-ambient-fake-key"
            os.environ["HF_TOKEN"] = "e6-ambient-fake-key"
            os.environ["HUGGING_FACE_HUB_TOKEN"] = "e6-ambient-fake-key"
            os.environ["HAUNT_EMBED_MODEL"] = "BAAI/bge-m3"
            if profile == "fts":
                os.environ["HAUNT_FTS_ONLY"] = "1"
                os.environ["HAUNT_OFFLINE"] = "1"
                os.environ.pop("HAUNT_MODEL_CACHE", None)
            else:
                os.environ.pop("HAUNT_FTS_ONLY", None)
                os.environ.pop("HAUNT_OFFLINE", None)
                os.environ["HAUNT_MODEL_CACHE"] = str(model_cache)
            embed.reset()
            with NetworkDeny() as network, _DeterministicFixtureIds():
                memory_ids: dict[str, str] = {}
                with Store("e6-calibration") as store:
                    for row in dataset["records"]:
                        observed = store.observe(
                            str(row["content"]),
                            origin="abstention-eval",
                            channel="python",
                            defer_embedding=True,
                        )
                        memory_ids[str(row["id"])] = observed.memory_id
                    if profile == "hybrid":
                        state = embed.state()
                        if (
                            not state.available
                            or state.fallback
                            or state.model_id != "BAAI/bge-m3"
                            or state.dim != 1024
                            or state.backend != "onnx"
                        ):
                            raise RuntimeError(
                                f"pinned hybrid model mismatch: {state!r}"
                            )
                        processed = store.process_embedding_jobs(limit=100)
                        if processed.get("processed") != len(dataset["records"]):
                            raise RuntimeError(
                                f"hybrid fixture embedding failed: {processed}"
                            )
                        if not store.vec_ok():
                            raise RuntimeError(
                                "pinned hybrid requires native sqlite-vec"
                            )

                # Fit observations and the fit-only capability are fully
                # analyzed before any held-out label object is created.
                with open_existing_readonly("e6-calibration") as read_store:
                    fit_observations = [
                        _observe_case(
                            by_case[case_id],
                            memory_ids,
                            profile=profile,
                            store=read_store,
                        )
                        for case_id in split["fit"][profile]
                    ]
                    fit_labels = _fit_labels(dataset, split, profile)
                    fit_analysis = analyze_fit(fit_observations, fit_labels)
                    boundary = float(
                        fit_analysis["selected_threshold"]
                        if fit_analysis["selected_threshold"] is not None
                        else fit_analysis[
                            "minimum_threshold_for_100pct_fit_negative_abstention"
                        ]
                    )
                    fit_metrics = _score(fit_observations, fit_labels.by_case, boundary)

                    held_observations = [
                        _observe_case(
                            by_case[case_id],
                            memory_ids,
                            profile=profile,
                            store=read_store,
                        )
                        for case_id in split["held_out"][profile]
                    ]
                    held_labels = _labels_for_ids(dataset, split["held_out"][profile])
                    held_metrics = _score(held_observations, held_labels, boundary)
                network_attempts = list(network.attempts)
                if network_attempts:
                    raise RuntimeError(
                        f"network access was attempted: {network_attempts}"
                    )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            embed.reset()

    semantic_ids = {
        case_id
        for case_id in split["held_out"][profile]
        if by_case[case_id].get("semantic_paraphrase") is True
    }
    semantic_labels = {
        key: value for key, value in held_labels.items() if key in semantic_ids
    }
    semantic_observations = [
        row for row in held_observations if row.case_id in semantic_ids
    ]
    semantic_metrics = (
        _score(semantic_observations, semantic_labels, boundary)
        if semantic_labels
        else None
    )
    held_overlap = _overlap_evidence(held_observations, held_labels)
    all_observations = [*fit_observations, *held_observations]
    batching_evidence = {
        "candidate_scope": "fixed_top_5",
        "maximum_candidate_count_observed": max(
            row.evidence["candidate_count"] for row in all_observations
        ),
        "maximum_coverage_sql_statements_per_recall_observed": max(
            row.evidence["diagnostics"]["coverage_sql_statement_count"]
            for row in all_observations
        ),
        "query_tokens_are_nfkc_casefold_deduplicated": True,
        "rrf_score_read_by_feature": False,
    }
    possible = bool(fit_analysis["possible_under_feature_definition"])
    return {
        "schema_version": 1,
        "report_id": f"haunt-abstention-feasibility-{profile}-v1",
        "status": "calibratable" if possible else "blocked",
        "runtime_policy_shipped": False,
        "dataset_canonical_sha256": canonical_hash(dataset),
        "split_canonical_sha256": canonical_hash(split),
        "feature_definition": FEATURE_DEFINITION,
        "feature_definition_sha256": canonical_hash(FEATURE_DEFINITION),
        "separation_from_e0": separation,
        "manual_semantic_audit": split["manual_semantic_audit"],
        "sealed_e0": e0_evidence,
        "public_runtime_policy": runtime_evidence,
        "profile": _profile_identity(profile),
        "fixture_physical_id_strategy": "sequential_uuid_for_reproducible_tie_breaks",
        "network_proof": {
            "deny_harness": (
                "socket_constructor_connect_connect_ex_send_sendall_sendto_"
                "create_connection_create_server_dns_http"
            ),
            "ambient_fake_keys": [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            ],
            "haunt_offline": profile == "fts",
            "model_cache_explicit": profile == "hybrid",
            "attempts": network_attempts,
        },
        "local_model_cache": cache_evidence,
        "evidence_query_batching": batching_evidence,
        "fit": {
            "labels_available_to_fit": sorted(fit_labels.by_case),
            "held_out_labels_available_to_fit": [],
            "analysis": fit_analysis,
            "diagnostic_boundary_from_fit_only": boundary,
            "metrics_at_diagnostic_boundary": fit_metrics,
            "cases": [
                _case_report(row, fit_labels.by_case[row.case_id], boundary)
                for row in fit_observations
            ],
        },
        "held_out": {
            "labels_loaded_after_fit_analysis": True,
            "metrics_at_fit_only_diagnostic_boundary": held_metrics,
            "pure_semantic_paraphrase_metrics": semantic_metrics,
            "evidence_overlap": held_overlap,
            "cases": [
                _case_report(row, held_labels[row.case_id], boundary)
                for row in held_observations
            ],
        },
        **(
            {
                "blocker": {
                    "code": "raw_evidence_fit_gate_unsatisfied",
                    "required": {
                        "unanswerable_abstention": 1.0,
                        "conditional_answerable_retention": 0.95,
                    },
                    "finding": (
                        "Approved raw retrieval evidence cannot separate related "
                        "but absent attributes from semantic answers in the pinned "
                        "hybrid fit cohort."
                    ),
                    "dominating_negative_positive_pairs": fit_analysis[
                        "dominating_negative_positive_pairs"
                    ],
                    "unblock_choices": [
                        "predeclare and review a new raw retrieval feature and a new E6 version",
                        "explicitly amend the contract to permit a reader or cross-encoder",
                        "leave calibrated hybrid abstention unshipped",
                    ],
                    "forbidden_shortcuts": [
                        "remove_or_relabel_hard_negatives",
                        "pad_the_positive_denominator_with_lexical_cases",
                        "threshold_or_normalize_rrf",
                        "fit_or_tune_on_held_out_labels",
                        "silently_fall_back_to_fts",
                    ],
                }
            }
            if not possible
            else {}
        ),
    }
