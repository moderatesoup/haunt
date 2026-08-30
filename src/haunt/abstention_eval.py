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
from typing import Any, Callable, Mapping, Sequence

from haunt import embed
from haunt.recall import Hit, recall
from haunt.rerank import RERANK_ENABLED_ENV
from haunt.store import ReadOnlyStore, Store, open_existing_readonly

SCHEMA_VERSION = 1
FEATURE_IMPLEMENTATION = "haunt-abstention-feasibility-v1"
FTS_PROFILE_ID = "fts5-porter-unicode61-top40-k5-v1"
HYBRID_PROFILE_ID = "bge-m3-onnx-1024-native-cosine-fts5-rrf60-top40-k5-v1"
HYBRID_ARTIFACT_MANIFEST_ID = "haunt-bge-m3-onnx-split-f8425123-v1"
HYBRID_ARTIFACT_MANIFEST_SHA256 = (
    "d767f2f4a020b36e5a1d26636460af6cc5981836258c6f38cc677de9cab143a2"
)
DATASET_MANIFEST_ID = "haunt-abstention-e6-composite-v1"
DATASET_MANIFEST_SHA256 = (
    "8c36449a8f38fed8daf86fe67e716e3ee53d700c29d4ba95fc4aa4268bb9c64e"
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "abstention_eval" / "v1"
DEFAULT_E0_CORPUS = ROOT / "tests" / "fixtures" / "retrieval_eval" / "corpus.json"
# The hybrid artifact manifest gates embed initialization, so it has to be
# reachable from an installed wheel, where ROOT points outside the package and
# tests/ was never shipped. It lives beside the code and stays E6 evidence:
# one file, hash-pinned, read by both the runtime check and the E6 harness.
HYBRID_MANIFEST_DIR = Path(__file__).resolve().parent / "data"
E6_EVIDENCE_PATHS = (
    "src/haunt/abstention_eval.py",
    "src/haunt/data/hybrid-model-manifest.json",
    "scripts/reproduce_abstention_eval.py",
    "scripts/benchmark_abstention_evidence.py",
    "tests/fixtures/abstention_eval",
    "tests/test_abstention_feasibility.py",
)
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
AuditHook = Callable[[str, Mapping[str, Any]], None]

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


def _emit_audit(
    hook: AuditHook | None, event: str, details: Mapping[str, Any] | None = None
) -> None:
    if hook is not None:
        hook(event, details or {})


def _read_json_component(
    fixture_dir: Path,
    filename: str,
    *,
    component: str,
    audit_hook: AuditHook | None,
) -> dict[str, Any]:
    path = fixture_dir / filename
    _emit_audit(
        audit_hook,
        f"before_open:{component}",
        {"filename": filename},
    )
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{component} must be a JSON object")
    return value


def load_inputs(
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    *,
    audit_hook: AuditHook | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load records, unlabeled queries, and split membership only.

    Neither label file nor the composite manifest is opened here. In
    particular, callers can complete fit retrieval and threshold analysis
    without the held-label file being readable.
    """
    record_source = _read_json_component(
        fixture_dir,
        "records.json",
        component="records",
        audit_hook=audit_hook,
    )
    query_source = _read_json_component(
        fixture_dir,
        "queries.json",
        component="queries",
        audit_hook=audit_hook,
    )
    split = _read_json_component(
        fixture_dir,
        "split.json",
        component="split",
        audit_hook=audit_hook,
    )
    for name, source in (("records", record_source), ("queries", query_source)):
        if source.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported E6 {name} schema")
    if split.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E6 split schema")
    if record_source.get("dataset_id") != query_source.get("dataset_id"):
        raise ValueError("E6 records/query dataset IDs differ")
    records = record_source.get("records")
    cases = query_source.get("cases")
    if not isinstance(records, list) or not isinstance(cases, list):
        raise ValueError("E6 inputs require records and unlabeled cases")
    if any("label" in row or "relevant" in row for row in cases):
        raise ValueError("queries.json must be physically unlabeled")
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
        raise ValueError("E6 split must partition every unlabeled case exactly once")
    by_case = {row["id"]: row for row in cases}
    for phase in ("fit", "held_out"):
        for profile in ("fts", "hybrid"):
            if len(split[phase][profile]) != 40:
                raise ValueError(f"{phase}/{profile} requires exactly 40 cases")
            if any(
                by_case[case_id]["profile"] != profile
                for case_id in split[phase][profile]
            ):
                raise ValueError("profile/split mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": record_source["dataset_id"],
        "records": records,
        "cases": cases,
    }, split


def _load_phase_labels(
    fixture_dir: Path,
    *,
    phase: str,
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    audit_hook: AuditHook | None,
) -> dict[str, FitLabel]:
    if phase not in {"fit", "held_out"}:
        raise ValueError("invalid E6 label phase")
    component = "fit_labels" if phase == "fit" else "held_labels"
    filename = "fit-labels.json" if phase == "fit" else "held-labels.json"
    source = _read_json_component(
        fixture_dir,
        filename,
        component=component,
        audit_hook=audit_hook,
    )
    if (
        source.get("schema_version") != SCHEMA_VERSION
        or source.get("dataset_id") != dataset["dataset_id"]
        or source.get("phase") != phase
        or not isinstance(source.get("labels"), list)
    ):
        raise ValueError(f"invalid E6 {phase} label file")
    rows = source["labels"]
    ids = [row.get("id") for row in rows]
    expected = {
        case_id for profile in ("fts", "hybrid") for case_id in split[phase][profile]
    }
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"{phase} labels must exactly match split membership")
    by_query = {row["id"]: row for row in dataset["cases"]}
    result: dict[str, FitLabel] = {}
    for row in rows:
        label = row.get("label")
        relevant = row.get("relevant")
        if label not in {"answerable", "unanswerable"} or not isinstance(
            relevant, list
        ):
            raise ValueError(f"invalid {phase} label row")
        if not all(isinstance(value, str) for value in relevant):
            raise ValueError(f"invalid {phase} relevant IDs")
        if (label == "answerable") != bool(relevant):
            raise ValueError(f"{phase} answerability/relevant IDs disagree")
        query = by_query[row["id"]]
        if label == "answerable" and query["profile"] == "hybrid":
            if (
                query.get("semantic_paraphrase") is not True
                or query.get("required_lexical_overlap") is not False
            ):
                raise ValueError("every hybrid answerable must be a pure paraphrase")
        result[str(row["id"])] = FitLabel(
            label=str(label), relevant=tuple(str(value) for value in relevant)
        )
    for profile in ("fts", "hybrid"):
        selected = [result[case_id] for case_id in split[phase][profile]]
        if sum(row.label == "answerable" for row in selected) < 20:
            raise ValueError(f"{phase}/{profile} requires 20 answerable cases")
        if sum(row.label == "unanswerable" for row in selected) < 20:
            raise ValueError(f"{phase}/{profile} requires 20 unanswerable cases")
    return result


def _phase_label_source(
    phase: str, labels: Mapping[str, FitLabel], *, dataset_id: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "phase": phase,
        "labels": [
            {
                "id": case_id,
                "label": label.label,
                "relevant": list(label.relevant),
            }
            for case_id, label in labels.items()
        ],
    }


def _composite_dataset(
    dataset: Mapping[str, Any],
    fit_labels: Mapping[str, FitLabel],
    held_labels: Mapping[str, FitLabel],
) -> dict[str, Any]:
    labels = {**fit_labels, **held_labels}
    if set(labels) != {row["id"] for row in dataset["cases"]}:
        raise ValueError("fit and held labels do not cover the composite dataset")
    return {
        "schema_version": dataset["schema_version"],
        "dataset_id": dataset["dataset_id"],
        "records": dataset["records"],
        "cases": [
            {
                **row,
                "label": labels[row["id"]].label,
                "relevant": list(labels[row["id"]].relevant),
            }
            for row in dataset["cases"]
        ],
    }


def _verify_dataset_manifest_after_scoring(
    fixture_dir: Path,
    *,
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    fit_labels: Mapping[str, FitLabel],
    held_labels: Mapping[str, FitLabel],
    audit_hook: AuditHook | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json_component(
        fixture_dir,
        "dataset-manifest.json",
        component="dataset_manifest",
        audit_hook=audit_hook,
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_id") != DATASET_MANIFEST_ID
        or canonical_hash(manifest) != DATASET_MANIFEST_SHA256
        or manifest.get("verification_phase") != "after_held_out_scoring"
    ):
        raise ValueError("invalid E6 composite dataset manifest")
    components = {
        "records": {
            "schema_version": dataset["schema_version"],
            "dataset_id": dataset["dataset_id"],
            "records": dataset["records"],
        },
        "queries": {
            "schema_version": dataset["schema_version"],
            "dataset_id": dataset["dataset_id"],
            "cases": dataset["cases"],
        },
        "split": split,
        "fit_labels": _phase_label_source(
            "fit", fit_labels, dataset_id=str(dataset["dataset_id"])
        ),
        "held_labels": _phase_label_source(
            "held_out", held_labels, dataset_id=str(dataset["dataset_id"])
        ),
    }
    component_hashes = {
        name: canonical_hash(value) for name, value in components.items()
    }
    expected_components = manifest.get("components")
    if not isinstance(expected_components, dict):
        raise ValueError("E6 manifest has no component lock")
    for name, digest in component_hashes.items():
        expected = expected_components.get(name, {}).get("canonical_sha256")
        if expected != digest:
            raise ValueError(
                f"E6 manifest {name} hash mismatch: expected={expected} actual={digest}"
            )
    composite = _composite_dataset(dataset, fit_labels, held_labels)
    composite_hash = canonical_hash(composite)
    expected_composite = manifest.get("composite_dataset", {}).get("canonical_sha256")
    if expected_composite != composite_hash:
        raise ValueError(
            "E6 composite dataset hash mismatch: "
            f"expected={expected_composite} actual={composite_hash}"
        )
    return composite, {
        "manifest_id": manifest["manifest_id"],
        "manifest_canonical_sha256": canonical_hash(manifest),
        "verified_after_held_out_scoring": True,
        "component_canonical_sha256": component_hashes,
        "composite_dataset_canonical_sha256": composite_hash,
    }


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


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"E6 isolation evidence requires git: {exc}") from exc
    if completed.returncode != 0:
        # A guard that cannot read history cannot certify isolation either.
        raise RuntimeError(
            "E6 isolation evidence requires git metadata: "
            f"git {args[0]}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _uncommitted(paths: Sequence[str]) -> list[str]:
    """Paths under ``paths`` that differ from HEAD, untracked additions included."""
    changed = _git("diff", "--name-only", "HEAD", "--", *paths)
    added = _git("ls-files", "--others", "--exclude-standard", "--", *paths)
    lines = (*changed.splitlines(), *added.splitlines())
    return sorted({line for line in lines if line})


def _e6_attributed_diff(paths: Sequence[str]) -> dict[str, Any]:
    """Diff of ``paths`` carried by E6's own change-set.

    Diffing the whole tree against a pinned revision cannot separate an E6
    runtime-policy change from unrelated work that shares the branch, so
    attribution is per commit: only commits carrying E6's evidence surface,
    plus uncommitted edits to that surface, count as E6's.

    Working-tree edits that no commit can attribute are reported separately
    in uncommitted_paths rather than folded into diff_empty, so a caller
    never reads an empty diff as a diff that was read and found empty.
    """
    # A shallow clone grafts the history away and its one commit adds every
    # file, so attribute nothing to it and say so rather than overclaim.
    attributable = _git("rev-parse", "--is-shallow-repository").strip() != "true"
    commits = (
        _git("rev-list", "HEAD", "--", *E6_EVIDENCE_PATHS).split()
        if attributable
        else []
    )
    diff = "".join(
        _git("show", "--format=", commit, "--", *paths) for commit in commits
    )
    uncommitted = _uncommitted(paths)
    # Uncommitted work carries no commit to attribute it by. Dirty E6 evidence
    # claims the whole working-tree diff for E6; dirt without it is
    # unattributable concurrent work, which must not fail this gate -- but
    # must not read as a checked-and-empty diff either, because nothing here
    # looked at it. sealed_e0_evidence has byte_mismatches for that; every
    # caller of this helper now gets the same leg.
    e6_dirty = bool(_uncommitted(E6_EVIDENCE_PATHS))
    if e6_dirty:
        diff += _git("diff", "HEAD", "--", *paths)
    return {
        "history_attributable": attributable,
        "e6_commits": commits,
        "paths": list(paths),
        "diff": diff,
        "diff_empty": not diff,
        "uncommitted_paths": uncommitted,
        "uncommitted_unattributed": bool(uncommitted) and not e6_dirty,
    }


def sealed_e0_evidence() -> dict[str, Any]:
    tracked = _git(
        "ls-tree", "-r", "--name-only", "HEAD", "--", *E0_SEALED_PATHS
    ).splitlines()
    byte_mismatches: list[str] = []
    byte_sha256: dict[str, str] = {}
    for relative in tracked:
        committed_bytes = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        working_bytes = (ROOT / relative).read_bytes()
        if working_bytes != committed_bytes:
            byte_mismatches.append(relative)
        byte_sha256[relative] = hashlib.sha256(working_bytes).hexdigest()
    return {
        **_e6_attributed_diff(E0_SEALED_PATHS),
        "tracked_file_count": len(tracked),
        "byte_mismatches": byte_mismatches,
        "working_tree_byte_sha256": byte_sha256,
    }


def public_runtime_evidence() -> dict[str, Any]:
    return _e6_attributed_diff(PUBLIC_RUNTIME_PATHS)


def verify_local_hybrid_cache(
    cache_root: Path,
    *,
    manifest_dir: Path = HYBRID_MANIFEST_DIR,
    audit_hook: AuditHook | None = None,
    hash_max_bytes: int | None = None,
) -> dict[str, Any]:
    """Require the committed BGE-M3 artifact identity before embed init.

    Every manifested file is checked for presence, type and exact size.
    hash_max_bytes caps content hashing: a file whose manifest size exceeds
    it is reported with sha256 None and stands on its size alone. The default
    hashes every file, which is the evidence E6 records.

    manifest_dir defaults to the packaged manifest and exists so a caller can
    point at a tampered copy; a relocated manifest still has to carry the
    pinned canonical hash, so it cannot bless a different artifact identity.
    """
    manifest = _read_json_component(
        manifest_dir,
        "hybrid-model-manifest.json",
        component="hybrid_model_manifest",
        audit_hook=audit_hook,
    )
    if canonical_hash(manifest) != HYBRID_ARTIFACT_MANIFEST_SHA256:
        raise RuntimeError("invalid committed hybrid artifact manifest identity")
    policy = manifest.get("variant_policy")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_id") != HYBRID_ARTIFACT_MANIFEST_ID
        or manifest.get("model_id") != "BAAI/bge-m3"
        or manifest.get("dimension") != 1024
        or manifest.get("backend") != "onnx"
        or manifest.get("selected_variant") != "nonquantized_split_onnx"
        or not isinstance(policy, dict)
        or policy.get("allowed_variants") != ["nonquantized_split_onnx"]
        or policy.get("quantized_variant_allowed") is not False
        or policy.get("external_data_sidecar_required") is not True
        or not isinstance(files, list)
    ):
        raise RuntimeError("invalid committed hybrid artifact manifest")
    selected_dir_value = policy.get("selected_directory")
    forbidden_values = policy.get("forbidden_relative_paths")
    if not isinstance(selected_dir_value, str) or not isinstance(
        forbidden_values, list
    ):
        raise RuntimeError("hybrid artifact manifest has invalid variant policy")
    expected: dict[str, dict[str, Any]] = {}
    for row in files:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("relative_path"), str)
            or not isinstance(row.get("size"), int)
            or not isinstance(row.get("sha256"), str)
        ):
            raise RuntimeError("hybrid artifact manifest has invalid file entry")
        relative = str(row["relative_path"])
        if (
            relative in expected
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("hybrid artifact manifest has unsafe/duplicate path")
        expected[relative] = row
    sidecar = "BAAI-bge-m3/onnx/model.onnx_data"
    if sidecar not in expected:
        raise RuntimeError("hybrid artifact manifest omits required ONNX sidecar")

    problems: list[str] = []
    selected_dir = cache_root / selected_dir_value
    expected_selected = {
        relative
        for relative in expected
        if str(Path(relative).parent) == selected_dir_value
    }
    actual_selected = (
        {
            str(path.relative_to(cache_root))
            for path in selected_dir.rglob("*")
            if path.is_file()
        }
        if selected_dir.is_dir()
        else set()
    )
    extra = sorted(actual_selected - expected_selected)
    if extra and policy.get("reject_unlisted_files_in_selected_directory") is True:
        problems.append(f"extra_selected={extra}")
    forbidden = [
        str(value)
        for value in forbidden_values
        if isinstance(value, str) and (cache_root / value).exists()
    ]
    if forbidden:
        problems.append(f"variant_mismatch_forbidden={sorted(forbidden)}")

    actual_rows: list[dict[str, Any]] = []
    for relative, row in expected.items():
        path = cache_root / relative
        if not path.exists() or not path.is_file():
            marker = "missing_sidecar" if relative == sidecar else "missing"
            problems.append(f"{marker}={relative}")
            continue
        if path.is_symlink():
            problems.append(f"symlink={relative}")
            continue
        size = path.stat().st_size
        if size == 0:
            problems.append(f"zero_byte={relative}")
        if size != row["size"]:
            problems.append(
                f"size_mismatch={relative}:expected={row['size']}:actual={size}"
            )
        actual_rows.append(
            {
                "relative_path": relative,
                "size": size,
                "expected_sha256": row["sha256"],
                "path": path,
            }
        )
    if problems:
        raise RuntimeError(
            "hybrid cache artifact preflight mismatch: " + "; ".join(problems)
        )

    verified: list[dict[str, Any]] = []
    for row in actual_rows:
        if hash_max_bytes is not None and row["size"] > hash_max_bytes:
            verified.append(
                {
                    "relative_path": row["relative_path"],
                    "size": row["size"],
                    "sha256": None,
                }
            )
            continue
        digest = _file_hash(row["path"])
        if digest != row["expected_sha256"]:
            raise RuntimeError(
                "hybrid cache artifact hash mismatch: "
                f"{row['relative_path']}:expected={row['expected_sha256']}:actual={digest}"
            )
        verified.append(
            {
                "relative_path": row["relative_path"],
                "size": row["size"],
                "sha256": digest,
            }
        )
    return {
        "cache_root_supplied": True,
        "matched_manifest_id": manifest["manifest_id"],
        "manifest_canonical_sha256": canonical_hash(manifest),
        "model_id": manifest["model_id"],
        "dimension": manifest["dimension"],
        "backend": manifest["backend"],
        "selected_variant": manifest["selected_variant"],
        "variant_policy": policy,
        "files": verified,
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
    labels: Mapping[str, FitLabel], split: Mapping[str, Any], profile: str
) -> FitLabels:
    return FitLabels({case_id: labels[case_id] for case_id in split["fit"][profile]})


def _labels_for_ids(
    labels: Mapping[str, FitLabel], case_ids: Sequence[str]
) -> dict[str, FitLabel]:
    return {case_id: labels[case_id] for case_id in case_ids}


def _observe_case(
    case: Mapping[str, Any],
    label: FitLabel,
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
        relevant = set(label.relevant)
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
    if any(not math.isfinite(row.strength) for row in observations):
        raise ValueError("fit evidence strengths must be finite")
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
    if not math.isfinite(minimum_boundary):
        raise ValueError("no finite threshold exists above the strongest negative")
    required = math_ceil_95(len(conditional_positives))
    eligible = [
        row for row in conditional_positives if row.strength >= minimum_boundary
    ]
    possible = len(eligible) >= required
    selected_threshold: float | None = None
    selected_threshold_strategy: str | None = None
    fit_gate_validation: dict[str, Any] | None = None
    if possible:
        # The smallest representable float strictly above the strongest
        # negative is the only boundary we need. A midpoint can round back to
        # max_negative when the positive is exactly one ULP above it, which
        # would retain the negative because scoring uses strength >= threshold.
        selected_threshold = minimum_boundary
        selected_threshold_strategy = "nextafter_max_negative_toward_positive_infinity"
        fit_gate_validation = _score(observations, labels.by_case, selected_threshold)
        if not (
            selected_threshold > max_negative
            and fit_gate_validation["gate_100_percent_unanswerable_abstained"]
            and fit_gate_validation["gate_95_percent_conditional_answerable_retained"]
        ):
            raise RuntimeError("fit threshold failed its predeclared gates")
    domination = _dominating_pairs(negatives, conditional_positives)
    return {
        "possible_under_feature_definition": possible,
        "selected_threshold": selected_threshold,
        "selected_threshold_strategy": selected_threshold_strategy,
        "selected_threshold_fit_gate_validation": fit_gate_validation,
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
    audit_hook: AuditHook | None = None,
) -> dict[str, Any]:
    if profile not in {"fts", "hybrid"}:
        raise ValueError("profile must be fts or hybrid")
    audit_events: list[dict[str, Any]] = []

    def audit(event: str, details: Mapping[str, Any]) -> None:
        recorded = dict(details)
        if "fit_report" in recorded:
            recorded = {
                "fit_report_canonical_sha256": canonical_hash(recorded["fit_report"]),
                "boundary": recorded["fit_report"]["boundary"],
            }
        audit_events.append({"event": event, "details": recorded})
        _emit_audit(audit_hook, event, details)

    dataset, split = load_inputs(fixture_dir, audit_hook=audit)
    fit_phase_labels = _load_phase_labels(
        fixture_dir,
        phase="fit",
        dataset=dataset,
        split=split,
        audit_hook=audit,
    )
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
            RERANK_ENABLED_ENV,
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
        cache_evidence = verify_local_hybrid_cache(model_cache, audit_hook=audit)

    network_attempts: list[str] = []
    with TemporaryDirectory(prefix=f"haunt-e6-{profile}-") as home:
        try:
            os.environ["HAUNT_HOME"] = home
            os.environ["OPENAI_API_KEY"] = "e6-ambient-fake-key"
            os.environ["ANTHROPIC_API_KEY"] = "e6-ambient-fake-key"
            os.environ["HF_TOKEN"] = "e6-ambient-fake-key"
            os.environ["HUGGING_FACE_HUB_TOKEN"] = "e6-ambient-fake-key"
            os.environ["HAUNT_EMBED_MODEL"] = "BAAI/bge-m3"
            # recall() honours HAUNT_RERANK_ENABLED. Left set, this harness
            # would measure reranked retrieval and write reranked final_rank
            # into committed evidence.
            os.environ.pop(RERANK_ENABLED_ENV, None)
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
                            fit_phase_labels[case_id],
                            memory_ids,
                            profile=profile,
                            store=read_store,
                        )
                        for case_id in split["fit"][profile]
                    ]
                    fit_labels = _fit_labels(fit_phase_labels, split, profile)
                    fit_analysis = analyze_fit(fit_observations, fit_labels)
                    boundary = float(
                        fit_analysis["selected_threshold"]
                        if fit_analysis["selected_threshold"] is not None
                        else fit_analysis[
                            "minimum_threshold_for_100pct_fit_negative_abstention"
                        ]
                    )
                    fit_metrics = _score(fit_observations, fit_labels.by_case, boundary)
                    fit_boundary_report = {
                        "analysis": fit_analysis,
                        "boundary": boundary,
                        "metrics": fit_metrics,
                    }
                    audit(
                        "fit_boundary_complete",
                        {"fit_report": fit_boundary_report},
                    )

                    # This is the first operation permitted to open or parse
                    # held labels. Fit analysis and its boundary are immutable.
                    held_phase_labels = _load_phase_labels(
                        fixture_dir,
                        phase="held_out",
                        dataset=dataset,
                        split=split,
                        audit_hook=audit,
                    )

                    held_observations = [
                        _observe_case(
                            by_case[case_id],
                            held_phase_labels[case_id],
                            memory_ids,
                            profile=profile,
                            store=read_store,
                        )
                        for case_id in split["held_out"][profile]
                    ]
                    held_labels = _labels_for_ids(
                        held_phase_labels, split["held_out"][profile]
                    )
                    held_metrics = _score(held_observations, held_labels, boundary)
                    composite_dataset, dataset_manifest_evidence = (
                        _verify_dataset_manifest_after_scoring(
                            fixture_dir,
                            dataset=dataset,
                            split=split,
                            fit_labels=fit_phase_labels,
                            held_labels=held_phase_labels,
                            audit_hook=audit,
                        )
                    )
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
        "dataset_canonical_sha256": canonical_hash(composite_dataset),
        "split_canonical_sha256": canonical_hash(split),
        "dataset_manifest": dataset_manifest_evidence,
        "label_access_audit": {
            "held_labels_physical_file": "held-labels.json",
            "fit_boundary_precedes_held_label_open": (
                next(
                    index
                    for index, row in enumerate(audit_events)
                    if row["event"] == "fit_boundary_complete"
                )
                < next(
                    index
                    for index, row in enumerate(audit_events)
                    if row["event"] == "before_open:held_labels"
                )
            ),
            "events": audit_events,
        },
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
