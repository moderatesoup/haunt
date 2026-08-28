"""E6 abstention feasibility is frozen as evidence, not shipped policy."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import socket
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from haunt import abstention_eval, embed
from haunt.abstention_eval import (
    FEATURE_DEFINITION,
    FitLabels,
    _coverage_many,
    _evidence,
    analyze_fit,
    canonical_hash,
    classify_vector_profile,
    evaluate_profile,
    load_inputs,
    sealed_e0_evidence,
    separation_evidence,
)
from haunt.recall import Hit
from haunt.rerank import RERANK_ENABLED_ENV
from haunt.store import Store, open_existing_readonly

FIXTURE = Path(__file__).parent / "fixtures" / "abstention_eval" / "v1"
REPORTS = FIXTURE / "reports"


def _hit(
    memory_id: str,
    *,
    score: float,
    fts_rank: int | None,
    vec_rank: int | None,
    vec_distance: float | None,
    vec_metric: str | None,
) -> Hit:
    return Hit(
        memory_id=memory_id,
        event_id="event",
        score=score,
        tier="episodic",
        content="backup algorithm",
        role="user",
        event_time="2026-08-26T00:00:00.000000Z",
        valid_from="2026-08-26T00:00:00.000000Z",
        valid_to=None,
        tool_name=None,
        fts_rank=fts_rank,
        fts_rank_raw=-1.5 if fts_rank is not None else None,
        vec_rank=vec_rank,
        vec_distance=vec_distance,
        vec_metric=vec_metric,
        final_rank=1,
    )


def _sparse_manifest_cache(root: Path) -> dict:
    manifest = json.loads((FIXTURE / "hybrid-model-manifest.json").read_text("utf-8"))
    for row in manifest["files"]:
        path = root / row["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(row["size"])
    return manifest


def test_dataset_split_is_large_predeclared_and_separate_from_e0():
    initial_events = []

    def initial_audit(event, details):
        initial_events.append((event, dict(details)))

    dataset, split = load_inputs(FIXTURE, audit_hook=initial_audit)
    assert [event for event, _ in initial_events] == [
        "before_open:records",
        "before_open:queries",
        "before_open:split",
    ]
    assert not (FIXTURE / "dataset.json").exists()
    assert (FIXTURE / "fit-labels.json").is_file()
    assert (FIXTURE / "held-labels.json").is_file()
    by_case = {row["id"]: row for row in dataset["cases"]}
    fit_labels = abstention_eval._load_phase_labels(
        FIXTURE,
        phase="fit",
        dataset=dataset,
        split=split,
        audit_hook=None,
    )
    held_labels = abstention_eval._load_phase_labels(
        FIXTURE,
        phase="held_out",
        dataset=dataset,
        split=split,
        audit_hook=None,
    )
    all_labels = {**fit_labels, **held_labels}

    assert len(dataset["records"]) == 40
    assert len(dataset["cases"]) == 160
    assert all("label" not in row and "relevant" not in row for row in dataset["cases"])
    for phase in ("fit", "held_out"):
        for profile in ("fts", "hybrid"):
            rows = [all_labels[key] for key in split[phase][profile]]
            assert sum(row.label == "answerable" for row in rows) == 20
            assert sum(row.label == "unanswerable" for row in rows) == 20
    hybrid_positives = [
        by_case[case_id]
        for case_id, label in all_labels.items()
        if by_case[case_id]["profile"] == "hybrid" and label.label == "answerable"
    ]
    assert len(hybrid_positives) == 40
    assert all(row["semantic_paraphrase"] is True for row in hybrid_positives)
    assert all(row["required_lexical_overlap"] is False for row in hybrid_positives)
    assert not any(separation_evidence(dataset)["overlaps"].values())
    assert split["manual_semantic_audit"]["reviewed_on"] == "2026-08-26"
    composite, manifest = abstention_eval._verify_dataset_manifest_after_scoring(
        FIXTURE,
        dataset=dataset,
        split=split,
        fit_labels=fit_labels,
        held_labels=held_labels,
        audit_hook=None,
    )
    assert canonical_hash(composite) == (
        "8119f4508d3582bc665a5a0117940c6eeca593de56f33999563ddf188846264c"
    )
    assert manifest["verified_after_held_out_scoring"] is True


def test_sealed_e0_artifacts_have_zero_diff():
    evidence = sealed_e0_evidence()
    assert evidence["diff_empty"] is True, evidence["diff"]
    assert evidence["diff"] == ""
    assert evidence["tracked_file_count"] > 0
    assert evidence["byte_mismatches"] == []


def test_public_runtime_guard_is_scoped_to_the_e6_change_set(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src" / "haunt").mkdir(parents=True)
    runtime = repo / "src" / "haunt" / "recall.py"
    evidence = repo / "src" / "haunt" / "abstention_eval.py"

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "e6@example.invalid")
    git("config", "user.name", "E6")
    runtime.write_text("runtime\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "runtime before E6")
    evidence.write_text("evidence\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "E6 evidence")
    monkeypatch.setattr(abstention_eval, "ROOT", repo)
    clean = abstention_eval.public_runtime_evidence()
    assert clean["diff_empty"] is True
    assert clean["history_attributable"] is True

    runtime.write_text("concurrent work\n", encoding="utf-8")
    git("commit", "-qam", "unrelated runtime change")
    assert abstention_eval.public_runtime_evidence()["diff_empty"] is True

    evidence.write_text("evidence and policy\n", encoding="utf-8")
    runtime.write_text("abstention policy\n", encoding="utf-8")
    assert abstention_eval.public_runtime_evidence()["diff_empty"] is False

    git("commit", "-qam", "E6 ships an abstention policy")
    shipped = abstention_eval.public_runtime_evidence()
    assert shipped["diff_empty"] is False
    assert "src/haunt/recall.py" in shipped["diff"]

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(abstention_eval, "ROOT", shallow)
    degraded = abstention_eval.public_runtime_evidence()
    assert degraded["history_attributable"] is False
    assert degraded["diff_empty"] is True

    monkeypatch.setattr(abstention_eval, "ROOT", tmp_path / "not-a-repo")
    with pytest.raises(RuntimeError, match="requires git"):
        abstention_eval.public_runtime_evidence()


def test_public_runtime_guard_reports_what_it_could_not_attribute(
    tmp_path, monkeypatch
):
    """Uncommitted runtime edits carry no commit to attribute them by.

    The gate deciding whether to diff the working tree keys on E6's own
    evidence surface, so a dirty runtime file beside clean evidence left the
    payload unrun and diff_empty True -- certifying what was never read.
    sealed_e0_evidence has byte_mismatches for that; this leg had nothing.
    """
    repo = tmp_path / "repo"
    (repo / "src" / "haunt").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "abstention_eval").mkdir(parents=True)
    runtime = repo / "src" / "haunt" / "recall.py"
    evidence = repo / "src" / "haunt" / "abstention_eval.py"

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "e6@example.invalid")
    git("config", "user.name", "E6")
    runtime.write_text("runtime\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "runtime before E6")
    evidence.write_text("evidence\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "E6 evidence")
    monkeypatch.setattr(abstention_eval, "ROOT", repo)
    clean = abstention_eval.public_runtime_evidence()
    assert clean["diff_empty"] is True
    assert clean["uncommitted_paths"] == []
    assert clean["uncommitted_unattributed"] is False

    # Concurrent work on a shared branch is not E6's and must not fail the
    # gate, but it cannot pass as a diff that was read and found empty.
    runtime.write_text("concurrent work\n", encoding="utf-8")
    dirty = abstention_eval.public_runtime_evidence()
    assert dirty["diff_empty"] is True
    assert dirty["uncommitted_paths"] == ["src/haunt/recall.py"]
    assert dirty["uncommitted_unattributed"] is True

    # An untracked evidence file is an E6 edit too, and claims the diff.
    (repo / "tests" / "fixtures" / "abstention_eval" / "extra.json").write_text(
        "{}\n", encoding="utf-8"
    )
    attributed = abstention_eval.public_runtime_evidence()
    assert attributed["diff_empty"] is False
    assert attributed["uncommitted_unattributed"] is False
    assert "src/haunt/recall.py" in attributed["diff"]


def test_reproduction_seals_the_rerank_flag_out_of_the_measurement(monkeypatch):
    """recall() honours HAUNT_RERANK_ENABLED, so evidence must not inherit it."""
    monkeypatch.setenv(RERANK_ENABLED_ENV, "1")
    seen: list[str | None] = []

    class _Reached(Exception):
        pass

    def spy(*_args, **_kwargs):
        seen.append(os.environ.get(RERANK_ENABLED_ENV))
        raise _Reached

    monkeypatch.setattr(abstention_eval, "recall", spy)
    with pytest.raises(_Reached):
        evaluate_profile("fts", fixture_dir=FIXTURE)
    assert seen == [None]
    assert os.environ[RERANK_ENABLED_ENV] == "1"


def test_sealed_e0_guard_separates_e0_maintenance_from_e6(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src" / "haunt").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "retrieval_eval").mkdir(parents=True)
    corpus = repo / "tests" / "fixtures" / "retrieval_eval" / "corpus.json"
    evidence = repo / "src" / "haunt" / "abstention_eval.py"

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "e0@example.invalid")
    git("config", "user.name", "E0")
    corpus.write_text('{"records": []}\n', encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "seal E0 corpus")
    evidence.write_text("evidence\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "E6 evidence")
    monkeypatch.setattr(abstention_eval, "ROOT", repo)
    assert sealed_e0_evidence()["tracked_file_count"] == 1

    # E0 re-freezing its own artifacts is not an E6 change to them.
    corpus.write_text('{"records": ["stemming case"]}\n', encoding="utf-8")
    git("commit", "-qam", "add a non-prefix inflection case and re-freeze")
    maintained = sealed_e0_evidence()
    assert maintained["diff_empty"] is True
    assert maintained["byte_mismatches"] == []

    evidence.write_text("evidence reaching into E0\n", encoding="utf-8")
    corpus.write_text('{"records": ["retuned for E6"]}\n', encoding="utf-8")
    git("commit", "-qam", "E6 retunes the sealed corpus")
    assert sealed_e0_evidence()["diff_empty"] is False


def test_feature_definition_forbids_rrf_truth_and_reader_shortcuts():
    forbidden = set(FEATURE_DEFINITION["forbidden_inputs"])
    assert {
        "fused_rrf_score",
        "renormalized_rrf_score",
        "fact_truth",
        "source_reputation",
        "reader_model",
        "cross_encoder",
    } <= forbidden
    assert FEATURE_DEFINITION["candidate_scope"] == "fixed_pre_abstention_top_5"


def test_coverage_deduplicates_terms_and_batches_top_five(tmp_path, monkeypatch):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_OFFLINE", "1")
    embed.reset()
    with Store("coverage") as store:
        first = store.observe("backup algorithm", defer_embedding=True).memory_id
        ids = [first]
        for index in range(4):
            ids.append(
                store.observe(
                    f"unrelated filler {index}", defer_embedding=True
                ).memory_id
            )
    with open_existing_readonly("coverage") as store:
        once, once_diag = _coverage_many(store, "backup algorithm", ids)
        repeated, repeated_diag = _coverage_many(
            store, "BACKUP backup Backup algorithm ALGORITHM", ids
        )
    assert once[first] == repeated[first]
    assert repeated[first]["query_token_count"] == 2
    assert repeated[first]["coverage"] == 1.0
    assert once_diag["sql_statement_count"] == 1
    assert repeated_diag["sql_statement_count"] == 1
    embed.reset()


def test_absent_component_cannot_invent_strength_and_rrf_is_never_read(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HAUNT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HAUNT_FTS_ONLY", "1")
    monkeypatch.setenv("HAUNT_OFFLINE", "1")
    embed.reset()
    with Store("membership") as store:
        memory_id = store.observe("backup algorithm", defer_embedding=True).memory_id
    vector_only = _hit(
        memory_id,
        score=-1000.0,
        fts_rank=None,
        vec_rank=1,
        vec_distance=0.8,
        vec_metric="cosine_distance",
    )
    with open_existing_readonly("membership") as store:
        low_rrf = _evidence(store, "backup algorithm", [vector_only])
        vector_only.score = 1000.0
        high_rrf = _evidence(store, "backup algorithm", [vector_only])
    candidate = low_rrf["candidates"][0]
    assert candidate["fts"]["coverage"] == 1.0
    assert candidate["fts"]["eligible_for_strength"] is False
    assert candidate["vector"]["eligible_for_strength"] is True
    assert low_rrf["strength"] == pytest.approx(0.2)
    assert high_rrf == low_rrf
    assert low_rrf["diagnostics"]["rrf_score_read_by_feature"] is False
    assert ".score" not in inspect.getsource(abstention_eval._evidence)
    embed.reset()


@pytest.mark.parametrize(
    ("kwargs", "state", "matches"),
    [
        (
            dict(
                requested=False,
                sqlite_vec_available=True,
                native_table=True,
                persisted_embedding_rows=1,
                candidate_metrics=[],
            ),
            "not_requested",
            False,
        ),
        (
            dict(
                requested=True,
                sqlite_vec_available=True,
                native_table=True,
                persisted_embedding_rows=1,
                candidate_metrics=["cosine_distance"],
            ),
            "native_cosine_candidate",
            True,
        ),
        (
            dict(
                requested=True,
                sqlite_vec_available=True,
                native_table=True,
                persisted_embedding_rows=1,
                candidate_metrics=[],
            ),
            "native_cosine_empty",
            True,
        ),
        (
            dict(
                requested=True,
                sqlite_vec_available=True,
                native_table=False,
                persisted_embedding_rows=1,
                candidate_metrics=["l2_distance"],
            ),
            "persisted_l2_candidate",
            False,
        ),
        (
            dict(
                requested=True,
                sqlite_vec_available=True,
                native_table=False,
                persisted_embedding_rows=1,
                candidate_metrics=[],
            ),
            "persisted_l2_empty",
            False,
        ),
        (
            dict(
                requested=True,
                sqlite_vec_available=False,
                native_table=False,
                persisted_embedding_rows=0,
                candidate_metrics=[],
            ),
            "sqlite_vec_unavailable_empty",
            False,
        ),
        (
            dict(
                requested=True,
                sqlite_vec_available=True,
                native_table=False,
                persisted_embedding_rows=0,
                candidate_metrics=[],
            ),
            "native_table_absent_empty",
            False,
        ),
    ],
)
def test_vector_profile_identity_distinguishes_every_execution_arm(
    kwargs, state, matches
):
    result = classify_vector_profile(**kwargs)
    assert result["state"] == state
    assert result["matches_pinned_native_cosine"] is matches


def test_held_file_is_unreachable_until_fit_boundary_and_cannot_change_it(tmp_path):
    normal_boundary = []

    def capture_normal(event, details):
        if event == "fit_boundary_complete":
            normal_boundary.append(deepcopy(details["fit_report"]))

    normal = evaluate_profile("fts", fixture_dir=FIXTURE, audit_hook=capture_normal)
    assert len(normal_boundary) == 1
    events = [row["event"] for row in normal["label_access_audit"]["events"]]
    assert events.index("fit_boundary_complete") < events.index(
        "before_open:held_labels"
    )
    assert events.index("before_open:held_labels") < events.index(
        "before_open:dataset_manifest"
    )

    poisoned_fixture = tmp_path / "poisoned"
    shutil.copytree(FIXTURE, poisoned_fixture)
    (poisoned_fixture / "held-labels.json").write_text(
        "THIS FILE MUST NOT BE OPENED BEFORE THE FIT BOUNDARY",
        encoding="utf-8",
    )
    poison_state = {"boundary": False, "held_open_after_boundary": False}

    def poison_audit(event, details):
        if event == "fit_boundary_complete":
            poison_state["boundary"] = True
            assert canonical_hash(details["fit_report"]) == canonical_hash(
                normal_boundary[0]
            )
        if event == "before_open:held_labels":
            assert poison_state["boundary"], "held labels opened before fit boundary"
            poison_state["held_open_after_boundary"] = True

    with pytest.raises(json.JSONDecodeError):
        evaluate_profile("fts", fixture_dir=poisoned_fixture, audit_hook=poison_audit)
    assert poison_state == {
        "boundary": True,
        "held_open_after_boundary": True,
    }

    flipped_fixture = tmp_path / "flipped"
    shutil.copytree(FIXTURE, flipped_fixture)
    held_source = json.loads((flipped_fixture / "held-labels.json").read_text("utf-8"))
    for row in held_source["labels"]:
        if row["label"] == "answerable":
            row["label"] = "unanswerable"
            row["relevant"] = []
        else:
            row["label"] = "answerable"
            row["relevant"] = ["deliberately-wrong-held-label"]
    (flipped_fixture / "held-labels.json").write_text(
        json.dumps(held_source), encoding="utf-8"
    )
    flipped_boundary = []

    def capture_flipped(event, details):
        if event == "fit_boundary_complete":
            flipped_boundary.append(deepcopy(details["fit_report"]))

    # The changed held file is invalid against the frozen query semantics or
    # manifest, but that failure is necessarily after fit is complete.
    with pytest.raises(ValueError):
        evaluate_profile("fts", fixture_dir=flipped_fixture, audit_hook=capture_flipped)
    assert len(flipped_boundary) == 1
    assert canonical_hash(flipped_boundary[0]) == canonical_hash(normal_boundary[0])
    assert list(FitLabels.__dataclass_fields__) == ["by_case"]
    assert list(inspect.signature(analyze_fit).parameters) == ["observations", "labels"]


def test_network_deny_records_attempt_and_restores_socket_constructor():
    original = socket.socket
    with abstention_eval.NetworkDeny() as deny:
        with pytest.raises(RuntimeError, match="blocked network access"):
            socket.socket()
        assert deny.attempts
    assert socket.socket is original
    restored = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    restored.close()


def test_fts_reproduction_is_deterministic_offline_and_calibratable():
    first = evaluate_profile("fts", fixture_dir=FIXTURE)
    second = evaluate_profile("fts", fixture_dir=FIXTURE)
    assert first == second
    assert first["status"] == "calibratable"
    assert first["runtime_policy_shipped"] is False
    assert first["fit"]["analysis"]["selected_threshold"] == 0.875
    held = first["held_out"]["metrics_at_fit_only_diagnostic_boundary"]
    assert held["pre_abstention_recall_at_5"] == 1.0
    assert held["gate_100_percent_unanswerable_abstained"] is True
    assert held["gate_95_percent_conditional_answerable_retained"] is True
    assert first["network_proof"]["haunt_offline"] is True
    assert first["network_proof"]["attempts"] == []
    assert first["evidence_query_batching"] == {
        "candidate_scope": "fixed_top_5",
        "maximum_candidate_count_observed": 5,
        "maximum_coverage_sql_statements_per_recall_observed": 1,
        "query_tokens_are_nfkc_casefold_deduplicated": True,
        "rrf_score_read_by_feature": False,
    }


def test_hybrid_requires_explicit_verified_cache(tmp_path):
    with pytest.raises(RuntimeError, match="requires --model-cache"):
        evaluate_profile("hybrid", fixture_dir=FIXTURE)
    with pytest.raises(RuntimeError, match="artifact preflight mismatch"):
        evaluate_profile("hybrid", fixture_dir=FIXTURE, model_cache=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing="),
        ("zero", "zero_byte="),
        ("size", "size_mismatch="),
        ("sidecar", "missing_sidecar="),
        ("extra", "extra_selected="),
        ("variant", "variant_mismatch_forbidden="),
    ],
)
def test_hybrid_manifest_rejects_cache_shape_before_hashing(
    tmp_path, mutation, message
):
    cache = tmp_path / mutation
    if mutation == "missing":
        cache.mkdir()
    elif mutation == "zero":
        manifest = json.loads(
            (FIXTURE / "hybrid-model-manifest.json").read_text("utf-8")
        )
        for row in manifest["files"]:
            path = cache / row["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    else:
        _sparse_manifest_cache(cache)
        if mutation == "size":
            with (cache / "BAAI-bge-m3/onnx/model.onnx").open("r+b") as stream:
                stream.truncate(123)
        elif mutation == "sidecar":
            (cache / "BAAI-bge-m3/onnx/model.onnx_data").unlink()
        elif mutation == "extra":
            (cache / "BAAI-bge-m3/onnx/unapproved.onnx").write_bytes(b"extra")
        elif mutation == "variant":
            (cache / "BAAI-bge-m3/onnx/model_quantized.onnx").write_bytes(
                b"wrong variant"
            )
    with pytest.raises(RuntimeError, match=message):
        abstention_eval.verify_local_hybrid_cache(cache, fixture_dir=FIXTURE)


def test_same_dimension_wrong_cache_fails_hash_before_embed_init(tmp_path, monkeypatch):
    cache = tmp_path / "same-dimension-wrong-cache"
    manifest = _sparse_manifest_cache(cache)
    config_row = next(
        row for row in manifest["files"] if row["relative_path"].endswith("config.json")
    )
    claimed_dimension = json.dumps({"hidden_size": 1024}).encode("utf-8")
    wrong_config = claimed_dimension + b" " * (
        config_row["size"] - len(claimed_dimension)
    )
    (cache / config_row["relative_path"]).write_bytes(wrong_config)

    def embed_init_must_not_run():
        raise AssertionError("embed initialization ran before artifact verification")

    monkeypatch.setattr(embed, "reset", embed_init_must_not_run)
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        evaluate_profile("hybrid", fixture_dir=FIXTURE, model_cache=cache)


def test_manifest_id_cannot_bless_alternate_artifact_hashes(tmp_path):
    fixture = tmp_path / "altered-manifest"
    shutil.copytree(FIXTURE, fixture)
    manifest_path = fixture / "hybrid-model-manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest identity"):
        abstention_eval.verify_local_hybrid_cache(
            tmp_path / "unused-cache", fixture_dir=fixture
        )


def test_committed_hybrid_report_is_an_honest_blocker():
    report = json.loads((REPORTS / "hybrid-blocked.json").read_text("utf-8"))
    _, split = load_inputs(FIXTURE)
    assert (
        report["dataset_canonical_sha256"]
        == "8119f4508d3582bc665a5a0117940c6eeca593de56f33999563ddf188846264c"
    )
    assert report["split_canonical_sha256"] == canonical_hash(split)
    assert report["dataset_manifest"]["verified_after_held_out_scoring"] is True
    assert report["status"] == "blocked"
    assert report["runtime_policy_shipped"] is False
    assert report["blocker"]["code"] == "raw_evidence_fit_gate_unsatisfied"
    analysis = report["fit"]["analysis"]
    assert analysis["possible_under_feature_definition"] is False
    assert analysis["fit_positive_retained_at_minimum_negative_boundary"] == 6
    assert analysis["conditional_fit_positive_count"] == 20
    assert analysis["required_fit_positive_retained_for_95pct"] == 19
    held = report["held_out"]["metrics_at_fit_only_diagnostic_boundary"]
    assert held["pre_abstention_recall_at_5"] == 0.75
    assert held["gate_100_percent_unanswerable_abstained"] is True
    assert held["conditional_answerable_retention"] == 0.4
    assert held["gate_95_percent_conditional_answerable_retained"] is False
    semantic = report["held_out"]["pure_semantic_paraphrase_metrics"]
    assert semantic["answerable_cases"] == 20
    assert semantic["pre_abstention_recall_at_5"] == 0.75
    assert semantic["conditional_answerable_retention"] == 0.4
    assert report["network_proof"]["attempts"] == []
    assert report["network_proof"]["haunt_offline"] is False
    assert report["profile"]["model_id"] == "BAAI/bge-m3"
    assert report["profile"]["dimension"] == 1024
    assert report["profile"]["vector_backend"] == "sqlite_vec_native"
    assert report["profile"]["vector_metric"] == "cosine_distance"
    assert (
        report["local_model_cache"]["matched_manifest_id"]
        == "haunt-bge-m3-onnx-split-f8425123-v1"
    )
    assert report["local_model_cache"]["selected_variant"] == (
        "nonquantized_split_onnx"
    )
    assert report["local_model_cache"]["files"] == [
        {
            "relative_path": "BAAI-bge-m3/onnx/config.json",
            "sha256": "f24afd5de914fba8c668426c43d208a1a54022500c63b2c160be20891686fce8",
            "size": 698,
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx",
            "sha256": "f84251230831afb359ab26d9fd37d5936d4d9bb5d1d5410e66442f630f24435b",
            "size": 724923,
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx_data",
            "sha256": "1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4",
            "size": 2266820608,
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/tokenizer.json",
            "sha256": "6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790",
            "size": 17082821,
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/tokenizer_config.json",
            "sha256": "7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4",
            "size": 1173,
        },
    ]
    assert all(
        case["vector_profile"]["matches_pinned_native_cosine"]
        for phase in ("fit", "held_out")
        for case in report[phase]["cases"]
    )
    assert all(
        not (set(case["relevant"]) & set(case["fts_only_returned"]))
        for phase in ("fit", "held_out")
        for case in report[phase]["cases"]
        if case["label"] == "answerable"
    )
    held_paraphrases = [
        case for case in report["held_out"]["cases"] if case["label"] == "answerable"
    ]
    assert len(held_paraphrases) == 20

    fit_pairs = {
        (row["negative_case_id"], row["positive_case_id"])
        for row in analysis["dominating_negative_positive_pairs"]
    }
    held_pairs = {
        (row["negative_case_id"], row["positive_case_id"])
        for row in report["held_out"]["evidence_overlap"][
            "dominating_negative_positive_pairs"
        ]
    }
    assert ("hybrid-fit-u04", "hybrid-fit-a16") in fit_pairs
    assert ("hybrid-fit-u11", "hybrid-fit-a18") in fit_pairs
    assert ("hybrid-held-u01", "hybrid-held-a03") in held_pairs
    assert ("hybrid-held-u06", "hybrid-held-a06") in held_pairs


def test_reports_prove_fit_label_separation_and_no_public_policy():
    for name in ("fts.json", "hybrid-blocked.json"):
        report = json.loads((REPORTS / name).read_text("utf-8"))
        assert report["fit"]["held_out_labels_available_to_fit"] == []
        assert report["held_out"]["labels_loaded_after_fit_analysis"] is True
        assert (
            report["label_access_audit"]["fit_boundary_precedes_held_label_open"]
            is True
        )
        events = [row["event"] for row in report["label_access_audit"]["events"]]
        assert events.index("fit_boundary_complete") < events.index(
            "before_open:held_labels"
        )
        assert events.index("before_open:held_labels") < events.index(
            "before_open:dataset_manifest"
        )
        assert report["sealed_e0"]["diff_empty"] is True
        assert report["sealed_e0"]["byte_mismatches"] == []
        assert report["public_runtime_policy"]["diff_empty"] is True
        assert not any(report["separation_from_e0"]["overlaps"].values())
        assert (
            report["evidence_query_batching"][
                "maximum_coverage_sql_statements_per_recall_observed"
            ]
            == 1
        )
    assert not (Path(abstention_eval.__file__).parent / "abstention.py").exists()


def test_latency_evidence_covers_1k_10k_100k_with_one_batched_statement():
    report = json.loads((REPORTS / "latency.json").read_text("utf-8"))
    assert report["deterministic_gate"] == {
        "corpus_sizes": [1000, 10000, 100000],
        "coverage_sql_statements_per_top_five": 1,
    }
    assert report["timing_is_observational_not_a_cross_machine_gate"] is True
    assert [row["corpus_rows"] for row in report["measurements"]] == [
        1000,
        10000,
        100000,
    ]
    assert all(
        row["coverage_sql_statement_counts"] == [1] for row in report["measurements"]
    )
    assert all(
        row["non_gating_observation_below_10ms_p95_overhead"] is True
        for row in report["measurements"]
    )
