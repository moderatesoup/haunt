"""E6 abstention feasibility is frozen as evidence, not shipped policy."""

from __future__ import annotations

import inspect
import json
import socket
from copy import deepcopy
from pathlib import Path

import pytest

from haunt import abstention_eval, embed
from haunt.abstention_eval import (
    FEATURE_DEFINITION,
    FitLabels,
    RetrievalObservation,
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


def _observation(case_id: str, strength: float, returned: tuple[str, ...]):
    return RetrievalObservation(
        case_id=case_id,
        returned=returned,
        evidence={
            "strength": strength,
            "diagnostics": {"native_cosine_top_two_distance_margin": 0.1},
        },
        vector_profile={},
    )


def test_dataset_split_is_large_predeclared_and_separate_from_e0():
    dataset, split = load_inputs(FIXTURE)
    by_case = {row["id"]: row for row in dataset["cases"]}

    assert len(dataset["records"]) == 40
    assert len(dataset["cases"]) == 160
    for phase in ("fit", "held_out"):
        for profile in ("fts", "hybrid"):
            rows = [by_case[key] for key in split[phase][profile]]
            assert sum(row["label"] == "answerable" for row in rows) == 20
            assert sum(row["label"] == "unanswerable" for row in rows) == 20
    hybrid_positives = [
        row
        for row in dataset["cases"]
        if row["profile"] == "hybrid" and row["label"] == "answerable"
    ]
    assert len(hybrid_positives) == 40
    assert all(row["semantic_paraphrase"] is True for row in hybrid_positives)
    assert all(row["required_lexical_overlap"] is False for row in hybrid_positives)
    assert not any(separation_evidence(dataset)["overlaps"].values())
    assert split["manual_semantic_audit"]["reviewed_on"] == "2026-08-26"


def test_sealed_e0_artifacts_have_zero_diff():
    evidence = sealed_e0_evidence()
    assert evidence["diff_empty"] is True, evidence["diff"]
    assert evidence["diff"] == ""
    assert evidence["tracked_file_count"] > 0
    assert evidence["byte_mismatches"] == []


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


def test_fit_capability_cannot_reach_poisoned_held_labels():
    class Poison:
        def __str__(self):
            raise AssertionError("held label was read")

        def __iter__(self):
            raise AssertionError("held relevant list was read")

        def __eq__(self, other):
            raise AssertionError("held label was compared")

    dataset, split = load_inputs(FIXTURE)
    original = deepcopy(dataset)
    poisoned = deepcopy(dataset)
    flipped = deepcopy(dataset)
    held_ids = {
        case_id
        for profile in ("fts", "hybrid")
        for case_id in split["held_out"][profile]
    }
    for row in poisoned["cases"]:
        if row["id"] in held_ids:
            row["label"] = Poison()
            row["relevant"] = Poison()
    for row in flipped["cases"]:
        if row["id"] in held_ids:
            row["label"] = (
                "unanswerable" if row["label"] == "answerable" else "answerable"
            )
            row["relevant"] = ["deliberately-wrong-held-label"]

    fit_ids = split["fit"]["hybrid"]
    original_by_case = {row["id"]: row for row in original["cases"]}
    observations = [
        _observation(
            case_id,
            0.9 if original_by_case[case_id]["label"] == "answerable" else 0.2,
            tuple(original_by_case[case_id]["relevant"] or ["irrelevant"]),
        )
        for case_id in fit_ids
    ]
    result_original = analyze_fit(
        observations, abstention_eval._fit_labels(original, split, "hybrid")
    )
    result_poisoned = analyze_fit(
        observations, abstention_eval._fit_labels(poisoned, split, "hybrid")
    )
    result_flipped = analyze_fit(
        observations, abstention_eval._fit_labels(flipped, split, "hybrid")
    )
    assert canonical_hash(result_original) == canonical_hash(result_poisoned)
    assert canonical_hash(result_original) == canonical_hash(result_flipped)
    assert result_original["selected_threshold"] == pytest.approx(0.55)
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
    with pytest.raises(RuntimeError, match="explicit verified HAUNT_MODEL_CACHE"):
        evaluate_profile("hybrid", fixture_dir=FIXTURE, model_cache=tmp_path)


def test_committed_hybrid_report_is_an_honest_blocker():
    report = json.loads((REPORTS / "hybrid-blocked.json").read_text("utf-8"))
    dataset, split = load_inputs(FIXTURE)
    assert report["dataset_canonical_sha256"] == canonical_hash(dataset)
    assert report["split_canonical_sha256"] == canonical_hash(split)
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
    assert report["local_model_cache"]["files"] == [
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx",
            "sha256": "f84251230831afb359ab26d9fd37d5936d4d9bb5d1d5410e66442f630f24435b",
            "size": 724923,
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/tokenizer.json",
            "sha256": "6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790",
            "size": 17082821,
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx_data",
            "sha256": "1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4",
            "size": 2266820608,
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
