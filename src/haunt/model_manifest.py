"""Runtime identity of the on-device embedding model artifact.

`haunt.embed` must prove the cached BGE-M3 bytes are the ones the committed
manifest names before onnxruntime is allowed to execute the graph. That check
is a *runtime* obligation on every model load, so it cannot live in the
evaluation harness: importing `haunt.abstention_eval` to reach it pulls in
`haunt.recall`, `haunt.store`, `haunt.rerank` and the whole `urllib`/`ssl`/
`email` stack that only the E6 network-deny harness needs.

This module holds the artifact constants, the canonical JSON hashing the
manifest identity rests on, the cache verifier, and the small JSON/audit
helpers the verifier is built from. It imports the standard library only, so
`haunt.embed` can bind it at module scope without a cycle.

`haunt.abstention_eval` re-exports every name except MANIFEST_SCHEMA_VERSION,
so E6 evidence keeps one source for the manifest it attests. The traffic is not
one-way: `_read_json_component` is also the E6 harness's fixture reader, which
is why its parameter is still named `fixture_dir`. Splitting it in two would
duplicate a JSON reader to make a boundary look tidier than it is.

This file is listed in abstention_eval.E6_EVIDENCE_PATHS. It has to be: the
manifest document beside it is E6 evidence, the report's `local_model_cache`
field is this verifier's return value, and attributing the document without
its enforcer silently narrows the E6 isolation gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

# The manifest document's own schema, NOT the E6 evidence schema. The two are
# both 1 today and describe different artifacts: this one versions
# hybrid-model-manifest.json, `abstention_eval.SCHEMA_VERSION` versions the E6
# fixture set whose value is baked into DATASET_MANIFEST_SHA256. Keeping them
# separate lets either move without silently invalidating the other.
MANIFEST_SCHEMA_VERSION = 1

HYBRID_ARTIFACT_MANIFEST_ID = "haunt-bge-m3-onnx-split-f8425123-v1"
HYBRID_ARTIFACT_MANIFEST_SHA256 = (
    "d767f2f4a020b36e5a1d26636460af6cc5981836258c6f38cc677de9cab143a2"
)

# Package data, so an out-of-tree wheel install still resolves it; `tests/`
# is never shipped and must not be on this path.
HYBRID_MANIFEST_DIR = Path(__file__).resolve().parent / "data"

AuditHook = Callable[[str, Mapping[str, Any]], None]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
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
