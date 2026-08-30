"""The cached BGE-M3 bytes are checked against the committed manifest.

_download_bge_m3 pins repo and revision, but that pin only binds the download
path. A model dropped straight into the cache never passes through it, and the
haunt-model-source.json marker beside it is written by whoever wrote the
cache -- so the recorded identity attests nothing on its own. These tests hold
the load path to verifying the bytes before onnxruntime is handed the graph,
and to saying so out loud on the paths where it cannot.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from fnmatch import fnmatch
from pathlib import Path

import pytest

import haunt
from haunt import abstention_eval, embed, model_manifest
from haunt.model_manifest import canonical_hash, verify_local_hybrid_cache
from haunt.paths import models_dir

PACKAGE_DIR = Path(haunt.__file__).resolve().parent
HYBRID_MANIFEST = model_manifest.HYBRID_MANIFEST_DIR / "hybrid-model-manifest.json"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _fake_cache(root: Path) -> Path:
    """Enough layout for _local_bge_m3_ready, with none of a real model."""
    onnx = root / embed.BGE_M3_DIRNAME / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "model.onnx").write_bytes(b"graph")
    (onnx / "model.onnx_data").write_bytes(b"weights")
    (onnx / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root / embed.BGE_M3_DIRNAME


def _fake_quant_cache(root: Path) -> Path:
    """The third-party quantized layout, which no committed manifest covers."""
    model = root / embed.BGE_M3_DIRNAME
    (model / "onnx").mkdir(parents=True)
    (model / "onnx" / "model_quantized.onnx").write_bytes(b"quantized-graph")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def _must_not_run(message: str):
    def boom(*_args, **_kwargs):
        raise AssertionError(message)

    return boom


def _clean_env(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("HAUNT_MODEL_CACHE", str(root))
    monkeypatch.delenv("HAUNT_EMBED_SKIP_MODEL_VERIFY", raising=False)
    monkeypatch.delenv("HAUNT_EMBED_QUANT_FALLBACK", raising=False)


def test_load_verifies_the_cache_before_building_the_session(tmp_path, monkeypatch):
    _clean_env(monkeypatch, tmp_path)
    _fake_cache(tmp_path)
    calls: list[tuple[Path, dict]] = []

    def refuse(cache_root, **kwargs):
        calls.append((cache_root, kwargs))
        raise RuntimeError("hybrid cache artifact hash mismatch: planted model")

    monkeypatch.setattr(model_manifest, "verify_local_hybrid_cache", refuse)
    monkeypatch.setattr(
        embed,
        "OnnxEmbedder",
        _must_not_run("an unverified graph reached onnxruntime"),
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        embed._load_onnx_bge_m3()
    assert calls == [(models_dir(), {"hash_max_bytes": embed.VERIFY_HASH_MAX_BYTES})]


def test_a_verified_cache_loads(tmp_path, monkeypatch):
    _clean_env(monkeypatch, tmp_path)
    root = _fake_cache(tmp_path)
    monkeypatch.setattr(
        model_manifest,
        "verify_local_hybrid_cache",
        lambda *_a, **_k: {"matched_manifest_id": "test", "files": []},
    )
    monkeypatch.setattr(embed, "OnnxEmbedder", lambda ready: f"model:{ready}")
    model, nbytes = embed._load_onnx_bge_m3()
    assert model == f"model:{root}"
    assert nbytes > 0


@pytest.mark.parametrize(
    ("flag", "cache"),
    [
        ("HAUNT_EMBED_SKIP_MODEL_VERIFY", _fake_cache),
        ("HAUNT_EMBED_QUANT_FALLBACK", _fake_quant_cache),
    ],
)
def test_skipping_verification_takes_an_env_opt_in_and_is_recorded(
    tmp_path, monkeypatch, capsys, flag, cache
):
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv(flag, "1")
    root = cache(tmp_path)
    monkeypatch.setattr(
        model_manifest,
        "verify_local_hybrid_cache",
        _must_not_run("verification ran under an explicit skip"),
    )
    embed._verify_bge_m3_cache(root)
    assert "embed_m3_unverified" in capsys.readouterr().err


def test_planted_quantized_files_do_not_skip_verification(tmp_path, monkeypatch):
    """The quantized layout is unmanifested, so it must not be self-certifying."""
    _clean_env(monkeypatch, tmp_path)
    root = _fake_quant_cache(tmp_path)
    calls: list[Path] = []

    def record(cache_root, **_kwargs):
        calls.append(cache_root)
        return {"matched_manifest_id": "test", "files": []}

    monkeypatch.setattr(model_manifest, "verify_local_hybrid_cache", record)
    embed._verify_bge_m3_cache(root)
    assert calls == [models_dir()]


def test_an_uninstalled_manifest_is_reported_not_assumed(
    tmp_path, monkeypatch, capsys
):
    """The manifest ships inside the package now; absence means a broken install."""
    _clean_env(monkeypatch, tmp_path)
    root = _fake_cache(tmp_path)

    def absent(*_a, **_k):
        raise FileNotFoundError("no hybrid-model-manifest.json")

    monkeypatch.setattr(model_manifest, "verify_local_hybrid_cache", absent)
    embed._verify_bge_m3_cache(root)
    err = capsys.readouterr().err
    assert "embed_m3_unverified" in err
    assert "manifest not installed" in err


def _synthetic_fixture(
    tmp_path: Path, cache: Path, contents: dict[str, bytes]
) -> tuple[Path, str]:
    """A manifest over throwaway files, keyed to the real one's structure."""
    manifest = json.loads(HYBRID_MANIFEST.read_text("utf-8"))
    manifest["files"] = [
        {
            "relative_path": relative,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for relative, body in contents.items()
    ]
    fixture = tmp_path / "manifest"
    fixture.mkdir()
    (fixture / "hybrid-model-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for relative, body in contents.items():
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return fixture, canonical_hash(manifest)


GRAPH = b"graph-bytes"
SIDECAR = b"external-tensor-bytes-past-the-cap"
CONTENTS = {
    "BAAI-bge-m3/onnx/model.onnx": GRAPH,
    "BAAI-bge-m3/onnx/model.onnx_data": SIDECAR,
}


@pytest.fixture()
def capped_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    fixture, digest = _synthetic_fixture(tmp_path, cache, CONTENTS)
    monkeypatch.setattr(model_manifest, "HYBRID_ARTIFACT_MANIFEST_SHA256", digest)
    return cache, fixture


def test_hash_cap_checks_oversized_files_by_size_and_the_rest_by_content(
    capped_cache,
):
    cache, fixture = capped_cache
    report = verify_local_hybrid_cache(cache, manifest_dir=fixture, hash_max_bytes=16)
    assert report["files"] == [
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx",
            "size": len(GRAPH),
            "sha256": hashlib.sha256(GRAPH).hexdigest(),
        },
        {
            "relative_path": "BAAI-bge-m3/onnx/model.onnx_data",
            "size": len(SIDECAR),
            "sha256": None,
        },
    ]


def test_hash_cap_still_catches_a_substituted_graph(capped_cache):
    cache, fixture = capped_cache
    (cache / "BAAI-bge-m3/onnx/model.onnx").write_bytes(b"other-graph")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        verify_local_hybrid_cache(cache, manifest_dir=fixture, hash_max_bytes=16)


def test_hash_cap_catches_a_resized_sidecar_but_not_a_restuffed_one(capped_cache):
    """The residual gap, stated as a test: size is all the cap can promise."""
    cache, fixture = capped_cache
    sidecar = cache / "BAAI-bge-m3/onnx/model.onnx_data"
    sidecar.write_bytes(b"X" * len(SIDECAR))
    verify_local_hybrid_cache(cache, manifest_dir=fixture, hash_max_bytes=16)

    sidecar.write_bytes(b"X" * (len(SIDECAR) + 1))
    with pytest.raises(RuntimeError, match="size_mismatch"):
        verify_local_hybrid_cache(cache, manifest_dir=fixture, hash_max_bytes=16)


def test_uncapped_verification_still_hashes_every_file(capped_cache):
    cache, fixture = capped_cache
    (cache / "BAAI-bge-m3/onnx/model.onnx_data").write_bytes(b"X" * len(SIDECAR))
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        verify_local_hybrid_cache(cache, manifest_dir=fixture)


def test_the_manifest_the_check_reads_lives_inside_the_package():
    """A wheel ships the package, never tests/, so the gate has to live here.

    Under tests/fixtures the check resolved through a repo checkout root that
    does not exist off-repo, and every wheel install fell through the
    FileNotFoundError leg above with verification never running.
    """
    assert HYBRID_MANIFEST.is_file()
    assert PACKAGE_DIR in HYBRID_MANIFEST.resolve().parents


def test_the_wheel_declares_the_manifest_as_package_data():
    """Living under src/haunt is not enough; setuptools ships listed data only."""
    text = PYPROJECT.read_text("utf-8")
    section = text.partition("[tool.setuptools.package-data]")[2].partition("\n[")[0]
    relative = HYBRID_MANIFEST.resolve().relative_to(PACKAGE_DIR)
    patterns = re.findall(r'"([^"]+)"', section)
    assert any(fnmatch(str(relative), pattern) for pattern in patterns), section


def test_a_built_wheel_actually_contains_the_manifest(tmp_path):
    """The end of the chain: what pip installs, not what the checkout holds."""
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
            str(PYPROJECT.parent),
        ],
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        pytest.skip(f"cannot build a wheel here: {built.stderr.strip()[-300:]}")
    wheel = next(iter(tmp_path.glob("haunt-*.whl")))
    relative = HYBRID_MANIFEST.resolve().relative_to(PACKAGE_DIR)
    with zipfile.ZipFile(wheel) as archive:
        assert f"haunt/{relative}" in archive.namelist()


# --- runtime/evaluation boundary -------------------------------------------
# The manifest check is a runtime obligation on every model load. It used to
# live in abstention_eval, so embed reached it through a function-local import
# that existed purely to dodge a cycle. These four tests pin the boundary.


def test_embed_import_does_not_drag_in_the_evaluation_module():
    """Loading the embedding backend must not import the E6 harness.

    abstention_eval pulls in recall, store, rerank and the whole urllib/ssl/
    email stack that only its network-deny harness needs. None of that belongs
    in a hook's import path.
    """
    code = (
        "import sys, haunt.embed; "
        "print(' '.join(str(int(m in sys.modules)) for m in ("
        "'haunt.abstention_eval','haunt.recall','haunt.rerank',"
        "'urllib.request','http.client','ssl')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["0", "0", "0", "0", "0", "0"], out.stdout


def test_patching_the_runtime_module_actually_reaches_embed(tmp_path, monkeypatch):
    """The patch seam must be live, not merely present.

    embed binds the *module* and resolves the attribute per call. Rebinding it
    as `from haunt.model_manifest import verify_local_hybrid_cache` would freeze
    the reference, and every test above that patches model_manifest would keep
    passing while exercising the real verifier. This asserts the seam by using
    it: if the patch does not reach embed, no SeamReached is raised.
    """

    class SeamReached(RuntimeError):
        pass

    _clean_env(monkeypatch, tmp_path)
    root = _fake_cache(tmp_path)

    def sentinel(*_args, **_kwargs):
        raise SeamReached("patch reached embed")

    monkeypatch.setattr(model_manifest, "verify_local_hybrid_cache", sentinel)
    with pytest.raises(SeamReached):
        embed._verify_bge_m3_cache(root)


def test_abstention_eval_re_exports_the_same_objects():
    """E6 evidence and the runtime check must attest one implementation."""
    for name in (
        "verify_local_hybrid_cache",
        "canonical_bytes",
        "canonical_hash",
        "_file_hash",
        "_emit_audit",
        "_read_json_component",
        "AuditHook",
        "HYBRID_MANIFEST_DIR",
        "HYBRID_ARTIFACT_MANIFEST_ID",
        "HYBRID_ARTIFACT_MANIFEST_SHA256",
    ):
        assert getattr(abstention_eval, name) is getattr(model_manifest, name), name


def test_the_two_schema_versions_are_independent_and_currently_equal():
    """They version different artifacts and are free to diverge.

    model_manifest.MANIFEST_SCHEMA_VERSION versions hybrid-model-manifest.json.
    abstention_eval.SCHEMA_VERSION versions the E6 fixture set, and its value is
    baked into DATASET_MANIFEST_SHA256. Sharing one constant would mean bumping
    either silently invalidated the other.
    """
    assert model_manifest.MANIFEST_SCHEMA_VERSION == 1
    assert abstention_eval.SCHEMA_VERSION == 1
    assert "SCHEMA_VERSION" not in vars(model_manifest)
