"""On-device embeddings. Prefer BGE-M3 ONNX (1024-d) via onnxruntime.

Never calls a remote LLM or embedding API. Never fakes vectors.

Load order when HAUNT_EMBED_MODEL is BAAI/bge-m3 (the default):
  1. Local ONNX under ~/.haunt/models (or $HAUNT_MODEL_CACHE), verified
     against the committed artifact manifest before onnxruntime sees it
  2. Download BAAI/bge-m3 ONNX + tokenizer from Hugging Face
  3. Newer fastembed if it lists BAAI/bge-m3
  4. BAAI/bge-small-en-v1.5 via fastembed (384-d) — automatic fallback

Set HAUNT_EMBED_MODEL=off or HAUNT_FTS_ONLY=1 for FTS-only. Set
HAUNT_OFFLINE=1 to prohibit model/network initialization entirely.
Existing namespace DBs created at another dim must be rebuilt
(`haunt bootstrap --reembed`, or the store auto-rebuilds on mismatch).
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from haunt.paths import models_dir
from haunt.util import diag, dumps, env_flag, env_int, loads

DEFAULT_REQUESTED = "BAAI/bge-m3"
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"
BGE_M3_ID = "BAAI/bge-m3"
BGE_M3_DIRNAME = "BAAI-bge-m3"
BGE_M3_PATTERNS = [
    "onnx/model.onnx",
    "onnx/model.onnx_data",
    "onnx/tokenizer.json",
    "onnx/tokenizer_config.json",
    "onnx/config.json",
]
# The revision whose onnx/ file sizes and SHA-256s are the ones in
# src/haunt/data/hybrid-model-manifest.json, which ships in the wheel and which
# abstention_eval.verify_local_hybrid_cache enforces byte for byte.
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_QUANT_REPO = "onnx-community/bge-m3-ONNX"
# No committed manifest covers this variant -- the hybrid manifest's
# variant_policy forbids it -- so this pin is its only identity.
BGE_M3_QUANT_REVISION = "25b9af8e87a38eb120cfe87125383677b9cd309e"
BGE_M3_QUANT_PATTERNS = [
    "onnx/model_quantized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
]
BGE_M3_SOURCE_FILE = "haunt-model-source.json"
# Files at or below this size are hashed in full against the committed
# manifest on every load (graph, tokenizer, configs -- about 18MB together).
# The one file above it is model.onnx_data, 2.2GB of external tensors that
# cost seconds to hash and that cannot change the graph onnxruntime runs; it
# is checked by exact size instead. verify_local_hybrid_cache with no cap
# still hashes it, which is what the E6 harness does.
VERIFY_HASH_MAX_BYTES = 64 * 1024 * 1024
# Rows per ONNX forward pass. Throughput peaks between 16 and 32 on the
# corpus this was measured against and falls off above that; the smaller of
# the two flat values also bounds peak activation memory.
ONNX_SUB_BATCH = 16

_lock = threading.Lock()
_state: "EmbedState | None" = None


@dataclass(frozen=True)
class EmbedState:
    model_id: str
    requested: str
    dim: int
    available: bool
    fallback: bool
    backend: str = "none"
    error: str | None = None
    download_bytes: int | None = None


def _env_model() -> str:
    raw = (os.environ.get("HAUNT_EMBED_MODEL") or DEFAULT_REQUESTED).strip()
    return raw


def fts_only() -> bool:
    if env_flag("HAUNT_FTS_ONLY"):
        return True
    model = _env_model().lower()
    return model in {"off", "none", "fts", "fts5", "disabled"}


def offline() -> bool:
    """True when Haunt must not initialize/download a model or use sockets."""
    return env_flag("HAUNT_OFFLINE")


def _max_len() -> int:
    return env_int("HAUNT_EMBED_MAX_LEN", default=512, lo=8, hi=8192)


def _supported_fastembed() -> dict[str, int]:
    try:
        from fastembed import TextEmbedding

        out: dict[str, int] = {}
        for row in TextEmbedding.list_supported_models():
            name = row.get("model")
            dim = row.get("dim")
            if name and dim:
                out[str(name)] = int(dim)
        return out
    except Exception:
        return {}


def _dir_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        if p.is_file() and ".cache" not in p.parts:
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _bge_m3_dir() -> Path:
    return models_dir() / BGE_M3_DIRNAME


def _find_onnx(root: Path) -> Path | None:
    candidates = [
        root / "onnx" / "model.onnx",
        root / "model.onnx",
        root / "onnx" / "model_quantized.onnx",
        root / "model_quantized.onnx",
    ]
    for c in candidates:
        if c.is_file():
            if c.name == "model.onnx":
                data = c.with_name("model.onnx_data")
                if data.is_file() or c.stat().st_size > 10_000_000:
                    return c
                if not data.is_file():
                    continue
            return c
    return None


def _find_tokenizer(root: Path) -> Path | None:
    for c in (root / "onnx" / "tokenizer.json", root / "tokenizer.json"):
        if c.is_file():
            return c
    return None


def _local_bge_m3_ready(root: Path | None = None) -> Path | None:
    root = root or _bge_m3_dir()
    if _find_onnx(root) and _find_tokenizer(root):
        return root
    return None


def _quant_fallback_enabled() -> bool:
    """Opt in to the third-party quantized repo when the official one is gone."""
    return env_flag("HAUNT_EMBED_QUANT_FALLBACK")


def _repo_unavailable_errors() -> tuple[type[BaseException], ...]:
    """Hub errors meaning the pinned repo/revision itself cannot be had.

    Deliberately excludes timeouts, 5xx, DNS and disk errors. Empty when
    huggingface_hub is too old to expose any of them by name.
    """
    try:
        from huggingface_hub import errors
    except ImportError:
        try:
            from huggingface_hub import utils as errors  # type: ignore[no-redef]
        except ImportError:
            return ()
    found = (
        getattr(errors, name, None)
        for name in (
            "RepositoryNotFoundError",
            "RevisionNotFoundError",
            "EntryNotFoundError",
            "GatedRepoError",
        )
    )
    return tuple(cls for cls in found if isinstance(cls, type))


def _record_bge_m3_source(root: Path, repo_id: str, revision: str) -> None:
    """Record which repo produced these bytes. Best effort; never raises."""
    try:
        (root / BGE_M3_SOURCE_FILE).write_text(
            dumps({"repo_id": repo_id, "revision": revision}), encoding="utf-8"
        )
    except OSError as exc:
        diag("embed_m3_source_unrecorded", error=str(exc))


def bge_m3_source(root: Path | None = None) -> dict[str, str] | None:
    """Repo and revision the cached BGE-M3 came from, or None if unrecorded."""
    marker = (root or _bge_m3_dir()) / BGE_M3_SOURCE_FILE
    try:
        value = loads(marker.read_text(encoding="utf-8"), default={})
    except OSError:
        return None
    return value if isinstance(value, dict) and value else None


def _verify_bge_m3_cache(root: Path) -> None:
    """Require the cached BGE-M3 bytes to match the committed manifest.

    Raises RuntimeError on any mismatch, which drops the caller back to the
    fastembed fallback rather than executing an unattested graph. Skipped --
    with a diag each time, never silently -- when the operator opted into the
    quantized variant no committed manifest covers, when the manifest is not
    reachable (it is package data, so only a damaged install loses it), or
    when HAUNT_EMBED_SKIP_MODEL_VERIFY is set for a hand-placed model.
    """
    if env_flag("HAUNT_EMBED_SKIP_MODEL_VERIFY"):
        diag("embed_m3_unverified", reason="HAUNT_EMBED_SKIP_MODEL_VERIFY")
        return
    onnx_path = _find_onnx(root)
    quantized = onnx_path is not None and "quantized" in onnx_path.name
    # Quantized files alone would be an easy way to dodge a check no committed
    # manifest can make, so the opt-in that is the only legitimate way to obtain
    # them has to be set too; without it the manifest rejects them by name.
    if quantized and _quant_fallback_enabled():
        diag("embed_m3_unverified", reason="quantized variant has no manifest")
        return
    # abstention_eval imports this module, so the import has to be deferred.
    from haunt.abstention_eval import verify_local_hybrid_cache

    try:
        evidence = verify_local_hybrid_cache(
            root.parent, hash_max_bytes=VERIFY_HASH_MAX_BYTES
        )
    except FileNotFoundError as exc:
        diag("embed_m3_unverified", reason="manifest not installed", error=str(exc))
        return
    diag(
        "embed_m3_verified",
        manifest_id=evidence["matched_manifest_id"],
        size_only=[
            row["relative_path"] for row in evidence["files"] if row["sha256"] is None
        ],
    )


def _download_bge_m3(root: Path) -> Path:
    """Download official BGE-M3 ONNX (+ tokenizer) into root. Local files only after this."""
    if offline():
        raise RuntimeError("offline mode forbids embedding model download")
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=BGE_M3_ID,
            revision=BGE_M3_REVISION,
            local_dir=str(root),
            allow_patterns=BGE_M3_PATTERNS,
        )
    except _repo_unavailable_errors() as exc:
        official_exc: Exception = exc
    else:
        if _local_bge_m3_ready(root):
            _record_bge_m3_source(root, BGE_M3_ID, BGE_M3_REVISION)
            return root
        official_exc = RuntimeError("BAAI/bge-m3 ONNX files missing after download")
    # onnxruntime executes whatever graph lands here, so switching publishers
    # is a decision, not a retry.
    if not _quant_fallback_enabled():
        raise official_exc
    diag("embed_m3_official_unavailable", error=str(official_exc))
    snapshot_download(
        repo_id=BGE_M3_QUANT_REPO,
        revision=BGE_M3_QUANT_REVISION,
        local_dir=str(root),
        allow_patterns=BGE_M3_QUANT_PATTERNS,
    )
    if _local_bge_m3_ready(root) is None:
        raise RuntimeError(f"BGE-M3 ONNX download failed (official: {official_exc})")
    _record_bge_m3_source(root, BGE_M3_QUANT_REPO, BGE_M3_QUANT_REVISION)
    return root


class OnnxEmbedder:
    """BGE-M3 (or compatible) ONNX embedder. Dense vectors only."""

    def __init__(self, root: Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path = _find_onnx(root)
        tok_path = _find_tokenizer(root)
        if onnx_path is None or tok_path is None:
            raise FileNotFoundError(f"ONNX model or tokenizer missing under {root}")
        self.root = root
        self.onnx_path = onnx_path
        self.tok = Tokenizer.from_file(str(tok_path))
        pad_id = 1
        pad_token = "<pad>"
        try:
            pad = self.tok.token_to_id(pad_token)
            if pad is not None:
                pad_id = int(pad)
        except Exception:
            pass
        max_len = _max_len()
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(direction="right", pad_id=pad_id, pad_token=pad_token)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self._np = np
        probe = list(self.embed(["haunt-dim-probe"]))
        self.dim = int(len(probe[0]))
        if self.dim <= 0:
            raise RuntimeError("ONNX embedder produced an empty vector")

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed texts, returning one vector per input in the input's order.

        Inputs are regrouped by token length internally, so the order the
        model sees is not the order given here; the returned list is always
        realigned to the caller's. Raises RuntimeError if the model answers a
        batch with a different number of vectors than it was given, rather
        than returning a list with holes in it.
        """
        batch = [t if (t or "").strip() else " " for t in texts]
        if not batch:
            return []
        np = self._np
        encs = self.tok.encode_batch(batch)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        lengths = attention_mask.sum(axis=1)
        # encode_batch pads every row out to the longest row it was given, so a
        # single truncation-length text makes the model compute those columns
        # for all of its batch-mates. Grouping similar lengths and trimming
        # each group to its own width removes that work. Padding is on the
        # right, so the trim can only drop pad columns, and a masked position
        # cannot reach the pooled vector -- the numbers are unchanged.
        order = np.argsort(lengths, kind="stable")
        out: list[list[float]] = [[] for _ in batch]
        for start in range(0, len(order), ONNX_SUB_BATCH):
            rows = order[start : start + ONNX_SUB_BATCH]
            width = max(1, int(lengths[rows].max()))
            hidden = self._forward(
                input_ids[rows, :width], attention_mask[rows, :width]
            )
            # out is pre-sized, so a short batch would leave [] holes that a
            # positional zip cannot detect and that surface far from here --
            # as a dimension-0 insert, or as a falsy query vector that skips
            # vector search. The batch that dropped rows is known only here.
            if len(hidden) != len(rows):
                raise RuntimeError(
                    f"ONNX model returned {len(hidden)} vectors for a batch "
                    f"of {len(rows)}"
                )
            for row, vec in zip(rows, hidden):
                out[int(row)] = [float(x) for x in vec]
        return out

    def _forward(self, input_ids: Any, attention_mask: Any) -> Any:
        """Run one padded batch and return its pooled hidden states."""
        np = self._np
        feeds: dict[str, Any] = {}
        if "input_ids" in self.input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        if not feeds:
            names = self.input_names
            if names:
                feeds[names[0]] = input_ids
            if len(names) > 1:
                feeds[names[1]] = attention_mask
        outs = self.sess.run(None, feeds)
        by_name = {n: o for n, o in zip(self.output_names, outs)}
        hidden = None
        if "sentence_embedding" in by_name:
            hidden = by_name["sentence_embedding"]
        else:
            hidden = outs[-1] if len(outs) > 1 else outs[0]
            for o in outs:
                if getattr(o, "ndim", 0) == 2:
                    hidden = o
                    break
        if hidden is None:
            raise RuntimeError("ONNX session returned no embedding output")
        if hidden.ndim == 3:
            hidden = hidden[:, 0, :]
        return hidden


def _load_onnx_bge_m3() -> tuple[OnnxEmbedder, int]:
    root = _bge_m3_dir()
    if not _local_bge_m3_ready(root):
        models_dir().mkdir(parents=True, exist_ok=True)
        _download_bge_m3(root)
    ready = _local_bge_m3_ready(root)
    if ready is None:
        raise RuntimeError(f"BGE-M3 ONNX not ready at {root}")
    # Before onnxruntime is handed the graph: the source marker is written by
    # whoever wrote the cache, so it attests nothing on its own.
    _verify_bge_m3_cache(ready)
    model = OnnxEmbedder(ready)
    return model, _dir_bytes(ready)


def _load_fastembed(model_id: str) -> Any:
    if offline():
        raise RuntimeError("offline mode forbids embedding backend initialization")
    from fastembed import TextEmbedding

    cache = models_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(model_name=model_id, cache_dir=str(cache))


def _wants_bge_m3(requested: str) -> bool:
    n = requested.strip().lower()
    return n in {BGE_M3_ID.lower(), "bge-m3", "bge_m3", "m3"}


def _load() -> EmbedState:
    if fts_only():
        return EmbedState(
            model_id="off",
            requested=_env_model(),
            dim=0,
            available=False,
            fallback=False,
            backend="off",
            error="FTS-only (embeddings disabled)",
        )
    if offline():
        return EmbedState(
            model_id="off",
            requested=_env_model(),
            dim=0,
            available=False,
            fallback=False,
            backend="off",
            error="offline mode (vector stage not run)",
        )
    requested = _env_model()
    last_err: str | None = None

    if _wants_bge_m3(requested):
        try:
            model, nbytes = _load_onnx_bge_m3()
            _load._model = model  # type: ignore[attr-defined]
            return EmbedState(
                model_id=BGE_M3_ID,
                requested=requested,
                dim=int(model.dim),
                available=True,
                fallback=False,
                backend="onnx",
                download_bytes=nbytes,
            )
        except Exception as exc:
            last_err = str(exc)
            diag("embed_m3_onnx_failed", error=last_err, requested=requested)

        supported = _supported_fastembed()
        if BGE_M3_ID in supported:
            try:
                model = _load_fastembed(BGE_M3_ID)
                probe = list(model.embed(["haunt-dim-probe"]))
                dim = int(len(probe[0]))
                _load._model = model  # type: ignore[attr-defined]
                return EmbedState(
                    model_id=BGE_M3_ID,
                    requested=requested,
                    dim=dim,
                    available=True,
                    fallback=False,
                    backend="fastembed",
                )
            except Exception as exc:
                last_err = str(exc)
                diag("embed_m3_fastembed_failed", error=last_err)

    try:
        supported = _supported_fastembed()
        model_id = requested
        is_fallback = False
        if model_id not in supported:
            if FALLBACK_MODEL in supported:
                model_id = FALLBACK_MODEL
                is_fallback = True
            elif supported:
                model_id = next(iter(supported))
                is_fallback = True
            else:
                raise RuntimeError("fastembed has no supported text embedding models")
        model = _load_fastembed(model_id)
        probe = list(model.embed(["haunt-dim-probe"]))
        dim = int(len(probe[0]))
        _load._model = model  # type: ignore[attr-defined]
        if is_fallback:
            diag(
                "embed_fallback",
                requested=requested,
                loaded=model_id,
                dim=dim,
                reason=last_err or "requested model is not available locally",
            )
        return EmbedState(
            model_id=model_id,
            requested=requested,
            dim=dim,
            available=True,
            fallback=is_fallback,
            backend="fastembed",
            error=last_err,
        )
    except Exception as exc:
        diag("embed_unavailable", error=str(exc), requested=requested, prior=last_err)
        return EmbedState(
            model_id="off",
            requested=requested,
            dim=0,
            available=False,
            fallback=True,
            backend="none",
            error=str(exc) if last_err is None else f"{exc} (m3: {last_err})",
        )


def state() -> EmbedState:
    global _state
    with _lock:
        if _state is None:
            _state = _load()
        return _state


def reset() -> None:
    """Drop cached model (tests / home changes)."""
    global _state
    with _lock:
        _state = None
        if hasattr(_load, "_model"):
            delattr(_load, "_model")


def available() -> bool:
    return state().available


def dimension() -> int:
    return state().dim


def l2_normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return vec
    return [x / n for x in vec]


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    st = state()
    if not st.available:
        return None
    model = getattr(_load, "_model", None)
    if model is None:
        return None
    raw = model.embed(texts)
    out: list[list[float]] = []
    for vec in raw:
        out.append(l2_normalize([float(x) for x in vec]))
    return out


def embed_one(text: str) -> list[float] | None:
    if not (text or "").strip():
        text = " "
    result = embed_texts([text])
    if not result:
        return None
    return result[0]


def warmup() -> EmbedState:
    """Force model download/load. Used by bootstrap."""
    reset()
    return state()
