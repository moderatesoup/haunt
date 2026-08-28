"""OnnxEmbedder.embed regroups its input by token length before running it.

The reordering is invisible to callers by contract: embed_texts() returns a
list positionally matched to its input, and process_embedding_jobs maps those
vectors back to memory_id by position, so a permutation bug would attach the
wrong vector to the wrong memory rather than fail loudly.

These tests drive a fake tokenizer and a fake ONNX session, so they exercise
the regrouping, the per-group padding trim and the realignment without
needing a real model. The fake session derives each vector only from its
unmasked tokens, which is the property the trim relies on, and records the
padded width it was handed so the trim itself is observable.

A model that answers a batch with the wrong number of vectors is the other
half of that contract: the realignment list is pre-sized, so the mismatch has
to raise here rather than leave holes for a later caller to trip over.
"""

from __future__ import annotations

import pytest

from haunt.embed import ONNX_SUB_BATCH, EmbedState, OnnxEmbedder

np = pytest.importorskip("numpy")

MAX_LEN = 12
PAD_ID = 1
FAKE_STATE = EmbedState(
    model_id="embed-batching-test-model",
    requested="embed-batching-test-model",
    dim=4,
    available=True,
    fallback=False,
)


class _FakeEncoding:
    def __init__(self, ids: list[int], attention_mask: list[int]):
        self.ids = ids
        self.attention_mask = attention_mask


class _FakeTokenizer:
    """One token per whitespace word, truncated then right-padded to the batch."""

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoding]:
        rows = []
        for text in texts:
            ids = [ord(w[0]) * 31 + len(w) for w in text.split()][:MAX_LEN]
            rows.append(ids or [PAD_ID])
        width = max(len(r) for r in rows)
        return [
            _FakeEncoding(
                r + [PAD_ID] * (width - len(r)), [1] * len(r) + [0] * (width - len(r))
            )
            for r in rows
        ]


class _FakeSession:
    """Reads only unmasked tokens, and records each batch's padded width."""

    def __init__(self) -> None:
        self.widths: list[int] = []

    def run(self, _outputs, feeds):
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        self.widths.append(int(ids.shape[1]))
        rows = []
        for row_ids, row_mask in zip(ids, mask):
            real = [int(i) for i, m in zip(row_ids, row_mask) if m]
            rows.append(_summarize(real))
        return [np.array(rows, dtype=np.float32)]


def _summarize(ids: list[int]) -> list[float]:
    if not ids:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(sum(ids)), float(len(ids)), float(ids[0]), float(ids[-1])]


def _embedder() -> OnnxEmbedder:
    e = OnnxEmbedder.__new__(OnnxEmbedder)
    e.tok = _FakeTokenizer()
    e.sess = _FakeSession()
    e.input_names = ["input_ids", "attention_mask"]
    e.output_names = ["last_hidden_state"]
    e._np = np
    return e


def _expected(text: str) -> list[float]:
    ids = [ord(w[0]) * 31 + len(w) for w in (text if text.strip() else " ").split()]
    return _summarize(ids[:MAX_LEN] or [PAD_ID])


def _varied() -> list[str]:
    """Lengths deliberately out of order, spanning several sub-batches."""
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
    texts = []
    for i in range(40):
        n = (i * 7) % 15 + 1
        texts.append(" ".join(words[j % len(words)] + str(i) for j in range(n)))
    return texts


def test_embed_returns_vectors_in_input_order():
    texts = _varied()
    out = _embedder().embed(texts)
    assert len(out) == len(texts)
    assert out == [_expected(t) for t in texts]


def test_embed_is_unchanged_by_regrouping(monkeypatch):
    """One batch for everything must give the same vectors as many groups."""
    texts = _varied()
    monkeypatch.setattr("haunt.embed.ONNX_SUB_BATCH", len(texts) * 2)
    reference = _embedder().embed(texts)
    monkeypatch.setattr("haunt.embed.ONNX_SUB_BATCH", 16)
    grouped = _embedder().embed(texts)
    assert np.allclose(
        np.array(reference), np.array(grouped), rtol=0.0, atol=1e-6
    )


def test_embed_trims_padding_to_each_group():
    texts = _varied()
    e = _embedder()
    e.embed(texts)
    lengths = sorted(min(len(t.split()), MAX_LEN) for t in texts)
    groups = [
        lengths[i : i + ONNX_SUB_BATCH]
        for i in range(0, len(lengths), ONNX_SUB_BATCH)
    ]
    assert len(groups) > 1
    assert e.sess.widths == [max(g) for g in groups]
    # Without the regrouping every row would be padded to the longest text.
    assert min(e.sess.widths) < max(lengths)


def test_embed_handles_empty_input():
    assert _embedder().embed([]) == []


def test_embed_handles_single_text():
    out = _embedder().embed(["one single line of text"])
    assert out == [_expected("one single line of text")]


def test_embed_handles_uniform_lengths():
    texts = [f"word{i} word{i} word{i}" for i in range(20)]
    e = _embedder()
    out = e.embed(texts)
    assert out == [_expected(t) for t in texts]
    assert set(e.sess.widths) == {3}


def test_embed_handles_blank_and_overlong_texts():
    long_text = " ".join(f"tok{i}" for i in range(MAX_LEN * 3))
    texts = ["", "   ", "\n\t", "short one", long_text]
    out = _embedder().embed(texts)
    assert out == [_expected(t) for t in texts]
    # Blanks become a single space rather than a zero-width row.
    assert all(vec[1] >= 1.0 for vec in out)
    # The overlong text is truncated, not passed through whole.
    assert out[-1][1] == float(MAX_LEN)


def test_embed_texts_and_embed_one_preserve_the_contract(monkeypatch):
    from haunt import embed as embed_mod

    monkeypatch.setattr(embed_mod, "_state", FAKE_STATE)
    monkeypatch.setattr(embed_mod._load, "_model", _embedder(), raising=False)
    texts = _varied()
    vectors = embed_mod.embed_texts(texts)
    assert vectors is not None and len(vectors) == len(texts)
    for text, vector in zip(texts, vectors):
        assert np.allclose(
            np.array(vector),
            embed_mod.l2_normalize(_expected(text)),
            rtol=0.0,
            atol=1e-6,
        )
    assert np.allclose(
        np.array(embed_mod.embed_one(texts[3])),
        np.array(vectors[3]),
        rtol=0.0,
        atol=1e-6,
    )


def test_embed_rejects_a_batch_the_model_answered_short():
    e = _embedder()
    full_run = e.sess.run

    def drop_last(outputs, feeds):
        return [full_run(outputs, feeds)[0][:-1]]

    e.sess.run = drop_last
    with pytest.raises(RuntimeError, match="returned 15 vectors for a batch of 16"):
        e.embed(_varied())


def test_embed_rejects_a_batch_the_model_answered_long():
    e = _embedder()
    full_run = e.sess.run

    def repeat_first(outputs, feeds):
        rows = full_run(outputs, feeds)[0]
        return [np.concatenate([rows, rows[:1]])]

    e.sess.run = repeat_first
    with pytest.raises(RuntimeError, match="returned 17 vectors for a batch of 16"):
        e.embed(_varied())
