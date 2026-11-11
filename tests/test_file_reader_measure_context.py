"""FileReader._measure must not feed more tokens than the rotary tables allow."""

import numpy as np
import pytest
import torch

from minagi.recur import RecurCoder, RecurConfig
from minagi.stream import FileReader


@pytest.fixture(scope="module")
def tiny_model():
    cfg = RecurConfig(
        vocab_size=64,
        d_model=32,
        n_head=4,
        d_ff=64,
        n_prelude=1,
        n_recur=1,
        n_coda=1,
        max_steps=1,
        block=16,
        use_pool=False,
        tie_embeddings=True,
    )
    model = RecurCoder(cfg)
    model.eval()
    return model


def test_measure_caps_chunk_to_context(tiny_model):
    # File longer than chunk; chunk larger than the model's rotary table.
    data = np.arange(80, dtype=np.uint8) % 60
    reader = FileReader(tiny_model, data, "heldout", chunk=64, context=16, device=torch.device("cpu"))
    loss = reader.step(learn=False)
    assert loss is not None
    assert reader.pos <= 16
    assert reader.seen <= 16


def test_measure_raises_on_main_without_cap(tiny_model, monkeypatch):
    """Document the old failure: uncapped n past rope length raises ValueError."""
    data = np.arange(80, dtype=np.uint8) % 60
    reader = FileReader(tiny_model, data, "heldout", chunk=64, context=16, device=torch.device("cpu"))

    def uncapped_measure(n):
        if reader.caches is None or reader.seen + n > reader.context:
            reader.caches = reader.model.empty_caches()
            reader.seen = 0
        a = reader.data[reader.pos:reader.pos + n + 1].astype(np.int64)
        x = torch.from_numpy(a[:-1]).to(reader.device).unsqueeze(0)
        y = torch.from_numpy(a[1:]).to(reader.device).unsqueeze(0)
        from minagi.precision import amp
        with amp(reader.device):
            return reader.model(x, y, caches=reader.caches, pos_offset=reader.seen)

    with pytest.raises(ValueError, match="rotary tables"):
        uncapped_measure(64)
