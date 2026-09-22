import types
import unittest
import weakref

import torch
import torch.nn as nn

import serve
from minagi.recur import RecurCoder, RecurConfig
from minagi.tokenizer import ByteTokenizer


class RecordingModel(nn.Module):
    def __init__(self, block=8):
        super().__init__()
        self.cfg = types.SimpleNamespace(block=block)
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []
        self.tokens = []
        self.full_cache_ref = None
        self.released_before_reset = None

    def empty_caches(self):
        return [{"k": None, "v": None}]

    def choose_for(self, tokens, free=False):
        return 0

    def forward(self, tokens, caches=None, pos_offset=0):
        length = tokens.shape[1]
        if pos_offset + length > self.cfg.block:
            raise AssertionError("forward exceeded rotary table")
        cache = caches[0]
        previous = 0 if cache["k"] is None else cache["k"].shape[-2]
        if pos_offset == 0 and previous == 0 and self.full_cache_ref is not None:
            self.released_before_reset = self.full_cache_ref() is None
        self.calls.append((pos_offset, length, previous))
        self.tokens.append(tokens.detach().clone())
        total = previous + length
        cache["k"] = torch.zeros(1, 1, total, 1)
        cache["v"] = torch.zeros(1, 1, total, 1)
        if total == self.cfg.block:
            self.full_cache_ref = weakref.ref(cache["k"])
        logits = torch.full((tokens.shape[0], length, 265), -100.0)
        nxt = (tokens + 1).clamp_max(255).unsqueeze(-1)
        logits.scatter_(-1, nxt, 100.0)
        return logits, None


class ServeContextTest(unittest.TestCase):
    def test_prefill_chunks_use_monotonic_positions(self):
        model = RecordingModel(block=8)
        tokens = torch.arange(8).unsqueeze(0)

        caches, logits, offset = serve._prefill_cache(model, tokens, chunk=3)

        self.assertEqual(model.calls, [(0, 3, 0), (3, 3, 3), (6, 2, 6)])
        self.assertEqual(caches[0]["k"].shape[-2], 8)
        self.assertEqual(logits.shape, (1, 2, 265))
        self.assertEqual(offset, 8)

    def test_generation_reprefills_recent_window_at_context_limit(self):
        model = RecordingModel(block=8)
        old_state = dict(serve.STATE)
        serve.STATE.update(model=model, tok=ByteTokenizer(), learner=None)
        try:
            events = list(serve.stream("abcdefgh", max_new=10))
        finally:
            serve.STATE.clear()
            serve.STATE.update(old_state)

        self.assertEqual(model.calls[0], (0, 8, 0))
        self.assertEqual(model.calls[1], (0, 4, 0))
        self.assertGreaterEqual(model.calls.count((0, 4, 0)), 2)
        self.assertEqual(model.tokens[1].tolist(), [[102, 103, 104, 105]])
        reset = [tokens.tolist() for call, tokens in zip(model.calls, model.tokens)
                 if call == (0, 4, 0)]
        self.assertIn([[107, 108, 109, 110]], reset)
        self.assertTrue(model.released_before_reset)
        self.assertTrue(all(pos + length <= model.cfg.block
                            for pos, length, _ in model.calls))
        self.assertEqual(sum("t" in event for event in events), 10)
        self.assertEqual("".join(event["t"] for event in events if "t" in event),
                         "ijklmnopqr")

    def test_chunked_prefill_matches_full_forward(self):
        torch.manual_seed(0)
        cfg = RecurConfig(vocab_size=265, d_model=8, n_head=1, d_ff=16,
                          block=8, n_prelude=1, n_recur=1, n_coda=0,
                          max_steps=1)
        model = RecurCoder(cfg).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        full = model(tokens)[0][:, -1]
        _, chunked, offset = serve._prefill_cache(model, tokens, chunk=3)

        torch.testing.assert_close(chunked[:, -1], full)
        self.assertEqual(offset, tokens.shape[1])


if __name__ == "__main__":
    unittest.main()
