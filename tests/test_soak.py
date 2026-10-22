"""M7: a save interrupted before its manifest must leave a loadable checkpoint."""

import errno
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

import train as T
from minagi import store
from minagi.create import create
from minagi.live import LiveLearner
from minagi.recur import RecurCoder, RecurConfig


def _tiny(tmp):
    path = os.path.join(tmp, "weights")
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=16, max_steps=1,
           top_k=1)
    model, _cfg, _pool, man = T.build_paged(path, "cpu", resident=1,
                                            ram_capacity=2)
    model.eval()
    return path, model, man


def _nonpaged(tmp):
    path = os.path.join(tmp, "weights")
    d = create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
               d_model=8, trunk_d_ff=16, n_head=1, block=16, max_steps=1,
               top_k=1)
    cfg = RecurConfig(**{k: v for k, v in d.items()
                         if k in RecurConfig.__dataclass_fields__})
    return path, RecurCoder(cfg), d


def _step(path):
    with open(os.path.join(path, "manifest.json")) as f:
        return json.load(f)["step"]


class InterruptedSaveTest(unittest.TestCase):
    def test_disk_full_keeps_old_manifest_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            store.save(model, path, step=1, cfg=man.get("cfg"))
            real = store._savez
            calls = {"n": 0}

            def full(p, arrays):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError(errno.ENOSPC, "No space left on device")
                return real(p, arrays)

            with mock.patch.object(store, "_savez", side_effect=full):
                with self.assertRaises(OSError):
                    store.save(model, path, step=2)

            self.assertEqual(_step(path), 1)
            self.assertFalse(os.path.exists(os.path.join(
                path, "manifest.json.tmp")))
            fresh, _cfg, _pool, _man = T.build_paged(
                path, "cpu", resident=1, ram_capacity=2)
            self.assertTrue(fresh is not None)

            # the failed attempt does not consume the learner's save slot
            store.save(model, path, step=2, cfg=man.get("cfg"))
            self.assertEqual(_step(path), 2)

    def test_manifest_failure_does_not_delete_pruned_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, cfg = _nonpaged(tmp)
            store.save(model, path, step=1, cfg=cfg)
            experts = os.path.join(path, "experts")
            extra = os.path.join(experts, "e99999.npz")
            shutil.copy(os.path.join(experts, "e00000.npz"), extra)
            real_replace = store.os.replace

            def break_manifest(src, dst):
                if os.path.basename(str(dst)) == "manifest.json":
                    raise OSError(errno.EIO, "interrupted manifest replace")
                return real_replace(src, dst)

            with mock.patch.object(store.os, "replace",
                                   side_effect=break_manifest):
                with self.assertRaises(OSError):
                    store.save(model, path, step=2)

            self.assertEqual(_step(path), 1)
            self.assertTrue(os.path.exists(extra),
                            "stale expert was deleted before the manifest")
            fresh, _cfg, _pool, _man = T.build_paged(
                path, "cpu", resident=1, ram_capacity=2)
            self.assertTrue(fresh is not None)

            store.save(model, path, step=2, cfg=cfg)
            self.assertEqual(_step(path), 2)
            self.assertFalse(os.path.exists(extra))

    def test_live_learner_retries_a_failed_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            learner = LiveLearner(model, chunk=16, context=16,
                                  weights_dir=path, manifest=man,
                                  save_every=0)
            learner.unsaved = 1
            with mock.patch.object(store, "save",
                                   side_effect=OSError(errno.ENOSPC, "full")):
                with self.assertRaises(OSError):
                    learner.save()
            self.assertEqual(learner.unsaved, 1)
            learner.save()
            self.assertEqual(learner.unsaved, 0)
            self.assertTrue(os.path.exists(os.path.join(path, "manifest.json")))


if __name__ == "__main__":
    unittest.main()
