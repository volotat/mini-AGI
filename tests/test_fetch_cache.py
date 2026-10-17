import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from corpora import fetch


def _argv(out, *extra):
    return ["fetch.py", "--dataset", "example/data", "--kind", "text",
            "--out", out, "--group", "40", *extra]


class FetchCacheTest(unittest.TestCase):
    def _owned_cache(self, root):
        path = os.path.join(root, "owned-datasets-cache")

        def make(prefix, dir):
            self.assertEqual(prefix, ".minagi-hf-datasets-")
            self.assertEqual(os.path.abspath(dir), os.path.abspath(root))
            os.makedirs(path)
            return path

        return path, make

    def test_default_uses_and_removes_only_owned_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            shared = os.path.join(tmp, ".cache", "huggingface", "datasets")
            os.makedirs(shared)
            sentinel = os.path.join(shared, "unrelated.arrow")
            with open(sentinel, "w") as f:
                f.write("keep")
            owned, make_cache = self._owned_cache(out)
            called = {}

            def load_dataset(name, config, **kwargs):
                called.update(name=name, config=config, **kwargs)
                with open(os.path.join(kwargs["cache_dir"], "owned.arrow"), "w") as f:
                    f.write("temporary")
                return [{"text": "a document long enough to become training text"}]

            module = types.SimpleNamespace(load_dataset=load_dataset)
            stdout = io.StringIO()
            with mock.patch.dict(sys.modules, {"datasets": module}), \
                    mock.patch.object(fetch.tempfile, "mkdtemp",
                                      side_effect=make_cache), \
                    mock.patch.object(sys, "argv", _argv(out)), \
                    mock.patch.dict(os.environ, {"HOME": tmp}), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(fetch.main(), 0)

            self.assertEqual(called["cache_dir"], owned)
            self.assertFalse(os.path.exists(owned))
            self.assertTrue(os.path.exists(sentinel))
            self.assertTrue(os.path.exists(
                os.path.join(out, "0000", "part-000000.txt")))
            self.assertIn("temporary dataset cache", stdout.getvalue())

    def test_keep_cache_uses_library_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            called = {}

            def load_dataset(name, config, **kwargs):
                called.update(kwargs)
                return [{"text": "a document long enough to become training text"}]

            module = types.SimpleNamespace(load_dataset=load_dataset)
            with mock.patch.dict(sys.modules, {"datasets": module}), \
                    mock.patch.object(fetch.tempfile, "mkdtemp",
                                      side_effect=AssertionError("cache allocated")), \
                    mock.patch.object(sys, "argv", _argv(
                        os.path.join(tmp, "out"), "--keep-cache")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fetch.main(), 0)

            self.assertNotIn("cache_dir", called)

    def test_owned_cache_is_removed_when_loading_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            owned, make_cache = self._owned_cache(out)

            def load_dataset(name, config, **kwargs):
                with open(os.path.join(kwargs["cache_dir"], "partial"), "w") as f:
                    f.write("partial")
                raise RuntimeError("download failed")

            module = types.SimpleNamespace(load_dataset=load_dataset)
            with mock.patch.dict(sys.modules, {"datasets": module}), \
                    mock.patch.object(fetch.tempfile, "mkdtemp",
                                      side_effect=make_cache), \
                    mock.patch.object(sys, "argv", _argv(out)), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    fetch.main()

            self.assertFalse(os.path.exists(owned))

    def test_streaming_cleans_cache_before_immediate_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            owned, make_cache = self._owned_cache(out)

            def load_dataset(name, config, **kwargs):
                return [{"text": "a document long enough to become training text"}]

            module = types.SimpleNamespace(load_dataset=load_dataset)
            with mock.patch.dict(sys.modules, {"datasets": module}), \
                    mock.patch.object(fetch.tempfile, "mkdtemp",
                                      side_effect=make_cache), \
                    mock.patch.object(fetch.os, "_exit",
                                      side_effect=SystemExit(0)) as exit_now, \
                    mock.patch.object(sys, "argv", _argv(out, "--streaming")), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    fetch.main()

            exit_now.assert_called_once_with(0)
            self.assertFalse(os.path.exists(owned))

    def test_cleanup_failure_prevents_successful_streaming_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            _, make_cache = self._owned_cache(out)

            def load_dataset(name, config, **kwargs):
                return [{"text": "a document long enough to become training text"}]

            module = types.SimpleNamespace(load_dataset=load_dataset)
            with mock.patch.dict(sys.modules, {"datasets": module}), \
                    mock.patch.object(fetch.tempfile, "mkdtemp",
                                      side_effect=make_cache), \
                    mock.patch.object(fetch.shutil, "rmtree",
                                      side_effect=OSError("cache is busy")), \
                    mock.patch.object(fetch.os, "_exit") as exit_now, \
                    mock.patch.object(sys, "argv", _argv(out, "--streaming")), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, "cache is busy"):
                    fetch.main()

            exit_now.assert_not_called()

    def test_conversion_and_cleanup_failures_are_both_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            _, make_cache = self._owned_cache(out)

            def load_dataset(name, config, **kwargs):
                raise ValueError("bad dataset")

            module = types.SimpleNamespace(load_dataset=load_dataset)
            with mock.patch.dict(sys.modules, {"datasets": module}), \
                    mock.patch.object(fetch.tempfile, "mkdtemp",
                                      side_effect=make_cache), \
                    mock.patch.object(fetch.shutil, "rmtree",
                                      side_effect=OSError("cache is busy")), \
                    mock.patch.object(sys, "argv", _argv(out)), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                        RuntimeError,
                        "bad dataset.*cache cleanup also failed.*cache is busy") as raised:
                    fetch.main()

            self.assertIsInstance(raised.exception.__cause__, ValueError)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_cache_path_is_resolved_before_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "target")
            os.makedirs(target)
            out = os.path.join(tmp, "out-link")
            os.symlink(target, out)
            owned, make_cache = self._owned_cache(target)

            def load_dataset(name, config, **kwargs):
                os.unlink(out)
                os.makedirs(out)
                return [{"text": "a document long enough to become training text"}]

            module = types.SimpleNamespace(load_dataset=load_dataset)
            with mock.patch.dict(sys.modules, {"datasets": module}), \
                    mock.patch.object(fetch.tempfile, "mkdtemp",
                                      side_effect=make_cache), \
                    mock.patch.object(sys, "argv", _argv(out)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fetch.main(), 0)

            self.assertFalse(os.path.exists(owned))


if __name__ == "__main__":
    unittest.main()
