"""Focused runtime regressions; no torch imports, training, or real checkpoints.

Run: python -m unittest discover -s tests -v
MINIAGI_RUNTIME_SOURCE can point at an untouched source tree for negative controls.
Only selected function definitions are compiled, with dependencies supplied below.
"""
import ast
import builtins
import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, call, patch

ROOT = Path(os.environ.get("MINIAGI_RUNTIME_SOURCE", Path(__file__).resolve().parents[1]))
SCRATCH = os.environ.get("MINIAGI_TEST_TMPDIR")


def load(relative, names, **namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    namespace.update(__file__=str(path), os=os, sys=sys, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ReplaceTests(unittest.TestCase):
    def setUp(self):
        self.clock = Mock()
        self.ns = load("minagi/store.py", {"_replace", "_savez"}, time=self.clock)
        # The baseline publishes directly through os.replace; exercise that
        # behavior too, rather than failing only because the helper is absent.
        self.ns.setdefault("_replace", lambda src, dst: os.replace(src, dst))

    def test_transient_lock(self):
        with patch.object(os, "replace", side_effect=[PermissionError(), PermissionError(), None]) as replace:
            self.ns["_replace"]("source", "target")
        self.assertEqual(replace.call_args_list, [call("source", "target")] * 3)
        self.assertEqual(self.clock.sleep.call_args_list, [call(.05), call(.1)])

    def test_exhausted_retry_preserves_error(self):
        error = PermissionError("still locked")
        with patch.object(os, "replace", side_effect=error) as replace:
            with self.assertRaises(PermissionError) as raised:
                self.ns["_replace"]("source", "target")
        self.assertIs(raised.exception, error)
        self.assertEqual(replace.call_count, 8)
        self.assertEqual(self.clock.sleep.call_args_list,
                         [call(.05 * 2 ** i) for i in range(7)])

    def test_other_errors_not_retried(self):
        with patch.object(os, "replace", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                self.ns["_replace"]("source", "target")
        self.clock.sleep.assert_not_called()

    def test_all_checkpoint_publish_sites_use_retry(self):
        tree = ast.parse((ROOT / "minagi/store.py").read_text(encoding="utf-8"))
        for name in ("save", "_save_paged", "_savez"):
            func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n.func.id for n in ast.walk(func)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertIn("_replace", calls, name)
        arrays = {"a": Mock(nbytes=12)}
        self.ns.update(np=Mock(), _replace=Mock())
        self.assertEqual(self.ns["_savez"]("bundle.npz", arrays), 12)
        self.ns["_replace"].assert_called_once_with("bundle.npz.tmp.npz", "bundle.npz")

    @unittest.skipUnless(sys.platform == "win32", "Windows delete-sharing semantics")
    def test_real_windows_reader_lock(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            src, dst = Path(directory) / "new", Path(directory) / "checkpoint"
            src.write_bytes(b"new")
            dst.write_bytes(b"old")
            handle = kernel.CreateFileW(str(dst), 0x80000000, 1, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            closed = threading.Event()
            def release():
                kernel.CloseHandle(handle)
                closed.set()
            timer = threading.Timer(.15, release)
            try:
                with self.assertRaises(PermissionError):
                    os.replace(src, dst)
                self.assertEqual(dst.read_bytes(), b"old")
                timer.start()
                self.ns["time"] = time
                self.ns["_replace"](src, dst)
                self.assertTrue(closed.is_set())
                self.assertEqual(dst.read_bytes(), b"new")
                self.assertFalse(src.exists())
            finally:
                if timer.ident is not None:
                    timer.join()
                else:
                    release()


class LogTests(unittest.TestCase):
    def test_unicode_samples_under_legacy_locale(self):
        ns = load("train.py", {"write_samples"}, repeat_rate=lambda text: 0,
                  chess_legality=lambda *args: None, refresh_plots=Mock())
        real_open = builtins.open
        def legacy_open(path, mode, **kw):
            kw.setdefault("encoding", "cp1252")
            return real_open(path, mode, **kw)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            path = Path(directory) / "samples.txt"
            with patch.dict(ns, open=legacy_open):
                ns["write_samples"](path, 1, 10, 1,
                    [("unicode", "prompt", "\u6f22\U0001f600\ufffd\ud800")], val=1., level=.2)
            text = path.read_text(encoding="utf-8")
            self.assertIn("\u6f22\U0001f600\ufffd?", text)
            self.assertIn("level +0.2000", text)

    def test_all_four_append_writers_are_explicit_utf8(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        writers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "open"
                   and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)
                   and n.args[1].value == "a"
                   # rates_path is an unrelated ASCII-only numeric telemetry file.
                   and ast.unparse(n.args[0]) != "rates_path"]
        self.assertEqual(len(writers), 4)
        for node in writers:
            options = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            self.assertEqual(options.get("encoding"), "utf-8", f"line {node.lineno}")


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.state = dict(procs=[], proc=None, on=True, since=123,
                          log='C:/a b/sample & log.txt', weights='C:/a b/weights')
        self.ns = load("train.py", {"refresh_plots"}, _PLOTS=self.state)
        self.exists = patch.object(os.path, "exists", return_value=True)
        self.exists.start()
        self.addCleanup(self.exists.stop)

    def test_direct_argv_and_all_children_gate_refresh(self):
        first, second = Mock(), Mock()
        first.poll.return_value = 0
        second.poll.return_value = None
        with patch.object(subprocess, "Popen", side_effect=[first, second]) as spawn:
            self.ns["refresh_plots"]()
            self.assertEqual(spawn.call_count, 2)
            for entry in spawn.call_args_list:
                argv = entry.args[0]
                self.assertEqual(argv[0], sys.executable)
                self.assertNotIn("shell", entry.kwargs)
                self.assertEqual(argv[argv.index("--log") + 1], self.state["log"])
                self.assertEqual(entry.kwargs["cwd"], str(ROOT))
            dashboard = spawn.call_args_list[1].args[0]
            self.assertEqual(dashboard[-2:], ["--last", "123"])
            self.assertEqual(dashboard[dashboard.index("--weights") + 1], self.state["weights"])
            self.ns["refresh_plots"]()
            self.assertEqual(spawn.call_count, 2)
            second.poll.return_value = 0
            spawn.side_effect = [Mock(), Mock()]
            self.ns["refresh_plots"]()
            self.assertEqual(spawn.call_count, 4)

    def test_spawn_failure_retains_already_started_child_and_disables(self):
        child = Mock()
        with patch.object(subprocess, "Popen", side_effect=[child, OSError("unavailable")]) as spawn:
            self.ns["refresh_plots"]()
            self.assertFalse(self.state["on"])
            self.assertEqual(self.state["procs"], [child])
            self.ns["refresh_plots"]()
            self.assertEqual(spawn.call_count, 2)

    def test_disabled_and_missing_tool_do_not_spawn(self):
        with patch.object(subprocess, "Popen") as spawn:
            self.state["on"] = False
            self.ns["refresh_plots"]()
            self.state["on"] = True
            with patch.object(os.path, "exists", return_value=False):
                self.ns["refresh_plots"]()
            spawn.assert_not_called()
            self.assertFalse(self.state["on"])


if __name__ == "__main__":
    unittest.main()
