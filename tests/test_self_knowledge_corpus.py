import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from corpora import build, chat, expand
from minagi.tokenizer import ByteTokenizer


def _decode_bin(path):
    ids = np.fromfile(path, dtype=np.uint16)
    return ByteTokenizer().decode(ids)


class SelfKnowledgeCorpusTest(unittest.TestCase):
    def test_default_chat_mode_still_harvests_code_pairs(self):
        pair = {"name": "answer", "doc": "Return the test answer.",
                "code": "def answer():\n    return 42"}
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "data_chat_char")
            stdout = io.StringIO()
            with mock.patch.object(chat, "harvest", return_value=[pair]) as harvest, \
                    mock.patch.object(sys, "argv", [
                        "chat.py", "--out", out, "--conversations", "2",
                        "--val", "1", "--seed", "2",
                    ]), contextlib.redirect_stdout(stdout):
                self.assertEqual(chat.main(), 0)

            harvest.assert_called_once()
            self.assertTrue(os.path.exists(os.path.join(out, "train.bin")))
            with open(os.path.join(out, "meta.json")) as f:
                meta = json.load(f)
            self.assertFalse(meta["self_only"])
            self.assertEqual(meta["pairs"], 1)

    def test_self_only_requires_yaml_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with mock.patch.object(chat, "SELF_SEED", []), \
                    mock.patch.object(chat, "harvest",
                                      side_effect=AssertionError("harvest called")), \
                    mock.patch.object(sys, "argv", [
                        "chat.py", "--out", os.path.join(tmp, "corpus"),
                        "--self-only", "--conversations", "1", "--val", "1",
                    ]), contextlib.redirect_stderr(stderr):
                self.assertEqual(chat.main(), 1)
            self.assertIn("self_knowledge.yaml", stderr.getvalue())

    def test_self_only_generation_expands_into_its_own_lane(self):
        seed = [("what are you", "I am the test model."),
                ("how do you route", "I route through test experts.")]
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "data_self_knowledge_char")
            out = os.path.join(tmp, "data")
            stdout = io.StringIO()
            with mock.patch.object(chat, "SELF_SEED", seed), \
                    mock.patch.object(chat, "harvest",
                                      side_effect=AssertionError("harvest called")), \
                    mock.patch.object(sys, "argv", [
                        "chat.py", "--out", source, "--self-only",
                        "--conversations", "6", "--val", "2", "--seed", "3",
                    ]), contextlib.redirect_stdout(stdout):
                self.assertEqual(chat.main(), 0)

            train = _decode_bin(os.path.join(source, "train.bin"))
            turns = re.findall(
                r"<user>\n.*?\n</user>\n<bot>\n(.*?)\n</bot>",
                train, flags=re.DOTALL)
            self.assertGreaterEqual(len(turns), 6)
            self.assertTrue(all(answer in {a for _, a in seed}
                                for answer in turns))
            with open(os.path.join(source, "meta.json")) as f:
                meta = json.load(f)
            self.assertTrue(meta["self_only"])
            self.assertEqual(meta["pairs"], len(seed))

            with mock.patch.object(expand, "ROOT", tmp), \
                    mock.patch.object(sys, "argv", [
                        "expand.py", "--out", out, "--only", "self-knowledge",
                        "--shard-chars", "10000",
                    ]), contextlib.redirect_stdout(stdout):
                self.assertEqual(expand.main(), 0)

            train_dir = os.path.join(out, "train", "self-knowledge")
            val_dir = os.path.join(out, "val", "self-knowledge")
            self.assertTrue(os.listdir(train_dir))
            self.assertTrue(os.listdir(val_dir))
            first = os.path.join(train_dir, sorted(os.listdir(train_dir))[0])
            with open(first) as f:
                expanded = f.read()
            self.assertIn("<user>", expanded)
            self.assertIn("<bot>", expanded)
            self.assertIn("test model", expanded)

    def test_all_builder_targets_self_knowledge_source_and_destination(self):
        with mock.patch.object(build, "_sub", side_effect=[0, 0]) as run:
            self.assertEqual(build.build_self_knowledge(250), 0)

        self.assertEqual(run.call_args_list, [
            mock.call("chat", "--out", "data_self_knowledge_char",
                      "--self-only", "--conversations", 250, "--val", 2),
            mock.call("expand", "--only", "self-knowledge"),
        ])
        self.assertEqual(build.BUILDERS["self-knowledge"][1],
                         "data/train/self-knowledge")

    def test_failed_forced_rebuild_is_not_hidden_by_stale_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "data", "train", "self-knowledge")
            os.makedirs(destination)
            with open(os.path.join(destination, "stale.txt"), "w") as f:
                f.write("old data")
            builders = dict(build.BUILDERS)
            builders["self-knowledge"] = (lambda _: 1, destination)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(build, "ROOT", tmp), \
                    mock.patch.object(build, "BUILDERS", builders), \
                    mock.patch.object(build.os, "chdir"), \
                    mock.patch.dict(sys.modules, {"datasets": mock.Mock()}), \
                    mock.patch.object(sys, "argv", [
                        "build.py", "--only", "self-knowledge", "--force",
                    ]), contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                self.assertEqual(build.main(), 1)

            self.assertIn("failed or produced nothing", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
