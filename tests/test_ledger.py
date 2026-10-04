import os
import json
import tempfile
import unittest

from minagi import ledger


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "receipts.jsonl")

    def test_append_and_verify_ok(self):
        lg = ledger.ReceiptLedger(self.path)
        lg.append("directive", {"lr": 1.05})
        lg.append("tool", {"name": "maat_screen"})
        ok, msg = lg.verify()
        self.assertTrue(ok, msg)
        self.assertEqual(len(lg.entries), 2)

    def test_chain_links_previous_hash(self):
        lg = ledger.ReceiptLedger(self.path)
        a = lg.append("a", {"x": 1})
        b = lg.append("b", {"y": 2})
        self.assertEqual(b["prev_hash"], a["hash"])

    def test_tamper_is_detected(self):
        lg = ledger.ReceiptLedger(self.path)
        lg.append("a", {"x": 1})
        lg.append("b", {"y": 2})
        with open(self.path) as f:
            lines = f.readlines()
        rec = json.loads(lines[0])
        rec["payload"]["x"] = 999
        lines[0] = json.dumps(rec) + "\n"
        with open(self.path, "w") as f:
            f.writelines(lines)
        ok, msg = ledger.ReceiptLedger(self.path).verify()
        self.assertFalse(ok)

    def test_persists_and_reloads(self):
        lg = ledger.ReceiptLedger(self.path)
        lg.append("a", {"x": 1})
        lg2 = ledger.ReceiptLedger(self.path)
        self.assertEqual(len(lg2.entries), 1)
        self.assertEqual(lg2.entries[0]["payload"], {"x": 1})

    def test_append_missing_dir_does_not_raise(self):
        import os
        bad = os.path.join(self.dir, "no", "such", "dir", "receipts.jsonl")
        lg = ledger.ReceiptLedger(bad)
        e = lg.append("a", {"x": 1})     # must not raise
        self.assertEqual(e["kind"], "a")


if __name__ == "__main__":
    unittest.main()
