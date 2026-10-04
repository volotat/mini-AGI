"""A tamper-evident, append-only receipt ledger.

Each entry is one JSON line; entry `hash = sha256(prev_hash + canonical(entry))`.
`verify()` replays the chain and reports the first mismatch. This is the
"record it" step of governance made concrete: nothing the agent did can be
silently rewritten after the fact.
"""

import hashlib
import json
import os

GENESIS = "0" * 64


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


class ReceiptLedger:
    def __init__(self, path):
        self.path = path
        self.entries = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))

    def _entry(self, kind, payload):
        prev = self.entries[-1]["hash"] if self.entries else GENESIS
        e = {"kind": kind, "payload": payload, "prev_hash": prev}
        e["hash"] = _digest(prev.encode("ascii") + _canonical(e))
        return e

    def append(self, kind, payload):
        e = self._entry(kind, payload)
        self.entries.append(e)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(e, sort_keys=True) + "\n")
        except OSError as err:
            import sys
            print(f"[ledger] write failed: {err}", file=sys.stderr)
        return e

    def verify(self):
        prev = GENESIS
        for i, e in enumerate(self.entries):
            body = {k: v for k, v in e.items() if k != "hash"}
            expect = _digest(prev.encode("ascii") + _canonical(body))
            if e.get("hash") != expect:
                return False, f"entry {i} hash mismatch"
            if e.get("prev_hash") != prev:
                return False, f"entry {i} prev_hash mismatch"
            prev = e["hash"]
        return True, "ok"
