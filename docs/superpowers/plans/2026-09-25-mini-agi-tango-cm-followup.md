# mini-AGI — Tango Maps + CM Health Certificate (Independence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the local agent independently auditable and interoperable: a tamper-evident receipt ledger, a generic HTTP/MCP tool bridge, and a complete-monotonicity health certificate reusing the machine-checked `cos`-obstruction from the RH work.

**Architecture:** Three pure-Python stdlib-only modules (`ledger.py`, `http_tool.py`, `health.py`), each unit-testable without torch/weights. Wired into the existing `serve.py`/`train.py` loops (tool calls, selfdir directives, checkpoint health) as advisory, never-raising additions. The CM certificate reuses only the *verified* complete-monotonicity algebra — it makes no RH claim.

**Tech Stack:** Python 3.10+, stdlib only (`hashlib`, `json`, `urllib.request`, `unittest`).

## Global Constraints

- No tokenizer/vocab change; markers stay literal ASCII.
- No new runtime dependency — stdlib only.
- Advisory and never-raising: a bad hash, a failed HTTP call, or a non-CM sequence must produce a record, never a crash.
- The receipt ledger is append-only and tamper-evident: each entry chains `sha256(prev_hash + canonical_json(entry))`.
- The CM certificate is a *diagnostic*, not a control: it never mutates the model, only emits a governance record.
- RH is NOT claimed anywhere: the health module cites the Lean theorem as the reference for the discrete-CM criterion, with no statement about RH.

---

### Task 1: `minagi/ledger.py` — hashed append-only receipt ledger

**Files:**
- Create: `minagi/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Produces: `class ReceiptLedger` with `__init__(path)`, `.append(kind, payload) -> dict` (returns the recorded entry), `.verify() -> (bool, str)` (checks the whole chain; `(True, "ok")` or `(False, reason)`), `.entries` (list). Module function `_canonical(obj) -> bytes` (sorted-key JSON, UTF-8, separators compact) and `_digest(data: bytes) -> str` (sha256 hex).

- [ ] **Step 1: Write the failing test**

`tests/test_ledger.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify RED**

Run: `python -m unittest tests.test_ledger -v`
Expected: FAIL (ModuleNotFoundError: minagi.ledger).

- [ ] **Step 3: Implement**

`minagi/ledger.py`:

```python
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
        with open(self.path, "a") as f:
            f.write(json.dumps(e, sort_keys=True) + "\n")
        return e

    def verify(self):
        prev = GENESIS
        for i, e in enumerate(self.entries):
            expect = _digest(prev.encode("ascii") + _canonical(e))
            if e.get("hash") != expect:
                return False, f"entry {i} hash mismatch"
            if e.get("prev_hash") != prev:
                return False, f"entry {i} prev_hash mismatch"
            prev = e["hash"]
        return True, "ok"
```

- [ ] **Step 4: Run to verify GREEN**

Run: `python -m unittest tests.test_ledger -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add minagi/ledger.py tests/test_ledger.py
git commit -m "feat(governance): hashed append-only receipt ledger"
```

---

### Task 2: `minagi/http_tool.py` — generic HTTP/MCP bridge

**Files:**
- Create: `minagi/http_tool.py`
- Test: `tests/test_http_tool.py`

**Interfaces:**
- Produces: `build_http_tool(name, spec) -> tools.Tool` where `spec` is a dict with `url`, `method` (default `POST`), `headers` (dict, optional), `timeout` (float, optional). `register_http_tools(registry, cfg, timeout=10.0) -> None` reads the `interop.tools` mapping (name → spec) from config and registers each as a `Tool` named after the key. The tool's `call(args)` POSTs (or GETs) `args` as JSON and returns the parsed JSON response; on any error returns `{"ok": False, "error": ...}`.

- [ ] **Step 1: Write the failing test**

`tests/test_http_tool.py`:

```python
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from minagi import http_tool, tools


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"echo": body}).encode())
    def log_message(self, *a):
        pass


class HttpToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_post_round_trip(self):
        t = http_tool.build_http_tool("echo", {
            "url": f"http://127.0.0.1:{self.port}/x", "method": "POST"})
        out = t.call({"a": 1})
        self.assertTrue(out.get("ok"))
        self.assertEqual(out["echo"], {"a": 1})

    def test_unreachable_returns_error(self):
        t = http_tool.build_http_tool("down", {
            "url": "http://127.0.0.1:1/nope", "timeout": 0.5})
        out = t.call({})
        self.assertFalse(out.get("ok"))
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify RED**

Run: `python -m unittest tests.test_http_tool -v`
Expected: FAIL (ModuleNotFoundError: minagi.http_tool).

- [ ] **Step 3: Implement**

`minagi/http_tool.py`:

```python
"""A generic HTTP/JSON tool, so the agent can reach external systems.

Reads `interop.tools` (name -> {url, method, headers, timeout}) from config and
registers each as a tool the model can call. This is the "any agent, any
harness" bridge: point it at the live Mekhat MCP or another agent's endpoint.
"""

import json
import urllib.error
import urllib.request

from .tools import Tool


def build_http_tool(name, spec):
    url = spec.get("url")
    method = (spec.get("method") or "POST").upper()
    headers = spec.get("headers") or {}
    timeout = float(spec.get("timeout") or 10.0)

    def call(args):
        data = json.dumps(args).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        hdrs.update(headers)
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8")
            try:
                out = json.loads(body)
            except ValueError:
                out = {"ok": True, "body": body}
            if isinstance(out, dict):
                out.setdefault("ok", True)
                return out
            return {"ok": True, "value": out}
        except Exception as e:               # noqa: BLE001
            return {"ok": False, "error": str(e)}

    return Tool(name, f"HTTP {method} {url}", spec, call)


def register_http_tools(registry, cfg, timeout=10.0):
    mapping = cfg.get("tools") or {}
    for name, spec in mapping.items():
        spec = dict(spec)
        spec.setdefault("timeout", timeout)
        registry.register(build_http_tool(name, spec))
```

- [ ] **Step 4: Run to verify GREEN**

Run: `python -m unittest tests.test_http_tool -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add minagi/http_tool.py tests/test_http_tool.py
git commit -m "feat(interop): generic HTTP/JSON tool bridge"
```

---

### Task 3: `minagi/health.py` — discrete complete-monotonicity certificate

**Files:**
- Create: `minagi/health.py`
- Test: `tests/test_health.py`

**Interfaces:**
- Produces: `discrete_cm(values: list[float], eps=1e-9) -> dict` returning `{"cm": bool, "first_flip": int | None, "depth": int, "verdict": str}`. A sequence is discretely completely monotone if `(-1)^k Δ^k a_n ≥ 0` for all reachable `k, n` (alternating-sign finite differences). The `cos`-obstruction reference: an oscillatory component forces a sign violation at some `k`. `certify(values, eps=1e-9) -> dict` wraps `discrete_cm` with a human verdict string (`"no-oscillation (CM)"` / `"oscillation detected at diff k"`). Both are pure functions; neither raises on empty/constant input.

- [ ] **Step 1: Write the failing test**

`tests/test_health.py`:

```python
import unittest

from minagi import health


class DiscreteCmTests(unittest.TestCase):
    def test_geometric_decay_is_cm(self):
        # r^n, 0<r<1: loss decaying to zero with no oscillation
        v = [0.5 ** n for n in range(8)]
        self.assertTrue(health.discrete_cm(v)["cm"])

    def test_constant_is_cm(self):
        self.assertTrue(health.discrete_cm([0.7] * 6)["cm"])

    def test_oscillation_is_not_cm(self):
        # alternating / bouncing sequence -> cos-obstruction
        v = [1.0, 0.2, 1.0, 0.2, 1.0, 0.2]
        self.assertFalse(health.discrete_cm(v)["cm"])

    def test_regress_then_improve_is_not_cm(self):
        v = [0.8, 0.7, 0.6, 0.5, 0.9, 0.85]   # dips then jumps back up
        self.assertFalse(health.discrete_cm(v)["cm"])

    def test_empty_and_short_do_not_raise(self):
        self.assertTrue(health.discrete_cm([])["cm"])
        self.assertTrue(health.discrete_cm([0.5])["cm"])

    def test_certify_verdict(self):
        self.assertIn("oscillation", health.certify([1, 0, 1, 0])["verdict"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify RED**

Run: `python -m unittest tests.test_health -v`
Expected: FAIL (ModuleNotFoundError: minagi.health).

- [ ] **Step 3: Implement**

`minagi/health.py`:

```python
"""A complete-monotonicity health certificate for the agent's own trajectories.

Reuses the *verified* complete-monotonicity algebra from the RH work
(05_research/RH/lean4/CmRhCore/Basic.lean): exp(-mu x) is completely monotone,
cos(b x) is NOT, and finite non-negative combinations of CM functions are CM.
The discrete analogue holds for sequences: a_n = r^n (0<r<1) is discretely CM;
an oscillatory (cos-like) component breaks the alternating-sign condition.

A CM loss trajectory is a non-negative combination of geometric decays: it
monotonically improves and cannot bounce back. An oscillation flag is the
"cos-obstruction": a sign violation in the alternating finite differences. This
is a diagnostic only — it never mutates the model.
"""


def _differences(v):
    cur = [float(x) for x in v]
    levels = [cur]
    while len(cur) > 1:
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
        levels.append(cur)
    return levels


def discrete_cm(values, eps=1e-9):
    """(-1)^k * Delta^k a_n >= 0 for all k, n (alternating-sign differences)."""
    v = [float(x) for x in values]
    if len(v) <= 1:
        return {"cm": True, "first_flip": None, "depth": 0}
    levels = _differences(v)
    for k, level in enumerate(levels):
        sign = 1 if k % 2 == 0 else -1
        for n, x in enumerate(level):
            if sign * x < -eps:
                return {"cm": False, "first_flip": k,
                        "depth": len(levels) - 1}
    return {"cm": True, "first_flip": None, "depth": len(levels) - 1}


def certify(values, eps=1e-9):
    r = discrete_cm(values, eps)
    if r["cm"]:
        r["verdict"] = "no-oscillation (CM)"
    else:
        r["verdict"] = f"oscillation detected at diff {r['first_flip']}"
    return r
```

- [ ] **Step 4: Run to verify GREEN**

Run: `python -m unittest tests.test_health -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add minagi/health.py tests/test_health.py
git commit -m "feat(governance): discrete complete-monotonicity health certificate"
```

---

### Task 4: Wire ledger + health into the loops

**Files:**
- Modify: `serve.py` (tool calls → ledger)
- Modify: `train.py` (selfdir directives + checkpoint health → ledger)
- Modify: `minagi/http_tool.py` is already importable; `serve.py` registers HTTP tools from config at startup
- Test: `tests/test_ledger.py` / `tests/test_health.py` unchanged (wiring is integration; verified by py_compile + review)

**Interfaces:**
- Consumes: `ledger.ReceiptLedger`, `health.certify`, `http_tool.register_http_tools`.
- Produces: `serve.py` builds a `ReceiptLedger(receipts_path)` and registers HTTP tools; `train.py` builds a ledger and, at the growth cadence, appends a `health` entry from `health.certify` over the recent held-out history.

- [ ] **Step 1: serve.py — ledger + HTTP tools**

At module scope near `REGISTRY` (Task 4 of the prior plan), add:

```python
from minagi import ledger as ledger_mod
from minagi import http_tool as http_tool_mod

RECEIPTS = ledger_mod.ReceiptLedger(os.path.join("runs", "receipts.jsonl"))
```

In `main()`, after `set_compute_dtype(...)`, add:

```python
    http_tool_mod.register_http_tools(REGISTRY, _g(c0, "interop", {}) or {})
```

In the tool loop (inside `stream()`), right after `result = tools_mod.execute(REGISTRY, call)` and before building `res_text`, add:

```python
            RECEIPTS.append("tool", {"name": call.name, "args": call.args,
                                     "ok": result.get("ok", False)})
```

- [ ] **Step 2: train.py — ledger + health at growth cadence**

At module scope (near `selfdir_log = []`), add:

```python
from minagi import ledger as ledger_mod
from minagi import health as health_mod
```

Inside `cmd_read`, after the loop-setup locals (near `last_held = None`), add:

```python
    receipts = ledger_mod.ReceiptLedger(os.path.join("runs", "receipts.jsonl"))
```

In the growth block (`if grower is not None and grow_every_steps ...`), after the `_selfdir_reflect(...)` call (when `args.selfdir`), add — using the `recent` deque of recent train losses:

```python
                    if args.selfdir and len(recent) >= 4:
                        h = health_mod.certify(list(recent)[-32:])
                        receipts.append("health", h)
                        if not h["cm"]:
                            print(f"    health: {h['verdict']}", flush=True)
```

- [ ] **Step 3: Verify**

Run: `python -m py_compile serve.py train.py minagi/ledger.py minagi/health.py minagi/http_tool.py` and `python -m unittest discover -s tests -v`.
Expected: compile OK; all unit tests pass (4 + 2 + 6 + prior 31).

- [ ] **Step 4: Commit**

```bash
git add serve.py train.py
git commit -m "feat(governance): wire receipts + health certificate into loops"
```

---

### Task 5: `config.yaml` — `receipts:` and `health:` sections

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Append**

```yaml
receipts:
  path: runs/receipts.jsonl   # append-only, tamper-evident record of agent actions

health:
  # Discrete complete-monotonicity certificate over the recent held-out loss
  # trajectory (reuses the verified cos-obstruction from the RH work).
  window: 32                  # how many recent evaluations the certificate covers
```

- [ ] **Step 2: Verify**

Run: `python -c "from minagi.config import load, get; c = load(); print(get(c, 'health.window'), get(c, 'receipts.path'))"`
Expected: `32 runs/receipts.jsonl`.

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "chore(config): receipts + health sections"
```

---

## Self-Review

- **Spec coverage:** receipts → Task 1 + wiring Task 4; HTTP/MCP bridge → Task 2 + Task 4; CM certificate → Task 3 + Task 4; config → Task 5. RH-stipulation is explicitly NOT claimed (Global Constraints); the Lean theorem is cited as reference in `health.py` docstring.
- **Placeholder scan:** none.
- **Type consistency:** `ReceiptLedger(path).append(kind, payload) -> dict` and `.verify() -> (bool, str)` used identically in Tasks 1 and 4; `health.certify(values) -> dict` with `verdict` key used in Tasks 3 and 4; `http_tool.register_http_tools(registry, cfg)` used in Task 4.
