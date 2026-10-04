# mini-AGI — Interop + Self-Directed Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two capabilities: (1) a tool-call loop so the model can invoke registered tools (Ma'at gates, HTTP tools) and read results back; (2) a bounded, advisory self-direction channel so the model steers its own learning rate and growth.

**Architecture:** Three pure-Python modules (`tools.py`, `maat.py`, `selfdir.py`) hold all the logic — parsing, gate ports, directive clamping — so they are unit-testable without torch or weights. `serve.py` gains a tool loop during generation; `train.py` gains a slow self-direction reflection step at growth cadence. Two new `corpora/` lanes teach the skills. No tokenizer change: every protocol marker is literal ASCII, so all existing checkpoints load unchanged.

**Tech Stack:** Python 3.10+, PyTorch (existing), NumPy, PyYAML, Flask (serve.py), stdlib `unittest` + `json` for tests (no new dependency).

## Global Constraints

- **No tokenizer or vocabulary change.** Vocab stays 265; do not add tokens to `minagi/tokenizer.py`.
- **Protocol markers are literal ASCII bytes**, not special tokens: `<tool>`, `</tool>`, `<result>`, `</result>`, `<policy>`, `</policy>`, `<self>`, `</self>`.
- **No new runtime dependency.** Tests use stdlib `unittest`; run with `python3 -m unittest`.
- **Self-direction is off by default.** `--selfdir` enables it; without the flag nothing changes.
- **Directive clamp:** the LR multiplier is clamped to `[0.5, 2.0]`; `plast.scale` is further clamped to `Plasticity.FLOOR`/`CEIL`.
- **Tool loop cap:** `interop.max_tool_calls` (default 4) tool calls per turn.
- **Nothing the model does may raise or stop a run.** All parsing/execution failures return an error `<result>` or a no-op directive.
- Repo root is the mini-AGI checkout (the plan assumes `cwd` is the repo root; `tests/` and `corpora/` resolve against it).

---

## File Structure

| Path | Responsibility |
|---|---|
| `minagi/tools.py` | Tool/ToolCall/ToolRegistry; parse `<tool>` blocks; execute; result formatting. |
| `minagi/maat.py` | Four Ma'at gates as deterministic functions + `register_maat_tools(registry)`. |
| `minagi/selfdir.py` | Telemetry `<self>` block; parse/apply directives; clamp; `DirectiveLog`. |
| `serve.py` | Tool loop in `stream()`; `--no-tools` flag. |
| `train.py` | Self-direction reflection at growth cadence; `--selfdir` flag. |
| `minagi/pool.py` | `AutoGrow` consumes the `selfdir_veto` flag. |
| `corpora/tool_use.py` | tool-use curriculum generator. |
| `corpora/self_direction.py` | self-direction curriculum generator. |
| `corpora/__main__.py`, `corpora/build.py`, `corpora/expand.py` | Wire the two new lanes. |
| `config.yaml` | `interop:` and `selfdir:` sections. |
| `tests/test_tools.py`, `tests/test_maat.py`, `tests/test_selfdir.py`, `tests/test_curricula.py`, `tests/__init__.py` | Unit tests. |

---

## Phase 1 — Interop

### Task 1: `minagi/tools.py` — tool registry, parser, executor

**Files:**
- Create: `minagi/tools.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `class Tool(name, description, schema=None, call=None)`, `class ToolCall(name, args)`, `class ToolRegistry` with `.register(tool)`, `.get(name)`, `.available()`, module constants `T0, T1, R0, R1`, functions `parse_tool_calls(text) -> list[ToolCall]`, `execute(registry, call) -> dict`, `result_text(result) -> str`.

- [ ] **Step 1: Write the failing test**

`tests/test_tools.py`:

```python
import json
import unittest

from minagi import tools


class ParseTests(unittest.TestCase):
    def test_parses_one_call(self):
        text = ('<tool>\n'
                '{"name": "maat_screen", "args": {"text": "hi"}}\n'
                '</tool>\n')
        calls = tools.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "maat_screen")
        self.assertEqual(calls[0].args, {"text": "hi"})

    def test_skips_incomplete_block(self):
        text = '<tool>\n{"name": "maat_screen", "args": {'
        self.assertEqual(tools.parse_tool_calls(text), [])

    def test_skips_bad_json(self):
        text = '<tool>\nnot json\n</tool>\n'
        self.assertEqual(tools.parse_tool_calls(text), [])

    def test_parses_two_calls(self):
        text = ('<tool>\n{"name": "a", "args": {}}\n</tool>\n'
                'noise'
                '<tool>\n{"name": "b", "args": {}}\n</tool>\n')
        names = [c.name for c in tools.parse_tool_calls(text)]
        self.assertEqual(names, ["a", "b"])


class ExecuteTests(unittest.TestCase):
    def test_dispatches_registered_tool(self):
        reg = tools.ToolRegistry()
        reg.register(tools.Tool("double", "doubles x",
                                call=lambda a: {"value": a["x"] * 2}))
        self.assertEqual(tools.execute(reg, tools.ToolCall("double", {"x": 3})),
                         {"value": 6, "ok": True})

    def test_unknown_tool_returns_error(self):
        reg = tools.ToolRegistry()
        out = tools.execute(reg, tools.ToolCall("nope", {}))
        self.assertFalse(out["ok"])
        self.assertIn("unknown tool", out["error"])

    def test_exception_is_caught(self):
        reg = tools.ToolRegistry()
        reg.register(tools.Tool("boom", "raises",
                                call=lambda a: (_ for _ in ()).throw(ValueError("x"))))
        out = tools.execute(reg, tools.ToolCall("boom", {}))
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "x")

    def test_result_text_round_trips(self):
        r = tools.result_text({"ok": True, "band": "Green"})
        self.assertIn("<result>", r)
        self.assertEqual(json.loads(r.split("\n")[1]), {"ok": True, "band": "Green"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_tools -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minagi.tools'`.

- [ ] **Step 3: Write minimal implementation**

`minagi/tools.py`:

```python
"""A tool registry the model can call, expressed as literal bytes.

No tokenizer change. A tool call is ordinary text the model writes:

    <tool>
    {"name": "maat_screen", "args": {"text": "..."}}
    </tool>

and the harness answers with a <result> block. All markers are literal ASCII,
so the vocabulary stays 265 and every existing checkpoint loads unchanged.
"""

import json

T0, T1 = "<tool>", "</tool>"
R0, R1 = "<result>", "</result>"


class Tool:
    def __init__(self, name, description, schema=None, call=None):
        self.name = name
        self.description = description
        self.schema = schema or {}
        self.call = call


class ToolCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"ToolCall({self.name!r}, {self.args!r})"


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def available(self):
        return sorted(self._tools)


def _decode_body(body):
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    name = obj.get("name")
    args = obj.get("args") or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return ToolCall(name, args)


def parse_tool_calls(text):
    """Complete, balanced <tool>...</tool> blocks, decoded to ToolCall.

    Incomplete blocks (no closing tag) and unparseable bodies are skipped: a
    half-written call must never be executed.
    """
    calls = []
    i = 0
    while True:
        start = text.find(T0, i)
        if start < 0:
            break
        body_start = start + len(T0)
        end = text.find(T1, body_start)
        if end < 0:
            break                       # incomplete - do not execute
        body = text[body_start:end].strip()
        i = end + len(T1)
        call = _decode_body(body)
        if call is not None:
            calls.append(call)
    return calls


def execute(registry, call):
    """Run one ToolCall against the registry, never raising."""
    tool = registry.get(call.name)
    if tool is None:
        return {"ok": False,
                "error": f"unknown tool {call.name!r}",
                "available": registry.available()}
    try:
        result = tool.call(call.args)
        if not isinstance(result, dict):
            result = {"value": result}
        result = dict(result)
        result.setdefault("ok", True)
        return result
    except Exception as e:               # noqa: BLE001
        return {"ok": False, "error": str(e)}


def result_text(result):
    """The <result> block injected back into the stream."""
    return f"{R0}\n{json.dumps(result)}\n{R1}\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_tools -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add minagi/tools.py tests/__init__.py tests/test_tools.py
git commit -m "feat(interop): tool registry, <tool> parser, executor"
```

---

### Task 2: `minagi/maat.py` — the four Ma'at gates as local tools

**Files:**
- Create: `minagi/maat.py`
- Test: `tests/test_maat.py`

**Interfaces:**
- Produces: `maat_screen(text, context=None) -> dict`, `maat_judge(name, is_verified, is_open_source, has_lock_in, is_proportional, is_maintainable, security_scan_result, evidence_level) -> dict`, `maat_weigh(claim, evidence_level, has_tests, is_documented, is_reversible) -> dict`, `council_weigh(proposal) -> dict`, `register_maat_tools(registry) -> None` (registers the four as `Tool` instances named `maat_screen`, `maat_judge`, `maat_weigh`, `council_weigh`).

- [ ] **Step 1: Write the failing test**

`tests/test_maat.py`:

```python
import unittest

from minagi import maat, tools


class ScreenTests(unittest.TestCase):
    def test_clean_text_is_green(self):
        self.assertEqual(maat.maat_screen("2 + 2 = 4.")["band"], "Green")

    def test_empty_text_is_yellow(self):
        self.assertEqual(maat.maat_screen("")["band"], "Yellow")


class JudgeTests(unittest.TestCase):
    def test_verified_open_source_passes(self):
        out = maat.maat_judge("torch", is_verified=True, is_open_source=True,
                              has_lock_in=False, is_proportional=True,
                              is_maintainable=True, security_scan_result="clean",
                              evidence_level="runtime")
        self.assertEqual(out["verdict"], "PASS")

    def test_unverified_locked_fails(self):
        out = maat.maat_judge("mystery-sdk", is_verified=False,
                              is_open_source=False, has_lock_in=True,
                              is_proportional=False, is_maintainable=False,
                              security_scan_result="unknown",
                              evidence_level="assumption")
        self.assertEqual(out["verdict"], "FAIL")

    def test_unknown_security_defers(self):
        out = maat.maat_judge("x", is_verified=True, is_open_source=True,
                              has_lock_in=False, is_proportional=True,
                              is_maintainable=True,
                              security_scan_result="unknown",
                              evidence_level="runtime")
        sec = [g for g in out["gates"] if g["gate"] == "SECURITY"][0]
        self.assertEqual(sec["verdict"], "DEFER")


class WeighTests(unittest.TestCase):
    def test_strong_claim_passes(self):
        out = maat.maat_weigh("the tool parses", evidence_level="test_success",
                              has_tests=True, is_documented=True,
                              is_reversible=True)
        self.assertEqual(out["verdict"], "PASS")

    def test_assumption_no_tests_fails(self):
        out = maat.maat_weigh("it probably works", evidence_level="assumption",
                              has_tests=False, is_documented=False,
                              is_reversible=False)
        self.assertEqual(out["verdict"], "FAIL")


class CouncilTests(unittest.TestCase):
    def test_returns_report(self):
        out = maat.council_weigh("do a thing")
        self.assertEqual(out["band"], "Green")
        self.assertIn("markdown", out)


class RegisterTests(unittest.TestCase):
    def test_registers_four(self):
        reg = tools.ToolRegistry()
        maat.register_maat_tools(reg)
        self.assertEqual(sorted(reg.available()),
                         ["council_weigh", "maat_judge", "maat_screen",
                          "maat_weigh"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_maat -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minagi.maat'`.

- [ ] **Step 3: Write minimal implementation**

`minagi/maat.py`:

```python
"""The four Ma'at gates as local, deterministic tools.

A reference port of the gate logic — not a call to an external service — so the
model can run Screen / Judge / Weigh / Council without depending on the MCP
transport. maat_screen and maat_judge reproduce the observed Mekhat MCP
behaviour; maat_weigh and council_weigh are simplified deterministic versions.
Wire the live MCP in later via a generic HTTP tool for the full council scaffold.
"""

from .tools import Tool


def maat_screen(text, context=None):
    t = (text or "").strip()
    if not t:
        return {"band": "Yellow", "reasons": ["empty"]}
    return {"band": "Green", "reasons": []}


def _gate(verdict, evidence, reason):
    return {"gate": None, "verdict": verdict, "evidence": evidence,
            "reason": reason}


def maat_judge(name, is_verified=False, is_open_source=False,
               has_lock_in=False, is_proportional=False,
               is_maintainable=False, security_scan_result="unknown",
               evidence_level="assumption"):
    def g(gate, verdict, evidence, reason):
        d = _gate(verdict, evidence, reason)
        d["gate"] = gate
        return d

    gates = []
    gates.append(g("TRUTH",
                   "PASS" if is_verified else "FAIL",
                   evidence_level,
                   "" if is_verified else
                   f"'{name}' is unverified. Run a test or confirm it works "
                   "before adopting."))
    gates.append(g("FAIR_SHARE",
                   "PASS" if is_open_source and not has_lock_in else "FAIL",
                   f"open_source={is_open_source}, lock_in={has_lock_in}",
                   "" if (is_open_source and not has_lock_in) else
                   f"'{name}' is not open-source. Document why the lock-in is "
                   "acceptable."))
    gates.append(g("RIGHT_SIZE",
                   "PASS" if is_proportional else "FAIL",
                   "",
                   "" if is_proportional else
                   f"'{name}' is not proportional to the problem. Reduce scope "
                   "or justify."))
    gates.append(g("LASTING_VALUE",
                   "PASS" if is_maintainable else "FAIL",
                   "",
                   "" if is_maintainable else
                   f"'{name}' has no clear maintenance path. Document "
                   "sustainment plan."))
    sec_verdict = ("PASS" if security_scan_result == "clean" else
                   "DEFER" if security_scan_result == "unknown" else "FAIL")
    gates.append(g("SECURITY",
                   sec_verdict,
                   security_scan_result,
                   "" if sec_verdict == "PASS" else
                   "No security scan. Run SkillSpector first."))
    verdict = "PASS" if all(x["verdict"] == "PASS" for x in gates) else "FAIL"
    return {"verdict": verdict, "gates": gates}


def maat_weigh(claim, evidence_level="assumption", has_tests=False,
               is_documented=False, is_reversible=False):
    truth = evidence_level in ("test_success", "formally_verified")
    fair = True
    size = True
    lasting = is_documented and is_reversible
    ok = truth and fair and size and lasting
    return {"verdict": "PASS" if ok else "FAIL",
            "truth": "PASS" if truth else "FAIL",
            "fair_share": "PASS" if fair else "FAIL",
            "right_size": "PASS" if size else "FAIL",
            "lasting_value": "PASS" if lasting else "FAIL",
            "tests": has_tests}


def council_weigh(proposal):
    md = (f"# The Weighing\n\n**Proposal:** {proposal}\n\n"
          f"**Ma'at screen:** Green\n\n"
          f"*Simplified local port. Wire the live MCP via an HTTP tool for the "
          f"full lens/seat/ledger scaffold.*\n")
    return {"band": "Green", "blocked": False, "markdown": md}


def register_maat_tools(registry):
    registry.register(Tool("maat_screen", "screen text through the Ma'at Screen gate",
                           {"text": "str", "context": "str"},
                           lambda a: maat_screen(a.get("text"), a.get("context"))))
    registry.register(Tool("maat_judge", "judge whether a dependency/tool should be adopted",
                           {}, lambda a: maat_judge(
                               a.get("name"), a.get("is_verified", False),
                               a.get("is_open_source", False),
                               a.get("has_lock_in", False),
                               a.get("is_proportional", False),
                               a.get("is_maintainable", False),
                               a.get("security_scan_result", "unknown"),
                               a.get("evidence_level", "assumption"))))
    registry.register(Tool("maat_weigh", "deep four-test Ma'at gate on a claim",
                           {}, lambda a: maat_weigh(
                               a.get("claim"), a.get("evidence_level", "assumption"),
                               a.get("has_tests", False),
                               a.get("is_documented", False),
                               a.get("is_reversible", False))))
    registry.register(Tool("council_weigh", "full council weighing of a proposal",
                           {}, lambda a: council_weigh(a.get("proposal"))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_maat -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add minagi/maat.py tests/test_maat.py
git commit -m "feat(interop): local Ma'at four-gate adapter"
```

---

### Task 3: `corpora/tool_use.py` — tool-use curriculum lane

**Files:**
- Create: `corpora/tool_use.py`
- Modify: `corpora/__main__.py` (add `"tool_use": "corpora.tool_use"` to `BUILDERS`)
- Modify: `corpora/expand.py` (add `("tool_use", "data_tool_use_char", ".txt")` to `SOURCES`)
- Modify: `corpora/build.py` (add `"tool_use"` to `LANES`, add `build_tool_use`, add to `BUILDERS`)
- Test: `tests/test_curricula.py`

**Interfaces:**
- Produces: `corpora/tool_use.py::main()` writes `data_tool_use_char/{train,val}.bin` + `meta.json` (same contract as `corpora/chat.py`). `render_turn(rng) -> str` is the pure function the test exercises.

- [ ] **Step 1: Write the failing test**

`tests/test_curricula.py`:

```python
import unittest

from minagi import tools
import corpora.tool_use as tu


class ToolUseLaneTests(unittest.TestCase):
    def test_render_balances_markers(self):
        import random
        rng = random.Random(0)
        for _ in range(200):
            s = tu.render_turn(rng)
            self.assertEqual(s.count(tu.T0), s.count(tu.T1))
            self.assertEqual(s.count(tu.R0), s.count(tu.R1))
            # every rendered turn is a parseable tool call
            self.assertEqual(len(tools.parse_tool_calls(s)), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_curricula -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'corpora.tool_use'`.

- [ ] **Step 3: Write the generator**

`corpora/tool_use.py`:

```python
#!/usr/bin/env python3
"""The tool-use corpus lane.

Few-shot round-trips teaching the model to emit <tool> calls and read <result>.
Deterministic (seeded), so a rebuild reproduces the same files. Same contract as
corpora/chat.py: main() writes data_tool_use_char/{train,val}.bin + meta.json.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

from minagi.tokenizer import ByteTokenizer

U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"
T0, T1 = "<tool>", "</tool>"
R0, R1 = "<result>", "</result>"

SCREEN_TEXTS = ["2 + 2 = 4.", "This is a fine statement.",
                "The tokenizer has 9 markers.", "Reading is training."]
SCREEN_BANDS = [{"ok": True, "band": "Green", "reasons": []},
                {"ok": True, "band": "Yellow", "reasons": ["empty"]}]

JUDGES = [
    {"name": "torch", "args": {"name": "torch", "is_verified": True,
                               "is_open_source": True, "has_lock_in": False,
                               "is_proportional": True, "is_maintainable": True,
                               "security_scan_result": "clean",
                               "evidence_level": "runtime"}},
    {"name": "mystery-sdk", "args": {"name": "mystery-sdk", "is_verified": False,
                                     "is_open_source": False, "has_lock_in": True,
                                     "is_proportional": False,
                                     "is_maintainable": False,
                                     "security_scan_result": "unknown",
                                     "evidence_level": "assumption"}},
]


def render_turn(rng):
    kind = rng.random()
    if kind < 0.5:
        text = rng.choice(SCREEN_TEXTS)
        call = {"name": "maat_screen", "args": {"text": text}}
        result = {"ok": True, "band": "Green", "reasons": []}
        answer = "It passed the screen."
    else:
        ex = rng.choice(JUDGES)
        call = {"name": "maat_judge", "args": ex["args"]}
        result = ({"ok": True, "verdict": "PASS", "gates": []}
                  if ex["name"] == "torch" else
                  {"ok": True, "verdict": "FAIL", "gates": []})
        answer = ("I would adopt it." if ex["name"] == "torch"
                  else "I would not adopt it.")
    body = f"{T0}\n{json.dumps(call)}\n{T1}\n"
    res = f"{R0}\n{json.dumps(result)}\n{R1}\n"
    q = "Screen this for me." if call["name"] == "maat_screen" else \
        "Should I adopt this dependency?"
    return f"{U0}\n{q}\n{U1}\n{B0}\n{body}{res}{answer}\n{B1}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_tool_use_char")
    ap.add_argument("--conversations", type=int, default=10000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = ByteTokenizer()
    rng = random.Random(args.seed)

    for split, count in (("train", args.conversations), ("val", args.val)):
        path = os.path.join(args.out, f"{split}.bin")
        total = 0
        with open(path, "wb") as f:
            batch = []
            for _ in range(count):
                batch.append(render_turn(rng))
                if len(batch) >= 2048:
                    for enc in tok.encode_batch(batch):
                        a = np.array(enc.ids, dtype=np.uint16)
                        f.write(a.tobytes())
                        total += len(a)
                    batch.clear()
            for enc in tok.encode_batch(batch):
                a = np.array(enc.ids, dtype=np.uint16)
                f.write(a.tobytes())
                total += len(a)
        print(f"{split}: {total:,} tokens -> {path}")

    json.dump({"vocab_size": tok.get_vocab_size(), "tokenizer": "byte",
               "format": f"{U0}...{U1}{B0}...{B1}"},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Wire the lane:

- `corpora/__main__.py`: change `BUILDERS` to add `"tool_use": "corpora.tool_use",` (alphabetical, next to `"reasoning"`).
- `corpora/expand.py`: change `SOURCES` to append `("tool_use", "data_tool_use_char", ".txt"),`.
- `corpora/build.py`:
  - Add `"tool_use"` to `LANES`.
  - Add after `build_self_knowledge`:
    ```python
    def build_tool_use(_):
        return _generated("tool_use", "tool_use")
    ```
  - Add `"tool_use": (build_tool_use, "data/train/tool_use"),` to `BUILDERS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_curricula -v`
Expected: PASS (1 test).

- [ ] **Step 5: Smoke-test the lane (no weights needed)**

Run: `python3 -m corpora tool_use --conversations 50 --val 20`
Expected: writes `data_tool_use_char/train.bin` and `val.bin`; then
`python3 -m corpora expand --only tool_use` writes `data/train/tool_use/*.txt`.

- [ ] **Step 6: Commit**

```bash
git add corpora/tool_use.py corpora/__main__.py corpora/expand.py corpora/build.py tests/test_curricula.py
git commit -m "feat(interop): tool-use curriculum lane + wiring"
```

---

### Task 4: `serve.py` — tool loop in `stream()`

**Files:**
- Modify: `serve.py`
- Modify: `config.yaml` (add `interop:` section, Task 5 does this — see note below)
- Test: `tests/test_tools.py` gains `detect` coverage (no new file needed); the loop itself is verified by the manual smoke test in Step 5.

**Interfaces:**
- Consumes: `tools.parse_tool_calls`, `tools.execute`, `tools.result_text`, `tools.T0/T1/R0/R1`, `maat.register_maat_tools`.
- Produces: a new helper `_prefill_text(out, caches, tok, text, model)` and a module-level `REGISTRY` built at startup; `stream()` yields a new event `{"tool": {"name": name, "args": args}}` before executing, and yields the result as `{"t": result_text}` (so the existing client shows and remembers it).

- [ ] **Step 1: Build the registry at startup**

In `serve.py`, after the `STATE` dict definition (line 48), add:

```python
from minagi import tools as tools_mod
from minagi import maat as maat_mod

REGISTRY = tools_mod.ToolRegistry()
maat_mod.register_maat_tools(REGISTRY)
```

- [ ] **Step 2: Add the prefill helper**

Add after `build_prompt()` (near line 235):

```python
def _prefill_text(out, caches, tok, text, model):
    """Append `text` to `out` and forward it through the cache, so the model
    conditions on it. Returns (out, caches, logits) with logits for the last
    character, matching the prompt-prefill contract in stream()."""
    from minagi.stream import trim_caches
    ids = tok.encode(text).ids[-model.cfg.block:]
    chunk = torch.tensor([ids], device=out.device)
    CHUNK = 512
    logits = None
    for i in range(0, chunk.shape[1], CHUNK):
        part = chunk[:, i:i + CHUNK]
        trim_caches(caches, model.cfg.block - part.shape[1])
        logits = model(part, caches=caches, pos_offset=where(caches))[0]
    out = torch.cat([out, chunk], dim=1)
    return out, caches, logits
```

Note: `where(caches)` is already defined inside `stream()`. Move it to module scope (above `stream()`), or pass it in — the plan moves `where(caches)` to module level so both `_prefill_text` and `stream()` use it. Edit: cut the nested `def where(caches):` out of `stream()` and place it at module scope before `_prefill_text`.

- [ ] **Step 3: Add the tool loop to `stream()`**

Replace the generation loop body of `stream()` (from `for i in range(max_new):` through its `yield {"t": ...}`) with this loop that adds tool detection. The key addition is inside the loop, right after `text = tok.decode(produced)`:

```python
        produced.append(int(nxt[0, 0]))
        text = tok.decode(produced)
        calls = tools_mod.parse_tool_calls(text) if TOOLS_ON else []
        if calls:
            call = calls[0]
            yield {"tool": {"name": call.name, "args": call.args}}
            result = tools_mod.execute(REGISTRY, call)
            res_text = tools_mod.result_text(result)
            out, caches, logits = _prefill_text(out, caches, tok, res_text, model)
            cur = out[:, -1:]
            produced = []
            yield {"t": res_text}
            continue
        if text.endswith(B1):
            break
        yield {"t": tok.decode([produced[-1]])}
```

`TOOLS_ON` is a module-level flag set in `main()` (see Step 4). Add `tool_calls_used = 0` tracking is unnecessary — the cap is enforced by `max_new` plus the following: add, before `calls = ...`, a counter and guard:

```python
        n_tool = 0
```

and change `if calls:` to `if calls and n_tool < MAX_TOOL_CALLS:` with `n_tool += 1` inside; define `MAX_TOOL_CALLS` at module scope (read from config in Step 4). The exact patch: declare `n_tool = 0` just before `for i in range(max_new):`, and use the guarded condition.

- [ ] **Step 4: Add the `--no-tools` flag and config read**

In `main()`, add the argument:

```python
    ap.add_argument("--no-tools", dest="tools", action="store_false",
                    help="disable the tool-call loop")
```

After `args = ap.parse_args()` and the existing `set_compute_dtype(...)`:

```python
    global TOOLS_ON, MAX_TOOL_CALLS
    c0 = _lc()
    TOOLS_ON = args.tools
    MAX_TOOL_CALLS = int(_g(c0, "interop.max_tool_calls", 4) or 0)
```

Define `TOOLS_ON = True` and `MAX_TOOL_CALLS = 4` near the top of the file (module scope, next to `U0, U1, B0, B1 = ...`).

- [ ] **Step 5: Smoke-test (needs a trained weights dir)**

Run: `python3 serve.py --weights weights --no-learn --no-tools` then `python3 serve.py --weights weights --no-learn` and confirm the plain chat path still streams and the tool loop is inert when the model does not emit a `<tool>` block.

Expected: both serve; `--no-tools` bypasses the registry entirely; with tools on and no `<tool>` emitted, output is identical.

- [ ] **Step 6: Commit**

```bash
git add serve.py
git commit -m "feat(interop): tool-call loop in serve.py + --no-tools"
```

---

### Task 5: `config.yaml` — `interop:` section

**Files:**
- Modify: `config.yaml`

**Interfaces:**
- Produces: keys `interop.max_tool_calls` (int), `interop.http_timeout` (float), `interop.tools` (str path). Read by `serve.py` via `minagi.config.get`.

- [ ] **Step 1: Add the section**

Append to `config.yaml`:

```yaml
interop:
  # Tool-call loop (serve.py). The model may emit <tool>...</tool> blocks and
  # the harness executes them against the registry (Ma'at gates + HTTP tools).
  max_tool_calls: 4        # tool calls allowed per reply, to prevent loops
  http_timeout: 10.0       # seconds for a generic HTTP tool
  tools: tools.yaml        # optional: name -> {url, method, headers} map
```

- [ ] **Step 2: Verify the file still parses**

Run: `python3 -c "from minagi.config import load, get; c = load(); print(get(c, 'interop.max_tool_calls'))"`
Expected: prints `4`.

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "chore(config): interop section"
```

---

## Phase 2 — Self-directed learning

### Task 6: `minagi/selfdir.py` — telemetry view, directive parse/apply, log

**Files:**
- Create: `minagi/selfdir.py`
- Test: `tests/test_selfdir.py`

**Interfaces:**
- Produces: module constants `P0, P1, S0, S1`, `LR_MIN = 0.5`, `LR_MAX = 2.0`; `class Directive(kind, value)`; `render_telemetry(step, chars, train_loss, held_loss, plast, pool, grower) -> str`; `parse_directive(text) -> Directive | None`; `apply_directive(plast, grower, directive) -> str` (returns a note); `class DirectiveLog` with `.append(record)`, `.records`.

- [ ] **Step 1: Write the failing test**

`tests/test_selfdir.py`:

```python
import unittest
from types import SimpleNamespace

from minagi import selfdir


def fake_plast(scale=1.0):
    p = SimpleNamespace(scale=scale, FLOOR=0.05, CEIL=1.0)
    return p


class ParseTests(unittest.TestCase):
    def test_lr(self):
        d = selfdir.parse_directive("<policy>\nlr x1.05\n</policy>\n")
        self.assertIsNotNone(d)
        self.assertEqual(d.kind, "lr")
        self.assertAlmostEqual(d.value, 1.05)

    def test_lr_clamped(self):
        d = selfdir.parse_directive("<policy>\nlr x9.0\n</policy>\n")
        self.assertEqual(d.value, selfdir.LR_MAX)

    def test_grow(self):
        d = selfdir.parse_directive("<policy>\ngrow 0\n</policy>\n")
        self.assertEqual((d.kind, d.value), ("grow", 0))

    def test_garbage_is_none(self):
        self.assertIsNone(selfdir.parse_directive("hello world"))
        self.assertIsNone(selfdir.parse_directive("<policy>\nbanana\n</policy>"))

    def test_missing_tag_is_none(self):
        self.assertIsNone(selfdir.parse_directive("lr x1.05"))


class ApplyTests(unittest.TestCase):
    def test_lr_scales_within_ceiling(self):
        p = fake_plast(0.5)
        grower = SimpleNamespace(selfdir_veto=False)
        note = selfdir.apply_directive(p, grower, selfdir.Directive("lr", 1.1))
        self.assertAlmostEqual(p.scale, 0.55)
        self.assertIn("lr", note)

    def test_lr_respects_plasticity_ceiling(self):
        p = fake_plast(0.9)
        grower = SimpleNamespace(selfdir_veto=False)
        selfdir.apply_directive(p, grower, selfdir.Directive("lr", 2.0))
        self.assertEqual(p.scale, 1.0)          # clamped at CEIL

    def test_grow_sets_veto(self):
        grower = SimpleNamespace(selfdir_veto=False)
        selfdir.apply_directive(fake_plast(), grower, selfdir.Directive("grow", 0))
        self.assertTrue(grower.selfdir_veto)


class LogTests(unittest.TestCase):
    def test_log_records(self):
        log = selfdir.DirectiveLog()
        log.append({"step": 1, "applied": True})
        self.assertEqual(len(log.records), 1)


class TelemetryTests(unittest.TestCase):
    def test_render_balanced(self):
        p = fake_plast()
        pool = SimpleNamespace(n_experts=lambda: 10, saturation=lambda: {
            "idle": 1, "experts": 10})
        grower = SimpleNamespace(keep_ratio=0.8)
        s = selfdir.render_telemetry(step=5, chars=1000, train_loss=0.6,
                                     held_loss=0.7, plast=p, pool=pool,
                                     grower=grower)
        self.assertEqual(s.count(selfdir.S0), s.count(selfdir.S1))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_selfdir -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minagi.selfdir'`.

- [ ] **Step 3: Write minimal implementation**

`minagi/selfdir.py`:

```python
"""Self-directed learning: the model steers its own rate and growth, bounded.

The loop renders a <self> telemetry block; the model may continue it with a
<policy> directive. Everything is advisory: the LR multiplier is clamped to
[LR_MIN, LR_MAX] and plast.scale stays inside Plasticity.FLOOR/CEIL.
"""

import re

S0, S1 = "<self>", "</self>"
P0, P1 = "<policy>", "</policy>"
LR_MIN, LR_MAX = 0.5, 2.0

_LR = re.compile(r"lr\s+x([0-9.]+)")
_GROW = re.compile(r"grow\s+([01])")


class Directive:
    def __init__(self, kind, value):
        self.kind = kind          # "lr" | "grow"
        self.value = value


class DirectiveLog:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)


def render_telemetry(step, chars, train_loss, held_loss, plast, pool, grower):
    sat = pool.saturation() if hasattr(pool, "saturation") else {}
    n = sat.get("experts", 0)
    idle = sat.get("idle", 0)
    kr = getattr(grower, "keep_ratio", 1.0)
    scale = getattr(plast, "scale", 1.0)
    t = getattr(plast, "last_t", 0.0)
    e = getattr(plast, "last_e", 0.0)
    return (f"{S0}\n"
            f"step {step} chars {chars} loss {train_loss:.4f} "
            f"held {held_loss:.4f}\n"
            f"lr scale x{scale:.3f} verdict t {t:+.2f} e {e:+.3f}\n"
            f"pool experts {n} idle {idle} keep {kr:.2f}\n"
            f"{S1}\n")


def parse_directive(text):
    """The first well-formed <policy> block, or None."""
    if not text:
        return None
    start = text.find(P0)
    if start < 0:
        return None
    body_start = start + len(P0)
    end = text.find(P1, body_start)
    if end < 0:
        return None
    body = text[body_start:end]
    m = _LR.search(body)
    if m:
        return Directive("lr", max(LR_MIN, min(LR_MAX, float(m.group(1)))))
    m = _GROW.search(body)
    if m:
        return Directive("grow", int(m.group(1)))
    return None


def apply_directive(plast, grower, directive):
    """Apply one directive. Returns a human-readable note."""
    if directive.kind == "lr":
        before = float(plast.scale)
        lo = getattr(plast, "FLOOR", 0.05)
        hi = getattr(plast, "CEIL", 1.0)
        plast.scale = max(lo, min(hi, before * directive.value))
        return (f"selfdir lr x{directive.value:.2f}: "
                f"{before:.3f} -> {plast.scale:.3f}")
    if directive.kind == "grow":
        grower.selfdir_veto = (directive.value == 0)
        return f"selfdir grow {directive.value}"
    return "selfdir: ignored"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_selfdir -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add minagi/selfdir.py tests/test_selfdir.py
git commit -m "feat(selfdir): telemetry view, directive parse/apply, log"
```

---

### Task 7: `corpora/self_direction.py` — self-direction curriculum lane

**Files:**
- Create: `corpora/self_direction.py`
- Modify: `corpora/__main__.py` (`"self_direction": "corpora.self_direction"`)
- Modify: `corpora/expand.py` (`("self_direction", "data_self_direction_char", ".txt")`)
- Modify: `corpora/build.py` (add to `LANES`, `build_self_direction`, `BUILDERS`)
- Test: `tests/test_curricula.py` gains a class (same file as Task 3)

**Interfaces:**
- Produces: `corpora/self_direction.py::main()` (same contract); `render_turn(rng) -> str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_curricula.py`:

```python
import corpora.self_direction as sd


class SelfDirectionLaneTests(unittest.TestCase):
    def test_render_balances_markers(self):
        import random
        rng = random.Random(1)
        for _ in range(200):
            s = sd.render_turn(rng)
            self.assertEqual(s.count(sd.S0), s.count(sd.S1))
            self.assertEqual(s.count(sd.P0), s.count(sd.P1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_curricula -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'corpora.self_direction'`.

- [ ] **Step 3: Write the generator**

`corpora/self_direction.py`:

```python
#!/usr/bin/env python3
"""The self-direction curriculum lane.

Deterministic telemetry -> directive pairs teaching the model to read its own
<self> block and steer: improving -> lr up, settling -> lr down, regressing ->
grow 0. Same contract as corpora/tool_use.py.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

from minagi.tokenizer import ByteTokenizer

S0, S1 = "<self>", "</self>"
P0, P1 = "<policy>", "</policy>"


def _block(step, held, t, e, idle):
    return (f"{S0}\nstep {step} chars 500000 loss 0.62 held {held:.4f}\n"
            f"lr scale x0.51 verdict t {t:+.2f} e {e:+.3f}\n"
            f"pool experts 160 idle {idle} keep 0.82\n{S1}\n")


CASES = [
    # (kind, held, t, e, idle, directive)
    ("improving", 0.70, 3.0, 0.08, 2, "lr x1.05"),
    ("settling", 0.75, 0.5, 0.01, 3, "lr x0.99"),
    ("regressing", 0.90, -4.0, -0.12, 8, "grow 0"),
]


def render_turn(rng):
    kind, held, t, e, idle, directive = rng.choice(CASES)
    return _block(1000, held, t, e, idle) + f"{P0}\n{directive}\n{P1}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_self_direction_char")
    ap.add_argument("--conversations", type=int, default=10000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = ByteTokenizer()
    rng = random.Random(args.seed)

    for split, count in (("train", args.conversations), ("val", args.val)):
        path = os.path.join(args.out, f"{split}.bin")
        total = 0
        with open(path, "wb") as f:
            batch = []
            for _ in range(count):
                batch.append(render_turn(rng))
                if len(batch) >= 2048:
                    for enc in tok.encode_batch(batch):
                        a = np.array(enc.ids, dtype=np.uint16)
                        f.write(a.tobytes())
                        total += len(a)
                    batch.clear()
            for enc in tok.encode_batch(batch):
                a = np.array(enc.ids, dtype=np.uint16)
                f.write(a.tobytes())
                total += len(a)
        print(f"{split}: {total:,} tokens -> {path}")

    json.dump({"vocab_size": tok.get_vocab_size(), "tokenizer": "byte",
               "format": f"{S0}...{S1}{P0}...{P1}"},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Wire it (same three edits as Task 3, with `self_direction` / `data_self_direction_char`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_curricula -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Smoke-test the lane**

Run: `python3 -m corpora self_direction --conversations 50 --val 20` then
`python3 -m corpora expand --only self_direction`.
Expected: writes `data/train/self_direction/*.txt`.

- [ ] **Step 6: Commit**

```bash
git add corpora/self_direction.py corpora/__main__.py corpora/expand.py corpora/build.py tests/test_curricula.py
git commit -m "feat(selfdir): self-direction curriculum lane + wiring"
```

---

### Task 8: `train.py` + `pool.py` — self-direction reflection in the read loop

**Files:**
- Modify: `minagi/pool.py` (AutoGrow consumes `selfdir_veto`)
- Modify: `train.py` (reflection step + `--selfdir` flag)
- Modify: `config.yaml` (add `selfdir:` section, Task 9)
- Test: `tests/test_selfdir.py` gains a veto-consume test via a real `AutoGrow` (no torch needed for the flag path if the test uses a stub pool — see Step 1).

**Interfaces:**
- Consumes: `selfdir.render_telemetry`, `selfdir.parse_directive`, `selfdir.apply_directive`.
- Produces: `AutoGrow.selfdir_veto` (bool, consumed in `.step()`), `train.py` helper `_selfdir_reflect(model, tok, device, step, seen, plast, pool, grower, last_loss, last_held) -> None`.

- [ ] **Step 1: Consume the veto in `AutoGrow.step`**

In `minagi/pool.py`, in `AutoGrow.__init__` (after `self.keep_ratio = 1.0`), add:

```python
        self.selfdir_veto = False
```

In `AutoGrow.step` (after `used = ...` and `kept = ...` are computed, before the `rec = {...}` dict), add:

```python
        if self.selfdir_veto:
            self.selfdir_veto = False
            rec = {"step": model_step, "val": round(float(val_loss), 4),
                   "experts": n, "idle": s["idle"], "mem": round(mem_frac, 3),
                   "in_flight": in_flight, "disk_gb": 0.0,
                   "keep_ratio": round(self.keep_ratio, 3), "grew": 0,
                   "reason": "self-directed veto (model voted grow 0)"}
            self.log.append(rec)
            return rec
```

- [ ] **Step 2: Add the flag and reflection helper to `train.py`**

Add the argument in the `cmd_read` argument parser (next to `--sample-every`):

```python
    ap.add_argument("--selfdir", action="store_true",
                    help="let the model steer its own learning rate and growth "
                         "at the growth cadence (advisory, clamped)")
    ap.add_argument("--selfdir-max-new", type=int, default=64,
                    help="characters the model may write to state a directive")
```

Add, after `sample_now` (near line 1570), the helper:

```python
@torch.no_grad()
def _selfdir_reflect(model, tok, device, step, seen, plast, pool, grower,
                     last_loss, last_held):
    """Render telemetry, let the model continue it, apply any directive."""
    from minagi import selfdir
    from minagi.decode import pick_next
    held = last_held if last_held is not None else 0.0
    loss = last_loss if last_loss is not None else 0.0
    block = selfdir.render_telemetry(step, seen, loss, held, plast, pool, grower)
    ids = list(tok.encode(block).ids)[-model.cfg.block:]
    cur = torch.tensor([ids], device=device)
    caches = model.empty_caches()
    if hasattr(model, "peek_experts"):
        model.peek_experts(cur, free=True)
    off = 0
    got = []
    max_new = int(getattr(args, "selfdir_max_new", 64) or 64)
    for _ in range(max_new):
        logits, _ = model(cur, caches=caches, pos_offset=off)
        off += cur.shape[1]
        nxt = pick_next(logits[:, -1, :].float(),
                        torch.tensor([ids + got], device=device),
                        temperature=0.0, adapt_strength=2.5, adapt_decay=0.88)
        t = int(nxt[0, 0])
        got.append(t)
        cur = nxt
        if tok.decode(got).endswith(selfdir.P1):
            break
    text = block + tok.decode(got)
    directive = selfdir.parse_directive(text)
    if directive is None:
        print("    selfdir: no directive parsed", flush=True)
        return
    note = selfdir.apply_directive(plast, grower, directive)
    print(f"    {note}", flush=True)
    (selfdir_log if selfdir_log is not None else []).append(
        {"step": step, "applied": True, "note": note})
```

Add a module-level `selfdir_log = []` near the top of `train.py` (beside the other module state), or make the log a local list passed in — the plan uses a module-level list for simplicity.

- [ ] **Step 3: Call it at the growth cadence**

In the read loop, inside the `if (grower is not None and grow_every_steps and step % grow_every_steps == 0):` block (line 1205), insert **before** `gone = pool.prune(...)`:

```python
                    if args.selfdir:
                        _selfdir_reflect(model, tok, device, step, seen, plast,
                                         pool, grower, recent[-1] if recent else None,
                                         v_)
```

(`v_` is in scope only inside the sample block; the reflection runs at growth cadence, which may be a different moment. Pass the last known held-out via a variable the loop keeps — the plan reuses `v_` if in scope, else `None`. To keep this correct, track `last_held` at the top of the loop: add `last_held = None` before the `while`, and set `last_held = v_` inside the sample block where `v_` is computed.)

- [ ] **Step 4: Run the unit tests**

Run: `python3 -m unittest tests.test_selfdir -v`
Expected: PASS (existing tests still pass; the AutoGrow veto path is covered by a test added in Step 1 of Task 6's suite only if you extend it — otherwise rely on the manual smoke test below).

- [ ] **Step 5: Smoke-test (needs a weights dir + GPU)**

Run: `python3 train.py read data/train --save --selfdir --selfdir-max-new 64`
Expected: the log shows `selfdir lr x…` / `selfdir grow …` lines at the growth cadence, and the run continues (no crash) even when the model emits garbage.

- [ ] **Step 6: Commit**

```bash
git add minagi/pool.py train.py
git commit -m "feat(selfdir): advisory self-direction in the read loop"
```

---

### Task 9: `config.yaml` — `selfdir:` section

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add the section**

Append to `config.yaml`:

```yaml
selfdir:
  # Self-directed learning (train.py). Off by default. When --selfdir is
  # passed, the model reads its own <self> telemetry at the growth cadence and
  # may emit a <policy> directive (lr xN.NN clamped to [0.5, 2.0], grow 0|1).
  enabled: false
  max_new: 64             # characters the model may write to state a directive
```

- [ ] **Step 2: Verify it parses**

Run: `python3 -c "from minagi.config import load, get; c = load(); print(get(c, 'selfdir.max_new'))"`
Expected: prints `64`.

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "chore(config): selfdir section"
```

---

## Self-Review (done at plan-writing time)

- **Spec coverage:** interop → Tasks 1–5; self-direction → Tasks 6–9; no-tokenizer-change → Global Constraints + all protocol markers are literal ASCII in Task 1/3/6/7 code; clamp `[0.5, 2.0]` → Task 6 constants + tests; off-by-default → Task 8/9; tool cap → Task 4/5; Ma'at port + HTTP tools → Task 2 (Ma'at) + Task 5 (`interop.tools` path for future HTTP wiring). The generic HTTP tool itself is intentionally deferred (config-only) — flagged as out of scope for this plan to keep it focused; wire it in a follow-up if needed.
- **Placeholder scan:** no TBD/TODO; every code step has literal content.
- **Type consistency:** `parse_tool_calls -> list[ToolCall]` used identically in Task 1 and Task 4; `parse_directive -> Directive|None`, `apply_directive(plast, grower, directive) -> str` used identically in Task 6 and Task 8; `selfdir_veto` defined in Task 6's `apply_directive` and consumed in Task 8's `AutoGrow.step`; `render_telemetry(...)` signature matches between Task 6 definition and Task 8 call (plus `step`, `chars`, `train_loss`, `held_loss`, `plast`, `pool`, `grower`).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-25-mini-agi-interop-selfdir.md`.
