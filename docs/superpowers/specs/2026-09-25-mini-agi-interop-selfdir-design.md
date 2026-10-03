# mini-AGI — Interop + Self-Directed Learning

Date: 2026-09-25
Status: Approved for planning

## Goal

Two capabilities, both in service of "patterns that survive and match its own
systems + others":

1. **Interop** — the model can *act*, not just predict: it can invoke registered
   tools (the Ma'at gates, generic HTTP/JSON tools, other agents) and read their
   results back into the stream.
2. **Self-directed learning** — the model reads its own training telemetry and
   emits *bounded, advisory* steering directives that the training loop honors,
   learning to steer its own learning rate and growth.

Both are built so the model can *learn* the skills (curriculum lanes) rather than
hard-coding them, and so nothing the model does can destabilize the run.

## Non-goals

- No change to the tokenizer or vocabulary. Vocab stays 265 (256 bytes + 9
  markers); every existing checkpoint loads unchanged.
- No change to the plasticity controller's safety rails (floor/ceiling).
- No training-loop restructure. The self-direction channel is additive and off
  by default.

## Cross-cutting decision: no tokenizer change

The tokenizer (`minagi/tokenizer.py`) has exactly 9 special markers
(`<think>`, `</think>`, `<user>`, `</user>`, `<bot>`, `</bot>`, `<g>`, `</g>`,
`<|endoftext|>`), vocab 265. Adding markers changes the embedding shape and
breaks every saved checkpoint.

Therefore all protocol markers in this design are **literal ASCII bytes**, not
special tokens. The model emits `<tool>`, `<policy>`, `<result>`, etc. as ordinary
byte sequences; the harness detects them on *decoded text*. None of these strings
overlap with the 9 existing markers, so the tokenizer treats them as plain bytes.

---

## Part 1 — Interop: tool-call loop

### 1.1 Tool spec

A tool is described by:

- `name` (str) — the identifier the model emits.
- `description` (str) — one line of intent, used in the curriculum and any prompt.
- `schema` (dict) — a minimal JSON-schema-ish shape for the args, used only for
  validation and curriculum generation (not enforced at runtime beyond JSON parse).
- `call(args: dict) -> dict` — the executor. Returns a JSON-serializable dict.

### 1.2 Protocol (on decoded text)

The model emits:

```
<tool>
{"name": "maat_screen", "args": {"text": "..."}}
</tool>
```

The harness responds by injecting:

```
<result>
{"ok": true, "band": "Green", "reasons": []}
</result>
```

- `name` is required and must match a registered tool.
- `args` must be a JSON object; parse failure → an error `<result>` is injected
  and the turn continues (the model sees the failure and can retry or abandon).
- Unknown tool → error `<result>` listing available tools.

### 1.3 `minagi/tools.py`

- `Tool` (dataclass).
- `ToolRegistry` — `register(tool)`, `get(name)`, `available()` (name + one-line
  description, for injection into error results).
- `parse_tool_calls(text) -> list[ToolCall]` — finds balanced `<tool>`/`</tool>`
  blocks, JSON-decodes the body, returns `(name, args)` pairs. Truncated block
  (no closing tag) returns nothing so a partially-emitted call is never executed.
- `execute(registry, call) -> dict` — dispatch, wrapping exceptions into a
  `{"ok": false, "error": ...}` result.

### 1.4 `minagi/maat.py` — local Ma'at adapter

Four functions mirroring the FRA four gates, returning band/verdict JSON:

- `maat_screen(text, context=None) -> {"band", "reasons"}`.
- `maat_judge(name, is_verified, is_open_source, has_lock_in, is_proportional,
  is_maintainable, security_scan_result, evidence_level) -> {"pass", "reasons"}`.
- `maat_weigh(claim, evidence_level, has_tests, is_documented, is_reversible) -> dict`.
- `council_weigh(proposal) -> dict`.

These are deterministic local implementations (a faithful port of the gate logic,
not a call to any external service). They are registered as tools. A generic
**HTTP/JSON tool** is also provided, configured from a small config file
(`tools.yaml`, keyed by name → `{url, method, headers}`), so the live Mekhat MCP
(or any other system) can be wired in later by URL without code changes.

### 1.5 `serve.py` integration

- Inside the existing `LOCK` in `stream()`: after each generated character is
  appended, scan the produced text for a complete `<tool>…</tool>` block.
- On a complete block: parse → execute → inject `<result>…</result>` into the
  output and continue generating (append to `out`, resume greedy decode).
- Cap `max_tool_calls` (default 4) per turn to prevent loops.
- The full exchange (including tool calls and results) goes through `remember()`,
  so the model learns from what a tool returned — consistent with "reads
  everything it does".
- The tool loop is **on by default but inert when no tools are registered**:
  a `<tool>` block naming an unregistered tool yields an error `<result>`. With
  zero registered tools nothing is ever executed. `--no-tools` disables the loop
  entirely, serving the plain chat path unchanged.

### 1.6 Curriculum: `tool-use` corpus lane

A new subject in `corpora/` that generates deterministic few-shot round-trips:

```
<user>\nScreen this text for me: "..."\n</user>\n<bot>\n<tool>\n{"name":"maat_screen","args":{"text":"..."}}\n</tool>\n<result>\n{"ok":true,"band":"Green","reasons":[]}\n</result>\n<bot continues with a plain answer>\n
```

Templates cover: screen, judge (adopt a dependency), weigh (before landing), and
a generic HTTP lookup. Generated deterministically (seeded), same pattern as the
existing `self-knowledge` lane. Lanes already on disk are left alone, matching
`corpora`'s existing behaviour.

---

## Part 2 — Self-directed learning

### 2.1 Protocol

At a slow cadence (growth/checkpoint), the loop renders a telemetry block and the
model may answer with a directive:

```
<self>
step 253398 chars 518.2M loss 0.6231 held 0.7511
lr 1.52e-04 scale x0.509 verdict t +2.73 e +0.0606
pool experts 169 idle 4 keep 0.82 pressure 0.31
</self>
<policy>
lr x1.05
</policy>
```

### 2.2 Directive grammar (tiny, bounded)

- `lr xN.NN` — multiply the plasticity `scale` by `N.NN`, clamped to `[0.5, 2.0]`.
- `grow 0` / `grow 1` — a vote against/for growth. Stored as a one-shot
  `selfdir_veto` flag (default absent = neutral) that `AutoGrow.step` reads: a
  `grow 0` veto blocks the next growth decision, `grow 1` clears any veto. The
  flag is consumed at the next decision, so a directive cannot silently veto
  growth forever.
- Anything else (or nothing) → ignored.

### 2.3 `minagi/selfdir.py`

- `TelemetryView(plast, pool, grower, step, chars, train_loss, held_loss) ->
  str` — renders the `<self>` block. Reuses the fields already present in
  `plast.state()`, `pool.saturation()`, `pool.report()`, `AutoGrow` recs.
- `parse_directive(text) -> Directive | None` — extracts the first well-formed
  `<policy>…</policy>`, parses `lr`/`grow`, applies clamps; returns `None` on
  garbage.
- `apply_directive(plast, grower, directive) -> note` — multiplies
  `plast.scale` by the clamped factor (never past `Plasticity.FLOOR`/`CEIL`) and
  records the growth vote. Returns a human-readable note for the log.
- `DirectiveLog` — appends `{step, directive, applied, scale_before,
  scale_after}` for every parse attempt; auditable and reversible.

### 2.4 `train.py` integration

- At the growth-decision cadence (`growth.every_chars`) and at checkpoint: render
  the telemetry block, then use it as the **prompt** — prime the model with the
  `<self>` block (via `peek_experts`, same as `sample_now`) and let it continue,
  generating a short (greedy, `max_new` ~64) continuation that is expected to
  contain a `<policy>` block. Parse that continuation, apply, log.
- Off by default; `--selfdir` enables it. Without the flag nothing changes.
- Compute is negligible (one short generation per growth window, not per step).

### 2.5 Curriculum: `self-direction` corpus lane

Deterministic telemetry→directive pairs from the run's own history shape:

- improving (positive `t`, positive `e`) → `lr x1.05`
- settling (flat) → `lr x0.99`
- regressing (held-out jumped) → `grow 0`

Generated as `<self>` block followed by `<policy>` block, seeded and reproducible.

---

## Part 3 — Testing

### Unit
- `test_tools.py`: parse valid block, truncated block (no execute), bad JSON,
  unknown tool, registry dispatch, exception wrapping.
- `test_maat.py`: each gate returns the expected band/verdict shape for known
  inputs.
- `test_selfdir.py`: directive parse (lr, grow, garbage, missing tag), clamp
  behaviour at `[0.5, 2.0]`, plasticity floor/ceiling respected, log records.
- `test_curricula.py`: `tool-use` and `self-direction` generators produce
  well-formed marker-balanced text.

### CPU integration
- Prime the model to emit a `maat_screen` call; verify a `<tool>` block is
  parsed, executed, and a `<result>` is injected.
- Feed a telemetry block; verify a valid directive is applied and a garbage one
  ignored; verify `--no-selfdir` disables the channel.

---

## Files

| Path | Change |
|---|---|
| `minagi/tools.py` | new — tool registry, parser, executor |
| `minagi/maat.py` | new — local Ma'at four-gate adapter |
| `minagi/selfdir.py` | new — telemetry view, directive parse/apply, log |
| `serve.py` | edit — tool loop in `stream()`, `--no-tools` |
| `train.py` | edit — self-direction cadence, `--selfdir`/`--no-selfdir` |
| `corpora/` | edit — `tool-use` and `self-direction` lane generators |
| `config.yaml` | edit — `interop:` and `selfdir:` sections (max_tool_calls, cadence) |
| `tests/test_tools.py` etc. | new — unit tests |
| `tests/test_interop_selfdir.py` | new — CPU integration |

## Risks & mitigations

- **Model can't yet emit clean JSON / directives** → the harness treats every
  parse failure as a no-op (never crashes the run); the curriculum lanes teach
  the shape; capability improves as the model does.
- **Self-direction destabilizes training** → all directives clamped to
  `[0.5, 2.0]`, applied on top of the plasticity controller's floor/ceiling,
  logged and reversible; off by default.
- **Tool loop infinite recursion** → `max_tool_calls` cap.
- **Tool execution blocks the single-threaded lock** → tools run inline under
  the lock, but the cap bounds worst-case latency; HTTP tools get a timeout.
