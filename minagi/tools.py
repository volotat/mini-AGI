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
    if not isinstance(obj, dict):
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
