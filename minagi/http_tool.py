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
        hdrs = {"Content-Type": "application/json"}
        hdrs.update(headers)
        try:
            data = json.dumps(args).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
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
    if not isinstance(mapping, dict):
        return
    for name, spec in mapping.items():
        spec = dict(spec)
        spec.setdefault("timeout", timeout)
        registry.register(build_http_tool(name, spec))
