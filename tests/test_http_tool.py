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

    def test_register_skips_non_dict_tools(self):
        reg = tools.ToolRegistry()
        http_tool.register_http_tools(reg, {"tools": "tools.yaml"})
        self.assertEqual(reg.available(), [])


if __name__ == "__main__":
    unittest.main()
