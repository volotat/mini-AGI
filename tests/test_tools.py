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

    def test_skips_non_dict_json(self):
        text = '<tool>\n[1, 2, 3]\n</tool>\n'
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
