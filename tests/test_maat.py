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
