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

    def test_lr_malformed_numeric_is_none(self):
        self.assertIsNone(selfdir.parse_directive("<policy>\nlr x1.2.3\n</policy>"))
        self.assertIsNone(selfdir.parse_directive("<policy>\nlr x.\n</policy>"))
        self.assertIsNone(selfdir.parse_directive("<policy>\nlr x..\n</policy>"))


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
