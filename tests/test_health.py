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
