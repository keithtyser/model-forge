from __future__ import annotations

import unittest

from model_forge import gates


class RequiredCheckGateTests(unittest.TestCase):
    def test_check_status_vocabulary(self) -> None:
        self.assertEqual(gates.check("a", True, "ok")["status"], "pass")
        self.assertEqual(gates.check("a", False, "no")["status"], "fail")
        self.assertEqual(gates.check("a", False, "no", required=False)["status"], "missing")

    def test_check_dict_shape_is_stable(self) -> None:
        self.assertEqual(
            gates.check("a", True, "ok"),
            {"name": "a", "status": "pass", "required": True, "message": "ok"},
        )

    def test_all_required_pass_ignores_optional_and_missing(self) -> None:
        checks = [
            gates.check("a", True, "ok"),
            gates.check("b", False, "skipped", required=False),
        ]
        self.assertTrue(gates.all_required_pass(checks))
        checks.append(gates.check("c", False, "bad"))
        self.assertFalse(gates.all_required_pass(checks))


class BlockingGateTests(unittest.TestCase):
    def test_gate_status_and_optional_gate(self) -> None:
        self.assertEqual(gates.gate_status("a", True, "ok").status, "pass")
        self.assertEqual(gates.gate_status("a", False, "no").status, "fail")
        self.assertEqual(gates.optional_gate("a", False, "warn").status, "warn")

    def test_failing_gates_and_blocked_until(self) -> None:
        g = [
            gates.gate_status("a", True, "ok"),
            gates.optional_gate("b", False, "soft"),
            gates.gate_status("c", False, "hard fail"),
        ]
        self.assertEqual([gate.name for gate in gates.failing_gates(g)], ["c"])
        self.assertEqual(gates.blocked_until(g), ["c: hard fail"])

    def test_gate_dict_serialization_matches_legacy(self) -> None:
        gate = gates.gate_status("a", True, "ok")
        self.assertEqual(gate.__dict__, {"name": "a", "status": "pass", "message": "ok"})


if __name__ == "__main__":
    unittest.main()
