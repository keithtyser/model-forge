"""Tests for compute-capability based arch classification + supported-quant derivation."""
from __future__ import annotations

import unittest

from model_forge.hardware import (
    arch_from_cc,
    detect_hardware_profile,
    supported_quant_for_cc,
)


class ArchFromCcTests(unittest.TestCase):
    def test_known_archs(self) -> None:
        self.assertEqual(arch_from_cc("8.0"), "ampere")    # A100
        self.assertEqual(arch_from_cc("8.6"), "ampere")    # A6000 / 3090
        self.assertEqual(arch_from_cc("8.9"), "ada")       # 4090 / L40
        self.assertEqual(arch_from_cc("9.0"), "hopper")    # H100
        self.assertEqual(arch_from_cc("10.0"), "blackwell")  # B200
        self.assertEqual(arch_from_cc("12.1"), "blackwell")  # GB10 / RTX 50xx
        self.assertEqual(arch_from_cc("7.5"), "volta_turing")

    def test_unknown_or_empty(self) -> None:
        self.assertEqual(arch_from_cc(""), "unknown")
        self.assertEqual(arch_from_cc("garbage"), "unknown")


class SupportedQuantTests(unittest.TestCase):
    def test_nvfp4_only_on_blackwell(self) -> None:
        self.assertIn("nvfp4", supported_quant_for_cc("12.1"))
        for cc in ("9.0", "8.9", "8.0", "7.5"):
            self.assertNotIn("nvfp4", supported_quant_for_cc(cc))

    def test_fp8_on_hopper_ada_not_ampere(self) -> None:
        self.assertIn("fp8", supported_quant_for_cc("9.0"))
        self.assertIn("fp8", supported_quant_for_cc("8.9"))
        self.assertNotIn("fp8", supported_quant_for_cc("8.0"))   # Ampere has no FP8 tensor cores

    def test_int8_awq_broadly_available(self) -> None:
        for cc in ("12.1", "9.0", "8.9", "8.0", "7.5"):
            self.assertIn("int8", supported_quant_for_cc(cc))
            self.assertIn("awq", supported_quant_for_cc(cc))

    def test_cpu_unknown_has_no_quant(self) -> None:
        self.assertEqual(supported_quant_for_cc(""), ())


class DetectProfileCapabilityTests(unittest.TestCase):
    def test_forced_profile_carries_simulated_capability(self) -> None:
        # simulate an A100 box without needing the hardware
        env = {"MODEL_FORGE_HARDWARE_PROFILE": "cuda_large_vram", "MODEL_FORGE_COMPUTE_CAP": "8.0"}
        profile = detect_hardware_profile(env)
        self.assertEqual(profile.compute_capability, "8.0")
        self.assertEqual(profile.supported_quant, ("int8", "awq", "gptq"))

    def test_hopper_capability(self) -> None:
        env = {"MODEL_FORGE_HARDWARE_PROFILE": "cuda_large_vram", "MODEL_FORGE_COMPUTE_CAP": "9.0"}
        self.assertEqual(detect_hardware_profile(env).supported_quant, ("fp8", "int8", "awq"))


if __name__ == "__main__":
    unittest.main()
