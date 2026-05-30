from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_forge.doctor import (
    Finding,
    check_generated_dataset_policy,
    check_machine_paths,
    check_model_family_configs,
    check_secret_literals,
    run_checks,
)


class DoctorTests(unittest.TestCase):
    def test_repo_hygiene_passes(self) -> None:
        self.assertEqual(run_checks(), [])

    def test_secret_literal_detection(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = "hf_" + "a" * 32
            (root / "card.md").write_text(f"token = {token}\n")
            findings = check_secret_literals(["card.md"], root)
        self.assertEqual(
            findings,
            [Finding("secret_literals", "literal secret-like value matched huggingface_token", "card.md", 1)],
        )

    def test_machine_path_detection(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            machine_path = "/" + "home/example/models/foo"
            (root / "config.yaml").write_text(f"local_dir: {machine_path}\n")
            findings = check_machine_paths(["config.yaml"], root)
        self.assertEqual(
            findings,
            [
                Finding(
                    "machine_paths",
                    "machine-specific absolute path found; use repo-relative paths, ~/models, or env overrides",
                    "config.yaml",
                    1,
                )
            ],
        )

    def test_generated_dataset_policy_allows_current_smoke_pack(self) -> None:
        files = [
            "datasets/generated/gemma4_26b_a4b_local_ft_v1/dataset.jsonl",
            "datasets/generated/other_full_dataset/dataset.jsonl",
        ]
        self.assertEqual(
            check_generated_dataset_policy(files),
            [
                Finding(
                    "generated_dataset_policy",
                    "generated dataset output is tracked without an explicit allowlist entry",
                    "datasets/generated/other_full_dataset/dataset.jsonl",
                )
            ],
        )

    def test_machine_path_check_skips_archived_roadmaps(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "docs" / "roadmaps"
            path.mkdir(parents=True)
            machine_path = "/" + "home/example/models/foo"
            (path / "example.md").write_text(f"historical path: {machine_path}\n")
            findings = check_machine_paths(["docs/roadmaps/example.md"], root)
        self.assertEqual(findings, [])

    def test_model_family_config_audit_passes_current_configs(self) -> None:
        self.assertEqual(check_model_family_configs(), [])


if __name__ == "__main__":
    unittest.main()
