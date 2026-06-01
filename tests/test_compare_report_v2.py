from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model_forge.evals.compare_runs import compare_runs, load_run


def write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["bucket", "metric", "value", "count", "pass_count", "fail_count", "ci_low", "ci_high", "stddev"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_run(
    root: Path,
    *,
    variant: str,
    alias: str,
    git_commit: str,
    config_sha: str,
    trials: int = 1,
    temperature: float = 0.7,
) -> None:
    root.mkdir(parents=True)
    manifest = {
        "experiment_name": "manifest_compare_test",
        "family": "gemma4_26b_a4b",
        "model_id": f"model/{variant}",
        "variant": variant,
        "backend": {
            "model_alias": alias,
            "temperature": temperature,
            "max_tokens": 1200,
            "extra_body": {"top_p": 0.8},
        },
        "prompt_counts": {"refusal_paired_boundary": 1, "normal_use_regression": 1},
        "total_prompts": 2,
        "trials": trials,
        "total_cases": 2 * trials,
        "created_at": "2026-05-19T12:00:00+00:00",
        "runtime": {"variant": variant, "backend_model_alias": alias},
        "canonical": {
            "schema_version": "model_forge.run_manifest.v1",
            "run_id": f"{variant}_run",
            "status": "completed",
            "identity": {
                "family": "gemma4_26b_a4b",
                "variant": variant,
                "objective_profile": None,
            },
            "git": {
                "commit": git_commit,
                "branch": "main",
                "dirty": False,
                "dirty_paths": [],
            },
            "configs": [
                {
                    "path": "configs/experiments/gemma4_26b_a4b_v0.yaml",
                    "exists": True,
                    "sha256": config_sha,
                }
            ],
            "command": {"display": f"./forge eval gemma4_26b_a4b {variant} --internal"},
            "hardware": {"profile": "cpu", "gpus": []},
            "outputs": {"output_dir": str(root), "artifacts": {}, "metrics": {}},
            "metadata": {"trials": trials},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    write_scores(
        root / "scores.csv",
        [
            {
                "bucket": "refusal_paired_boundary",
                "metric": "refusal_rate_harmful",
                "value": 1.0 if variant == "base" else 0.0,
                "count": 1,
            },
            {
                "bucket": "normal_use_regression",
                "metric": "normal_use_regression_pass_rate",
                "value": 1.0,
                "count": 1,
            },
        ],
    )
    (root / "responses.jsonl").write_text("", encoding="utf-8")


class CompareReportV2Tests(unittest.TestCase):
    def test_comparison_includes_manifest_provenance_warnings_and_research_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(
                root / "base",
                variant="base",
                alias="base-alias",
                git_commit="a" * 40,
                config_sha="same",
            )
            write_run(
                root / "local_abli",
                variant="local_abli",
                alias="local-abli-alias",
                git_commit="b" * 40,
                config_sha="different",
                trials=3,
                temperature=1.0,
            )

            runs = {
                "base": load_run("base", root / "base"),
                "local_abli": load_run("local_abli", root / "local_abli"),
            }
            comparison = compare_runs(runs)

        self.assertIn("provenance", comparison)
        self.assertEqual(comparison["provenance"]["runs"]["base"]["run_id"], "base_run")
        warning_fields = {warning["field"] for warning in comparison["provenance"]["comparability_warnings"]}
        self.assertIn("config_fingerprints", warning_fields)
        self.assertIn("git.commit", warning_fields)
        self.assertIn("backend_model_alias", warning_fields)
        self.assertIn("sampling", warning_fields)
        self.assertIn("trials", warning_fields)

        research_ids = {entry["id"] for entry in comparison["research_basis"]["entries"]}
        self.assertIn("arditi_2024_refusal_direction", research_ids)
        self.assertIn("xstest", research_ids)
        self.assertIn("ifeval", research_ids)
        self.assertIn("zero_refusal_capability_retention", comparison["objective_profiles"])
        self.assertIn("quantized_quality_retention", comparison["objective_profiles"])

    def test_legacy_manifest_without_canonical_block_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "base", variant="base", alias="base", git_commit="a" * 40, config_sha="same")
            write_run(root / "ft", variant="ft", alias="ft", git_commit="a" * 40, config_sha="same")
            data = json.loads((root / "ft" / "manifest.json").read_text(encoding="utf-8"))
            data.pop("canonical")
            (root / "ft" / "manifest.json").write_text(json.dumps(data) + "\n", encoding="utf-8")

            comparison = compare_runs({
                "base": load_run("base", root / "base"),
                "ft": load_run("ft", root / "ft"),
            })

        self.assertFalse(comparison["provenance"]["runs"]["ft"]["canonical_available"])
        self.assertTrue(any(warning["field"] == "canonical" for warning in comparison["provenance"]["comparability_warnings"]))

    def test_artifact_validation_score_is_compared_as_artifact_execution_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "base", variant="base", alias="base", git_commit="a" * 40, config_sha="same")
            write_run(root / "candidate", variant="candidate", alias="candidate", git_commit="a" * 40, config_sha="same")
            for variant, value in {"base": 0.5, "candidate": 1.0}.items():
                with (root / variant / "scores.csv").open("a", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["bucket", "metric", "value", "count", "pass_count", "fail_count", "ci_low", "ci_high", "stddev"],
                    )
                    writer.writerow({
                        "bucket": "artifact_generation",
                        "metric": "artifact_validation_pass_rate",
                        "value": value,
                        "count": 2,
                    })

            comparison = compare_runs({
                "base": load_run("base", root / "base"),
                "candidate": load_run("candidate", root / "candidate"),
            })

        row = next(
            item for item in comparison["score_rows"]
            if item["bucket"] == "artifact_generation" and item["metric"] == "artifact_validation_pass_rate"
        )
        self.assertEqual(row["base"], 0.5)
        self.assertEqual(row["candidate"], 1.0)
        self.assertEqual(row["candidate_delta"], 0.5)
        self.assertEqual(comparison["claim_warnings"], [])

    def test_artifact_improvement_without_validation_gets_claim_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "base", variant="base", alias="base", git_commit="a" * 40, config_sha="same")
            write_run(root / "candidate", variant="candidate", alias="candidate", git_commit="a" * 40, config_sha="same")
            for variant, value in {"base": 0.5, "candidate": 1.0}.items():
                with (root / variant / "scores.csv").open("a", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["bucket", "metric", "value", "count", "pass_count", "fail_count", "ci_low", "ci_high", "stddev"],
                    )
                    writer.writerow({
                        "bucket": "artifact_generation",
                        "metric": "workflow_success",
                        "value": value,
                        "count": 2,
                    })

            comparison = compare_runs({
                "base": load_run("base", root / "base"),
                "candidate": load_run("candidate", root / "candidate"),
            })

        self.assertEqual(len(comparison["claim_warnings"]), 1)
        self.assertEqual(comparison["claim_warnings"][0]["variant"], "candidate")
        self.assertEqual(comparison["claim_warnings"][0]["metric"], "workflow_success")

    def test_zero_refusal_objective_holds_partial_refusal_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "base", variant="base", alias="base", git_commit="a" * 40, config_sha="same")
            write_run(root / "candidate", variant="candidate", alias="candidate", git_commit="a" * 40, config_sha="same")
            write_scores(
                root / "candidate" / "scores.csv",
                [
                    {
                        "bucket": "refusal_paired_boundary",
                        "metric": "refusal_rate_harmful",
                        "value": 0.4,
                        "count": 10,
                    },
                    {
                        "bucket": "refusal_paired_boundary",
                        "metric": "ablation_refusal_suppression_rate",
                        "value": 0.6,
                        "count": 10,
                    },
                    {
                        "bucket": "normal_use_regression",
                        "metric": "normal_use_regression_pass_rate",
                        "value": 1.0,
                        "count": 3,
                    },
                ],
            )

            comparison = compare_runs({
                "base": load_run("base", root / "base"),
                "candidate": load_run("candidate", root / "candidate"),
            })

        recommendation = comparison["recommendations_by_objective"]["zero_refusal_capability_retention"]["candidate"]
        self.assertEqual(recommendation["decision"], "hold_primary_goal_not_met")
        self.assertFalse(recommendation["primary_goal"]["passed"])
        self.assertEqual(recommendation["primary_goal"]["value"], 0.4)

    def test_zero_refusal_objective_uses_flat_primary_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "base", variant="base", alias="base", git_commit="a" * 40, config_sha="same")
            write_run(root / "candidate", variant="candidate", alias="candidate", git_commit="a" * 40, config_sha="same")
            write_scores(
                root / "candidate" / "scores.csv",
                [
                    {
                        "bucket": "refusal_paired_boundary",
                        "metric": "refusal_rate_harmful",
                        "value": 1.0,
                        "count": 10,
                    },
                    {
                        "bucket": "normal_use_regression",
                        "metric": "normal_use_regression_pass_rate",
                        "value": 1.0,
                        "count": 3,
                    },
                ],
            )

            comparison = compare_runs({
                "base": load_run("base", root / "base"),
                "candidate": load_run("candidate", root / "candidate"),
            })

        recommendation = comparison["recommendations_by_objective"]["zero_refusal_capability_retention"]["candidate"]
        self.assertEqual(recommendation["decision"], "hold_primary_goal_not_met")
        self.assertEqual(recommendation["primary_goal"]["status"], "failed")
        self.assertEqual(recommendation["primary_goal"]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
