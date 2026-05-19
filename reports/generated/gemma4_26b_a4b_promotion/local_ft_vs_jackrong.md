# Promotion Report: local_ft_vs_jackrong

- Family: `gemma4_26b_a4b`
- Candidate: `local_ft`
- Reference: `ft`
- Objective: `capability_sft`
- Decision: `hold`
- Passed: `false`

| Status | Gate | Metric | Candidate | Target | Reason |
|---|---|---|---:|---:|---|
| FAIL | challenge_capability_matches_jackrong | `capability_preservation_challenge.normal_use_regression_pass_rate` | 0.7708 | 0.7812 | 0.7708 >= 0.7812 |
| PASS | normal_use_regression_floor | `normal_use_regression.normal_use_regression_pass_rate` | 1.0 | 0.95 | 1 >= 0.95 |
| PASS | paired_benign_quality_matches_jackrong | `refusal_paired_boundary.benign_answer_quality_rate` | 0.7333 | 0.5 | 0.7333 >= 0.5 |
| PASS | reasoning_style_floor | `reasoning_style_stability.workflow_success` | 0.9333 | 0.9 | 0.9333 >= 0.9 |
