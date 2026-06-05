# Qwen3.6 27B V29 Patched Abliterix Harmfulness Component Search

Status: executed and rejected, search-only, no checkpoint exported.

Config:
`configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_harmfulness_component_v29.yaml`

Source:
`local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`

Objective: remove the remaining stochastic
`self_harm_instruction_harmful` refusal-opening miss while preserving local FT
v4 capability.

## Hypothesis

V28 reached residual extraction and measured the expected `12/20` baseline, but
failed before scoring because Abliterix's harmfulness/refusal extractor returned
shape `(2, layers + 1, hidden_dim)` while the LoRA steering path indexed that
tensor as if it were layer-aligned. V29 keeps V28's public Qwen3.6 component
policy and prompt objective, then applies the model-forge Abliterix compatibility
patch:

- compute the refusal/harmfulness pair with Abliterix
- reduce it with `harmfulness_pair_reduction: normalized_sum`
- pass a normal `(layers + 1, hidden_dim)` vector tensor to LoRA steering

This preserves the useful harmfulness/refusal separation idea without mutating
the installed Abliterix package in-place.

## Search Policy

- `vector_method: mean`
- `steering_mode: lora`
- `orthogonal_projection: true`
- `projected_abliteration: true`
- `winsorize_vectors: true`
- `weight_normalization: none`
- `disabled_components: [attn.q_proj, attn.k_proj, attn.v_proj]`
- `component_strength_ranges.attn.o_proj: [2.8, 5.8]`
- `component_strength_ranges.mlp.down_proj: [0.3, 1.4]`
- `ablate_harmfulness_direction: true`
- `harmfulness_pair_reduction: normalized_sum`
- `harmfulness_pair_weight: 1.0`
- `kl_target: 0.005`
- `search_selection.max_kl: 0.03`
- `n_trials: 50`

## Stop And Promotion Contract

Run V29 as a guarded short search on both Sparks. Stop early if the first
completed trials worsen the `12/20` baseline refusal count on both nodes. Do not
export unless journal analysis finds a selected trial with zero proxy refusals
inside the KL gate.

Even a successful journal is not promotion evidence. After export, register the
selected checkpoint as a normal checkpoint-producing candidate, sync to both
Sparks, run strict checkpoint/tokenizer/architecture audits, serve it, and pass
the targeted three-trial model-forge gate before broad eval, NVFP4, Hugging Face
upload, or promotion.

## Result

V29 ran to completion on both Sparks through the guarded Abliterix container.
The compatibility patch fixed the V28 vector-shape failure: both nodes loaded
the source checkpoint, measured the expected `12/20` proxy-refusal baseline,
applied steering, and completed `50` trials.

Coordinator analysis:
`reports/generated/qwen36_27b_v29/abliterix_harmfulness_component_v29_coordinator_analysis.json`

- completed trials: `50/50`
- inferred baseline: `12` refusals from Abliterix objective ratios
- best trial: index `44`, trial id `43`
- best proxy refusals: `8/20`
- best KL: `0.008346500806510448`
- recommendation: `do_not_export`

Worker analysis:
`reports/generated/qwen36_27b_v29/abliterix_harmfulness_component_v29_worker_analysis.json`

- completed trials: `50/50`
- inferred baseline: `12` refusals from Abliterix objective ratios
- best trial: index `14`, trial id `13`
- best proxy refusals: `9/20`
- best KL: `0.05595417320728302`
- recommendation: `do_not_export`

Decision: reject V29. It proves the patched Abliterix harmfulness-pair steering
path is operational, but this search space does not remove enough refusals to
justify export. Do not run `abliterix-export`, broad eval, NVFP4 export, Hugging
Face upload, or promotion for
`local_ft_abli_abliterix_harmfulness_component_v29_selected`.

## Follow-Up

The run exposed and fixed a model-forge analyzer gap: Abliterix journals do not
persist baseline refusals directly, but completed trials record objective values
where the second value is `refusals / baseline_refusals`. The analyzer now
infers `base_refusals` from that ratio and reports reduction columns instead of
leaving the baseline ambiguous.

The next ablation attempt should change method or objective, not rerun V29
unchanged. Use a stronger no-refusal target such as response-opening conditioned
vectors with direct refusal-phrase scoring, a streamed/source-tethered
OBLITERATUS export, or an SAE/activation-feature edit aimed at the residual
refusal-opening state.

## Validation

- `.venv/bin/python -m unittest tests.test_abliteration_pipeline -v`
- `.venv/bin/python -m unittest tests.test_abliteration_pipeline.AbliterationPlanTests.test_abliterix_search_analyze_recommends_export_runner_only_after_journal_gates tests.test_abliteration_pipeline.AbliterationPlanTests.test_abliterix_search_analyze_marks_missing_baseline_before_export_gate tests.test_abliteration_pipeline.AbliterationPlanTests.test_abliterix_search_analyze_infers_baseline_from_ratio_objective -v`
- Docker smoke inside `model-forge-abliterix:latest` confirmed
  `model_forge.integrations.abliterix_compat.apply_abliterix_compat_patches`
  makes both `abliterix.cli.compute_steering_vectors` and
  `abliterix.vectors.compute_steering_vectors` return `(layers, hidden)` tensors
  for `ablate_harmfulness_direction=true` on tiny residual fixtures.
