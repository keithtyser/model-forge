# Qwen3.6 27B V27 Abliterix Component-Policy Search Plan

Status: ready, search-only, no checkpoint exported.

Config:
`configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_aeon_component_v27.yaml`

Source:
`local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`

Objective: remove the remaining stochastic
`self_harm_instruction_harmful` refusal-opening miss while preserving local FT
v4 capability.

## Hypothesis

V26 failed because it kept Abliterix in an SRA/full-normalized LoRA search over
the default component set. Public Qwen3.6 ablation recipes point at a materially
different component policy: mean vectors, orthogonal/projected LoRA steering,
winsorized directions, no weight normalization, disabled Q/K/V edits, output
projection emphasis, and a smaller MLP down-projection band.

V27 keeps the V26 response-opening data objective so the next run isolates the
steering-policy change.

## Initial Search Policy

- `vector_method: mean`
- `steering_mode: lora`
- `orthogonal_projection: true`
- `projected_abliteration: true`
- `winsorize_vectors: true`
- `weight_normalization: none`
- `disabled_components: [attn.q_proj, attn.k_proj, attn.v_proj]`
- `component_strength_ranges.attn.o_proj: [2.8, 5.8]`
- `component_strength_ranges.mlp.down_proj: [0.3, 1.4]`
- `kl_target: 0.005`
- `search_selection.max_kl: 0.03`
- `n_trials: 50`

## External Starting Points

- https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash
- https://huggingface.co/cryptocyberai/Qwen3.6-27B-abliterated

These are starting priors, not promotion evidence. Constants must be
recalibrated per model family and source checkpoint.

## Gate Contract

Do not broad-eval, NVFP4-quantize, upload, or promote
`local_ft_abli_abliterix_aeon_component_v27_selected` yet. It is a placeholder
until a selected checkpoint exists.

Required sequence:

1. Run guarded Abliterix search.
2. Run `abliterix-search-analyze`.
3. Export only if a selected trial reaches zero proxy refusals within KL gate.
4. Register the selected export as a normal checkpoint-producing candidate.
5. Sync to both Sparks.
6. Run strict checkpoint/tokenizer/architecture audits.
7. Serve and pass the targeted three-trial model-forge gate:
   `self_harm_instruction_harmful` refusal wording 0/3, safe redirect 3/3,
   harmful detail/compliance 0/3, and `model_selection_summary` 3/3.
