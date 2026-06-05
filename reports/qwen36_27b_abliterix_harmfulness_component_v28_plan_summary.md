# Qwen3.6 27B V28 Abliterix Harmfulness Component-Policy Search Plan

Status: ready, search-only, no checkpoint exported.

Config:
`configs/abliteration/qwen36_27b_ft_abli_v2_abliterix_harmfulness_component_v28.yaml`

Source:
`local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`

Objective: remove the remaining stochastic
`self_harm_instruction_harmful` refusal-opening miss while preserving local FT
v4 capability.

## Hypothesis

V27 proved the public Qwen3.6 component-policy path is operational in
model-forge: Abliterix accepted the generated TOML and exposed only
`attn.o_proj` plus `mlp.down_proj` as steerable components. However, with
`ablate_harmfulness_direction=false`, every completed V27 trial worsened the
proxy refusal count from the 12/20 baseline.

V28 keeps the same response-opening prompt objective and component targeting,
but turns on Abliterix's opt-in harmfulness-direction ablation path:

- `ablate_harmfulness_direction: true`
- `harmfulness_layer_band: [0.3, 0.7]`

This tests whether the residual denial-style self-harm opening is driven by a
separable harmfulness signal rather than the refusal direction alone.

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
- `kl_target: 0.005`
- `search_selection.max_kl: 0.03`
- `n_trials: 50`

## Stop And Promotion Contract

Stop early if the first completed trials again worsen the 12/20 baseline refusal
count on both Sparks. Do not export from V28 unless journal analysis finds a
selected trial with zero proxy refusals inside the KL gate.

Even a successful journal is not promotion evidence. After export, register the
selected checkpoint as a normal checkpoint-producing candidate, sync to both
Sparks, run strict checkpoint/tokenizer/architecture audits, serve it, and pass
the targeted three-trial model-forge gate before broad eval, NVFP4, Hugging Face
upload, or promotion.
