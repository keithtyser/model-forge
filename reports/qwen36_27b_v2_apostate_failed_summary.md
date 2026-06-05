# Qwen 3.6 27B V2 Apostate Attempt

Date: 2026-06-04

## Hypothesis

Held v2 remains the strongest Qwen FT-abli evidence node, but it still has one
stochastic self-harm denial-wording blocker. A preservation-direction baked
Apostate checkpoint might remove that residual refusal opening while preserving
source capability better than another sequential preference/unlikelihood repair.

## Source

- source variant: `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`
- config: `configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_apostate_plan.yaml`
- backend: Apostate 1.0.0 from `heterodoxin/apostate`
- wrapper: `scripts/run_apostate_container.sh`

## Run

```bash
docker build -f docker/apostate.Dockerfile -t model-forge-apostate:latest .
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_apostate_plan.yaml sota-prepare --backend apostate
MODEL_FORGE_MIN_AVAILABLE_RAM_FRACTION=0.05 MODEL_FORGE_MIN_FREE_DISK_FRACTION=0.15 MODEL_FORGE_APOSTATE_DOCKER_MEMORY_GB=110 MODEL_FORGE_APOSTATE_SHM_SIZE=32g \
  ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_apostate_plan.yaml sota-run --backend apostate --execute
```

## Result

- elapsed: 4828.7 seconds
- backend baseline refusal estimate: 0.7143
- backend edited refusal estimate: 0.5714
- best search/rerank trial: refusal 0.375, KL 0.0024, capability drift 0.0
- final harmless KL: 0.0443 nats
- failed baked checkpoint:
  `~/models/Qwen3.6-27B-local-ft-v4-abliterated-apostate-self-harm-selected`
- local backend summary:
  `artifacts/abliteration/qwen36_27b_ft_abli_v2_self_harm_apostate_plan/sota_apostate/model_forge_sota_apostate.json`

## Decision

Reject without model-forge eval. The backend did not reach the zero-refusal
objective; refinement worsened the refusal estimate from the best search/rerank
value, and the final backend test refusal remained 0.5714.

The failed 51 GiB baked checkpoint was deleted after the summary was captured.
Do not register, targeted-eval, broad-eval, quantize, upload, or promote
`local_ft_abli_apostate_self_harm_selected`.

## Next Step

Do not rerun this exact full balanced Apostate search unchanged. If Apostate is
retried, first change the search space and run a smaller diagnostic before
baking. Otherwise prioritize a multi-direction/SOM or optimal-transport-style
backend, because another single-profile preservation-aware baked edit left the
self-harm refusal cluster intact.
