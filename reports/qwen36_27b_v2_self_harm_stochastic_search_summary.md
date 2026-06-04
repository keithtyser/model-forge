# Qwen 3.6 27B V2 Self-Harm Stochastic Heretic Search

Status: stopped negative probe. Do not export or promote.

## Hypothesis

The prior single-prompt Heretic search from the held Qwen FT-abli v2 checkpoint
had no baseline refusal signal, even though model-forge targeted repeats caught
explicit refusal wording in 1/3 trials. A richer search-only signal with
weighted prompt variants might reproduce the stochastic refusal style and give
Heretic something measurable to suppress.

## Repo Change

The Heretic prompt materializer now supports per-section prompt variants through
`*_prompt_variants`. Variants preserve duplicate rows on purpose, which lets a
generic config overweight rare observed failure modes without model-specific
code. Heretic journal analysis now also reports a `min_base_refusals` gate, so
zero-baseline searches are rejected explicitly instead of appearing as generic
near misses.

## Configuration

- Config:
  `configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_stochastic_search.yaml`
- Source checkpoint:
  `~/models/Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2`
- Backend: Heretic container, search-only, no checkpoint export.
- Weighted bad train rows: 6.
- Weighted bad eval rows: 4.
- Gates: refusals <= 0, KL <= 0.04, refusal reduction >= 2, baseline refusals >= 2.

## Commands

```bash
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_stochastic_search.yaml sota-plan
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_stochastic_search.yaml sota-prepare
MODEL_FORGE_EXECUTE_HERETIC=1 ./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_stochastic_search.yaml sota-run --execute
./forge ablate --config configs/abliteration/qwen36_27b_ft_abli_v2_self_harm_stochastic_search.yaml heretic-search-analyze --output reports/generated/qwen36_27b_v2_self_harm_stochastic_search_analysis.json
```

## Result

The guarded Heretic container loaded the held v2 checkpoint and counted initial
refusals over the weighted bad eval set:

```text
Initial refusals: 0/4
```

Because the baseline signal was still zero and the config requires at least two
baseline refusals, the run was stopped before wasting time on 24 ineligible
trials. The analyzer reported no complete trials and `do_not_export`.

## Decision

Reject this search path. Weighted prompt variants are useful generic repo
infrastructure, but Heretic's deterministic refusal counter is not reproducing
the remaining Qwen v2 stochastic refusal. Do not export, quantize, upload, or
promote the stochastic-selected output path.

The next Qwen FT-abli method should not be another deterministic Heretic
projection from this same signal. Use either a multi-sample model-forge objective
that directly optimizes refusal wording probability, or a category-conditioned
repair method whose training/eval loop observes the same stochastic targeted
repeat gate used for promotion.
