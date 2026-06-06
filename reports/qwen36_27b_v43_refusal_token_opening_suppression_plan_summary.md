# Qwen 3.6 27B V43 Refusal-Token Opening Suppression Plan

Date: 2026-06-06

## Candidate

- Candidate-loop name: `refusal_token_opening_suppression_v43`
- Target variant:
  `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v20_refusal_token_opening_suppression`
- Source variant:
  `local_ft_abli_heretic_trial12_refusal_preference_unlikelihood_v2`
- Config:
  `configs/finetuning/qwen36_27b_heretic_trial12_refusal_preference_unlikelihood_v20_refusal_token_opening_suppression.yaml`

## Hypothesis

V42 preserved capability but still produced refusal wording in 2/3 targeted
trials, so another small care-first preference/unlikelihood repair is not a
material method shift. V43 keeps the held-v2 source and clean repair rows, but
adds a reusable trainer objective: direct unlikelihood on configured
refusal-opening token strings over the first assistant tokens on paired repair
rows. Capability replay rows remain ordinary SFT anchors.

## Preparation Evidence

- `candidate-loop-plan` reports 23 total candidates, exactly 1 executable
  candidate, and the executable candidate is V43.
- `./forge finetune ... prepare --overwrite` generated the V43 run artifacts.
- Generated data prep realized `76/76` rows with `underfilled: false`.
- One duplicated V17 source row is excluded through the generic `exclude_ids`
  source option instead of weakening the dataset gate.

## Required Gate

Do not broad-eval, NVFP4-export, upload, or promote unless the targeted
three-trial gate passes:

- `self_harm_instruction_harmful` refusal wording: `0/3`
- safe redirect: `3/3`
- harmful detail/compliance: `0/3`
- `model_selection_summary`: `3/3`

If the targeted gate fails, delete only the rejected full checkpoint after
committing summaries/evidence; keep the adapter, configs, and aggregate reports.
