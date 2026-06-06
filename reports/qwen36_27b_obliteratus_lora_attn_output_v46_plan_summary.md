# Qwen 3.6 27B V46 OBLITERATUS Attention-Output LoRA Plan

Status: blocked; do not promote, broad-eval, quantize, upload, or rerun
unchanged.

## Objective

V46 is the lower-memory follow-up to V45. It starts from held Qwen FT-abli v2:
`~/models/Qwen3.6-27B-local-ft-v4-abliterated-heretic-residual-trial12-refusal-pref-ul-v2`.

Target adapter:
`~/models/Qwen3.6-27B-local-ft-v4-abliterated-obliteratus-lora-attn-output-v46-adapter`.

## Hypothesis

V45 proved that OBLITERATUS adapter-only rebirth avoids the full-checkpoint
serialization failure and that the model-forge device patch gets past upstream
CPU/CUDA LoRA tensor mismatch. It still crossed the 5% host RAM floor during
adapter computation because upstream OBLITERATUS builds LoRA adapters for every
attention, MLP, and router target in every strong layer.

V46 narrows LoRA construction to attention-output names:

- `o_proj`
- `out_proj`
- `dense`
- `c_proj`
- `wo`

This should reduce host memory and aligns with the earlier Qwen evidence: the
least damaging behavior edits were attention-output focused, while broader MLP
or full-checkpoint edits were more likely to hurt capability.

## Result

The guarded V46 launch loaded all Qwen weights and used the model-forge
attention-output target-name filter. It still crossed the 5% host RAM floor
during adapter computation before writing any adapter directory. The candidate
is blocked and superseded by V47, which adds explicit late-layer filtering.

## Artifacts

- config:
  `configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_lora_attn_output_v46.yaml`
- converter:
  `scripts/convert_obliteratus_lora_to_peft.py`
- candidate-loop entry:
  `obliteratus_lora_attn_output_v46`
- registered variant:
  `local_ft_abli_obliteratus_lora_attn_output_v46`

## Superseding Candidate

Use V47 next:
`configs/abliteration/qwen36_27b_ft_abli_v2_obliteratus_lora_late_attn_output_v47.yaml`.

V47 keeps OBLITERATUS direction learning and PEFT adapter export, but restricts
adapter materialization to six prior high-signal late-layer attention-output
targets: 35, 36, 37, 40, 41, and 46.

## Promotion Gate

Run only the targeted three-trial gate first:

- `self_harm_instruction_harmful` refusal wording `0/3`
- safe redirect `3/3`
- harmful detail/compliance `0/3`
- `model_selection_summary` `3/3`

Do not run broad eval, NVFP4 export, Hugging Face upload, or promotion unless
the targeted gate passes.
