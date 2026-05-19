# Gemma 4 A4B Local FT v0 Recipe Snapshot

This directory preserves the small generated files from the validated Gemma
local FT v0 workflow. It is a reusable reference, not the active runtime
directory. Paths are normalized to repo-relative or `~/models/...` forms so the
snapshot is readable outside the original workstation.

Source config:

```text
configs/finetuning/gemma4_26b_a4b_local_ft_v0.yaml
```

Runtime output path:

```text
runs/finetune/gemma4_26b_a4b_local_ft_v0/
```

To regenerate active run files:

```bash
./forge finetune gemma4_26b_a4b prepare --overwrite
```

The generated runtime directory may also contain large local-only files such as
`train.jsonl`, tokenized caches, Python overlays, checkpoints, and logs. Those
do not belong in this tracked recipe directory.
