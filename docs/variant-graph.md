# Variant Graph

Model Forge variants are graph nodes, not only flat names. A node records where
the candidate came from, which transform produced it, what evidence exists, and
what should happen to the artifact next.

Inspect a family graph without loading a model:

```bash
./forge variants graph gemma4_26b_a4b
./forge variants graph gemma4_26b_a4b \
  --variant ft_local_abli_sota_internal_r7_selected_t34_transfer_nvfp4_modelopt \
  --json
```

Write a variant node:

```bash
./forge variants node gemma4_26b_a4b local_ft \
  --implementation-status implemented \
  --validation-state spark_single_node_validated \
  --promotion-decision inconclusive \
  --command './forge eval gemma4_26b_a4b local_ft --internal' \
  --spark-evidence-path reports/generated/example \
  --node-count 1 \
  --hardware-profile dgx_spark \
  --write
```

The default output is:

```text
reports/generated/variants/<family>/<variant>/variant_node.json
```

Generated nodes stay under `reports/generated/` unless a small reusable example
is intentionally promoted into docs or recipes. They can reference large local
checkpoints, but model weights do not belong in Git.

Each node includes:

- source variant and transform type
- model identity and served model name
- checkpoint path and optional artifact checksums
- validation implementation status and Spark validation state
- command, node count, topology, metrics, logs, and promotion decision
- retention decision, disk budget, and keep-until date

Validate a node:

```bash
./forge variants audit-node reports/generated/variants/gemma4_26b_a4b/local_ft/variant_node.json
```

`./forge doctor` validates any tracked `variant_node.json` files, so checked-in
examples cannot drift from the schema.

## Tokenizer And Chat Template Audit

Derived variants should preserve tokenizer files, special-token semantics, and
chat-template behavior unless a config explicitly documents a change. This is
especially important after adapter merges, refusal ablation exports,
quantization exports, and future GGUF conversions.

Run a metadata-only audit without loading model weights:

```bash
./forge variants tokenizer-audit gemma4_26b_a4b --variant local_abli
```

Metadata mode checks configured local dirs when present, hashes tokenizer files,
extracts special tokens and chat-template sources, and compares derived variants
against their `base_variant` source. Missing local dirs are warnings by default
so open-source configs remain portable.

For a release or promotion gate, use strict mode and a live tokenizer round
trip:

```bash
./forge variants tokenizer-audit gemma4_26b_a4b \
  --variant local_abli \
  --load-tokenizer \
  --strict
```

`--load-tokenizer` uses `AutoTokenizer.from_pretrained(..., local_files_only=True)`
and applies the chat template to a small message fixture, then tokenizes and
decodes it to catch broken templates before serving or publishing.
