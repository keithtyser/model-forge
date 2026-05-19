# datasets

Start narrow. No giant synthetic slop dump.

v0 dataset buckets:
- task-specific structured planning JSON
- refusal boundary probes
- normal-use regression set

Every dataset should have:
- source note
- schema
- generation method
- filter rule
- version tag

## Generated Data

Small smoke/generated packs can be tracked when they are useful for tests,
handoff, or review. Full generated training datasets should be published to
Hugging Face and represented in Git by manifests, dataset cards, review reports,
and publish plans.

The current tracked `datasets/generated/gemma4_26b_a4b_local_ft_v1/` pack is a
smoke-size quality-control artifact, not a final training dataset.
