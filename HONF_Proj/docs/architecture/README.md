# Forward architecture index

The maintained forward platform is the Stage-7 fixed/context-fusion model:

- primary profile: `src/config_core/forward/stage7_structured_context.json`;
- accepted scientific checkpoint: Run 1401 best-by-field, epoch 4585;
- historical compatibility checkpoint: Run 1000 best-by-field, epoch 9655;
- profile status index: `src/config_core/forward/profile_registry.json`;
- frozen checkpoint and schema evidence: `docs/experiments/stage1_7_freeze_manifest.json` and `tests/fixtures/forward_cleanup/`.

See [forward_core.md](forward_core.md), [thermalchannel.md](thermalchannel.md), and
[compatibility.md](compatibility.md) for ownership and compatibility boundaries.
