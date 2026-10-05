# Matched lean interaction continuation

The lean experiment starts two new development identities from the inspected
Global-C Run3402 exact epoch 500 checkpoint. G-fast enables the same-operator
one-group reader; Tensor-H adds the zero-output source-group control residual.
This is warm-start adaptation with 500 inherited epochs and 100–500 additional
epochs. It does not restart or change either formal Run3501/3502.

`tools/thermal_lean_continuation.py` prepares children without launching them.
Run it in the ModularDT environment from HONF_Proj, with outputs outside Git:

```bash
python tools/thermal_lean_continuation.py \
  --parent-checkpoint /data/wanglz/ModularDT/thermal_development/HONF_Forward_Runs/ThermalChannel/HONF_Forward_Runs/Run_3402_20261005_000243_thermal_native_context25_global-c_v1/epoch_0500_model.pt \
  --output-root /data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/continuation \
  --first-run-id 3601
```

The tool checks the exact declared parent SHA256
`1a8ac8d976edc72e55950eb3404145adc4ca124e7a529010e02e1a882c5ff051`,
the fixed25_v1 semantic manifest, 150 selected training and 22 exposed validation
cases, epoch 500, absolute horizon 1000, FP32, ordinary task autograd and the
existing epoch 101 physical objective/response amendment. It rejects previously
attached children. Preparation refuses existing child/profile paths.

Both children keep every common physical/control parameter by name and shape.
AdamW moments are remapped by parameter names using the parent's literal
single-group traversal order; the sorted display inventory is not used to
guess moment positions. The inspected parent has 233 common trainable tensors:
213 carry moments at 2000 completed updates and 20 have no saved moments. Both
children preserve that distinction.
Only `core.backend.tensor_residual.*` may be new. Its final gamma projections
must be exactly zero before attachment, and its moments start empty. The new
parameters use the inherited constant AdamW learning rate 3e-4 and weight
decay 1e-5. The native trainer has no separate learning-rate scheduler; its
absolute 1000 objective/callback schedule remains unchanged.

The generated checkpoint preserves the parent RNG streams, normalization,
Stage-A weights/state, selection age and campaign calibration/counters.
`attachment_receipt.json` and `campaign_training_state` record the exact source
path/hash and parameter/moment inventory. The child selector starts empty for
both arms, comparing only their own saved 100-epoch reviews. The inherited
parent is preserved as the attachment checkpoint rather than copied into a
child best-field alias. Ordinary strict checkpoint/run/data/normalizer resume
validation remains active.

`prepared_children.json` gives each child's native launch command. Use physical
GPU 1 for G-fast and GPU 2 for Tensor-H, with logical `cuda:0`. The initial stop is
absolute 600, additional 100; review the saved outputs, real task gradients,
parameter changes and measured epoch cost before continuing. Resume that
child's exact `epoch_0600_model.pt` with `--epochs 1000` only within the declared
budget. Retain absolute 600/700/800/900/1000 milestones, latest and best-field
aliases. Never resume a child from full-data weights or regenerate memberships.

Tensor admission and donor memberships are soft at additional 1–100, blend to
sparse projections at 101–200 and use exact sparse projections at 201–500. The
saved absolute age reconstructs this forward function through
`residual_parent_epoch=500`. Positive support in the soft/blended stages is not
reported as exact learned sparse K.

G-fast applies case-constant gains after native sums and before biased output
projections. Tensor-H's base-plus-contrast reductions preserve the same native
operator and zero-output attachment while supplying genuine residual task
gradients. Both retain all fine physical sources. Sparse admission/control
donors describe control information; they do not establish sparse physical
retrieval, executor savings or physical causality. Trained source plans,
gamma, reference centering, access/actions and dense-value provenance are
exported separately from the base Global-C value plan.

Run the all 22 full-role normal statistical evaluation at the first review and
final endpoint, retaining detailed fields/graphs only for the same fixed four
representative cases. Use `--stage 600`/`--stage 1000`,
`--evaluation-scope development`, `--capture-phase-graphs` and
`--organization-summary` with the maintained evaluator. The stage label is the
absolute checkpoint age; report additional age alongside it. Saved development
metrics are exposed validation evidence, not independent test results.

Immutable runtime source snapshots may be kept outside Git for launched
children. Keep the original project resource/configuration anchor so managed
resume identities and the existing physical atlas/Stage-A bindings do not
change merely because executable Python moved. Record snapshot hashes and
the anchor explicitly. Account for every associated GPU interval, including
failed starts, tests and evaluation, without touching unrelated processes.
