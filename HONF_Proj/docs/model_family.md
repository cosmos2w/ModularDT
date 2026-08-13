# Model families and baselines

`case_id` identifies physics/data; `model_family` identifies the reusable model
and artifact policy. `honf_forward` is implemented, `honf_inverse` is reserved,
and future baselines should receive their own stable family IDs. A baseline is
not a new physical case.

A model family owns its architecture schema, factory, generic execution needs,
checkpoint selector policy, and common artifacts. The installed case still
owns the data adapter, targets, physical losses, and evaluation semantics. Add
new families only from concrete model requirements; avoid a universal trainer
that weakens validation or silently ignores unsupported settings.

`honf_runtime.registry.require_model_family` is the launch gate. The built-in
registry enables `honf_forward`, marks `honf_inverse` explicitly unavailable,
and rejects unknown baseline IDs instead of silently dispatching them through
HONF. A future baseline must add a concrete implementation/spec and tests before
its ID becomes selectable.
