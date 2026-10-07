# ThermalChannel physical and data contract

## Physical problem

The intended application is steady channel flow containing a padded,
variable-size collection of circular heated solid modules, with fluid energy
transport and solid conduction interacting at module boundaries. The active
benchmark labels use a more limited generator: a prescribed analytic wake
velocity/pressure approximation followed by temperature evolution on one
shared fluid/solid grid. They are not a coupled Navier–Stokes CFD reference.

The locally inspected generator is
`1_Demo_ChannelThermal/src/simulate_channelthermal.py`. All 690 packed case
configurations inspected on 2026-10-06 use `flow_model="analytic_wake"`,
`apply_projection=false`, `nu=null`, and `viscosity_scale=1`. Velocity,
pressure and vorticity read layout, radius, domain/grid geometry and prescribed
inlet/Reynolds context; they do not read heating or temperature. The outlet
pressure mean fixes the pressure gauge. The optional projection routine also
has no thermal input, but is inactive in these packed cases. This dependency
applies to this benchmark variant; it does not establish heat-independent
flow for buoyant or temperature-dependent flow applications.

The thermal grid uses fixed fluid/solid diffusivities, upwind advection by the
prescribed flow, solid-cell heat deposition, inlet/wall Dirichlet temperature
and a zero-gradient outlet. At a fixed layout, context and common integration
step count, its temperature update is affine in module heating. The saved
converged-final extraction uses input-dependent stopping tolerances and FP32
storage, so saved endpoint superposition remains a measured numerical property,
not an exact certified steady/grid-error claim. No temperature-dependent
coefficients, buoyancy or radiation are used by the inspected generator.

`q_normal` is a sampled shared-grid flux proxy,
`-k_interface*(T_outside-T_surface)/delta`, with harmonic interface conductivity;
it is not a independently solved conjugate-interface flux. The raw
`h_proxy=abs(q_normal)/(abs(T_surface-T_outside)+1e-6)` and the preprocessed
`h_effective=q_normal/(T_surface-T_outside)` have ratio, epsilon, clipping and
validity-mask semantics. They may depend on heating even though flow does not.
The effective denominator is clamped below the configured epsilon and its
value is clipped to the configured range. Outside temperature, surface
temperature, `h_effective` and flux proxy are distinct quantities; none may be
silently substituted for another.

The source and saved-state audit is recorded in
[the dependency and thermal-response audit](../../docs/reports/_bk/20261006_154229Z_HONF_Dependency_Correct_Source_Audit.md).

The global learned field is ordered as:

```text
[u, v, p, omega, temperature]
```

The local module operator receives material/heat descriptors and an angular
Robin-condition port sequence. It predicts solid temperature at normalized
disk coordinates and the boundary response `[T_surface, q_normal]`.

## Global dataset

Logical ID: `thermal_channel_global_v1`.

The root stores `case_ids`, `splits`, feature-name arrays, normalization
statistics, and a `cases/<case_id>` group. The active dataset contains 690
cases: 600 train and 90 test. Important root attributes include `field_dim=5`,
`max_modules=12`, `n_interface_points=64`, `local_grid_size=64`, and
`target_mode=converged_final`.

Each case provides:

- `sampled_points`: `[x,y,u,v,p,omega,T]` rows used for continuous-field
  supervision;
- optional `sampled_point_weights` and group labels for weighted sampling;
- `module_centers [M,2]`, `heat_powers [M]`, and `module_present [M]`;
- material/domain metadata, including Reynolds number, inlet velocity,
  diffusivities, conductivities, module radius, and domain lengths;
- `interface_condition [M,P,C]` with the declared feature-name order;
- `interface_target [M,P,2] = [T_surface,q_normal]`;
- module internal temperature grids and masks;
- optional solved-field organizer targets used only for training diagnostics
  or auxiliary supervision;
- full `x_grid`, `y_grid`, and steady field arrays for evaluation.

The maintained interface-condition feature order is:

```text
[theta, normal_x, normal_y, T_outside,
 u_normal, u_tangent, h_proxy, h_effective]
```

The Stage-A teacher port projection is:

```text
[theta, normal_x, normal_y, T_outside, h_effective]
```

## Local Stage-A dataset

Logical ID: `thermal_disk_local_v1`.

The active dataset contains 1,034 samples: 919 train and 115 test. Every item
contains:

- `module_params [7]`: heat, solid conductivity/diffusivity, mean/std of
  `h_effective`, and mean/std of outside temperature;
- `port_tokens [P,5]`: `[theta,cos(theta),sin(theta),T_env,h]`;
- `internal_query_points [Ql,2]`: normalized coordinates inside the disk;
- `internal_temperature_targets [Ql,1]`;
- `interface_targets [P,2]`: `[T_surface,q_normal]`;
- optional local grids, masks, solver labels, modal counts, and roughness
  diagnostics used by evaluation.

The mixed Stage-A workflow combines these standalone samples with every active
module extracted from global cases. It fits one normalizer from all physical
training samples and injects that immutable transform into validation and
checkpoint evaluation.

## Padding and masks

Global cases use a fixed maximum number of slots. `module_present=1` marks an
active module; inactive slots must contribute zero to core assignments, local
inference, losses, and metrics. Slot order may change only when all
module-indexed tensors are permuted consistently.

## Normalization

Global normalization statistics are stored under the HDF5 root
`normalization` group. When normalization is enabled, the training reader owns
the transform and validation receives the same normalizer object. Stage-A
mixed training deliberately refits one transform over its combined training
sources; it never fits validation statistics.

Checkpoints embed normalization configuration and arrays. Evaluation must use
checkpoint-owned statistics and must not refit on the selected split.

## Integrity and versioning

`dataset_manifest.json` records the active file sizes, SHA-256 fingerprints,
required root keys, schemas, and split counts. Launch validation checks file
size and required keys before confirmation. Full hash verification is
available to resource tooling and should be run when a dataset is copied or
replaced.

Changing feature order, target meaning, normalization semantics, or required
keys requires a new schema and logical dataset ID, even if the HDF5 filename
stays `packed_dataset.h5`.
