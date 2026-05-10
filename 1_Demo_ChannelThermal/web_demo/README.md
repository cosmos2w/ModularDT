# ChannelThermal Web Demo

Interactive forward and inverse dashboard for `1_Demo_ChannelThermal`.

## Run

From the demo root:

```bash
conda activate ModularDT
./start_demo.sh
```

Frontend: <http://127.0.0.1:5174>  
Backend: <http://127.0.0.1:8001>

The frontend launcher reuses the local React/Vite dependency cache from
`0_Demo_MultiCylinder/web_demo/frontend/node_modules` when available; otherwise
it falls back to `npm install`.

## Forward Mode

Forward mode edits a steady heated-module layout on a large channel-domain
canvas:

- module center positions in the channel
- per-module heat power, shown as both module color and `q` labels
- reference case for grid, material, and flow conditions

Heat power is a first-class design variable. The module heat table lets you set
individual powers, set all modules to one value, randomize within the model's
expected range, or normalize to a requested total heat. The web backend passes
these explicit module powers into the forward verifier and marks results with
`heat_power_source: web_per_module`.

Flow is exposed as one primary control: `reference`, `u_in override`, or
`Re override`. The reference setting uses the dataset flow condition that the
surrogate was trained around. Overrides are kept available for diagnostics, but
`Re` and `u_in` are not shown as competing primary controls because changing
both at once can imply an inconsistent physical condition.

The backend loads the configured ChannelThermal forward checkpoint, runs the
autonomous predicted-port path, renders static field images, exports raw arrays,
and reports thermal KPIs.

The result view shows all steady fields at once: temperature, `u`, `v`,
pressure, and vorticity. Organizer output is shown first as a domain overlay
with environment tokens and modules colored by dominant hyperedge. The raw
`A_mh` and `A_eh` matrix image remains available under advanced matrices.

## Inverse Mode

Inverse mode launches `src/evaluate_inverse.py` as an asynchronous job. Target
presets come from both `inverse_targets/*.json` and
`inverse_targets_v2/*.json`. The backend tags every preset as `legacy_kpi` or
`design_intent`; the frontend prefers the first v2 design-intent preset by
default.

The design-intent builder writes v2 target JSON with:

- `scenario`: module-count range and heat-load policy
- `geometry_constraints`: spacing, clearances, spans, keep-out/protected/
  preferred boxes
- `thermal_limits`: solid temperature, module spread, pressure drop, wall hot
  delta, and outlet hot delta limits
- `objective_weights`: safety, uniformity, pressure, outlet mixing, wall
  protection, plume avoidance, and coverage weights
- `field_preferences`: outlet/wall/plume preferences and coverage targets
- `structure_constraints`: optional quantitative or sketch placement intent
- `heat_loads`: explicit heat conditioning

Heat-load modes include exact per-module values, per-module ranges, shared
exact value, shared range, total heat only, from reference, and none. For
per-module v2 targets, candidate cards and the "Use in forward" action preserve
the target heat-to-slot list so the forward editor receives the intended heat
identity.

Placement conditioning modes are encoded as:

- `none`: `structure_constraints.enabled=false`
- `sketch`: low-resolution `sketch_maps` channels for preferred, keepout,
  protected, and reference-soft maps
- `quantitative`: spans, coverage, mean pair distance, centroid/tolerance,
  vertical-stack avoidance, and box constraints
- `reference family`: a structured intent mode recorded in preferences and
  controlled by structure strength/reference options

Legacy presets still populate the KPI table and use the old KPI target path.
Design-intent presets keep the KPI table collapsed as an advanced section.

Each candidate is forward-verified by the configured frozen ChannelThermal
checkpoint before it is shown. Candidate cards include a mini heat-colored
layout, score, count, total heat, key KPIs, artifacts, and a "Use in forward"
button. That button copies centers and heat powers into forward mode without
auto-running; a "Run now" button appears in forward mode for the copied design.

If a job completes but candidates cannot be parsed, the backend returns
`complete_with_no_candidates`, stdout/stderr tails, and
`/api/inverse/jobs/{job_id}/debug-files` to inspect discovered output files.
