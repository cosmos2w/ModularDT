# ModularDT Multi-Cylinder Web Demo

## 1. Purpose

This demo is an interactive dashboard for the ModularDT multi-cylinder emulator. It lets a user choose a checkpoint, edit cylinder layouts in physical coordinates, run deterministic full-cycle inference, inspect animated flow fields, and view organizer/hypergraph structure plus KPI curves.

The deterministic path is always available when its checkpoint exists. Generative inference is available for Stage-2 latent-flow checkpoints produced by `src/train_gen.py`; the included Stage-1 AE-only entry remains visible as a disabled placeholder.

The web demo also includes an inverse-design mode. It wraps `src/evaluate_inverse.py` in an asynchronous backend job, lets a user define partial KPI targets and geometry preferences, ranks sampled designs with the learned forward verifier, and can launch real PhiFlow validation for a selected candidate.

## 2. Architecture

```text
React/Vite frontend
  -> FastAPI backend
    -> model manifest and example designs
    -> inverse model manifest and inverse target presets
    -> existing 0_Demo_MultiCylinder/src/model.py
    -> existing 0_Demo_MultiCylinder/src/evaluate_inverse.py
    -> Saved_Model / Saved_Model_Gen checkpoints
    -> Saved_Model_Inverse checkpoints
    -> rendered PNG frames, inverse jobs, and KPI JSON under storage/cache
```

The backend does not depend on `packed_dataset.h5` for web inference. It builds structure tensors directly from the user-defined cylinders and evaluates the existing ModularDT neural field over a generated physical grid.

## 3. Folder Structure

```text
web_demo/
  README.md
  run_backend.sh
  run_frontend.sh
  run_demo.md
  backend/
    app.py
    model_registry.py
    deterministic_service.py
    generative_service.py
    inference_service.py
    design_validation.py
    kpi_service.py
    hypergraph_service.py
    render_service.py
    cache.py
    inverse_registry.py
    inverse_service.py
    requirements.txt
  frontend/
    package.json
    src/
      App.tsx
      api.ts
      components/
      styles.css
  storage/
    model_manifest.json
    inverse_model_manifest.json
    inverse_target_presets/
    example_designs.json
    cache/
      inverse_jobs/
```

## 4. Model Manifest

Edit `storage/model_manifest.json` to change model paths or add checkpoints. Paths support `~` and are resolved with `pathlib.Path(...).expanduser()`.

Each model entry includes:

- `id`: stable API id used by the frontend.
- `label`: human-readable label.
- `mode`: `deterministic` or `generative`.
- `run_dir`: saved run directory.
- `checkpoint_name`: checkpoint filename, such as `best_model.pt` or `latest_model.pt`.
- `config_name`: resolved config filename in the run directory.
- `enabled`: whether inference is allowed.
- `preload`: whether the backend tries to load it on startup.

For quick local startup, actual preload is opt-in even when the manifest contains `preload: true`. Set this before launching the backend if you want startup preloading:

```bash
export MODULARDT_WEB_DEMO_ENABLE_PRELOAD=1
```

Example deterministic path:

```text
~/Desktop/src/ModularDT/0_Demo_MultiCylinder/Saved_Model/Case0004_20260425_135831
```

Example generative Stage-1 path:

```text
~/Desktop/src/ModularDT/0_Demo_MultiCylinder/Saved_Model_Gen/Gen_Casegen002_Stage1_20260425_131600
```

These are examples only; edit the manifest for your local runs.

## 5. Backend Setup

Use the same Python environment that can import `torch` and the existing ModularDT model code. For example:

```bash
conda activate ModularDT
cd 0_Demo_MultiCylinder/web_demo/backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Or use the helper script:

```bash
cd 0_Demo_MultiCylinder/web_demo
./run_backend.sh
```

The script does not force a virtual environment; activate your preferred environment first.

## 6. Frontend Setup

```bash
cd 0_Demo_MultiCylinder/web_demo/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Or use:

```bash
cd 0_Demo_MultiCylinder/web_demo
./run_frontend.sh
```

The frontend expects the backend at `http://127.0.0.1:8000`. Override it with:

```bash
VITE_API_BASE=http://127.0.0.1:9000 npm run dev
```

## 7. Running The Demo

1. Start the backend.
2. Start the frontend.
3. Open `http://127.0.0.1:5173`.
4. Select the deterministic model.
5. Load an example design or drag cylinders in the domain editor.
6. Run inference.
7. Play the field animation, switch fields, toggle the hypergraph overlay, and inspect KPI curves.

## 8. Display Smoothing

The backend can render presentation PNGs at a higher display resolution than the raw model grid. For example, a `192 x 96` inferred field can be rendered as a `576 x 288` bicubic display frame.

This is visual only. The raw model arrays are unchanged and remain the source for KPI calculations, quantitative metrics, and exported `fields.npz` data. Result JSON reports this under `rendering.kpi_source = "raw_model_grid"`.

## 9. Phase-Bin Control

Panel A exposes the requested phase-bin count for generation, GIF/display frames, playback, and KPI synchronization. The selected model config controls the maximum phase bins and policy.

The backend resolves `max_phase_bins` in this order:

1. `max_phase_bins` in `storage/model_manifest.json`
2. `web_demo.max_phase_bins` in the resolved config
3. `validation.phase_bins_to_eval` in the resolved config
4. `evaluation.cycle.phase_bins` in the resolved config
5. fallback `36`

`default_phase_bins` is read from the manifest, then `web_demo.default_phase_bins`, then the resolved maximum. `phase_bin_policy` may be `cap` or `reject`. With `cap`, a request above the max remains valid and runs at the max. With `reject`, the validation response blocks inference.

Example manifest fields:

```json
{
  "default_phase_bins": 36,
  "max_phase_bins": 36,
  "phase_bin_policy": "cap"
}
```

## 10. KPI Panel

Panel D shows KPI target curves for all numeric KPI series returned by the backend. The main five appear first: Mean `|omega|`, Enstrophy, Max `|omega|`, Kinetic energy, and Pressure range. Additional field mean and max-absolute curves are available through the panel's show-more control.

The moving dot on every curve is synchronized with the flow animation and phase slider. This panel is intended as the future inverse-design target interface.

## 11. Computation Domain Annotations

Panel B annotates the interactive computation domain with the current Reynolds number, cylinder count, periodic-domain status, and domain size. The Re badge updates live while editing Panel A.

## 12. Example Deterministic Model Path

The default deterministic manifest entry points to:

```text
Saved_Model/Case0004_20260425_135831
```

Expected files:

```text
best_model.pt
resolved_train_config.json
```

The checkpoint should contain one of:

- `model_state_dict`
- `model`
- `state_dict`

## 13. Generative Stage-2 Hook

Stage 1 reconstructs residual fields but is not a full conditional sampler. The frontend shows Stage-1 entries as pending and disables inference with:

```text
Generative stage-2 checkpoint pending.
```

When a Stage-2 latent-flow checkpoint exists, add or replace a manifest entry in `storage/model_manifest.json`:

```json
{
  "id": "gen_stage2_casegen001",
  "label": "Generative Stage 2 Case gen001",
  "mode": "generative",
  "stage": 2,
  "run_dir": "~/Desktop/src/ModularDT/0_Demo_MultiCylinder/Saved_Model_Gen/Gen_Casegen001_Stage2_20260425_013132",
  "checkpoint_name": "best_model.pt",
  "config_name": "resolved_train_gen_config.json",
  "preload": false,
  "enabled": true
}
```

The backend hook in `backend/generative_service.py` loads the Stage-2 checkpoint fields saved by `src/train_gen.py`:

- `ae_state_dict`
- `velocity_state_dict`
- optional `ema_state_dict`
- `stats`
- `cond_ch`
- `global_cond_dim`
- `deterministic_checkpoint_path`

`POST /api/infer` then runs the frozen deterministic model as the conditioner, samples the latent rectified flow for each phase, renders the ensemble mean, and exports `generated_samples`, `pred_sample_std`, and `deterministic_field` in the job NPZ.

## 14. API Endpoints

- `GET /api/health`
- `GET /api/models`
- `GET /api/example-designs`
- `GET /api/models/{model_id}/config`
- `POST /api/design/validate`
- `POST /api/infer`
- `GET /api/jobs/{job_id}/result`
- `GET /api/jobs/{job_id}/frames/{field}/{frame_id}`
- `GET /api/jobs/{job_id}/export.npz`
- `GET /api/inverse/models`
- `GET /api/inverse/target-presets`
- `GET /api/inverse/kpis`
- `POST /api/inverse/run`
- `GET /api/inverse/jobs/{job_id}`
- `GET /api/inverse/jobs/{job_id}/result`
- `GET /api/inverse/jobs/{job_id}/candidates`
- `GET /api/inverse/jobs/{job_id}/candidates/{candidate_id}`
- `POST /api/inverse/jobs/{job_id}/candidates/{candidate_id}/quick-validate`
- `POST /api/inverse/jobs/{job_id}/candidates/{candidate_id}/simulation-validate`
- `GET /api/inverse/jobs/{job_id}/candidates/{candidate_id}/simulation-status`
- `GET /api/inverse/jobs/{job_id}/files/{relative_path}`

`/api/models` reports `available`, `enabled`, `mode`, `stage`, `checkpoint_exists`, `config_exists`, and `reason_unavailable`.

`/api/inverse/run` writes `request.json`, `target.json`, `command.json`, `status.json`, `stdout.log`, `result.json`, and `candidates.json` under `storage/cache/inverse_jobs/{job_id}`. Candidate quicklook images and evaluator outputs are exposed through the safe file endpoint.

## 15. Troubleshooting

- `No module named 'torch'`: activate the training/evaluation environment, such as `conda activate ModularDT`.
- `No module named 'uvicorn'`: install backend requirements.
- Frontend cannot reach backend: confirm FastAPI is running on `127.0.0.1:8000` or set `VITE_API_BASE`.
- Checkpoint missing: edit `storage/model_manifest.json`.
- Config missing: confirm `config_name` matches the saved resolved config file.
- Generative inference disabled: expected until a Stage-2 checkpoint is available.
- Very slow inference: reduce `phase_bins`, `resolution_nx`, and `resolution_ny` in the parameter panel.
- Inverse model not found: edit `storage/inverse_model_manifest.json` and confirm `run_dir`, `checkpoint_name`, and `config_name` exist.
- Forward verifier missing: confirm the inverse request's verifier id exists in `storage/model_manifest.json`.
- Target KPI has no active rows: enable at least one KPI row in inverse mode before launching.
- Simulation is slow: real PhiFlow validation is intentionally separate from inverse sampling; reduce simulation grid/phase settings or let the polling view continue.
- CUDA device selection: set `MODULARDT_WEB_DEMO_DEVICE=cuda:0` before launching the backend. The inverse wrapper uses the same setting and falls back to CPU when CUDA is unavailable.
- Generative verifier grid mismatch: Stage-2 generative checkpoints require their AE grid. Add `default_resolution_nx` and `default_resolution_ny` to the generative entry in `storage/model_manifest.json`; the backend also reads `num_x`/`num_y` from the checkpoint and normalizes inverse requests before launch.

## 16. How To Add New Checkpoints

1. Copy or locate the saved run directory.
2. Confirm it has a checkpoint and resolved config.
3. Add an entry to `storage/model_manifest.json`.
4. Restart or refresh the backend.
5. Open the frontend model selector and choose the new entry.

For deterministic checkpoints, the backend imports `build_model_from_config` from `src/model.py`, loads the checkpoint state dict, moves the model to the selected device, and runs in eval mode.

## 17. How This Connects To ModularDT

The demo follows the ModularDT inference path:

```text
Structure
  -> organized interaction state
  -> behavior manifold
  -> multi-field reconstruction
```

The user layout becomes structure tensors: Reynolds number, cylinder count, centers, and mask. The model organizer forms module/environment/hyperedge states. The behavior head maps organized structure to dynamic memory. The decoder reconstructs `u`, `v`, `p`, and `omega` across phase `tau`.

## 18. Inverse Design Mode

Open the `Inverse design` mode from the top toggle. The workflow is:

1. Select an inverse model from `storage/inverse_model_manifest.json`.
2. Select the learned forward verifier from the existing forward model manifest.
3. Load a target preset or enable individual KPI rows manually.
4. Set constraints and preferences: `Re`, cylinder count range, minimum center distance, and optional x/y span.
5. Set sampling and verification controls: `n_samples`, `verify_top_k`, `save_verified_top_k`, `phase_bins`, `nx`, `ny`, inverse steps, seed, and deterministic/generative verifier options.
6. Run inverse design and poll the async job status.
7. Inspect one ranked candidate at a time in the carousel. The UI shows the design, validity, KPI comparison, saved quicklook images/GIFs, and action buttons.
8. Use `Quick validate` to return cached learned-forward verification when available, or run the selected forward verifier for an unverified candidate.
9. Use `Simulation validate` to launch real PhiFlow validation for the selected candidate. Only one simulation job is allowed at a time by default.
10. Use `Use in forward` to move the candidate layout back into forward test mode.

Inverse target presets live in:

```text
web_demo/storage/inverse_target_presets/
```

The inverse model manifest format is:

```json
{
  "inverse_models": [
    {
      "id": "inv_case0010_demo",
      "label": "Inverse Case0010 demo",
      "run_dir": "~/Desktop/src/ModularDT/0_Demo_MultiCylinder/Saved_Model_Inverse/<RUN>",
      "checkpoint_name": "best_model.pt",
      "config_name": "resolved_train_inverse_config.json",
      "enabled": true,
      "preload": false,
      "default_forward_verifier_id": "det_case0010"
    }
  ]
}
```

Backend limits are controlled in `backend/settings.py` and can be overridden with environment variables:

```bash
export MODULARDT_WEB_DEMO_MAX_INVERSE_N_SAMPLES=512
export MODULARDT_WEB_DEMO_MAX_INVERSE_VERIFY_TOP_K=64
export MODULARDT_WEB_DEMO_MAX_INVERSE_SAVE_TOP_K=16
export MODULARDT_WEB_DEMO_MAX_SIMULATION_JOBS=1
```
