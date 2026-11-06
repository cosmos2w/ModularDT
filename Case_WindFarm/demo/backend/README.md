# WindFarm demo backend

Start the local application from `Case_WindFarm/demo` after building the frontend:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/wanglz/miniconda3/envs/ModularDT/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

The backend reads the existing WindFarm compact metadata and memory-mapped native volume through `Dataset/links/wind_farm`. It loads `wind_r15_2_w2_train420_q8192_b24_full1000_v2/checkpoints/epoch_2500_model.pt` on CPU, using its saved model configuration and TRAIN-fitted transforms. The checkpoint has the same model tensors as the latest saved alias inspected on 2026-11-05. The dataset and checkpoint remain at their existing locations. `GET /api/meta` loads metadata without loading model weights; the first reconstruction loads the model.

`GET /api/cases/{id}` returns a target-free scene from the saved TRAIN or VALID populations. `POST /api/reconstruct` accepts its `case_id`, optional edited `turbines`, `domain`, and `wind`, and a `[nx, ny, nz]` resolution. Coordinates are in rotor diameters (80 m), velocities are in m/s, turbine hubs are fixed at 0.875D, and the represented altitude range is 0–6.25D. The saved model accepts 1–30 turbines, although its training cases had 6–30. Inflow speed is fixed to the training value of 9 m/s and direction is one of 270, 285 or 300 degrees. Coordinates belong to the direction-specific wind-aligned frame; changing the direction category does not rotate a layout. Edited layouts or support dimensions can be outside the training distribution and have no CFD reference.

For an unchanged case, the returned axes are sampled native cell centers, the flattened reference and prediction have the same x-major, z-fast, component-last order, and fieldwise errors use those exact points. `fluid_mask` is all true because those points are sampled from the native cell volume. For an edited case the grid spans the edited support and reference fields, errors, and CFD mask are null. TEST cases are excluded.

The organization response contains every E receiver position and all native `directory_rows` from the actual prepared **full-access** directory, retaining original M/E IDs and positive member weights. Anchor records additionally carry members for 12 evenly selected receiver rows used by the spatial display and its address selector; the Venn view can select from the complete directory. The stage names distinguish executed E←M and E←E relations. Directory weights are prepared access weights, not final query contributions, physical causal relations, or evidence of sparse-execution savings. This demo performs the checkpoint's full-read prediction and makes no speedup claim.
