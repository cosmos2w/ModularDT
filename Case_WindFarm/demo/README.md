# HONF / Wind Lab

A local, sponsor-facing desktop 3D workbench for the saved full-data WindFarm HONF model. The browser renders actual joint velocity predictions with movable sections, integrated streamlines, editable turbine layouts and the model's prepared organization. Three standard scientific maps beside the 3D view show the hub-height XY plane and two orthogonal vertical sections, with physical coordinates, a shared velocity scale and errors from their displayed reference samples.

## Start

```bash
cd /home/wanglz/Desktop/src/ModularDT/Case_WindFarm/demo
./run.sh
```

Open [Wind Lab](http://127.0.0.1:8765). The launcher builds the frontend and starts the Python inference server on localhost. It uses the existing `ModularDT` Conda environment and Node 26 installation. Set `HONF_DEMO_PYTHON`, `HONF_DEMO_NODE` or `HONF_DEMO_PORT` to override those machine paths. Dependencies, builds and caches stay beneath `/data2/wanglz/ModularDT/windfarm_sponsor_demo_20261105/`; source and the dependency lockfile remain here. The installation/build/cache budget is 700 MB, with an estimated 850 MB peak and a 500 GB free-space floor on DATA2. No datasets or checkpoints are copied.

## Interaction

- Drag the scene to orbit, use the wheel to zoom, and use the camera rail for top view, scene fitting or fullscreen.
- Choose **Move turbine**, then drag a turbine; select one to edit its X/Y coordinates. Add or remove turbines with the quantity controls. Scene changes trigger a debounced reconstruction.
- Choose **XY**, **XZ** or **YZ** and move the section slider. Select velocity magnitude or an individual component in the color legend.
- Toggle **Flow field / Learned hypergraph** to reveal the prepared organization in the same 3D coordinate frame. Choose one E-address hyperedge or all representative rows. Original M/E nodes and positive directory memberships remain identifiable; link curves are schematic. The scientific section maps remain visible in either mode.
- Click a scientific map to select its matching 3D section. Toggle streamlines and slices in the lower toolbar. **Display settings** controls vertical exaggeration and sampling resolution.
- **Reset** restores the selected dataset scene and view.

## Physical scope

The demo is pinned to `wind_r15_2_w2_train420_q8192_b24_full1000_v2/checkpoints/epoch_2500_model.pt`, the most recently saved full-population endpoint when this demo was built, and exposes its run, saved epoch and membership through `/api/meta`. It loads the native dataset without copying it; selectable examples come from TRAIN/VALID and leave TEST targets locked. The model predicts the three velocity components jointly in m/s; scene coordinates use rotor diameter D = 80 m and hub height 0.875 D.

The saved model has three categorical directions, 270°, 285° and 300°, and a fixed 9 m/s reference inflow profile. Direction selects the trained category in the existing layout frame; it is not an automatic physical rotation of the wind farm. Arbitrary inflow speed is not an input to this checkpoint. Edited layouts and displayed extents are model explorations, with no new CFD reference or boundary-condition solve. Streamline animation follows the reconstructed steady vector field and does not imply a time-dependent forecast.

The field overlay displays fieldwise error when a native reference is available; edited scenes explicitly show that the reference error is unavailable. Organization uses actual prepared E-address directory memberships with original M/E identities. Nodes keep their physical coordinates, link curves are schematic, and link brightness uses a fixed logarithmic mapping of stored positive weights without discarding small positive members. Display selection and retained weights are exposed in the response; these relations do not establish physical causality or measured sparse execution savings.

## Development

```bash
# Start the backend with ./run.sh, then use a second terminal:
PATH=/home/wanglz/.nvm/versions/node/v26.12.1/bin:$PATH npm run dev
```

Vite proxies `/api` to the localhost inference server. Build with `npm run build`. The backend serves the compiled frontend, so the sponsor demonstration needs only one local URL. The server binds to localhost by default; this first demo does not publish the model, dataset or service externally.
