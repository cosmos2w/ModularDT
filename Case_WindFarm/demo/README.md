# HONF / Wind Lab

A local, sponsor-facing desktop 3D workbench for the saved full-data WindFarm HONF model. The browser renders actual joint velocity predictions with movable sections, integrated streamlines, editable turbine layouts and the model's prepared organization. Three standard scientific maps beside the 3D view show the hub-height XY plane and two orthogonal vertical sections, with physical coordinates, a shared velocity scale and errors from their displayed reference samples.

## Start

```bash
cd /home/wanglz/Desktop/src/ModularDT/Case_WindFarm/demo
./run.sh
```

Open [Wind Lab](http://127.0.0.1:8765). The launcher builds the frontend and starts the Python inference server on localhost. It uses the existing `ModularDT` Conda environment and Node 26 installation. Set `HONF_DEMO_PYTHON`, `HONF_DEMO_NODE` or `HONF_DEMO_PORT` to override those machine paths. Dependencies, builds and caches stay beneath `/data2/wanglz/ModularDT/windfarm_sponsor_demo_20261105/`; source and the dependency lockfile remain here. The installation/build/cache budget is 700 MB, with an estimated 850 MB peak and a 500 GB free-space floor on DATA2. No datasets or checkpoints are copied.

## Open from VS Code Remote

Run this command in the active VS Code Remote integrated terminal, from any directory:

```bash
/home/wanglz/Desktop/src/ModularDT/Case_WindFarm/demo/open.sh
```

The script reuses a healthy server or starts one in `tmux` as `honf-wind-demo-8765`, waits for the metadata endpoint, and requests opening through VS Code's own remote browser helper. The server remains independent of the terminal. Use `--no-open` to print the address without opening a browser, or `--port 8766` to choose another port. To inspect a server started by this script, run `tmux attach -t honf-wind-demo-8765`.

If no browser tab appears, use **Ports → Forward a Port → 8765 → Open in Browser**, using the displayed **Forwarded Address**. VS Code may choose a different local port. Browser opening requires the current VS Code terminal's IPC connection; an ordinary SSH terminal or an old tmux terminal may lack it. The script prints instructions in that case and keeps the server on localhost. See [VS Code's port-forwarding documentation](https://code.visualstudio.com/docs/remote/ssh#_temporarily-forwarding-a-port).

## Interaction

- Drag the scene to orbit, use the wheel to zoom, and use the camera rail for top view, scene fitting or fullscreen.
- Open **Module design** for the distinct grid-based editor. Under **Turbine layout**, choose **Add** and click the floor, **Move** and drag a turbine, or **Delete** and click one. A selected turbine also exposes coordinates and a delete action. Under **Domain sizing**, drag the X/Y boundary handles or use the horizontal-extent sliders. Resizing keeps the domain center fixed, stays within 4–100 D, and keeps turbine centers at least 0.5 D from the boundary; 1–30 turbines are supported. Scene changes trigger a debounced reconstruction without blocking the editor.
- Choose a quantity in **Physical field**: magnitude, downstream, crosswind, vertical velocity or downstream deficit. Each uses a dedicated sequential or zero-centered diverging palette. Choose **XY**, **XZ** or **YZ** and move the section slider.
- **Wake focus** uses the 2nd–98th percentiles of actual grid samples in the union of turbine neighborhoods (−0.5 to 8 D downstream, ±1.5 D laterally, ±0.65 D from hub height). **Full range** uses the full returned grid. Signed quantities use symmetric limits around zero; colors saturate at the displayed endpoints without altering values or errors. All views share these limits. Streamlines are seeded around the real rotors and integrated through the predicted vector field.
- **Learned hypergraph** opens **Spatial links** by default, using the physical coordinates and one E-address row or the existing representative rows. **Focused links** hides weak connections while preserving at least 95% of each row’s positive access weight, every positive M membership and positive own-E access; the inspector reports displayed memberships and retained weight. **All links** restores the complete positive support. **Venn sets** retains its exact native-hyperedge intersections without this display filter. The underlying model and directory are unchanged.
- Drag the scientific sidebar’s left divider to enlarge the maps up to half the page width; dragging farther opens the full map workspace. Click a map to enlarge it, use the wheel to zoom at the pointer and drag to pan. **Show in 3D** selects the corresponding plane. **Back** or Escape restores the default sidebar layout and map views. Maps redraw from the physical samples at the current canvas size and crop; this adds display resolution, not new model or CFD samples. Toggle streamlines and slices in the lower toolbar. **Display settings** controls vertical exaggeration and model sampling resolution.
- **Reset** restores the selected dataset scene and view.

## Physical scope

The demo is pinned to `wind_r15_2_w2_train420_q8192_b24_full1000_v2/checkpoints/epoch_2500_model.pt`, the most recently saved full-population endpoint when this demo was built, and exposes its run, saved epoch and membership through `/api/meta`. It loads the native dataset without copying it; selectable examples come from TRAIN/VALID and leave TEST targets locked. The model predicts the three velocity components jointly in m/s; scene coordinates use rotor diameter D = 80 m and hub height 0.875 D.

The saved model has three categorical directions, 270°, 285° and 300°, and a fixed 9 m/s reference inflow profile. Direction selects the trained category in the existing layout frame; it is not an automatic physical rotation of the wind farm. Arbitrary inflow speed is not an input to this checkpoint. Edited layouts and displayed extents are model explorations, with no new CFD reference or boundary-condition solve. Streamline animation follows the reconstructed steady vector field and does not imply a time-dependent forecast.

Downstream deficit means the checkpoint's TRAIN-fitted vertical Ux profile minus reconstructed Ux at the same height, not a uniform 9 m/s subtraction. The three transparent scientific maps retain physical aspect ratios, light contour lines, separate colorbars and RMSE computed from their paired displayed samples in the current crop. The 3D field overlay displays selected-quantity RMSE on the displayed slice or full sampled volume. Edited scenes explicitly mark reference error unavailable.

Organization uses all actual prepared E-address rows with original M/E identities. Venn selection compares non-self, L2-normalized membership profiles, refines medoids and adds a farthest profile until the 90th-percentile residual is at most 0.5 or the explicit three-set display cap is reached. A chosen address pins the first row. This is a representative display selection, not a discovered group count. Every positive M/E membership of the selected rows, including own-E membership, is preserved; no E threshold is applied. Every original member is placed in its exact set intersection or outside the union. Venn coordinates, sphere areas and distances are abstract; turbine dragging and the physical top-view preset are disabled there, and Fit restores a clear face-on view. Spatial links retain physical coordinates. M-to-E arrows denote positive prepared-directory access, not separately captured contextual exchange gates; their curvature is schematic. Spatial link brightness uses stored positive weights, and the response retains selection/coverage details. These relations do not establish physical causality or measured sparse execution savings.

## Development

```bash
# Start the backend with ./run.sh, then use a second terminal:
PATH=/home/wanglz/.nvm/versions/node/v26.12.1/bin:$PATH npm run dev
```

Vite proxies `/api` to the localhost inference server. Build with `npm run build`. The backend serves the compiled frontend, so the sponsor demonstration needs only one local URL. The server binds to localhost by default; this first demo does not publish the model, dataset or service externally.
