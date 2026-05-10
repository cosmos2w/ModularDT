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

Forward mode edits a steady heated-module layout:

- module center positions in the channel
- per-module heat power
- reference case for grid, material, and flow conditions

The backend loads the configured ChannelThermal forward checkpoint, runs the
autonomous predicted-port path, renders static field images, exports raw arrays,
and reports thermal KPIs.

## Inverse Mode

Inverse mode launches `src/evaluate_inverse.py` as an asynchronous job. Target
presets come from `inverse_targets/*.json`; each candidate is forward-verified
by the configured frozen ChannelThermal checkpoint before it is shown.
