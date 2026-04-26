# Demo Presentation Script

Run all commands from the repository root unless a step says otherwise.

## 1. Start the backend

Open a terminal and activate the Python environment that has `torch` available:

```bash
conda activate ModularDT
```

Start the backend with the helper script:

```bash
cd 0_Demo_MultiCylinder/web_demo
./run_backend.sh
```

Leave this terminal running. The backend serves API endpoints only, so opening
`http://127.0.0.1:8000/` in a browser may show:

```json
{"detail":"Not Found"}
```

That is expected. The dashboard page is served by the frontend in Step 2.

Equivalent manual commands:

```bash
cd 0_Demo_MultiCylinder/web_demo/backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## 2. Start the frontend

Open a second terminal and run:

```bash
cd 0_Demo_MultiCylinder/web_demo
./run_frontend.sh
```

Equivalent manual commands:

```bash
cd 0_Demo_MultiCylinder/web_demo/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

If the backend is running on a different port, set the frontend API base:

```bash
VITE_API_BASE=http://127.0.0.1:9000 npm run dev -- --host 127.0.0.1 --port 5173
```

## 3. Open the dashboard

Open the app in a browser:

```bash
xdg-open http://127.0.0.1:5173
```

If `xdg-open` is not available, manually open:

```text
http://127.0.0.1:5173
```

## 4. Select the deterministic model

In the left panel, select:

```text
Deterministic Case0004
```

No terminal command is needed for this step.

## 5. Choose or edit a design

In the dashboard, pick an example design or drag cylinders in the computation domain.

No terminal command is needed for this step.

## 6. Run inference

Click the inference/run button in the dashboard and wait for the result to load.

Optional backend health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## 7. Present the deterministic result

In the dashboard:

1. Show the `omega` animation.
2. Scrub the phase slider.
3. Switch to `u`, `v`, and `p` without rerunning inference.
4. Toggle the hypergraph overlay to reveal organizer tokens, hyperedge sources, wake markers, and links.
5. Point to the KPI cards and the phase-synced moving dot.

No terminal command is needed for this step.

## 8. Optional generative setup

The current checked-in generative entry is Stage 1 AE only. To enable sampling, edit:

```bash
nano 0_Demo_MultiCylinder/web_demo/storage/model_manifest.json
```

Add or update a Stage-2 latent-flow entry with:

```json
{
  "mode": "generative",
  "stage": 2,
  "enabled": true
}
```

Then restart the backend:

```bash
cd 0_Demo_MultiCylinder/web_demo
./run_backend.sh
```

## 9. Close the demo

Close by describing the roadmap:

```text
generative uncertainty, inverse design, and target-driven layout optimization
```
