# Thermal model development protocol

Use the input-stratified membership and original case order declared by the selected recipe. Existing TRAIN150 and exposed validation manifests are external scientific artifacts; preserve their exact identities rather than creating replacements from model errors. Full-population training uses the declared TRAIN600 membership. TRAIN-only normalization, source split separation and exact resume identity are mandatory.

An epoch visits the declared training membership once. Small segmented fits establish execution, learning-path health, checkpointing and resume; they do not establish mature architecture quality. The standing local budget is at most 500 segmented epochs per round, with only the epochs needed for qualification. A first selected full-data assessment is normally at most 100 epochs; later continuation requires the authorization described in [AGENTS.md](../../AGENTS.md) or explicit user instructions.

Compare original-member geometry and IDs, initialization, query sampling, objectives, schedules and case/epoch budgets. Keep aggregate accuracy, fieldwise/local errors, organizer use and complete runtime/memory separate. Do not interpret sparsity, support, a small directory or a short preliminary trajectory as physical causality, convergence or end-to-end savings.

Save complete histories, initial/current loss curves, latest state, one best-field selector and declared 100-epoch milestones. Refresh curves every ten epochs and at clean stops. Separate sampled training reductions, full exposed validation metrics and gradient diagnostics. The [run guide](Run_Artifacts_and_Monitoring.md) describes layouts and compatibility.

On the maintained workstation, canonical Thermal run aliases are `Trained_Results/ThermalChannel/HONF_Forward_Runs/<run_name>` and remain bound to DATA2 payloads. The root lift does not migrate those payloads. Other installations must configure their own external data and output locations and preserve existing run identities. Historical plans, reports and absolute machine recipes remain local rather than shipping as current launch instructions.
