# Attention and content-hypergraph forward family

The opt-in `honf_content_hypergraph` family uses typed, measure-aware attention to retain native module and environment states, then decodes either those source states directly (A-D) or six content-bearing edges (A-H6). Thermal and Wind use the same core implementation and separate trained weights. The dataset adapters own legal features, coordinates, TRAIN transforms, native observations and physical units. Historical families retain their existing loaders and checkpoints.

## Architecture and physical capability

The initial core has width128, four attention heads, two simultaneous MM/ME/EM blocks, a width64 shared reader and one compact-support local reader. A-H6 uses six edges with full receiver softmax access. Its edge-to-query work replaces dense far-source nonlinear reading. The local reader evaluates every eligible pair, including overlapping neighborhoods: radius3 times module radius for Thermal and1.5 rotor diameters for Wind. Incidence is normalized over edges; actual pooling weights additionally use the donor measure exactly once. Physical IDs are correspondence metadata and never learned ordinal embeddings. A-D omits the final EM/environment-FFN update because simultaneous final MM/ME reads use the preceding E state and no A-D consumer uses the final E output; its source states exactly match the full block with the common initialized tensors. A-H6 retains that E update for edge pooling, and both execution policies are recorded explicitly.

Thermal has four signed response channels per edge. Its source-indexed coefficient bank is `[batch,edge,channel,source]`; temperature applies `Phi @ (C @ h)` plus the selected local source coefficients. Actual heating enters after the nonlinear geometry computation, so zero heating gives zero physical temperature and flow is computationally heat-independent. The adapter transforms flow before native curl and derives fluid, surface, outside, material and flux-proxy observations from one temperature field. Thermal quantities remain packed-dataset native units. Wind has a nonlinear three-component residual head with the stored TRAIN profile and residual transform in m/s; it offers ordinary input derivatives without an exact affine finite-control capability.

Wind uses the maintained native residual conversion `U = profile(z) + residual_normalized * safe_std * u_ref_mps`. The stored absolute-velocity mean remains provenance metadata and is not added after the profile. Adapter state schema v2 records this law explicitly and rejects the invalidated v1 conversion state. Earlier round checkpoints that added the absolute mean are retained as invalidated evidence; they cannot silently load with corrected semantics.

Prepared states belong to one model state, scene, source inventory, normalizer and coordinate frame. Parameter/buffer updates and geometry/identity changes invalidate reuse. The full smooth function is the default for geometry derivatives. Ordinary and precise heat application contract the factors without exporting a dense source kernel; dense K export is an explicit operation with its own output cost. Precise application casts retained factors and reductions to FP64 and does not claim FP64 neural inference.

## Numerical contract

Freeze new-family software assertions before native qualification: FP64 reference algebra uses `atol=1e-9, rtol=1e-7`; FP32 primary outputs use `atol=3e-6` in TRAIN-scaled spaces and `rtol=3e-5`; scaled gradient comparisons use `atol=1e-5, rtol=1e-4`. Physical absolute tolerances multiply by the corresponding stored TRAIN scale. Derived q/curl comparisons propagate the primary error budget through the actual native linear observation: use the absolute row sum of its interpolation/difference coefficients, including conductivity and spacing. These are arithmetic checks, not physical accuracy requirements or revised historical tolerances. Preserve a failed optimized result and retain the qualified reference path.

## Development and exact resume

Use the ModularDT environment from `HONF_Proj`. Both new arms use the fixed Thermal TRAIN150/DEV22 or Wind TRAIN72/DEV24 membership, the same TRAIN transforms, seed, effective batch and `case_epoch_v1` query streams. Effective batches are48 Thermal cases and24 Wind direction rows. Primary counts are1024 fluid queries plus native auxiliaries in Thermal and4096 total queries over five roles in Wind. The LR schedule warms for20 epochs, holds to1000 and decays to3e-6 at2500; stopping at100 does not replace that schedule.

The measured primary packing uses24 Thermal cases and12 Wind rows per microbatch. Both arms of a dataset keep the same packing. Wind recipes declare32 GiB of host catalogue capacity for development and128 GiB for the conditional full-data follow-up; these are RAM limits, with zero new persistent-cache bytes. The provider reuses the same checksummed geometry-only store read-only, retains exact loaded catalogues within that bound, and builds cold misses only in RAM. The helper retains its2 GiB default for callers without an explicit capacity. Query draws, measures and numerical precision are independent of this execution choice.

```bash
conda activate ModularDT
CUDA_VISIBLE_DEVICES='' python tools/content_hypergraph_train.py \
  --recipe src/config_core/forward/content_hypergraph/thermal_a-h6_development.json \
  --device cpu --dry-run
CUDA_VISIBLE_DEVICES=GPU-233fcd85-5c6a-6212-3f44-655252afec70 \
  python tools/content_hypergraph_train.py \
  --recipe src/config_core/forward/content_hypergraph/thermal_a-h6_development.json \
  --device cuda:0 --output /absolute/owned/new_run --stop-after 100 --yes
CUDA_VISIBLE_DEVICES=GPU-233fcd85-5c6a-6212-3f44-655252afec70 \
  python tools/content_hypergraph_train.py \
  --recipe src/config_core/forward/content_hypergraph/thermal_a-h6_development.json \
  --device cuda:0 --output /absolute/owned/new_run --stop-after 500 \
  --resume /absolute/owned/new_run/checkpoints/latest_model.pt --yes
```

Use the Wind recipe and physical GPU1 UUID `GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e` for the other dataset. Replace `a-h6` by `a-d` for the matched direct control. Run a pair to100, review its physical learning and measured forecast, then continue healthy comparisons at500/1000 and within budget to2500. Save every100-epoch milestone plus latest and best-field aliases. A clean-stop request uses the maintained TrainingEngine `CLEAN_STOP_REQUEST.json` interface and saves an epoch-safe state. The engine rejects rewinds or changed data/schedule identities on resume.

Dry-run constructs and describes the model/data contract on CPU with zero optimizer updates and no run allocation. Disposable learnability and packing probes are separate measured operations; they never become scientific initial weights. Set `PYTHONDONTWRITEBYTECODE=1` and route pytest temporary payloads into the round-owned disposable directory.

## Self-contained loading

```python
from honf_runtime.content_hypergraph import load_content_hypergraph_checkpoint

model, adapter, checkpoint = load_content_hypergraph_checkpoint(
    "/absolute/run/checkpoints/best_by_field_mse_model.pt", device="cpu"
)
```

This loader reconstructs the stored core and native transforms without opening source datasets, refitting normalization or loading a frozen learned partner. It checks workflow, family, recipe checksum, constructor metadata and the exact tensor inventory. The Thermal public loader also dispatches this workflow to the same strict implementation.

## Conditional full-TRAIN follow-up

The separate `*_a-h6_full1000.json` recipes initialize fresh weights, moments and full-TRAIN transforms. Their prospective LR schedule holds through2000 and ends at5000, while `approved_stop_after=1000` imposes the actual execution boundary. The launcher refuses any stop above1000 and requires a dataset-specific measured qualification receipt. Development checkpoint weights or moments never initialize this follow-up.

Qualification requires common development age at least1000, lawful native behavior, measured edge-content use and nontrivial TRAIN/DEV fields; A-H6 selection and priority-field ratios at most1.25 versus A-D; each resolved component ratio at most2; complete high-M Q8192 inference at most2 times A-D; and a forecast that fits the remaining campaign allowance and evaluation reserve. A-D alone cannot qualify as an organized model. Failure of a modest research guardrail blocks that dataset's automatic follow-up and remains reported separately from mature-reference distance.

```bash
CUDA_VISIBLE_DEVICES='' python tools/content_hypergraph_train.py \
  --recipe src/config_core/forward/content_hypergraph/thermal_a-h6_full1000.json \
  --stop-after 1000 --device cpu --dry-run
# Launch only after the measured qualification and available-budget review.
CUDA_VISIBLE_DEVICES=GPU-233fcd85-5c6a-6212-3f44-655252afec70 \
  python tools/content_hypergraph_train.py \
  --recipe src/config_core/forward/content_hypergraph/thermal_a-h6_full1000.json \
  --stop-after 1000 --device cuda:0 --output /absolute/owned/fresh_full_run \
  --qualification /absolute/measured_qualification.json --yes
```

Continuation beyond1000 is manual future work requiring a new authorization and an explicit authorization-aware recipe/schema; the current full1000 launcher rejects it. There is no queued5000 command. WindTEST stays locked, unexposed Thermal targets stay locked, and solver usage remains326/326 with zero new attempts.

## Evidence and storage

Use full native endpoint metrics, fixed-weight TRAIN/DEV comparisons, source-to-edge pooling, receiver access and same-weight interventions to assess learning and actual content use. An arrow or nonzero gradient establishes a computational path; it does not identify physical causality. Count complete prediction, native extraction/CPU return, optimizer-boundary work and memory, requested versus unique union queries, and actual local/edge rows. Fixed-B density probes atQ1024/4096/8192 and the declared larger stress point measure memory scaling; reduced work does not itself establish faster execution.

Record a hard disk-byte ceiling and reserve before materialization. Reuse verified raw geometry catalogues by exact identity; learned prepared states expire at a weight update. Preserve source data, historical runs, declared milestones, native evidence and compact provenance. Only round-owned disposable caches and superseded selected-figure payloads qualify for routine cleanup. Generated arrays, figures, checkpoints and one-time renderers remain ignored; push durable code/tests/configuration/guides/report after auditing the entire outgoing history.
