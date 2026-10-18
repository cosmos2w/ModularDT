# HET-MF electro-thermal dataset pilot and formal readiness

The new independent Case_ElectroThermal generator produces conservative electrical potential, sheet current, Joule heat, and temperature-rise targets from 12 × 12 heterogeneous material controls and finite-area electrical contacts. A frozen pilot generated 286 distinct physical states and 320 accepted raw grid cases on verified `/data2`; no HONF model was trained and no formal dataset was generated. The tested primary grid is 96 × 96, the qualified GPU2 solver policy is FP64, and compact, standard, and expanded formal releases remain approval-bound proposals.

## Decision and scope

The 96² grid passes the prospective 96²-versus-192² bands on all twelve fixed representatives; 48² is not selected because the 48²-versus-192² potential discrepancy reaches 1.024% on R07, above the 1% field band. Four input-selected cases also have saved 384² references. The pilot contains twelve representative scenes, six controlled quartets, and a 256-state stratified throughput panel; the overlapping quartet bases are reused. Its 320 accepted raw files comprise 286 native-96 cases plus 34 refinement cases. The first FP32 raw export failed the post-storage thermal residual check, was retained as a diagnostic record, and was removed after verification; all accepted raw targets are FP64.

GPU2 was shared with an existing `flame` process under the user's explicit concurrent-use authorization. Its physical UUID was `GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39`; the other process stayed present with 3,456 MiB reported before and after the pilot, and the HET-MF process reserved at most 182,452,224 bytes during B32 timing. Shared-device measurements are not uncontended performance claims. Three repeats of the complete 256-state B32 panel passed; GPU FP32 and mixed precision failed representative true-residual qualification, so the formal script accepts only FP64. The script requires a separate exact formal approval record and exclusive GPU2 for any later release run.

## Measured numerical evidence

| Check | Saved pilot result | Decision |
|---|---:|---|
| 96² vs 192² maximum potential / temperature relative L2 | 0.202% / 0.121% | Both below 1% |
| 96² vs 192² maximum total Joule / terminal-current error | 0.530% / 0.215% | Below 8% / 2% |
| 96² vs 192² maximum reconstructed face-current discrepancy | 0.465% | Below 5% |
| 96² vs 192² peak temperature discrepancy | 0.076% | Below 3% |
| Maximum post-storage electrical / thermal local relative residual, 320 files | `6.40e-13` / `1.82e-11` | Below FP64 reference target |
| Maximum post-storage current / electrical-power / thermal-power imbalance | `4.71e-14` / `3.02e-14` / `2.84e-12` | Below `1e-5` balance bands |
| GPU2 FP64 representative field/reference discrepancy | maximum measured relative L2 `8.27e-11` across saved fields and terminal currents | Below GPU/CPU comparison bands |
| Four-case 4 × 4 contact quadrature pre-normalization mass error | maximum `7.19e-6`; normalized mass error `4.44e-16` | Below 0.2% band |

The finite-volume solve uses harmonic internal-face transmissibilities, insulating exterior faces, and unique-face resistive Joule losses partitioned by the two half-cell resistances. Current and energy balances are recalculated from stored arrays, while the 4 × 4 contact rule normalizes every finite-area footprint to unit discrete mass. The 2 × 2 rule reached a `1.52e-4` pre-normalization mass error on the four-case check, and the 4 × 4 footprint differed from 8 × 8 by at most `2.88e-5` relative L2; the 4 × 4 rule remains the pilot policy.

![R05 saved material fields, potential, current, temperature, heat sources, and terminal currents](../Dataset/figures/R05_n96.png)

R05 is a native-96 representative with four contacts and dimensionless targets; its saved current sum is approximately `-4.55e-15`, supplied electrical power is about `1.55`, and the temperature hotspot lies near the upper right contacts. This visually inspected page uses reconstructed cell current for display while conservation uses native face fluxes.

![Refinement, conservation, complete CPU time, and measured byte forecast](../Dataset/figures/pilot_numerical_cost_board.png)

The twelve-case maximum 96²-to-192² potential error is `0.00202`, the largest stored thermal residual is `1.82e-11`, the 256-state complete CPU mean is `0.0606` seconds per native-96 case, and the plotted storage includes both retained raw per-case sources and planned raw shards. The board is a compact view of the saved numerical diagnostics, not a continuum convergence proof.

## Predictor: gains, misses, next step

**Gain.** All 256 throughput states retain complete, input-only material and contact records with four environment families (104 correlated mosaics, 52 layered barriers, 52 inclusions, 48 corridors) and contact counts 2/3/4/6/8/12 represented 43/43/43/43/42/42 times. Median contact-loss share is `0.420` of total Joule heat, median spatial temperature coefficient of variation is `0.230`, and the twelve representatives show resolved nonuniform potential, heat, and temperature fields. The raw reference and five-channel processed targets are saved separately; generator provenance and solver labels are excluded from model inputs.

**Miss.** No predictor was trained or evaluated, and the exposed pilot cannot establish model generalization or formal sample sufficiency. Formal audit rows with 10 or 16 contacts and composite environments have no matched pilot throughput stratum. The dimensionless sheet model has no electrochemistry, temperature-dependent conductivity, calibrated physical units, or intra-contact temperature field.

**Next step.** If a formal release is approved, use the frozen input manifest, native 96² FP64 reference, and a separately bound GPU2/storage/time approval; preserve root-disjoint TRAIN/DEV/TEST and fit any later normalization on ordinary TRAIN only. A compact 2,304-state release is the lowest-cost first option; audit M10/M16 and composite challenge rows before any larger release claim.

## Organizer: gains, misses, next step

**Gain.** Four representative terminal conductance matrices satisfy `I=C u` with maximum row-sum error `1.32e-13`, symmetry error `3.33e-16`, and minimum symmetric eigenvalue no lower than `-1.51e-14` numerical roundoff. Controlled quartets contain electrical and thermal interactions: Q01 potential interaction is `0.00220` of endpoint scale, Q02 voltage-pair potential interaction is effectively zero (`4.40e-16`) while temperature interaction is `0.00195`, and Q05 electrical potential interaction is exactly zero under its thermal-only coefficient changes. The Q00 and Q05 selected interaction fields were checked again at 192²; their measured interaction amplitudes exceed the corresponding grid-difference floor by factors of roughly 257–1,166 where a nonzero ratio is defined.

**Miss.** Most four-solve interaction fractions are small (`~5e-7` to `2e-3`), so this pilot does not establish a strong sparse-interaction challenge. The terminal matrices are physical response operators with distributed nonzero entries; no learned graph, executed sparse support, executor speedup, or physical causal edge set was measured.

**Next step.** Keep the measured terminal matrices, quartet fields, and numerical floors as organizer evaluation targets. Before claiming useful sparse routing on a later model, measure actual selected and full-access execution, field fidelity, and intervention response on the same frozen cases.

![Physical terminal response matrices and controlled quartet interaction fractions](../Dataset/figures/physical_responses.png)

The native-96 R02/R05/R10 terminal matrices show signed, approximately symmetric physical coupling rather than thresholded graph labels; Q01/Q02/Q03 reach interaction fractions near `1e-3`, while Q04/Q05 are near `1e-6` or smaller. The log plot displays exact zeros at a labelled plotting floor and is not a learned-support map.

![Q02 voltage-pair four-solve potential and temperature interaction](../Dataset/figures/quartet_Q02_n96.png)

Q02 has potential interaction `4.40e-16` relative to endpoint scale and temperature interaction `0.00195`, consistent with voltage-linear electrical response and quadratic Joule heating at fixed geometry. All four endpoint fields and the difference-of-differences were rendered from saved native-96 arrays.

## Inverse: gains, misses, next step

**Gain.** The finite-contact controls and saved unit-voltage response matrices expose a deterministic forward map suitable for a future inverse task; terminal currents are solved outputs, including signed intermediate-voltage contact currents. Changing only thermal conductance or cooling leaves the electrical potential exactly unchanged in the saved R02/R08/R09 null comparison.

**Miss.** No inverse optimization, candidate design trail, constraint feasibility, or physical design validation was run. The generated contacts are parameterized sheet-interface controls, not validated battery or current-collector designs; sampling diversity alone cannot establish valid inverse designs.

**Next step.** Define explicit current, hotspot, manufacturability, and out-of-distribution constraints before running inverse experiments, and preserve every attempted sample and physical solve outcome.

![Thermal-only material changes and unchanged electrical parent potential](../Dataset/figures/R02_R08_R09_electrical_null_n96.png)

R02/R08/R09 alter thermal `kappa` and/or `beta` with the same electrical input, and the saved native-96 potential difference is exactly zero while temperature fields change. This is a controlled null for the one-way coupling law, not an optimized inverse design.

## Throughput and storage forecast

The complete CPU FP64 sparse-direct panel measured `0.0606` seconds per case on average (`0.0636` seconds p90) including preparation, solve, validation, FP64 HDF5 write, and post-storage audit. The batched GPU2 replay timed preparation, transfer, FP64 PCG, CPU finalization, and in-memory validation without duplicate output: B1 `0.740`, B4 `0.197`, B16 `0.0535`, and B32 `0.0290` seconds per case; B1/B4/B16 used 32 stratified cases with three repeats and B32 used the full 256 cases with three repeats. B32 electrical/thermal iteration p50/p90 were 582/622 and 599.5/627, respectively. The separate CPU torch replay measured B1/B4/B16 three full repeats and one B32 batch, then stopped to retain the plan's 4,000-evaluation ceiling.

Forecasts below join each exact input-only formal plan to measured native-96 pilot file sizes and CPU strata; the GPU column adds measured CPU non-solver write/audit overhead (`0.0176` seconds per case) to the shared-device B32 rate, then adds the four-case mechanical formal-shard packing rate (`0.0138` seconds per case), selected plotting, measured fixed QA, and prospective 5% mean or 10% conservative retry/time reserves. The GPU estimate can double-count some validation, and neither column is a promised formal runtime: process startup, plan construction, formal full-release audit/normalization, 192²/384² challenge groups, and unmeasured M10/M16/composite tails require separate approval limits. The two times are mean/conservative extrapolations, not a measured formal release.

| Profile | States; train environment/layout roots | CPU FP64 minutes mean / conservative | GPU2 shared FP64 minutes mean / conservative | Raw FP64 source + shard GiB | Processed shard GiB | Fixed QA GiB | Peak temporary GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compact | 2,304; 120/240 | 4.49 / 4.73 | 3.92 / 4.11 | 2.36 | 0.40 | 0.069 | 0.047 |
| Standard | 4,608; 240/480 | 8.41 / 8.89 | 7.27 / 7.66 | 4.72 | 0.80 | 0.069 | 0.047 |
| Expanded | 9,216; 480/960 | 15.95 / 16.91 | 13.68 / 14.45 | 9.45 | 1.61 | 0.069 | 0.047 |

The compact/standard/expanded selected-plot allowances are about 75/132/227 seconds and projected figure storage is roughly 0.010/0.018/0.032 GiB. A further 60 ordinary TRAIN environments correspond to 720 states and add about 0.015 CPU or 0.012 GPU-shared forecast hours before extra plots/QA. The 4-case raw HDF5 write/read compression comparison measured none at 2,390,608 bytes and 0.0056/0.0058 seconds, LZF at 2,226,378 bytes and 0.0393/0.0101 seconds, and gzip-1 at 2,086,858 bytes and 0.0512/0.0177 seconds; the retained source writer and formal shard test use LZF for a moderate size/latency balance. A four-case formal-shard mechanical test wrote 2,158,218 raw plus 749,672 processed bytes in 0.0552 seconds, verified accepted source hashes and array shapes, and reused the identical shard without rewriting.

## Formal gate, provenance, and retained figures

The compact, standard, and expanded input-only plan hashes are `cdffb87e9effdf5fcef331839e50b2dbe26dbadfc4f2ed5ed16ead4c127c1ae4`, `a94ddd9e0374fb9a971f4856944b2f8cc475749db6b2f797cddccbd3afa29e6b`, and `f7f25efd5194cdec7dcc16d94ada30c2d92027e4334a21928e5531935df2ae99`. Every profile passed `generate --dry-run` with zero physical solves and zero writes; a non-dry-run invocation without an approval record was rejected. The pilot plan hash is `e1cfcb37c073ecf95422c4479ac45d0598ebcd47e90b5dd46784b472679e56d9`, and its source file SHA-256 is `52839fca8593b496b349c1bd1c9bb2a62b8d066ec3bfd3f181717e5b9ab3b94b`. See [generation guide](HETMF_GENERATION_GUIDE.md) for the approval-bound command and [local pilot manifest](../Dataset/links/active_manifest.json) for the exact payload inventory.

The key PDF masters are [R05 fields](../Dataset/figures/R05_n96.pdf), [numerical and cost board](../Dataset/figures/pilot_numerical_cost_board.pdf), [physical responses](../Dataset/figures/physical_responses.pdf), [Q02 quartet](../Dataset/figures/quartet_Q02_n96.pdf), and [thermal null](../Dataset/figures/R02_R08_R09_electrical_null_n96.pdf); the remaining representative and quartet PDF masters are in the same ignored `/data2` figure directory. The selected small raster companions above exist only for direct Markdown embedding. All generated plans, arrays, diagnostics, logs, figures, and receipts remain under `/data2/wanglz/ModularDT/datasets/hetmf/pilots/pilot_20261017_v1`; `Dataset/` contains safe ignored browsing symlinks, and generated data is not committed to Git.

Round cleanup removed 75 verified disposable files totaling 16,657,593 logical bytes, plus 13 pytest-only symlinks: one failed FP32 raw export (294,474 bytes), 23 superseded/redundant figure exports (6,053,227 bytes), 17 temporary benchmark/test/shard files (9,926,134 bytes), and 34 ignored source bytecode files (383,758 bytes). Exact inode, link, size, and duplicate-file hash allowlists and deletion receipts remain under the ignored `receipts/` directory. The protected raw, processed, input, figure, and diagnostic inventories were unchanged across temporary cleanup; the final payload manifest lists 320 raw files (234,945,519 logical bytes), 286 processed files (59,477,068 bytes), 31 selected figure files (6,142,251 bytes), and the three input-only formal plans. The verified `/data2` ext4 mount UUID is `a47e4da3-3b56-40a7-baba-b675eeb0e999`; the closeout free-space snapshot was 3,380,965,335,040 bytes, above the 50 GiB minimum headroom, under the pilot's 15 GiB retained plus 5 GiB temporary hard budget.

The pilot used NumPy 2.4.3, SciPy 1.17.1, PyTorch 2.6.0+cu124, h5py 3.16.0, matplotlib 3.10.8, and PyYAML 6.0.3 in the ModularDT Conda environment. The isolated source branch started at reviewed `53cba42bc73661306e469c7905be9422b235d672`; Round9 and its GPU0/GPU1 jobs were not modified. The formal release remains ungenerated pending an explicit choice of profile, native grid and any high-resolution challenge groups, GPU2 scheduling, and approved time/storage budgets.

Verification closed with 33 passing case-local tests, an independent full post-storage audit accepting all 320 raw cases, inspected representative/quartet/response/null/numerical figures, three read-only formal dry runs, an expected rejection of unapproved formal execution, and a real four-case mechanical shard pack/reuse test whose duplicate shard payloads were removed after its receipt was saved. These checks demonstrate generator and pilot readiness under the stated physical model; they do not constitute a generated or validated formal release.
