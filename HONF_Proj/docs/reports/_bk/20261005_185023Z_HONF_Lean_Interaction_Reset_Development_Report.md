# Lean interaction reset: matched Global-C continuation

Completed measured development experiment, 5 October 2026. We made Global-C's redundant one-group reader cheaper and tested whether a new source-group control residual earns useful receiver-dependent organization. Both arms continued the same selected-data Global-C e500 for 500 additional epochs. Tensor-H improves mean fluid/surface/material temperature RMSE by 19.95%/12.05%/11.26% against matched G-fast, while pressure worsens 21.26% and all four flow-channel means worsen. It costs 26.63% more per complete epoch. Receiver-mean access is effectively interchangeable with learned receiver access on the fixed-four intervention panel, so meaningful receiver-specific utility was not demonstrated.

**Decision:** retain opt-in G-fast as the broad-field development base, with the disclosed strict FP32 VJP misses; preserve Tensor-H as a thermal-focused research reference. Do not promote the residual hypothesis to a general interaction or inverse solution, force another sparsity repair, or restart formal training automatically. The formal runs were already interrupted before this task; a graceful pause could not be executed. All new experimental GPU work finished within 3.43331 aggregate GPU-hours of the 12-hour allowance. Evidence root: `/data/wanglz/ModularDT/thermal_development/lean_interaction_20261005`.

| Result area | Gains | Misses and limits | Next decision |
|---|---|---|---|
| Predictor | Four of eight core means improve; all eight improve against the inherited parent after extra training | u/v/p/omega worsen against the matched continuation; q-proxy maximum worsens; one seed and exposed development validation | Keep G-fast for broad-field work; retain the measured thermal tradeoff |
| Organizer | One once-wrapper tensor plan; exact admission/donor support, phase sharing and honest exports; modest zero-residual thermal reliance | Receiver-mean access changes core metrics by at most 0.003%; physical fine rows and dense control arithmetic remain unchanged | No useful receiver-specific organization established; no forced repair campaign |
| Inverse | Finite model input gradients with two-step local FD agreement; exposed TRAIN0348 thermal responses beat zero-change | Known flow-null leakage remains; two cross-arm sensitivity signs disagree; held positive response truth absent | No inverse training, search or designs; no inverse readiness claim |

The plan's three scientific questions have distinct outcomes: **A, added learning value:** a conditional thermal gain with flow regressions and extra parameters. **B, representation faithfulness:** input-conditioned control support and declared global/dense paths are faithfully exported, with no physical-causality or hardware-sparsity inference. **C, response transfer:** not established; measured exposed responses and model self-consistency do not replace held physical-response evidence.

This experiment asks whether receiver-dependent source-group controls add useful information after ordinary global calibration has an inexpensive execution path. G-fast specializes the existing Global-C one-group reader. Tensor-H adds one input-conditioned, source-anchored residual organizer while preserving every original fine physical source. Both start from the same development Global-C epoch-500 checkpoint, with identical selected cases, inherited common weights and optimizer state, losses, query sampling, and absolute schedule. This is warm-start adaptation, with 500 inherited epochs; it is not a comparison from fresh initialization.

## Formal histories and measured epoch cost

The requested graceful checkpoint-and-pause could not be performed: Runs 3501/3502 had already exited with `KeyboardInterrupt` at 2026-10-05 16:20:20 UTC, before this task began. Ownership, original commands, GPU assignments, process records, configurations, logs, and durable states were inspected. No signal was sent, no unrelated process was stopped, and no formal code was hot-edited. Their existing failed manifests were preserved. Neither formal run was restarted or relabelled as gracefully paused.

| Original run | Completed work on arrival | Durable recovery | Conditional remaining cost |
|---|---|---|---|
| Tree-C Run3501, GPU 1 | 97 complete training/validation epochs | No checkpoint exists; the model/optimizer state of those epochs cannot be recovered from scalar logs | 211.81 h for a hypothetical remaining 4,903 epochs at the last-20 mean; **not a resumable ETA**, and no post-e101 cost was observed |
| Global-C Run3502, GPU 2 | 431 complete training/validation epochs; training 432 completed, validation 432 interrupted | Model, optimizer, RNG, normalizers and campaign state at e400, 5,200 optimizer updates | 45.18 h for 4,600 epochs from durable e400; 44.87 h from logged full e431 would incorrectly assume its unsaved state was available |

Both resolved formal recipes used Q1024, microbatch 8/effective 48, receiver chunks 128, FP32 and activation checkpointing; Tree used `local_context_shadow`. The 600-case training epoch has 13 optimizer updates and 614,400 primary fluid queries, with 90 validation cases. Parameter count alone does not characterize that executed work.

| Measured formal interval | Training mean (s) | Validation mean (s) | Complete mean (s) |
|---|---:|---:|---:|
| Tree e1–25 | 147.3747 | 8.2996 | 155.6744 |
| Tree e26–97 | 147.2011 | 8.3720 | 155.5731 |
| Global e1–25 | 30.6499 | 1.9924 | 32.6424 |
| Global e26–100 | 30.6379 | 1.9972 | 32.6351 |
| Global e101–431 | 33.4341 | 2.0798 | 35.5139 |

Global's post-e101 response work totals 163.2825 s over 331 epochs, approximately 0.4933 s per epoch, **already included in training time**. Tree never reached that transition. Peak allocated GPU memory was 7,810.0 MiB Tree and 6,334.8 MiB Global. Unallocated launch elapsed time was 27.277 s Tree and 101.750 s Global; it includes startup, checkpoint/plot activity and interrupted work and cannot be disaggregated retrospectively. The conditional forecasts assume unchanged schedule and contention; they are not confidence intervals.

Global's resumable `latest_model.pt` SHA256 is `20400d02316941cbe436f7bf7a1bb8132b6d1504569e12962391068934d3dd39`. Its tensors agree with the e400 milestone. The following manual recipe preserves the original formal identity and is **unexecuted**:

```bash
cd /home/wanglz/Desktop/src/ModularDT/HONF_Proj
CUDA_VISIBLE_DEVICES=2 /home/wanglz/miniconda3/envs/ModularDT/bin/python train.py \
  --config src/config_core/forward/thermal_native_context/global-c_full5000.json \
  --resume-checkpoint /data/wanglz/ModularDT/thermal_formal/HONF_Forward_Runs/ThermalChannel/HONF_Forward_Runs/Run_3502_20261005_080820_thermal_native_context_full5000_global-c_v1/latest_model.pt \
  --device cuda:0 --yes
```

There is no honest Tree resume recipe without recovering a separately saved state or explicitly choosing a fresh run. The existing trainer has no supported signal-triggered save; interrupting it is not a resumable pause mechanism. This round did not add or test a generic pause mechanism.

## Cost diagnosis and Global specialization

G-fast is opt-in under `native_context_global_control_honf`. Non-attention case gains move after the native source sum and before any biased output projection. QE's source-constant score cancels in softmax; its head gain moves after the completed head context and before attention's output affine. Masks, MM self-exclusion, source measures, denominators, native keys/values, coarse/local paths, nonlinear updates and phase-current global controls remain intact. Historical defaults and state keys remain compatible.

Corrected profiler-off native samples at saved development weights, physical GPU 1, Q1024, microbatch 8/effective 48, three warmed boundaries per mode. These use the maintained native physical FIELD loss/backward, saved AdamW, native-denominator accumulation and work telemetry, with eight deterministic-query cases repeated six times for a boundary. The scheduled TRAIN0348 response callback is omitted (`response_callback=None`), so these are representative physical updates, **not complete scheduled-objective epochs**. The actual development epochs include the callback and establish complete-objective cost. Tree uses its own historical learned checkpoint, so its timing comparison measures execution at the same panel, not identity of learned functions:

| Panel | Global-C update (s) | G-fast update (s) | Tree-C update (s) | G-fast saving | Global / G-fast hard inference (s) |
|---|---:|---:|---:|---:|---:|
| Eight M1 cases | 1.563383 | 1.528314 | 8.170850 | 2.24% | 0.118200 / 0.110908 |
| Eight M12 cases | 3.040110 | 2.884885 | 14.100941 | 5.11% | 0.213299 / 0.194046 |

High-M B8 disjoint forward/backward/loss medians are Tree 1.888966/0.353737/0.001616 s versus G-fast 0.220435/0.221647/0.001188 s. Instrumented host scopes separately identify Tree's three largest sibling forward entries: `organizer.prepare`, 6 calls/1.352934 s; `TypedHypergraphField._numerical_access`, 110 calls/0.110888 s; and `SharedInterfaceContext.read_local`, 23 calls/0.102921 s. Inside prepare, 144 `AdaptiveReceiverHypergraph._index` calls take 0.927149 s, including 144 `CaseLocalReceiverTree.build` calls/0.653177 s, including canonical indexing/0.139478 s. **Nested scopes are not added.** These host dispatch/wait scopes are not isolated CUDA kernel times.

The observed 144 builds for eight cases verify 18 builds per case and the static full-training-epoch implication of `600 × 3 phases × 3 indexes × 2 hard/soft = 10,800` builds. Global prepare has three calls/0.023572 s; G-fast three/0.024370 s. Numerical-access host scopes fall from Global's 55 calls/0.020760 s to G-fast's 55/0.004527 s. Fine-MLP CPU dispatch remains about 0.0171 s for Global and G-fast; this is not its device execution time.

![Measured formal and development timing, memory, parameter and fine-row costs](../../../diagnostics/generated/lean_reset_20261005/figures/01_cost.png)

Figure 1. Saved formal logs give mean complete epochs 155.60 s Tree and 35.51 s post-e101 Global. Actual additional 500 epochs average 9.79 s G-fast and 12.40 s Tensor-H; peak training allocation is 5.19/6.34 GiB. The representative Q1024 FIELD-update benchmark excludes its scheduled response callback; the development epoch includes it. The host scopes and nested build counts explain execution structure, but no corrected-policy CUDA kernel attribution is available. The extra 120,351 Tensor parameters do not reduce fine-value rows. [PDF master](../../../diagnostics/generated/lean_reset_20261005/figures/01_cost.pdf).

All three high-M benchmark modes execute **4,918,656 padded fine MLP input rows in 55 invocations**: MM 3,456/3, ME 55,296/3, EM 55,296/3, QM 282,624/23 and QE 4,521,984/23. Low-M G-fast uses 1,837,336 rows/31 calls: MM 24/3, ME 4,608/3, EM 4,608/3, QM 9,472/11 and QE 1,818,624/11. The native low/high batches differ in padded module extent and physical output work, so their cost difference is not solely an organizer effect. The specialization removes redundant controls, not fine-source work.

An initial disposable benchmark selected the wrong default whole-wrapper shadow policy. Its training numbers were withdrawn and replaced with configured profiler-off measurements above. The three short profiler captures were from that initial policy, so they do not establish the configured training CUDA critical path. The profiler capture limit was not expanded to repair this attribution. Both attempts and all disposable updates remain charged to the resource ledger; no benchmark checkpoint is a scientific child state.

Native output and actual AdamW-step checks pass for G-fast at low/high M. The high-M strict VJP comparison has five misses at unchanged rtol2e-5/atol1e-6: maximum errors in encoder weights are 1.24e-5 and 6.68e-6, port-head weights 1.70e-6 and 1.22e-6, and heat input 5.81e-6. Their small FP32 ordering differences are disclosed rather than converted to a tolerance pass. The query-input comparison passes. This supports bounded development execution, not an automatically approved formal resume.

## Scientific changes and comparison identity

Tensor-H uses shared descriptors for each present module anchor plus the positive-measure environmental centroid, with no slot-ID embedding. Its padded capacity is Mmax+1 and its valid proposal count M+1; neither is learned K. One fresh input-only plan constructs admission, group keys and measure-normalized M/E donor densities per complete wrapper and is explicitly carried through P0/P1/P2. Phase-current projected source content is pooled with batched matrix multiplication; shared small controls receive ordinary task autograd. Final gamma projections initialize exactly zero.

The directed donor prior uses length 0.25 in coordinates divided by the native adapter scale, corresponding to 3×1.5 for the 12×6 frame. It is a modelling prior, not measured physical causality. Additional 1–100 use soft admission and measure-softmax density, 101–200 blend linearly to sparse projection, and 201–500 use exact sparse admission/density. A positive soft/blended proposal is not thresholded into a claimed exact sparse K. Density integrates against physical source measure; donor probability is measure×density.

Receiver keys depend on native coordinates, role and prescribed global context. Bounded receiver logits give positive access to every admitted group at every eligible receiver, so query participation generally has Kq=case K; differing access weights do not constitute independently sparse query subgraphs. Their bounded logits weight admission. Residual source controls subtract an eligible, target-free reference mean before tanh: module/environment references for their native routes and eight measure-weighted role-centroid slots from the fixed native receiver anchor panel for query routes. MM excludes self pairs. The reference does not depend on a requested query minibatch. Centering removes the **pre-tanh** constant mode; it does not assert an exactly zero post-tanh mean. Base-plus-contrast reductions preserve zero-output attachment and live gamma gradients while evaluating fine values once.

Planning, global calibration, reference centering and native coarse/local context remain full-information paths. Phase-current donors also retain ancestry from earlier MM/ME/EM, predicted ports and Stage-A. Sparse donor support restricts the direct fixed-plan control-content path; it does not erase those other dependencies. No recursive Tree, local/whole shadow, K/diversity target, new solve, inverse head or alternate sparse value executor is introduced.

| Code path | Disposition |
|---|---|
| Fine nonlinear MM/ME/EM/QM/QE, source measures, prepared source projections, Stage-A/predicted ports, native coarse/local | Retained |
| Global calibration and historical model/optimizer compatibility | Retained; compact execution opt-in |
| Tree-C/Tree-Lite, recursive indexes, local-shadow tools, all historical checkpoints | Frozen research reference; preserved |
| Generic one-group control expansion | Replaced in G-fast execution only |
| Recursive/independently rebuilt candidate planning | Replaced by one explicit input-only tensor plan |
| Dense physical values and inherited response/reference limits | Retained; no executor-saving or inverse-readiness claim |

The parent is Run3402 `epoch_0500_model.pt`, SHA256 `1a8ac8d976edc72e55950eb3404145adc4ca124e7a529010e02e1a882c5ff051`. Fixed 25_v1 manifest fingerprint is `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`. It selects 150 training/22 exposed validation cases input-only, preserves module-count strata, source partition separation and selected-training normalization. Both new identities use seed 0, Q1024, micro 8/effective 48, FP32, the inherited constant AdamW LR 3e-4/weight decay 1e-5, and the original absolute 1000 objective/response schedule. There is no separate LR scheduler.

Run3601 G-fast has 4,670,953 parameters in 233 trainable tensors. Run3602 Tensor-H has 4,791,304 in 273, adding 120,351 parameters/40 tensors (+2.58%). The parent has 213 common tensors with AdamW moments at step 2000 and 20 without saved moments. Both preserve this exact distinction; only Tensor's new residual moments start empty. RNG, normalizers, calibration and work counters are inherited. Both child best-field selectors start empty and compare only their saved additional 100 reviews; the parent is not silently a child best alias. The untouched parent campaign metadata still contains historical `Global-C`, `parent=null` and `matched_fresh_initialization=true` labels. Those inherited labels do not describe this continuation: each child’s attachment receipt records the exact parent hash and 500 inherited epochs, and is the authoritative lineage.

Training executes immutable source snapshots with the original project resource anchor, preserving configuration/Stage-A/atlas identities. The first Tensor launch failed before any update because its initial snapshot anchor changed resource identity; its 2.413 s interval is charged and the original anchor was restored. The running trainers were not hot-edited. Monitoring retains e600/700/800/900/1000, latest and best-field aliases; no intermediate 25 snapshots.

## Review, predictor, organizer and inverse results

The first five measured complete development epochs forecast 9.5125 s G-fast and 11.8733 s Tensor-H: approximately 15.85/19.79 minutes for 100 additional epochs and 1.321/1.649 GPU-associated training hours for 500. This early forecast includes epoch training and validation scopes, not startup/plot/export. The first actual Tensor epoch has nonzero gamma task gradient/update (0.008618/0.004124); the next has nonzero content/admission/donor/receiver gradients (5.85e-5/1.84e-5/3.74e-6/5.43e-6) and updates. This establishes a live learned branch, not useful physical organization.

Native Tensor zero-attachment checks **against the attached G-fast parent** pass strict low-M/high-M field, port, material, common parameter, heat and query gradients. This means tolerance agreement, not bitwise equality: high-M q-normal proxy differs by at most 2.9087e-5; maximum common/input VJP errors are 4.7684e-7 low-M and 1.9073e-6 high-M. This does not remove the separate original-generic-Global versus G-fast VJP misses above. CPU tests cover module permutation, unequal-mass environmental atom splitting, query batching, soft/blended/exact sparse ages, once-wrapper phase sharing, empty donor types, eligible pre-tanh centering, input finite differences away from knots, admission-boundary continuity, and excluded fixed-plan content dependencies while QE physical values remain full-access. These tests establish contracts; the trained evaluation below establishes scientific performance.

A release-reader conformance cleanup makes the QE residual score explicit and omits its constant base score from numerical logits and exported executed score channels. The running immutable snapshots still contain the equivalent addition of a **frozen exactly-zero** base score: its weight/bias are zero in parent e500 and saved e600 and final e1000. No running code, mathematical recipe, model/state keys or optimizer identity changed. The release test uses a deliberately arbitrary nonzero frozen base score with live residuals and verifies exact output invariance, rather than relying on the campaign's zero special case. Final saved-state and native execution checks are reported below.

### Additional 100 review and extension decision

Both children completed absolute 600 with 400 new optimizer updates, 15,000 training case visits and 15,360,000 primary training queries each. Full-grid statistical evaluation covered the same 22 validation cases, with all 24 roles finite. GPU-associated training-process intervals were 961.345 s G-fast (0.26704 h) and 0.33514 h Tensor-H; full-role evaluation added 23.10 s/0.00642 h and 0.00940 h respectively. Startup, plots and checks belong to those process intervals rather than being added again to training scopes.

| Core role | G-fast mean RMSE | Tensor-H mean RMSE | Tensor change vs G-fast |
|---|---:|---:|---:|
| Fluid u | 0.0421732 | 0.0573560 | +36.00% |
| Fluid v | 0.00484959 | 0.00495247 | +2.12% |
| Fluid p | 0.0199087 | 0.0219784 | +10.40% |
| Fluid omega | 0.195323 | 0.186206 | −4.67% |
| Fluid temperature | 1.468299 | 1.315445 | −10.41% |
| Surface temperature | 1.894922 | 1.638093 | −13.55% |
| Material temperature | 1.710449 | 1.579900 | −7.63% |
| q-normal proxy | 4.116671 | 4.070681 | −1.12% |

These are equal-case mean RMSEs in denormalized native benchmark units, not interchangeable SI quantities. Near means fluid receivers within two native module radii; far is its fluid complement. Material peak compares per-module maxima; ports compare raw initial/refined final outside-temperature and h tokens against teacher-port targets. The evaluator's pressure difference is mean p on the minimum-x fluid edge minus mean p on the maximum-x fluid edge; it is not asserted to be a separately validated physical pressure-drop functional. Negative change favors Tensor-H. It improves temperature in 17/22 cases, surface 15/22 and material 14/22, but u worsens in every case. Tail changes agree with this concern: u p 90/max worsen 43.11%/36.96%, pressure 7.57%/6.99%; temperature p 90/max improve 12.43%/10.21%, surface 9.31%/30.36% and material 6.64%/19.25%.

The thermal gain is not uniform across strata. At M3 (n 6), fluid/surface/material changes are +1.07%/+0.07%/+7.38%; M5 (n 6), −11.75%/−4.17%/−1.02%; M7 (n 6), −18.33%/−30.14%/−25.19%; M10 (n 4), −6.64%/−7.65%/+2.59%. Relative to the inherited e500, both children's surface/material errors have increased at this review; a gain against matched continuation does not imply universal gain against the parent.

The actual Tensor plan is at age 100/fraction 0 in every evaluated case. Positive admission supports equal M+1 (4/6/8/11 in the four strata), and donor supports remain positive throughout the prescribed soft stage. This is **not learned exact sparse K**. P2 gamma RMS is MM 0.00559, ME 0.02457, EM 0.00600, QM 0.000649 and QE 0.003566. Both arms execute 43,972,896 fine MLP rows/3,542 calls in full 22 evaluation. No physical executor saving is measured.

Saved e600 audits additionally verify equal Torch/CUDA RNG state, dataset binding and campaign schedule after the matched 100 epochs. G-fast retains 213 moment states at step 2400. Tensor retains the same 213 plus 40 new moment states at step 400; the 20 common no-moment tensors remain without moments. The review released unchanged matched continuation to absolute 1000: finite outputs, live residual updates, meaningful thermal gains and measured budget support the authorized additional 500 experiment. Velocity/pressure misses are explicitly retained; no sweep, objective change or extra architecture was introduced to repair them during comparison.

### Additional 200 transition

The saved Tensor absolute 700 checkpoint was probed on fixed validation case 0687 (M10) with eight native queries. It builds one shared wrapper plan, has sparse_fraction1.0, and its admission/M/E densities exactly equal recomputation by the declared sparse projections. Actual admission has **K4 of 11 valid proposals**. Density memberships over all valid proposals have 73 positive M and 1,139 positive E entries, integrating against their physical measures within 1.2e-7 of one. This is a trained exact-support example, not an all 22 K distribution or a forced M-to-K rule. The CPU probe uses no GPU, updates or new solves. Dense tensor arithmetic still evaluates allocated proposal pooling/actions; coefficient zeros do not establish hardware control or physical-value savings.

### Final additional 500 endpoint: predictor and A, added learning value

Both children completed absolute 1000/additional 500. The common best-field rule evaluates the same saved native monitoring checkpoints at 600/700/800/900/1000, and **both selected 1000**. Thus exact-endpoint and selected-checkpoint evaluations reuse identical saved measurements rather than duplicating evaluations. All 24 reported roles are finite on all 22 matched validation cases. These are exposed fixed25_v1 development metrics, not full-population or independent-test evidence. No confidence interval or multi-seed result is implied.

The final full-role table below uses equal-case mean RMSE in each role's denormalized native units. Negative percentage favors Tensor-H. p 90 and maximum columns compare each arm's marginal distribution; they are not the paired worst-case delta. Cases improved count paired per-case RMSE. Core roles are bold.

| Role | G-fast mean | Tensor-H mean | Mean change | p 90 change | Maximum change | Cases improved |
|---|---:|---:|---:|---:|---:|---:|
| far/omega | 0.0660679 | 0.0804763 | +21.81% | +22.89% | +22.57% | 1/22 |
| far/p | 0.0134785 | 0.0165687 | +22.93% | +13.91% | +46.71% | 2/22 |
| far/temperature | 1.16265 | 0.938012 | -19.32% | -20.36% | -14.70% | 14/22 |
| far/u | 0.0290492 | 0.0296384 | +2.03% | +9.10% | +3.11% | 10/22 |
| far/v | 0.00346979 | 0.00405206 | +16.78% | +16.16% | +13.37% | 2/22 |
| final_port/h_effective | 0.89437 | 0.793863 | -11.24% | -19.18% | -22.54% | 18/22 |
| final_port/outside_temperature | 1.77781 | 0.953183 | -46.38% | -37.62% | -29.42% | 22/22 |
| **fluid/omega** | 0.122856 | 0.133456 | +8.63% | +11.88% | +3.91% | 6/22 |
| **fluid/p** | 0.0143447 | 0.017395 | +21.26% | +19.76% | +50.88% | 3/22 |
| **fluid/temperature** | 1.17451 | 0.940162 | -19.95% | -19.59% | -12.20% | 14/22 |
| **fluid/u** | 0.0306896 | 0.0317718 | +3.53% | +11.18% | +4.85% | 8/22 |
| **fluid/v** | 0.00377759 | 0.00435925 | +15.40% | +7.88% | +3.67% | 1/22 |
| initial_port/h_effective | 11.8021 | 11.8285 | +0.22% | +0.17% | +0.10% | 1/22 |
| initial_port/outside_temperature | 1.13386 | 1.82412 | +60.88% | +29.62% | +50.08% | 0/22 |
| inlet_outlet_pressure_difference | 0.00841499 | 0.00574119 | -31.77% | -13.48% | -24.73% | 18/22 |
| **material_temperature** | 1.04896 | 0.93089 | -11.26% | -7.26% | -27.94% | 11/22 |
| module_material_peak | 1.15867 | 0.984887 | -15.00% | -10.25% | -31.53% | 12/22 |
| near/omega | 0.278226 | 0.285592 | +2.65% | +8.20% | -8.63% | 7/22 |
| near/p | 0.0182509 | 0.0210678 | +15.43% | +23.05% | +51.28% | 7/22 |
| near/temperature | 1.24004 | 0.918134 | -25.96% | -19.09% | -24.87% | 18/22 |
| near/u | 0.0373988 | 0.0417406 | +11.61% | +22.63% | +29.48% | 5/22 |
| near/v | 0.00500943 | 0.00567524 | +13.29% | +13.06% | +4.43% | 3/22 |
| **q_normal_proxy** | 3.72215 | 3.54808 | -4.68% | -13.40% | +5.93% | 16/22 |
| **surface_temperature** | 1.23966 | 1.0903 | -12.05% | -13.66% | -25.43% | 13/22 |

Tensor-H improves 10/24 means and misses 14/24, including four of eight core-role misses. Pressure's maximum worsens 50.88%; its worst case shifts from 0663 to 0291. Fluid-u worst case remains 0279. Fluid-temperature worst cases differ,0299 for G-fast and 0684 for Tensor-H; material worst case 0663 remains shared. q-proxy mean improves 4.68%, but its maximum worsens 5.93% and worst case shifts 0663→0645. Better edge pressure difference (−31.77%) does not negate the worse fluid pressure field (+21.26%); these are different quantities. Initial outside-port temperature worsens 60.88% in all 22 cases, whereas final refined outside temperature improves 46.38% in all 22. Refinement and initial-port fidelity must remain separate claims.

Per-module-count changes reveal the thermal tradeoff and preserve every selected stratum:

| M (n) | u | v | p | omega | Fluid T | Surface T | Material T | q proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 (6) | -12.10% | +19.11% | +22.93% | +7.85% | -9.13% | -3.87% | -5.32% | +3.10% |
| 5 (6) | +7.10% | +19.00% | +31.72% | +7.96% | -29.09% | -19.59% | -22.74% | -8.90% |
| 7 (6) | +7.40% | +17.28% | +9.60% | +10.70% | -21.85% | -15.32% | -14.83% | -7.91% |
| 10 (4) | +19.74% | +5.58% | +22.00% | +7.51% | -16.95% | -3.94% | +8.00% | -3.54% |

In particular, M10 material temperature worsens 8.00%, despite the overall 11.26% material gain. The parent comparison reuses the verified identical 22 case IDs and normalization:

| Core role | Inherited e500 mean | G-fast change vs inherited | Tensor-H change vs inherited |
|---|---:|---:|---:|
| fluid/u | 0.0545362 | -43.73% | -41.74% |
| fluid/v | 0.00629477 | -39.99% | -30.75% |
| fluid/p | 0.0199239 | -28.00% | -12.69% |
| fluid/omega | 0.196482 | -37.47% | -32.08% |
| fluid/temperature | 1.46366 | -19.76% | -35.77% |
| surface_temperature | 1.47714 | -16.08% | -26.19% |
| material_temperature | 1.24095 | -15.47% | -24.99% |
| q_normal_proxy | 4.27964 | -13.03% | -17.09% |

Both children improve all eight means against inherited e500. Those gains include 500 more epochs of learning and cannot be credited to reader factorization alone. The matched G-fast comparison is the residual's added-learning-value evidence; Tensor-H also has 2.58% more trainable parameters. Mature Dense/Tree references have different histories and are not relabelled as matched baselines in this experiment.

![Paired learning against additional epochs and measured training scope, with full-grid endpoint role comparisons](../../../diagnostics/generated/lean_reset_20261005/figures/02_learning.png)

Figure 2. The arms inherit 500 epochs and visit the same 75,000 additional training cases each. Curves show actual sampled native validation, with raw values and a 10-epoch rolling median; time axes accumulate measured training scopes, not the complete GPU-process ledger. Soft admission ends at+100, blending at+200, then exact sparse projection runs through+500. Full 22 thermal measurements are shown only at inherited 500, additional 100 and additional 500. Final sampled-validation MSE is 0.0218112 G-fast versus 0.0217048 Tensor-H, while the full-grid field table shows the mixed physical-channel result. [PDF master](../../../diagnostics/generated/lean_reset_20261005/figures/02_learning.pdf).

![Four fixed validation physical fields and unclipped temperature and velocity residuals at absolute 1000](../../../diagnostics/generated/lean_reset_20261005/figures/03a_physical_fields.png)

Figure 3 a. Saved exact 1000 arrays for the input-selected fixed cases 0277/0291/0294/0687 use the same native 12×6 coordinates, receiver masks and stored analytic-wake/shared-grid references. Residuals are prediction minus reference; common unclipped scales are T−0.889..43.073, ΔT±6.280 and Δu±0.338 in dataset-native units, with no asserted SI/CFD calibration. Case 0277 is a visible thermal miss: Tensor-H fluid-T RMSE 0.8574 versus G-fast 0.5854, despite its better u RMSE 0.02776 versus 0.03114. The fixed-four details are not a favorable-case selection or an all 22 statistic.

![Matched interface residuals and per-module material temperatures for the same four checkpoints and cases](../../../diagnostics/generated/lean_reset_20261005/figures/03b_interface_material.png)

Figure 3 b. Interface surface-T residuals use±4.167 and q-normal proxy residuals±19.199 common unclipped native scales. Material summaries retain module identity and reference values. These saved numerical outputs support the thermal gains and their nonuniform module-level misses; the M10 all 22 stratum's material mean worsens 8.00%. q-normal remains a benchmark proxy rather than a validated physical flux. No new forward calls were needed to render either physical page. [Two-page PDF master](../../../diagnostics/generated/lean_reset_20261005/figures/03_physical.pdf).

### Final organizer: B, representation faithfulness and measured utility

At additional 500, exact sparse fraction is 1. Across 22 cases, learned admission K ranges 1–4, mean 2.136, against mean 6.909 valid proposals; allocated capacity stays 13 for the present Mmax 12 dataset. At fixed M3, K varies 2–3; M5,1–4; M7,1–2; M10 hasK 2 for all four cases. Variation within M3/M5/M7 is measured; M10 supplies no within-stratum K variation. All four detailed cases happen to haveK 2 and do not summarize the population.

Active-group M donor-pair counts range 4–9/3–14/5–12/10–14 in M3/5/7/10; environmental counts 209–345/88–429/108–239/179–217. Donors may recur in several groups. Full-panel active-group totals are 177 M and 4,910 E positive pairs. Those supports describe direct control-content pooling, not removed physical sources. P2 gamma RMS is MM 0.01613, ME 0.09900, EM 0.03724, QM 0.02400 and QE 0.02682; QE includes gain/score channels. Nonzero gamma is evidence of a live control, not useful causal structure.

![Actual Tensor-H admitted source groups, measure-weighted donors, receiver access and QM residual actions](../../../diagnostics/generated/lean_reset_20261005/figures/04a_organizer_groups.png)

Figure 4 a. Exact 1000 exports show input-selected active groups on fixed exposed validation 0277/0291/0294/0687, allK 2 of 4/6/8/11 valid proposals. Selected-group M/E donors are 2/109,3/121,5/112 and 4/101. QM gain-contrast RMS over the source-measure grid is 0.00137/0.00376/0.01596/0.01567. Receiver-access maps vary weakly, and all admitted groups retain positive access at each eligible receiver. Normal action RMS is algebraically reconstructed from saved actual native access, gamma, densities, phase content and centering arrays; control variants below record actual native actions. Global planning/centering/calibration/coarse/local and upstream state ancestry remain separate full-information paths. Fine physical values remain dense; no causal interpretation is made.

Only the three declared variants were newly executed on the fixed four; normal arrays were reused, giving 12 new wrappers and zero optimizer updates. Zero gamma keeps the current trained global/fine/coarse/local base. Reference-mean access uses the declared ν-weighted reference mean and recomputes centering. Removing the input-admission argmax group zeros its gamma contribution without renormalizing admissions or changing physical source inventory. The context restores the model afterwards.

| Same-weight control, fixed4 macro RMSE change vs normal | u | v | p | omega | Fluid T | Surface T | Material T | q proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero residual | +1.483% | +0.170% | +0.421% | +0.487% | +0.427% | +2.024% | +1.525% | +0.369% |
| Reference-mean receiver access | −0.00287% | +0.00014% | −0.00035% | +0.00020% | −0.00005% | −0.00150% | −0.00056% | −0.00045% |
| Remove one input-selected group's residual | −0.00706% | +0.00776% | +0.1280% | +0.02765% | +0.4670% | +0.6038% | +0.9248% | −0.0283% |

Removing residuals has modest thermal impact; averaging receiver access is numerically ineffective at this panel's metric scale. Zero residual even improves edge pressure difference by 5.62%, although it worsens core fluid-p mean 0.42%. Current branch reliance is much smaller than the trained G-fast/Tensor thermal gap: the co-adapted base retains most thermal advantage. A same-weight intervention does not isolate every effect of the training trajectory or prove equivalent training after deleting the planner. The receiver-dependent organization hypothesis therefore **misses its meaningful-utility claim**; exact support and a live gamma branch alone do not rescue it.

![Actual fixed-four thermal field effects of zero residual, mean receiver access and removal of one group](../../../diagnostics/generated/lean_reset_20261005/figures/04b_organizer_effects.png)

Figure 4 b. Normal predictor-reference errors and the three actual control-minus-normal fields use separate shared unclipped scales. Zero-residual field-effect RMSE is 0.03846/0.01258/0.04622/0.09126; reference-mean access effects are only 0.000120/0.0000127/0.0000437/0.000151; group-removal effects are 0.01825/0.00838/0.04438/0.00377 native T units. These are field changes, not RMSE improvement against physical truth. The nearly blank mean-access maps support the receiver-specific utility miss. No solve or inverse search occurred. [Two-page PDF master](../../../diagnostics/generated/lean_reset_20261005/figures/04_organizer.pdf).

The full 22 normal evaluation executes identical fine-value work in both arms: **43,972,896 padded MLP rows/3,542 invocations**. MM 9,504/66, ME 152,064/66, EM 152,064/66, QM 2,568,192/1,672 and QE 41,091,072/1,672. Dense tensor control pooling/actions also remain allocated across proposals/sources. Neither coefficient zeros nor learned K imply measured hardware control savings or value-executor savings. The full-access fallback/global paths are explicitly retained, not included in a claimed sparse reduction.

### Response, inverse and C, transfer evidence

Final frozen e1000 response evaluation uses the existing TRAIN0348 atlas and the same four numerical input probes. Each arm executes 47 wrappers, eight input VJPs, zero optimizer/solver/inverse calls. Frozen parameter snapshots remain bitwise unchanged (G-fast 344 tensors; Tensor-H 384). Every valid prediction and derivative is finite; atlas NaNs occur only in excluded reference cells. TRAIN0348 belongs to exposed selected training, and its stored±0.125 fixed-total heat-transfer targets are analytic/shared-grid benchmark responses, not independent CFD.

| TRAIN0348 response RMSE | G-fast positive / negative transfer | Tensor change vs G-fast positive / negative |
|---|---:|---:|
| Fluid T | 0.08273 / 0.07283 | +2.65% / −13.06% |
| Surface T | 0.09439 / 0.09262 | +2.87% / −21.22% |
| Material T | 0.06805 / 0.06522 | −0.60% / −3.11% |
| q-normal proxy | 0.32139 / 0.32842 | −2.27% / +1.47% |

Both beat zero-change thermal response references: G-fast fluid-T improvement 47.04%/53.38%, surface 63.19%/63.88%, material 79.66%/80.50%, q-proxy 53.22%/52.19%; Tensor fluid-T 45.64%/59.47%, surface 62.13%/71.54%, material 79.78%/81.11%, q-proxy 54.28%/51.49%. Relative Tensor gains are mixed rather than universal.

The known heat-only flow reference is exactly unchanged, but **both models fail the strict null**. G-fast atlas RMS increments are u 0.001297/0.001100, v0.0001113/0.0001144, p 0.0007602/0.0005190 and omega 0.006261/0.006276. Tensor leakage changes vs G-fast are u+2.14%/−10.19%, v+27.57%/+6.94%, p−24.43%/−24.75%, omega+4.82%/−9.07%. Training's null coefficient remains 0; it was not silently increased to repair this failure.

For each fixed-four case, Q14 input-only locations contain six observed fluid-T sensors, six held T locations and two inlet/outlet pressure probes. Reverse AD differentiates only mean observed T, with two directions: first/last active modules' opposite heat changes, and first-module x displacement. Central FD usesε0.001/0.0005 and saves every channel/interface/material response. All 16 scalar AD/two-step FD signs agree locally per arm. Maximum relative discrepancies are G-fast 0.6344% heat/0.3490% geometry and Tensor 0.2842%/0.2811%. This is numerical self-consistency of the actual model, not validation of physical sensitivity.

Cross-arm AD signs disagree for 0291 heat (G-fast+0.10975, Tensor−0.07750) and 0687 module-x (−0.22955 versus+0.18506). There is no positive held derivative reference to choose a correct sign. Tensor's P0 plan is bitwise shared through P1/P2, and no actual sparse support changes occur under these small FD perturbations. Constructed-knot CPU tests separately check continuity; this panel does not prove correctness through arbitrary real support changes. Small-step heat-null RMS normalized by selected-training σ remains nonzero: G-fast u 4.49e-5..9.17e-5, v2.49e-5..7.83e-5, p 5.14e-5..1.25e-4, omega 4.50e-5..2.98e-4; Tensor u 3.55e-5..7.99e-5, v3.54e-5..1.05e-4, p 3.25e-5..1.29e-4, omega 5.10e-5..1.33e-4.

![Existing exposed response fidelity, known heat-null leakage and model-only AD versus two-step finite differences](../../../diagnostics/generated/lean_reset_20261005/figures/05_response.png)

Figure 5. Final 1000 TRAIN0348 responses beat a zero-change thermal predictor but remain mixed between arms. Fixed-four+0.001 heat changes leak into flow whose analytic reference increment is 0. Bottom panels show reverse AD and two finite-difference steps for six-sensor mean T; maximum scalar discrepancy is 0.6344% G-fast and 0.2842% Tensor. Their agreeing local numerics do not settle the two cross-arm sign disagreements or physical response transfer. No held positive family, new solve, inverse model, search, design or sample trail was produced. [PDF master](../../../diagnostics/generated/lean_reset_20261005/figures/05_response.pdf).

The existing [fixed-four physical-reference request](../../../../../../../../../data/wanglz/ModularDT/thermal_development/tree_faithfulness_20261004/follow_on_reference_request_fixed4.json) remains prepared and unexecuted: four aligned baselines plus opposite feasible fixed-total transfers, at most 12 local analytic/shared-grid benchmark solves. This round retains its exact inputs and receiver alignment without executing it, adding atlas families to training, or treating it as CFD evidence. A read-only preservation receipt records the request hash. The new fixed-four derivative directions and small numerical steps are model probes, not substitutes for these reference families.

### Completed work, resource budget and final execution checks

Every arm's 500 additional epochs visit 75,000 selected training cases and 11,000 exposed validation cases, with 9,500 training microbatches,2,000 native optimizer boundaries/updates,76,800,000 primary training fluid positions and 11,264,000 primary validation positions. Auxiliary TRAIN0348 uses 1,000 actual wrappers,1,000 baseline/variant examples and 4,444,700 all-role query positions per arm; these are separate from the primary fluid numerator. Both arms' per-epoch work sequences are identical. No null-training work is executed at coefficient 0.

| Actual 501–1000 work scope | G-fast | Tensor-H |
|---|---:|---:|
| Sum epoch training seconds | 4,482.472 | 5,754.260 |
| Sum epoch validation seconds | 412.212 | 444.077 |
| Mean training / validation / complete seconds | 8.96494 / 0.82442 / 9.78937 | 11.50852 / 0.88815 / 12.39667 |
| Auxiliary response seconds, already inside training | 217.874 | 296.978 |
| Response fraction of training | 4.86% | 5.16% |
| Full main training-process envelope, including restart/save/plot | 4,975.120 s / 1.38198 h | 6,278.835 s / 1.74412 h |
| Peak allocated training memory | 5.19 GiB | 6.34 GiB |

Complete epoch overhead is 1.26634×. Representative final native inference, on the same physical GPU 2 and original engineering B8/Q1024 M1/M12 panels, is 0.115711/0.204111 s G-fast versus 0.150015/0.259211 s Tensor-H (1.29647×/1.26995×). These are synchronized, recorder/profiler-off, warmed completeP 0/P1/P2 calls, three measured samples per panel; panel inference is not an all 22 mean or training step.

Final exact 1000 state audits preserve common RNG streams, dataset/normalization/calibration and absolute schedules. Both retain 213 common moments at step 4000 and 20 no-moment common tensors; Tensor's 40 new states have step 2000. Latest and best-field 1000 agree with the exact milestone. The QE base-score weight/bias remain exactly zero. Native release-reader versus immutable-training-reader parity with identical trained weights passes original output, parameter, heat and query VJP tolerances on B1/Q16 fixed validation0277 (M3) and0687 (M10), including native internal/port outputs: M10 maximum output 1.4067e-5 and gradient 7.1526e-7, not bitwise equality. This verifies the explicit zero-base-score omission and does not reclassify the earlier generic-Global specialization VJP misses.

The final unique-process ledger records **3.433310 aggregate GPU-associated hours**, physical GPU 1=1.611256 h and GPU 2=1.822055 h, including a conservative 762 s initial engineering reservation, all failed attempts, engineering steps, evaluation, interventions and responses. These envelopes include CPU loading/save time associated with each GPU process; they are not CUDA kernel busy hours. Nested response/boundary timings are not charged again. There are 39 disposable optimizer updates plus 4,000 main additional updates, below the 100-disposable cap. No same-GPU process intervals overlap and no receipt remains active. Last GPU work exited 18:35:44 UTC; accounting closed 18:36:04 UTC,2.22367 elapsed hours after the 16:22:39 start, against the 8-hour deadline 00:22:39 UTC. Report/commit CPU closure is recorded separately in the local final task receipt. Pre-task formal history is excluded from this new-work charge and has no overlap.

Actual per-arm complete-epoch scopes forecast another 100 selected-data epochs at 16.32 min G-fast/20.66 min Tensor, or 500 at 1.360 h/1.722 h before process-envelope overhead; actual 500 main envelopes above provide the stronger observed budget. Full 600/90 Tensor training/validation and its full scheduled-callback cost are **unmeasured**. No fresh 5 k recipe or formal readiness is asserted. The original Global formal e400 recovery recipe remains a separately chosen manual action with its original estimated 45.18 h remainder; Tree has no durable state to resume.

Predictor next step: use the measured broad-field G-fast baseline with numerical limitations visible. Organizer next step: preserve this thermal result, treat receiver-specific utility as missed, and do not begin another forced-sparsity sequence. Inverse next step: preserve the unexecuted held-reference request; finite model derivatives alone do not justify inverse use. No automatic formal restart, new physical solve, inverse campaign, additional model portfolio or scientific deletion follows this report.

### Figure and evidence index

Five selected PDF groups, with seven small PNG companions embedded above: [1 cost](../../../diagnostics/generated/lean_reset_20261005/figures/01_cost.pdf), [2 learning](../../../diagnostics/generated/lean_reset_20261005/figures/02_learning.pdf), [3 physical fields/interface/material](../../../diagnostics/generated/lean_reset_20261005/figures/03_physical.pdf), [4 actual organizer and interventions](../../../diagnostics/generated/lean_reset_20261005/figures/04_organizer.pdf), [5 responses/derivatives](../../../diagnostics/generated/lean_reset_20261005/figures/05_response.pdf). Every selected page was visually inspected. The final exports supersede this round's early-review presentation copies; checkpoint masters and numerical arrays remain intact. Figures are local ignored artifacts, so this report's figure links require the preserved evidence checkout rather than a bare Git clone.

Key measured receipts under the evidence root are [formal cost/disposition](../../../../../../../../../data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/formal_cost_and_disposition.json), [final all-role endpoint](../../../../../../../../../data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/final_endpoint_measured_summary.json), [paired saved-state audit](../../../../../../../../../data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/paired_e1000_state_audit.json), [actual query work](../../../../../../../../../data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/final_query_work_ledger.json), [GPU budget](../../../../../../../../../data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/final_gpu_budget_ledger.json), [trained native parity/timing](../../../../../../../../../data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/organizer/final_native_parity_timing.json), [organizer utility](../../../../../../../../../data/wanglz/ModularDT/thermal_development/lean_interaction_20261005/organizer/final_absolute1000_additional500/residual_utility_summary.json) and the [continuation guide](../../guides/Thermal_Lean_Continuation.md). These receipts retain exact paths, protocol, controls and numerical source identities.

### Validation and artifact boundary

The final affected CPU suite passes **180 tests with one resource-dependent skip** in 5.21 s. It covers the new readers/attachment/export, native organizers and historical topology replay, full-role evaluator/interventions, fixed-quarter profiles/training/evaluation/config binding and inverse-scope guards. The passing invocation uses `PYTHONPATH=src:Case_ThermalChannel/src:tools` in the ModularDT environment. The skipped evaluator test requires retained native Thermal resources; the actual GPU model/checkpoint evaluations and numerical checks are reported separately, rather than describing a CPU fixture as physical evidence.

All new Python files pass Ruff. The preparation CLI validates the pinned parent hash before trusted deserialization; a negative test confirms a wrong-hash file never reaches the loader. Modified legacy files retain the same 99 baseline lint findings, with no added findings. The Tensor disabled factory option is explicitly covered: absent/False preserve the historical backend/state inventory, True opts in and a wrong-architecture True request rejects. These are compatibility checks, not predictor/organizer measurements.

All checkpoints, numeric arrays, receipts, source snapshots, profiler traces, one-time analysis/renderers and figure exports remain in local ignored paths. The five selected figure groups use PDF masters and small PNG companions via the ignored report figure directory. Every outgoing object path, including files added and subsequently deleted in earlier outgoing commits, is audited before pushing; the repository pre-push gate supplements that audit. The final push/tip verification is recorded at task completion.
