# Native-context organizer comparison: measured 100-epoch review

Tree-C and Global-C have both completed their first 100 development epochs and stopped before epoch101. They restore the same native coarse/local physical context while testing receiver-dependent collective organization against a trained case-global controller. Tree-C improves six of eight core physical means at this age, including a 21.57% reduction in u RMSE, but its fluid-temperature RMSE is 9.35% worse and its final effective-h port error is 40.29% worse. The result is a prediction tradeoff, not organizer confirmation or positive heat-response generalization.

The decision recorded before continuation is to complete both healthy arms to500 under the original absolute1000 schedule. Nonzero known-null responses and the mature Dense reference winning are measured limitations, not reasons to stop healthy reconstruction at100. No1000 extension is released. Its current Tree forecast misses the science cutoff; reconsider only from actual500 evidence and a refreshed forecast.

## Identity, exposure and health

This is fixed25_v1: 150 original-training and 22 canonical-validation cases, manifest `933b0138ba2f8447a1ecadfe31fd0bb2cb4a05607d3ac3d9f0dc79419f196044`. Normalization was fitted only on those selected training cases. Frozen Stage-A retains its own checkpoint transform. Initialization/query seed is 0; selection seed is 20261004. Both arms use FP32, Q1024, effective batch 48, microbatch 8 and the same absolute 1000-epoch schedule. Each 100 run completed 15,000 training case visits, 400 ordinary updates and 15.36 million primary training queries. Per-epoch validation adds 2.2528 million primary queries per arm over this interval; diagnostic/auxiliary work is separate. These 22 repeatedly exposed cases are development evidence, not independent testing.

Tree-C Run3401 uses `native_context_tree_honf`; Global-C Run3402 uses `native_context_global_control_honf`. Their344 common materialized tensors matched at initialization. Actual complete low/high-M wrappers, finite updates, optimizer coverage, save/reload and staged control gradients were checked before training. Tree-C has4,803,256 trainable scalars and Global-C4,670,953; each retains1,035,139 common frozen scalars. The additional132,303 Tree parameters include its planner and score machinery. Task values and all24-role evaluation results are finite. No persistent instability or disconnected intended component was found.

| Arm | Literal checkpoint | SHA256 | Saved field-MSE selector | Training source |
|---|---|---|---:|---|
| Tree-C | Run3401 `epoch_0100_model.pt` | `05c990ab3e3f4347363bb0a09983b51a6abcfb9f5fac364f0b3ce09d51b1b217` |0.1642898321| `7bfd13ba54d4f8ccdff61887f5c5bcda64d3b0a6` |
| Global-C | Run3402 `epoch_0100_model.pt` | `085e73acab9ef52558d679fe2d09381a3b0852dd3c44fed4d9fa8fcff07a6015` |0.1710690856| `03a0707d4c54f9f6d8ae75e7e0505e604eb7783c` |

Tree's first launch failed before any optimizer update because canonical configuration forwarded a Tree-only option to a historical constructor. The targeted factory correction is published in7bfd13b; the failed workspace and5.064 associated seconds remain recorded. Later normalization support in496d4a4 preserves these development defaults and weights. Evaluations imported that newer default-preserving source; their physical checkpoint identities above are unchanged.

## Predictor: gains, misses and all22 evidence

Every cell below is an equal-case physical RMSE in the saved dataset's native units. Near/far definitions, pressure functional, material masks and port definitions are those of the maintained24-role evaluator. q is a benchmark heat-flux proxy; the reference uses analytic flow and stored thermal fields, not an independent CFD/SI calibration. Lower is better. The percentage is Tree/Global minus1. All22 cases contribute to each role; pooled metrics remain separately available in the saved summaries.

| Role | Tree mean / median / p90 | Global mean / median / p90 | Tree change in mean | Tree worst (case) | Global worst (case) |
|---|---|---|---:|---|---|
| far/omega |0.247372 /0.245243 /0.283114|0.254119 /0.252863 /0.291804|−2.66%|0.307923 (0688)|0.318097 (0688)|
| far/p |0.0420126 /0.0424051 /0.0484734|0.0401638 /0.0407536 /0.0447975|+4.60%|0.054411 (0688)|0.0465139 (0688)|
| far/temperature |3.75242 /4.02066 /4.53349|3.42339 /3.52701 /4.47155|+9.61%|5.29144 (0644)|5.06138 (0644)|
| far/u |0.0969556 /0.0967208 /0.10861|0.120223 /0.118306 /0.143965|−19.35%|0.12228 (0688)|0.146481 (0688)|
| far/v |0.00585651 /0.00574979 /0.00689295|0.00601601 /0.00597268 /0.00736875|−2.65%|0.00743709 (0688)|0.00811482 (0688)|
| final_port/h_effective |2.56697 /2.53526 /2.91746|1.82973 /1.78995 /2.09477|+40.29%|3.20095 (0674)|2.30909 (0294)|
| final_port/outside_temperature |3.71291 /3.90475 /5.3351|3.58989 /3.4281 /5.30892|+3.43%|6.08931 (0687)|5.99189 (0687)|
| fluid/omega |0.350955 /0.347504 /0.419167|0.356856 /0.353952 /0.417296|−1.65%|0.438421 (0688)|0.437328 (0686)|
| fluid/p |0.043267 /0.0432273 /0.0528628|0.0422917 /0.0432371 /0.0468994|+2.31%|0.0539713 (0687)|0.0510706 (0687)|
| fluid/temperature |3.82907 /4.13913 /4.61763|3.50182 /3.46658 /4.48963|+9.35%|5.22746 (0663)|4.95037 (0644)|
| fluid/u |0.096464 /0.0968444 /0.107117|0.122993 /0.118353 /0.149598|−21.57%|0.115148 (0688)|0.152318 (0287)|
| fluid/v |0.00710788 /0.00687264 /0.00910687|0.00717434 /0.00682131 /0.0091028|−0.93%|0.00923918 (0687)|0.00941177 (0687)|
| initial_port/h_effective |12.374 /12.4586 /12.7361|12.5931 /12.6489 /12.8823|−1.74%|12.9003 (0684)|12.974 (0684)|
| initial_port/outside_temperature |4.52741 /4.23683 /6.75701|5.15879 /5.08196 /7.34228|−12.24%|7.71017 (0687)|8.27702 (0687)|
| inlet_outlet_pressure_difference |0.0209458 /0.0183595 /0.0379657|0.0287579 /0.030737 /0.045983|−27.17%|0.0446077 (0279)|0.0485892 (0677)|
| material_temperature |2.99329 /3.05443 /4.5773|3.28814 /3.11892 /5.15786|−8.97%|5.62438 (0687)|5.8626 (0687)|
| module_material_peak |2.79888 /2.5058 /4.50183|3.44379 /3.12487 /4.91947|−18.73%|6.00519 (0687)|6.60222 (0687)|
| near/omega |0.688352 /0.688744 /0.742907|0.697732 /0.691364 /0.752169|−1.34%|0.784667 (0279)|0.819411 (0644)|
| near/p |0.0490679 /0.0483171 /0.0586161|0.0522892 /0.0513962 /0.060392|−6.16%|0.0695679 (0687)|0.0716751 (0291)|
| near/temperature |3.98796 /4.20164 /5.37976|3.682 /3.54052 /5.33625|+8.31%|6.33209 (0663)|5.76443 (0687)|
| near/u |0.0933424 /0.0935748 /0.10887|0.137173 /0.134559 /0.176992|−31.95%|0.142437 (0287)|0.200029 (0287)|
| near/v |0.0116803 /0.0120313 /0.0144698|0.0115747 /0.0118949 /0.0150189|+0.91%|0.0158562 (0677)|0.0157178 (0677)|
| q_normal_proxy |7.9136 /8.19126 /9.8447|8.23252 /8.49321 /10.2599|−3.87%|11.5351 (0645)|11.9359 (0645)|
| surface_temperature |3.30889 /3.49576 /5.01598|3.64984 /3.47336 /5.58991|−9.34%|6.02373 (0687)|6.28569 (0687)|

Tree improves16/24 role means. Its u advantage covers22/22 cases; material/surface/q advantages cover16/17/21 cases respectively. Fluid temperature improves only7/22, and final-port h improves0/22. These descriptive paired counts are not significance claims. Tree's lower surface mean coexists with its slightly worse median. Material/temperature tails and near/far flow differences are retained.

| M (validation count) | Fluid-T mean Tree / Global | Material-T mean Tree / Global | Surface-T mean Tree / Global | u mean Tree / Global |
|---|---|---|---|---|
|3 (6)|3.190982 /3.216962|1.910864 /2.009443|2.093596 /2.288843|0.090870 /0.106804|
|5 (6)|3.581947 /3.439224|2.596681 /3.360414|2.888089 /3.682293|0.093486 /0.123185|
|7 (6)|4.287862 /3.641821|3.661046 /3.845314|4.057716 /4.280032|0.100624 /0.125773|
|10 (4)|4.468691 /3.812984|4.210231 /4.262004|4.639798 /4.697375|0.103082 /0.142816|

Each of these M categories has only four selected training cases. High-M fluid-T misses remain substantial; high-M material/surface mean gains are modest. Tree's pooled fluid/material/surface/q RMSE is3.933689/3.551319/3.906171/8.380178, rather than the equal-case means above. The fixed-four native field, signed residual, interface and per-module material figures were rendered from saved arrays and visually inspected with shared unclipped scales. They show better Tree u residuals but mixed thermal residuals; case0687 retains a large module error. No fabricated material spatial coordinates were used.

## Organizer: what is measured and what remains

Actual P0/P1/P2 organization and native-context evidence were recorded during the normal all22 wrappers, with detailed maps only for0277/0291/0294/0687. Both retain the all-source coarse path and native local neighborhoods alongside their fine interactions. Those bypasses prevent a Tree fine-support map from describing the complete predictor dependency graph. Tree planner descriptors are an additional disclosed all-source control dependency. Global-C has one full-source collective control per mechanism, without a learned receiver partition or fabricated structural penalty.

The trained physical tradeoff is preliminary evidence about added organization. Fixed-weight identity dependence alone will not establish uniqueness. Final selected-weight route-uniform, identity, root-union, effective reassignment and conditional one-group interventions remain scheduled after500. Logical memberships, coarse attention cells, local gathered rows and fine rectangular executor rows are separate units; no sparse executor saving is claimed from this100 review. All22 phase inventory and the actual native-path maps remain in the evidence directory below.

The actual all22/all15-route inventory gives Tree 15 allocated slots per route, MM/ME frontiers of 3–8 and EM/QM/QE frontiers of 8. P2 QM/QE have 8 participating groups and participation-equivalent counts of 7.300–7.874. Exact source-resolved receiver signatures are 191–259 for QM and 193–262 for QE, versus one Global signature for either query route. These signatures concatenate masked densities and projected actions; Global MM's 3–10 signatures arise from eligibility/self-exclusion and do not imply a learned receiver partition. Tree pooled QM support is 98.079%/94.295%/94.499% in P0/P1/P2 and MM support is 97.266% in P1/P2. Its QE/ME/EM support and every Global route's support are 100%.

Fine hook work agrees exactly between arms by case/route: each all22 panel executes 9,504 MM rows in 66 calls, 152,064 ME and EM rows each in 66 calls, 2,568,192 QM rows in 1,672 calls and 41,091,072 QE rows in 1,672 calls. These are actual retained rectangular rows, including padding and port/detail-query streams, not primary training Q1024 work. Coarse source/self attention allocations total 143,616/5,632 cells per phase; coarse receiver allocations are 540,672/540,672/5,767,168 cells in P0/P1/P2. Local gathered MLP rows are 9,448/9,594/58,832, against allocated geometric pairs of 202,752/202,752/2,162,688. Native geometric work matches by case/phase. Hidden-context norm maps are not energy or measured attention weights. The paired native-path figures were visually inspected; no sparse-work saving is measured.

## Inverse boundary and forward-response checks

No inverse model or search was trained. Known fixed-geometry heat-null evaluation completed22 cases,172 transfer variants and194 complete wrappers per arm. Equal-case mean variant RMS, with each increment divided by its selected-training channel standard deviation, is:

| Channel | Tree raw / train-scaled RMS | Global raw / train-scaled RMS |
|---|---|---|
|u|0.00167822 /0.00415420|0.00100576 /0.00248961|
|v|0.0000999657 /0.00217765|0.0000809342 /0.00176307|
|p|0.000922149 /0.00605792|0.000450273 /0.00295800|
|omega|0.00226876 /0.00222666|0.00179421 /0.00176092|

Both violate the known benchmark null; Tree leakage exceeds Global in every channel. Fluid-temperature sensitivity0.080441/0.047524 is model-only, because positive held thermal truth is absent. There is no division by a zero target. The disabled extra null-training coefficient remains zero in both continuations; this review does not tune it on validation.

On exposed TRAIN0348, finite response error averaged over opposite±0.125 heat transfers is Tree/Global/zero-change: fluid-T0.168463/0.156454/0.156223; surface-T0.250472/0.247095/0.256417; material-T0.211032/0.199832/0.334492; q-proxy0.608970/0.631055/0.686996. Both improve surface/material/q responses over zero change, but both miss fluid-T. Tree's q response is better than Global, whereas Global's thermal responses are better. Native response masks use7464 fluid,640 interface and30,960 material entries. This three-state panel is distinct from the older eleven-state atlas/mask and supplies no held-out response-transfer claim.

Four heat VJPs and ordinary centered rebuilds per arm are finite. Tree heat AD/FD discrepancies for0277/0291/0294 are0.030087/0.014540/0.005658%, with the same discrete support. Tree0687 gives0.005019% but crosses one actual P1/MM source/control support entry, so it is not a same-support verification. Global's four same-support discrepancies are0.038493/0.012386/0.195484/0.019341%. Context derivatives are FD-only: the existing native operator float-casts prescribed context, so context AD is not established. Tree0687's negative context probe also changes one P1/MM support entry.

State and hook ownership restore correctly on normal and exceptional exits for4/4 representatives in each arm. No-grad alpha0 versus grad-enabled alpha0 has maximum interface Ts/q differences Tree4.7684e−6/9.3460e−5 and Global3.8147e−6/3.5286e−5 at0687. These are different execution modes, not repeated-same-mode drift, a state leak or a physical response floor. The bounded old Tree900 diagnosis localized8/10 q-component failures while10/10 Ts components passed; its unadopted FP64 pooling trial did not repair all failures. That historical result remains a recorded miss.

## A/B/C and continuation

**A, added organizer value:** the matched trained Tree has a useful u/material/surface/pressure-functional tradeoff at100, but Global is stronger on fluid-T/final-port h and null responses. Final receiver-dependent utility beyond global calibration is unresolved. **B, faithful interpretation:** native bypasses and support crossings are disclosed, old numerical misses are localized, and current instrumentation ownership passes; final whole-rebuild/VJP and conditional-donor checks remain pending. **C, transferable response knowledge:** finite exposed-training surface/material/q responses beat zero change, but fluid-T misses and known-null leakage persist. There is no held changed-heat family among the selected172 cases. The existing12-solve request is prepared and unexecuted.

Measured native training/validation epoch means are37.8763s Tree and8.6829s Global; latest20 means37.6166/8.6404s. These are training scopes, not interleaved inference benchmarks. The all100 maxima44.8148s Tree (epoch35) and9.0775s Global are retained; the isolated Tree spike has no proven contention cause. Full first100 child lifetimes are3808.507s/888.629s. Response jobs add20.3803s/16.5534s, each27 wrappers/four heat VJPs. Normal/null evaluations, failed startup and bounded old diagnostics are separately included in campaign process accounting.

At05:23:20 UTC, actual total allocation is1.46716 GPU-associated process-hours and1.88434 elapsed hours. Allocation includes full child lifetimes and failures, not active GPU utilization. The conservative05:19 forecast uses the inclusive epoch maxima,15s load,10s per future cadence and an explicitly **unmeasured**3s/epoch atlas allowance: paired500 projects8.15241 associated hours before the unpriced final package, with Tree ending about10:38:59 UTC excluding review delay. Tree1000 projects17:18:17 UTC and misses the14:30:17 science cutoff. The absolute15:30:17 deadline reserves the final elapsed hour for closeout. Refresh the forecast from actual e101 auxiliary cost; do not equate the allowance with measured cost.

Release both literal100 checkpoints to500 only after this measured review is saved. Keep the original optimizer/RNG/data identity and absolute schedule; apply the common physical policy1→2 amendment at101 and selected TRAIN0348 callback, with null-training0. Preserve Tree's completed epoch26 measure calibration exactly. Continue100-epoch monitoring/selection/plots and light all22 physical milestones at200/300/400. The CPU resume guards and calibration-helper replay passed, but those are preparation checks, not101 learning. GPUs1/2 are available; Tree usesGPU1 and GlobalGPU2. No1000, formal optimizer, new solve, full-data development expansion or inverse search is released.

## Saved evidence

Evidence root: `/data/wanglz/ModularDT/thermal_development/native_context_20261005`. Training workspaces are the Run3401/3402 paths in `prepared_e100_to_e500_commands.json`. The source catalogue, all22 identity, actual phase inventory, fixed-four arrays and native path maps are in `evaluation/{tree-c,global-c}_e100_normal`; null variants in the corresponding `_e100_null` directories. `report_numerics.json` retains all24 means/medians/p90/worst/per-M/pooled metrics and paired counts. `figure_preparation/e100_information/e100_organization_inventory.{json,md}` retains actual route/work totals and count semantics. `response_preparation/paired_e100_response_readout.json` joins the exact100 response jobs. `paired_e100_review_forecast.json` and `budget_snapshot.json` retain accounting assumptions and intervals.

Saved-only100 learning, physical and paired response presentation figures were visually inspected. These temporary development exports remain ignored in `figure_preparation`, with their PDF masters and source receipts. Final selected figures will be embedded and indexed in the completed report; preparation images do not substitute for completed500 measurements.
