# Four-model HONF comparison: accuracy, efficiency, organization and structural response

**Completed comparison, updated 2026-10-07.** The new system replaces the classical predicted-port/local-surrogate/refinement chain with an ordinary dependency-separated flow predictor (D-sep, Run3901) and a source-resolved thermal response model (R-Direct, Run3902). Both fresh full-TRAIN stages completed 5,000 epochs. This report compares that composed system with HONF1401, Dense1804 and sparse-incidence HONF1502, reorganizing the saved evidence into four presentation themes. The models exchange advantages: R-Direct is faster and improves streamwise velocity and mean surface temperature; Dense1804 remains stronger on absolute fluid temperature and several difficult tails. The revision adds no training, model evaluation or reference solves.

**Predictor — gains, misses, evidence and next step.** On canonical89, R-Direct reduces mean u RMSE by 33.4% and surface-temperature RMSE by 27.5% against Dense1804; six saved signed heating states show 16.2% lower fluid-response and 14.8% lower surface-response RMSE. It worsens nominal fluid-temperature RMSE by 44.3%, vorticity by 53.7% and mean module-peak error by 6.4%; material/surface tails and geometry response remain limitations. The precise prepared arithmetic passes 288 checks, which establishes numerical consistency rather than physical accuracy. Retain R-Direct for fixed-layout response research and Dense1804 as a nominal-temperature reference; use the completed checkpoints for any further analysis rather than treating this comparison as authorization for another fit.

**Organizer — gains, misses, evidence and next step.** Saved receiver-local response coefficients preserve physical heater identities and support input-range-dependent donor covers. Equal-count nearest/upstream controls show a useful learned response structure, but many patches still need all sources. HONF1502 has genuine zero incidences yet retains rectangular computation; the measured R-Direct speed advantage is full-access execution. None of these views proves physical causality, global grouping superiority or sparse-executor savings. A future organizer assessment would have to measure actual execution work at controlled fidelity.

**Inverse — gains, misses, evidence and next step.** Every model selects the same correct minima in the small existing heating pools, with zero observed saved-reference regret. Reusable R-Direct kernels make those evaluations convenient, but the experiment gives no unique inverse-design advantage. There was no continuous search, inverse generator, new physical solve or independently certified design. Further design claims would require separate validation, especially after geometry changes.

The earlier DEV22/e2500 preparation narrative is preserved in the [historical readiness archive](_bk/HONF_RDirect_Historical_Readiness_Assessment.md). Its no-launch statement is superseded by the completed formal fits and is not the current execution status. R-group, historical controls, checkpoints, original arrays and PDF masters remain preserved.

## Reading and presentation guide

The four sections below follow the requested presentation order: **1 Accuracy → 2 Efficiency → 3 Module–environment organization → 4 Responses to structural changes**. Each selected figure is a standalone presentation page with a PDF master and a raster companion for this Markdown report. Detailed numerical tables, checkpoint hashes and evidence receipts follow in appendices so the main narrative can be read without repeatedly crossing historical experiments. The selected pages are also supplied as a [PowerPoint deck](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/HONF_Four_Model_Comparison.pptx) and [combined PDF](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/HONF_Four_Model_Comparison.pdf); the deck embeds the inspected figures, while individual PDF pages retain vector/text quality where available.

| Theme | Selected presentation pages | Main question |
|---|---|---|
| Accuracy | Figures 1–5 | Which fields improve, and where do local/tail errors remain? |
| Efficiency | Figures 6–7 | What complete-call time and measured memory are paid? |
| Organization | Figures 8–11 | What information is grouped, routed or retained by physical source? |
| Structural response | Figures 12–16 | What happens after moving obstacles or changing heating? |

For an architectural reading, **Sections 3.1.1–3.1.6** summarize each model and trace structural inputs through encoders, organization/context layers and output heads to physical fields, including the classical physical phases and the R-Direct heating Jacobian. The mathematical exposition is in this report; the existing presentation deck contains the sixteen selected figure pages.

## Comparison contract

The primary population is all **89 canonical source-TEST cases**: original TEST0273 is excluded because its inputs duplicate TRAIN0001. The original90 compatibility panel includes that duplicate and is kept separate. Both were repeatedly exposed to validation and monitoring, so neither is a blind test. Shared physical receivers, fluid masks, near/far definitions, interface/material queries and ordered case IDs are preserved. The fixed detailed layouts are 0277/M3, 0291/M5, 0294/M7 and 0687/M10. References are the existing analytic-wake/shared-grid generator and saved alternatives; there is no independent CFD certification or cross-seed population study.

The main figures use literal **e5000 endpoints**. A separate appendix reports saved-best states: R-Direct thermal e4200 with frozen flow e5000, Dense1804 e4738, HONF1401 e4585 and HONF1502 e4794. Each system uses its checkpoint-native normalization and complete forward path, then denormalizes outputs before physical error calculation. Classical models retain their predicted-port/local-surrogate/refinement paths. Architectures, objectives, normalization bases, histories and component ages differ; this is an actual-system comparison, not a matched architecture ablation.

Fresh formal training uses all **600 original TRAIN cases**, seed 0, FP32, TRAIN-only normalization, effective batch 48/microbatch 8, Q1024 primary fluid queries and learning rate 3e-4 through e2000 followed by cosine decay to 3e-6 at e5000. Each stage records 3,000,000 case visits, 65,000 optimizer updates and 3,072,000,000 primary fluid queries. Together the two stages spend **6,000,000 visits and 130,000 updates**, with two distinct 5,000-epoch optimizations; they are not one jointly trained 5,000-epoch model. The strict fixed25_v1 quarter-data development protocol remains distinct.

## 1. Accuracy: field-specific gains and remaining local failures

### 1.1 Compare physical channels separately

R-Direct leads on u and p, while Dense1804 leads on v and nominal temperature; HONF1401 has the lowest pooled vorticity relative L2. The exact ranking depends on the declared metric: Dense1804 has the lowest equal-case mean vorticity RMSE. Pooled relative L2 weighs reference energy across cases, whereas mean RMSE gives each case equal weight. A single five-channel Euclidean score would mix physical units and hide these distinctions.

![Five pooled physical-channel relative L2 errors for the four e5000 systems](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/01_accuracy_relative_l2.png)

**Figure 1 — Per-field accuracy.** Canonical89 endpoint errors are independently recomputed from raw reference energies and matched error sums. Each channel is dimensionless within its own relative-error definition; no mixed-unit aggregate is used as a selector. See Appendix A for raw-native mean/p90 RMSE and paired intervals. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/01_accuracy_relative_l2.pdf).

### 1.2 Average temperature improvement does not remove difficult tails

R-Direct reduces mean surface-temperature RMSE from 0.429475 to 0.311198 against Dense1804 and wins 74/89 cases. Its material mean improves from 0.329348 to 0.307594, but the paired 95% interval crosses zero. Surface/material p90 nevertheless worsens to 0.705303/0.716253 versus 0.557424/0.434210. Module-peak mean is 0.338021 versus 0.317573, with p90 0.832998 versus 0.530304. A small typical error therefore does not certify reliable hotspots.

![Four-model temperature error distributions with a 90-percent case-fraction reference](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/02_thermal_accuracy_distribution.png)

**Figure 2 — Thermal distributions.** Canonical89 literal-e5000 physical temperature errors retain every case rather than trimming difficult examples. R-Direct’s worst module-peak case is 0287 (1.30422 RMSE), worst material case is 0294 (0.944081), and worst surface/fluid-temperature case is 0295 (0.985360/0.870243). The plot reports existing generator-based errors in stored native temperature units. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/02_thermal_accuracy_distribution.pdf).

### 1.3 Representative fields explain what summary scores miss

The predeclared 0291/M5 case shows nominal temperature and its spatial residuals; it was not selected because one model won. The high-module-count 0687/M10 companion separates large-scale streamwise velocity from vorticity. Near-boundary vorticity remains a substantial D-sep limitation: canonical89 near-omega mean RMSE is 0.0889746 against Dense1804’s 0.0524350. The earlier near-boundary refinement was rejected; it is not silently substituted into this composition.

![Reference and four-model nominal temperature residuals on fixed TEST0291 M5](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/03_temperature_case0291.png)

**Figure 3 — Nominal temperature, 0291/M5.** The shared saved native grid, masks and source geometry support reference/prediction or residual comparison under common physical scales. Residuals are predicted minus reference, with an unclipped shared range ±1.977658. Case-specific relative L2 is 7.94%/5.51%/6.81%/8.02% for HONF1401/Dense1804/HONF1502/R-Direct; this fixed case illustrates rather than replaces the population statistics. Temperature uses native dataset units, not asserted SI units. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/03_temperature_case0291.pdf).

![Reference and four-model streamwise velocity residuals on fixed TEST0687 M10](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/04a_velocity_case0687.png)

**Figure 4 — Streamwise velocity, 0687/M10.** The ten-source fixed case displays the measured D-sep/thermal composition and the three classic endpoints on identical receivers. The strong canonical89 u result, 0.00432883 RMSE versus Dense1804’s 0.00650424, does not establish improvement in every flow quantity. Case-specific u relative L2 is 1.59%/1.19%/1.39%/0.64% for HONF1401/Dense1804/HONF1502/R-Direct; the shared residual range ±0.067265 is unclipped. Velocity values follow the saved generator’s native units. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/04a_velocity_case0687.pdf).

![Reference and four-model vorticity residuals on fixed TEST0687 M10](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/04_vorticity_case0687.png)

**Figure 5 — Vorticity, 0687/M10.** A separate flow-detail page prevents the stronger u prediction from obscuring structured omega errors. Case-specific omega relative L2 is 4.12%/2.74%/2.98%/6.01% for HONF1401/Dense1804/HONF1502/R-Direct. The displayed residual scale uses the pooled 99.5th percentile (±0.144983), saturating 0.5025% of cells; the full absolute maximum is 1.384754. All RMSE/relative-L2 statistics use unclipped saved arrays. Canonical89 omega mean RMSE is 0.0375592 for R-Direct versus 0.0244341 for Dense1804; no independent CFD or discretization-error floor is available. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/04_vorticity_case0687.pdf).

### 1.4 Ports: better reported temperature is not proof of an equivalent closure

Final outside-temperature mean/p90 RMSE improves to 0.300105/0.682245 for R-Direct versus Dense1804’s 0.851560/1.036130. This is a useful output comparison, but the new model does not possess the classical initial-port/refinement trajectory: initial-port metrics remain N/A. Surface and material outputs are read from the common learned temperature response, rather than generated by a separately frozen Stage-A model driven by predicted ports.

The native q and effective-h values are proxies. R-Direct uses `q = −harmonic(k)/delta × (T_outside − T_surface)`, so `h = q/(T_surface − T_outside)` is largely `harmonic(k)/delta` away from clamps. Small h errors consequently have an algebraic explanation and do not independently establish learned transport closure or continuum heat-flux accuracy. The h mean includes a retained outlier, case 0282 (0.748986). Appendix A keeps native port roles separate from field errors.

## 2. Efficiency: complete application cost and historical training context

### 2.1 Matched inference is the strongest cost comparison

Inference means a trained model receives a case and returns predicted fields. The common GPU-2 benchmark uses all four fixed layouts, 8192 native query rows, one warmup and seven rotated complete-call repetitions, native inner chunks and maps off. Input staging and output-to-CPU copy are included. The reported mean averages the four case-median latencies; it is not a bare neural-layer timer or a thousands-of-modules scalability result.

| Model | Complete-call latency (ms/case) | Relative to R-Direct | Incremental peak allocation (MiB) | Mean registered Linear input rows/case |
|---|---:|---:|---:|---:|
| R-Direct + D-sep | 25.239 | 1.00× | 21.845 | 657,497 |
| HONF1401 | 52.530 | 2.08× | 49.228 | 708,342 |
| Dense1804 | 163.290 | 6.47× | 38.772 | 4,456,756.75 |
| HONF1502 | 229.506 | 9.09× | 40.952 | 4,407,766 |

Incremental memory subtracts each invocation’s loaded-model baseline and is not total GPU residency. Linear-row counts are actual registered-layer input evaluations, not FLOPs; layer widths, attention, stencils and backend differences prevent treating their ratios as theoretical complexity. Support/work recording runs separately from the timing calls. All four systems execute complete full-access prediction; none of this benchmark measures a sparse executor.

![Matched four-model inference latency, incremental memory and accuracy-cost tradeoff](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/05_inference_cost_tradeoff.png)

**Figure 6 — Measured inference tradeoff.** Complete-call GPU-2 latency orders R-Direct, HONF1401, Dense1804, HONF1502. Any error axis retains the canonical89 endpoint identity even though timing uses the fixed-four layouts. Lower full-access cost is measured; the plot does not attribute the entire improvement to architecture or learned sparsity. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/05_inference_cost_tradeoff.pdf).

### 2.2 Training time and training peak memory answer a different question

Training computes losses, backpropagates through intermediate states and updates parameters; it generally requires more memory than inference. The actual new flow stage uses 2928.424 s training (0.81345 h) and 3041.918 s process time, with peak allocated memory 235.99 MiB. Thermal uses 17035.424 s training (4.73206 h) and 17348.404 s process time, with peak 875.69 MiB. Combined training is **5.5455 h**; combined process time including startup, validation and saving is **5.6640 h**. Both histories end at e5000/65,000 optimizer steps. This measured completion supersedes the earlier 6.07 h startup forecast.

| System/stage | Training (h) | Logged/process total (h) | Peak allocated memory (MiB) | Comparability |
|---|---:|---:|---:|---|
| New D-sep Run3901 | 0.81345 | 0.84498 | 235.99 | Actual fresh full600 stage |
| New R-Direct Run3902 | 4.73206 | 4.81900 | 875.69 | Actual fresh full600 stage; flow fixed |
| Combined new system | 5.5455 | 5.6640 | Two sequential peaks above | Two e5000 optimizations |
| HONF1401 | N/A | N/A | N/A | No aligned training receipt in this comparison |
| Dense1804, historical | 23.02 | 25.55 | 27,210 | Distinct device/code/objective history |
| HONF1502, historical | 10.76 | 12.15 | 24,273 | Distinct device/code/objective history |

The historical figures come from the [mature comparison’s trainer summaries](run1501_1502_comparison/HONF_1404_1804_1501_1502_Mature_Comparison.md). Historical totals include validation; the new process total additionally accounts for startup/saving. These numbers show recorded resource scale, not a strictly matched training-speed or memory ablation. HONF1404 is not substituted for the requested HONF1401 when a receipt is absent.

![Actual two-stage training resources with explicitly historical Dense1804 and HONF1502 references](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/06_training_resources.png)

**Figure 7 — Training resources.** Fresh Run3901/Run3902 measurements are shown separately from historical 1804/1502 logs; 1401 remains unavailable. The new stages run sequentially, so their training times add while their peak allocations are not summed. Training allocation and the invocation-incremental inference memory in Figure 6 use different baselines. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/06_training_resources.pdf).

### 2.3 Why the new system is lighter

The classic systems retain predicted ports, frozen local Stage-A responses and refinement. Freezing Stage-A parameters does not remove the derivative path through its predicted inputs. The new flow reader directly predicts u/v/p/omega, while thermal contextualizes the layout and constructs source responses without that repeated port/local-surrogate chain. Training separates the flow and thermal graphs and uses microbatch 8 with accumulation to effective batch 48. Shorter paths, smaller active networks and reduced concurrently retained activations provide plausible code-level reasons for the lower cost; their individual shares have not been quantified by controlled ablations.

HONF1502’s zero associations do not currently remove rectangular reader work, and group construction/routing adds overhead: its actual linear-row count remains close to Dense1804 and complete inference is slower. Conversely, fixed geometry and receivers allow R-Direct to reuse prepared K for multiple heating candidates, but the table measures complete calls, not cached applications; no separate repeated-application speedup factor is claimed.

## 3. Module–environment organization: latent routing versus physical response

### 3.1 Four architectures answer different organization questions

![Four model architectures distinguishing latent hyperedges, dense pairwise interactions and direct source response](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_01_architecture.png)

**Figure 8 — Information organization.** This is an architecture schematic, not a learned physical interaction graph. HONF1401 uses six latent hyperedges; Dense1804 uses contextual pairwise messages; HONF1502 adds sparse incidences over twelve group slots; R-Direct preserves physical heater columns with a separate heat-independent D-sep flow component. A group ID is exchangeable and does not identify a physical module. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_01_architecture.pdf).

| Organization property | HONF1401 | Dense1804 | HONF1502 | R-Direct + D-sep |
|---|---|---|---|---|
| Main thermal organization | Six latent hyperedges | Contextual dense pairwise | Twelve latent group slots, sparse incidences | Physical heater-to-receiver kernels |
| Environment role | Learned memberships plus classical paths | Contextual environment/module messages | Sparse memberships plus classical paths | Layout context; not independently actuated heaters |
| Receiver reads | Hyperedge weights | Source/environment messages | Sparse group weights controlling source reader | Signed K_i(x), temperature/heating unit |
| Predicted-port → Stage-A → refinement | Retained | Retained | Retained | Removed from current SourceResponseThermalModel |
| What a small drawn support means | Small positive weights | Dense learned messages | Exact zero routing, full rectangular execution | Post-fit bounded cover; full prediction retained |

The classical P0 → frozen Stage-A local response → P1 refinement → P2 final organization/field read remains a meaningful global–local decomposition. It couples predicted interfaces to a reusable local model and then corrects those interfaces. The current `SourceResponseThermalModel` does **not** use that chain; the older `DependencySeparatedThermalModel` is a different implementation and should not be confused with the current source-response system. Its flow output is composed alongside the thermal output, not supplied as the thermal reader’s input. Thermal discrete-residual training uses saved TRAIN velocities, not D-sep-predicted velocities. Removing the chain changes the physical modeling assumptions, so lower resource cost does not demonstrate that the former interface design was unnecessary.

#### 3.1.1 Shared notation: structure in, physical fields out

The following equations describe the **checkpoint-bound architectures compared here**, at the level of learned blocks rather than every affine layer. They explain how Figure 8 becomes a field reconstruction. Let the input structure be a masked module set, prescribed material and operating conditions, and heating; let the output be the field at requested physical coordinates:

$$\begin{array}{l}\mathcal S=(\{(\mathbf c_i,m_i,h_i)\}_{i=1}^{M_{\mathrm{pad}}},\boldsymbol\mu,\mathbf b),\\ \mathbf x_q\in\Omega,\\ \widehat{\mathbf y}(\mathbf x_q)=[\hat u,\hat v,\hat p,\hat\omega,\hat T](\mathbf x_q).\end{array}$$

Here, center coordinates are $\mathbf c_i$, presence masks are $m_i$, heating powers are $h_i$, $\boldsymbol\mu$ contains material coefficients/radius, and $\mathbf b$ contains prescribed conditions such as Reynolds number, inlet velocity and domain lengths. Padded absent modules contribute zero. Environment tokens represent spatial samples and input-derived geometry/context, with coordinates $\mathbf e_j$ and quadrature weights $w_j$; they are not observed target fields. Encoders map these inputs to module states $\mathbf z_i$, environment states $\mathbf v_j$ and global context $\mathbf g$; Fourier/relative-coordinate features supply the receiver geometry. Below, $E,\Phi,\Psi,D$ denote learned neural blocks, and $\varepsilon$ denotes a numerical denominator safeguard. These equations describe neural reconstruction, not an independent PDE solve or a proof that its residual vanishes.

| Model | Input-to-intermediate path | Receiver/output path | One-sentence interpretation |
|---|---|---|---|
| HONF1401 | Encoded modules/environment → six learned hyperedge summaries | Query-to-hyperedge routing plus enabled pairwise/global/local contexts → field heads | Compress information into latent interaction channels, then read them spatially. |
| Dense1804 | Encoded modules/environment → dense MM/ME/EM contextual messages | Fine module messages and environment attention, plus coarse/local branches → field heads | Preserve contextual fine sources and read them directly at each receiver. |
| HONF1502 | Fine contextual sources → twelve sparse-incidence groups with 16-dimensional controls | Sparse query routes control fine-source messages/attention → field heads | Use groups to control access to source values, rather than replace every source by a group value. |
| R-Direct + D-sep | Heat-independent layout context → physical-source temperature kernels; separate flow states | Kernel × physical heating → shared temperature grid/role extraction; flow decoder → composition | Separate nonlinear geometry preparation from linear heating application. |

#### 3.1.2 Classical physical phases: organization is inside a larger predictor

For **1401, 1804 and 1502**, the organization/reader blocks below operate within the retained thermal wrapper. Use $\mathcal O_r$ for phase-r organization, $p^{(r)}$ for interface-port tokens, $L_A$ for the frozen local Stage-A model, and $F$ for fusion of its response into module states. The deployed predicted-port path with one refinement is:

$$\begin{array}{l}\mathcal S\xrightarrow{E}(\mathbf z^{(0)},\mathbf v,\mathbf g)\xrightarrow{\mathcal O_0,P_{\mathrm{port}}}p^{(0)},\\ \ell^{(0)}=L_A(\boldsymbol\mu,\mathbf h,p^{(0)}),\\ \mathbf z^{(1)}=F(\mathbf z^{(0)},\ell^{(0)}).\end{array}$$

$$\begin{array}{l}o^{(1)}=D_{\mathrm{outside}}(\mathcal O_1(\mathbf z^{(1)},\mathbf v,\mathbf g)),\\ p^{(1)}=R_{\mathrm{port}}(\mathbf z^{(1)},p^{(0)},o^{(1)},\ell^{(0)}),\\ \ell^{(1)}=L_A(\boldsymbol\mu,\mathbf h,p^{(1)}),\\ \mathbf z^{(2)}=F(\mathbf z^{(0)},\ell^{(1)}).\end{array}$$

$$\begin{array}{l}\widehat{\mathbf y}_{\mathrm{norm}}(\mathbf x)=D_{\mathrm{field}}(\mathbf x,\mathcal O_2(\mathbf z^{(2)},\mathbf v,\mathbf g)),\\ \widehat{\mathbf y}_{\mathrm{phys}}=\boldsymbol\mu_y+\boldsymbol\sigma_y\odot\widehat{\mathbf y}_{\mathrm{norm}}.\end{array}$$

Thus P0 predicts provisional ports, P1 reads outside temperatures after the first local response and refines the ports, and P2 reads the final field after refreshing the local response. Local solid/interface predictions also use the local-response outputs and wrapper assembly. $\boldsymbol\mu_y,\boldsymbol\sigma_y$ are checkpoint-native TRAIN field statistics, distinct from material parameters $\boldsymbol\mu$. Frozen Stage-A parameters still permit gradients through their inputs during training. The three models differ in how $\mathcal O_r,D$ organize/read information, while retaining this physical coupling chain.

#### 3.1.3 HONF1401: latent hyperedge summaries and query reads

**Summary:** each present module and each environment token distributes information across six latent hyperedges. For a given physical phase, let $A^M\in\mathbb R^{M_{\mathrm{pad}}\times6}$ and $A^E\in\mathbb R^{N_E\times6}$ be learned memberships; softmax normalizes each present source row over hyperedges. A simplified but implementation-faithful summary/read is:

$$\begin{array}{l}A^M_{ik}=m_i\,\operatorname{softmax}_k(a^M_{ik}),\\ A^E_{jk}=\operatorname{softmax}_k(a^E_{jk}),\\ \mathbf H_k=\Phi_H\!\left(\frac{\sum_i A^M_{ik}\Phi_M(\mathbf z_i)}{\sum_i A^M_{ik}+\varepsilon}+\frac{\sum_j w_j A^E_{jk}\Phi_E(\mathbf v_j)}{\sum_j w_j A^E_{jk}+\varepsilon}\right).\end{array}$$

$$\begin{array}{l}\alpha_{qk}=\operatorname{softmax}_k\!\left(\frac{Q(\mathbf x_q)^\top K(\mathbf H_k)}{\sqrt d}+b_{qk}^{\mathrm{geom}}\right),\\ \mathbf c_H(\mathbf x_q)=\sum_{k=1}^{6}\alpha_{qk}V(\mathbf H_k).\end{array}$$

$$\widehat{\mathbf y}_{\mathrm{norm}}(\mathbf x_q)=D_{1401}\!\left(\operatorname{LN}[\mathbf c_H+\mathbf c_{\mathrm{pair}}+\mathbf c_{\mathrm{global}}+\mathbf c_{\mathrm{direct}}+\mathbf c_{\mathrm{near}}]\right).$$

The last equation includes the enabled auxiliary contexts, with their projections/gates absorbed into $\mathbf c$; it deliberately avoids portraying the six hyperedges as the entire predictor. Module–environment context enters the source states, and learned geometry/mechanism features enter routing and pairwise reads. The output heads predict field channels (including any configured mean/residual split), followed by wrapper assembly and denormalization. H0–H5 are latent slots, not heater identities, and positive normalized routes need not be physical transport fractions.

#### 3.1.4 Dense1804: contextual fine sources and dense receiver reads

**Summary:** retain individual fine module/environment states and contextualize them with typed dense messages. MM denotes module→module, ME environment→module, and EM module→environment. Writing $N_M=\sum_i m_i$, $\delta_{is}$ for relative-coordinate features, and folding global conditioning into $\Psi$, one preparation pass has the form:

$$\begin{array}{l}\mathbf a_i^{MM}=\frac{\sum_{l\ne i}m_l\Psi_{MM}(\mathbf z_i,\mathbf z_l,\delta_{il})}{1+N_M},\\ \mathbf a_i^{ME}=\frac{\sum_j w_j\Psi_{ME}(\mathbf z_i,\mathbf v_j,\delta_{ij})}{\sum_jw_j+\varepsilon},\\ \mathbf a_j^{EM}=\frac{\sum_i m_i\Psi_{EM}(\mathbf v_j,\mathbf z_i,\delta_{ji})}{1+N_M}.\end{array}$$

$$\begin{array}{l}\mathbf z'_i=m_i[\mathbf z_i+\Phi_M(\mathbf z_i,\mathbf a_i^{MM},\mathbf a_i^{ME},\mathbf g)],\\ \mathbf v'_j=\mathbf v_j+\Phi_E(\mathbf v_j,\mathbf a_j^{EM},\mathbf g).\end{array}$$

At each receiver, fine module messages and multihead environment attention supply the main context. The following attention expression shows one head; actual head concatenation/output projection is absorbed into $\Phi_E^{\mathrm{out}}$:

$$\begin{array}{l}\mathbf c_M(\mathbf x_q)=\Phi_M^{\mathrm{out}}\!\left(\frac{\sum_i m_i\Psi_{QM}(\mathbf z'_i,\mathbf x_q-\mathbf c_i,\mathbf g)}{1+N_M}\right),\\ a_{qj}=\operatorname{softmax}_j\!\left(\frac{Q_q^\top K_j}{\sqrt{d_h}}+b_{qj}^{\mathrm{geom}}+\log w_j\right),\\ \mathbf c_E(\mathbf x_q)=\Phi_E^{\mathrm{out}}\!\left(\sum_j a_{qj}V_j\right).\end{array}$$

$$\widehat{\mathbf y}_{\mathrm{norm}}(\mathbf x_q)=D_{\mathrm{modern}}(\mathbf x_q,\mathbf c_M,\mathbf c_E,\mathbf c_{\mathrm{coarse}},\mathbf c_{\mathrm{local}},\mathbf g).$$

The field heads read the fused main $(\mathbf c_M,\mathbf c_E)$, eight-token coarse, local and global contexts; their learned projections/gates are omitted here for readability. This preparation/read repeats within the classical phases. There is no learned hyperedge identity to interpret: fine source values, geometry and nonlinear contextual messages carry the interactions, and every valid source participates in the dense computation.

#### 3.1.5 HONF1502: sparse group controls over fine-source values

**Summary:** use twelve group slots to sparsify associations while keeping the Dense1804-style fine source states. Source memberships $B^M,B^E$ use entmax15, query routes $\alpha_q$ use sparsemax, and the final environment-membership refinement uses sparsemax in this literal 1502 configuration. These normalizers can produce exact zeros. Group controls $\mathbf t_k\in\mathbb R^{16}$ are learned from membership-weighted module/environment moments, with a final tanh; they are not the full-width physical source values. For either source type $s$:

$$\begin{array}{l}\rho_{qs}=\sum_{k=1}^{12}\alpha_{qk}B_{sk},\\ \boldsymbol\zeta_{qs}=\sum_{k=1}^{12}\alpha_{qk}B_{sk}\mathbf t_k.\end{array}$$

Here $\rho$ controls source access and $\boldsymbol\zeta$ supplies a source-specific control moment inside the nonlinear module message or environment score modulation. For normalized source measures $\omega_i^M=m_i/N_M$, define the module weighted sum and overlap mass:

$$\begin{array}{l}\mathbf U_q=\sum_i\omega_i^M\rho_{qi}\Psi_{QM}(\mathbf z'_i,\mathbf x_q-\mathbf c_i,\boldsymbol\zeta_{qi}),\\ \kappa_q^M=\sum_i\omega_i^M\rho_{qi},\\ \mathbf c_M(q)=12\left[\frac{N_M}{1+N_M}W_M\mathbf U_q+\kappa_q^M\mathbf b_M\right].\end{array}$$

For environment tokens, the reader adds $\log\rho_{qj}+\log\omega_j^E$ to the controlled content/geometry score and masks $\rho_{qj}=0$ before softmax. Its output is scaled by $12\kappa_q^E$, where $\kappa_q^E=\sum_j\omega_j^E\rho_{qj}$; unsupported rows return zero. The final field heads follow the $D_{\mathrm{modern}}$ structure above, with separately trained weights, fusing these controlled fine-source contexts with coarse/local/global paths and the classical wrapper. Consequently, a sparse group route is neither a low-rank temperature factorization nor evidence that execution skipped the corresponding rectangular tensor rows. The saved implementation measured here retains that rectangular work.

#### 3.1.6 R-Direct + D-sep: nonlinear layout preparation, linear heating application

**Summary:** remove the classical phases and construct two parallel predictors. The D-sep input excludes heating; its encoded geometry/operating context passes through two source–source updates, followed by a receiver-relative source read, a background branch and a four-channel decoder:

$$\begin{array}{l}\mathbf f_i^{(0)}=E_F(\mathbf c_i,\boldsymbol\mu,\mathbf b,m_i),\\ \mathbf f_i^{(r+1)}=m_i\!\left[\mathbf f_i^{(r)}+\Phi_F\!\left(\mathbf f_i^{(r)},\frac{\sum_{l\ne i}m_l\Psi_F(\mathbf f_i^{(r)},\mathbf f_l^{(r)},\mathbf c_i-\mathbf c_l)}{\max(N_M-1,1)}\right)\right],\\ r=0,1.\end{array}$$

$$[\hat u,\hat v,\hat p,\hat\omega]_{\mathrm{norm}}(\mathbf x)=D_F\!\left(\frac{\sum_i m_i\Psi_{QF}(\mathbf f_i^{(2)},\mathbf x-\mathbf c_i)}{\max(N_M,1)}+\mathbf c_{\mathrm{background}}(\mathbf x,\mathbf b),\mathbf x,\mathbf b\right).$$

The thermal branch independently encodes heating-free source/environment inputs, then performs two rounds of source–source, environment→source and source→environment contextual updates. Its near and far heads predict **one scalar coefficient per physical heater per grid receiver**, using these whole-layout states, global context and receiver-relative geometry:

$$\begin{array}{l}K_i(\mathbf x;\mathcal G)=\frac{m_i}{s_h}\left[\beta_i(\mathbf x)K_i^{\mathrm{near}}(\mathbf x;\mathcal G)+(1-\beta_i(\mathbf x))K_i^{\mathrm{far}}(\mathbf x;\mathcal G)\right],\\ \hat T_{\mathrm{grid}}(\mathbf x)=\sum_i K_i(\mathbf x;\mathcal G)h_i.\end{array}$$

Here $\mathcal G=(\{\mathbf c_i,m_i\},\boldsymbol\mu,\mathbf b)$ excludes heating, $s_h$ is the frozen forcing scale, and the smooth near weight $\beta_i$ equals one within two radii and zero beyond four radii. The benchmark's zero inlet/wall/initial temperature makes the offset zero; this is a capability restriction, not a claim that arbitrary boundary conditions have zero background. Native stencil extraction $\mathcal E_r$ turns the same learned grid into fluid, solid, surface and outside temperatures; q/effective-h are subsequent algebraic proxies:

$$\begin{array}{l}\hat T_r=\mathcal E_r\hat T_{\mathrm{grid}},\\ \hat q=-\frac{k_{\mathrm{harm}}}{\delta}(\hat T_{\mathrm{outside}}-\hat T_{\mathrm{surface}}),\\ \widehat{\mathbf y}(\mathbf x)=[\operatorname{denorm}(D_F)(\mathbf x),\mathcal E_{\mathrm{fluid}}\hat T_{\mathrm{grid}}(\mathbf x)].\end{array}$$

Flow is denormalized using its TRAIN statistics; thermal extraction is already in native temperature units after forcing-scale conversion. The thermal branch does not consume predicted flow in this compared model. At fixed $\mathcal G$, its temperature Jacobian is exactly the prepared kernel matrix, $\partial\hat T/\partial h_i=K_i$, giving $\Delta\hat T=K\Delta\mathbf h$. Geometry/material/operating changes require rebuilding K and the flow states. This exact model identity explains reusable heating responses and source-local covers; it does not guarantee that each learned column equals a true physical Green's function.

**Implementation anchors.** The classical phase map follows [ChannelThermalModel](../../Case_ThermalChannel/src/channelthermal/model.py); the organization/read equations follow [legacy organizer](../../src/honf_forward_core/organizer.py), [legacy decoder](../../src/honf_forward_core/decoder.py), [DensePairwiseField](../../src/honf_forward_core/interface_fields/dense_pairwise.py), [group-control router](../../src/honf_forward_core/interface_fields/group_control_router.py) and [group-control reader](../../src/honf_forward_core/interface_fields/group_control_pairwise.py). The composed response follows [D-sep adapter](../../Case_ThermalChannel/src/channelthermal/dependency_flow.py), [geometry flow reader](../../src/honf_forward_core/interface_fields/geometry_flow_field.py), [source-response operator](../../src/honf_forward_core/interface_fields/source_response_operator.py) and [native thermal adapter](../../Case_ThermalChannel/src/channelthermal/source_response.py). Configuration-specific slot counts, normalizers and active branches refer to the saved runs in Appendix B; these links locate implementation families, whose later options need not belong to those checkpoints.


### 3.2 Measured organization on the same physical layout

The saved 0291/M5 example preserves literal physical IDs M0–M4 and 7837 fluid receivers. Three input-defined tiles from the existing 4×4 partition contain 443/394/457 valid rows; selection uses geometry rather than residuals. Group diagrams describe the classic routing branch, with their other pairwise/coarse/local/refinement paths still active. Dense1804 provides a pairwise architecture reference; no latent-group membership is invented for it.

![Observed source, environment and receiver organization from saved four-model evidence on case0291](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_02_observed_organization.png)

**Figure 9 — Saved organization, 0291/M5.** All panels use actual saved diagnostics: 1401/1502 show five module-incidence rows, mean incidence over 192 environment tokens and three regional mean query-route rows; Dense1804 shows regional main/coarse/local context-norm fractions; R-Direct shows signed mean physical K. Context norms are neither contribution fractions nor sensitivities. Quantities/color scales are intentionally distinct, and full R-Direct prediction retains all five donors. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_02_observed_organization.pdf).

| Saved 0291 organization evidence | HONF1401 | HONF1502 | R-direct |
|---|---:|---:|---:|
| Nonzero real module → group memberships |30/30|27/60|N/A: no latent-group memberships|
| Nonzero environment → group memberships |1152/1152|399/2304|Context only; not independently actuated heating sources|
| Nonzero links on the common 7837 fluid receivers |47022/47022 group routes|26923/94044 group routes|39185/39185 physical kernel coefficients|
| Mean/range receiver support |6 /6–6 groups|3.43537 /2–6 groups|5 /5–5 physical sources|

For the near-source tile `py2_px1`, 1401’s largest query weight is on H4 (0.402), while 1502’s is on G5 (0.438). R-direct’s strongest RMS link is from physical heater 4 (4.38 temperature/heating units), larger than the link from heater 2 (3.48), whose center is closest to the tile centroid. This makes the model’s receiver-dependent donor structure directly readable while retaining the same physical heater IDs across regions.

The table counts exact saved zeros, without a plotting threshold. It shows that 1502 has sparser learned associations, while all twelve slots remain phase-occupied; its group 0 is never queried. Three groups have zero module membership but positive environment membership. These distinctions are visible in the graph and prevent “occupied,” “read,” and “module-bearing” from being treated as the same quantity. Probability rows sum to one, but the physical R-direct coefficients do not: among its 39185 fluid source coefficients, 16031 are negative. A negative coefficient describes the learned model’s signed heating sensitivity, not independently established physical cooling or causality.

For the three tiles above, the 1% TRAIN-box donor cover retains 4/4/5 sources respectively. It omits source 0 in the first two tiles with triangle bounds 0.000980622 and 0.001799245 temperature units, below the declared 0.004408216 budget; the third tile keeps all five and has zero omission. This selector uses `r_i RMS_P(K_i)`, so it reflects both receiver sensitivity and the allowed heating-change radius. Source 0’s radius is only 0.00833756 because its current heating lies near the TRAIN-box lower bound; a small cover score is not proof that this heater has a weak raw response everywhere. The covers can overlap and vary by receiver region, providing an intuitive local organization without inventing global learned groups.

![Spatial receiver organization and source support for the four models on fixed0291](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_02b_spatial_organization.png)

**Figure 10 — Spatial organization.** Saved e5000 diagnostics use the same 7837 fluid cells: dominant hyperedge for 1401, maximum environment-attention weight for 1804, positive query-group count for 1502 and argmax absolute physical-source K for R-Direct. Gray/blank cells exclude solids. The different colors encode different quantities, not a common causal influence score. Environment contextualizes R-Direct coefficients but is not an independently actuated heat column. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_02b_spatial_organization.pdf).

![Physical-source-to-receiver donor network with bounded overlapping local covers on case0291](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_02c_receiver_donors.png)

**Figure 11 — Receiver-local donor graph.** Three input-defined 0291 tiles show physical source-to-region links: width is regional RMS(K_i), signed labels report regional mean K_i in temperature/heating units, and solid/dashed styles distinguish retained/omitted sources in the coincident 1%/2% covers. The respective cover sizes are 4/4/5; full prediction still uses all five. The nearest-source tile is most responsive in RMS to M4 (4.38) rather than its nearest M2 (3.48). Cover scores also depend on the allowed input radius, so omitted M0 is not an independently validated irrelevant heater. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_02c_receiver_donors.pdf).

### 3.3 Local covers retain responses, not certified causal donors

At fixed geometry, `Delta T(x) = sum_i K_i(x) Delta h_i`. Each K_i depends on whole-layout source/source and source/environment context plus the near/far blend. The response therefore preserves source identity without implying geometrically independent modules. After geometry changes K must be rebuilt. One saved excitation direction per layout does not independently certify every coefficient column.

The bounded view uses `q_i(P) = r_i ||sqrt(W_P) K_i||_2` and the smallest descending-strength cover whose omitted strength sum is below the chosen fraction of sealed TRAIN response RMS 0.440821614. Its heating box 0.503670633–1.999153733 and radii are frozen from the predeclared 150-TRAIN calibration subset plus four TRAIN addenda; all IDs were verified inside the formal 600. This is a conservative reused TRAIN calibration, not a newly fitted full 600 response domain. Every measured model-omission triangle bound holds within the declared box. Equal-count nearest/upstream controls have their own measured bounds, which need not meet the learned selector’s requested epsilon.

Mean actual model omission RMS at 1% is 0.0000442085/0.00424039/0.00264733 for learned/nearest/upstream; at 2% it is 0.000118852/0.00430935/0.00283159. Averages include the secondary 0277 span and describe four layouts with one saved excitation direction each. They do not establish general grouping superiority, causal donor support or execution savings; no forced K or R-group retraining was performed.

The full endpoint audit covers sixteen patches per fixed layout, or 64 patch matrices. At 1%/2% budgets the mean retained source counts are 5.7031/5.5156, and 37/34 of 64 patches keep every donor. Across 112 saved patch-response rows per budget, learned/nearest/upstream covered physical-response RMSE is 0.0081411/0.0097130/0.0088265 at 1%, and 0.0081230/0.0096913/0.0089158 at 2%. These physical-reference errors remain separate from the much smaller model-omission errors. The result supports a useful local explanatory selector within its declared TRAIN box, with full-access fallback explicitly counted.

## 4. Responses to structural changes: geometry remains an unresolved limitation

### 4.1 Moving a module requires a new contextual response

A structural change moves an obstacle and changes the flow/thermal context. This is more demanding than applying new heating to a fixed K. All four models are cold-applied separately after geometry rebuilding. The comparison retains sixteen unique existing perturbations (four original-TRAIN anchors × four signed moves), native joins and common-fluid masks. These exposed saved states are neither newly solved references nor an unseen-layout generalization test.

![Four-model statistics over sixteen saved obstacle-move temperature responses](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_03_geometry_statistics.png)

**Figure 12 — Geometry-response statistics.** Mean fluid-temperature response RMSE is 0.135146/0.113314/0.126869/0.116515 for R-Direct/Dense1804/HONF1401/HONF1502. R-Direct is 19.3% worse than Dense1804 on the aggregate. Dots retain all sixteen (case, state) identities; correlated moves are not independent statistical samples. Direction/amplitude and absolute error are assessed separately; a favorable individual-state RMSE does not qualify geometry response. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_03_geometry_statistics.pdf).

![Saved TRAIN0348 i_plus structural temperature response and four endpoint predictions](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_04_geometry_response.png)

**Figure 13 — Structural response, TRAIN0348 i_plus.** The retained state moves M4 by +0.075 in x and uses 7456 baseline/state common fluid receivers. All signed predictions share one native-temperature scale; the stated signs refer to spatial means, not a certified physical sign law. The reference mean temperature change is +0.0881768, while R-Direct predicts −0.00300634: its mean sign and amplitude fail. The reference is a saved generator alternative, and geometry-specific coefficients are rebuilt rather than reusing a fixed-layout kernel. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_04_geometry_response.pdf).

![Four-model residuals for the saved TRAIN0348 i_plus structural response](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_05_geometry_residual.png)

**Figure 14 — Structural response residuals.** On the displayed move, R-Direct/Dense1804/HONF1401/HONF1502 RMSE is 0.181461/0.197101/0.210595/0.197835. R-Direct’s lower single-state RMSE coexists with a wrong mean direction; both evidence streams remain visible. Shared signed error scales enable direct spatial comparison. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_05_geometry_residual.pdf).

### 4.2 Heating response is a separate, more successful capability

Fixed-layout heating changes evaluate source amplitudes rather than structure. Three saved layouts contribute six primary plus/minus changes. Layout 0277 has no counted baseline, so only its secondary minus-to-plus span is used; the packed nominal baseline is never substituted. R-Direct improves fluid and surface response means against Dense1804 but is essentially tied on material response.

| Primary signed-response mean RMSE | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid temperature | 0.00884525 | 0.0105621 | 0.0196841 | 0.0137821 |
| Surface temperature | 0.0153821 | 0.0180586 | 0.0428163 | 0.0253747 |
| Normal flux proxy | 0.0554767 | 0.122495 | 0.201084 | 0.120676 |
| Material temperature | 0.0187677 | 0.0187927 | 0.0357857 | 0.0242 |

![Four-model saved heating-response error statistics](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/07_heat_response.png)

**Figure 15 — Heating-response accuracy.** Six primary signed states on 0291/0294/0687 give R-Direct 0.00884525 fluid, 0.0153821 surface and 0.0187677 material response RMSE. The corresponding Dense1804 values are 0.0105621/0.0180586/0.0187927. These are response changes against counted saved baselines, not nominal-field errors. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/07_heat_response.pdf).

![Four-model heating-response predictions for saved case0291 transfer_plus](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/08_heat_response_case0291.png)

**Figure 16 — Heating change, 0291 transfer_plus.** The saved change transfers 0.057515 heating units from literal M1 to M4 at fixed total heating. Reference mean fluid changes for minus/plus are −0.0289189/+0.0289138, while R-Direct gives −0.0402401/+0.0402401: a 39.2% amplitude excess remains despite improved aggregate response RMSE. The shared response scale and signed-mean annotations retain that amplitude failure. [PDF master](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/08_heat_response_case0291.pdf).

Heat-only u/v/p/vorticity increments are exactly zero under the prescribed heat-independent D-sep flow assumption. This is a model/reference assumption, not newly discovered buoyancy physics. R-direct’s material response is essentially tied with Dense1804; one excitation direction per layout does not validate every source column independently. No certified physical sign or discretization-error floor is available.

The endpoint precision audit covers four TRAIN anchors and four fixed response layouts. All 288 precise endpoint/increment role checks pass; maximum endpoint discrepancy is 1.42109e-14 and maximum native VJP discrepancy is 8.88178e-16. All 250 failing legacy FP32 endpoint checks remain recorded. Precise cold/prepared outputs agree and weights remain frozen; arithmetic consistency is distinct from physical accuracy. The additional four original-TRAIN heat-response families have mean temperature response RMSE 0.004052/0.003496/0.003865 for fluid/surface/material and 0.06898 for flux proxy; these are exposed calibration evidence, not a new held-out comparison.

### 4.3 Finite saved-pool inverse choices do not establish a design advantage

All four endpoints correctly choose `transfer_plus` for the primary 0291/0294/0687 pools and `transfer_minus` for secondary 0277, with zero observed saved-reference regret. This demonstrates reuse of existing finite alternatives, not successful continuous inverse optimization or a new certified design. Geometry-response direction and amplitude remain unqualified. No inverse/grouping superiority is required to retain R-Direct as a useful fixed-layout response reference.

## Appendix A. Numerical audit tables

### A.1 Canonical89 endpoint physical RMSE and paired evidence

Cells below are equal-case mean / case p90 physical RMSE on canonical89. Lower is better; units follow each stored native field. Module peaks are native per-module material maxima, then per-case RMSE over modules. No normalized training loss is used as a substitute for physical error.

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid u | 0.00432883 / 0.00494489 | 0.00650424 / 0.00835394 | 0.00960653 / 0.0122716 | 0.00970305 / 0.0122938 |
| Fluid v | 0.000459456 / 0.000515626 | 0.000331887 / 0.000460536 | 0.000428815 / 0.000589572 | 0.000531272 / 0.000743702 |
| Fluid p | 0.00197505 / 0.00258988 | 0.00224869 / 0.00359335 | 0.00416286 / 0.00697683 | 0.00423869 / 0.00617787 |
| Fluid vorticity | 0.0375592 / 0.0599751 | 0.0244341 / 0.053056 | 0.0260128 / 0.0425843 | 0.028128 / 0.0576348 |
| Fluid temperature | 0.342425 / 0.49042 | 0.237375 / 0.389084 | 0.312974 / 0.43949 | 0.267942 / 0.405377 |
| Surface temperature | 0.311198 / 0.705303 | 0.429475 / 0.557424 | 0.515004 / 0.639925 | 0.520884 / 0.666743 |
| Material temperature | 0.307594 / 0.716253 | 0.329348 / 0.43421 | 0.421362 / 0.575138 | 0.416504 / 0.543476 |
| Module material peak | 0.338021 / 0.832998 | 0.317573 / 0.530304 | 0.478912 / 0.795911 | 0.466868 / 0.683553 |

The material mean improvement over Dense1804 is small (6.6%) and its paired interval crosses zero. R-direct wins module-peak RMSE on 48/89 cases, yet its mean and p90 are worse because of large misses. Its worst module-peak case is 0287 (1.30422), worst material case is 0294 (0.944081), and worst surface/fluid-temperature case is 0295 (0.985360/0.870243). These failures remain in the numerical evidence and figures.

| Role | R-direct minus Dense mean RMSE | Paired 95% interval | R-direct wins / 89 |
|---|---:|---:|---:|
| Fluid u | -0.00217541 | [-0.00247266, -0.00185821] | 84 |
| Fluid temperature | 0.105049 | [0.0893296, 0.121076] | 6 |
| Surface temperature | -0.118277 | [-0.156624, -0.0805346] | 74 |
| Material temperature | -0.0217544 | [-0.0592276, 0.0154034] | 66 |
| Module material peak | 0.0204473 | [-0.0332275, 0.0768823] | 48 |
| Near vorticity | 0.0365396 | [0.0294168, 0.0432492] | 9 |

Intervals use 10,000 paired case resamples with seed 0. They describe one exposed panel, not uncertainty across training seeds or a causal architecture effect; no multiple-comparison adjustment is implied.

### A.2 Near/far fields and native ports

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Near u | 0.00709583 / 0.0101229 | 0.0077493 / 0.0102006 | 0.00862538 / 0.0112242 | 0.00985822 / 0.0145282 |
| Near v | 0.000711308 / 0.00080918 | 0.000556422 / 0.000765985 | 0.000825787 / 0.00125774 | 0.000930896 / 0.00120486 |
| Near p | 0.00322881 / 0.00505866 | 0.00295946 / 0.00506513 | 0.00502164 / 0.00750185 | 0.00562898 / 0.00774446 |
| Near omega | 0.0889746 / 0.140299 | 0.052435 / 0.111844 | 0.0503735 / 0.0943597 | 0.0610032 / 0.133315 |
| Near temperature | 0.322688 / 0.652558 | 0.272913 / 0.44262 | 0.32214 / 0.468802 | 0.288976 / 0.513653 |
| Far u | 0.00347158 / 0.00390372 | 0.00621897 / 0.0081149 | 0.00971469 / 0.0120168 | 0.00959339 / 0.0124043 |
| Far v | 0.000401012 / 0.000445395 | 0.000266325 / 0.000348165 | 0.000302075 / 0.000416664 | 0.000411112 / 0.000527492 |
| Far p | 0.0015958 / 0.00185837 | 0.00207149 / 0.00310254 | 0.0039744 / 0.00675726 | 0.0039055 / 0.00567716 |
| Far omega | 0.0133834 / 0.0152836 | 0.0113225 / 0.0141754 | 0.0168142 / 0.0228455 | 0.0129732 / 0.0162815 |
| Far temperature | 0.344597 / 0.476561 | 0.228885 / 0.374928 | 0.311169 / 0.450316 | 0.262362 / 0.394255 |

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Normal flux proxy | 1.43326 / 2.36954 | 1.53376 / 2.39451 | 2.015 / 2.63486 | 1.68273 / 2.2754 |
| Final outside temperature | 0.300105 / 0.682245 | 0.85156 / 1.03613 | 0.751084 / 0.927114 | 0.74174 / 0.937015 |
| Final effective h | 0.0109522 / 0.000113759 | 0.59063 / 0.857544 | 0.650998 / 0.913205 | 0.547675 / 0.80066 |
| Initial outside temperature | N/A | 0.767828 / 0.965704 | 2.51877 / 3.63974 | 0.748176 / 0.972072 |
| Initial effective h | N/A | 2.58072 / 2.859 | 7.58199 / 7.67967 | 0.533734 / 0.695866 |
| Inlet–outlet pressure difference | 0.00132857 / 0.0024923 | 0.00102587 / 0.00210062 | 0.00301812 / 0.00545861 | 0.00151789 / 0.00325609 |

Near-boundary vorticity is a substantial R-direct miss. Its frozen ordinary D-sep flow was chosen in the earlier matched refinement pair, so this endpoint comparison does not turn the rejected near-boundary refinement into a selected model. Initial-port history is not applicable to R-direct and is left N/A.

Normal flux and effective h are native proxies. The R-direct construction uses `q = −harmonic(k)/delta × (T_outside − T_surface)`, making `h = q/(T_surface − T_outside)` largely equal to `harmonic(k)/delta` away from clamps. Its tiny median/tail h errors therefore do not independently demonstrate learned heat-transfer accuracy. The h mean is driven by outliers, including case 0282 (0.748986). These values do not certify continuum boundary flux.

### A.3 Relative error, module-count strata and original90 compatibility

| Fluid channel | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| u | 0.00562176 | 0.00817124 | 0.0120969 | 0.0122766 |
| v | 0.0176551 | 0.0131344 | 0.0173283 | 0.0211146 |
| p | 0.0197704 | 0.0239279 | 0.0460201 | 0.0423585 |
| omega | 0.0592198 | 0.0420716 | 0.04136 | 0.0474568 |
| temperature | 0.0445331 | 0.0324413 | 0.0416466 | 0.0369812 |

These are pooled per-channel physical relative L2 values, using independently recomputed raw-reference energies and matched error sums. Combining the five channels into one Euclidean score mixes units and is retained only as legacy compatibility evidence in the linked numerical artifact, not as a model selector.

| Module count / cases | Fluid T mean: R / 1804 / 1401 / 1502 | Peak mean: R / 1804 / 1401 / 1502 |
|---|---|---|
| M3 / 24 | 0.229433 / 0.138517 / 0.203121 / 0.173835 | 0.172608 / 0.221684 / 0.284585 / 0.321997 |
| M5 / 25 | 0.333088 / 0.241382 / 0.288362 / 0.280522 | 0.379199 / 0.276853 / 0.389138 / 0.43043 |
| M7 / 25 | 0.440736 / 0.304683 / 0.41303 / 0.356227 | 0.502733 / 0.449585 / 0.667972 / 0.589891 |
| M10 / 15 | 0.374921 / 0.276692 / 0.363001 / 0.250401 | 0.259529 / 0.318843 / 0.62436 / 0.554356 |

Original90 compatibility means, including duplicate 0273:

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid temperature | 0.341547 | 0.236197 | 0.311722 | 0.266698 |
| Surface temperature | 0.309506 | 0.429352 | 0.514088 | 0.520344 |
| Material temperature | 0.305823 | 0.328713 | 0.419776 | 0.415278 |
| Module material peak | 0.335927 | 0.316299 | 0.47553 | 0.46461 |

### A.4 Saved-best states, kept separate

| Physical row | R-direct | Dense1804 | HONF1401 | HONF1502 |
|---|---:|---:|---:|---:|
| Fluid temperature | 0.34613 / 0.49669 | 0.218656 / 0.369171 | 0.284443 / 0.425218 | 0.250088 / 0.394572 |
| Near temperature | 0.325044 / 0.642304 | 0.249452 / 0.417311 | 0.305512 / 0.443519 | 0.27201 / 0.496213 |
| Far temperature | 0.348651 / 0.487839 | 0.212036 / 0.352819 | 0.280375 / 0.427392 | 0.244487 / 0.367545 |
| Surface temperature | 0.312285 / 0.690755 | 0.448389 / 0.602693 | 0.458676 / 0.589466 | 0.4587 / 0.631604 |
| Material temperature | 0.308358 / 0.727054 | 0.340027 / 0.43283 | 0.356474 / 0.479992 | 0.35276 / 0.485158 |
| Module material peak | 0.335256 / 0.842339 | 0.346586 / 0.58687 | 0.359877 / 0.585881 | 0.358339 / 0.612205 |

R-direct uses thermal e4200 plus frozen flow e5000; Dense1804/HONF1401/HONF1502 use e4738/e4585/e4794 respectively. Endpoint e5000 slightly improves R-direct’s full-grid fluid, surface and material means relative to e4200, while peak mean slightly worsens. The different monitoring selectors and checkpoint ages remain explicit.

## Appendix B. Exact checkpoint and dataset identities

The classics retain their legacy normalizer hash `5cbe6473e6449e403626520e63ad1ca1cf283cd802d257d904c42e877c4f9129`, distinct from the fresh formal normalizer. Saved configs refer to the same packed source dataset but do not carry the new ordered TRAIN membership seal. These differences remain part of the comparison contract.

| Model | Literal endpoint | Saved best |
|---|---|---|
| R-direct | e5000; `5aafc23cb86ea4d02706cda5749e97e67e42886328e1e3758148138e8736f2e5`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/RDirect_e5000/summary.json) | e4200; `28d19f6d5dbdf8f104363d70b2f19a8bb45df8ebf9be2c635bbdc74b2283ddc1`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/r_direct_best_e4200_original90_retry2/summary.json) |
| Run1804 | e5000; `9d0b83c562cecc2ffc52c3a08c993dfa8f47ae0e769f54ed6ffdd966047d63d9`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1804_e5000_latest/summary.json) | e4738; `71ed480ff0396813491c650dd11d887195174019b373fbd9a1fb25505142c066`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1804/summary.json) |
| Run1401 | e5000; `cee978f0461db928b72c3b6b66cb0ef56647674f82d6ef2660a8380754a829d6`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1401_e5000_latest/summary.json) | e4585; `5be150bd6b4fc79599af62c767fba84490ba50edcc8cc8ce85026ae27a1846b3`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1401/summary.json) |
| Run1502 | e5000; `20af85200796f639039d4853425fe91ee76e19e7caf5b5b68863badfe7690895`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1502_e5000_latest/summary.json) | e4794; `08d86a573c7f7d86463bde231eb9f84a745602fca2b8f97142f39540d33a85bb`; [receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/classic_fields/Run1502/summary.json) |

Run1401’s literal endpoint is the run-root `epoch_5000_model.pt`, not `checkpoints/latest.pt`; its comparison identifier `Run1401_e5000_latest` does not change that path. Run1804 and Run1502 endpoints are their SHA-bound `checkpoints/latest.pt` files. R-direct’s thermal saved-best alias contains actual epoch 4,200, with the same frozen flow e5000 and exact formal memberships, schedule and normalization bindings as its endpoint. There is no standalone e4200 milestone. The sampled-monitor selector (0.002256388845984311 at e4200) remains intact; full-grid evaluation is reported separately and does not relabel the saved-best alias.

Formal TRAIN-ID SHA256 is `c991acbcdced62e887385bc2763d4da7f4686256f72e44f797b69185522f3e3e`; normalization binding is `b06a65cc1e84176badea9b77431ae5f751ea9672295afd3aad883d0ed61abd86`; validation binding is `64a183f1aecb3672f284617009697e46ac3b32d848f73b321c3cac6de5b2a346`; source metadata SHA256 is `1bba5ab5c0535fabab2f2434de33eccb67184ef3fd7d7881eb01f5088e15e211`. Development remains fixed25_v1 150/22. Neither development resume checks nor the literal-e5000 formal CLI guard were weakened.

## Appendix C. Evidence, retained failures and revision scope

This presentation revision reads saved evidence on CPU and creates local figures; it does not launch model calls, GPU work, optimizer updates or reference solves. The original solver ledger remains 326/326. Old raw arrays, checkpoints, PDF masters and failed receipts remain untouched. Newly selected PDF pages are the retained presentation masters; small PNG companions are present because this Markdown report embeds figures directly. All generated figures and one-time renderers stay in ignored local directories and are excluded from Git.

Read-only checks passed 61 relevant tests covering formal profiles, strict loaders, formal evaluation, the classic bridge, flow dependency and training identity. The preceding comparison’s code change added only exact age/SHA-whitelisted Run1401 identities and rejection tests; existing development/formal checks remain strict. Eighty physical fluid-field RMSE checks were independently recomputed in float64 from the saved fixed-four arrays. The duplicate R-direct endpoint pass on the other authorized GPU matched all 90 metric dictionaries exactly; that repeated execution is recorded, not additional independent scientific evidence. All evaluated frozen states remained unchanged. Those numerical checks belong to the preceding completed-model assessment, not fresh executions during this presentation revision. This revision independently reconstructs plotted statistics and verifies saved-array joins; all sixteen presentation pages were visually inspected. The browser preview loads all sixteen images and fifteen tables without horizontal page overflow at 1280px and 900px. All 63 local links in this report and 39 links in the historical archive resolve; prose paragraphs and captions remain on one source line. The [presentation export receipt](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/export_validation.json) and [accuracy/efficiency](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/accuracy_efficiency_manifest.json), [organization/geometry](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/org_geo_manifest.json) and [heating-response](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/ppt_comparison_20261007/heat_response_manifest.json) source manifests record numerical checks and provenance. The PPTX embeds each figure as an image; the standalone PDFs retain vector text and chart geometry where applicable.

An initial legacy 1401 benchmark attempt failed before completion because its backend lacks the modern executor recorder; a retry measures common linear-layer rows and labels unsupported modern work as N/A. Two selected-best preflights failed on array equality and an assumed nonexistent e4200 milestone; both receipts remain, and the successful evaluation uses the actual best alias with exact endpoint bindings. The maintained formal CLI still requires literal e5000. The local selected-best wrapper preserves actual e4200 metadata and verifies the sealed endpoint identities rather than weakening that guard. Historical FP32 failures, source checkpoints, raw arrays and PDF masters are preserved.

The compact artifact index and native summaries make the result reproducible without another fit: [figure index](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/figure_and_metrics_index.json), [endpoint table](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/endpoint_native_metrics_table.json), [saved-best table](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/figures/final/selected_best_native_metrics_table.json), [identity audit](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/identity_audit.json), [normalization audit](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/normalization_audit.json), [relative physical errors](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/physical_relative_l2.json), [independent array checks](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/independent_physical_checks.json), [counted heat replay](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/RDirect_e5000_counted/summary.json), [precision audit](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/r_direct_e5000_precision/summary.json), [receiver-local summary](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/flow_pair/receiver_local_formal_e5000/summary.json), [geometry replay](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/geometry/summary.json), [saved-pool comparison](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/stored_pool_comparison.json), [native timing](../../diagnostics/generated/formal5000_launch_20261007/evaluation/comparison_20261007/native_cost_retry1/receipt.json). The [maintained evaluator](../../tools/thermal_source_response_evaluate.py) supports the exact endpoint identities above with `--mode classic-fields --formal-panel original90 --formal-reference-checkpoint <Run3902 epoch_5000_model.pt>` and the corresponding classic identifier; R-direct uses `--mode fields --formal-panel original90`. Use a fresh output directory, the ModularDT environment and each literal checkpoint path from the linked summaries. The local one-time comparison scripts remain beside these artifacts; they are not uploaded. The [formal recipe](../guides/Thermal_RDirect_Formal5000.md) remains the lineage/configuration reference, while the measured completion and costs above supersede its earlier launch forecast.
