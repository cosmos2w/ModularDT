# HET-MF datasets and HONF training readiness

**Review date:** 18 October 2026. **Case:** `hetmf_contact_sheet_v1`. **Scope:** code review, physical and storage diagnostics, complete Standard and Expanded processed H5 archives, and a tested connection to the general HONF model. **Training performed:** zero optimizer updates.

The case now has two self-contained H5 files that a later training program can open directly: [Standard](Dataset/standard.h5), containing 4,608 physical states, and [Expanded](Dataset/expanded.h5), containing 9,216. Each archive contains the prescribed physical inputs, five target fields, explicit training/development/test memberships, source provenance, and normalization fitted on ordinary TRAIN only. Expanded preserves every Standard input and split unchanged. Its additional states were solved with the independent CPU FP64 reference while Round-10 continued using the GPUs. Unchanged Standard source fields are shared as immutable hardlinks with their original solver and approval identities.

## 1. The physical problem in plain language

Imagine a thin square sheet carrying electrical current. Several round contacts connect the sheet to external voltage terminals. The material may conduct electricity easily in one region and resist it in another. It may spread heat well along one path and poorly along another, and cooling strength can vary across the sheet. Given the material maps and the contacts, the reference calculates where current flows, where electrical energy becomes heat, and how hot the sheet becomes above ambient.

The contacts are the physical modules in this case. A module is described by its location, radius, total contact conductance, and applied terminal voltage. Its internal electrical or thermal state is not resolved. A temperature value underneath a contact is the temperature of the sheet. The environment is fully given to the predictor through 144 material-control records; it is not a hidden field that the model must infer from incomplete measurements.

All coordinates and physical quantities use fixed dimensionless reference units. The square occupies $\Omega=[0,1]^2$. These results have not been calibrated to volts, amperes, watts, or kelvin for a particular device. The model includes steady electrical conduction, finite contact resistance, Joule heating, in-plane thermal conduction, and distributed cooling. It does not include electrochemistry, fluid flow, temperature-dependent conductance, or a time history.

![Given electrical, thermal, and cooling fields with the prescribed contacts](Dataset/readiness/figures/given_inputs.png)

Figure 1 shows the actual given input fields for the first input-selected Standard example, reconstructed on the 96 × 96 grid from 12 × 12 material controls. White circles mark finite-area contacts, and terminal voltages are labelled on the electrical map. Each material channel has its own dimensionless color scale. The material controls are interpolated samples, rather than 144 discontinuous physical blocks. [PDF master](Dataset/readiness/figures/given_inputs.pdf).

## 2. What is supplied and what must be predicted

| Entity | Given information | Physical meaning |
|---|---|---|
| Contact $i$ | $(x_i,y_i,r_i,\gamma_i,u_i)$ | Center, radius, total connection conductance, and external terminal voltage |
| Material record $e$ | Position, spacing, sampling area, $s_e,\kappa_e,\beta_e$ | Electrical sheet conductance, thermal sheet conductance, and distributed cooling |
| Query | $(x,y)$ or an exact native cell ID | A location where the field should be predicted |
| Model output | $(v,j_x,j_y,\theta,q)$ | Potential, two sheet-current components, temperature rise, and total Joule heat density |

The number of contacts varies. Ordinary core cases use $M\in\{2,3,4,6,8,12\}$; supplemental count challenges include $M=10$ and $M=16$. The contact radius lies in the prescribed finite range, so a contact covers a resolved area rather than a single grid point. The parameter $\gamma_i$ is total conductance and remains independent of radius: enlarging a contact at fixed $\gamma_i$ changes spreading geometry without automatically increasing total conductance.

The material inputs vary through correlated mosaics, layered barriers, inclusions, and conductive corridors. These shapes can reroute current and change heat spreading. Electrical and thermal material fields need not have the same pattern. The predictor receives their numerical records; family names, random seeds, solver errors, split labels, solved terminal currents, and target-derived hot spots are kept in provenance or diagnostics and excluded from model features.

| Target channel | Meaning and interpretation |
|---|---|
| `potential` ($v$) | Electrical potential on the sheet; its spatial differences drive current |
| `jx_cell` ($j_x$) | Reconstructed sheet-current density in the positive horizontal direction |
| `jy_cell` ($j_y$) | Reconstructed sheet-current density in the positive vertical direction |
| `temperature_rise` ($\theta$) | Sheet temperature above the reference ambient level |
| `q_total` ($q$) | Areal heat generation from sheet resistance plus finite contact loss |

These five channels describe two coupled steady PDE solutions and their derived quantities. They are not five independent physical systems. The raw reference additionally retains sheet/contact heating separately, electrical and thermal fluxes on native cell faces, and terminal currents. The convenient cell-centered current channels are suitable targets and displays; conservation is checked using the native face quantities.

## 3. Equations and the direction of physical coupling

Each contact has a smooth finite-area density $\chi_i(x)$ with $\int_\Omega\chi_i\,dx=1$. Its profile is a normalized quartic disk. The electrical problem is

$$
-\nabla\cdot(s\nabla v)+\sum_{i=1}^{M}\gamma_i\chi_i(v-u_i)=0,
\qquad \mathbf j=-s\nabla v.
$$

The contact voltage belongs to the external terminal. The sheet potential under that contact is solved through a finite connection resistance; the code does not also impose an ideal-voltage boundary on the same pad. A contact can deliver current to the sheet or receive current from it. For intermediate terminal voltages, the sign follows from the complete solved system.

The continuum heat law and thermal equation are

$$
q=q_{\mathrm{sheet}}+q_{\mathrm{contact}},\qquad
q_{\mathrm{sheet}}=s|\nabla v|^2,\qquad
q_{\mathrm{contact}}=\sum_i\gamma_i\chi_i(v-u_i)^2,
$$

$$
-\nabla\cdot(\kappa\nabla\theta)+\beta\theta=q.
$$

The electrical equation is solved first. Its electrical energy loss becomes the thermal source, and the thermal equation is solved second. The side boundaries have zero normal electrical and thermal flux. Distributed cooling removes heat through the $\beta\theta$ term. This is one-way coupling: changing $\kappa$ or $\beta$ alone changes the thermal response while preserving the electrical solution.

At fixed geometry and material properties, potential and current depend linearly on terminal voltages, while heat and temperature depend quadratically. This is why the HONF adapter keeps voltage as a prescribed input feature and uses the general nonlinear output head. The existing affine thermal-control head would not represent all five electrothermal outputs correctly.

The finite-volume reference forms heating from the unique-face electrical energy,

$$
P_f=T_f^s(v_p-v_r)^2,
$$

and apportions each face loss to its two neighboring half-cells according to their electrical resistances. It does not square an averaged cell current to manufacture the reference heat source. This preserves electrical energy at material transitions. All contact loss is deposited in the sheet once, as a declared version-1 modeling assumption.

Terminal currents and the principal global checks are

$$
I_i=\gamma_i\int_\Omega\chi_i(u_i-v)\,dx,\qquad
\sum_i I_i=0,
$$

$$
\sum_i u_iI_i=\int_\Omega q\,dx,\qquad
\int_\Omega\beta\theta\,dx=\int_\Omega q\,dx.
$$

The revised audit also reconstructs local electrical/thermal residuals and checks stored face currents, heat fluxes, separate heat contributions, and terminal-current vectors against the same physical inputs. A small global balance error alone is insufficient to prove local field accuracy.

## 4. Dataset membership and experimental meaning

| Membership | Standard | Expanded | Intended use |
|---|---:|---:|---|
| Ordinary TRAIN | 2,880 | 5,760 | Fit predictor weights and any fitted normalization |
| DEV | 480 | 960 | Exposed model-development checks |
| Joint-unseen TEST | 480 | 960 | Final evaluation with both environment and layout roots held out |
| New environment / known layout | 128 | 256 | Environmental transfer |
| Known environment / new layout | 128 | 256 | Layout transfer |
| New pair / known components | 128 | 256 | Recombine familiar components in unseen pairs |
| Extrapolation/interpolation panel | 128 | 256 | M10, M16, higher electrical contrast, and composite morphology |
| Counterfactual audit | 256 | 512 | 64 or 128 four-state controlled comparisons |
| **Total** | **4,608** | **9,216** | Distinct physical input states within each profile |

Core TRAIN, DEV, TEST, and audit roots are separated before pairing. The code now verifies actual memberships, environment/layout root separation, physical-input uniqueness, panel component-exposure flags, new-pair exclusion, and complete quartet lineage. The supplemental panels deliberately reuse some TRAIN components; their labels identify that exposure. Their targets must not be used to fit normalization or model weights in the declared core protocol.

Standard contains 240 ordinary TRAIN environment roots and 480 TRAIN layout roots. Each environment appears with twelve layouts, and each layout appears with six environments. Expanded has 480 TRAIN environment roots and 960 TRAIN layout roots with the same reuse counts. These counts matter: the number of rows is larger than the number of independent material realizations. Across all splits and panels, Standard has 544 environment roots and 992 layout roots; Expanded has 1,088 and 1,984.

**Measured design limitation.** The frozen TRAIN pairing places every individual layout with environments from only one morphology family, even though that layout sees six distinct environments. Thus layout identity and morphology exposure are partly associated in TRAIN. DEV and TEST layouts see broader family exposure. The review preserves the frozen Standard and Expanded input hashes and records this limitation explicitly. A future claim of training on independently varied morphology for each layout requires a prospectively revised pairing recipe and a new dataset identity; the larger Expanded row count does not remove this association.

Expanded is an exact physical-input superset of Standard, with the same split for every shared state. It is suitable for a declared data-size comparison, but the two archives are not independent experimental populations. A checkpoint must remain bound to its exact profile, membership, and normalization. Fitting on Expanded and evaluating shared Standard cases cannot be described as evaluation on an independent dataset.

## 5. Files and standardized H5 layout

The browsing paths [Dataset/standard.h5](Dataset/standard.h5), [Dataset/expanded.h5](Dataset/expanded.h5), [Dataset/formal_standard](Dataset/formal_standard), [Dataset/formal_expanded](Dataset/formal_expanded), and [Dataset/readiness](Dataset/readiness) can be connected to the public bundle with `python -m hetmf link-shared --bundle /data/public/ModularDT-HETMF-Dataset/v1 --case-dir .`. In that distribution, the formal-release links expose selected receipts and figures; the protected original FP64 arrays remain in the owner archive on `/data2`. The real `Dataset` directory retains the versioned schemas and documentation. The H5 files themselves are standalone files, with no virtual datasets or external H5 links.

| H5 path | Shape or contents | Role |
|---|---|---|
| `inputs/environment_records` | `[N,144,8]`, float64 | Columns: center x/y, spacing x/y, sampling area, s, kappa, beta |
| `inputs/contacts` | `[N,16,5]`, float64 | Columns: x, y, radius, total gamma, terminal voltage |
| `inputs/contact_mask` | `[N,16]`, boolean | Marks real contacts; padded records are zero |
| `targets/fields` | `[N,96,96,5]`, float32 | Physical dimensionless targets in the fixed five-channel order |
| `queries/xy` | `[9216,2]`, float64 | Native cell-center coordinates, with x varying fastest |
| `queries/cell_area` | `[9216]`, float64 | Finite-volume cell area, 1/9216 |
| `splits/<name>` | Integer archive row indices | Complete explicit partition into the eight memberships above |
| `metadata/` | State/root IDs, physical and solver hashes, challenge/quartet membership | Provenance excluded from predictor features |
| `provenance/` | Frozen plan/approval, original contact IDs, source audits and inherited lineage | Reconstruction and integrity evidence |
| `normalization/` | FP64 means, population standard deviations, counts, TRAIN membership fingerprint | Optional loader-side target transformation |
| `integrity/` | Per-case checksums; archive payload seal stored in attributes | Detect changes to numerical arrays or archive metadata |

The ordinary per-case `processed/` folder in a formal release is not the training archive. Legacy Standard `processed_shards/` remains retained evidence. The standardized training interface opens the two single H5 files directly. All canonical input values are preserved; the stored targets are not normalized. The loader can apply $z=(F-\mu_{\mathrm{TRAIN}})/\sigma_{\mathrm{TRAIN}}$ on read, and the corresponding physical prediction is $F=\mu_{\mathrm{TRAIN}}+\sigma_{\mathrm{TRAIN}}z$. Constant columns use neutral scale one. The archive stores input moments for inspection, while the adapter uses fixed logarithmic transforms of positive conductances rather than fitted target information.

| Archive | Actual bytes | Physical states | Payload SHA-256 |
|---|---:|---:|---|
| Standard | 891,914,636 | 4,608 | `8ae7dfb62e76ed1654910131fa8db23853beac351bae5b694b3e7062d100399f` |
| Expanded | 1,786,766,567 | 9,216 | `245dd9fa8852e0abb01b69096b54137e462123a4b7d28bc439eb9a140cae7850` |

The input-plan hashes remain Standard `a94ddd9e0374fb9a971f4856944b2f8cc475749db6b2f797cddccbd3afa29e6b` and Expanded `f7f25efd5194cdec7dcc16d94ada30c2d92027e4334a21928e5531935df2ae99`. The exporter verifies every accepted source checksum and physical audit, computes the canonical target conversion, seals the complete archive, independently verifies its contents and TRAIN statistics, and publishes the result by atomic rename. The full verifier detects altered fields, input columns, query measures, split labels, provenance, and normalization metadata rather than trusting an accepted flag alone.

## 6. Representative saved physical fields

The following six examples were selected from input metadata before target values were read: four ordinary Standard morphologies at different contact counts and two new Expanded challenge cases. White contact outlines show geometry; each panel has its own color range. Brightness should not be compared across panels without reading the colorbar. All fields are physical reference solutions with dimensionless units; there is no trained checkpoint in these figures.

![Standard correlated mosaic M2 and layered barrier M6 reference fields](Dataset/readiness/figures/representatives_1.png)

Figure 2 compares the first input-selected correlated-mosaic M2 case with the first layered-barrier M6 case. Columns show the given electrical conductance, solved potential, reconstructed current magnitude, total heating, and temperature rise. `formal:train:e0000:m2:l0` has peak temperature 0.171, peak heat density 12.96, and contact-loss share 35.8%; `formal:train:e0001:m6:l0` has peak temperature 0.3272, peak heat density 45.61, and contact-loss share 46.0%. [PDF master](Dataset/readiness/figures/representatives_1.pdf).

![Standard inclusion M8 and corridor M12 reference fields](Dataset/readiness/figures/representatives_2.png)

Figure 3 shows inclusion and corridor environments with M8 and M12 contacts. Conducting paths can carry current away from a straight line between contacts, while the thermal field spreads through its own conductivity and cooling maps. `formal:train:e0003:m8:l0` has peak temperature 0.4351, peak heat density 34.59, and contact-loss share 64.1%; `formal:train:e0004:m12:l0` has peak temperature 0.4696, peak heat density 18.85, and contact-loss share 34.1%. [PDF master](Dataset/readiness/figures/representatives_2.pdf).

![New Expanded M16 and composite morphology reference fields](Dataset/readiness/figures/representatives_3.png)

Figure 4 shows an M16 challenge and a composite-morphology challenge newly introduced by Expanded. These cases have explicit panel membership rather than ordinary TRAIN membership. `formal:panel:extrapolation:m16:0032` has peak temperature 2.476, peak heat density 126.7, and contact-loss share 50.7%; `formal:panel:extrapolation:composite:0032` has peak temperature 0.2937, peak heat density 59.86, and contact-loss share 60.9%. [PDF master](Dataset/readiness/figures/representatives_3.pdf).

![Same-input 96 to 192 grid refinement for all six selected examples](Dataset/readiness/figures/refinement_audit.png)

Figure 5 compares the exact same physical inputs on 96² and 192² grids using conservative cell averaging and native face restriction. These errors measure spatial-discretization discrepancy, which is different from a linear solver residual. For these six cases, maximum relative L2 discrepancies are 0.124% for potential, 0.088% for temperature, 0.277% for total heating, 0.229% for either native current component, and 0.117% for terminal currents; maximum peak-temperature discrepancy is 0.157%. The fourteen retained refinements also include all eight endpoints of the two quartets. They qualify this fixed panel, rather than every possible input or a grid-independent continuum solution. [PDF master](Dataset/readiness/figures/refinement_audit.pdf).

## 7. Modular structure for later predictor, organizer, and inverse experiments

A future model receives two distinct source sets: the contacts $M=\{m_i\}$ and environmental records $E=\{e_j\}$. A query $x$ asks for five values $F(x)=(v,j_x,j_y,\theta,q)$. A modular predictor can encode each source, organize relationships relevant to that query, and decode the local field. Useful relationships include contact–environment effects, redistribution between contacts, and joint effects of multiple environmental regions. Electrical and thermal pathways may differ, so their organization should be checked per field.

The current adapter connects these prescribed sources to the real general `SceneBatch` and `QueryBatch` contracts. Its module features are $(r,\log\gamma,u)$, with contact coordinates supplied separately. Its environment features are $(\log s,\log\kappa,\log\beta)$, with control coordinates and sampling measures supplied separately. It uses two coordinate dimensions and five output channels. Variable contact counts have explicit masks; ordering IDs preserve correspondence and are not learned numeric embeddings. Environmental measure is 1/144 per control record. Contact area $\pi r^2$ is the declared model pooling measure, while $\gamma$ remains the independent physical total conductance.

The tested model is the general full-access A2-D core, with affine thermal response disabled. Its local window of three contact radii is a model detail window, not a boundary on the actual physical influence of a contact. A subsequent H/S organizer can be designed and compared through the same physical input contract. Its memberships, attention, exact-zero support, and executed work will require direct measurement. The data generator supplies no learned graph, sparse causal edge labels, or ground-truth hyperedge memberships; elliptic conduction can produce dense long-range influence.

Counterfactual quartets provide a physical test of joint response. For two controlled input changes A and B, the reference stores $F_{00},F_{10},F_{01},F_{11}$ and computes

$$
\mathcal I_{AB}=F_{11}-F_{10}-F_{01}+F_{00}.
$$

This asks whether applying both changes differs from adding their separate effects. Standard has 32 contact–environment quartets, 16 voltage–voltage contact quartets, and 16 two-patch electrical-environment quartets. Expanded doubles those counts. The broader six-type intervention catalogue belongs to the earlier pilot; it should not be described as six formally sampled intervention families in these releases.

![Actual contact displacement and electrical material-patch counterfactual responses](Dataset/readiness/figures/counterfactual_me_patch_move.png)

Figure 6 shows a contact displacement and a separate 15% electrical-control patch change. Columns plot the isolated changes and their joint interaction for potential and temperature. All four endpoints are solved at both 96² and 192² to assess the interaction against its own grid discrepancy. For `formal:audit:q0000`, maximum interaction magnitudes are 0.000365 for potential and 0.000314 for temperature. Their interaction norms are 299.6 and 292.5 times their own 96²-versus-192² discrepancies, respectively, supporting a resolved non-additive response on this panel. [PDF master](Dataset/readiness/figures/counterfactual_me_patch_move.pdf).

![Actual two-terminal-voltage counterfactual responses](Dataset/readiness/figures/counterfactual_mm_voltages.png)

Figure 7 shows two independent voltage changes with all other inputs fixed. The potential interaction should be consistent with zero because the fixed-geometry electrical problem is linear in voltage; heat and temperature can have nonzero pair interactions because Joule heating is quadratic. For `formal:audit:q0002`, voltage increments are +0.075 and -0.06. The potential interaction peaks at 5.07e-11, consistent with numerical error, while the temperature interaction peaks at 0.00155 and has norm 791.1 times its own grid discrepancy. The potential null is plotted on the scale of its isolated voltage changes to avoid magnifying solver noise. [PDF master](Dataset/readiness/figures/counterfactual_mm_voltages.pdf).

An inverse experiment would choose legal contact positions, radii, conductances, voltages, or explicitly authorized material controls to meet a requested field or temperature objective. Minimizing temperature alone admits turning the electrical device off, so a useful design problem must retain a specified current, power, voltage range, or receiver service. The present NumPy archive adapter supports predictor gradients through model weights; it does not expose differentiable design tensors. A later inverse adapter and fresh reference-solver checks are still required before candidate designs can be called valid.

## 8. Review findings, corrections, and limits

| Review finding | Correction or retained limitation | Verification |
|---|---|---|
| Formal shards lacked canonical inputs and a formal training reader | Added standalone H5 export and a worker-safe formal loader | Full archive verification plus pickle, fork, spawn, and two-worker DataLoader checks |
| Split validation checked totals without proving actual membership | Validate exact split counts, roots, physical hashes, panel exposure, and quartet ownership | Negative split/root tests and live full-profile design audits |
| Bounds ignored intervention overrides | Include exact overridden material values in approval checks | Out-of-bound intervention rejection |
| Shard reuse trusted header identities without checking stored numerical content | Compare all packed arrays and IDs to accepted source bytes and retain file checksums | Corrupted shard-field and metadata tests; all legacy Standard shard pairs audited |
| Stored terminal currents and internal face/heat arrays were incompletely checked | Reconstruct currents, electrical/thermal face fluxes, and separate heating from saved inputs | Deliberately corrupted stored-field tests; all exported sources pass strengthened audit |
| Profile-specific approvals blocked legitimate nested reference reuse | Preserve inherited raw bytes and original solver/approval identity; bind source release explicitly | Hardlink/inheritance/resume/forged-lineage tests |
| Original GPU solver identity did not fingerprint implementation versions | Preserve immutable legacy identity; new CPU references record numerical source fingerprints and library versions | `numerical_recipes.json` plus source ledger validation; no retroactive legacy claim |
| TRAIN layout and morphology exposure are associated | Preserve and disclose the frozen design | Every TRAIN layout has one morphology family in both profiles |
| Older case documentation still said no formal release existed | Update current physical/data registry and browsing guidance | Current release and training-archive receipts agree |

The initial full Standard audit checked 4,608 raw-file hashes and all 73 raw/processed shard pairs. Every legacy processed value equalled the float32 conversion of the corresponding FP64-derived target. Maximum absolute conversion discrepancies by channel were approximately $(2.97\times10^{-8},4.76\times10^{-7},4.76\times10^{-7},1.19\times10^{-7},7.62\times10^{-6})$. These are storage precision discrepancies, not model prediction errors. The compact per-file checksum and provenance inventory remains in [the initial review receipt](Dataset/readiness/diagnostics/h5_initial_review_audit.json).

The strengthened physical audit accepted all 4,608 Standard and all 9,216 Expanded source records. Across Expanded, maximum relative electrical and thermal residuals were 1e-10 and 1e-10; maximum relative current, electrical-power, and thermal-power balance errors were 1.07e-10, 8.87e-14, and 1.39e-12. Stored terminal currents, electrical face currents, thermal face fluxes, and separate Joule-heat arrays agreed exactly with their reconstructed FP64 definitions in these checks. Exterior normal flux was zero. These are reference-solver and storage checks, rather than a trained model's accuracy.

Expanded generation completed in 360.88 seconds with 4,608 new CPU FP64 solves and 4,608 audited inherited cases. The inherited sources share exactly 2,585,515,278 apparent bytes with Standard through the same inodes; their numerical bytes and original solver/approval provenance were preserved. The independent cross-profile audit also found exact agreement of all 4,608 shared target arrays, prescribed inputs, contact identities, split memberships, and source-lineage columns. This generation time includes reuse and audit and should not be compared as though all 9,216 cases were solved anew.

The engineering test suite passed 79 tests. Three Python fork deprecation warnings arose in the deliberate fork-worker checks after Torch thread initialization; the spawn route was also tested, and is recommended for that parent state. The real Standard and Expanded H5 archives each passed a CPU forward/loss-backward diagnostic through the actual general HONF implementation. Each diagnostic used one M2 TRAIN case and one M16 challenge case with 17 native queries, checked finite gradients, contact permutation, mixed contact padding, query chunking, and exclusion of targets/diagnostic labels from model inputs. It performed zero optimizer steps and verified unchanged parameters. These two challenge cases are now development-exposed interface examples; their random-model loss is not a predictive-accuracy result. The retained receipts are [Standard](Dataset/readiness/diagnostics/standard_honf_smoke.json) and [Expanded](Dataset/readiness/diagnostics/expanded_honf_smoke.json).

| Later task | What this round gains | What remains unmeasured | Next concrete step |
|---|---|---|---|
| Predictor | Complete archives, frozen splits, TRAIN normalization, actual general-model connection | Trained field accuracy, sample efficiency, conservation of predictions | Select an explicit training budget and case TaskProvider/campaign recipe; retain archive and normalizer binding |
| Organizer | Separate contact/environment entities and physical quartet references | Learned interaction fidelity, actual sparse execution, complete runtime savings | Compare full-access and selected organization on matched cases and field-specific intervention responses |
| Inverse | Physical inputs and constraints are reconstructible; references are available | Differentiable design interface, candidate feasibility, optimized objectives | Define electrical service and geometry constraints, implement live design tensors, and verify candidates with the reference solver |

No sparse-work or executor saving was measured in this round. The full-access A2-D connection is an interface check. No predictor was fitted, no learned support was presented as physical causality, and no inverse design was optimized.

## 9. How a later training program connects

Use the ModularDT Conda environment and put this case's `src` and `HONF_Proj/src` on the import path. The general model implementation is imported lazily by the case adapter. A minimal CPU batch connection is:

```python
from torch.utils.data import DataLoader
from hetmf.training_data import HETMFDataset
from hetmf.honf_adapter import ElectroThermalHONFAdapter

dataset = HETMFDataset("Dataset/standard.h5", split="train",
                      query_count=1024, seed=0, normalize=True)
adapter = ElectroThermalHONFAdapter(device="cpu")
loader = DataLoader(dataset, batch_size=2, shuffle=False,
                    num_workers=0, collate_fn=adapter.collate)
batch = next(iter(loader))
model = adapter.build_model(context_dim=32, num_heads=4)
prediction = model.decode(model.prepare(batch.scene), batch.query).nonlinear_field
loss = ((prediction - batch.targets).square().mean(-1)
        * batch.query_weights).sum(-1).mean()
```

DataLoader workers should read and collate on CPU; the main training process can move a complete batch with `batch.to(device)` and place the model on the same device. Use a guarded Python entry point and spawn workers when the parent has already initialized Torch threads. Query sampling is deterministic from the seed, epoch, stable physical case ID, and input hash, so a shared physical state receives the same native query draw in Standard and Expanded even if its archive row changes. Call `dataset.set_epoch(epoch)` when changing training query draws, and account for worker copies when using persistent workers. Full evaluation uses every native cell rather than the training subsample.

The stored target normalizer is profile-specific and its TRAIN-membership fingerprint is exposed by the loader. Always invert it before reporting physical field errors and balances. Input IDs, provenance, and target labels remain separate from the prescribed feature tensors. A future shared training-engine CasePlugin/TaskProvider and an explicitly approved schedule are still needed for a campaign; the adapter is the tested connection point, rather than a training launch script.

The reusable case commands are `export-training`, `verify-training`, `link-training`, `honf-smoke`, and `diagnose-releases`. Expanded generation uses `generate --device cpu` with its exact CPU/inheritance approval. The [generation guide](docs/HETMF_GENERATION_GUIDE.md) records the invocations and the [declarative readiness configuration](configs/training_readiness.yaml) records feature and model contracts.

## 10. Storage, retention, and artifact index

The round reserved 24 GiB retained payload, 4 GiB temporary staging, a 256 MiB source-worktree allowance, and 100 GiB minimum `/data2` headroom before materialization. The original Standard sources and selected historical pilot evidence remain protected. The retained new unique payload, including both training archives, additional Expanded references, audit references/figures/receipts, and the isolated source worktree, measured 5,461,779,925 apparent bytes (5.09 GiB), excluding the inherited Standard source inodes. Cleanup removed 2,158 disposable regular files totaling 123,906,779 payload bytes, plus 170 fixture symlinks and empty temporary directories. The selected figure versions were regenerated atomically after their old identities were inventoried; all fourteen retained native-192 reference checksums stayed unchanged and the diagnostic replay performed zero new solves. Measured `/data2` free space after cleanup was 3,367,972,798,464 bytes (3.06 TiB). The cleanup inventory, absence checks, and retention summary are in [cleanup_receipt.json](Dataset/readiness/receipts/cleanup_receipt.json) and [closeout_storage.json](Dataset/readiness/receipts/closeout_storage.json).

Retain the two standalone H5 files, exact plans and approvals, all unique physical reference arrays, inherited-source lineage, TRAIN statistics and archive seals, physical-audit and model-interface receipts, selected native-192 reference arrays, PDF masters, and the small PNG companions required for this report. Disposable test fixtures and plotting caches have explicit cleanup inventories and receipts. These generated payloads, figures, logs, and one-time helpers remain on `/data2` through ignored browsing links; durable case code, tests, schemas, configuration, and this Markdown report are the Git deliverables.

The short figure index is: [given inputs](Dataset/readiness/figures/given_inputs.pdf); [M2/M6 fields](Dataset/readiness/figures/representatives_1.pdf); [M8/M12 fields](Dataset/readiness/figures/representatives_2.pdf); [Expanded challenges](Dataset/readiness/figures/representatives_3.pdf); [grid refinement](Dataset/readiness/figures/refinement_audit.pdf); [contact–environment response](Dataset/readiness/figures/counterfactual_me_patch_move.pdf); [voltage-pair response](Dataset/readiness/figures/counterfactual_mm_voltages.pdf). Exact input-selected case identities, per-case measurements, design diagnostics, and refinement values are in [readiness.json](Dataset/readiness/diagnostics/readiness.json).

### Storage maintenance: 2026-10-24

An explicitly approved cleanup retired the 73 legacy Standard raw shards and 73 processed shards after exact comparison of every array and membership against the 4,608 retained original FP64 cases and current Standard training archive. Both training archives, all original and refined simulations, source lineage, normalizers, ledgers, scientific diagnostics, and selected report figures remain retained. The physical dataset identities and measurements above are unchanged. Duplicate frozen plans and approvals share hardlinks while preserving paths and bytes. The home Dataset directory now uses six clearly named browsing links; pilot figures and its manifest are reached beneath `Dataset/pilot`. Worktree removal remains deferred. The [storage index](Dataset/README.md), [cleanup receipt](Dataset/readiness/receipts/tidy_20261024_v1/cleanup_receipt.json), and [exact shard verification inventory](Dataset/readiness/receipts/tidy_20261024_v1/shard_redundancy_verification.json) document the current layout and evidence. Earlier storage totals and packing receipts describe their dated closeouts.

## 11. Coworker preparation: 2026-10-26

The curated main-branch preparation adds portable training and evaluation interfaces for three general joint-field models: **Six-hyperedge organization controlling contextualized member reads**, **Dense MM/ME/EM interaction and direct field reads**, and **A2-D**. These names describe the implemented information access; no knowledge of historical run numbers is needed. The first model disables the additional global/near value reads and direct hyperedge values so contextualized member access follows the six-hyperedge relations. The dense model performs MM, ME, and EM contextual exchanges and direct field reads. A2-D retains a full-access context and nonlinear query read with no affine thermal control branch. Each model receives prescribed contact voltage before interaction and predicts the same five targets jointly.

The shared repository is `/data/public/ModularDT-HETMF`; the small read-only data/evidence bundle is `/data/public/ModularDT-HETMF-Dataset/v1`. Original simulation archives remain protected and unchanged. Published `standard.h5` and `expanded.h5` preserve the existing bytes, memberships, normalizers, and provenance, and their embedded plans do not depend on access to private original paths. The generated figures and selected evidence accompanying this report are published outside Git. The [coworker handoff](docs/COWORKER_HANDOFF.md) gives installation, portable linking, processing, model configuration, training, resume, and evaluation commands.

Preparation qualifies finite CPU forward/backward calls without fitting the HONF models. The common epoch trainer's visitation, state saving, RNG restoration, and resume behavior are tested separately with a small linear toy model; that engineering test is not a HET-MF learning experiment. Default configurations declare a first segment of 100 full-TRAIN epochs for a later explicitly selected model. Batch size 48 and 1,024 queries are starting settings; the qualification checks two cases and 17 queries, so it does not establish the requested training batch's GPU memory or runtime. There are still no trained HET-MF field-accuracy, learned-organization, inverse-design, or sparse-execution results.
