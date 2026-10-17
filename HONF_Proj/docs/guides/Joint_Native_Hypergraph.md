# Joint native M/E hypergraph

Round9 introduces `joint_native_hypergraph_v1`, a separate family over the original A2-D contextualizer, native adapters and physical objectives. Original module and environment records retain separate identities and measures. Environment values are evaluated at each query before their reduction conditions the module query. The conditioned query reaches the nonlocal module reader, local reader and field head, including Thermal's signed source coefficients. Environment records do not acquire heater columns.

## Four matched models

| Arm | Value access | Training ancestry |
|---|---|---|
| D9-M | Original dense M | Literal segmented D8 e2500 through e5000 |
| D9-ME | Dense E followed by E-conditioned dense M | D8 e2500, dense E warm-up through e3000, continued through e5000 |
| H9-ME | One typed native seed directory, with selected E and M values | Exact shared D9-ME e3000 weights and Adam moments, then e3001–5000 |
| S9-ME | Independently learned query-to-M and query-to-E gates, same value function | Exact same shared D9-ME e3000 boundary, then e3001–5000 |

The managed warm-up stage is named `D9-ME-warm`. It is shared ancestry, not an additional terminal comparison arm. All models retain absolute-epoch case/query streams and the original TRAIN normalization. The common warm-up executes once per dataset. The complete intended campaign contains 36,000 Thermal and 45,000 Wind new optimizer updates; a launched or partially trained stage does not establish completion.

## Typed organization and value semantics

H captures actual raw scores from the final MM, ME, EM and EE contextual attention stages. Destination and donor types define each route. Independent convex head mixtures, bounded scales and TRAIN-calibrated membership biases produce native seed/member incidence. Every seed retains its eligible own record and a best-scoring cross-type record. A pre-E query dispatch chooses one native seed. Original module local protection and the nearest positive-measure environment record join the selected list before deduplication and value normalization. These architectural floors are reported separately from learned membership.

Geometric keys and declared type resolve distinguishable ties. Indistinguishable tied maxima are rejected. Native IDs identify records and reproducible noise, but never act as learned geometric features. Duplicate-key groups are computed per scene; query tie checking is linear in the query-by-seed inventory and does not construct query-by-seed-by-seed tensors. Each stochastic scene reuses its membership sample across receiver tiles and associated calls, with distinct type/route noise namespaces.

E values use original query-specific nonlinear messages and a within-E measure-weighted reduction. A bias-free projection A_E maps that context into the original query state. Only A_E starts at zero; the E reader starts nondegenerately. This preserves the original field boundary while permitting an initial gradient on A_E and subsequent gradients on E parameters. Module information attention is recomputed from the conditioned query. Its weights never replace Thermal's signed, full-amplitude physical source coefficients.

Thermal retains T=Kh with zero-background heat-affine capability, heat-independent routing/flow, native interpolation, curl, flux-proxy extraction and separately recomputed peaks. Wind retains original native cell IDs, the aligned TRAIN-fitted background profile and scaled velocity residual in m/s. Dense contextualization remains upstream, so a record omitted from the final read can still influence selected contextualized states. Final read support is executed model organization, not a causal boundary over physical inputs.

### Dependency contract

| Quantity | Permitted inputs | Meaning of the restriction |
|---|---|---|
| Prepared M/E states | Prescribed geometry, material/environment descriptors, dense contextualization and scene summaries | Architectural dependence; final sparsity does not remove this ancestry |
| Scene membership | Actual MM/ME/EM/EE scores, typed calibration and prescribed scene RNG | Query-independent organization; targets and heat amplitudes are excluded |
| Query dispatch and protection | Original query state and coordinates, prepared M/E scores, original local support and nearest positive-measure E | One dispatch before E-value conditioning; an architectural execution rule |
| E context and conditioned query | Individually evaluated selected E messages, E measures and bias-free A_E | Learned information path; E records are not independent physical interventions in the existing data |
| Module information and local field decoding | Conditioned query, selected original M values and original local features | E can alter native M messages, rather than only adding a final offset |
| Thermal signed K and flow | Prescribed heat-free scene and query inputs, including the permitted E path | Heat-affine generator contract: heat amplitudes enter only through the original source contraction |
| Thermal temperature, derived roles and peaks | Signed K, original heat amplitudes and unchanged native extraction | Physical source columns stay tied to original modules; peaks are recomputed |
| Wind physical velocity | Native model residual, original TRAIN-fitted scaling and aligned background profile | Dataset/provider contract; no interpolated targets or changed reference frame |

These contracts distinguish a generator's physical laws from the available dataset variation and the model's chosen organization. Successful graph intervention establishes an executed information path. Physical interaction identification additionally requires compatible saved reference states for every endpoint of the intervention.

## Training and execution

The first 200 sparse epochs use hard-forward dense router gradients at every macro. Subsequent dense-router steps use `((sparse_epoch_zero_based + macro_index_zero_based) % 4 == 0)`. This exposes every macro position once in four epochs, including the partial final batch. Saved clocks determine the cadence on resume. The original Thermal response cadence remains separate. Selected steps use actual packed native values, while dense-router steps deliberately evaluate omitted values to provide restorative gate gradients.

M and E have separate expected/relaxed work constraints and serialized dual states. Their terminal removable-row retention objectives are 0.85 and 0.60, respectively, above the actual mandatory/protected union. Work is aggregated over native requests within a scene and then across cases; the separate saved response scene is explicitly charged. A constraint is hinged once over the complete macro, not independently over microbatches. Initial membership and dual coefficients require measured fixed-TRAIN calibration on at most eight batches; zero defaults are not scientific admission evidence.

`dense_ME` is the all-open E-conditioned control. `dense_typed_reference` evaluates the same physical function with hard masks and zero-sensitive multiplicative normalization. `flat_typed_selected` gathers original query/source rows before wide Fourier and nonlinear value features. Optional non-reentrant checkpointing encloses value evaluation and reduction, returning compact contexts and source coefficients. It reuses supports and does not resample routes or increment counters during recomputation. Traces are opt-in and bounded.

Measure allocated and reserved memory in fresh processes with one model, identical precision, effective/microbatch configuration, query counts, physical roles and optimizer state. Report selected-step and dense-router-step peaks separately, together with the maximum over a complete four-epoch cycle. A selected-step reduction does not imply a lower full-training capacity requirement. Complete host-input/native-CPU-output latency is descriptive; a speedup is not an admission condition.

## Managed interfaces and evidence

`tools/joint_native_train.py` provides recipe creation, CPU preparation, fitting, exact resumption, status and standalone inference export. Recipes bind literal source checkpoint hashes, inherited provider identity, absolute stage ages, common microbatch, value execution settings, fixed-TRAIN calibration and the explicit resource reservation. The launcher uses the original TrainingEngine and native providers. It does not create a second training objective or silently resume across different data or schedules.

The public `load_content_hypergraph_checkpoint` dispatches this family to its standalone loader. Inference reconstructs the model, adapter schema, capability and normalizer from embedded checkpoint content without reading ancestor checkpoints or TRAIN data. Native evaluation separately rebuilds and verifies the source-bound provider; TRAIN audits and exposed DEV selectors retain their distinct Wind quotas. Prepared queries reject stale weights, geometry, type catalogues, normalizers and consumed backward graphs. Counterfactual route replay validates stored typed IDs against newly rebuilt scene states.

Scientific checkpoints are retained every 100 epochs, plus the shared branch boundary and explicit review/final stops, with latest and best-field aliases. Full native history and ten-epoch curves remain enabled. Resource monitoring samples the authorized GPU, records observation gaps, and requests a fully resumable epoch-boundary stop before the declared free-memory floor or reserved closeout time is exhausted. Generated checkpoints, arrays, figures, temporary fixtures and receipts belong under the reserved ignored `/data2` root.

The [work plan](../../UpgradePlan/HONF_Round9_Joint_Native_Hypergraph_Work_Plan.md) defines the scientific horizon, native boards, physical-response audits and reporting bands. The [future physical data specification](Joint_Native_Future_Physical_Data.md) separates available saved reference perturbations from missing independent environmental evidence. Implementation tests, exact ancestry, completed learning, field utility, sparse memory and physical interaction identification are separate claims and require their own retained measurements.
