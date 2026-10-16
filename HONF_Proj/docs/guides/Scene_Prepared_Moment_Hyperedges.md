# Scene-prepared moment hyperedges

This family extends attention-inherited gathering by computing compact donor statistics and a processor once per scene, at the existing environment anchors. It reuses the ordered contextualizer, native datasets, normalization, deterministic case/query sampling, physical extraction and TrainingEngine. The model has separate Thermal and Wind weights. Soft associations remain overlapping, positive weighted distributions; their learned support is not a physical causal graph.

## Shared representations

| Variant | Anchor state read by a query | New processor |
|---|---|---|
| C0 | Projected inherited typed means | None |
| M | Projected means plus a residual from typed moments | Two affine layers, hidden width 128 |
| H | The same inputs, dimensions and parameter inventory as M | The same layers with SiLU between them |

The inherited last-cycle EM and EE scores define separate module and environment associations. Their payloads come from the final contextualized token states. Each head retains its full width-32 value mean and projects donor values to eight moment coordinates. There is no nonlocal query/source value MLP, learned selector, extra environment grid or additional contextual feedback cycle.

For each typed association row, let \(s=\sum_iP_i\), \(\xi_i=(x_i-o)\oslash L\), \(r_i=\xi_i-\xi_a\), and \(u_i=L_hV_{hi}\). The retained statistics are

\[
\bar u=\sum_iP_i u_i,\qquad d=\sum_iP_i r_i,\qquad C=\sum_iP_i r_i\otimes u_i,
\]

\[
\Sigma^x=\sum_iP_i r_i r_i^\mathsf{T}-dd^\mathsf{T},\qquad v^u=\sum_iP_i u_i^{\odot2}-\bar u^{\odot2}.
\]

The cross moment \(C\) is the raw relative position/value moment, not a centered cross covariance. Only the symmetric upper triangle of spatial covariance and the diagonal latent variance are retained. Latent statistics have latent or nondimensional units; they are not energy or calibrated uncertainty. Module and environment measures retain their separate native meanings.

Moment computation uses donor-only lifts followed by the existing P contraction. It does not construct an anchor-by-donor-by-value feature array. Relative centering happens after aggregation, including the actual floating-point row mass: \(d=\sum P\xi-s\xi_a\) and \(C=\sum P(\xi\otimes u)-\xi_a\otimes\bar u\). No silent row renormalization is permitted.

## Descriptor and numerical contract

The descriptor order is normalized anchor coordinate, normalized anchor characteristic lengths, operating state of width 128, then module and environment blocks in that order. Each type contains its full head-major means, followed head by head by the eight-coordinate mean, relative centroid, raw position/value cross moment, spatial covariance upper triangle, eight-coordinate latent variance, validity flag and association-weighted normalized donor characteristic lengths. The resulting descriptor has width 708 in 2-D and 814 in 3-D. Upper triangles follow `torch.triu_indices` order: `00,01,11` in 2-D and `00,01,02,11,12,22` in 3-D.

Production arithmetic is FP32 with TF32 and autocast disabled for matched experiments. An independent FP64 literal calculation checks the moment implementation. A deterministic live shift from the first valid donor of the same type reduces variance cancellation. For \(z=u-c\), \(t=\sum Pz\) and \(q=\sum Pz^2\), the literal variance is evaluated as \(q-t^2+2ct(1-s)+s(1-s)c^2\). This retains the written formula when row mass differs slightly from one. Empty typed rows have exact zero means, moments and lengths, with validity zero.

The roundoff allowance for a latent variance coordinate is \(\gamma_{4N+16}(\epsilon)\max(|\sum Pu^2|,\bar u^2,\mathrm{tiny})\), where \(\gamma_k(\epsilon)=k\epsilon/(1-k\epsilon)\), \(\epsilon\) is the input P dtype epsilon and N counts positive donors. The spatial diagonal uses the analogous second-moment scale. Only negative diagonal values within this predeclared allowance may be clamped; larger negatives must be flagged. Signed covariance off-diagonals are retained, and no variance square root is used.

The M/H output projection and its bias start at zero; the first processor layer uses ordinary nonzero initialization. Thermal transport output heads also start at zero. No additional zero residual gate is used. New M/H tensors have matching names and initialization, so the initial complete function recovers C0 while output-layer gradients can become active immediately.

## Query reuse and cache ownership

The existing type/head projection is applied to the two typed means once at each anchor. A query gathers one width-128 anchor state through its four/eight geometry-owned cover slots; the original projection bias is added exactly once after gathering. Typed query contexts are trace-only intermediates. The field head and Thermal `response_phi` retain their input widths; `response_phi` already consumes global and local context.

Receiver topology belongs to the prepared scene. It retains discrete sorted-axis source offsets and the lattice-to-environment-token map, native identities, legal bounds and version bindings. Every query cover gathers axis coordinates from the live scene tensor, preserving query and selected-anchor geometry gradients. A factory validation seal avoids repeating semantic reductions after construction, while tensor versions, ownership and topology checks remain active. Manual covers retain full semantic validation.

Prepared moment and transport arrays must participate in the existing weight, buffer, scene, normalizer, receiver and consumed-graph checks. A membership intervention with fixed contextual states rebuilds every dependent mean, moment, processor output and heat factor. A real geometry intervention rebuilds contextualization as well. Exported hypergraph views refer to actual prepared arrays rather than a second organizer calculation.

The Torch indexed weighted reduction remains the reference executor. Compilation is an explicit optional diagnostic. The installed compiler's forward or first-gradient success alone does not qualify higher-order native training; its derivative, cold-cost and fallback evidence must be reported separately.

## Thermal source-affine transport

Heating is excluded from contextualization, association scores, moments, processor activations, receiver covers and coefficient normalization. For native normalized module measure \(m_i\), signed carrier \(t_{ihr}\), and heat-independent processor outputs \(a^0_{ahr},a^\ell_{ahr}\), the source coefficient is

\[
K^{\mathrm{anchor}}_{ahri}=\frac{P^M_{ahi}}{m_i}t_{ihr}\left[1+a^0_{ahr}+\sum_\ell a^\ell_{ahr}(\xi_i-\xi_a)_\ell\right].
\]

M and H use the same diagonal carrier mixing; C0 retains the original transport. The original compact local source response remains present. Ordinary prediction, heat increments, JVP, VJP and bounded explicit K export must use the same source identities and coefficient law. Precise accumulation casts raw factors before arithmetic instead of casting an already rounded FP32 coefficient cache. Temperature obeys \(T(\mathcal X,h)=K(\mathcal X)h\), zero heating gives zero temperature, and flow remains heat independent. Endpoint maxima are recomputed at both endpoints.

## Controlled development continuation

Round5 C0/M/H runs branch from each dataset's own Round4 C-soft e500 checkpoint; Wind uses C, not Q. Each run preserves common weights, named AdamW states, RNG, sampler/provider state and the full 500-row history prefix. New parameter optimizer states start at their actual birth. All inherited and new parameters train jointly. The primary endpoint is absolute e2500, with reviews at e600, e1000, e1500 and e2500; secondary selection uses the common saved e600–e2500 cadence and earliest exact ties. Parent e500 is not an M/H selected checkpoint.

Thermal retains TRAIN150/DEV22, sampling seed 0, effective batch 48 with microbatch 24, primary Q1024, native auxiliaries and the scheduled response addendum. Wind retains TRAIN72/DEV24, initialization seed 0, sampling seed 42, effective batch 16 with microbatch 8, and Q8192 split into 6144 volume and 2048 hub queries. Its Q10240 monitor and separate five-role native evaluation board retain their distinct definitions. Both use all-parameter AdamW at constant 3e-4, weight decay 1e-5, betas (0.9,0.999), epsilon 1e-8 and clip 1, including final partial macro-batches.

Loss curves refresh every 10 epochs with tqdm progress. Full validation history, latest/best states and declared checkpoint milestones are retained; separate validation files are written only at milestones and final/clean-stop reviews. A conditional cached first-order reconstruction requires its own TRAIN diagnosis, resource gate and child identity; it does not modify a running H control. Fresh full-TRAIN probes require the separate qualification decision and stop at e1000. Existing full-TRAIN D and H6 protocols, historical replay, TEST locks and solver usage 326/326 remain separate contracts.

## Standalone construction and loading

Use the ModularDT interpreter with `HONF_Proj/src`, `HONF_Proj/Case_ThermalChannel/src` and `HONF_Proj/Case_WindFarm/src` on `PYTHONPATH`. These functions reconstruct the saved native model and TRAIN transform without constructing a training provider, opening a dataset, creating an optimizer or launching training. Call them with a trusted Thermal or Wind Round5 checkpoint. CPU construction does not require a GPU allocation.

```python
from honf_runtime.content_hypergraph import load_content_hypergraph_checkpoint


def load_thermal(checkpoint):
    model, physical_adapter, payload = load_content_hypergraph_checkpoint(
        checkpoint, device="cpu"
    )
    assert payload["experiment_identity"]["recipe"]["dataset"] == "thermal"
    assert physical_adapter is model
    return model, payload


def load_wind(checkpoint):
    core, physical_adapter, payload = load_content_hypergraph_checkpoint(
        checkpoint, device="cpu"
    )
    assert payload["experiment_identity"]["recipe"]["dataset"] == "wind"
    return core, physical_adapter, payload
```

The saved native metadata also supports explicit construction before loading tensor weights. Thermal's adapter contains its shared core; Wind keeps the shared core and physical conversion adapter separate. This example uses the exact constructor metadata from already validated payloads and does not fit new normalization statistics.

```python
from channelthermal.content_hypergraph import ThermalInheritedAttentionAdapterV1
from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1
from windfarm.content_hypergraph import WindInheritedAttentionAdapterV1


def construct_thermal(payload):
    native = payload["experiment_identity"]["provider_identity"]
    model = ThermalInheritedAttentionAdapterV1.from_state_payload(
        native["adapter_state"], device="cpu"
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def construct_wind(payload):
    native = payload["experiment_identity"]["provider_identity"]
    core = MomentHyperedgeCoreV1.from_config(native["model_config"])
    core.load_state_dict(payload["model_state_dict"], strict=True)
    adapter = WindInheritedAttentionAdapterV1.from_state_payload(
        native["adapter_state"], device="cpu"
    )
    return core.eval(), adapter
```

For a resumed scientific run, use the maintained `moment_hyperedge_train.py train --resume latest` route with its sealed recipe and original output directory. Standalone inference loading does not restore an optimizer or authorize changing dataset identity. The family loader rejects incompatible schemas, modified native bindings and a different parent lineage.
