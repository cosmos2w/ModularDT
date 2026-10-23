# U-H upgrade design: soft merging and directly constructed compact hyperedge directories

**Status:** detailed proposal for a future model-development round; not an implementation report, a qualified numerical amendment, or training authorization.

**Prepared:** 22 October 2026. This note connects the existing attention-derived, applied unified organization to a proposed compact group representation. It does not modify existing U-H/U-D/U-S models, checkpoints, active experiments, data memberships, or historical results.

**Design revision:** 23 October 2026, following the full-data epoch-1000 comparison. The user authorized retaining and sharpening this proposal, not implementing it or starting another fit. The selected design below resolves the first coarsener, compact scorer, transition and comparison policies; its thresholds and resource budget must be frozen before any later execution.

**Central idea:** initialize organizational candidates from original M/E records, learn which candidates are functionally redundant, and gradually replace them with fewer live groups. After compaction, construct and query only the live K-group directory. Preserve every original record, its individual query-dependent value, and every Thermal source column.

## 1. Executive recommendation

Treat physical records and organizational groups as separate axes. The current equality between their counts is an initialization choice, not a physical requirement. Start with N = M + E native organizational candidates, allow their membership functions to become compatible, and compress redundant candidates into K live groups, where K can depend on the scene and need not be prescribed as six or any other universal constant.

The proposed upgrade has two explicit levels. First, qualify compact final retrieval while retaining the existing encoder; this isolates the consequences of replacing the native-seed directory. Second, if encoder efficiency is a required goal, replace dense contextual pair scoring with an explicitly group-factorized contextual operator whose applied group memberships are reused by final retrieval. The second level is a new mathematical model and must be evaluated as such.

The required endpoint is not “a soft assignment matrix was added.” It is “the ordinary compact forward constructs K-by-N group membership, computes query-to-K access, and does not reconstruct retired N-by-N directories.” A claim of reduced encoder complexity additionally requires the second level; compact retrieval alone leaves the old encoder's dense pair scoring intact.

Keep the scientific question explicit: can fewer learned, input-dependent groups preserve useful native fields and responses while reducing organizational overhead? Group count, actual M/E value work, peak memory, complete time, and physical interpretation remain separate outcomes.

### 1.1 Claims and selected scope

Current native U-H already defines a weighted overlapping hypergraph. This proposal's architectural contribution is a smaller, reusable organizational axis that is independent of the physical record axis; it does not convert an otherwise invalid graph into a mathematically valid hypergraph. The Level U operator has distinct donor membership C and receiver participation R, so the complete applied structure is a typed, directed group-mediated operator rather than a single undirected adjacency. Neither representation identifies physical causal edges.

| Decision | Selected first design | Explicit limit |
|---|---|---|
| Initial milestone | Level R, same-destination-type groups, original encoder and individual decoder values | Not yet compact applied encoding |
| Portable coarsener | Shared learned sibling compatibility on one deterministic, typed geometry hierarchy; Section 11.3 | Restricted candidate family, not globally optimal clustering |
| Compact scorer | Shared pre-E query-to-live-descriptor scorer, trained on native objectives | No deployed native-logit summation or exact parent equivalence |
| Structural transition | Soft compatibility learning followed by one versioned hard-cut activation at a review boundary | Variable K introduces topology seams |
| Scientific starting point | Literal full-data U-H epoch 1000 parent; separate matched adaptation identity | Not a fresh quarter-data experiment or exact resume |
| Primary success | Useful directory compression at preserved fields/tails/responses and measured organizational cost | Fewer groups alone is insufficient |

Level U remains conditional on a useful Level R result and a measured contextual bottleneck. A fixed-K factorized primitive is an engineering control for Level U; it is not evidence that adaptive scene compression works. The [U-S companion proposal](HONF_US_Geometry_Conditioned_Implicit_Hypergraph_Upgrade_Design.md) pursues a separate spatial-selection prior, and neither design is a required add-on to the other.

## 2. What is already implemented in U-H

### 2.1 The completed mechanism to preserve

Existing U-H does not derive its directory from a separate geometric clustering pass. Each contextual stage forms typed attention scores from current record states and geometry, transforms those scores into continuous incidence, applies the incidence to the actual contextual value exchange, and passes the updated states into the next stage. Final retrieval reuses the actual applied incidences of the last contextual cycle, with a separate retrieval self-identity convention.

~~~mermaid
flowchart TD
    X["Original local M / E records"] --> Z["Current record states"]
    Z --> L["Typed attention logits"]
    L --> W["Measure-aware attention weights"]
    L --> B["Calibrated continuous incidence"]
    W --> A["Applied gated message coefficients"]
    B --> A
    A --> U["Update original M / E states"]
    U --> NEXT["Next stage and next cycle"]
    NEXT --> LAST["Last-cycle applied incidences"]
    LAST --> DIR["Native-seed retrieval directory"]
    Q["Receiver query"] --> PI["Seed access weights"]
    DIR --> READ["Composed access to original records"]
    PI --> READ
    READ --> OUT["E-conditioned M response and native fields"]
~~~

**Figure 1. Existing U-H dependency chain.** The incidence controls values during encoding and then provides retrieval organization. It is not merely an attention visualization produced after a dense encoder has completed. Dense scores and normalizers nevertheless remain control-plane dependencies.

The contextual order is MM -> EM -> EE -> ME, repeated twice. Names specify destination first and donor second. Thus EM means that E receives from the updated M states. Each stage has its own source-state version; four blocks from the same cycle do not describe one simultaneous physical adjacency.

### 2.2 Existing equations

For route t, head h, destination r and donor i, the current mechanism can be summarized as follows; masks and reference-measure conventions are implicit here and remain explicit in implementation:

\[
L^{t,h}_{ri}
=Q^{t,h}(z_r)\cdot K^{t,h}(z_i)+G^{t,h}_{ri},
\]

\[
A^t_{ri}
=\kappa_t\sum_h\omega_{t,h}
\left(L^{t,h}_{ri}-\max_{j\in\mathcal E_t(r)}L^{t,h}_{rj}\right)+\beta_t,
\qquad
b^t_{ri}=\psi(A^t_{ri}),
\]

\[
\psi(a)=\frac{[a]_+^2}{1+[a]_+^2}.
\]

The eligible donor set is denoted by \(\mathcal E_t(r)\); it is not the environment-record type E. The calibration mixes heads separately for each typed route. It uses raw attention logits relative to their eligible row maximum, not softmax weights interpreted directly as memberships.

For one head:

\[
w_{ri}=\frac{\mu_i\exp L_{ri}}
{\sum_{j\in\mathcal E_t(r)}\mu_j\exp L_{rj}},
\quad
Z_r=\sum_i b_{ri}w_{ri},
\quad
D_\varepsilon(Z)=Z+\varepsilon(1-Z)^2,
\quad \varepsilon=10^{-3},
\]

\[
\alpha_{ri}=\frac{b_{ri}w_{ri}}{D_\varepsilon(Z_r)},
\qquad
u_r=\sum_i\alpha_{ri}V_i.
\]

The output projection, correctly mass-weighted projection bias, residual and local FFN then update the destination. Zero donor mass must produce zero donor contribution. Own-record residual processing remains a distinct path.

The existing final native directory has N rows, one per native seed. Query retrieval uses:

\[
\pi(q)=\operatorname{entmax}_{1.5}(s(q)),
\qquad
a(q)=\pi(q)B^{\mathrm{ret}},
\quad
B^{\mathrm{ret}}\in[0,1]^{N\times N}.
\]

For M, the smooth local envelope is retained:

\[
g_i^M(q)=\ell_i(q)+(1-\ell_i(q))a_i^M(q),
\qquad
g_j^E(q)=a_j^E(q).
\]

### 2.3 What existing evidence does and does not establish

The latest completed comparison uses literal full-data epoch 1000 U-D/H/S and classic1404/1804 checkpoints on the same already-exposed canonical89 cases. U-S leads seven of nine mean errors within the U triplet and six of nine among all five models; U-H leads fluid temperature and vorticity. U-H fluid-T mean RMSE is 0.394110 versus U-S 0.421801, but H's material/peak-T p90 remains 12.5%/18.1% above classic1804. Classic normalization, objectives and training draws differ, and launch-source closure is incomplete across the U arms. These are measured predictor comparisons, not an organizer-only causal ablation.

On input-selected case 0692/M10, U-H has 202 live native-record anchors, 4874 positive group memberships out of 40804 live possibilities, and 12.82 entropy-effective anchors per sampled query. Per-query concentration does not establish a scene-wide compressible K: queries may require different anchors, and detailed traces cover only eight fixed queries per case. Redundancy and role coverage must be measured before selecting a compression level.

The saved six-case, 48-query diagnosis locates a specific support-expansion route. Every live M-anchored group's E row is fully positive in those cases, so any positive M-seed coefficient is sufficient to open all 192 E records. This occurs even at M-seed mass0.000366 in case 0293/query 4680. Convex merging cannot remove that support without changing the membership/access function. Do not infer that every admitted contribution is large, that the phenomenon holds for every scene, or that physically necessary environmental influence should simply be cut off.

Across canonical89, H's positive final M/E support is 72.65%/39.79% of eligible rows; S's is 48.71%/44.46%. All three configured dense executors nevertheless evaluate 100% of final M/E value rows. The new comparison measures neither complete-call acceleration nor a memory advantage, and does not replay inverse decisions with the epoch 1000 weights. See the [current comparison and support diagnosis](../../docs/reports/HONF_Round11_Thermal_Development_Report.md#full-data-epoch-1000-comparison-u-dhs-and-classic-14041804).

Historical completed U10-e5000 development results use exposed DEV22 and an input-stratified TRAIN24 panel. Their packed final E retention is 22.00% for U-H on DEV22 and 32.81% on TRAIN24, while their selected M1 illustration reads every E record; their six-case scan finds full E support at 36.22% of queries. These belong to a different training lineage and panel and remain historical evidence rather than the ranking of the new full-data models.

U-H still has field tradeoffs, including module-peak and surface/material temperature misses, and weak saved-candidate ranking. H/S typed work is unequal. Round11 also reports that dense-weighted execution lowers allocated memory in one same-function comparison without a speedup, while segment execution is not faster in that comparison. These are reasons to test organizational compression carefully, not reasons to assume it works.

Neither current U-H nor this proposal establishes independent physical M-E causality. The current review establishes epoch 1000 measurements, not current worker progress or completed epoch 5000 formal results. Original records, mature checkpoints and historical negative results remain preserved.

## 3. The problem: native seeds are useful initialization, expensive permanent organization

Native seeding avoids an arbitrary fixed latent edge bank and gives clear initial correspondence. Its cost is that every record remains a candidate group even when many groups become redundant.

| Quantity | Current native-seed organization | Desired compact organization |
|---|---|---|
| Original physical/input records | N | N, unchanged |
| Live group candidates | N | K, scene-dependent |
| Group-member directory | N by N | K by N |
| Query-group coefficients | Q by N | Q by K |
| Thermal source columns | M | M, unchanged |
| Query-specific nonlinear values | Individual admitted records | Individual admitted records |
| Historical seed provenance | Native IDs | Optional lineage metadata, not an executable N-group bank |

Dense multiplication of a sparse-looking query vector by a dense directory does not skip its zero entries automatically. Similarly, a group whose final coefficient is small but positive still admits a full original-record value computation. Candidate-count reduction and value-support reduction must both be measured.

For nonnegative composition:

\[
\{i:a_i(q)>0\}
=\bigcup_{k:\pi_k(q)>0}\{i:B_{ki}>0\}.
\]

Consequently, merging groups by support union is a poor default: it can reduce the number of groups while increasing final record work.

## 4. Scope and ownership boundaries

This document authorizes no model revision, launch, pause, migration, stop, solver work, or deletion. A future implementation must use a separate branch/worktree when needed to avoid changing code imported by active workers. Existing formal runs remain on their declared source and protocol.

The new compact model must have an explicit capability name, for example compact_unified_groups_v1. That name is illustrative. It must not be loaded as an exact resume of native-seed U-H. Two-stage group compression is a scientific change even when its initial error is small.

Preserve original IDs, typed axes, presence versus information-measure eligibility, TRAIN-only normalization, query/role draws, native losses, source heat semantics and source contraction. E records do not become heaters. Do not add a hidden global neural summary, parent-prediction target, error-dependent route, or silently expanded local protection window.

A compressed group descriptor may summarize member features for control scoring. It is explicitly a routing representation, not a new physical object, source column, or decoder value. Such descriptors are a proposed architectural change and their sufficiency is not assumed.

## 5. Core design: separate record identity, group identity and physical values

### 5.1 Three persistent meanings

1. **Record identity:** original M/E IDs, states, geometry, masks, measures and physical source correspondence. No record is deleted by group merging.
2. **Group identity:** a scene-local organizational slot with membership over original records and optional lineage from initial seeds. A merged group need not have one privileged physical center.
3. **Value identity:** the per-record contextual value projection or query-conditioned nonlinear decoder value. Final values remain tied to original records.

For example, merging a group seeded by M1 with a group seeded by E1 does not merge the module M1 with the environment record E1. It merges two proposals for organizing record access.

~~~mermaid
flowchart LR
    subgraph START["Native initialization"]
        A["M1-seeded group"]
        B["E1-seeded group"]
        C["M2-seeded group"]
        D["E2-seeded group"]
        E["Other native groups"]
    end
    A --> G1["Live group A"]
    B --> G1
    C --> G2["Live group B"]
    D --> G2
    E --> G3["Other live groups"]
    G1 --> R["Membership over all original records"]
    G2 --> R
    G3 --> R
    R --> V["Original record values and M source columns remain separate"]
~~~

**Figure 2. Proposed organizational merging.** This schematic illustrates possible cross-type seed lineage. The first implementation should use same-destination-type merges to isolate typed-stage semantics; cross-type lineage is a separately tested extension in Section 10.

### 5.2 Soft assignment during learning

Let S be a row-stochastic assignment of initial organizational candidates to live groups:

\[
S\in[0,1]^{N\times K},
\qquad \sum_kS_{sk}=1.
\]

S describes seed lineage or organizational assignment, not record access. One original record may still be a donor member of many groups. Soft assignments therefore do not force the final hypergraph into a disjoint node partition.

At the native boundary K = N and S can start as identity. Initially similar groups are encouraged to approach a common functional role. At a later structural update, redundant live slots are physically removed and the remaining assignments are compacted. Continuous weights can be learned by gradients; changing the allocated number of groups is a discrete model-state transition.

Keeping a soft N-by-N assignment, multiplying it into a full native directory, and masking redundant rows is not the target implementation.

For the selected sibling-only prototype, S is explanatory lineage notation rather than a required dense tensor. Store the seed-to-live-slot index map and original seed IDs, plus the disjoint soft sibling coefficients during tying. This is sufficient to form the declared group means and reconstruct bounded diagnostic lineage; do not materialize an N-by-K assignment merely because the algebra can be written that way.

### 5.3 Compact directory and query access

After compaction:

\[
C(X)\in[0,1]^{K(X)\times N(X)},
\qquad
\widetilde\pi(q,X)\in[0,1]^{K(X)},
\qquad
\sum_k\widetilde\pi_k=1
\]

for a nonempty live bank. The ordinary compact retrieval is:

\[
\boxed{a(q,X)=\widetilde\pi(q,X)C(X).}
\]

C must be constructed directly from live group control descriptors and original record states. Neither the N original seed rows nor their original N-by-N pair scores may be a prerequisite of an ordinary compact retrieval call.

An empty group bank is not permitted for a nonempty scene during the first prototype. Keep at least one valid group per required typed receiver bank, without forcing every group to contain every donor type. A genuinely empty donor type still returns its declared zero message.

## 6. What makes two groups mergeable?

### 6.1 Membership agreement, by donor type

For candidate groups u and v, measure continuous coefficient discrepancy separately for M and E:

\[
d_t(u,v)=
\frac{\sum_{i\in t}\nu_i\,|C_{ui}-C_{vi}|}
{\sum_{i\in t}\nu_i+\epsilon},
\qquad t\in\{M,E\}.
\]

Here nu is a declared diagnostic weighting, not an extra multiplier in the physical source law. Report both unweighted record coefficients and any measure-weighted diagnostic. Also report exact-positive support overlap and support symmetric difference. A mass-only comparison can hide many weak but executed reads.

Do not let E's larger catalogue dominate a pooled distance. Compare M and E separately, then combine only through a declared criterion. Geometry proximity alone is insufficient, and distant but functionally similar groups need not be prohibited.

### 6.2 Query usage and complementarity

On a fixed input-selected query panel, record usage:

\[
u_k=\frac{1}{Q_{\mathrm{panel}}}\sum_q\pi_k(q),
\qquad
o_{uv}=\frac{1}{Q_{\mathrm{panel}}}\sum_q\min(\pi_u(q),\pi_v(q)).
\]

Inspect whether two groups serve distinct query roles, interface locations or M strata. Similar memberships can still require distinct query-score behavior if their descriptors or temperatures differ. Low average usage is not sufficient for deletion: rare surface, material or peak-relevant queries can matter disproportionately.

Use query-role-stratified statistics and a small sensitivity test. A low-use group is a retirement candidate, not automatically a redundant one.

### 6.3 Functional effect is the decisive check

For a proposed merge, evaluate the same model before and after the provisional replacement on fixed inputs. Compare final access, E context, E-conditioned query, M source coefficients, native fields, per-role losses and parameter/input gradients. Then compare complete fields and tails after adaptation. No field improvement should be inferred from directory similarity alone.

The function-preservation probe is a diagnostic on input-selected cases. Scientific model selection still uses the predeclared exposed DEV panel; do not select a favorable intervention or reference subset after reading errors.

### 6.4 Acceptance hierarchy

| Criterion | Purpose | Decision consequence |
|---|---|---|
| Same masks, type rules and source IDs | Prevent semantic invalidity | Reject invalid merge |
| Low typed membership distortion | Identify redundancy | Candidate only |
| Compatible query usage | Avoid combining distinct functions blindly | Candidate ranking |
| Bounded native field/gradient change | Protect the learned function locally | Provisional merge or reject |
| No severe tail/stratum/response regression after adaptation | Protect scientific usefulness | Retain compressed candidate or revert it |
| Lower measured complete cost | Establish practical benefit | Performance conclusion only when measured |

Select numerical expectations from the existing native regression policy before measurement. Do not invent a successful universal merge threshold in this planning note or loosen a threshold after a failure.

## 7. Merge algebra and its limits

### 7.1 Exact merge at the access level

If two directory rows are identical:

\[
C_u=C_v=C_c,
\qquad
\pi_c(q)=\pi_u(q)+\pi_v(q),
\]

then:

\[
\pi_u(q)C_u+\pi_v(q)C_v=\pi_c(q)C_c.
\]

This proves exact access preservation at the represented rows. It does not prove that a newly trained single group scorer reproduces pi_c, or that encoder states remain identical after contextual factorization changes.

In particular, applying entmax to one averaged logit is generally not equivalent to summing two old entmax probabilities. A temporary exact reference may sum old probabilities for qualification, but a deployed compact scorer must evaluate only K live groups. Fit and test that scorer explicitly.

### 7.2 Approximate merge

A reasonable provisional row is a convex combination:

\[
C_c=\lambda C_u+(1-\lambda)C_v,\qquad 0\leq\lambda\leq1.
\]

Initialize lambda from declared fixed-panel usage, with a defined fallback for zero total usage. Keep the resulting row in [0,1]. The access residual is:

\[
\Delta a(q)=
\pi_u(q)(C_u-C_c)+\pi_v(q)(C_v-C_c).
\]

Hence:

\[
\|\Delta a(q)\|_\infty
\leq
\pi_u(q)\|C_u-C_c\|_\infty
+\pi_v(q)\|C_v-C_c\|_\infty.
\]

This is an access bound only. It is not a native-output or gradient guarantee: guarded normalization, E-conditioning, nonlinear values and source response can amplify access changes. If rows differ, a single fixed mixture is exactly equivalent only under additional conditions such as a matching query-wise mixture ratio.

### 7.3 Do not merge by Boolean union

Union keeps every positive member and can inflate actual nonlinear reads. Averaging also has union support when both mixture weights are positive, although its coefficient distortion is better controlled. Therefore convex merging still needs exact-positive support accounting; it is not automatically a sparsifier.

Do not add an unreported positive cutoff to make a merged row appear sparse. If the continuous gate must change to obtain useful exact zeros, version that as a scientific function change and measure its field and derivative effects.

### 7.4 Identity conventions must change explicitly

Native U-H adds a retrieval identity for every native seed. A compact group with several ancestral seeds has no single universal self-record. Forcing every ancestral identity into every merged group can create broad compulsory support and defeat compression.

Preserve original own-record encoder residuals, current local M envelopes and every physical source column. Define compressed group memberships on their own equations, without a blanket native-seed self=1 rule. Compare this transition with the native model; it is intentional model change, not a silent compatibility shortcut.

## 8. Two implementation levels: isolate retrieval, then eliminate encoder pair scoring

### 8.1 Level R: compact retrieval over an unchanged applied encoder

Retain the existing two-cycle U-H contextual preparation. On a bounded calibration panel, learn a compressed final directory and its query scorer from the last applied native incidence. After adaptation, construct the compact final directory directly from prepared states and live control descriptors.

At ordinary inference, Level R must not form an extra full native retrieval directory. However, its unchanged encoder still constructs native contextual scores and incidences. It therefore cannot claim removal of all N-by-N work, even if the final retrieval-specific materialization is eliminated.

This level is useful for isolating whether group compression preserves decoder utility. It is also a reference control for the more ambitious encoder change. Its compact final directory is derived from the applied encoder during transition but is not literally the encoder's compact applied operator, because that encoder has not yet been compacted. Label this limitation.

### 8.2 Level U: compact groups govern both encoding and decoding

Level U is the intended architectural endpoint if encoder cost matters. Replace each dense record-to-record contextual operation with record-to-live-group and live-group-to-record factors. Derive group memberships from current attention-like control scores, apply them to actual contextual values, and reuse final applied memberships for retrieval.

Do not retain the old full pair attention as an uncharged parallel branch. The old U-H can be used as a frozen, bounded initialization or diagnostic reference, but its predictions are not physical training targets and it is not part of the deployed compact model.

~~~mermaid
flowchart TD
    LOCAL["Original local M / E states"] --> DESC["K live control descriptors<br/>No retired native group bank"]
    LOCAL --> GD["Group-to-donor logits<br/>K by N"]
    DESC --> GD
    LOCAL --> RG["Receiver-to-group logits<br/>N by K"]
    DESC --> RG
    GD --> C["Applied continuous group membership C"]
    C --> FACT["Factorized typed contextual value exchange"]
    RG --> FACT
    FACT --> UPDATED["Updated original M / E states"]
    UPDATED --> REPEAT["Continue MM → EM → EE → ME<br/>Repeat second cycle"]
    REPEAT --> FINAL["Final applied K-by-N memberships"]
    QUERY["Receiver query"] --> QG["Query-to-K scoring"]
    FINAL --> ACCESS["Compose final access without N-by-N reconstruction"]
    QG --> ACCESS
    ACCESS --> VALUES["Original query-specific E / M values"]
~~~

**Figure 3. Proposed compact unified endpoint.** The original record axis remains N throughout. A compact contextual intermediate is allowed for the existing linear contextual value reduction, while final decoder values remain individually query-conditioned. Control descriptors are not pooled physical values.

## 9. A concrete compact contextual operator

### 9.1 Directly construct only live factors

For typed route t = XY, let r index original destination records of type X, i index original donors of type Y, and k index live groups associated with the destination bank. Let p_k be a scene-dependent control descriptor with typed lineage metadata.

Compute group-to-donor logits from current stage states:

\[
L^{t,h}_{ki}
=Q_g^{t,h}(p_k)\cdot K^{t,h}(z_i)+G_g^{t,h}(p_k,x_i).
\]

Use the inherited form of route head calibration and psi to construct C^t_ki, then form per-head measure-aware base weights and admitted mass:

\[
w^{t,h}_{ki}
=\operatorname{softmax}_{i\in\mathrm{eligible}\ Y}
\left(L^{t,h}_{ki}+\log\mu_i\right),
\qquad
d^{t,h}_{ki}=C^t_{ki}w^{t,h}_{ki}.
\]

For no positive-measure donors, all d values are zero. Compute destination-to-group access directly:

\[
R^{t,h}_{rk}
=\operatorname{entmax}_{1.5,k}
\left(F^{t,h}(z_r,p_k)\right),
\qquad
\sum_kR^{t,h}_{rk}=1
\]

when a valid group bank exists. Both score families depend on live record states and declared geometry. They are not obtained by first calculating all native r-to-i scores.

Define the implicit unnormalized donor operator:

\[
\widehat u^{t,h}_{ri}=\sum_kR^{t,h}_{rk}d^{t,h}_{ki}.
\]

Its matrix need not be formed. Since each group's admitted mass is at most one and R is row-stochastic, each destination's total mass lies in [0,1].

### 9.2 Factorized execution with the same guarded-mass convention

Compute:

\[
U^{t,h}_k=\sum_i d^{t,h}_{ki}V^{t,h}_i,
\qquad
m^{t,h}_k=\sum_i d^{t,h}_{ki},
\]

\[
n^{t,h}_r=\sum_kR^{t,h}_{rk}U^{t,h}_k,
\qquad
Z^{t,h}_r=\sum_kR^{t,h}_{rk}m^{t,h}_k,
\]

\[
o^{t,h}_r=\frac{n^{t,h}_r}{D_\varepsilon(Z^{t,h}_r)}.
\]

Apply the contextual output projection and bias scaled by actual reduced mass Z/D, followed by the existing destination mask, own-record residual and FFN. Multihead bias handling must follow the same explicit convention as the maintained contextual implementation.

U_k is a transient contextual reduction with a known original-donor expansion. It is not a learned standalone hyperedge value substituted into the final decoder. Capturing a bounded provenance trace should recover R, d, the donor IDs and message mass; ordinary inference should retain only what its execution and autodiff require.

### 9.3 Self exclusion without constructing a dense pair matrix

MM currently excludes contextual self exchange. For same-type routes that require self exclusion, identify the corresponding original donor index j(r), compute:

\[
\delta^{t,h}_r
=\sum_kR^{t,h}_{rk}d^{t,h}_{k,j(r)},
\]

\[
n^{t,h}_r\leftarrow n^{t,h}_r-\delta^{t,h}_rV^{t,h}_{j(r)},
\qquad
Z^{t,h}_r\leftarrow Z^{t,h}_r-\delta^{t,h}_r.
\]

This diagonal correction uses O(NK) work and removes self contribution before guarded normalization. Test cancellation, empty support and exact gradient presence. A clamp used solely to hide negative mass from an incorrect subtraction is not acceptable; investigate numerical errors using an independent explicit reference.

This correction is exact for the proposed factorized unnormalized operator. It does not make that operator equal to the old per-destination softmax attention. Other destination-specific eligibility restrictions require a similarly structured correction; if they demand a full arbitrary r-by-i mask, the O(NK) claim may fail. Preserve the current eligibility rules or disclose an unsupported capability rather than silently dropping them.

### 9.4 This is a new function, including at K = N

Old U-H uses individual receiver-to-donor score normalization. The proposed operator uses group-to-donor score normalization followed by receiver-to-group mixing. Equal tensor dimensions, copied projections and identity lineage do not establish equality between those equations. At a special exactly representable setting they may coincide; no general parent-function equality is claimed.

Qualify the new operator against its own independent literal implementation, and separately measure deviation from the inherited U-H checkpoint. Do not call the architectural transition an executor-only optimization.

## 10. Reuse final applied compact memberships in retrieval

### 10.1 First implementation: preserve destination-type group banks

Keep a compact M-destination group bank of size K_M and E-destination bank of size K_E; K = K_M + K_E. Merge native M-seeded candidates with compatible M-seeded candidates first, and E with compatible E. Every group can still contain both M and E donor members.

Maintain live group slots across all four stages and both cycles, while recomputing their descriptors and coefficients from the appropriate stage states. The final compact retrieval rows are:

\[
C^{\mathrm{ret}}=
\begin{bmatrix}
C^{MM}&C^{ME}\\
C^{EM}&C^{EE}
\end{bmatrix}
\in[0,1]^{K\times N}.
\]

These are the actual last-cycle group membership gates used by the compact contextual operator. Reuse them without a second independently calibrated directory or a full native directory. The constituent blocks retain distinct stage/source-state metadata.

The contextual donor weights d and destination-group weights R are additional parts of the applied value plan. C alone is not a complete frozen contextual plan and must not be used as one in intervention tests.

### 10.2 Later extension: cross-type seed lineage

An eventual mixed-lineage group can have type fractions rho_k^M and rho_k^E summing to one. A declared possible retrieval construction is C_k^t = rho_k^M C_k^(Mt) + rho_k^E C_k^(Et), using actual applied group gates from both destination roles. This preserves [0,1] bounds but changes support and attenuation; it is not automatically equivalent to same-type banking.

The source-state versions of those routes differ, and averaged geometry/representation may be unsuitable across types. Test this extension separately only after same-type merging is useful. Do not interpret the mixed descriptor as an averaged physical module/environment value or force cross-type merging to reach a predetermined K.

### 10.3 Query scoring must itself be compact

Compute decoder query-to-group logits directly from q0 and live p_k, with an explicit new group-scoring capability. The selected scorer is a shared normalized query/group dot product plus a small relative-geometry MLP, with separate destination-bank calibration and one entmax over the concatenated valid K slots. Its geometry uses the group's control centroid and extent in the declared physical frame; donor coordinates remain original in every physical value. Copy compatible query/key/geometry projections from the selected parent, initialize only unmatched tensors with seed0, and record the complete common/new/dropped parameter map. Do not compute N native seed logits and pool their probabilities on every deployed query. That latter construction is an exact-reference or transition tool, not the compact endpoint.

The compact scorer is trained by the retained native objectives, not by treating old predictions as physical truth. Identical membership rows do not make its probabilities equal to summed old entmax weights. At K=N, diagnose the scorer and changed retrieval-identity convention before attributing any loss to compression. Reusing an existing projection is initialization only.

Retain the pre-E routing order to avoid circular dependence. Final M information attention still uses q_E after reading E. Cheap query-to-record scores and normalization used by the retained native reader may remain O(QN); this residual is reported, not hidden.

## 11. How to initialize, learn and compact without abrupt uncontrolled changes

### 11.1 Native initialization

The selected comparison source is thermal_u_h_full5000_v1/checkpoints/epoch_1000_model.pt, with its saved full-TRAIN600 input capability, A2-D TRAIN-only normalization, physical heads and source-specific weights. Freeze its hash and complete identity in the future run manifest. This selects the latest evaluated parent rather than an unreviewed latest alias; it does not stop or alter the parent's running experiment. Initialize one organizational control descriptor per native record, with lineage identity S = I and no learned native-ID embedding. A different parent or fresh initialization is a separately named design amendment, not an automatic fallback.

This native initialization is a boundary condition for discovering redundant groups, not a requirement to recreate N groups forever. Bound any inherited full-directory capture to a small fixed input panel and a short declared transition phase. Do not attach full-directory diagnostics to every scientific forward.

For Level U, first qualify the factorized contextual equations with K = N on small native inputs. Inherited weight copying is initialization, not function equality. Adaptation must account for this architecture change before attributing any further degradation specifically to merging.

### 11.2 Scene-conditioned group descriptors

For the first design, a live group's control descriptor is a shared projection of the arithmetic mean of its ancestral original-record states, its input-coordinate centroid, per-axis extent, and a destination-bank type embedding. Group means use lineage multiplicity, not physical heat or quadrature weights; group sizes and extents are organizational controls rather than physical source aggregation. For a route at stage v, use only the destination states available at v and bind the resulting descriptor to that source-state version. Recompute descriptors at every applicable contextual stage in Level U and after preparation for Level R. A centroid used for control scoring does not replace donor coordinates in individual physical values.

Use shared, permutation-equivariant parameterizations across scenes. Do not store trainable per-case/per-native-ID groups. For unseen inputs, infer an organization from input records; never consult target fields or checkpoint-specific case errors.

The descriptor and assignment calculations must consume the live K-axis once compacted. A soft assignment S with N-by-K size can be transient, but rebuilding an N-by-N native directory to create it is forbidden in the compact path.

### 11.3 Candidate generation without another quadratic hot path

Use one typed binary geometry hierarchy, built once per exact input geometry, instead of an approximate-neighbor service or a trainable per-case clustering table. Build M-destination and E-destination trees separately. Canonically sort distinct normalized coordinate tuples lexicographically in the declared channel frame. At each hierarchy level, pair adjacent entries left-to-right and carry an unpaired final entry unchanged to the next level; repeat until a single root remains. This gives depth at most ceil(log2(number_of_bundles)) and defines odd-count pairing without record-order ties. Exact-coordinate ties remain unsplit terminal bundles; v1 leaves every record in such a bundle unmerged and does not pick a representative by native ID. Different records at the same coordinate are therefore not incorrectly declared interchangeable. Sorting costs O(N log N), hierarchy metadata costs O(N), and geometry-dependent ordering changes remain explicit topology seams. This candidate recipe is permutation-equivariant but does not claim arbitrary rotation invariance of a channel with fixed boundaries.

Only sibling nodes are merge candidates. Each node carries a count, centroid, extent, and a symmetric mean of input-only local descriptors. Build these by bottom-up reductions without evaluating candidate membership rows. A shared, symmetric compatibility MLP f receives the child descriptor sum, absolute difference, absolute separation and extents. It has the same weights for every scene within a destination bank. The merge logit is b_v=f(features(left,right)); the relaxed compatibility is t_v=sigmoid(b_v). Geometry determines candidate availability, while learned compatibility determines acceptance. A geometrically local hierarchy restricts the merge family: it can miss distant functional redundancy, and no global-optimum or arbitrary-group-discovery claim is made.

The frozen topology policy deliberately consumes the local input capability rather than contextual states, avoiding a circular requirement to compute a full native encoder before choosing compact groups. Thermal E inputs lie on a fixed grid and their local capability removes module-distance/occupancy summaries. Consequently, for fixed domain, material and operating inputs, this first E cut does not acquire explicit dependence on module translations; M cuts can change with module geometry, and the live group descriptors/memberships still depend on the model's contextual states. Scene-dependent K is permitted, not guaranteed to vary with every layout perturbation. Do not introduce a hidden layout-summary branch to strengthen that claim. A later context-conditioned coarsener would need its own noncircular compact preparation design and separate comparison.

The selected v1 structural scope is one sibling level above singleton terminal leaves. A pair is eligible only when both children contain one record; a coordinate-tie bundle stays unmerged. Thus a record can join at most one sibling in the first compaction, K=N-number_of_accepted_pairs, with K>=ceil(N/2) and K_X>=ceil(N_X/2) in each nonempty destination bank X. A particular tree can have fewer eligible pairs and a stricter attainable bound. The resulting reduction is deliberately modest and cannot justify the illustrative K=32 example below. Higher-level merging requires a separately reviewed extension after this primitive is useful. This makes the first learned cut inspectable and prevents an unchecked cascade of individually acceptable merges.

During soft tying, blend each leaf's live control descriptor toward its sibling-pair live descriptor using a(s)*t_v, retaining the allocated leaf slots. The local phase factor a(s) is zero during common warm-up and rises linearly from 0 to 1 during epochs 21-40; the input-only compatibility MLP is separate from these live contextual descriptor projections. Native task gradients reach t_v through the blended descriptors. Compute the typed membership-discrepancy penalty on the actually blended child rows already constructed by this transition forward, with equal weight for the two donor types when present; do not add pre-blend directory captures. The original physical values are never blended. This temporary uncompressed phase is charged at its actual N-slot cost. At structural activation, the shared hard rule merges a sibling pair when b_v>0, otherwise retains its leaves; b_v=0 means no merge. The whole policy, rather than case-specific merge overrides, must pass the combined fixed-panel field/gradient/support tests. If it fails, preserve the failure and leave hard compaction disabled; do not search a DEV-tuned threshold grid.

After activation, construct descriptors only for the resulting live cut and build only its K-by-N membership rows. The hierarchy and compatibility evaluation use input descriptors and do not require a native N-by-N directory. The selected bounded-depth construction is O(N log N) and policy evaluation is O(N); record actual depth, sorting and host/device-transfer costs. No hierarchy lookup or grouping is repeated per query tile. A bounded old-directory reference is allowed only for diagnosis and soft-transition checks, never as a deployed grouping prerequisite.

### 11.4 Soft tying, then a structural event

Use smooth redundancy pressure to bring accepted candidate groups closer. Keep native task losses dominant. Do not discretely delete groups in the middle of an optimizer update or prepared-scene lifetime.

At one declared review boundary, preserve the current checkpoint, activate the complete proposed sibling-cut policy in a candidate, and perform the fixed numerical/field probes. Accept the policy only when its combined effect is acceptable; independently harmless merges need not remain harmless when combined. No accepted/rejected per-TRAIN-case merge catalogue is deployed.

After acceptance, physically compact live arrays and continue adaptation. Model parameters, group-state maps and execution plans change version; all prepared scenes become invalid. Keep the pre-merge checkpoint as the explicit rollback state during the trial.

Use shared scoring/calibration parameters with scene-owned group descriptors, so reducing a scene's K does not replace neural modules or AdamW slots for every training example. Name and own the input-only coarsener encoder/f separately from the live contextual group descriptor projection/query scorer. At accepted structural activation, freeze the former; subsequent native adaptation trains the latter, memberships and inherited native model. Thus weight updates do not silently change the learned cut policy, while new geometries still generate scene-dependent cuts. Save the frozen policy, hierarchy recipe, activation version and structural schedule in the model checkpoint; regenerate scene plans from inputs, and retain in-flight plan/RNG information required by an exact mid-update resume. The hard topology decision has no ordinary autodiff gradient; this is soft learning followed by a structural transition, not globally differentiable optimization of integer K. Do not learn a lookup table of permanent TRAIN-case groupings and call it a generalizing group generator.

When any explicit per-slot trainable parameter is introduced, define its optimizer transition separately: retain unchanged shared AdamW moments, remove retired-slot states, and initialize or explicitly transform merged-slot states under a named policy. Averaging weights or moments is not an exact optimizer resume. Apply the same optimizer treatment to the comparison control and report the transition.

### 11.5 Avoid converting soft merging into a new geometry seam

Discrete, scene-dependent merge decisions can jump under arbitrarily small geometry perturbations. A continuous gate inside each group does not remove that new boundary.

For the first prototype, test derivatives on fixed-topology intervals, rebuild legitimate geometry metadata, and separately probe coarsener topology changes. A robust deployment option is a fixed compact capacity with soft slot participation; that can preserve smoother behavior but computes its allocated capacity, not necessarily the number of nonzero slots. Another is a sparse hierarchy with smooth sibling blending; its additional work and remaining transition rules need testing.

Do not silently freeze stale assignments for perturbed geometry and call the resulting derivative the full live-model derivative. Do not promise global C1 continuity across hard group-count changes.

## 12. Training objective and collapse control

Use the existing native objective and separately account for compression terms:

\[
\mathcal L
=\mathcal L_{\mathrm{native}}
+\lambda_{\mathrm{work}}\mathcal L_{\mathrm{actual/proxy\ work}}
+\lambda_{\mathrm{tie}}\mathcal L_{\mathrm{redundancy}}
+\lambda_{\mathrm{capacity}}\mathcal L_{\mathrm{group\ capacity}}.
\]

The work term retains separate M, E and contextual scopes. A surrogate is explicitly a surrogate; actual positive support remains an integer measurement.

One possible redundancy term is a candidate-restricted fused penalty:

\[
\mathcal L_{\mathrm{redundancy}}
=\sum_{(u,v)\in\mathcal P}\eta_{uv}
\left(\omega_M d_M(u,v)+\omega_E d_E(u,v)\right),
\]

where candidate weights eta are determined by input/usage compatibility and do not depend on favorable target errors. Avoid a global all-pairs attraction that collapses every group into the same broad directory. Retain task gradients through descriptors, memberships, query scoring and contextual exchanges.

For group-capacity pressure, a continuous participation gate can provide a differentiable occupancy surrogate during soft tying. Its mass is not allocated K and not measured work saved. Reclaimed computational capacity is counted only after unused slots are physically removed or skipped by a demonstrated sparse executor.

The selected v1 capacity surrogate is K_soft=N-sum_v sigmoid(b_v), summed only over the disjoint eligible singleton sibling pairs. It equals expected K under hypothetical independent Bernoulli merges with those probabilities, but the selected model samples no such merges and deploys the deterministic b_v>0 rule. Use equal-case K_soft/N as the weak capacity penalty; the forward still allocates N slots during soft tying. This closed-form surrogate does not require a global assignment matrix, recursive sampling, a fixed K quota or an additional dual controller. Redundancy pressure compares the sibling membership rows by donor type, while task gradients through the blended descriptors oppose harmful tying. Calibrate the two weak penalty scales once from input-selected TRAIN gradient magnitudes relative to the native objective, retain that calibration record, and freeze the scales before any DEV comparison. These are engineering calibration choices, not measured optimal values or physical constants.

A row-stochastic assignment's L1 norm is constant, so adding its plain L1 norm does not encourage fewer groups. Assignment entropy encourages decisiveness but does not by itself reduce the number of occupied groups. A mandatory equal-size balance loss can prevent useful compression. Choose the objective for its actual effect, not its name.

Start with task-only adaptation of the new representation, then ramp weak redundancy/capacity pressure, then compact once at a review and recover. Do not simultaneously introduce new losses for physical laws, new local radii, E cutoff policies or a different optimizer. An ordinary field miss is a result; nonfinite computation or a semantic violation blocks only the affected candidate.

Exact-zero group gates can have zero local derivatives. Closed groups are not guaranteed to recover merely because dense values are computed. Use identity/native initialization, gradual pressure and shared parameters, and explicitly test recovery or acknowledge its absence.

### 12.1 Level U uses a new factor-work schema

The existing native contextual controller counts original receiver-donor incidence. It cannot be carried into Level U by reconstructing R*d and measuring its positive entries: that would restore the N-by-N allocation and work this capability is intended to remove. Select compact_context_factor_work_v1 for Level U, with separate donor-factor and receiver-factor counters for each typed route/head, exact-positive d and R support, their smooth support proxies, and allocated score/contraction rectangles K_X*N_Y and N_X*K_X. Report group count, padded group capacity and dense executed factors independently. A composed original-record support trace is permitted only as a bounded diagnostic, not an ordinary loss or controller dependency.

For the first Level U factorized comparison, preserve the parent's final-reader M/E work definitions and targets, but replace the native context objective explicitly. Reset the new context duals equally in the K=N factorized control and compact candidate; retain the same case/request accumulation and equal-case macro denominators. Use one combined context constraint formed from the equal-weight mean of per-route donor-factor and receiver-factor proxy retention, with exact-positive retention as controller feedback; do not let E or head duplication dominate by raw cardinality. Nonempty eligible factors define each denominator and empty routes contribute no spurious penalty. The selected initial target is 1.0 with context pressure disabled during the first task-recovery phase; activate one frozen target only if a TRAIN-only support/field diagnosis justifies further factor pruning. No numerical pruning target is imported from the native pair controller as though its meaning were unchanged.

This deliberately separates two sources of cost: structural K reduction changes dense allocated factor work, while smaller positive d/R support changes logical factor support and helps execution only when the selected backend actually skips it. The v1 dense factor contractions charge their entire rectangles regardless of positive support. Capacity pressure drives the former; a context gate proxy does not establish the latter. Version this objective change in configuration and checkpoints, and include factor-work derivatives and replay in real TrainingEngine qualification.

## 13. Making K portable to new scenes

Do not hard-code one globally fixed number of hyperedges as a claimed physical truth. Use a scene-conditioned group generator with a declared capacity policy, such as an upper budget depending on N and typed inventory. The realized K may vary by scene within that budget.

For the first implementation, use exactly the same-type singleton-sibling policy from Section 11.3 and K_soft from Section 12. No recursive higher-level merging, hard target K or additional candidate search is enabled. Group descriptors use the declared lineage means; final membership must still be computed from current stage states. The hierarchy constrains ancestry rather than donor membership: a group may retrieve original donors outside its ancestral sibling pair.

At inference, constructing N cheap local seed descriptors is permitted. Reconstructing every old seed's N-member row is not. Report the coarsener's input-feature, candidate-search and assignment costs even when no dense matrix is materialized.

Keep grouping invariant to input record permutation, up to a corresponding permutation of internal group slots. Exact ties need symmetric behavior or a documented indeterminate case; never make geometry behavior depend on arbitrary numeric source IDs.

For empty types and small scenes, retain explicit valid behavior without a hidden all-record decoder fallback. A small scene may legitimately retain K = N. Compression is an empirical decision, not a compulsory per-scene quota.

## 14. Complexity and actual compute claims

The table suppresses head/width constants and separates preparation, routing and values. It assumes fixed context depth and regular factor contractions; sparse positive coefficients alone do not change dense-kernel complexity.

| Scope | Native U-H | Level R | Compact Level U |
|---|---|---|---|
| Context pair scores | O(N squared) | O(N squared) | O(NK), subject to structured eligibility |
| Additional native retrieval directory | O(N squared) | No extra native directory; encoder tensors remain | None |
| Group membership construction | Native contextual rows | O(KN) from prepared states | O(KN) per typed stage |
| Query-group scores | O(QN) | O(QK) | O(QK) |
| Dense access composition | O(QN squared) | O(QKN) | O(QKN) |
| Retained native query-record scores | O(QN) | O(QN) | O(QN), unless separately redesigned |
| Nonlinear query values | Actual positive original records | Same semantics, measured support | Same semantics, measured support |
| Coarsener/merge discovery | Absent | Additional, priced | Additional, priced |

For N = 204, Q = 8192 and hypothetical K = 32, the routing-product scalar multiply count falls from Q*N*N = 340,918,272 to Q*K*N = 53,477,376, a 6.375-fold reduction in this isolated product. This is shape arithmetic, not measured end-to-end speedup. The example K is illustrative and is outside the first sibling-only prototype's attainable reduction; that prototype has K>=102 for 204 live records and at most a twofold reduction of this product. Count only live eligible records when predicting a particular scene's bound, and also report actual batch padding.

The new operator's uncorrected per-head donor matrix has rank at most K_X on a typed route. Diagonal self-removal can raise that matrix's rank, and residual/FFN paths mean there is no corresponding rank bound on the whole predictor. Nevertheless, factorization imposes a real contextual capacity restriction. A useful Level R result does not establish that Level U will preserve surface/material or peak information.

Dense Q-by-K times K-by-N multiplication still costs O(QKN) even when entmax output has zeros. An optional packed group executor may exploit active groups, but its gather, deduplication, synchronization and packing cost must be charged. Do not make the first prototype depend on a custom CUDA kernel.

Encoder factorization and smaller K can increase final support or require more original values. Conversely, a faster dense-weighted native reader may compute every value despite sparse logical access. Report all of these outcomes separately.

## 15. Worked example: merge two groups without merging their records

Consider original records M1, M2, M3, E1, E2, E3 and E4. The following numbers are a synthetic access-algebra example, not trained or physical data:

| Group | M1 | M2 | M3 | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1.00 | 0.80 | 0 | 0.60 | 0.40 | 0 | 0 |
| B | 0.98 | 0.78 | 0 | 0.62 | 0.39 | 0 | 0 |
| C | 0 | 0 | 1.00 | 0 | 0 | 0.70 | 0.60 |

Groups A and B have close member coefficients; C serves a different region/role. For equal weighting, C_AB = (A+B)/2. If a query uses pi_A = pi_B = 0.4 and pi_C = 0.2, replacing A/B by pi_AB = 0.8 gives exactly the same access vector. If another query uses 0.7/0.1/0.2, that replacement is generally approximate. Thus near-identical membership is useful, but query mixture and learned compact scoring still matter.

~~~mermaid
flowchart LR
    Q["Receiver query"] --> OLD["Before<br/>A: 0.4, B: 0.4, C: 0.2"]
    Q --> NEW["After<br/>AB: 0.8, C: 0.2"]
    OLD --> V1["Seven original records<br/>Individual query-conditioned values"]
    NEW --> V2["The same seven original records<br/>Individual query-conditioned values"]
~~~

**Figure 4. Exact access equality in one special query mixture.** The example does not claim a globally exact merge, equal encoder states, or identical source-response coefficients after training. In the physical model, M1/M2/M3 remain three independent source columns.

## 16. Suggested implementation interfaces

Keep one scene-owned compact organization behind the existing prepare/decode boundary. Callers should not coordinate assignment reconstruction, contextual masks or retired native groups.

~~~python
from dataclasses import dataclass
from typing import Mapping
import torch
from torch import Tensor

@dataclass
class CompactOrganization:
    # Proposed contract; not an existing repository class.
    record_ids: Tensor
    record_types: Tensor
    group_mask: Tensor
    group_control: Tensor
    retrieval_membership: Tensor       # [batch, K_live, N]
    stage_factors: Mapping[str, object]
    source_state_versions: tuple
    # Optional compressed lineage for interpretation, never a native B bank.
    lineage: object

def compact_access(query_group_logits: Tensor, membership: Tensor,
                   group_mask: Tensor, record_eligible: Tensor,
                   entmax15_impl) -> tuple[Tensor, Tensor]:
    """Illustrative core only; query masks/empty scenes belong in its wrapper."""
    allowed_groups = group_mask[:, None, :].expand_as(query_group_logits)
    pi = entmax15_impl(query_group_logits, allowed_groups)
    access = torch.bmm(pi, membership)
    access = access * record_eligible[:, None, :].to(access.dtype)
    return access, pi

def explicit_merge_reference(pi: Tensor, native_rows: Tensor,
                             assignment: Tensor) -> Tensor:
    """Diagnostic only: useful for tests, forbidden in compact deployment."""
    usage = pi.mean(dim=1)                          # [batch, N_seed]
    weights = usage[..., None] * assignment        # [batch, N_seed, K]
    denominator = weights.sum(dim=1).clamp_min(1e-12)
    compact_rows = torch.bmm(weights.transpose(1, 2), native_rows)
    compact_rows = compact_rows / denominator[..., None]
    compact_pi = torch.bmm(pi, assignment)
    return torch.bmm(compact_pi, compact_rows)
~~~

The explicit merge reference is deliberately labeled as nondeployable. It constructs from native rows and can be useful for access-distortion tests; its clamp does not define zero-use group initialization or a production fallback. The compact model must instead call build_live_membership with K descriptors directly.

~~~python
def build_live_membership(group_control, record_states, geometry, eligibility,
                          route_score, route_calibration, positive_gate):
    """Direct K-by-N control construction; no native seed-row expansion."""
    logits = route_score(group_control, record_states, geometry)
    # Expected: [batch, heads, K_live, N_donor].
    assert logits.shape[-2] == group_control.shape[1]
    return positive_gate(route_calibration(logits, eligibility)) * eligibility

def factor_context(receiver_to_group, donor_coefficients, donor_values,
                   eps=1e-3):
    """One-head reference, without optional self-exclusion correction."""
    # R: [batch, N_receiver, K], d: [batch, K, N_donor].
    # V: [batch, N_donor, width]; never build [N_receiver, N_donor].
    group_numerator = torch.bmm(donor_coefficients, donor_values)
    group_mass = donor_coefficients.sum(dim=-1, keepdim=True)
    numerator = torch.bmm(receiver_to_group, group_numerator)
    mass = torch.bmm(receiver_to_group, group_mass)
    denominator = mass + eps * (1.0 - mass).square()
    return numerator / denominator, mass / denominator
~~~

These sketches do not implement eligibility, multihead bias, source masking, empty types, diagonal corrections, storage ownership or native losses. They identify the intended tensor axes and complexity. They must not be copied into a runner and treated as a qualified model.

At the public interface:

~~~python
prepared = model.prepare(scene, diagnostics=False)
prediction = model.decode(prepared, native_queries)

# A bounded explicit request, separate from ordinary training:
trace = model.describe_organization(prepared, query_ids=fixed_query_ids)
~~~

prepared binds the exact scene, record inventory, normalization, model weights, compact topology and execution settings. It is invalid after a model update or accepted compaction. Trace allocation is opt-in. Autograd must retain the live routing derivatives even when no trace is requested.

## 17. Repository implementation map and alternatives

| Existing area | Suggested responsibility | Keep out of the caller |
|---|---|---|
| continuous_joint/unified.py | Select legacy or compact capability; manage stage/source ownership | No case-name dispatch in numerical kernels |
| continuous_joint/preparation.py | Factorized typed contextual primitive and independent dense reference | No hidden fallback to full native attention |
| continuous_joint/routing.py | Query-to-live-group access and final local-envelope composition | No reconstruction of retired native seed logits |
| continuous_joint/incidence.py | Reuse calibrated continuous gates; add compact group-to-donor contract | No assumption that group_count equals record_count |
| A small new compact_groups module, if justified | Live descriptors, assignments, proposals and compaction state | No second training engine or campaign service |
| continuous_joint/reader.py | Retain original query-conditioned value execution and physical scalar law | No pooled group value substituted for E/M messages |
| Maintained research runtime | Local schedules, explicit capability, save/load and exact resume | No automatic formal launch or large model portfolio |

Three alternatives should be distinguished:

| Alternative | Advantage | Limitation | Recommendation |
|---|---|---|---|
| Prune unused native seeds only | Smallest intervention | Does not merge overlapping useful groups; encoder remains dense | Useful diagnostic/control |
| Soft merge final directories | Isolates decoder organization; smaller query bank | Does not eliminate old contextual scoring; approximate scorer transition | First model-level milestone |
| Direct compact contextual factors plus reused retrieval | Targets both organization stages and removes native pair scoring | Changes attention equations; requires new representation and derivative tests | Architectural endpoint after qualification |

Do not simultaneously benchmark an unbounded set of merge losses, coarseners and controller schedules. Use one simple implementation per level and only advance when the previous measurement identifies a useful remaining problem.

## 18. Numerical tests and structural checks

### 18.1 Mathematical correctness of the new function

- Independent FP64 literal contextual loop versus factorized R/d contractions, including bias and message mass.
- Exact identical-row merge, approximate-row error calculation and failure of the naive averaged-logit equivalence.
- Dense same-function versus compact/packed value executor, with parameter and coordinate VJPs/JVPs.
- Self exclusion by explicit masking versus diagonal correction, including cancellation and one-record scenes.
- Empty support, empty type, zero information measure, invalid padding and absent donors; preserve explicit zero-gradient paths where the optimizer contract requires them.
- Query duplicates, query-alone/chunked/mixed batches, source permutation, variable M/E and group-slot permutation.
- Group support entering/leaving zero, entmax ties, and all-zero mass limits under fixed compact topology.

### 18.2 Semantic and physical preservation

- Exact original source-column order, zero heat, heat scaling/superposition and heat-independent flow.
- Native surface/material extraction, curl, q-proxy and independently recomputed module peaks.
- Frozen complete contextual-plan disconnected-value tests versus separate live-control sensitivity tests.
- Geometry changes rebuild all legitimate metadata and group descriptors; test fixed-topology intervals separately from coarsener transition events.
- No target/reference data in descriptors, assignments, candidate generation or query routing.
- Save/load, accepted-compaction resume, per-scene group padding/masks, stale prepared-state rejection and optimizer-state treatment.

### 18.3 Proving the compact path is actually compact

Instrument score builders and ordinary allocations. Verify that group-donor score dimensions use K, query-group dimensions use K, and no ordinary compact operation calls the native N-by-N contextual score builder or reconstructs its directory. A source-code promise or zero-mask inspection is insufficient.

For variable K batching, report sum(K_case) and allocated batch padding. Padding every scene to the original N or to an unnecessarily large global maximum can erase savings. One bounded scene loop at preparation can be a correct first implementation; its CPU cost remains measured.

Count all positive M/E values, rectangular dense values, group/donor score pairs, receiver/group score pairs, contextual projections, grouping work, temporary reference calls and backward recomputation. Do not alias logical counts to executed counts when the backend differs.

Retain the known numerical limitations of legacy source-order/segment comparisons. New-function correctness is tested against its own independent reference; old failures are neither retroactively passed nor used to avoid diagnosing a new failure.

## 19. Bounded training and evaluation strategy

### 19.1 Stage sequence

1. **Read-only diagnosis:** inspect a fixed input-selected M1/small/medium/high-M panel; measure redundancy, typed support, query usage, descriptor cost and existing per-field errors. Reuse existing arrays when their identity matches.
2. **Primitive implementation:** build same-function test fixtures and a real native forward/backward qualification before any fit.
3. **Level R adaptation:** a common uncompressed representation warm-up, followed by one K=N Level R control and one sibling-tying candidate with matched initialization, native loss and case/query budget; retain the native H parent as a frozen inference reference.
4. **One compaction review:** accept only the combined shared sibling-cut policy, then recover at the same declared local schedule. Do not repeatedly tune K or per-case decisions against DEV.
5. **Level U conditional prototype:** only if Level R is useful and encoder cost remains material, compare compact contextual equations with a matched uncompressed-factorized control. This separates factorization cost from merging cost; it is not an additional large portfolio.
6. **Decision:** retain the measured useful representation, a partial retrieval-only improvement, or a negative result. Prepare a separate formal-run recommendation if justified.

This is an ordering recommendation, not authorization to launch all stages. Freeze the actual allowed trajectories, training hours and stop horizon in a later execution request. Keep active full-data U-H/U-D/U-S runs untouched.

### 19.2 Data and local age

For this retained full-data comparison direction, select a separately named matched adaptation from the exact U-H epoch 1000 parent on its original TRAIN600 and exposed canonical89, preserving the saved TRAIN600 normalizer. This is a proposed future execution recipe, not authorization to train. Do not refit normalization, switch to TRAIN150/DEV22, or add a portfolio of subset screens under the same identity. The repository's quarter-data default remains available for a separately requested fresh development experiment, with its own initialization and response contract; it is not a silent alternative to this adaptation.

The proposed bounded Level R pair uses 100 local adaptation epochs, each visiting all 600 training cases, effective batch48, microbatch24, Q1024, FP32 and seed0. Retain the parent's case/query sampling and response addendum exactly, including eligible response anchors and native objective weights. Both arms use fresh AdamW state with the same saved optimizer-group/LR recipe and local schedule; carrying weights is transfer initialization, not optimizer resume. Copy the parent M/E/native-context controller state and continue its scientific controller clock from 1000, independently recording the new local adaptation age. Group-tying pressure uses local age and must not inherit a1000-epoch wait. A future launcher must serialize these separate clocks explicitly rather than overload an epoch offset.

Here1000 denotes the parent's scientific sparse age, not the raw stored controller coordinate, which uses the existing3000 offset. Load that coordinate from the checkpoint and advance its age consistently. Bind sampler epochs 1001-1100 for the proposed adaptation while recording local epochs 1-100 separately; both branches reuse the same corresponding case/query streams. Never feed raw epoch 1000 into a controller that requires its 3000-based coordinate convention.

Each epoch has twelve48-case optimizer macros and one final 24-case macro. Accumulate two24-case microbatches with the macro's48-case loss denominator for ordinary macros; normalize the final macro by its actual24 cases, without phantom padding or duplicate cases. Preserve the inherited equal-case native and work loss reductions and once-per-epoch response placement. Record case/query hashes and partial-macro handling in both children so altered packing does not silently change the comparison.

Use a common first 20 local epochs to adapt the uncompressed Level R representation and scorer. Save its common state, then fork one K=N Level R control and one soft-tying candidate with identical copied weights, optimizer state, controller state and future case/query streams. During epochs 21-40, linearly ramp the fixed weak tying/capacity penalties only in the candidate. At the declared epoch 40 boundary, run the combined hard-cut probes and either activate/freeze the shared policy or preserve its failure without compaction. Epochs41-100 recover at the same remaining schedule; the K=N control receives the same native case/update budget. These20/40/100 phase boundaries are selected practical defaults, not convergence findings or an optimized horizon. Report common warm-up work once and the complete work of each arm thereafter.

The inherited native H epoch 1000 is a frozen inference reference; the trained K=N Level R arm is the matched compression control. This avoids attributing the compact scorer/identity transition to merging. Count the common phase in both arms'100-epoch scientific age but only once in physical resource totals: each full100-epoch arm represents60000 case visits and 1300 optimizer updates at the inherited packing, before response/auxiliary work. The shared 20 plus two80-epoch branches perform108000 visits and 2340 updates total, not120000 visits or 2600 physical updates.

Separate parent training age, experiment-local epoch and group-compression age. Do not let a new short run inherit a thousand-epoch waiting phase accidentally. A changed membership, normalization or group capability starts an explicitly labeled experiment, not an exact resume of a historical native-seed run.

No extension to 500, fresh full5000 fit or automatic Level U launch follows from this recipe. A later execution instruction must set actual GPU/time/disk allowances and freeze numerical tolerances and field/tail tradeoff bands before measurement. Retain the shared epoch 20 handoff, epoch 40 rollback/review and epoch 100 final states plus latest/best aliases; these are explicitly declared boundaries rather than a new25-epoch cadence. Refresh curves every10 epochs. A resource stop retains the exact attained phase and resumable state.

### 19.3 Field, organizer and inverse evidence

Report fluid/surface/material T, module peaks, u/v/p/omega and q-proxy by mean, p90 and populated M strata. In particular, do not let improved fluid T hide worse surface/material or peak errors.

Show low-M and high-M applied compact contextual factors, group lineage, final positive M/E access, K by scene, support-union inflation and group occupancy. Include dense/unhelpful cases rather than selecting only successful compression.

Replay a few old and newly found support transitions at represented step sizes. Repeat compatible saved heat/geometry responses and six-candidate ranking only when needed for the selected candidate; no new solver, independently forced environment truth, missing factorial corner or generative design is supplied by this proposal.

Use selected-candidate feasibility and original objective/constraint definitions. A better grouping score is not evidence of better physical response or design decisions.

### 19.4 Cost and storage

Measure complete same-function old/new executor scopes separately from independently trained model comparisons. Report preparation/coarsening, decoder forward, backward, optimizer, transfers, validation/output and cold-start costs with explicit inclusion boundaries. Use a disposable model/optimizer for engineering cost checks; do not mutate scientific runs.

No generated payload is needed to read or implement this design document. Before any future output-producing work, verify the prescribed /data2-backed workspace symlinks and mount, record measured free capacity, and declare hard retained/temporary/peak budgets and headroom. Reuse exact shared data/cache identities; do not duplicate datasets or complete native catalogues for a small probe.

The future round should retain selected checkpoints, full history, compact numerical summaries and a few inspected PDF figures. Delete only its verified disposable profiling/test outputs under the standing allowlist rules. No historical checkpoint, active-run artifact, required reference or migration backup becomes disposable because group compression is introduced.

## 20. Failure modes, stop conditions and honest conclusions

| Observation | Interpretation | Response |
|---|---|---|
| K decreases but final support widens | Organizational compression is not value sparsity | Inspect union inflation; do not report sparse work gains |
| K decreases but time grows | Coarsening/irregular execution outweighs savings | Keep numerical result; reject speed claim |
| Field means improve but peaks/tails fail | Incomplete physical-field preservation | Report/reject according to declared bands; no pooled-score pass |
| Factorized K=N control already loses fields | Attention reparameterization is a confound | Fix/assess factorization before blaming merging |
| Groups become identical broad sets | Collapse without useful organization | Reduce pressure or reject candidate; no global-summary bypass |
| Group switch creates a finite geometry jump | New topology-selection seam | Label scope; test smooth-capacity alternative before inverse use |
| Compact path still builds old pair scores | Compute objective not implemented | Remove dependency or restrict claim to Level R |
| Only selected queries get faster | Limited workload result | Report exact shapes and complete scope |
| Learned groups correlate with saved physics | Descriptive association | Do not infer physical causal edges |

A failed new numerical primitive blocks that primitive, not unrelated read-only analysis. A resource stop preserves the same experiment state. Ordinary negative scientific results do not authorize another architecture portfolio or indefinite training.

## 21. Proposed deliverables for a future implementation round

1. A clearly versioned compact organizational capability behind the maintained prepare/decode interface.
2. Direct K-group score/directory construction with original record/source axes preserved.
3. A working soft-tying and explicit-compaction transition with a resumable checkpoint contract.
4. A separate label and qualification for any factorized encoder function; no implicit equivalence to native U-H.
5. Independent numerical tests, compact-path allocation/call evidence and a bounded real native training result.
6. A short comparison of fields, tails, applied organization, actual work, complete cost, seams and existing decision diagnostics.
7. A clear decision about whether compact retrieval alone, full compact organization, or neither is useful.

Success is not guaranteed by fewer groups. A useful negative result can establish that native groups were not sufficiently redundant, that compact contextual factors cost too much accuracy, or that a simpler same-function executor remains preferable.

As a preliminary engineering estimate, one experienced developer using the maintained runner and existing infrastructure may need1-2 working weeks for Level R and approximately4-8 weeks for the complete adaptive Level U endpoint and numerical/resume qualification. These are scope estimates, not promises or measured throughput; a fixed-K primitive can be much quicker. Fitting, GPU availability and recovery after scientific regressions are additional. Do not extrapolate the isolated routing-product ratio into a wall-time or memory budget, and do not change the existing24-case/Q1024/FP32 protocol merely to increase VRAM occupancy.

## 22. Source map, external precedents and visualization index

### Repository evidence read for this proposal

- [Round11 development report](../../docs/reports/HONF_Round11_Thermal_Development_Report.md): completed development fields, actual work, organization, response/ranking and executor limitations.
- Local, unpublished Round11 practical-development input: `../HONF_Round11_Thermal_First_Practical_Development_Work_Plan.md`. It informs ordinary research workflow and preservation boundaries; this two-document revision does not publish that separate user-owned plan. The tracked Round11 report and protocol guide below provide the durable evidence.
- [Round10 continuous/unified plan](../HONF_Round10_Continuous_Unified_Organization_Work_Plan.md): continuous gate, guarded normalization, E-first native values and applied unified preparation.
- [Thermal development protocol](../../docs/guides/Thermal_Model_Development_Protocol.md): fixed memberships, TRAIN-only normalization and review/retention rules.
- [Unified preparation implementation](../../src/honf_forward_core/continuous_joint/unified.py): applied contextual policy and reuse of actual last-cycle incidences.
- [Contextual value reduction](../../src/honf_forward_core/continuous_joint/preparation.py): actual coefficients, guarded mass, dense-weighted/segment execution and full-plan interventions.
- [Continuous incidence](../../src/honf_forward_core/continuous_joint/incidence.py): typed gate calibration and native retrieval identity convention.
- [Continuous routing](../../src/honf_forward_core/continuous_joint/routing.py): query-to-native seed weights, matrix composition and local M envelope.
- [Score calibration](../../src/honf_forward_core/joint_native/routing.py): convex head mixture, relative logits and bounded scalar calibration.
- [Continuous native reader](../../src/honf_forward_core/continuous_joint/reader.py): individual query-conditioned values and distinct execution backends.

These are source observations at document preparation, not an independent replay of scientific checkpoints or a claim about current formal-run progress. Resolve later source changes before implementation without resetting active work.

### Primary external precedents

[DiffPool: Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) learns differentiable soft cluster assignments and hierarchical graph representations. It motivates learning assignments rather than hard-coding groups. It does not validate this proposal's preservation of native decoder values, compact scorer, physical law or computation savings.

[MinCutPool: Spectral Clustering with Graph Neural Networks for Graph Pooling](https://proceedings.mlr.press/v119/bianchi20a.html) uses a continuous relaxation to learn clustering for pooling. It motivates differentiable grouping objectives; it does not establish that graph partitions coincide with useful Thermal query-retrieval groups.

Borrow assignment and compression ideas selectively. Do not transplant pooled node features into the physical decoder or impose balanced clusters as physical truth. All compact-context formulas, merge decisions and training recommendations in this note are project-specific proposals requiring their own tests.

### English visualization index

| Figure | Purpose | Evidence status |
|---|---|---|
| 1. Existing applied organization | Connect attention scores, encoder exchange and final retrieval | Source-derived schematic |
| 2. Native candidates to live groups | Separate merged organization from retained physical records | Proposed schematic |
| 3. Direct compact unified execution | Show K-by-N construction and removal of native pair-score dependency | Proposed architecture |
| 4. Synthetic merge algebra | Explain exact equality for one special query mixture | Illustrative numerical example |

All visuals are inline Mermaid source. No new raster, PDF, dataset, checkpoint, profiling trace or numerical export is created by this design note.

## 23. Document-level verification

For the 23 October revision, Markdown with tables was parsed and rendered to HTML in memory, all 11 local links resolved to this two-document change or existing tracked files, and prose paragraphs were checked for hard wrapping. All three Python sketches pass syntax parsing; the two definition blocks load in an isolated CPU namespace. Current in-memory FP64 checks compare the factorized contraction and VJPs with an explicit operator, verify diagonal self-removal, and verify the new disjoint-sibling capacity expectation and gradient by enumerating all merge outcomes. Shared-prefix case/update arithmetic was checked independently. Original proposal preimages and compact validation receipts are retained locally under /data2, outside Git.

Earlier checks of zero-mass behavior, merge examples and compact-access shapes remain elementary draft evidence; they do not qualify the new coarsener, compact scorer, factor-work controller, source checkpoint adaptation or save/resume implementation. No native model was executed, no fit or GPU operation was performed, and the revised capability remains unimplemented.

The third Python block is an illustrative future model-interface call, not an executable existing-model example. The four Mermaid blocks are retained as source; graphical Mermaid layout was not rendered or visually qualified in this documentation task. These checks establish internal consistency of the note and small algebra examples, not implementation completeness, native model parity, physical efficacy or measured savings.
