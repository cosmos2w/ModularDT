# HONF Redesign — 13 October 2026

## From pairwise computation with organizing controls to reusable, content-bearing hypergraph decoding

**Status:** Mathematical and architectural design proposal. This document does not launch, terminate, modify, or supersede the execution instructions of an ongoing experiment. A subsequent document should specify implementation stages, training budgets, acceptance criteria, and formal-run recipes.

**Research objective:** Learn from multi-field reconstruction, discover a reusable organization of module–environment information, and make that organization available for evaluating and eventually generating modular designs. Forward accuracy, computational efficiency, and physically useful organization are joint objectives; none is a substitute for the others.

**Evidence boundary:** The user's complete two-part proposal is the starting point [U1]. Repository inspection was pinned to `f10aa863a5faae8db5cdb969d16692729e79d7ce` on `agent/honf-core-next`, which was the remote tip retrieved during this review. That is newer than the preceding `a6e898d` discussion. Its committed consolidation report is considered below, but no local GPU experiment, uncommitted working tree, or runtime process was inspected or changed. Proposed formulas and design choices are not reported experimental results. References [R1–R15] support constituent methods, not the untested performance of this HONF design.

---

## 1. The central decision

The proposed direction is sound: **use standard attention to contextualize physical tokens, and put the hypergraph where it can replace expensive decoding work rather than merely modulate it.** The essential architecture should be

\[
\boxed{
\text{dataset-owned physical inputs}
\longrightarrow
\text{typed attention contextualizer}
\longrightarrow
\text{content-bearing hyperedges}
\longrightarrow
\text{receiver-conditioned edge decoding + local detail}
\longrightarrow
\text{dataset-owned physical outputs}.
}
\tag{1}
\]

This is a substantive redevelopment, not another gate added around the present dense source reader. Its principal compression is learned as part of field reconstruction. It should not require discovering, after training, that an arbitrary subset of a dense sum happens to be dispensable.

Three distinctions make the proposal viable.

**Attention is a computational building block; the hypergraph is the reusable representation.** The contextualizer need not itself discover a sparse physical graph. Its job is to give each module and environment token a competent description of its surroundings using efficient, well-supported operations. The decoding bank then organizes those descriptions into a small set of reusable computational units.

**An edge is not a scalar.** Six hyperedges can each carry a vector of field-relevant content and several independent forcing-response modes. Edge count, latent width, response rank, and the number of edges read by one receiver are different design variables. Confusing them would recreate the most serious compression risk in the pasted proposal.

**Compression and selection are separate.** First establish an accurate decoder using all of a compact edge bank. Then determine whether selecting fewer edges gives a worthwhile accuracy–work tradeoff. Full access to six prepared edges can already be much cheaper than constructing nonlinear messages from every physical source at every query; an initial model does not need a complicated adaptive edge-count controller.

The recommended initial mathematical family has a small fixed bank, such as **six hyperedges**, ordinary multihead softmax attention for contextualization, multiple thermal-response channels per edge, and one explicitly local source-detail path. A useful starting specification is given in Section 15. The exact widths are design proposals, not empirically selected hyperparameters.

## 2. What the sources support, and what needs correction

### 2.1 What is accepted from the user's proposal

The two supplied parts consistently advocate preserving physical input ownership, replacing pair-dependent value networks with donor-projected attention, preparing hyperedge content once, retaining source-resolved thermal response, and selecting edge IDs before expensive nonlinear edge reading [U1]. These are the foundations of this document.

The proposed global/local decomposition is also worth retaining. Interfaces, material interiors, and immediate near-body structure should not be forced through a handful of scalar global averages. However, the local branch must be a bounded detail path, not an unrestricted dense decoder that makes the edge bank optional.

Part 1 proposes a particularly cheap projected-edge read. Part 2 proposes a richer query–edge nonlinear read. These are different decoders, not two descriptions of the same operation. Section 7 distinguishes them and recommends a small shared edge-conditioned reader as the main design, with the simpler projected-value decoder retained as an architectural reference.

### 2.2 Four corrections to avoid another misleading reset

**First, attention is not a guaranteed accuracy-preserving substitution.** The current value function depends nonlinearly on destination, donor, and signed relative geometry before reduction. Standard attention usually projects donor values independently of the destination and puts pair dependence in scores. This removes considerable work, but changes the function class at fixed width and depth. Geometry, tokenwise nonlinearities, multiple heads, residual connections, and retained native tokens are therefore important; the name “Transformer” alone does not replace the lost pair-value expressivity.

**Second, modern attention does not automatically mean linear attention.** Fused exact softmax attention and kernel-based linear attention make different changes. The former changes execution and memory traffic while retaining the softmax formula; the latter changes the attention kernel or its representation. For current short module/environment catalogues, ordinary cross-attention is the stronger default. Large point-cloud or boundary catalogues can motivate linear or local/global variants later [R1, R2, R7, R8].

**Third, the latest P/R1 decoder does not directly read every environment token at every query.** Its nonlinear readout evaluates source-conditioned messages and a pooled/global context. Environment information reaches that readout through the prepared states. Dense query–environment reading is present in historical families, but should not be attributed indiscriminately to the current source-resolved core [C1, C2]. The new design removes the current repeated query–source value construction; it should not claim savings from eliminating a current query–environment stack that is not there.

**Fourth, six scalar global thermal modes cannot represent an arbitrary high-rank heater-response operator.** In the Part-1 formula \(K^{\mathrm{global}}=\Phi C\), a single scalar mode per edge imposes rank at most six. A more capable contextualizer does not remove this algebraic restriction. Sections 8–9 supply a multi-mode construction without increasing the number of graph nodes.

### 2.3 What changed in the latest repository snapshot

The latest committed report states that the existing Wind P/R1 pair completed its original e2500 horizon; R1 retained an approximately 5.31% transverse-priority improvement on its exposed development panel. More importantly for this redesign, its bounded final-read study found that neither frozen primary host admitted a useful omission frontier under the declared physical safeguards. No learned omission scorer was fitted [C5]. This is evidence against expecting post-fit deletion to solve the current representation's cost problem, not evidence against learning a different compressed representation.

The same report identifies a real Thermal classic-input problem: the earlier evaluator supplied raw heating/interface inputs where the checkpoints expected saved global normalization. A corrective helper was added, but remaining GPU interface-tolerance failures kept the mature Thermal board quarantined [C5, C6]. Consequently, this document does not use that disputed board to claim that the new family already beats the classics or that a particular large error proves an architectural failure.

The previous multiview experiment found no added R2 benefit and retained dense final reads [S1]. It therefore does not justify more custom views, an entropy target, or a more elaborate sparse controller as the main redesign.

### 2.4 Lessons retained, procedures not retained

The valuable results are the physical-input whitelist, source identities, joint trainability, exact case-qualified heat dependency, source-response semantics, native extraction, and explicit prepared-state ownership. The expensive pair-value implementation, recursive organization, historical frozen Stage-A chain, repeated exact-function retrofit, and uncontrolled all-source bypass are not architectural requirements.

Earlier advice was too focused on preserving a particular dense computation while adding organization or trying to remove its individual terms. The new proposal changes the computation being learned. It should be judged by physical fidelity and its declared laws, not by whether newly initialized attention reproduces an old trained pair-MLP bit for bit. Historical replay compatibility remains a separate obligation.

## 3. Relevant research: what to borrow and what not to infer

The redesign is a synthesis of established methods. Calling latent attention “hypergraph-inspired” does not by itself establish a new scientific contribution. The distinguishing HONF target is a typed, source-identifiable, receiver-addressable representation with explicit physical output and control laws.

| Primary source | Relevant contribution | Use in HONF; limit of the analogy |
|---|---|---|
| Set Transformer, ICML 2019 [R3] | Attention on unordered sets; inducing-point compression and attention pooling | Use permutation-respecting token processing and learned slots. It does not establish that six slots suffice for these physical fields. |
| Perceiver IO, ICLR 2022 [R4] | Decouples input processing from output queries through a latent representation | Use query-independent preparation and arbitrary output locations. Preserve physical measures and source-control identities explicitly. |
| GNOT, ICML 2023 [R5] | Heterogeneous input encoders, normalized linear cross-attention, geometric expert gating | Use typed inputs and an optional efficient attention backend. Do not copy query–query processing that makes an independently requested field value depend on unrelated queries. |
| Transolver, ICML 2024 [R6] | Learns slice memberships, pools content, processes slice tokens, and reconstructs spatial representations | Supports content-bearing organization rather than control-only groups. Its usual mesh-to-mesh construction is not automatically an arbitrary-query, source-resolved response interface. |
| Linear attention and Galerkin Transformer, ICML 2020 / NeurIPS 2021 [R7, R8] | Efficient reassociation and operator-learning attention formulations | Useful alternatives for genuinely large catalogues. Arbitrary pairwise geometry masks can destroy the factorization enabling linear cost. |
| Latent Neural Operator, NeurIPS 2024 [R9] | Physics-cross-attention between geometric and latent spaces, including off-grid decoding | Supports latent preparation plus separate spatial decoding. Its reported inverse tasks do not validate HONF modular-design responses. |
| GINO, NeurIPS 2023 [R15] | Geometry-aware input/output operators around a learned global representation | Supports separating geometry interfaces from the central representation. HONF need not adopt its Fourier latent grid or treat its geometry transfer as already established here. |
| Transolver++, ICML 2025 [R10] | Extends physical-state attention to million-point geometries with adaptive mechanisms and parallelism | Relevant if environment catalogues become large; not a reason to import its full machinery into present short-token problems. |
| LinearNO / “Transolver Is a Linear Transformer,” AAAI 2026 [R11] | Re-examines slice attention and reports that slice/deslice operations, rather than slice interactions alone, can explain effectiveness | A useful warning: explicitly test representation and edge processing separately. Do not require extra edge–edge attention just to make a graph look richer. |
| DeepONet, Nature Machine Intelligence 2021 [R12] | Branch/trunk operator representation | A useful analogy for coefficient–spatial-basis separation. HONF's heat branch must be linear in the declared affine controls; a generic nonlinear branch network would not preserve that law. |
| Sparse sequence models / adaptively sparse Transformers, ACL and EMNLP 2019 [R13] | Entmax mappings with exact zero weights and differentiable sparse attention | A possible later edge-selection law. Exact zero weights do not themselves imply fewer executed operations or lower wall-clock time. |

FlashAttention-2 and the official scaled-dot-product-attention documentation are relevant for implementation reuse [R1, R2]. Their reported performance is not a forecast for HONF: sequence length, masks, additive geometry bias, dtype, device, and backend eligibility all matter.

The most useful borrowed pattern is therefore **contextualize native tokens → prepare a compact bank → query that bank**, not a wholesale replacement with the latest named model.

## 4. General physical problem and reusable interface

### 4.1 Scene, controls, and receivers

Let the prescribed configuration be

\[
\mathcal G=(\mathcal M,\mathcal E,\xi,\Omega,\mathscr D),
\tag{2}
\]

with

\[
\mathcal M=\{(\mathrm{id}_i,c_i,a_i,\mu_i,p_i)\}_{i=1}^{M},
\qquad
\mathcal E=\{(\mathrm{id}^{E}_a,x_a,b_a,\nu_a,p^E_a)\}_{a=1}^{N_E}.
\tag{3}
\]

Here \(c_i,x_a\in\mathbb R^D\) are coordinates; \(a_i,b_a\) are typed prescribed attributes; \(p_i,p^E_a\) are presence indicators; \(\mu_i,\nu_a\) are positive integration or aggregation measures; \(\xi\) contains operating and boundary parameters; and \(\mathscr D\) declares allowed dependencies and output capabilities. Module and environment measures can have different physical meanings and are never silently added as if they had identical units.

Receivers are

\[
\mathcal Q=\{(q_j,\mathrm{role}_j,\mathrm{id}^{Q}_j,\omega_j)\}_{j=1}^{Q}.
\tag{4}
\]

The quadrature weights \(\omega_j\) belong to extraction or loss integration. They are not generally inputs to the pointwise field value. A requested field value should not change merely because the evaluation sampler changes its weights.

Actual controls are a separate object \(u\). For the current ThermalChannel capability, write \(u=h=(h_1,\ldots,h_M)\). These are physical source amplitudes, not arbitrary node features. For a nonlinear dataset, controls such as yaw, thrust settings, or actuation can instead be configuration inputs and require rebuilding the scene representation.

Physical IDs establish correspondence. They are not learned numerical embeddings that allow memorization of a layout or source ordinal. One actuator may later own several geometric tokens, but those tokens must map back to one physical control ID with an explicit ownership rule; duplicating tokens must not duplicate the actuator's heating.

### 4.2 Input encoders

Each dataset produces typed raw descriptors and fixed TRAIN-owned transforms. The common model lifts them into a shared width:

\[
Z_M^0=\mathcal E_M(a,c,\xi),
\qquad
Z_E^0=\mathcal E_E(b,x,\xi),
\qquad
z_{\mathrm{op}}=\mathcal E_{\mathrm{op}}(\xi).
\tag{5}
\]

The lifts may differ in input dimension and parameter values between datasets. The attention, edge preparation, and decoding algorithms should be the same implementation. The initial objective is architectural reuse with separate trained weights, not unsupported zero-shot transfer between Thermal and Wind.

For current ThermalChannel, \(Z_M^0,Z_E^0,z_{\mathrm{op}}\) are independent of actual heating. The prohibition includes indirect paths through normalized total heat, predicted heat-dependent ports, or heat-conditioned routing. For Wind, only quantities known at prediction time may enter; CFD target velocities are never environment features.

### 4.3 Which global information is allowed at decoding?

Distinguish a prescribed operating embedding \(z_{\mathrm{op}}\) from a learned whole-scene summary \(z_{\mathrm{scene}}\). The latter can help form edge seeds and edge content, but should not be passed as an unrestricted parallel input to a large final coordinate decoder. That would let the model encode the complete solution outside the nominal hypergraph.

The main decoder receives edge content, receiver geometry, prescribed conditions, and explicit local detail. If a learned global background token is required, count it as an explicit edge within the bank and include it in all work and intervention accounting. Wind's TRAIN-fitted physical background profile is a different, adapter-owned output convention and remains visible separately.

## 5. Efficient module–environment contextualization

### 5.1 Typed, measure-aware multihead attention

Retain three directions, using the existing convention:

\[
\mathrm{MM}: M\leftarrow M,
\qquad
\mathrm{ME}: M\leftarrow E,
\qquad
\mathrm{EM}: E\leftarrow M.
\tag{6}
\]

For route \(t:A\leftarrow B\), layer \(\ell\), and attention head \(b\), define

\[
Q_i^{t,b}=W_Q^{t,b}\operatorname{LN}(z_i^A),\quad
K_j^{t,b}=W_K^{t,b}\operatorname{LN}(z_j^B),\quad
V_j^{t,b}=W_V^{t,b}\operatorname{LN}(z_j^B).
\tag{7}
\]

Donor projections are evaluated once per token and route/head, not once per destination–donor value pair. Let \(\eta_j^B\) denote the donor measure and \(m_{ij}^t\) the eligibility mask. Then

\[
\ell_{ij}^{t,b}
=
\frac{\langle Q_i^{t,b},K_j^{t,b}\rangle}{\sqrt{d_b}}
+
\beta^{t,b}(x_i^A,x_j^B,\xi)
+
\log\left(\frac{\eta_j^B}{\eta_{\mathrm{ref}}^B}\right),
\tag{8}
\]

\[
\alpha_{ij}^{t,b}
=
\frac{m_{ij}^t\exp(\ell_{ij}^{t,b})}
{\sum_k m_{ik}^t\exp(\ell_{ik}^{t,b})},
\qquad
A_i^{t,b}=\sum_j\alpha_{ij}^{t,b}V_j^{t,b}.
\tag{9}
\]

The log measure uses a type-specific reference so its argument is dimensionless; any positive constant reference cancels within the row. Invalid or zero-measure donors are masked, not assigned a small artificial physical mass. An empty row returns a zero attention update. In particular, an M=1 MM row has no other module donor and must not produce an all-masked softmax NaN. Self information remains in the residual path.

Different heads have different learned projections. They are ordinary attention heads, not physical hyperedges, and no head is assigned a physical meaning by its index.

### 5.2 Relative geometry without another expensive pair-value network

Geometry should distinguish signed downstream/upstream, transverse and vertical relationships where the dataset defines them. A cheap headwise bias can use normalized relative displacement, distance, characteristic lengths, and boundary descriptors:

\[
\beta^{t,b}_{ij}
=
\vartheta_{t,b}^{\mathsf T}
\gamma_t\!\left(
\frac{x_i-x_j}{L},
\frac{x_i-x_j}{\lambda_i+\lambda_j},
\xi_{\mathrm{geom}}
\right).
\tag{10}
\]

A small shared geometry projection is acceptable. The prohibited regression is rebuilding a wide nonlinear \([z_i,z_j,\gamma_{ij}]\) value for every pair. Geometry in scores is a modeling choice, not a proof that all destination-dependent value functions are represented at the chosen width.

The geometry basis must be smooth enough for the intended coordinate derivatives. Module IDs, query ordering, and source ordinal are not geometry. Do not impose an upstream-only causal mask: diffusion, pressure, and whole-layout dependence are not generally one-way processes. Any hard physical exclusion belongs to the adapter's declared model, not to an aesthetic preference for sparse arrows.

### 5.3 Residual updates and information re-injection

Use standard pre-normalized residual blocks [R14]. For example, with route projections absorbing head concatenation,

\[
\begin{aligned}
\bar Z_M^{\ell+1}
&=Z_M^{\ell}
+W_{MM}^{\ell}A_{MM}^{\ell}
+W_{ME}^{\ell}A_{ME}^{\ell},\\
Z_M^{\ell+1}
&=\bar Z_M^{\ell+1}
+\operatorname{FFN}_M^{\ell}
\bigl(\operatorname{LN}(\bar Z_M^{\ell+1})\bigr),\\
\bar Z_E^{\ell+1}
&=Z_E^{\ell}+W_{EM}^{\ell}A_{EM}^{\ell},\\
Z_E^{\ell+1}
&=\bar Z_E^{\ell+1}
+\operatorname{FFN}_E^{\ell}
\bigl(\operatorname{LN}(\bar Z_E^{\ell+1})\bigr).
\end{aligned}
\tag{11}
\]

All route reads in a simultaneous layer use the pre-update states. Presence masks are reapplied after residual updates. Sequential updates are a different block definition and must not vary silently between datasets.

These residual paths already retain each token's own state. A lightweight original-feature re-injection can additionally be defined as

\[
Z_t^{\ell+1}\leftarrow
Z_t^{\ell+1}+\lambda_{t,\ell}P_{t,\ell}Z_t^0,
\qquad t\in\{M,E\},
\tag{12}
\]

with a small learned scalar or channel gate, initially zero. This is an optional standard skip mechanism, not a second field network or a mandatory new subsystem. Do not use it to insert forbidden heating, target flow, or reference fields.

At the end of contextualization, individual module and environment states still exist. **Compression is late:** it occurs when forming the decoding bank, not before all meaningful module–environment exchange.

### 5.4 Exact, linear, and mixed attention

The default is exact multihead softmax attention with a standard supported implementation. Its dense interaction arithmetic is approximately

\[
O\!\left((M+N_E)d^2
+\left[M(M-1)+2MN_E\right]d\right)
\tag{13}
\]

per block, excluding small geometry-bias work and FFN constants. The current wide pair-MLP preparation instead includes terms of order \(\left[M(M-1)+2MN_E\right]d^2\) for comparable widths. This is an architectural cost distinction, not a measured speed ratio.

A normalized positive-kernel attention alternative is

\[
A(q)
=
\frac{\varphi(q)^{\mathsf T}
\left(\sum_j\eta_j\varphi(k_j)v_j^{\mathsf T}\right)}
{\varphi(q)^{\mathsf T}\left(\sum_j\eta_j\varphi(k_j)\right)}.
\tag{14}
\]

It permits shared donor accumulation [R7]. But an arbitrary pairwise term \(\exp(\beta_{ij})\), a general pair mask, or destination-dependent values prevents this reassociation unless it has additional separable structure. It is not correct to attach all original pairwise geometry machinery and still label the result linear attention.

For larger domains, a mixed scheme can combine exact local attention with a global kernel/latent attention path. This is a future backend choice. The initial HONF should not import sequence-model state updates, autoregressive KV assumptions, or several attention variants merely because they are available.

## 6. The hypergraph as prepared field content

### 6.1 Edge seeds and spatial registration

Let \(H\) be the registered number of decoding hyperedges, with learnable seed vectors \(s_g\), \(g=1,\ldots,H\). Seeds are shared across scenes and independent of physical source ordering. They can be conditioned on \(z_{\mathrm{op}}\) and a scene summary used only during preparation.

Each edge also has a spatial anchor \(a_g\) and a positive length scale \(\sigma_g\), expressed in the adapter's coordinate frame. Anchors may begin at distinct reference-domain positions and receive smooth scene-conditioned shifts. They are reference locations for gathering and querying, not hard physical cells or identified source objects. A weak geometric preference is acceptable; it should not force all physically related donors into a compact Euclidean ball.

For an irregular domain, anchors may lie in a containing reference region. The adapter supplies validity and geometry features. A latent anchor need not itself be a physical fluid receiver; demanding that every anchor project onto the current mesh introduces unnecessary discontinuities.

### 6.2 Overlapping typed incidence and its normalization

Use module-to-edge and environment-to-edge assignment scores,

\[
s^M_{ig}
=\frac{\langle k_M(z_i),q_H(s_g)\rangle}{\sqrt{d_H}}
+\beta^M(c_i,a_g,\xi),
\qquad
s^E_{ag}
=\frac{\langle k_E(e_a),q_H(s_g)\rangle}{\sqrt{d_H}}
+\beta^E(x_a,a_g,\xi).
\tag{15}
\]

A simple first model uses soft assignments across edges:

\[
B^M_{ig}=\operatorname{softmax}_{g}(s^M_{ig}),
\qquad
B^E_{ag}=\operatorname{softmax}_{g}(s^E_{ag}).
\tag{16}
\]

These are nonnegative incidence strengths. A node can participate in several edges. They are not physical energy fractions. Softmax over edges and attention softmax over donors are different normalizations; the implementation and exported maps must say which is used.

Define typed edge masses and actual pooling weights,

\[
D_g^M=\sum_i p_i\mu_i B^M_{ig},
\qquad
W^M_{gi}=\frac{p_i\mu_i B^M_{ig}}{D_g^M},
\tag{17}
\]

\[
D_g^E=\sum_a p_a^E\nu_a B^E_{ag},
\qquad
W^E_{ga}=\frac{p_a^E\nu_a B^E_{ag}}{D_g^E}.
\tag{18}
\]

An absent type produces a zero summary and an explicit absence indicator. Use stable normalized sums on nonempty types; an arbitrary epsilon must not turn zero-measure donors into information. Keep the actual masses as separate, nondimensionalized features so averaging does not silently discard extensive information. Physical module count can be a prescribed summary; raw environment sample count should not become a physical feature when it changes only because the same measure was resampled.

### 6.3 Content-bearing edge states

Project donor content once and pool separately by type:

\[
S_g=\sum_i W^M_{gi}V_Mz_i,
\qquad
E_g=\sum_a W^E_{ga}V_Ee_a.
\tag{19}
\]

Then form

\[
H_g
=s_g+
\Phi_H\!\left[
S_g,E_g,z_{\mathrm{op}},\gamma(a_g),
\log\bar D_g^M,\log\bar D_g^E,
\mathrm{type\ availability}
\right],
\tag{20}
\]

where \(\bar D\) uses declared reference measures and the log encoding has an explicit absent-type convention. \(\Phi_H\) is a shared tokenwise network, not a separately trained expert for every edge.

The default can stop here after a tokenwise residual FFN. An edge–edge self-attention block is optional, not fundamental: the module/environment states have already been contextualized, and more edge mixing can make donor interpretation less direct. If edge–edge attention or re-attention to native tokens is added, record its ancestry; the final incidence map alone no longer describes all information used by the edge [R11].

The bank is prepared once for a fixed scene and model state. This does not mean once for an entire training run: weight updates or changed geometry invalidate learned cached content. Fixed raw geometry descriptors can be cached more broadly under their own identity.

### 6.4 The actual hypergraph object

The prepared organization is not just \(\{H_g\}\). It is

\[
\mathfrak H(\mathcal G)=
\left(
\mathcal M,\mathcal E,
\{H_g,a_g,\sigma_g\}_{g=1}^{H},
B^M,B^E,W^M,W^E,
\text{context ancestry},
\text{output capabilities}
\right).
\tag{21}
\]

For Thermal, this object additionally carries the source-indexed response coefficients defined below. It explicitly distinguishes physical modules, environmental sample records, latent decoding edges, and eventual receiver access. The model may learn useful nonlocal groups; a geometrically compact edge is not required for legitimacy.

## 7. Receiver decoding: what should actually execute

### 7.1 Lightweight addressing

Construct edge keys once and receiver queries independently:

\[
\kappa_g=W_K^H\operatorname{LN}(H_g),
\qquad
r(x)=W_Q^H[\gamma(x),z_{\mathrm{op}},b_Q(x)],
\tag{22}
\]

\[
\ell_g(x)=\frac{r(x)^{\mathsf T}\kappa_g}{\sqrt{d_r}}
+\beta_Q(x,a_g,\sigma_g),
\qquad
\alpha_g(x)=\operatorname{softmax}_g\ell_g(x).
\tag{23}
\]

The adapter-owned \(b_Q(x)\) contains only prescribed receiver geometry such as boundary distance, height, or material region. It cannot include the unknown solution. A fluid/surface/material label must not create three unrelated temperature functions at the same physical location; native Thermal extraction uses one temperature field.

Addressing is input-only and, for an affine capability, independent of heating. Initially all H edges are read. No target or target-error map chooses which edge executes.

### 7.2 Two legitimate decoder forms

**Projected-value decoder:**

\[
c_H(x)=\sum_g\alpha_g(x)V_HH_g,
\qquad
\hat y(x)=D[\gamma(x),c_H(x),c_{\mathrm{loc}}(x),z_{\mathrm{op}}].
\tag{24}
\]

This is close to ordinary cross-attention and directly implements the cheap Part-1 proposal. Its nonlinear query network runs once per receiver. Because it does not execute a large MLP for every edge–query pair, pruning six cached values may offer little additional benefit; its efficiency comes mainly from representation compression.

**Shared edge-conditioned decoder, recommended main form:**

\[
p_g=W_H\operatorname{LN}(H_g),
\tag{25}
\]

\[
m_g(x)=W_O\,\sigma\!\left(
p_g+W_x\gamma(x)
+W_b b_Q(x)
+W_{\mathrm{rel}}\gamma\!\left(\frac{x-a_g}{\sigma_g}\right)
+b\right),
\tag{26}
\]

\[
c_H(x)=\sum_g\alpha_g(x)m_g(x).
\tag{27}
\]

All edges share the reader weights. \(p_g\) is query-independent and cached; query-only projections are computed once per receiver. The nonlinearity still depends on the edge–receiver relationship, but its work scales with H rather than M or \(N_E\). This is defined directly as a new network, not a numerically exact factoring retrofit of old weights.

Use a narrow edge reader rather than recreating a large, generic pair MLP over concatenated state copies. The reader supplies both nonlinear field features and spatial thermal-response functions. This gives one common decoding mechanism with different lawful output heads.

The projected-value decoder is a useful simpler reference. The main design should not combine both complete decoders in parallel unless evidence later demonstrates a distinct need.

### 7.3 An explicit local detail path

Let \(\mathcal N(x;\mathcal G)\) contain sources whose geometric neighborhoods cover x. This set is determined before local value evaluation. A smooth compact-support weight can be

\[
\chi_i(x)=w\left(\frac{\|x-c_i\|}{R_i}\right),
\qquad
w(t)=(1-t)_+^4(4t+1).
\tag{28}
\]

Here \(R_i\) is a declared geometry-scaled local radius, not a physical influence cutoff. The global edge bank still represents long-range influence. All admissible local pairs are retained; do not silently drop a boundary neighbor to satisfy a K quota. Dense local overlap increases work and must remain visible.

A shared local reader returns flow features and, where applicable, thermal coefficients:

\[
\left(\ell_i^F(x),k_i^{\mathrm{loc}}(x)\right)
=\Psi_{\mathrm{loc}}[z_i,\gamma(x-c_i),a_i,b_Q(x),z_{\mathrm{op}}].
\tag{29}
\]

It is evaluated only for \(i\in\mathcal N(x;\mathcal G)\). Geometric descriptor work can still include cheap distance searches; not every coordinate comparison is an expensive neural read. The model must account for both.

The nonlinear field head uses

\[
c_{\mathrm{loc}}^F(x)
=\sum_{i\in\mathcal N(x)}\chi_i(x)\,\bar\mu_i\ell_i^F(x),
\qquad
\hat F(x)=D_F[c_H(x),c_{\mathrm{loc}}^F(x),\gamma(x),z_{\mathrm{op}}].
\tag{30}
\]

\(\bar\mu_i\) is an explicitly chosen nondimensional feature-aggregation measure. It is not automatically the actuator-volume factor in a physical source law.

The local path is a residual detail representation. It does not independently replace the global field. Both branches are trained together, and the complete field is observed through native output operators. If accuracy requires evaluating most physical sources almost everywhere, the proposed compression has not succeeded.

### 7.4 Query-set consistency

For fixed scene and model, require

\[
\hat y(x;\mathcal Q_1)=\hat y(x;\mathcal Q_2)
\quad\text{whenever }x\in\mathcal Q_1\cap\mathcal Q_2,
\tag{31}
\]

up to declared arithmetic tolerance. Edge preparation cannot depend on the current output-query batch. Do not use query self-attention, normalization across requested query points, or current-query-set pooling to determine the field.

Input meshes or environment catalogues may be processed collectively because they are fixed parts of the scene. They are not interchangeable with an arbitrary set of output requests. This distinction is especially important when adapting mesh-based Transformer papers to HONF.

## 8. ThermalChannel: source-resolved affine response through hyperedges

### 8.1 Keep the law, change the representation

For the current qualified benchmark, temperature is modeled as

\[
\hat T(x;\mathcal G,h)
=b_T(x;\mathcal G)+\sum_{i=1}^{M}K_i(x;\mathcal G)h_i.
\tag{32}
\]

The present zero-boundary/zero-initial-temperature capability sets \(b_T=0\). Nonzero boundary data require a declared lifting or offset capability, not a silent change to a normalization mean. Heat independence of the coefficients makes the model affine; it does not, by itself, enforce all boundary values. Native boundary extraction and any hard boundary construction must remain explicit.

The structural states, incidence, edge contents, routing, source coefficients, and local coefficients are prepared without actual h. Flow has no path from h. This is stronger and simpler than trying to learn a zero heat-to-flow derivative through a penalty.

### 8.2 Several response channels inside each edge

Let each edge carry r thermal-response channels. Define a small source–edge coefficient network evaluated during preparation:

\[
C_{gri}(\mathcal G)
=
B^M_{ig}(\mathcal G)\,
\left[
\Psi_C\left(
H_g,z_i,\gamma(c_i-a_g),a_i,z_{\mathrm{op}}
\right)
\right]_r.
\tag{33}
\]

The physical source ID labels column i. Both positive and negative coefficients are allowed. The incidence factor couples source-channel membership to the exported organization; a separate unrelated dense coefficient matrix should not be presented as the same graph.

The incidence in Equation (33) is the node-to-edge assignment, not the donor-normalized pooling weight. A physical heating amplitude is not automatically multiplied by the attention quadrature. Geometry, volume, and prescribed source units can affect \(\Psi_C\), but the attention measure and the physical actuation measure are different objects. For general volumetric heating, total supplied power is determined by a dataset-owned volume/source operator, not necessarily \(\sum_i h_i\).

Equation (33) uses a small H-by-M coefficient calculation once per prepared scene. It does not reinstate the Q-by-M nonlinear decoder. One can cache source and edge projections before forming these coefficients.

Obtain signed spatial basis functions from the common edge reader:

\[
\phi_{gr}(x;\mathcal G)
=\alpha_g(x;\mathcal G)\,[W_Tm_g(x;\mathcal G)]_r.
\tag{34}
\]

Define grouped control amplitudes

\[
a_{gr}(h;\mathcal G)=\sum_i C_{gri}(\mathcal G)h_i.
\tag{35}
\]

Then

\[
\boxed{
\hat T(x;\mathcal G,h)
=b_T(x;\mathcal G)
+\sum_{g=1}^{H}\sum_{r'=1}^{r}
\phi_{gr'}(x;\mathcal G)a_{gr'}(h;\mathcal G)
+\sum_{i\in\mathcal N(x)}
\chi_i(x)k_i^{\mathrm{loc}}(x;\mathcal G)h_i.
}
\tag{36}
\]

The summation index \(r'\) distinguishes a channel index from the number r of channels. All coefficients on the right are independent of h. The first response term provides collective global behavior; the second supplies source-local detail. They are additive jointly learned components, not two independently solved physical contributions.

The resulting source kernel is

\[
K_i(x;\mathcal G)
=
\sum_{g,r'}\phi_{gr'}(x;\mathcal G)C_{gr'i}(\mathcal G)
+\chi_i(x)k_i^{\mathrm{loc}}(x;\mathcal G).
\tag{37}
\]

Thus physical source identity is retained even though ordinary prediction contracts controls through a compact edge-channel representation. Exporting K is optional; evaluating every K column is not required for every ordinary field call.

### 8.3 Why nonlinear processing must end before control application

It would be incorrect to build \(a=Ch\), pass a through a nonlinear FFN, and still claim exact affine thermal response. The same problem occurs with LayerNorm of heat-carrying tokens, heat-dependent routing, or a nonlinear final decoder consuming both \(c_H\) and a.

The valid ordering is: nonlinear geometry/context processing → nonlinear spatial basis evaluation → **linear control contraction**. Arbitrarily rich nonlinear functions of geometry are allowed. Nonlinearity in the separately applicable controls is not.

For fixed \(\mathcal G\),

\[
\frac{\partial\hat T}{\partial h_i}=K_i,
\qquad
\Delta\hat T=K\Delta h,
\qquad
\frac{\partial^2\hat T}{\partial h_i\partial h_j}=0.
\tag{38}
\]

These are exact laws of the proposed network in exact arithmetic, not proof that K is the true physical response operator. Signed heat transfers and signed basis functions do not establish causal source attribution.

### 8.4 Units and normalization

One consistent convention makes \(\phi\) dimensionless, assigns C and \(k^{\mathrm{loc}}\) units of temperature per unit source amplitude, and expresses h in the adapter's native source units. Other factor scalings are possible, but the exported K must always have a declared output/control unit.

For fixed TRAIN-owned diagonal scales \(S_h,S_y\), a network trained with normalized amplitudes and response coefficients obeys

\[
\tilde h=S_h^{-1}h,
\qquad
K_{\mathrm{physical}}=S_y\tilde K S_h^{-1}.
\tag{39}
\]

Loss normalization may subtract a TRAIN temperature mean. It must not introduce a nonzero physical prediction at h=0 when the declared capability requires zero. In particular, predicting a normalized zero and then blindly adding the data mean would violate the physical offset contract.

Current Thermal quantities remain packed-dataset native units unless a physical-unit mapping is actually established. This model design does not relabel them as Kelvin, watts, or SI heat flux.

### 8.5 One temperature field, native derived observations

Let \(\mathcal O_{\mathcal G}^{a}\) be a native extraction operator for role a at fixed geometry. For linear interpolation and the maintained difference-based flux proxy,

\[
\hat y^a=\mathcal O_{\mathcal G}^{a}\hat T,
\qquad
K^a=\mathcal O_{\mathcal G}^{a}K.
\tag{40}
\]

Fluid temperature, surface temperature, outside temperature, and material samples should therefore derive from the same grid/field representation. Preserve the current coordinate transforms, module-attached receiver IDs, interpolation stencils, conductivity convention, and fluid/solid validity rules [C3]. Do not invent an independent temperature head for every evaluation role simply to improve each metric separately.

The native normal-flux proxy is not interchangeable with an arbitrary continuum AD normal derivative. Effective heat-transfer coefficients, clipped ratios, and material maxima are generally nonlinear functions of the field even when point temperatures are affine. For a module peak,

\[
\Delta T_{\max,i}
=\max_{x\in\Omega_i}\hat T(x;h+\Delta h)
-\max_{x\in\Omega_i}\hat T(x;h),
\tag{41}
\]

not \(\max_x\Delta\hat T(x)\).

The nonlinear flow head predicts physical u, v, p through the existing normalization convention. In the native-curl capability,

\[
\hat\omega=D_x^{\mathrm{native}}\hat v-D_y^{\mathrm{native}}\hat u,
\tag{42}
\]

using the actual valid native stencils after physical conversion. An auxiliary learned omega head is not necessary. A generic zero-divergence constraint is not inherited automatically: the archived Thermal generator's analytic flow has its own documented limitations [S2].

### 8.6 One jointly trainable model

All context, edge, local, nonlinear-flow, and affine-temperature weights belong to one new model. Both flow and thermal losses train the shared contextualizer and edge content. There is no frozen D-sep dependency, frozen local Stage-A model, or P0/P1/P2 port-refinement loop in this design.

This does not require adding predicted flow back into thermal inputs. Shared learned geometry/context already permits both outputs to use common representation. A future one-way predicted-flow feature edge is a separate hypothesis, not a default component, and may never carry reference flow at inference.

## 9. Hyperedge number, content width, and response rank

### 9.1 The rank restriction in the scalar-edge proposal

On Q fixed receivers, flatten the pair \((g,r')\) into R=Hr modes. The global thermal operator is

\[
K^{\mathrm{global}}=\Phi C,
\qquad
\Phi\in\mathbb R^{Q\times R},
\quad
C\in\mathbb R^{R\times M}.
\tag{43}
\]

Therefore

\[
\operatorname{rank}(K^{\mathrm{global}})\leq\min(Q,Hr,M).
\tag{44}
\]

For r=1 and H=6, the bound is six regardless of the contextualizer's width. Many independent heater columns can be required even when one nominal temperature field is smooth. A model can fit the observed heating combination while failing other source combinations.

For a weighted reference residual operator, the best rank-R Frobenius approximation has squared error equal to the squared singular-value tail. Consequently, a genuine non-negligible tail beyond R is an irreducible limitation of that global factorization. This is an algebraic statement about a specified operator, not a claim that the unavailable fully excited physical K has already been measured.

The sparse local term can restore full total rank through source-specific functions. It does not prove that the far response has low rank. Far-separated heaters or multiple downstream plumes may still require independent nonlocal modes. “Local correction fixes the bottleneck” must be demonstrated, not assumed.

### 9.2 Six edges with multiple channels

H=6 and r=4 provide 24 global response modes. This removes a rank-six ceiling for the present Thermal source counts without making 24 separate graph nodes. It does not guarantee accuracy: assignment structure, learned features, spatial basis capacity, and training data can still restrict the fitted operator.

The independent design variables are:

| Quantity | Meaning |
|---|---|
| M | Number of actual physical modules/actuators |
| \(N_E\) | Number of environment representation records |
| H | Registered decoding hyperedge capacity |
| d | Vector width of contextual tokens/edge content |
| r | Independent affine response channels per edge |
| Hr | Upper bound on global affine response rank |
| \(k(x)\) | Number of edges actually executed for receiver x |
| \(n_{\mathrm{loc}}(x)\) | Number of local physical-source reads |

Six content-rich edges are not equivalent to six scalar averages, six attention heads, or six physical modules. Increasing d alone does not increase Equation (44)'s rank when r is unchanged.

### 9.3 What to do if fixed capacity is insufficient

First determine whether the failure is local spatial detail, lost native-token context, insufficient global response rank, or poor assignment. They call for different changes. Increase local basis capability for near-interface residuals; increase r for independently required thermal source modes; increase H when the scene needs additional distinct information groups; improve contextualization when full-edge decoding already lacks interaction knowledge.

The next architecture should not begin with birth/death controllers, recursive trees, or a required small-K histogram. A later adaptive-capacity family can use an overcomplete edge bank with learned occupancy or smooth split/refinement, but its output law, source identity, and cache semantics must remain unchanged. Increasing geometric domain size does not imply that fixed H will generalize indefinitely; capacity may have to scale with physical complexity.

### 9.4 Identification is a data issue as well as a model issue

The existing opposite-transfer records provide one independent control direction per fixed layout, not two. If \(\Delta h\) spans a low-dimensional subspace, matching \(K\Delta h\) does not identify every column of K. The family of changes \(K+N\) satisfying \(N\Delta h=0\) is indistinguishable on those observations.

Source-resolved storage, exact gradients, and a large enough Hr make further identification possible; they do not create missing excitation. TRAIN-qualified operator residuals can add useful structure, but remain different from independently solved source-response labels. Future inverse claims must reflect this distinction.

## 10. Wind and broader nonlinear modular problems

### 10.1 Wind uses the same bank without an affine control law

Wind uses the same typed attention contextualizer, edge preparation, query addressing, shared edge reader, and local detail machinery. Its adapter supplies 3-D geometry and receiver features, and its output head predicts a nonlinear velocity residual:

\[
\hat{\boldsymbol U}(x)
=\boldsymbol U_{\mathrm{profile}}(x;\xi)
+\mathcal T_{\mathrm{TRAIN}}^{-1}
\left(D_U[c_H(x),c_{\mathrm{loc}}^F(x),\gamma(x),z_{\mathrm{op}}]\right).
\tag{45}
\]

Here \(\mathcal T_{\mathrm{TRAIN}}^{-1}\) is precisely the stored residual transform, including any required centering/scaling. The profile is fitted only from the permitted TRAIN inventory. The adapter retains rotor-diameter coordinates, physical source identities, native direction handling, and output conversion to m/s [C4]. No second rotation or unverified frame conversion is introduced.

Turbine geometry changes rebuild all affected scene, edge, local-neighborhood, and receiver state. Yaw/thrust changes likewise require a new configuration unless a separately validated response capability is defined. A Thermal-style exact finite-increment method is not available merely because the Wind network is differentiable.

### 10.2 How nonlinear many-body dependence enters

A module's representation \(z_i\) depends on other modules and the environment. Edge states depend on sets of these contextualized tokens, and the field head is nonlinear. Thus a source-preserving or edge-factorized architecture need not assume geometrically independent modules.

The thermal affine law is linear only in the qualified controls at fixed structure. Geometry-to-K can be highly nonlinear. For Wind, both geometry and allowed actuation dependence remain nonlinear. The common architecture does not impose superposition where the dataset does not support it.

### 10.3 Generality means reusable operators and explicit capabilities

Other modular datasets can change the module encoder, environment representation, prescribed coefficient fields, output channels, and observation operators without replacing the attention/edge/decoder implementation. Battery modules may carry geometry, conductivity and operating descriptors; geothermal wells may carry trajectories and boundary conditions; other flow domains may require richer surfaces or inlet fields. These are possible input contracts, not claims that one scalar module token is sufficient for all such geometries.

A nonlinear multiphysics dataset in which heating changes flow must place heating on a permitted configuration path. It cannot reuse ThermalChannel's heat-null restriction. A steady coupled response can be represented as a nonlinear mapping from all prescribed inputs without claiming an exact affine reusable state. Time evolution or solver-like cyclic updates would require an explicit state/temporal contract and are outside the initial static redesign.

## 11. Sparse edge execution and derivative-safe reuse

### 11.1 Put sparsity after representation adequacy

The full-edge model is the initial reference:

\[
\mathcal A(x)=\{1,\ldots,H\}.
\tag{46}
\]

Only after it reconstructs the required fields competently should a sparse edge policy choose \(\mathcal A(x)\subseteq\{1,\ldots,H\}\). The order must be: cached keys → cheap query scores → support selection → gather content → expensive edge reader → lawful output contraction.

Evaluating every edge message and masking afterward is a dense implementation of a sparse-weight function. It is a valid numerical reference, but not sparse execution. Conversely, selecting only a few edges while allowing them to expand into all M sources recreates dense final reading; any candidate-source refinement must be deduplicated and separately bounded.

### 11.2 Support law

The most conservative initial model uses full softmax. A later entmax map can produce exact zero edge weights:

\[
\alpha(x)=\operatorname{entmax}_{\beta}(\ell(x)),
\qquad 1<\beta\leq2,
\qquad
\mathcal A(x)=\{g:\alpha_g(x)>0\}.
\tag{47}
\]

This provides a continuous sparse weighting law with piecewise differentiation properties [R13]. It is not the same as hard top-K followed by a renormalized softmax. A hard top-K crossing can switch between messages with different values and create output discontinuity even when the competing scores are equal.

With a six-edge bank, full execution may be the best hardware choice. Adaptive sparsifiability is a property worth investigating; it is not an obligation to make the first model execute fewer than six edges.

### 11.3 Attention mass is not output-error importance

For fixed unrenormalized weights and unchanged local content,

\[
\Delta c_H(x)=\sum_{g\notin\mathcal A(x)}\alpha_g(x)m_g(x).
\tag{48}
\]

The field change also depends on the nonlinear head. A local Lipschitz constant, if actually bounded on the relevant set, gives

\[
\|\Delta\hat F(x)\|
\leq L_D\,\|\Delta c_H(x)\|
\leq L_D\sum_{g\notin\mathcal A(x)}\alpha_g(x)\|m_g(x)\|.
\tag{49}
\]

Absent such a bound, these are sensitivity diagnostics rather than certificates. Small routing mass does not imply small physical effect. Renormalizing retained weights changes retained contributions too; it is a different approximation from deleting terms at fixed weights.

For affine Thermal response, let \(\Delta\Phi=\Phi_{\mathrm{full}}-\Phi_{\mathrm{sparse}}\) include all changes due to selection and renormalization. If the allowed control increment satisfies \(|\Delta h_i|\leq\rho_i\),

\[
\|\Delta T_{\mathrm{full}}-\Delta T_{\mathrm{sparse}}\|_{W}
\leq
\sum_{g,r'}
\|\Delta\phi_{gr'}\|_W
\sum_i|C_{gr'i}|\rho_i,
\tag{50}
\]

when the local path is unchanged. Additional local omission has its own term. This is a bound on distortion of the learned operator inside a declared control box, not on its error against physical truth.

A heating-independent sparse operator requires a support rule based on configuration, receivers, and an explicitly fixed control-domain/budget declaration. Selecting support separately from the current h generally destroys global heat affinity, even if each individual fixed-support contraction is linear.

### 11.4 Smooth and sparse interfaces are distinct

For geometry optimization, the default should be full smooth edge access plus smooth local support. Sparse deployment can use the same architecture with an explicit support law, but its derivative scope must be stated. A fixed-support gradient is not the derivative through an arbitrary hard support change.

The complete thermal geometry derivative includes changes in spatial functions, source coefficients, memberships, anchors and local weights:

\[
\partial_{\mathcal G}T
=(\partial_{\mathcal G}\Phi)Ch
+\Phi(\partial_{\mathcal G}C)h
+(\partial_{\mathcal G}K^{\mathrm{loc}})h
+\partial_{\mathcal G}b_T.
\tag{51}
\]

For material-attached receivers \(x=x(\mathcal G)\), the receiver-motion contribution must also be included. Native interpolation and mask changes can make geometry dependence piecewise smooth; the neural network's smooth activation does not remove those adapter boundaries.

Adding or removing a module changes a discrete inventory. This is a new scene, not an ordinary derivative with respect to an unchanged physical-source axis.

## 12. What “physically meaningful organizer” should mean here

### 12.1 Three distinct evidential levels

**Computational meaning:** Incidences show the actual donor content pooled into edges; receiver access selects actual edge content; local paths and all context ancestry are disclosed. Removing or replacing the specified information changes the executed prediction as expected.

**Physical usefulness:** The organization preserves or improves reconstruction and relevant finite responses under comparable training and work. Groups are not useful merely because their spatial pictures are recognizable.

**Physical identification:** Claims that particular modules jointly cause a measured response require independent physical perturbation evidence, sufficient excitation, and numerical qualification. Attention diagrams, latent signs, and nonzero gradients do not establish this strongest level.

The new architecture makes the first level substantially more direct: the edge bank carries the values used by the decoder. It creates a plausible path to the second level. It does not automatically establish the third.

### 12.2 A structural incidence is not a causal support mask

After contextualization, \(z_i\) can contain information about all modules and environment records. Even if \(B^M_{ig}=0\), source i can affect edge g indirectly through another contextualized donor. An absent direct membership means that direct pooling route is absent, not that the physical source is irrelevant.

Similarly, H learned from fixed geometry and environment is a representation of that scene's information. A source deletion intervention that recomputes the full scene is different from deleting one decoder edge at fixed context. Both are legitimate tests, but answer different questions.

### 12.3 Identifiability of latent factors

Latent edge ordering has no inherent meaning. Within thermal modes, factor scalings and, where the parameterization permits, basis transformations can leave K unchanged:

\[
\Phi C=(\Phi A)(A^{-1}C)
\tag{52}
\]

for an invertible compatible A. Structural incidence constraints restrict some transformations but do not turn every internal mode into a unique physical quantity. Report source-indexed K, reconstructed contributions, spatial support, and relevant subspaces rather than comparing mode-number signs across seeds as though they were physical measurements.

Useful displays should distinguish normalized incidence, actual measure-weighted pooling, receiver access, signed edge contribution, and source-response coefficient. Combining them into one arrow-width scale loses their meanings.

### 12.4 Avoid making the graph mandatory by arbitrary penalties

A reconstruction-trained organizer can legitimately use all six edges, prefer one edge for a simple case, or retain long-range links. Do not require uniform edge occupancy, a prescribed degree histogram, or visually different heads as a success condition. If all edges are redundant, investigate whether the field decoder bypasses them, whether capacity is excessive, or whether the task genuinely needs a simpler representation.

The relevant controls are ordinary latent cross-attention and a simpler geometric assignment at comparable capacity and work. A hypergraph interpretation must add explanatory or operational usefulness, not merely rename a set of learned tokens.

## 13. What remains shared and what remains dataset-owned

### 13.1 Reusable core versus physical adapter

| Component | Shared responsibility | Dataset-owned responsibility |
|---|---|---|
| Scene | Typed token collections, masks, measures, identities and dependency capabilities | Meaning of module/environment features, geometry ownership, legal controls and units |
| Encoding | Token-lift interface and positional-feature machinery | Input widths, TRAIN transforms, geometric descriptors and prescribed conditions |
| Contextualization | The same typed attention, normalization, FFN and residual code | Allowed route types, coordinate frame and physically declared masks |
| Organization | Seed bank, typed incidence, pooled content, anchors and receiver addressing | Appropriate geometry scales and configuration parameters; not a separate organizer implementation |
| Decoding | Shared edge-conditioned and local readers; nonlinear and affine output interfaces | Number and meaning of outputs, control components and fixed physical transforms |
| Thermal extraction | Generic prepared response and linear observation interfaces | Native grid, cell-center interpolation, module-material coordinates, conductivity/proxy formulas, curl stencils |
| Wind extraction | Generic nonlinear prediction and local linearization interfaces | Rotor-relative coordinates, TRAIN-fitted height profile, residual normalization, velocity frame and m/s conversion |
| Training | One engine, batching abstractions, optimizer/state handling and loss-term protocol | Fixed dataset memberships, sampling roles, lawful objectives, physical scales and target extraction |
| Reuse | Ownership, versioning, structured outputs and invalidation rules | Which input changes are separately applicable and which require rebuilding |

The current `InteractionScene`, `DependencySpec`, prepared-response objects, and dataset extractors are useful conceptual starting points [C1–C4]. Their purpose should be retained without requiring that a new attention model inherit all historical architecture-dispatch branches or guard implementations. A clean family can share the physical contracts while leaving old checkpoints replayable in their own unchanged families.

### 13.2 Four reusable mathematical objects

A clear interface separates the following states:

**PrescribedScene:** geometry, coefficients, typed IDs, masks/measures, operating conditions, coordinate transforms and capability declaration. No labels or reference solution.

**PreparedOrganization:** contextual token states, edge bank, typed incidence, cached key/value projections, source-response coefficients where applicable, and scene/model/normalizer ownership. Independent of output-query batching and, where declared, applicable controls.

**PreparedQuery:** requested receiver identities and coordinates, edge scores/support, spatial basis values or projected features, local pairs, and native observation stencils. A query object belongs to one prepared organization and one support policy.

**ResponseApplication:** physical outputs, optional contributions, derived observations, and derivative/finite-increment semantics. It should not claim a capability stronger than that supplied by its owner.

This separation avoids recomputing geometry and source organization when only Thermal heating changes. It also makes it explicit why moving a turbine or changing a model weight invalidates prepared content.

### 13.3 Fundamental invariances and their limits

Under a physical-source permutation P,

\[
C(\mathcal G_P)=C(\mathcal G)P^{\mathsf T},
\qquad
h_P=Ph,
\qquad
\hat T(\mathcal G_P,h_P)=\hat T(\mathcal G,h).
\tag{53}
\]

Source-indexed outputs and incidence rows transform accordingly. Environment records are permutation invariant in their aggregate effect. Query permutations only reorder corresponding outputs.

Splitting one environment record into identical copies with measures summing to the original should preserve weighted attention and edge pooling in exact arithmetic: its weighted numerator and denominator contributions are unchanged. This argument assumes no raw environment-record count is injected as physics. Arbitrary remeshing is a quadrature-approximation issue, not exact finite-array identity.

A duplicated physical module is not generally a representation refinement: it may represent a second source. Preserve the distinction between token replication and adding a real actuator. Any multi-token module encoder needs an explicit physical-ID aggregation map.

Translation and rotation behavior must match the problem's boundaries and coordinate frame. An inlet channel or a wind direction breaks unrestricted rotational symmetry unless the entire physical configuration, boundaries, and vector outputs are transformed consistently. Do not claim SE(3) equivariance from relative coordinates alone.

### 13.4 Prepared reuse is mathematical, not a hidden detached computation

Caching learned preparation during inference is valid while its ownership is unchanged. During training, preparation must be rebuilt after parameter updates, and its graph must remain live for the update in which it is used. A detached cache of old edge content is a different optimization problem.

For fixed-geometry affine heating,

\[
J_h v=\Phi(Cv)+K^{\mathrm{loc}}v,
\qquad
J_h^{\mathsf T}w=C^{\mathsf T}(\Phi^{\mathsf T}w)
+(K^{\mathrm{loc}})^{\mathsf T}w.
\tag{54}
\]

These products can be evaluated without materializing the full Q-by-M kernel. Explicit K export is available when needed, with its unavoidable output size and work recorded separately.

Precise accumulation for small increments can be supported by the response application. It does not require declaring every neural operation bitwise invariant across batch shapes, attention kernels, or devices. The new family's numerical contract should distinguish structural laws, floating-point error, physical response error, and historical-replay identity.

## 14. Computational implications and honest efficiency claims

### 14.1 Where work is replaced

Let \(N=M+N_E\), \(P_{\mathrm{ctx}}=M(M-1)+2MN_E\), and L be contextualizer depth. A schematic old preparation/read cost is

\[
C_{\mathrm{old}}
\sim L\,P_{\mathrm{ctx}}\,C_{\mathrm{pair\text{-}MLP}}
+QM\,C_{\mathrm{source\text{-}reader}}
+C_{\mathrm{head}}+C_{\mathrm{native}}.
\tag{55}
\]

Thermal may have distinct nonlinear and affine source readers, so its actual native scope must count both. Q is the executed receiver/stencil union, not necessarily the user's requested primary query count.

For the proposed architecture,

\[
\begin{aligned}
C_{\mathrm{new}}
\sim\;&L\,[N d^2+P_{\mathrm{ctx}}d]
+C_{\mathrm{edge\ preparation}}(N,H,d)
+C_C(H,M,r)\\
&+QH\,C_{\mathrm{router}}
+Q\bar k\,C_{\mathrm{edge\ reader}}
+Q\bar n_{\mathrm{loc}}\,C_{\mathrm{local}}\\
&+Q\,C_{\mathrm{head}}
+C_{\mathrm{contraction}}(Q,Hr)
+C_{\mathrm{native}}.
\end{aligned}
\tag{56}
\]

For full-edge decoding, \(\bar k=H\). Source-channel preparation \(C_C\) is absent from a purely nonlinear Wind head. Native extraction, data movement, geometry preparation, support construction and validation remain real costs.

At M30, replacing 30 nonlinear source reads by six edge reads reduces that *counted operation family* by a factor of five before adding local reads and accounting for different widths. At M3, six edges are more than three sources. There is no universal speedup from H=6. The appropriate claim is a potentially favorable large-query/many-source regime, with a small low-M overhead that must be measured.

### 14.2 Memory and reuse

The main representation advantage is avoiding a wide Q-by-M-by-d final source-value tensor. A thermal global basis with Q8192 and Hr=24 contains 196,608 coefficients, or **0.75 MiB in FP32**, before local data, neural activations, query buffers, and native extraction. The C matrix for M12 contains only 288 coefficients. These are illustrative tensor sizes, not a predicted process-memory peak.

Actual inference can stream receiver chunks, and training must retain or recompute activations for gradients. A dense, well-packed six-edge operation can outperform irregular sparse packing even when sparse packing evaluates fewer edge rows. Backend dispatch should preserve the same chosen weighting law; switching silently between a full-softmax function and a truncated function is not merely an execution optimization.

For a fixed scene and many heating candidates, prepare \(\Phi,C,K^{\mathrm{loc}}\) once and apply the new controls. This reuse has a clear operation to amortize. For geometry search, preparation must be recomputed as geometry changes. Report those workloads separately.

### 14.3 Standard primitives, not an efficiency label

Use supported attention, linear projections, normalization, FFNs and batched reductions wherever possible. Fused attention may avoid materializing full attention weights, but exported diagnostic maps may require a separate explicit path. Additive geometry bias and donor measures must not be removed solely to obtain a faster backend.

The minimum efficiency evidence eventually needs complete native calls and training updates, in addition to arithmetic counts. A redesigned model can be faster because it learns a different factorized representation; it need not be an exact sparse subgraph of P. Conversely, low parameter count or unused VRAM does not prove that additional layers will improve accuracy.

## 15. Recommended initial architecture, without a training campaign

The initial family should be narrow enough to explain in one diagram and broad enough to avoid an artificial rank bottleneck.

| Choice | Recommended starting definition | Reason |
|---|---|---|
| Input contract | Typed modules and environmental records with explicit IDs, measures and prescribed conditions | Retains source correspondence and physical ownership |
| Context width | d=128 as a starting point | Comparable scale to the current common representation; not a mandatory universal width |
| Attention | Four heads, exact measure-aware softmax, typed MM/ME/EM | Standard building blocks; separates interaction directions without pair-value MLPs |
| Context depth | Two residual blocks initially | Sufficient for a first nontrivial exchange; increase depth only for an identified representation need |
| Original-token re-injection | Supported as one simple zero-initialized skip; off initially | Avoid adding an untested mechanism before ordinary residual attention is evaluated |
| Edge capacity | H=6, shared scene-conditioned seeds and soft spatial registration | A deliberate compact decoding bank, not a fixed physical partition |
| Edge content | Separate M/E pooling, mass features, one shared fusion/FFN | Prevents environmental record count from overwhelming module content |
| Edge–edge processing | None initially | No need to mix an already contextualized bank merely for appearance |
| Edge reader | Shared narrow reader, e.g. internal width 64, with cached edge and query projections | Preserves nonlinear spatial conditioning at much lower repeated source count |
| Local detail | One smooth compact-support source reader, with all actual local pairs retained | Protects near-body and interface detail without restoring a dense global source reader |
| Thermal response channels | r=4 per edge initially | Six edges do not impose only six global heater-response modes |
| Thermal flow | Fresh nonlinear u/v/p head; native derived curl | One jointly trained self-contained model |
| Thermal temperature | Source-indexed C, signed spatial modes, linear heating contraction, zero-offset native capability | Exact case-qualified affine response and source identity |
| Wind output | Nonlinear three-component residual with the current TRAIN profile/transform | No false affine law |
| Decoder support | All six edges initially | Establish representation adequacy before sparse-access pressure |
| Later sparse law | An explicit smooth sparse edge map or separately qualified hard policy | Selection must precede the edge reader, with derivative scope declared |

This is one family with capability-specific heads, not an ensemble. It does not contain the current dense decoder as a permanent parallel fallback. The old decoder remains an external architectural reference and a possible transitional comparison, not a hidden inference branch.

The source proposal's key statement is retained: **replacement, not addition**. The only deliberate exception is a bounded local-detail residual, whose presence and cost are explicit.

## 16. Questions that should organize the next implementation discussion

This section defines scientific comparisons, not commands, run counts, schedules, budgets, or launch authorization.

### 16.1 Separate contextualization from decoding

A conceptual two-axis comparison is useful:

| Contextualizer | Decoder | Question |
|---|---|---|
| Existing source-conditioned pair values | Existing source-resolved decoder | Reference computation |
| Typed donor-value attention | Existing source-resolved decoder | Is the new fusion stage competent before compression? |
| Typed donor-value attention | Full content-bearing edge decoder plus local detail | Does learned representation compression retain or improve physical fields? |
| Same selected edge model | Explicit sparse edge decoder | What additional error–work tradeoff is introduced by selection? |

A failed attention-plus-source decoder points to contextualization, encoding, optimization, or insufficient donor-value expressivity—not automatically to the edge count. A competent full-edge model with poor sparse behavior points to the access law or intrinsic distributed content—not automatically to poor fusion. These distinctions prevent changing several mechanisms in response to one aggregate loss curve.

### 16.2 Training principles, not a loss portfolio

The common engine should optimize one physically defined multi-field objective with dataset-owned roles and TRAIN-owned scales:

\[
\mathcal L_{\mathrm{field}}
=\sum_{a}\lambda_a\,
\mathbb E_{\mathcal G}\!
\left[
\|S_a^{-1}(\mathcal O^a_{\mathcal G}\hat y-y^a)\|_{W_a}^{2}
\right].
\tag{57}
\]

The nonlinear and affine heads share representation parameters and are jointly trainable. Existing qualified response/operator terms may remain visible as additional supervision, but do not proliferate losses merely to force a graph pattern. Field-role balance, source-excitation supervision, and sparse-work pressure answer different questions.

The first compressed representation should not be required to pay a strong sparse-work penalty before it can reconstruct the field. Later sparse learning can retain a full-edge reference or a declared mixed-support objective; details belong in the subsequent work plan. Repeatedly adjusting physical losses to repair a representation miss would obscure the architectural question.

### 16.3 Keep comparison maturity separate from mechanism evidence

Segmented development, matched architecture comparisons, and mature full-dataset practical reference gaps remain separate. Preserve fixed Thermal150/DEV22 and Wind72/DEV24 identities unless explicitly changed, TRAIN-only normalization, source locks, and existing native physical evaluation conventions. Do not infer the performance of an untrained full-data model from either a short development fit or a differently trained classic.

The classical Thermal board currently under reconciliation is not a sound numerical promotion target until its native path is qualified. That issue should not force the new architecture to inherit the old wrapper or normalize inputs incorrectly for apparent agreement.

### 16.4 What a meaningful result would look like

The full-edge decoder reconstructs native fields and important local structure without dense source-query values. Its typed incidences and receiver accesses correspond to the content actually used. A modest fixed H has enough within-edge response rank and local detail to avoid obvious compression failures. Shared code executes both datasets with their own lawful interfaces. Sparse execution, if introduced, has a measured complete-cost or work advantage at an explicitly measured physical-error tradeoff.

A negative result is also interpretable: the attention contextualizer may lack necessary geometry-dependent expressivity; the six-edge bank may need more response channels or edges; the local residual may be insufficient for the relevant sharp structures; or sparse access may not pay at such a small H. These are bounded model questions, not reasons to rebuild another elaborate organizing controller.

## 17. Open issues and decisions deliberately deferred

**Expressivity of donor-value attention.** It is more economical than a wide pair-value MLP, but whether it captures the Wind transverse wakes and Thermal geometry interactions at the chosen depth remains empirical. Geometry-aware scores and local detail address plausible limitations; they do not prove equivalence.

**Adequacy of H=6 and r=4.** These remove a particular rank-six restriction and provide a manageable first bank. Neither is a physical law. Source-count growth, widely separated plumes, and larger domains may require increased or hierarchical capacity.

**Physical meaning of learned edges.** Structural registration and source-indexed coefficients make interpretation more explicit. Unique physical attribution still needs data that identify responses, not just nominal field reconstruction.

**Numerical precision.** Efficient attention and packing can change floating-point results. The new family should preserve declared physical laws, masks, units, derivatives and error tolerances without pretending to be a bitwise copy of the old pairwise function. Historical failures are not erased.

**Environment representation.** E8 versus E512 changes the representation supplied to a model. A richer catalogue of valid prescribed geometry may help, but adding copies of the same information is not equivalent to adding physical knowledge. Catalogue choice is an interface question to keep separate from the attention mechanism.

**Locality and geometry differentiation.** Smooth local windows do not fix discontinuous mesh validity, moving material samples, or hard learned support. Inverse consumers require the complete derivative of the deployed interface and independent physical-response checks.

**Execution policy.** Full small-bank decoding may remain preferable to sparse GPU gathers. That outcome would still be a useful redesign if it provides accurate, source-aware organized decoding with better cost than the repeated source-value baseline.

**Pending work.** This design does not direct a running Codex task to abandon its safeguards, alter experiments, or consume new solver attempts. Source records from the preceding task can inform the next coding discussion without making that task's narrow pruning approach a permanent architectural constraint.

## 18. Final architectural statement

The redesigned HONF should be understood as a **typed attention encoder with a reusable, source-aware hypergraph decoding bank**, not a dense pairwise network decorated with graph controls.

Its mathematical commitments are:

\[
\begin{gathered}
\text{individual physical tokens remain available during contextualization},\\
\text{hyperedges carry the content used by spatial decoding},\\
\text{global repeated work scales with edge capacity, not dense source reads},\\
\text{local source detail is explicit and geometrically bounded},\\
\text{affine controls remain linear and source-indexed when physics permits},\\
\text{nonlinear datasets use the same core without inheriting that affine law}.
\end{gathered}
\tag{58}
\]

The new research hypothesis is not “attention always wins” or “six hyperedges are physically sufficient.” It is that **standard efficient contextualization, late content-bearing compression, sufficient within-edge response capacity, and a small local detail path offer a much better foundation for HONF than trying to make every stage of the existing dense pair-value computation simultaneously sparse, interpretable, and numerically identical.**

---

## Appendix A. Provenance and inspected implementation map

### A.1 User and project sources

**[U1]** `Pasted text.txt`, supplied in this conversation. Entire two-part proposal read, including its final full-edge-before-sparse recommendation. Part 1 supplies the attention, incidence, global/local affine-response and replacement principles; Part 2 supplies the selection-before-value evaluation and optional fine-source branch. The additional rank analysis, recommended multi-mode thermal factorization, explicit global-bypass restriction, and capability separation in this document are design synthesis, not quotations from that file.

**[S1]** `HONF_Multiview_Native_Routing_20261012.md` and the companion evidence index. These report the bounded fresh P/R1/R2 comparison, two-view implementation, dense executed values, saved responses, and comparison limits. No unseen local arrays were independently recomputed for this document.

**[S2]** `HONF_Joint_Formal_Comparison_20261010.md`, especially the native Thermal output/curl and generator-scope discussion. This supports retaining the dataset's actual extraction laws rather than imposing an idealized incompressible-flow target on its analytic reference.

**[S3]** The supplied report titled *Learned source-response interface: consolidation and maturation*, and `HONF_RDirect_Formal_Readiness_and_Receiver_Local_Organization_Report.md`. These provide the historical affine-response, local/global, excitation-rank and native-output lessons. The first is identified by its supplied title, not an inferred repository filename.

### A.2 Pinned repository review

The remote tip retrieved for this review was `f10aa863a5faae8db5cdb969d16692729e79d7ce`. The diff from `a6e898d` contains the active final-source reader, corresponding adapter changes/tests, a Thermal classic-input helper, and new consolidation/reconciliation reports. It does not replace dense context pair-value MLPs with standard attention.

| Source key | Repository path under `HONF_Proj/` | Relevant inspected content |
|---|---|---|
| [C1] | `src/honf_forward_core/interface_fields/interaction_core.py` | Typed physical scene/dependency objects; measure reductions; dense destination/donor pair-value construction; residual source/environment updates; prepared-state ownership; new selected-pair plan and reader |
| [C2] | `src/honf_forward_core/interface_fields/interaction_preserving_joint.py` | Direct versus collective modes, two-round rebuild semantics, distinct nonlinear and affine read capabilities, ownership and intervention boundaries |
| [C3] | `Case_ThermalChannel/src/channelthermal/source_response.py` and `joint_regional.py` | Heat-independent context whitelist; native zero-offset capability; source/native receiver identity; self-contained joint adapter; reuse of native grid/role extraction |
| [C4] | `Case_WindFarm/src/windfarm/interaction_preserving.py` and `joint_regional.py` | Shared nonlinear core binding, TRAIN profile and residual normalization, 3-D rotor-relative receiver features, physical/source ownership |
| [C5] | `docs/reports/HONF_R1_Consolidation_and_Sparse_Reads_20261013.md` | Completed Wind continuation, failed primary-host active-read omission frontier, retained dense default, saved Thermal response limits and complete-work/cost distinctions |
| [C6] | `docs/reports/HONF_R0_Thermal_Classic_Native_Input_Reconciliation_20261013.md` and the loading diff identified by the commit comparison | Versioned classic-input correction and retained quarantine; this document uses [C5]'s summary and does not claim to have independently qualified the corrected GPU path |

The code review establishes what the inspected functions implement. Reported training outcomes and runtime measurements remain report-derived. No claim of full-repository test execution, local data access, or verification of all historical artifacts is made here.

## Appendix B. Primary research references

The following references were checked through official proceedings, author-hosted arXiv papers, or official documentation. Performance numbers from those papers are intentionally not imported as predicted HONF gains.

**[R1]** Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024; preprint 2023. arXiv:2307.08691. Primary record: `https://arxiv.org/abs/2307.08691`. Relevance: fused exact attention and hardware-aware execution, not a new physical organizing law.

**[R2]** PyTorch. *torch.nn.functional.scaled_dot_product_attention* and the official SDPA tutorial. Documentation consulted on 13 October 2026. `https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention`; `https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html`. Relevance: supported attention primitives, mask/bias semantics and backend restrictions. The locally installed version and eligible backend must be checked during implementation; current online documentation is not a claim about that environment.

**[R3]** Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi and Yee Whye Teh. *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* ICML 2019, PMLR 97:3744–3753. `https://proceedings.mlr.press/v97/lee19d.html`.

**[R4]** Andrew Jaegle and colleagues. *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR 2022. arXiv:2107.14795. `https://arxiv.org/abs/2107.14795`.

**[R5]** Zhongkai Hao and colleagues. *GNOT: A General Neural Operator Transformer for Operator Learning.* ICML 2023, PMLR 202:12556–12569. `https://proceedings.mlr.press/v202/hao23c.html`.

**[R6]** Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang and Mingsheng Long. *Transolver: A Fast Transformer Solver for PDEs on General Geometries.* ICML 2024, PMLR 235:53681–53705. `https://proceedings.mlr.press/v235/wu24r.html`.

**[R7]** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas and François Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020, PMLR 119:5156–5165. `https://proceedings.mlr.press/v119/katharopoulos20a.html`.

**[R8]** Shuhao Cao. *Choose a Transformer: Fourier or Galerkin.* NeurIPS 2021. arXiv:2105.14995. `https://proceedings.nips.cc/paper_files/paper/2021/hash/d0921d442ee91b896ad95059d13df618-Abstract.html`.

**[R9]** Tian Wang and Chuang Wang. *Latent Neural Operator for Solving Forward and Inverse PDE Problems.* NeurIPS 2024. arXiv:2406.03923. `https://proceedings.neurips.cc/paper_files/paper/2024/hash/39f6d5c2e310a5a629dcfc4d517aa0d1-Abstract-Conference.html`.

**[R10]** Huakun Luo and colleagues. *Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries.* ICML 2025, PMLR 267:41432–41449. `https://proceedings.mlr.press/v267/luo25o.html`.

**[R11]** Wenjie Hu, Sidun Liu, Peng Qiao, Zhenglun Sun and Yong Dou. *Transolver Is a Linear Transformer: Revisiting Physics-Attention Through the Lens of Linear Attention.* AAAI 2026, 40(1):408–416. DOI:10.1609/aaai.v40i1.37003. `https://ojs.aaai.org/index.php/AAAI/article/view/37003`. Relevance: a current primary-source counterpoint to attributing every benefit of latent slicing to an explicit latent interaction stage.

**[R12]** Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang and George Em Karniadakis. *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* Nature Machine Intelligence 3:218–229, 2021. DOI:10.1038/s42256-021-00302-5. `https://doi.org/10.1038/s42256-021-00302-5`.

**[R13]** Ben Peters, Vlad Niculae and André F. T. Martins. *Sparse Sequence-to-Sequence Models.* ACL 2019. `https://aclanthology.org/P19-1146/`. Also: Gonçalo M. Correia, Vlad Niculae and André F. T. Martins. *Adaptively Sparse Transformers.* EMNLP-IJCNLP 2019. `https://aclanthology.org/D19-1223/`. Relevance: continuous exact-zero attention mappings; neither paper establishes HONF's physical accuracy or sparse GPU efficiency.

**[R14]** Ruibin Xiong and colleagues. *On Layer Normalization in the Transformer Architecture.* ICML 2020, PMLR 119:10524–10533. `https://proceedings.mlr.press/v119/xiong20b.html`. Relevance: standard pre-normalized Transformer blocks as a starting design rather than custom normalization control.

**[R15]** Zongyi Li and colleagues. *Geometry-Informed Neural Operator for Large-Scale 3D PDEs.* NeurIPS 2023. arXiv:2309.00583. `https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html`. Relevance: geometry-aware operator interfaces and local/global representation separation; the proposed HONF does not require adopting a Fourier latent grid.

---

**One-sentence model definition:** HONF encodes a prescribed modular scene using measure-aware attention, organizes contextualized source and environment information into a small bank of content-bearing hyperedges, and reconstructs fields through receiver-specific edge reads and bounded local detail, with source-resolved affine or nonlinear output laws supplied explicitly by the dataset contract.
