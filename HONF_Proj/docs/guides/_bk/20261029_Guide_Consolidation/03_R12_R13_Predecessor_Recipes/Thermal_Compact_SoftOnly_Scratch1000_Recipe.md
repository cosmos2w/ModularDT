# Thermal Round12 soft-only scratch1000 v2

**Archived reference — 29 October 2026.** This guide records a predecessor model or a dated experiment, not current architecture policy or standing launch authorization. Use the [current guide index](../../../README.md); see the [archive review](../README.md) for the reason it was moved.

This separately named full-data run learns prediction and soft retrieval organization jointly from random initialization. It removes the scratch v1 epoch40 hard cut and policy freeze, corrects the bank/scene/candidate normalization and calibration coverage, and adds bounded supervision from existing TRAIN geometry differences. It tests whether adjacent same-type candidates can share useful retrieval controls. Actual K remains N throughout; neither compression nor speed is an outcome of this experiment. The [executable frozen configuration](https://github.com/cosmos2w/ModularDT/blob/7981c173e8b1d29ee2b3addb7b972d0f15003542/HONF_Proj/configs/research/thermal_compact_softonly_full1000_scratch_v2.json) preserves original records, all physical source columns, the single-level candidate rule, input-only E policy and the original source law. The earlier [Round12 synthesis](../../../../reports/HONF_Round12_Consolidated_Scientific_Report.md) concerns different checkpoints and supplies no v2 accuracy result.

## Mathematical objective and boundaries

For local epoch e, alpha(e)=0.1+0.45(1−cos(pi(e−1)/199)) for 1≤e≤200, and alpha(e)=1 thereafter. Both sibling slots survive; their live descriptors use the existing symmetric soft blend with alpha times sigmoid(b). Policy and predictor remain trainable at every epoch. No strict-cut threshold, acceptance quota, hard topology transition or post-cut recovery exists in this recipe. Fixed-input E probabilities remain insensitive to module translations under the existing input constraint; live contextual descriptors, C and query access can still depend on geometry.

For bank B∈{M,E}, let P_sB be its eligible sibling set in scene s and S_B the scenes with at least one eligible pair. The raw losses are L_tie,B=mean_{s∈S_B} mean_{v∈P_sB} stopgrad(p_v) D_v and L_cap,B=mean_{s∈S_B} mean_{v∈P_sB}(1−p_v). D retains the original equal-nonempty-donor-type mean absolute difference between the two live C rows. Empty banks contribute no gradient and no fictitious valid scene. This bankwise capacity surrogate expresses soft sharing preference only; it is not measured K reduction. The existing alpha multiplier remains outside each structural term.

At the beginning of epochs1,100,200, a fixed input-only TRAIN panel of at most96 cases is replayed without an optimizer step. Cases are sorted by ID within every available module-count category and interleaved round-robin, covering all categories and their M candidates; no case is selected by prediction error. Both losses for each bank use exactly that bank's complete compatibility-MLP parameter set, with absent VJPs filled by zero. Each coefficient is 0.02 times the native policy-gradient norm divided by its own raw policy-gradient norm, including the current alpha, or zero when either norm vanishes. No positive floor or desired-merge amplification is permitted. Native means ordinary task/auxiliary/heat-response gradients plus the maintained completed-macro M/E/context work-controller VJPs, before clipping. Record same/opposed signs, signed dot products, zero/missing gradients and actual candidate/scene coverage. Coefficients from epoch200 remain fixed through1000.

At every primary update, sum the four weighted structural gradients and multiply them by min(1,0.1||g_native,full||/||g_structure||), using zero-safe arithmetic. This downward-only bound is applied to the combined full-model structural gradient, not independently to each term. It preserves native gradient competition and is separate from AdamW weight decay and final clip1. The native work-controller contracts are unchanged.

Every fourth primary update adds one geometry-response microbatch before the same optimizer step; no extra update is created. Cycle the sixteen existing baseline-to-i_minus/i_plus/j_minus/j_plus differences from TRAIN families0001/0318/0333/0348 deterministically. Use512 common valid Eulerian fluid receivers, up to64 common module-local material receivers per module and16 module-local surface angles, with original module IDs and unchanged heat amplitudes. Physical components have separate scales fitted as equal-family/direction RMS from these sixteen TRAIN differences only (numerical scale floor1e-8). The loss is the equal-component mean squared scaled difference error. Geometry calibration uses one fixed difference per TRAIN family, with direction offset by family index to cover i_minus/i_plus/j_minus/j_plus, and targets10% of the native shared-predictor gradient norm, excluding compatibility-policy parameters. This shared-slice norm determines one coefficient applied to the full geometry gradient, including policy parameters. Its coefficient is recalibrated only at1/100/200 and thereafter fixed; a downward-only per-update cap at10% of the current native shared norm bounds unexpected transients. Native response forwards retain gradients in evaluation mode to exclude extra requests from the primary work-controller denominator. Record all response scene/query exposure separately from primary training and calibration.

Response-development families0304/0320/0335/0350 never enter a differentiable training or scale/calibration path. Mixed-geometry quartets are validation only for every family. All eight references are existing original-TRAIN physical families, with the latter four held out specifically from response training; this distinction does not convert them into independent TEST data. No solver, new physical dataset or locked TEST read is allowed.

## Frozen training, retention and budgets

| Item | Frozen contract |
| --- | --- |
| Initialization | Seed0 random weights; fresh TRAIN600-only statistics; no checkpoint/pretrained state |
| Membership | Full TRAIN600; canonical exposed VAL89, excluding0273; exact existing factory identities |
| Optimizer | AdamW3e-4, WD1e-5, betas(0.9,0.999), eps1e-8; clip1 |
| Primary work | Effective48/micro24/Q1024;600 visits and13 updates per epoch;1000 epochs gives600000 visits,13000 updates and614400000 primary queries |
| Extra geometry work |3250 response microbatches if all13000 updates complete; extra scene/query counts recorded from executed panels |
| Calibration | Three no-step96-case maximum TRAIN replays plus their four-family geometry panels; restored model/Adam/RNG/sampler state |
| Controllers | Fresh native coordinate3000→4000, original fixed fresh lambda0, source law and work penalties |
| Runtime | Only explicit cuda:0; preserve visible-device list and every saved CUDA RNG stream; maintained OOM rollback/repacking |
| Regular validation/output | Exposed VAL89 every10 epochs, tqdm enabled, loss plots every10, complete history |
| Checkpoints |100/200/300/400/500/600/700/800/900/1000 plus best/latest; clean-stop final resumable state; no cut snapshots |
| Time | Qualification≤2 GPU hours; training≤36; internal organizer/terminal/best validation≤4; no automatic extension |
| Storage | Combined retained run/preparation evidence≤12GiB; disposable temporary≤1GiB; DATA2 free headroom≥200GiB |

Qualification uses four disposable maintained macros at phases1/100/200/201, all three no-update calibrations, an attached response microbatch, strict own-state model/named-Adam/RNG restoration and measured peak memory/checkpoint size. Its weights never initialize the scientific run. A conservative projection uses the measured ordinary macro time times13000 with75% margin and1800 seconds overhead; qualification/calibration time is recorded separately. A failed qualification prevents launch. The finite wall-time guard requests the maintained epoch-boundary clean stop with600 seconds reserved for the remaining epoch/save, and storage checks reserve atomic checkpoint space. The run never expands beyond1000 epochs or its declared budget.

## Prespecified validation and decisions

Internal organizer reviews at100/300/600/1000 use fixed input-only TRAIN representatives. Report every bank's probabilities, eligible counts, actual K=N, raw C amplitudes, nondegenerate normalized row differences, query access, gradient competition and bounded two-slot full-sharing interventions. For each nonempty eligible donor type, normalize each C row by its L1 mass and compute half the L1 difference; average across types. Row mass≤1e-8 is degenerate and cannot pass. A promising measured candidate requires p≥0.8, normalized row difference≤0.1 and intervention-induced RMS change≤0.1 native RMSE separately for fluid/surface/material temperature and recomputed module peaks. Report measured-versus-eligible coverage, repeated-pair stability and failures; high probabilities, small training tie loss or attractive graphs alone do not pass.

The review panel uses the lowest-ID case from the minimum, next, median and maximum available module-count categories. Across that entire panel, intervene on at most two pairs per bank, ranking promising pairs first, then probability and stable case/slot IDs. If no pair passes the first two gates, use bounded highest-probability unqualified probes; a probe does not become a promising group merely because its intervention is small. Every review has a120-second bound checked between native cases/interventions, and its actual time is charged to the shared4-hour validation budget. Query-access summaries use maintained query tiles.

At the explicit terminal checkpoint and the independently identified selected-best checkpoint, perform full native exposed VAL89 evaluation of u,v,p,omega,fluid T,surface T,material T,q proxy and recomputed module peaks. Keep equal-case mean,p90,worst case,module-count strata,physical masks and native units. Replay all existing single geometry, heat and mixed geometry references, splitting TRAIN4 from held-out response DEV4, absolute error from relative error, and mixed responses from unary responses. Near-null references require absolute error interpretation. Cache/source laws, graph appearance and AD/FD consistency are not substitutes for physical response accuracy. This run does not conduct a candidate-ranking campaign, claim a verified inverse design or identify physical causes.

## Launch and status

Use the maintained runner in one persistent background terminal, with no CUDA_VISIBLE_DEVICES override:

```bash
/home/wanglz/miniconda3/envs/ModularDT/bin/python -B -u HONF_Proj/tools/honf_research.py train \
  --config HONF_Proj/configs/research/thermal_compact_softonly_full1000_scratch_v2.json \
  --device cuda:0 --devices cuda:0 --epochs 1000 \
  --output HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_compact_softonly_full1000_scratch_v2
```

Retain preparation/qualification receipts and resolved input/reference identities in ignored DATA2 diagnostics; numerical outputs and scientific figures remain ignored. Perform one immediate startup check after the scientific launch, then end the interaction without polling or creating external monitors. Startup establishes execution only; training and scientific validation remain incomplete until the internal bounded steps finish.
