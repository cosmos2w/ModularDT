# Choosing placement edits in ThermalChannel

The engineering decision is where to move fixed-load heat sources to lower the material smooth-peak temperature while respecting a pressure-rise limit. Physical evaluations of edited layouts and cheaper estimates of their response have different costs. This investigation compares what can be done with either resource. All methods start at the same layout and can retain it.

A subsequent prospective study tests the frozen response model on 12 fresh geometries. [Unseen-layout results](#prospective-check-on-unseen-geometries-with-pretrained-r)

## Reference-only search

The released v1 corpus `data/response/new/fit_corpus.json` supplies 96 families and four converged edits per family, including 160 edits moving multiple modules. Ordered module slots, heat loads and controls stay fixed within a family. The endpoints use the release's analytic-wake and shared-grid thermal model. The target is the native material smooth peak J with smoothing scale 0.4149511814117432. An edit earns cooling equal to −ΔJ if ΔJ < −0.012 and ΔP is at most the chosen cap. All other outcomes earn zero. Temperatures and pressures use native dataset units. The zero-rise request uses the sign of the stored pressure delta, without an uncertainty band around zero.

A direct search sees an edit's stored J and P changes only after choosing to evaluate it. It retains its best verified feasible layout so far. Cooling is averaged first over all 24 permutations used for initial randomization and tie-breaking, then equally across the 96 families. A physical check below means a stored endpoint lookup. No new simulation was launched.

| Direct search | How it chooses the next edit | Mean cooling at two checks, ΔP ≤ 0.010 | Mean cooling at two checks, ΔP ≤ 0 |
| --- | --- | ---: | ---: |
| Random without replacement | Probe an untried edit uniformly | 0.3004 | 0.1571 |
| Largest displacement | Probe the largest remaining change to the ordered center coordinates | 0.3428 | 0.1374 |
| Maximin coverage | Maximize distance to the nearest tested layout, including the baseline | 0.3449 | 0.1433 |
| Local linear response | Fit minimum-norm slopes for J and P to observed displacements, then choose the predicted coolest feasible edit | 0.3060 | 0.1549 |
| Constrained Bayesian optimization | Choose the edit with largest expected cooling improvement weighted by its estimated probability of pressure feasibility | 0.3179 | 0.1502 |

**What changed with the budget.** At the looser cap, only one of 384 edits exceeded the pressure allowance. Maximin produced mean cooling of 0.1891, 0.3449 and 0.4283 at one, two and three checks. Random search gave 0.1762, 0.3004 and 0.3906. At four checks every direct method exhausted the bank and recovered the same mean of 0.4584, with a resolved opportunity in 83 of 96 families. With zero pressure rise, 174 of 384 edits exceeded the cap. The attainable mean fell to 0.2619, with opportunities in 60 families. A strategy that samples larger moves early lost its two-check advantage.

**How much did feedback help?** The linear and Bayesian policies start uniformly in the table. Giving them the same largest-displacement first probe as maximin changed their two-check means to 0.3177 and 0.3435 at the looser cap. A random second probe after that start yielded 0.3081. At the zero-rise cap the corresponding values were 0.1662, 0.1421 and 0.1609. The adaptive methods could exploit one observation, but their ordering depended on the constraint. In a paired bootstrap over families, the maximin-minus-random difference at two checks was +0.0446 (95% interval −0.0055 to +0.0923) at the looser cap and −0.0138 (−0.0480 to +0.0198) at zero rise. These are empirical intervals for this family collection.

The Bayesian control used independent zero-mean Gaussian processes over center displacements. Its squared-exponential kernel used the median candidate displacement norm as length scale, the baseline supplied an observed zero change, and a diagonal term of 1e−8 stabilized the solve. Per-output scales were the largest absolute observed changes with floors 0.012 for J and 0.001 for P. Acquisition required improvement over the better of the incumbent cooling and 0.012. The linear model used the same baseline, preferred the least predicted pressure violation if all remaining edits exceeded the cap, and used the same tie order. Neither method fitted its hyperparameters on the outcomes of this study.

## Screening candidates before spending one physical check

A predictor can inspect candidate layouts and then submit at most one edit for physical verification. There is no second attempt after rejection. The comparison below uses a geometry-only regressor and the frozen B0 and R field predictors. All have access to the baseline and the four candidate layouts. A shortlist controls how many candidate responses the field model may evaluate. Baseline scoring is additional. The measured outcome is verified cooling per family, including zeros for abstentions and rejections.

**Geometry-only control.** Two ridge regressions predict ΔJ and ΔP from geometry and prescribed heat loads. Each sees 380 labeled edits from the other 95 families and excludes all four edits from the target family. Features sum, over modules, products of signed and absolute x/y motion and motion magnitude with load, normalized position, wall proximity and upstream/downstream neighboring heat weighted by distance. Standardization uses only the fitting families. The ridge penalty is fixed at 10. This control ranks all four candidates, submits the coolest predicted improvement within the cap, and needs no native field prediction.

**Field-model controls.** B0 learned native state reconstruction. R also learned responses on the FIT families evaluated here. Each uses its saved epoch-3000 adapter, restricted access and the same smooth-peak observation. Candidate order is uniform, largest displacement first, hot-receiver proximity, or environment-directory relevance. The last two rank edits by the displacement-weighted share of module relevance; they need a baseline field evaluation. Only the shortlisted candidates' predicted J and P enter the choice. For this retrospective comparison, each distinct prediction was computed once or reused from the preceding replay; budgets below count what an individual decision consumes. [Scoring and relevance definitions](../../Case_ThermalChannel/src/channelthermal/inverse/placement_benchmark.py)

| Candidate screen | ΔP ≤ 0.010, two estimates | ΔP ≤ 0.010, four | ΔP ≤ 0, two | ΔP ≤ 0, four |
| --- | ---: | ---: | ---: | ---: |
| Geometry ridge | — | 0.2638 | — | 0.1037 |
| B0 uniform | 0.1752 | 0.2563 | 0.0661 | 0.1120 |
| B0 largest displacement | 0.2001 | 0.2563 | 0.0651 | 0.1120 |
| B0 proximity | 0.2306 | 0.2563 | 0.0711 | 0.1120 |
| B0 directory | 0.2374 | 0.2563 | 0.0867 | 0.1120 |
| R uniform | 0.2916 | 0.4401 | 0.1444 | 0.2382 |
| R largest displacement | 0.3345 | 0.4401 | 0.1288 | 0.2382 |
| R proximity | 0.3621 | 0.4401 | 0.1567 | 0.2382 |
| R directory | 0.3897 | 0.4401 | 0.1818 | 0.2382 |

At a pressure allowance of 0.01, R's directory-minus-uniform difference at two candidate evaluations was +0.0982 (95% interval +0.0527 to +0.1405). Its difference from hot-receiver proximity was +0.0276 (95% interval -0.0111 to +0.0630). With directory guidance, the mean best physically attainable cooling among the two selected candidates was 0.3989. Verification retained 0.3897. The remaining difference reflects selection or abstention after shortlisting.

At a pressure allowance of 0, R's directory-minus-uniform difference at two candidate evaluations was +0.0373 (95% interval +0.0071 to +0.0685). Its difference from hot-receiver proximity was +0.0251 (95% interval +0.0037 to +0.0499). With directory guidance, the mean best physically attainable cooling among the two selected candidates was 0.1942. Verification retained 0.1818. The remaining difference reflects selection or abstention after shortlisting.

| Four-candidate screen | Pressure allowance | Physical checks issued / 96 | Checks over pressure cap | Families retaining resolved cooling |
| --- | ---: | ---: | ---: | ---: |
| Geometry ridge | 0.01 | 95 | 1 | 50 |
| Geometry ridge | 0 | 78 | 32 | 20 |
| B0 | 0.01 | 93 | 0 | 56 |
| B0 | 0 | 71 | 19 | 28 |
| R | 0.01 | 91 | 0 | 82 |
| R | 0 | 69 | 3 | 52 |

The geometry-only model gave 0.2638 mean cooling at the looser cap, close to B0's 0.2563. B0 succeeded in more families. At zero rise the geometry model submitted 32 over-cap edits in 78 attempts; R submitted three in 69. A verifier rejected all such edits. The tight-cap result is sensitive to pressure resolution. In total, 137 of 384 stored ΔP values lie within 0.0001 of zero. An engineering deployment would need a pressure uncertainty margin. Training exposure also differs. The ridge model excluded the evaluation family; R had access to its endpoints during response training. Their gap combines differences in information and method.

## Combining a ranking with physical search

A screen can also set the order of a direct search. Rank all four estimates by predicted pressure excess, then predicted J, and physically check the first two even if the model expects little cooling. Keep the best verified incumbent. This uses four cheap candidate estimates and two physical checks per family. It tests the value of committing a second check and of allowing the verifier to override model pessimism.

| Ordering for two physical checks | ΔP ≤ 0.010 | ΔP ≤ 0 |
| --- | ---: | ---: |
| Uniform, no fitted response model | 0.3004 | 0.1571 |
| Maximin geometry, no fitted response model | 0.3449 | 0.1433 |
| Geometry ridge, trained on other families | 0.3524 | 0.1951 |
| B0 native-field response | 0.3671 | 0.2089 |
| R native-field response, FIT exposure | 0.4529 | 0.2523 |
| Best available layout after four physical checks | 0.4584 | 0.2619 |

If an independent reference evaluation dominates cost, spending it on a ranked edit can be worthwhile even with a simple geometry response model. The geometry ridge increased two-check mean cooling over random by 0.0520 at the looser cap and 0.0380 at zero rise. When all four outputs are estimated, predictor quality and the verification rule determine the outcome; the order in which the model reads all four cannot improve it. Wall-time savings, full-access comparison and prospective reference-solve failures were outside this stored-endpoint experiment.

## Protecting downstream temperatures

The availability of a good edit changes with the design request. Of the 96 families, 44 have a baseline and four endpoints that all clear the three previously defined downstream rectangles. On this same subset, 37 families offered resolved cooling with the 0.010 pressure allowance alone. Requiring zero rise in the lower-intake maximum and upper-intake mean left 24 families. Also requiring zero rise in outlet temperature standard deviation left 23. Each statistic uses the same valid native fluid receivers across a family's five configurations. [Footprints and previous two-family replay](README.md#r16-b0-and-r-refresh)

The corresponding mean best available cooling was 0.4540, 0.3050 and 0.3032. Two uniformly drawn physical checks retained 0.3088, 0.1840 and 0.1828 on average. The earlier two-family example had no resolved opportunity under intake protection. Across the broader geometry-compatible subset there are enough feasible and infeasible outcomes to test a request-aware ranking. These counts use stored physical fields; the 44-family temperature-constrained screen has yet to be evaluated with either checkpoint.

## Implications for the design study

**Predictor.** A compact geometry-to-response model is a useful engineering baseline when the request is fixed and endpoint labels exist. The next comparison should give competing response predictors the same training families and reserve complete unseen layouts for evaluation. Recompute one verification attempt per family, the number of rejected proposals, and mean verified cooling. The current corpus supports a family-excluded geometry model and an exposed R replay. It cannot recover an unexposed assessment of the already-trained R checkpoint from these endpoints.

**Organizer.** Directory relevance was useful when only part of the bank could be scored. At two evaluations it improved on uniform shortlisting at both pressure caps. Against proximity the paired interval included zero at the looser cap. On unseen families compare both with the same scorer and budget, and retain an explicit geometry-only shortlist. The existing narrative can claim a shortlist effect on these FIT families. The evidence does not identify physical interactions from a relevance score.

**Inverse algorithm.** For a fixed four-edit bank, use uniform and maximin as reference-only controls and geometry-ranked verification as a learned control. Give constrained Bayesian optimization a larger legal domain and a sequence of observations before drawing a conclusion about sequential adaptation. A prospective test could start with the same legal baseline and first verified probe for every method, hold physical attempts fixed, and allow new edits after feedback. Compare a coordinate or mesh poll, constrained Bayesian optimization, and surrogate-ranked physical search. Repeat on unseen families under both a pressure-only and an intake-protection request. Record all attempted solves, rejected and duplicate layouts, accepted cooling and end-to-end cost. The four-edit banks do not exercise mesh refinement, multi-generation evolution, continuous gradients or learned generation of new layouts. Those methods need a verifier for newly proposed coordinates. [Mesh adaptive direct search](https://doi.org/10.1137/040603371) · [Constrained Bayesian optimization](https://proceedings.mlr.press/v32/gardner14.html)

Advance a more elaborate search policy if its paired verified cooling improves on the strongest simple control at a matched physical budget across unseen families and its over-cap submission rate stays acceptable for the application. The current comparisons are retrospective. Family bootstrap intervals use 10,000 paired resamples of the 96 families; they describe variation across this collection and do not correct for the multiple policy comparisons.

## Prospective check on unseen geometries with pretrained R

The same frozen epoch-3000 R checkpoint was tested on newly sampled layouts. Its directory shortlisted useful moves, but the model frequently predicted the wrong direction of temperature change. At the primary pressure limit, a geometry-only regressor delivered more cooling with one verification. When both shortlisted edits were physically checked, the directory recovered substantially more cooling than proximity.

**Panel and verification.** The pilot used four layouts each with two, four and six modules. Each reused operating controls and heat loads from a released family and drew new centers with the channel generator. The 12 baselines and 48 edits were checked against the packed dataset and both released edit catalogues for duplicate geometry. All layouts satisfy the configured clearances, pair separation and downstream footprint requirements. Each family has two candidate module slots, with one half-grid and one 1.5-grid displacement per slot in sampled directions. The family and tie-order seed was 18439, with one common randomized candidate order per family. All one-check recommendations were saved before running the new physical simulations. The native NumPy analytic-wake and shared-grid thermal solver first reproduced a distributed edited endpoint's final pressure, temperature, material temperature and masks exactly. All 60 new baseline and endpoint solves converged. Only final frames were serialized, using the ordinary convergence-check cadence.

The scorer used restricted access, saved normalization and smoothing scale 0.4149511814117432. The geometry control used the previously specified ridge model and penalty, fitted on all 384 released FIT edits for J and P and the 176 compatible FIT edits for intake deltas. The pressure-only and intake requests allow a nominal ΔP of 0.010. The zero-rise request allows a nominal ΔP of zero. All methods applied a fixed conservative margin of 0.0001 to an edited layout's nominal limit in both screening and verification. This margin is a study setting in native pressure units; it has no calibrated confidence interpretation. Intake protection also requires zero rise in the lower-intake peak and upper-intake mean. Acceptance requires reference cooling greater than 0.012.

**Verified mean cooling.** Each cell gives the mean across all 12 families, with one allowed physical check followed by the result with two allowed checks. With one check, a model can abstain. With two checks, both proposals are physically evaluated and the best acceptable result is retained even if the predictor was pessimistic. Geometry ranks four cheap estimates. Each R shortlist uses one baseline forward and two candidate forwards. Full-pool R scores all four candidates as a diagnostic. Reference calls are charged per policy; the experiment reused physical results for identical layouts.

| Policy | Pressure cap, one / two | Zero rise, one / two | Intake protection, one / two |
| --- | ---: | ---: | ---: |
| Direct random physical checks | 0.0528 / 0.1030 | 0.0351 / 0.0554 | 0.0484 / 0.0515 |
| Geometry-ranked verification | 0.0921 / 0.1720 | 0.0290 / 0.0744 | 0.0031 / 0.1538 |
| R uniform shortlist | 0.0061 / 0.1030 | 0.0017 / 0.0554 | 0 / 0.0515 |
| R hot-receiver proximity | 0.0141 / 0.1024 | 0 / 0.0800 | 0 / 0.0436 |
| R directory | 0.0044 / 0.1799 | 0 / 0.1108 | 0 / 0.1211 |
| R full-pool ranking | 0.0141 / 0.1710 | 0 / 0.0922 | 0 / 0.0467 |
| Best possible in the four-edit bank | 0.2651 | 0.1108 | 0.1581 |

**Predictor.** Of 35 edits with resolved absolute ΔJ, R and the geometry regressor each predicted the direction correctly for 17. Mean absolute ΔJ error was 0.2834 for R and 0.2469 for geometry. R had smaller mean absolute ΔP error, 0.000365 versus 0.000599. For family `new2_2`, the directory selected the two edits that physically cooled the material by 0.3696 and 0.8869. R predicted warming for both and abstained. The geometry model selected the 0.8869 edit. A useful next predictor check is whether uncertainty in ΔJ flags sign errors on new layouts.

**Organizer.** At the primary pressure cap, the directory shortlist included a resolved feasible edit in all 10 families with an opportunity; proximity covered nine. Checking both directory proposals achieved 0.1799 mean cooling versus 0.1024 for proximity. Both R shortlists focus on one of two movable modules because their relevance scores rank slots and tie the two moves of a selected slot. This design tests module prioritization; a larger move bank with multi-module proposals is needed to test richer organization.

**Inverse decisions.** With one allowed physical check under the primary pressure limit, directory guidance issued five checks and retained one improvement. Proximity issued seven and retained two. Geometry issued 12 and retained three. The mean paired one-check difference for directory minus geometry was −0.0876, with a descriptive 95% family-bootstrap interval of −0.2396 to 0.0000. Under intake protection, R full-pool ranking issued five checks and all five violated a constraint; the bank contained an acceptable improvement in six families. A practical next comparison is a geometry-ranked first proposal followed by a physically verified directory alternative, evaluated on fresh layouts with the same two-check allowance. These results apply to this seeded, small single-slot edit bank and the channel solver. End-to-end acceleration, full-access fallback and hardware performance were unmeasured.
