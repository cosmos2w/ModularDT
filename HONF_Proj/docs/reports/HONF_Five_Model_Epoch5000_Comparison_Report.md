# Five-model HONF comparison at epoch 5,000

Date: 2026-09-12; analysis executed September 11–12. This report compares the completed Legacy 1401, latent-attention 1801, Dense 1804, geometry-envelope Reader 1805, and regional-response 1806 runs. It adds the missing mature latent and regional evaluations to the established evidence rather than replacing prior reports.

**Dense and Regional are the leading reconstruction models, with a checkpoint-sensitive ordering.** Exact-5,000 pooled normalized fluid relative L2 is **0.029661 for Dense and 0.031188 for Regional**, a 5.15% Regional gap (10.57% in MSE). Using each run's saved best-by-validation-field checkpoint through 5,000 instead, **Regional reaches 0.028192 and Dense 0.028960**. Regional improves on Legacy and both compressed-reader alternatives under both policies, while retaining a useful regional path. Its early accuracy lead does not persist at the exact final epoch; a blanket claim that Dense always wins would also be unsupported.

**Latent's completed long run does not close the accuracy gap.** Its exact-endpoint L2 is **0.090399**, versus Reader's **0.065865** and Legacy's **0.037401**. Latent worsens from its 2,500 endpoint and has the largest difficult-case tail. Attention activity alone therefore does not establish useful reconstruction capacity.

The report separates measured reconstruction, frozen-checkpoint reliance, geometric organization, and execution. Independent physical-response validation remains pending. The companion [HTML figures](../../diagnostics/generated/interface_operator_study/five_model_epoch5000/figures/index.html) provide interactive comparisons and anchor views from the same local evidence.

## Model identities and fair comparison scope

| Run | Role | Main response computation | Completion |
|---|---|---|---|
| 1401 | Historical HONF baseline | Historical organizer and field/physical coupling | 5,000 |
| 1801 | Latent-attention baseline | 16 learned main latents with fixed geometric references, plus 8 common coarse latents | 5,000 |
| 1804 | Dense baseline | Fine simultaneous MM/ME/EM messages, dense environmental attention, direct query–module read | 5,000 |
| 1805 | Sparse geometry-envelope HONF design | Compact geometric groups with a geometry-envelope group read | 5,000 |
| 1806 | Regional-response HONF design | Dense fine joint messages, 2×2 pre-update pooling into 48 regional states, shared port/field reads | 5,000 |

Runs 1800 and 1803 were failed Dense execution attempts; 1804 is the successful original Dense baseline. Run 1802 completed 500 epochs but its main group read became negligible. Run 1805 changed that read normalization; Run 1806 instead tested compression after Dense's fine collective messages. Neither 1805 nor 1806 is the latent baseline. The history is documented in the [original interface study](HONF_Interface_Study_Report.md), [reader recovery](HONF_Group_Reader_Recovery_Report.md), and [regional early assessment](HONF_Regional_Response_Compression_Report.md).

The four interface-field families share the established physical wrapper, predicted ports, frozen Stage A, one refinement, common coarse/local paths, losses, and width policy. Legacy remains a historical reference with its original architecture and configuration; it is not a parameter-matched implementation of the newer physical interfaces. These are purpose-built adaptations, not exact reproductions of published architectures.

Reader 1805 keeps Run 1802's compact-support architecture and changes its read to a geometry-envelope-weighted attention: supported values are normalized relative to one another and their mixture is scaled by the total geometric envelope. This prevents the old null competitor from suppressing the entire useful read. Regional 1806 takes a different route: preserve Dense's fine MM/ME/EM joint messages, pool encoded environment and module-conditioned EM responses within 2×2 regions **before** the existing environmental-update MLP, and read the resulting 48 states at both ports and field queries. The regional route replaces the fine environmental read; the direct query–module and common coarse/local paths remain. The five-model results are consistent with preserving fine collective-response preparation being valuable, but they do not isolate every architectural difference between Reader and Regional.

## Evaluation protocol and evidence coverage

Primary accuracy uses the **exact stored epoch-5,000 checkpoints**, all **90 development-holdout cases**, and all **8,192 field query locations** before the fluid mask is applied. Checkpoint-owned normalization and predicted-port physical coupling are retained. Near-interface and far-fluid masks use surface distances 0–0.25 and at least 1.0 in dataset coordinates; the separate spacing/wall strata use radius-normalized geometry.

Existing 1401/1804/1805 tables at 500, 2,500, and 5,000 are reused in place. New standard-evaluator passes cover 1801/1806 at exact 2,500 and 5,000. Their epoch-500 tables already existed. Thus the full-grid trajectory below has **15 model/epoch datasets**, distinct from sampled training-validation histories. No training was launched for this analysis and no checkpoint was copied.

All datasets contain the same 90 unique IDs. The reducer reconciled **75,600 target/count comparisons over 56 common columns**, **540 stored pooled values against per-case numerators**, and **40 five-channel MSE decompositions**. The common global denominator is 3,496,800 fluid scalar values, with target squared norm 3,237,304.06052. Pooled relative L2 is `sqrt(sum SSE / sum target SSE)`; pooled MSE is `sum SSE / number of values`. Equal-case statistics summarize each case's own relative L2 without size weighting. These definitions should not be interchanged.

This remains a one-seed study on a repeatedly inspected development holdout. Case counts and paired deltas describe this cohort; they do not measure training-seed uncertainty or establish an untouched-test result. Best-selected validation checkpoints are considered separately from exact endpoints.

## Exact-5,000 reconstruction

| Model | Pooled MSE | Pooled L2 | Equal-case mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Legacy 1401 | 0.00129505 | 0.037401 | 0.034705 | 0.031616 | 0.060236 | 0.080615 |
| Latent 1801 | 0.00756557 | 0.090399 | 0.072500 | 0.053792 | 0.189624 | 0.240695 |
| Dense 1804 | 0.00081448 | 0.029661 | 0.026433 | 0.022646 | 0.052161 | 0.064715 |
| Reader 1805 | 0.00401630 | 0.065865 | 0.054565 | 0.039926 | 0.130381 | 0.177697 |
| Regional 1806 | 0.00090053 | 0.031188 | 0.028221 | 0.025707 | 0.050395 | 0.073074 |

Dense beats Legacy on 80/90 field cases, Latent on 90/90, and Reader on 90/90. Regional beats Legacy on 81/90, Latent on 90/90, and Reader on 90/90, but beats Dense on only **23/90**. Regional has a slightly better p95 than Dense (0.050395 versus 0.052161), yet a worse maximum (0.073074 versus 0.064715) and mean. The tail comparison is mixed, not uniformly better.

| Metric | Legacy 1401 | Latent 1801 | Dense 1804 | Reader 1805 | Regional 1806 |
|---|---:|---:|---:|---:|---:|
| Near-interface normalized field | 0.033693 | 0.068733 | 0.035086 | 0.047437 | 0.035932 |
| Far-fluid normalized field | 0.041778 | 0.085094 | 0.026495 | 0.063757 | 0.028105 |
| Normalized u | 0.021273 | 0.047995 | 0.014359 | 0.029002 | 0.013856 |
| Normalized v | 0.017297 | 0.070603 | 0.013121 | 0.059069 | 0.015015 |
| Normalized p | 0.049151 | 0.124772 | 0.025573 | 0.082507 | 0.028527 |
| Normalized omega | 0.041203 | 0.095011 | 0.041888 | 0.066364 | 0.041869 |
| Normalized temperature | 0.052285 | 0.108424 | 0.040705 | 0.085696 | 0.044749 |

Regional is slightly better than Dense in normalized u and effectively tied in vorticity, while v, pressure, and temperature are worse. This differs from epoch 500, where v supplied the entire net Regional gain. Legacy remains best near interfaces and in vorticity. Comparing a single aggregate field metric would hide these differences.

The five channels contribute respectively −2.638, +11.836, +22.575, −0.349, and +54.629 × 10⁻⁶ to Regional-minus-Dense global MSE. They reconcile to the +86.052 × 10⁻⁶ total gap. Temperature supplies about 63.5% of the net gap, with pressure and transverse velocity accounting for most of the remainder.

The corresponding denormalized field-channel relative L2 values are:

| Metric | Legacy 1401 | Latent 1801 | Dense 1804 | Reader 1805 | Regional 1806 |
|---|---:|---:|---:|---:|---:|
| Physical u | 0.012064 | 0.027217 | 0.008143 | 0.016446 | 0.007857 |
| Physical v | 0.017297 | 0.070603 | 0.013121 | 0.059069 | 0.015015 |
| Physical p | 0.045904 | 0.116529 | 0.023883 | 0.077057 | 0.026643 |
| Physical omega | 0.041203 | 0.095011 | 0.041888 | 0.066364 | 0.041869 |
| Physical temperature | 0.041622 | 0.086313 | 0.032404 | 0.068220 | 0.035623 |

Relative errors in normalized and physical channels can differ because normalization may include a nonzero offset. They are not independent repetitions of the same measurement.

## Ports, internal response, flux, and engineering quantities

| Metric | Legacy 1401 | Latent 1801 | Dense 1804 | Reader 1805 | Regional 1806 |
|---|---:|---:|---:|---:|---:|
| Provisional outside T | 0.226292 | 0.099400 | 0.066390 | 0.076243 | 0.067997 |
| Final outside T | 0.063595 | 0.098344 | 0.073197 | 0.081575 | 0.079636 |
| Provisional heat-transfer coefficient | 0.513136 | 0.237539 | 0.178643 | 0.234237 | 0.183216 |
| Final heat-transfer coefficient | 0.050224 | 0.048707 | 0.046379 | 0.048912 | 0.047771 |
| Internal T | 0.034742 | 0.060621 | 0.025744 | 0.044122 | 0.026133 |
| Surface T | 0.045998 | 0.078534 | 0.037510 | 0.059751 | 0.039070 |
| Interface normal heat flux | 0.131790 | 0.120709 | 0.103487 | 0.105010 | 0.105374 |

These are physical pooled relative L2 values over their appropriate active receivers. Dense leads internal/surface temperatures, final heat-transfer coefficient, and flux among the compared models. Regional is close in internal temperature and flux, while final outside temperature remains worse than Legacy and Dense. Reader's flux is also close to Dense despite its substantially worse field reconstruction; this does not imply equivalent full physical response.

Existing KPI mean absolute errors, in dataset physical units:

| Metric | Legacy 1401 | Latent 1801 | Dense 1804 | Reader 1805 | Regional 1806 |
|---|---:|---:|---:|---:|---:|
| Pressure drop | 0.003042 | 0.003504 | 0.001031 | 0.002140 | 0.001310 |
| Outlet temperature | 0.261771 | 0.246849 | 0.155314 | 0.275935 | 0.194222 |
| Active-module mean temperature | 0.237370 | 0.210691 | 0.113180 | 0.216052 | 0.098947 |

Regional has the best active-module mean-temperature KPI, while Dense leads pressure-drop and outlet-temperature errors. An average-module KPI and the full internal-temperature field weight errors differently, so these rankings can legitimately differ.

## Full-grid convergence and geometry-dependent errors

| Model | L2 @500 | L2 @2,500 | L2 @5,000 | Change 2,500→5,000 |
|---|---:|---:|---:|---:|
| Legacy 1401 | 0.117148 | 0.045285 | 0.037401 | -17.41% |
| Latent 1801 | 0.145333 | 0.085111 | 0.090399 | +6.21% |
| Dense 1804 | 0.098741 | 0.048835 | 0.029661 | -39.26% |
| Reader 1805 | 0.139648 | 0.074600 | 0.065865 | -11.71% |
| Regional 1806 | 0.096652 | 0.047089 | 0.031188 | -33.77% |

Regional stays close to Dense through training but the ranking reverses: it leads slightly at 500 and 2,500, then trails at 5,000. Legacy's previously observed intermediate advantage at 2,500 also reverses by 5,000. Reader improves more slowly. Latent's 6.21% L2 deterioration from 2,500 to 5,000 requires the history/best-checkpoint qualification below rather than an assumption that more epochs always improve endpoint accuracy.

Pooled normalized fluid L2 across the 13 predefined, overlapping strata:

| Stratum | n | Legacy 1401 | Latent 1801 | Dense 1804 | Reader 1805 | Regional 1806 |
|---|---:|---:|---:|---:|---:|---:|
| Heating CV: high_cv_>=0p35 | 20 | 0.045459 | 0.124807 | 0.037560 | 0.092023 | 0.036764 |
| Heating CV: low_cv_<0p25 | 28 | 0.034996 | 0.079013 | 0.026163 | 0.055427 | 0.029237 |
| Heating CV: medium_cv_0p25_to_0p35 | 42 | 0.034349 | 0.075777 | 0.027172 | 0.055612 | 0.029317 |
| Modules: 10 | 15 | 0.033057 | 0.058815 | 0.025238 | 0.042252 | 0.029800 |
| Modules: 3 | 25 | 0.027952 | 0.042131 | 0.017586 | 0.034699 | 0.018071 |
| Modules: 5 | 25 | 0.036677 | 0.090002 | 0.030289 | 0.065456 | 0.033918 |
| Modules: 7 | 25 | 0.045887 | 0.127405 | 0.037814 | 0.092108 | 0.036625 |
| Spacing: crowded_<1r | 45 | 0.039565 | 0.094248 | 0.032275 | 0.070260 | 0.034358 |
| Spacing: intermediate_1r_to_2p5r | 30 | 0.036979 | 0.097813 | 0.028664 | 0.067175 | 0.028863 |
| Spacing: separated_>=2p5r | 15 | 0.027878 | 0.044543 | 0.018416 | 0.037968 | 0.020722 |
| Wall distance: interior_>=2p5r | 12 | 0.024660 | 0.047089 | 0.019434 | 0.037453 | 0.019835 |
| Wall distance: middle_1p5r_to_2p5r | 50 | 0.038302 | 0.092509 | 0.029677 | 0.068159 | 0.031013 |
| Wall distance: near_<1p5r | 28 | 0.039417 | 0.097863 | 0.032519 | 0.069360 | 0.034647 |

Regional beats Legacy, Latent, and Reader in every listed stratum. Against Dense it wins only **M7** and **high heating CV**; the other 11 favor Dense. Strata overlap and covary, so these are conditional comparisons rather than isolated causal effects of module count or heating. M7 is harder than M10 for several models, illustrating why module count alone is not a sufficient complexity measure.

The hardest cases are not identical across models. Case **0298** is the maximum-error case for Latent (0.240695), Dense (0.064715), and Reader (0.177697), while Regional reaches 0.058321 there. Regional's worst case is **0283** (0.073074), which is also Dense's second hardest (0.063069). Legacy's maximum is **0297** (0.080615). These cases combine crowding, wall proximity, and heterogeneous heating. The HTML includes the four established anchors; `reduction/worst_cases.csv` retains each model's ten hardest cases at each epoch, including unfavorable Regional examples.

The four anchors used in the field-map preview and interventions have the following exact-endpoint fluid L2:

| Anchor | Legacy | Latent | Dense | Reader | Regional |
|---|---:|---:|---:|---:|---:|
| 0273 | 0.022920 | 0.038605 | 0.015654 | 0.029223 | 0.016566 |
| 0653 | 0.022296 | 0.049093 | 0.021157 | 0.038247 | 0.022022 |
| 0298 | 0.063619 | 0.240695 | 0.064715 | 0.177697 | 0.058321 |
| 0302 | 0.040667 | 0.188235 | 0.055464 | 0.122322 | 0.040930 |

Regional improves over Dense on the two difficult anchors but trails it slightly on the two ordinary anchors. Legacy is narrowly best on 0302. The four-anchor mean favors Regional over Dense, whereas the 90-case aggregate favors Dense: selected diagnostic anchors must not replace population-level evaluation.

## Sensitivity to checkpoint selection

All five saved `best_by_field_mse_model.pt` checkpoints were separately evaluated on the same 90 full-grid cases. Their actual stored epochs were read with the existing trusted loader. Selection uses the run's logged validation field MSE, not the full-grid results below; nevertheless, validation and evaluation use the same development split, so this is selection sensitivity rather than independent testing. Another **25,200 target/count comparisons** matched the exact-endpoint reference.

| Model | Selected epoch | Pooled MSE | Pooled L2 | Equal-case mean | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Legacy 1401 | 4,585 | 0.00095379 | 0.032097 | 0.029823 | 0.050556 | 0.074153 |
| Latent 1801 | 4,973 | 0.00584817 | 0.079479 | 0.056078 | 0.175542 | 0.229308 |
| Dense 1804 | 4,738 | 0.00077646 | 0.028960 | 0.025616 | 0.052812 | 0.063981 |
| Reader 1805 | 4,777 | 0.00386739 | 0.064633 | 0.053594 | 0.127950 | 0.172508 |
| Regional 1806 | 4,933 | 0.00073582 | 0.028192 | 0.024725 | 0.049550 | 0.078212 |

Regional's selected checkpoint beats Dense's on **60/90 cases** and by **2.65% pooled L2**, although its worst case remains worse. This changes the interpretation from an unconditional Dense accuracy win to a close competition with endpoint variability. Latent improves 12.08% relative to its exact-final L2 but remains well behind Reader, Legacy, Dense, and Regional. Its selected late checkpoint also beats its own exact-2,500 L2 of 0.085111: therefore the claim that *all* late Latent models deteriorated would be incorrect. The exact-final deterioration, weak difficult-case accuracy, and substantial generalization gap remain real.

Other selected-checkpoint metrics also change ordering. These are pooled relative L2 values, with normalized field rows and denormalized physical-response rows:

| Selected-checkpoint metric | Legacy | Latent | Dense | Reader | Regional |
|---|---:|---:|---:|---:|---:|
| Near-interface field | 0.032536 | 0.058505 | 0.035043 | 0.046701 | 0.034123 |
| Far-fluid field | 0.034760 | 0.072309 | 0.025369 | 0.062884 | 0.024492 |
| Field temperature | 0.046768 | 0.103521 | 0.038670 | 0.083601 | 0.036451 |
| Final outside T | 0.071710 | 0.095190 | 0.066986 | 0.080956 | 0.066842 |
| Internal T | 0.029460 | 0.059580 | 0.026805 | 0.043700 | 0.026696 |
| Surface T | 0.040548 | 0.077551 | 0.038985 | 0.059217 | 0.039013 |
| Interface normal heat flux | 0.133311 | 0.109615 | 0.106147 | 0.109735 | 0.099310 |

Regional's selected checkpoint improves far-field aggregate reconstruction, field temperature, and flux relative to selected Dense; their surface-temperature errors are nearly tied. Legacy remains best near interfaces. Thus even physical-response rankings should name the checkpoint policy. These selected results do not replace the exact-5,000 intervention and timing experiments.

## Logged convergence and historical training cost

All five runs contain unique, consecutive epochs 1–5,000, completed manifests, and loadable exact-5,000 checkpoints. Continuation histories require named-column handling: 1804 changes from 286 to 324 CSV columns and 1801 from 286 to 336. The maintained history reader was used; source logs were not rewritten. Stored normalization and optimizer metadata are retained in `history/history_summary.json`.

Logged field MSE uses the established 1,024 sampled points per case, whereas the primary reconstruction tables use full grids. The logged values are useful for learning dynamics but cannot replace the full-grid L2 values above.

| Model | Train field MSE @5k | Validation field MSE @5k | Last-100 validation median | Best logged validation field MSE | First epoch ≤0.01 |
|---|---:|---:|---:|---:|---:|
| Legacy 1401 | 0.00079253 | 0.00193250 | 0.00201585 | 0.00150997 | 831 |
| Latent 1801 | 0.00175019 | 0.01143270 | 0.00957220 | 0.00887321 | 2,357 |
| Dense 1804 | 0.00034537 | 0.00171428 | 0.00182193 | 0.00152102 | 651 |
| Reader 1805 | 0.00082725 | 0.00555277 | 0.00614892 | 0.00536316 | 1,498 |
| Regional 1806 | 0.00044548 | 0.00198075 | 0.00188920 | 0.00161927 | 601 |

Regional reaches the common field threshold slightly earlier than Dense, and both learn useful reconstruction much earlier than Latent or Reader. First crossings are descriptive observations, not stable-convergence guarantees or acceptance gates. Historical logged training-plus-validation time to this crossing is 12,664 s for Regional, 12,872 s for Dense, 30,434 s for Reader, and 42,256 s for Latent; Legacy has no comparable per-epoch timing. The HTML plots trajectories against both epochs and accumulated logged time.

Latent's exact-final validation field MSE is 28.8% above its best logged value. Its final train/validation temperature MSE is 0.000990/0.012042, versus Dense's 0.000555/0.001681 and Regional's 0.000703/0.002099. This is evidence of a substantial generalization gap and a noisy late endpoint in this configuration, not proof that latent attention as a class cannot work. Regional's last-100 validation-temperature median is 0.001617, slightly better than Dense's 0.001767, even though its final temperature sample and full-grid exact-final temperature error are worse.

| Model | Trainable parameters | Logged train time (h) | Logged validation time (h) | Sum (h) | Largest logged allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| Legacy 1401 | 2,473,510 | N/A | N/A | N/A | N/A |
| Latent 1801 | 4,542,489 | 22.314 | 2.465 | 24.779 | 23,039.5 |
| Dense 1804 | 4,395,409 | 25.498 | 2.801 | 28.299 | 27,210.3 |
| Reader 1805 | 3,778,064 | 26.669 | 3.042 | 29.711 | 29,285.3 |
| Regional 1806 | 4,395,409 | 27.921 | 3.151 | 31.072 | 23,544.8 |

Parameter counts come from the actual optimizer inventory and exclude frozen parameters. Dense and Regional have identical trainable counts: Regional's savings arise from shared response computation, not a smaller learned width. Latent has the most parameters yet is the fastest newer-family training run in these logs. Reader has fewer parameters but the highest observed training allocation and slower training than Dense.

These are **historical execution observations**, not controlled training-speed ratios. Runs used different periods, GPUs, continuations, and potentially shared devices; 1806's continuation shared GPU 1 with another process. Manifest start-to-end times include pauses (for example, 1801's 521,740 s versus 89,206 s of logged training/validation) and must not be treated as active compute. Legacy's manifest interval is 21,792 s, but its missing per-epoch timing prevents a like-for-like training-time decomposition. CSV fields named `*_memory_mb` divide by 1024² and are reported here as MiB. The matched inference measurements below provide the controlled execution comparison.

The learned routes remain trainable at maturity. Reader's epoch-5,000 group preparation/read gradient norms are 0.01063/0.001985, with update norms 0.01499/0.008660. Regional preparation/read gradients are 0.001469/0.012196 and updates 0.007532/0.007484; its direct-module, coarse, and local gradients are 0.05006, 0.02671, and 0.01313. These are logged batch observations, not contribution percentages. The intervention errors below, rather than nonzero gradients, establish usefulness.

## Mature port/field usefulness

Frozen-checkpoint removals use the four established anchors **0273, 0653, 0298, 0302**. P0 removes the main source read at provisional module ports and recomputes downstream physical response. P1 removes only the interface-feedback read while preserving normal P0. P2 removes only the final field read, leaving the physical loop intact. Positive error deltas mean removal worsens ground-truth error and therefore supports usefulness. The values below are unweighted four-anchor means of per-case relative-L2 changes, not the pooled 90-case scores.

| Model / removed phase | Field ΔL2 | Final outside T ΔL2 | Internal T ΔL2 | Surface T ΔL2 | Flux ΔL2 |
|---|---:|---:|---:|---:|---:|
| Latent P0 | +0.001670 | +0.038544 | +0.028863 | +0.029439 | +0.199115 |
| Latent P1 only | +0.009917 | +0.169983 | +0.115603 | +0.143100 | +0.098986 |
| Latent P2 | +0.340949 | ≈0 | ≈0 | ≈0 | ≈0 |
| Reader P0 | +0.007376 | +0.023016 | +0.025991 | +0.027808 | +0.199549 |
| Reader P1 only | +0.013330 | +0.239758 | +0.155557 | +0.178149 | +0.093305 |
| Reader P2 | +0.338342 | ≈0 | ≈0 | ≈0 | ≈0 |
| Regional P0 | +0.000254 | −0.001253 | +0.001152 | +0.001182 | +0.050490 |
| Regional P1 only | +0.010348 | +0.007093 | +0.009920 | +0.018228 | +0.064950 |
| Regional P2 | +0.376834 | ≈0 | ≈0 | ≈0 | ≈0 |

The same removals produce the following **normalized mean absolute prediction differences**, a different quantity from error changes. These stored tensor summaries include padding and all field-grid locations; the ground-truth errors above instead use active physical receivers and fluid locations:

| Model / phase | Field difference | Interface difference | Internal difference |
|---|---:|---:|---:|
| Latent P0 / P1 / P2 | 0.01157 / 0.01896 / 0.32876 | 0.12092 / 0.12206 / ≈0 | 0.02313 / 0.09399 / ≈0 |
| Reader P0 / P1 / P2 | 0.00787 / 0.02325 / 0.23490 | 0.12352 / 0.14371 / ≈0 | 0.02602 / 0.14417 / ≈0 |
| Regional P0 / P1 / P2 | 0.00124 / 0.00835 / 0.19611 | 0.04569 / 0.04838 / ≈0 | 0.00444 / 0.01957 / ≈0 |

Latent and Reader have strongly consequential paths despite their weak normal reconstruction. A larger ablation error or prediction difference does not make them better models. Regional P2 is useful on all four anchors; removing it raises mean near-field L2 by 0.258860 and far-field L2 by 0.539513. P1 is useful in the mean across field and physical metrics. P0 contributes substantially to flux but has small, mixed field/temperature effects.

Examples of material exceptions include Regional P0 removal improving field error on 0653, outside temperature on 0273/0653/0302, and internal temperature on 0653/0302. Regional P1 removal improves outside/internal temperature on 0298 and far-field error on 0653. Latent P0 removal improves global field error on 0273 and 0298; Latent P1 removal improves flux on 0298. Reader P0 removal improves outside/internal/surface temperatures on 0302. The complete anchor-level signs and near/far deltas are in `interventions/phase_summary.csv` and `phase_analysis.md`.

Existing mature Dense interventions have a different component scope and are kept separately labelled:

| Dense removal | Field ΔL2 | Internal T ΔL2 | Surface T ΔL2 | Flux ΔL2 |
|---|---:|---:|---:|---:|
| Environmental read throughout physical loop and final field | +0.290121 | +0.012499 | +0.025298 | +0.048910 |
| P2 environmental read only | +0.293881 | ≈0 | ≈0 | ≈0 |
| P2 direct module read only | +0.451462 | ≈0 | ≈0 | ≈0 |
| P2 coarse path only | +0.671207 | ≈0 | ≈0 | ≈0 |
| P2 local path only | +0.444353 | ≈0 | ≈0 | ≈0 |

For Regional, removing P2 direct-module and coarse reads raises field L2 by 0.460049 and 0.567707, respectively. Thus native regional sharing retains the common coarse path and direct module read as substantial contributors; it has not replaced all communication. Dense's throughout-loop removal slightly improves mean outside-temperature error (−0.001974) despite worsening the other listed physical errors.

There is no compatible Legacy phase artifact in this bounded comparison, so no fabricated P0/P1/P2 score is assigned to 1401. Dense's throughout-loop result is not an isolated P0 or P1 result. P2 physical changes are numerical zero, with occasional signs around 10⁻⁷; they are not evidence of physical feedback. This preserves the earlier correction of the broad decoder-hook attribution. Frozen removals demonstrate reliance of the trained model, not the performance of a retrained architecture without the component.

The earlier mature Dense **frozen projected-key/value coarsening** remains a separate negative result. Pooling 192 projected sources to 48 raised four-anchor field L2 by 0.006396 for P2-only approximation and 0.006152 for the recomputed full physical loop; the corresponding eight-case deltas were 0.009645 and 0.009338, with all eight field errors worsening. Full-loop flux also worsened by 0.015569 across eight cases. This previously executed experiment is reused from `regional_response/frozen_coarsening/run1804_epoch5000_frozen_192_to_48.json`. It does not contradict the trained native Regional result: averaging projected states after nonlinear preparation and learning nonlinear preparation after pooling are different operations.

## Geometric organization and derivative evidence

The families organize communication differently; there is no shared ground-truth topology score that makes their source counts directly comparable.

| Model | Geometric organization | Learned information | Principal limitation |
|---|---|---|---|
| Legacy | Historical fixed-projection organizer, six historical edges | Historical affinities and routed features | Affinity targets differ from newer-family read semantics |
| Latent | 16 main sources with fixed 4×4 geometric references | Module-conditioned latent states and receiver attention | Global availability provides no disconnected-source zero test |
| Dense | 192 fine environmental sites on real cases | Fine MM/ME/EM responses, environmental and direct-module reads | Repeated fine environmental reading is expensive |
| Reader | Compact overlapping geometric supports; 28–58 groups across cases | Group values, geometry-envelope read weights | Local support availability alone does not ensure accurate collective response |
| Regional | Fixed 2×2 fine-cell membership; 48 regional states on all 90 cases | Fine module-conditioned messages, pooled pre-update nonlinear responses, port/field attention | Fine joint messages and direct query–module work remain |

Reader's mean source-group count is 44.97, mean query degree 13.89 (maximum 16), mean non-null mass 0.6583, and mean reported covered-volume ratio 0.4752. These describe actual sparse organization. They are not a ranking of physical organization quality, and the covered-volume diagnostic is not a guarantee of reconstruction coverage.

Over the four anchors, Latent's head-averaged P2 attention has normalized entropy 0.785–0.807 over 16 sources and an effective count `exp(entropy)` of 8.89–9.44. Regional's corresponding values are 0.552–0.567 over 48 sources and 8.80–9.32. At active P0 ports, the effective counts are 8.87–9.82 for Latent and 7.65–8.93 for Regional. Similar effective counts coexist with very different reconstruction errors: attention concentration does not explain quality by itself. Padding is excluded at ports and the stored fluid mask is used at field receivers.

Mean main/coarse/local context-norm fractions over 90 cases are 0.454/0.449/0.097 for Latent, 0.407/0.510/0.083 for Dense, 0.411/0.493/0.096 for Reader, and 0.440/0.477/0.084 for Regional. Dense/Regional main context includes both direct-module and environmental/regional reads. These norm fractions are not causal percentages; the large coarse-path ablation effects confirm why assigning all reconstruction to the named main route would be misleading.

Legacy's historical module-affinity relative L2 averages 0.5766, but it measures a different target. Its environmental-affinity comparison has a **192-versus-288 shape mismatch on every case**. No cross-family topology score is inferred from these incompatible exports. The useful organization evidence is the combination of spatially indexed states, geometry-stratified reconstruction, and output-specific interventions. The HTML shows membership/support geometry separately from learned attention and state norms.

### Quadrature mass and local derivative checks

Duplicating the 192 environmental samples to 384 while splitting their mass preserves the same geometry and quadrature. Regional membership IDs are duplicated with the samples, retaining 48 regions. The maximum normalized field changes on anchor 0273 are:

| Legacy | Latent | Dense | Reader | Regional |
|---:|---:|---:|---:|---:|
| 8.58×10⁻⁶ | 7.39×10⁻⁶ | 4.77×10⁻⁶ | 6.20×10⁻⁶ | 8.82×10⁻⁶ |

Mean absolute field differences are approximately 0.8–1.1×10⁻⁷. The first diagnostic attempt failed because the old duplication helper discarded regional membership metadata; preserving that metadata fixed the diagnostic, with a focused regression test. Both the failed and successful execution logs remain local. This checks mass consistency at one geometry; duplicating samples adds no spatial information and is not validation at a finer physical resolution.

At exact 5,000, the Regional encoded-module probe perturbs one active encoded source in a unit all-ones direction with geometry and global input fixed. It freshly prepares the regional response and reads the **same P2-prepared states** at 32 field probes and eight actual port locations. It excludes direct-module, coarse, and local paths:

| Anchor | Mean regional-state JVP vector norm | Mean port-read JVP vector norm | Mean field-read JVP vector norm |
|---|---:|---:|---:|
| 0273 | 0.166072 | 0.159980 | 0.128270 |
| 0298 | 0.102802 | 0.117880 | 0.103054 |

Signed regional-state responses span −0.0866 to +0.1041 on 0273 and −0.0556 to +0.0433 on 0298. Thus a common regional representation carries module-dependent changes to both receiver types. Port locations sampled from P2 states are not relabelled as P0 computation. Every active module is globally available to every region, and every receiver can access every positive-mass regional state: no structurally disconnected subset exists here on which an exact zero derivative should be expected.

Centered finite differences at encoded-state steps 0.0005/0.001 have mean field-read relative vector discrepancies of 3.04%/1.37% on 0273 and 3.48%/1.77% on 0298. Regional-state discrepancies are at most 1.05%/0.52%. The smaller step is worse, consistent with finite-precision cancellation, without proving it is the only cause. These support dependence and approximate AD/FD agreement, not high-precision derivative validation.

The full physical model's coordinate probe moves the first active module in +x and measures mean normalized field temperature:

| Anchor | Autograd | FD at 0.005 | FD at 0.01 | Relative differences |
|---|---:|---:|---:|---:|
| 0273 | +0.01201299 | +0.01202077 | +0.01199618 | 0.065% / 0.140% |
| 0298 | −0.00880825 | −0.00880957 | −0.00880510 | 0.015% / 0.036% |

The negative response on 0298 is a signed directional response, not a failed derivative check. These are local model self-consistency measurements; they do not supply solver responses for displaced physical geometry. Reader's connected/disconnected influence and support-key-transition measurements in the recovery report remain **epoch-500 evidence** and were not silently promoted to 5,000. Regional's fixed environmental membership is a different mechanism from Reader's moving compact supports, so those transition tests do not establish a common topology ranking.

## Matched inference cost and scaling

All five exact-5,000 models were measured sequentially on **physical GPU 0, RTX 6000 Ada (49,140 MiB)**, using the same two real anchors, 8,192 queries, no detailed routing exports, outer query chunk 32,768, and inference receiver chunk 2,048. There were two warmups/five repetitions on anchors and one warmup/three repetitions on synthetic shapes. CUDA synchronization brackets wall timing. Training and scientific full-grid evaluation keep the configured 128-receiver chunk; the 2,048 override is an execution experiment only.

Full physical-forward timing includes preparation and field decoding. Memory is total live peak allocation, with peak reserved memory kept separate; it includes the required model/input state, excludes unrelated retained prepared states, and releases models/shapes between measurements. P05–P95 below describes the small repeat sample, not a confidence interval.

| Model | Anchor | Median ms [P05–P95] | Allocated MiB | Reserved MiB |
|---|---|---:|---:|---:|
| Legacy 1401 | 0273 | 27.103 [26.575–29.126] | 384.24 | 458 |
| Legacy 1401 | 0653 | 25.324 [24.854–25.976] | 384.24 | 458 |
| Latent 1801 | 0273 | 28.338 [27.176–34.098] | 112.09 | 138 |
| Latent 1801 | 0653 | 27.106 [26.918–28.431] | 112.24 | 138 |
| Dense 1804 | 0273 | 33.429 [33.191–34.275] | 460.01 | 618 |
| Dense 1804 | 0653 | 33.590 [33.348–34.291] | 460.01 | 618 |
| Reader 1805 | 0273 | 38.029 [37.393–38.944] | 115.68 | 218 |
| Reader 1805 | 0653 | 38.633 [37.741–39.449] | 138.29 | 254 |
| Regional 1806 | 0273 | 30.717 [30.280–31.394] | 149.00 | 214 |
| Regional 1806 | 0653 | 30.875 [30.331–31.571] | 149.00 | 214 |

Regional is **8.1% faster than default Dense** on the mean of the two anchor medians and uses **67.6% less peak allocated memory**. Legacy and Latent are faster than Regional on these real cases. Reader is slowest on the real anchors despite its low inference allocation.

The median physical-preparation-plus-one-query / prepared-field-decode times on anchor 0273 are 23.44/6.61 ms (Legacy), 22.32/6.79 (Latent), 22.88/13.05 (Dense), 29.18/13.42 (Reader), and 24.92/8.63 (Regional). Regional pays more preparation cost than Dense and saves repeated field-read cost. The same pattern holds on 0653. Independently measured phase medians need not add to the complete-forward median. Legacy has no comparable separate encoding/layout phase and is explicitly marked unsupported for that phase, rather than assigned zero work.

Large-shape measurements use synthetic inputs with the displayed `(M, E, Q)` sizes. They measure execution, not reconstruction against physical references:

| Model | Shape (modules / fine environment / queries) | Median ms [P05–P95] | Allocated MiB | Reserved MiB |
|---|---|---:|---:|---:|
| Legacy 1401 | 32 / 768 / 65,536 | 115.834 [115.820–132.329] | 3700.79 | 4320 |
| Legacy 1401 | 128 / 3072 / 262,144 | 1701.731 [1662.062–1703.718] | 14428.29 | 17520 |
| Latent 1801 | 32 / 768 / 65,536 | 82.166 [77.785–82.866] | 232.55 | 272 |
| Latent 1801 | 128 / 3072 / 262,144 | 224.421 [221.731–232.466] | 241.10 | 320 |
| Dense 1804 | 32 / 768 / 65,536 | 351.148 [348.911–358.238] | 1727.50 | 2042 |
| Dense 1804 | 128 / 3072 / 262,144 | 5324.988 [5320.778–5340.698] | 6708.15 | 9038 |
| Reader 1805 | 32 / 768 / 65,536 | 132.584 [131.085–135.279] | 231.65 | 294 |
| Reader 1805 | 128 / 3072 / 262,144 | 456.162 [446.691–473.361] | 242.83 | 374 |
| Regional 1806 | 32 / 768 / 65,536 | 164.052 [163.565–165.610] | 485.95 | 782 |
| Regional 1806 | 128 / 3072 / 262,144 | 2120.541 [2116.168–2123.892] | 2268.10 | 2896 |

Regional is **2.14× and 2.51× faster than Dense** on these two shapes. On the largest, its allocation is 2,268 versus Dense's 6,708 MiB. However, Regional remains **9.45× slower than Latent and 4.65× slower than Reader** there. Legacy is also faster than Regional on that shape but allocates 14,428 MiB (14.09 GiB). Every variant completed without OOM. The compressed families therefore occupy different accuracy/cost tradeoffs; Regional is not a universal efficiency winner.

Untimed hooks count actual selected operation rows in Dense and Regional. Each of the following entries is **Dense → Regional**:

| Work over complete physical forward | Real anchor (12 padded module slots, E=192) | M32 / E768 / Q65,536 | M128 / E3,072 / Q262,144 |
|---|---:|---:|---:|
| Fine MM message rows | 432 → 432 | 3,072 → 3,072 | 49,152 → 49,152 |
| Fine ME and EM message rows, each | 6,912 → 6,912 | 73,728 → 73,728 | 1,179,648 → 1,179,648 |
| Direct query–module rows | 116,736 → 116,736 | 2,228,256 → 2,228,256 | 35,651,712 → 35,651,712 |
| Environmental-update MLP rows | 576 → 144 | 2,304 → 576 | 9,216 → 2,304 |
| Environmental receiver-geometry rows | 1,867,776 → 466,944 | 53,478,144 → 13,369,536 | 855,641,088 → 213,910,272 |

These counters explain what native grouping saves and what remains. They count instrumented operation rows, not FLOPs or a count of uniquely active physical modules. Real anchors have three/five active modules inside 12 padded slots. Regional has 48/192/768 response states at the three environment sizes; its common coarse path still sees the fine environment. Other families lack the same operation hooks, so empty counter dictionaries are missing instrumentation, not zero computation.

### Execution-only variants and numerical limits

The opt-in summary export, larger inference chunk, and prepared-state reuse are kept separate from architecture/training changes. Dense's extra projection-cache variant takes 34.711/33.559 ms on the two anchors versus default 33.429/33.590 ms, and 348.568/5,386.243 ms on the synthetic shapes versus default 351.148/5,324.988 ms. It supplies no consistent additional speed gain in this measurement. Native Regional uses its existing prepared projections. No new cache or production arithmetic was introduced in this task.

At unchanged `rtol=2e-5, atol=2e-6`, comparisons of chunk 128 with 2,048 show small numerical failures: Latent field fails on both anchors, Reader field on 0273, and all four newer families' interface tensors on both anchors. Dense/Regional field checks pass; all models' port/internal checks pass; Legacy is exact. Interface maxima are 4.90×10⁻⁶–1.07×10⁻⁵ with relative L2 1.28×10⁻⁶–2.18×10⁻⁶. Independent same-chunk controls also show small repeat differences, so chunk partitioning is not the only possible source. All booleans and discrepancies remain in the timing JSON; no tolerance was changed and exact numerical equivalence is not claimed. Primary accuracy tables use the original configured chunk.

Sources: `timing/five_model_timing.json` and `timing/methods_numerics.md`. These new matched timings supersede neither the historically logged training costs nor the earlier epoch-500 timings; they answer a separate, controlled inference question.


## What changes relative to the earlier conclusions

1. **The baseline comparison is now mature.** Dense 1804 and Latent 1801 both completed 5,000 epochs and were evaluated under the same full-grid protocol. The old missing long-run Latent comparison is resolved. Legacy 1401 remains valuable as a fast historical reference and for its near-interface accuracy. Failed early Dense execution IDs 1800/1803 are not additional scientific baselines.
2. **Regional's early promise survives, but the ordering needs qualification.** Its small aggregate lead over Dense at 500/2,500 reverses at exact 5,000; its validation-selected checkpoint regains a small lead. The evidence supports comparable reconstruction with reduced environmental read cost, not uniform superiority or a final scientific failure. Temperature and pressure remain the main exact-endpoint gaps, and case 0283 remains a difficult exception.
3. **Regional's P0 physical role matures.** At epoch 500, P0-removal deltas for internal temperature, surface temperature, and flux were −0.006920, −0.008655, and −0.003960: retaining P0 worsened those means. At 5,000 the deltas are +0.001152, +0.001182, and +0.050490, supporting usefulness on those outputs. Its early all-anchor field benefit becomes mixed; P1/P2 remain useful with the stated exceptions.
4. **Reader recovery fixed a useful path but did not solve reconstruction.** Mature interventions now answer the old open question: group reads affect and generally improve field/physical outputs. Nevertheless, Reader remains well behind Legacy, Dense, and Regional on both exact and selected reconstruction. Its small inference memory and favorable large-shape speed are real, while its training-memory and real-anchor costs are unfavorable. Non-null mass, group count, or ablation magnitude cannot substitute for accuracy.
5. **Latent is a credible efficiency baseline with weak accuracy in this configuration.** It is fastest on both synthetic shapes and has the lowest allocation on the largest; its read has physical/field usefulness. Its late selected checkpoint improves over its exact 2,500 endpoint, so blanket late deterioration is too strong. Its large development error and difficult-case tail persist even after selection. No width sweep or claim about all latent-attention architectures follows from this one run.
6. **Compression savings are partial and workload-dependent.** Regional reduces environmental update/read work and inference memory relative to Dense while retaining fine joint communication. It is not the fastest scalable model among all five, and historical training logs do not establish a training-speed gain. Frozen post-projection coarsening remains the distinct, previously negative approximation experiment; native pre-update nonlinear regional training is not equivalent to it except in the singleton limit.

**Next research recommendation:** retain Dense and Latent as the two current interface-family baselines and Regional as the leading accuracy-oriented new design; preserve Reader as the compact-support comparison. Use a stated checkpoint-selection policy consistently in subsequent comparisons, report exact milestones alongside it, and seek seed replication plus the pending independent physical-response evidence before claiming a robust accuracy advantage. For the next bounded mechanism question, prioritize the difficult thermal cases and the remaining fine ME/EM/direct-module costs. Any further compression should preserve the demonstrated collective-response computation and be assessed against actual error/time/memory, rather than optimizing source count. This report launches no new training, architecture, sweep, or automatic continuation.

## Reproduction and artifacts

The working branch is `agent/honf-core-next`, starting from the completed regional-study source `c955c62`. This task changes only analysis/diagnostic tooling and reporting, not model arithmetic or training settings. Exact commands are recorded in [commands.txt](../../diagnostics/generated/interface_operator_study/five_model_epoch5000/commands.txt). All paths below are relative to `HONF_Proj/`.

- Main local study: `diagnostics/generated/interface_operator_study/five_model_epoch5000/`.
- New exact evaluations: `evaluation/tables/` and selected-anchor `evaluation/debug_npz/` under that study.
- Best-by-validation-field evaluations: `best_field_evaluation/tables/`, reduced separately in `reduction/best_selected*.csv` and `best_selected.json`.
- Fifteen-dataset reduction and source references: `reduction/comparison.json`, with pooled, paired, channel-contribution, stratum, and difficult-case CSVs.
- Previous exact endpoints: `diagnostics/generated/interface_operator_study/epoch5000_comparison/evaluation/` and `epoch2500_comparison/evaluation/`.
- Mature Dense/Reader phase evidence: `diagnostics/generated/interface_operator_study/regional_response/diagnosis/`.
- New phase evidence, geometry checks, histories, and timing: `interventions/`, `geometry/`, `history/`, and `timing/` in the study directory.
- Interactive offline figures: `figures/index.html` with the corresponding renderer `tools/diagnostics/render_honf_maturity_html.py`.

The HTML contains 15 quantitative/organization panels plus the interactive field-map panel. It embeds Plotly locally, supports all **100 anchor/model/channel combinations**, and uses common reference/prediction ranges and symmetric error ranges across models. Geometry membership/state norms and learned attention are selectable separately. Firefox browser verification found 16 populated plots, no missing panels, and successful representative changes of all anchor controls; the 100 map combinations also passed scale/data checks. Screenshots and the result are saved as `figures/preview_*.png` and `figures/preview_qa.json`. Standard SVG organization plotting avoids a WebGL requirement. Additional standalone history figures are `figures/history_validation_trajectories.html` and `figures/history_cost_convergence.html`.

To regenerate the reductions and HTML from the existing local artifacts, run from `HONF_Proj/`:

```bash
rtk proxy env PYTHONPATH=src:Case_ThermalChannel/src /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/diagnostics/analyze_honf_maturity.py
rtk proxy /home/wanglz/miniconda3/envs/ModularDT/bin/python tools/diagnostics/render_honf_maturity_html.py
```

Formal checkpoints remain in `Trained_Results/ThermalChannel/HONF_Forward_Runs/Run_<ID>_.../epoch_5000_model.pt`. Existing security and trusted loading remain intact. The **16 independent physical-reference requests remain pending**; no model self-consistency check or attention map is substituted for trustworthy solver outputs.

The new evaluator invocations process four exact datasets (1801/1806 at 2,500 and 5,000) and five selected checkpoints, or **810 case-model evaluations**. Previously evaluated matched baseline tables are reused in place. The model reduction, quadrature consistency check, role-scoped interventions, derivative probes, and repeated timings all completed with the numerical limitations documented above. The ordinary full CPU test suite passed **417 tests**, with three CUDA-only tests skipped, in 17.34 s. GPU 0 executed the actual model evaluations and diagnostics; no training or checkpoint copies were created.
