# Thermal finite-response research task

The explicit `thermal_finite_response_v1` task compares the inherited joint predictor with absolute edited-state anchors (A), additional finite-change supervision (B), and B with experimental rank-8 contextual pair content (C). It preserves the original primary membership, TRAIN600 transforms, native observations, local asset and prescribed physical inputs. Original-model loading and default core configuration remain unchanged.

The maintained commands are `tools/thermal_response_corpus.py`, `tools/relation_anchored_train.py` and `tools/thermal_finite_response_evaluate.py`. Physical generation uses the existing CPU reference solver. Store raw states, attempt ledgers, scale calculations, training histories and evaluation arrays outside Git in the declared DATA2 round directory. Plans and scientific reports remain local.

The literal common source is P64 checkpoint epoch 100 inherited from R14 epoch 5000, so its absolute learned age is 5100. A new epoch e advances the contextual clock to 5100+e. Qualification inherits these same weights and transforms; it never supplies initialization or optimizer state to formal training.

## Objective and data roles

One baseline/edit pair accompanies each accumulated primary batch, on its first microbatch. Both endpoint state objectives are averaged and weighted by 0.25. The paired pack has denominator one, independent of primary case count and accumulation. B/C add the mean full/restricted finite loss: 0.8 times normalized inherited eight-channel field losses, plus 0.1 each for native J and pressure-drop differences. Native functionals retain gradients and use FP64 reduction; inverse evaluators that freeze model parameters are unsuitable for fitting.

The prospective recipe assigns FIT and response-holdout roles before physical generation. Only FIT labels enter paired training, scale estimation and initial gradient balancing. Original TRAIN-origin holdout baselines remain primary training examples. Their edited labels are held out; this does not make them unseen layouts or independent TEST. Canonical exposed VAL89 remains development evaluation under its historical protocol.

Fixed action-class scales are equal-family FIT RMS values floored by the larger of 1% of the inherited absolute scale and measured TRAIN observation/convergence sensitivity. Practical response resolution is a separate TRAIN-derived quantity. The initial balance command measures twelve actual accumulated TRAIN600 objectives and all inherited trainable parameter gradients, without an optimizer update. It sets `lambda_delta = min(1, 0.25 * median(norm(g_absolute)/norm(g_delta)))`. Keep this coefficient and the scales fixed across B/C and every continuation. Saved provider identity includes the actual scale values; changing the scale file invalidates an ordinary same-run resume.

## Execution

Use explicit physical GPU UUIDs through `CUDA_VISIBLE_DEVICES`, after checking current ownership. Use one fitting process per authorized device, TF32 disabled and a writable device-specific KeOps directory. Effective batch 48, physical microbatch 24 and 1024 primary fluid queries preserve the common primary stream. Pair initialization uses isolated RNG so C does not shift shared case/query draws.

Declare the formal optimizer horizon as 500 from the start and use the ordinary engine stop/resume interface for shared reviews; do not amend the optimizer schedule at epoch 100. For example:

```bash
python tools/relation_anchored_train.py calibrate --recipe B_balance.json --output DATA2/balance --device cuda:0
python tools/relation_anchored_train.py fit --recipe A.json --output DATA2/A --stop-after 100 --device cuda:0
python tools/relation_anchored_train.py fit --recipe A.json --output DATA2/A --resume latest --stop-after 500 --device cuda:0
python tools/relation_anchored_train.py native --recipe A.json --checkpoint DATA2/A/checkpoints/epoch_0100_model.pt --output DATA2/A_review100 --device cuda:0
```

Actual recipe, corpus, scale and checkpoint paths are required; the example placeholders do not define an experiment. Preserve optimizer/RNG/provider state, complete histories, latest/best states and declared milestones across resume. Initial and periodic loss curves include total TRAIN and validation panels, with validation pending before its scheduled result. Qualification and formal arms require sparse read-only monitoring under the applicable round plan.

## Interpretation

Compare A/B/C at shared completed horizons. Report direct predictor fields, frozen-local/port readouts, finite physical responses, actual applied organization and complete work/cost separately. A changing organization or successful autodifferentiation is not physical response verification. Positive pair-content gathers do not establish end-to-end acceleration; dense score and geometry preparation remain.

Bind any implementation claim to its source commit. Experimental C has a separate self-contained schema and preserves both ordinary weighted reduction backends. Frozen-control and all-open contextual interventions are explicitly unsupported for C. No adoption or default-core change follows from implementation alone.
