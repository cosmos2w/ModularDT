# HONF Interaction-Preserving Joint Core Recovery: Segmented Stage Review

**Status:** the six fresh segmented P/P-G/P-H siblings completed matched review checkpoints at epochs 500 and 1,000 and were authorized to continue unchanged through absolute epoch 2,500. The e1,000 results show consistent improvement, especially for learned P-H Wind wakes and Thermal flow/temperature tails, but do not establish mature recovery against stronger historical physical references; final native fields, actual graph interventions, GPU inference costs, and the recovery/organizer decision remain pending.

The implementation restores a nonlinear source-conditioned MM/ME/EM path for fresh P before adding one shallow collective residual. P-G uses fixed geometric memberships, while P-H learns residual membership scores from the same start; Thermal retains one jointly trained flow/temperature model and Wind retains one shared nonlinear core. Full-data training, inverse training, and new solver runs have not been launched.

## Predictor

All six siblings improved their primary fixed-DEV trajectories from e500 to e1000. Thermal measurements use the same fixed25 DEV22 cases and Q1024 queries in native dataset units; flow RMSEs below are mean u/v/p/omega in that order, temperature pairs are mean/p90, and material-peak values are sampled peak RMSE mean/p90.

| Arm | Field score | Flow mean RMSE u/v/p/omega | Fluid temperature mean/p90 | Surface temperature mean/p90 | Material temperature mean/p90 | Sampled material-peak RMSE mean/p90 | q-proxy RMSE mean/p90 |
|---|---:|---|---|---|---|---|---|
| P | 0.04550 | 0.05385 / 0.01113 / 0.02281 / 0.48290 | 0.77456 / 1.03810 | 0.73872 / 1.05385 | 0.69665 / 0.94115 | 0.78100 / 1.16267 | 2.64141 / 3.40033 |
| P-G | 0.03263 | 0.04559 / 0.00867 / 0.02086 / 0.40649 | 0.68885 / 1.02122 | 0.67172 / 1.01288 | 0.61896 / 0.97334 | 0.59947 / 0.91386 | 2.37494 / 2.89360 |
| P-H | 0.02565 | 0.04769 / 0.00754 / 0.02217 / 0.33954 | 0.66220 / 0.89851 | 0.64424 / 0.84828 | 0.62937 / 0.80145 | 0.67021 / 0.89041 | 2.43802 / 3.17537 |

P-H leads on flow-v and omega means and temperature p90s; P-G has lower flow-u and pressure means and the lowest sampled material-peak and q-proxy means. The Thermal gate found both organized arms passed the twelve priority field case-mean/p90 checks against same-age P, but no arm has yet been selected on mature native fields or tails.

Wind measurements use the same fixed DEV24 panel, corrected TRAIN-fitted component scales, and m/s. Vector and component p95 values are quantiles across per-row RMSEs, not within-row pointwise errors.

| Arm | Near-turbine vector mean / row-p95 / worst | Near Ux/Uy/Uz component means | Downstream vector mean / row-p95 / worst | Downstream Ux/Uy/Uz component means | Downstream Ux/Uy/Uz component row-p95 |
|---|---|---|---|---|---|
| P | 0.48112 / 0.69075 / 0.71622 | 0.46321 / 0.09334 / 0.08868 | 0.30356 / 0.43236 / 0.44955 | 0.29472 / 0.05302 / 0.04857 | 0.42359 / 0.06502 / 0.05873 |
| P-G | 0.47188 / 0.59643 / 0.61143 | 0.45630 / 0.09670 / 0.06881 | 0.29511 / 0.36340 / 0.38250 | 0.28798 / 0.05228 / 0.03539 | 0.35722 / 0.08228 / 0.05049 |
| P-H | 0.45550 / 0.54072 / 0.54372 | 0.44313 / 0.08955 / 0.05393 | 0.29697 / 0.34819 / 0.37404 | 0.29095 / 0.05001 / 0.03101 | 0.34211 / 0.06746 / 0.04141 |

P-H passed all twelve near/downstream component mean and row-p95 guards against P at e1000; P-G missed the downstream Uy row-p95 guard by 26.6%. P-H is promising, but its mean wake-vector errors of 0.45550 m/s near turbines and 0.29697 m/s downstream remain above the stronger segmented W2302 J-geometry reference of 0.32086/0.21662 m/s, so sibling-relative improvement does not yet qualify as recovery.

The historical Thermal reference is a D-sep flow plus R-direct temperature composite, not a self-contained jointly trained checkpoint. Its full-native and new weighted-Q1024 values are not directly interchangeable: exact saved-array reindexing covers only four DEV cases, the current point weights range from 1 to 3, and for DEV0277 the archived omega RMSE shifts from 0.10223 on the full uniform fluid grid to 0.26360 on unweighted exact-Q1024 points and 0.33319 under the current weighted reducer. The exact-subset receipt contains e1,000 P and P-G rows but no P-H rows, so it cannot yet support a full three-arm historical comparison.

## Organizer

Fixed-weight TRAIN probes confirmed that all fitted MM/ME/EM parameters receive gradients and that the registered collective value, update, output, learned membership, and receiver-access modules are active where defined. Those probes establish that the computation is exercised, not that the learned graph improves physical fidelity. The matched same-weight DEV graph interventions, actual receiver-to-donor content chains, bit-exact restoration receipts, and accelerator cost comparisons are not yet part of the qualified final result; no organizer is promoted and no structural revision has been used.

## Inverse and protected boundaries

No inverse training, inverse design campaign, or new solver run was performed. Solver usage remains 326/326 with no allowance remaining, and WindTEST remains locked. The inherited 647 bindings and newer formal artifacts were inventoried read-only and preserved; detailed path hashes, locks, calibration lineage, recipe differences, and resource receipts are in the [technical appendix](HONF_Interaction_Preserving_Recovery_20261010_Appendix.md).

The root's e1,000 gate continued all six segmented siblings unchanged to epoch 2,500. It did not trigger the user's conditional full-data 1,000-epoch comparison against Thermal1404/1804/3906 and Wind2102/2103/2204: the user's authorization remains conditional on credible matched segmented-2,500 physical recovery, a useful preferred recipe, and an explicit recovery/organizer gate, and has not become a standing launch instruction. No full-data run has been launched; any extension beyond 1,000 remains the user's separate decision.

The e500/e1000 stage boards, exact recipes, calibration provenance, historical query-scope audit, native-measurement workflow, and protection/resource receipts are recorded in the [technical appendix](HONF_Interaction_Preserving_Recovery_20261010_Appendix.md). Selected native field, residual, graph-chain, response, and cost figures will be added only after the mature checkpoint review; the CPU-only schema smokes are not result figures.
