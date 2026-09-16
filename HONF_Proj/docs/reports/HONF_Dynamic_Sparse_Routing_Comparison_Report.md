# Dynamic sparse routing: common comparison

This comparison is being populated by Goals 1 and 2. Run 2000 and the separately authorized Run 2100 are training; no completed routing accuracy or acceleration claim is available yet. At the user's request, Run-2000 final analysis waits for a later instruction; see its [analysis handoff](HONF_Routing_Run2000_Epoch500_Handoff.md). Run 2100 has its own first-50-epoch interim handoff option. Goal 3 remains unexecuted.

| Candidate | Routing descriptors | Managed run status | Exact-500 assessment |
|---|---|---|---|
| Run 2000 | Physical module hubs | One fresh run on physical GPU 2, authorized through 500 | Pending; see [Run-2000 report](HONF_Routing_Run2000_Epoch500_Report.md) |
| Run 2100 | Three-step fixed-data mean-shift candidates | One fresh run on physical GPU 1, authorized through 500 | Pending; see [Run-2100 report](HONF_Routing_Run2100_Epoch500_Report.md) |
| Run 2200 | Finite dictionary | Outside this task; not launched by this task | Pending Goal 3 report |

All eventual rows must distinguish exact epoch 500 from saved-best-by-validation-field and state the selected checkpoint's actual epoch. They must use the same complete 90-case development population for accuracy, and the same GPU and stated chunking policy for timing. A later candidate's report must reuse prior results rather than silently retraining old models.

The established exact-500 pooled fluid relative L2 references are Legacy 1401: 0.1171479881; Dense 1804: 0.0987410316; Regional 1806: 0.0966520664. They are historical observations, not fixed acceptance thresholds. Best-through-5000 checkpoints are not substitutes for unavailable parent best-through-500 weights.

The research question is whether routing before fine pair evaluation preserves useful physical prediction while reducing measured end-to-end cost. Exact routing zeros, hub counts, branch-removal effects, and model AD/FD consistency alone do not establish physical sparsity or solver-validated influence. Formal ThermalChannel barrier benefit remains **Evidence Missing**.

Run-2100 has completed its healthy first-50 assessment and continues its single authorized run to 500; final analysis is deferred under the user's interim option. Exact-50 sampled validation field/temperature MSE is 0.196162/0.122869, with finite useful router and full-model updates. Training retains 99.73% QM and 99.995% QE work. All five bounded anchor field probes retain dense fine support despite candidate attraction; same-weight module-hub intervention pooled relative L2 is 0.294192 versus mean shift 0.295857. CPU candidate generation also costs more. The interim recommendation is unresolved: finish 500 before deciding on extension, with no demonstrated attraction-related support or accuracy gain yet. These probes are not the 90-case endpoint or a trained Run-2000 comparison.
