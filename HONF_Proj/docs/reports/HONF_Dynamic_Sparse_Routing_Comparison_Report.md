# Dynamic sparse routing: common comparison

This comparison is being populated by Goal 1. Run 2000 is still training; no completed routing accuracy or acceleration claim is available yet. At the user's request, final analysis waits for a later instruction after the tools are prepared; see the [analysis handoff](HONF_Routing_Run2000_Epoch500_Handoff.md). Goals 2 and 3 are outside the current authorization and remain pending.

| Candidate | Routing descriptors | Managed run status | Exact-500 assessment |
|---|---|---|---|
| Run 2000 | Physical module hubs | One fresh run on physical GPU 2, authorized through 500 | Pending; see [Run-2000 report](HONF_Routing_Run2000_Epoch500_Report.md) |
| Run 2100 | Mean-shift candidates | Outside this task; concurrent work not assessed here | Pending Goal 2 report |
| Run 2200 | Finite dictionary | Outside this task; not launched by this task | Pending Goal 3 report |

All eventual rows must distinguish exact epoch 500 from saved-best-by-validation-field and state the selected checkpoint's actual epoch. They must use the same complete 90-case development population for accuracy, and the same GPU and stated chunking policy for timing. A later candidate's report must reuse prior results rather than silently retraining old models.

The established exact-500 pooled fluid relative L2 references are Legacy 1401: 0.1171479881; Dense 1804: 0.0987410316; Regional 1806: 0.0966520664. They are historical observations, not fixed acceptance thresholds. Best-through-5000 checkpoints are not substitutes for unavailable parent best-through-500 weights.

The research question is whether routing before fine pair evaluation preserves useful physical prediction while reducing measured end-to-end cost. Exact routing zeros, hub counts, branch-removal effects, and model AD/FD consistency alone do not establish physical sparsity or solver-validated influence. Formal ThermalChannel barrier benefit remains **Evidence Missing**.
