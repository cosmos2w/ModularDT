"""Sampled task/work policy gradients for the interaction-group frontier.

The previous macro's action-independent baselines are updated only after the
current Adam boundary. Conditional native gradients remain engine-owned.
"""

import copy

import torch

from honf_runtime.thermal_soft_organization import gradient_norm


class InteractionGroupObjectiveRuntime:
    def __init__(self, model, provider=None):
        self.model = model
        self.core = model.core
        self.provider = provider
        self.state = {
            "schema": "interaction_group_objective_v1",
            "task_baseline": 0.0,
            "work_baseline": 0.0,
            "work_coefficient": 0.0,
            "work_qualified": False,
            "updates": [],
        }
        self.pending = []

    def collect(self, predictions, per_case_native_losses, epoch, case_ids=None):
        if not self.core.adaptive_groups:
            return
        p = predictions.native_prepared.organization
        logps = torch.stack(
            [sum(layer[b].frontier.log_probability for layer in p.layers) for b in range(len(p.scene.module_features))]
        )
        costs = []
        prices = self.provider.recipe["objective"].get("work_prices", {})
        for b in range(len(logps)):
            cost = 0.0
            for layer in p.layers:
                for name, count in layer[b].work_counts.items():
                    cost += float(prices.get(name, 0.0)) * count
            query_rows = predictions.native_prepared.query_batch.num_queries
            slots = min(8, p.final_groups[b].frontier.K)
            query_counts = {
                "group_scores": query_rows * slots * self.core.num_heads,
                "group_value_reads": query_rows * slots,
                "query_module_scores": query_rows * p.scene.module_mask.shape[1],
                "spatial_cell_visits": query_rows * 9,
            }
            cost += sum(float(prices.get(name, 0.0)) * count for name, count in query_counts.items())
            costs.append(cost)
        work = logps.new_tensor(costs)
        task = ((per_case_native_losses.detach() - self.state["task_baseline"]) * logps).sum()
        work_term = ((work - self.state["work_baseline"]) * logps).sum()
        named = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        task_g = torch.autograd.grad(task, tuple(named.values()), retain_graph=True, allow_unused=True)
        work_g = torch.autograd.grad(work_term, tuple(named.values()), retain_graph=True, allow_unused=True)
        self.pending.append(
            (
                len(logps),
                {n: g.detach() for n, g in zip(named, task_g) if g is not None},
                {n: g.detach() for n, g in zip(named, work_g) if g is not None},
                float(per_case_native_losses.detach().sum()),
                float(work.sum()),
                sum(len(g.frontier.reached) for layer in p.layers for g in layer),
            )
        )

    def before_optimizer_step(self, model, optimizer, losses, work, *, epoch, update_index):
        if not self.pending:
            return
        named = dict(model.named_parameters())
        total = sum(row[0] for row in self.pending)
        task_g = {}
        work_g = {}
        for _, tg, wg, *_ in self.pending:
            for target, source in ((task_g, tg), (work_g, wg)):
                for n, g in source.items():
                    target[n] = target.get(n, 0) + g / total
        native = {n: p.grad.detach() for n, p in named.items() if p.grad is not None}
        nt, nw, nn = gradient_norm(task_g), gradient_norm(work_g), gradient_norm(native)
        # Work price is fixed by the no-step TRAIN qualification. No dividing
        # a null/noisy norm here or retuning at the epoch100 review.
        coefficient = float(self.provider.recipe["objective"].get("work_policy_coefficient", 0.0))
        _, ramp = self.core.layers[0].frontier.schedule(epoch)
        scale = coefficient * ramp
        cap = min(1.0, 0.1 * nn / (scale * nw)) if scale * nw > 0 else 1.0
        for source, factor in ((task_g, 1.0), (work_g, scale * cap)):
            for n, g in source.items():
                if named[n].grad is None:
                    named[n].grad = g.clone() * factor
                else:
                    named[n].grad.add_(g, alpha=factor)
        slices = {}
        for label, select in {
            "organizer": lambda n: ".frontier." in n,
            "shared_encoder": lambda n: n.startswith(
                ("core.module_encoder.", "core.environment_encoder.", "core.operating_encoder.")
            ),
            "exchange": lambda n: n.startswith("core.layers.") and ".frontier." not in n,
        }.items():
            names = [n for n in named if select(n)]
            tg = {n: task_g[n] for n in names if n in task_g}
            wg = {n: work_g[n] for n in names if n in work_g}
            a, b = gradient_norm(tg), gradient_norm(wg)
            dot = sum(float((tg[n] * wg[n]).sum()) for n in tg if n in wg)
            slices[label] = {
                "task_norm": a,
                "work_norm": b,
                "cosine": dot / (a * b) if a * b else None,
                "missing_task": sum(n not in tg for n in names),
                "missing_work": sum(n not in wg for n in names),
            }
        task_mean = sum(row[3] for row in self.pending) / total
        work_mean = sum(row[4] for row in self.pending) / total
        self.state["task_baseline"] = 0.9 * self.state["task_baseline"] + 0.1 * task_mean
        self.state["work_baseline"] = 0.9 * self.state["work_baseline"] + 0.1 * work_mean
        self.state["work_coefficient"] = coefficient
        self.state["work_qualified"] = coefficient > 0
        self.state["updates"].append(
            {
                "epoch": epoch,
                "update": update_index,
                "native_policy_norm": nt,
                "slices": slices,
                "raw_work_policy_norm": nw,
                "native_full_norm": nn,
                "work_ramp": ramp,
                "work_cap": cap,
                "cases": total,
                "reached_nodes": sum(row[5] for row in self.pending),
                "task_mean": task_mean,
                "work_mean": work_mean,
            }
        )
        work["frontier_work_vjp_cases"] = total
        self.pending = []

    def state_dict(self):
        return {"state": copy.deepcopy(self.state), "pending": copy.deepcopy(self.pending)}

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state["state"])
        self.pending = copy.deepcopy(state.get("pending", []))
