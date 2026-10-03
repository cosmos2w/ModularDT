"""Evaluation-only counts of actual five-route fine MLP forward inputs."""

from contextlib import AbstractContextManager
from math import prod

import torch
from torch import nn

FINE_KERNEL_MODULES = {
    "MM": "mm_message",
    "ME": "me_message",
    "EM": "em_message",
    "QM": "query_module_message",
    "QE": "env_geometry_bias",
}


class FineKernelWork(AbstractContextManager):
    """Count successful module forwards without retaining values or graphs.

    A row is one leading-axis cell of the tensor actually entering a fine
    message/geometry MLP. Masked/padded cells within an executed rectangle
    count. This does not measure eligible or unique source pairs, attention,
    organizer/coarse/local physics, backward recomputation or hardware work.
    Install only around a separate evaluation work pass, outside latency and
    backward scopes. Existing module hooks are preserved on exit, including
    exceptional exit.
    """

    def __init__(self, backend: nn.Module):
        self.modules = {route: getattr(backend, name, None) for route, name in FINE_KERNEL_MODULES.items()}
        if not all(isinstance(module, nn.Module) for module in self.modules.values()):
            raise TypeError("Fine work requires all five Dense/Typed physical MLP modules")
        if len({id(module) for module in self.modules.values()}) != len(self.modules):
            raise ValueError("Fine physical routes must use distinct MLP modules")
        self.records: dict[str, dict[str, int]] = {}
        self._handles = []
        self._active = False

    def _hook(self, route):
        def record(_module, arguments, keywords, _output):
            values = arguments[0] if arguments else keywords.get("x")
            if not torch.is_tensor(values) or values.ndim < 1:
                raise TypeError("A fine physical MLP input must be a tensor with a feature axis")
            entry = self.records[route]
            entry["padded_input_rows"] += prod(values.shape[:-1])
            entry["calls"] += 1

        return record

    def __enter__(self):
        if self._active:
            raise RuntimeError("A fine work recorder cannot enter twice concurrently")
        self.records = {route: {"padded_input_rows": 0, "calls": 0} for route in self.modules}
        self._active = True
        try:
            for route, module in self.modules.items():
                self._handles.append(module.register_forward_hook(self._hook(route), with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exception):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        return False

    def snapshot(self):
        """Return plain independent counters suitable for saved evidence."""
        return {
            "scope": "Successful five-route physical MLP forwards; actual leading-axis input cells including rectangle padding",
            "excluded": "Eligible/unique pairs, attention cells, policy/coarse/local physics, backward and hardware kernel counts",
            "routes": {route: dict(value) for route, value in self.records.items()},
        }


__all__ = ["FINE_KERNEL_MODULES", "FineKernelWork"]
