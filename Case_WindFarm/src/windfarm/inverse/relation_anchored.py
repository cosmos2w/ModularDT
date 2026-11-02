"""Differentiable, fixed-weight native Wind design evaluation.

Prepared objects are valid only for their original design and model state.
Multiple receiver blocks may share one preparation and one backward pass.
"""
from __future__ import annotations

import torch


class WindDesignEvaluator:
    def __init__(self, model, case, *, module_ids=None):
        self.model, self.case, self.module_ids = model, case, module_ids
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def prepare(self, centers):
        scene = self.model.make_scene(self.case, centers, module_ids=self.module_ids)
        return scene, self.model.core.prepare(scene)

    def evaluate_prepared(self, prepared, coordinates, *, query_ids=None,
                          read_mode="full", return_trace=False):
        scene, organization = prepared
        query = self.model.make_query(scene, coordinates, query_ids=query_ids)
        output = self.model.core.decode(organization, query,
            intervention={"read_mode": read_mode}, return_trace=return_trace)
        velocity, _, _ = self.model.transforms.physical_velocity(output.joint_fields, query.coordinates)
        return velocity, output

    def evaluate_design(self, centers, coordinates, *, query_ids=None,
                        read_mode="full", return_trace=False):
        prepared = self.prepare(centers)
        velocity, output = self.evaluate_prepared(prepared, coordinates,
            query_ids=query_ids, read_mode=read_mode, return_trace=return_trace)
        return velocity, output, prepared
