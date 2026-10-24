"""Capability hook for exact typed macro constraints without retaining microbatch graphs."""
from __future__ import annotations
import math
from collections.abc import Mapping
import torch


class JointMacroObjective:
    """Accumulate d(mean residual)/dtheta; hinge only after the complete native macro."""
    def __init__(self, core, optimizer, epoch, update_index):
        self.core=core;self.epoch=epoch;self.update_index=update_index
        self.capable=callable(getattr(core,'joint_begin_update',None))
        self.enabled=self.capable
        self.residuals={};self.gradients={};self.denominators={};self.controller={};self.telemetry={}
        self.additive_values={};self.additive_gradients={};self.additive_denominators={}
        self.parameters=tuple(dict.fromkeys(p for g in optimizer.param_groups for p in g['params'] if p.requires_grad)) if self.capable else ()
        if self.enabled:
            needed=('joint_begin_microbatch','joint_microbatch_result','joint_constraint_duals','joint_after_update')
            if not all(callable(getattr(core,name,None)) for name in needed):
                raise TypeError('Joint core must implement the complete typed macro constraint capability.')
            kinds = getattr(core, 'joint_constraint_kinds', lambda: ('M','E'))()
            if not isinstance(kinds,(tuple,list)) or any(not isinstance(kind,str) or not kind for kind in kinds):
                raise ValueError('Joint constraint kinds must be nonempty names.')
            if len(set(kinds)) != len(kinds) or not {'M','E'}.issubset(kinds):
                raise ValueError('Joint constraint capability must declare unique M/E and any additional kinds.')
            self.constraint_kinds = set(kinds)
            self.state=core.joint_begin_update(epoch,update_index)
            if not isinstance(self.state,Mapping) or self.state.get('absolute_epoch')!=epoch:
                raise ValueError('Joint core failed to restore the absolute TRAIN clock.')
            self.enabled=bool(self.state.get('constraints_enabled',True))

    def begin_microbatch(self):
        if self.enabled:self.core.joint_begin_microbatch()

    def observe(self, expected_case_count=None):
        if not self.enabled:return
        result=self.core.joint_microbatch_result()
        constraints=result.get('constraints')
        if not isinstance(constraints,Mapping) or set(constraints)!=self.constraint_kinds:
            raise ValueError('Joint native macro residual kinds differ from the declared capability.')
        for typed,row in constraints.items():
            value=row.get('residual_numerator');count=row.get('denominator');controller=row.get('controller_numerator')
            if (not torch.is_tensor(value) or value.numel()!=1 or not value.requires_grad
                    or type(count) is not int or count<0 or not torch.isfinite(value.detach())):
                raise ValueError('Joint constraint requires a differentiable finite numerator and exact case count.')
            if expected_case_count is not None and count!=expected_case_count:
                raise ValueError('Joint typed denominator omitted/duplicated native cases or scheduled response requests.')
            gradients=torch.autograd.grad(value,self.parameters,retain_graph=True,allow_unused=True)
            current=self.gradients.setdefault(typed,{})
            for p,g in zip(self.parameters,gradients):
                if g is not None:
                    if p in current:current[p].add_(g.detach())
                    else:current[p]=g.detach().clone()
            self.residuals.setdefault(typed,[]).append(value.detach().reshape(()))
            self.denominators[typed]=self.denominators.get(typed,0)+count
            if not torch.is_tensor(controller) or controller.numel()!=1:
                raise ValueError('Joint controller numerator must be a detached scalar.')
            self.controller.setdefault(typed,[]).append(controller.detach().reshape(()))
        # Scene-level structural terms have their own exact case denominator.
        # Accumulate their VJPs before the native task backward frees this
        # microbatch, then apply the mean once over the complete macro. They do
        # not enter the work hinge or its replayed gradients.
        additive=result.get('additive_losses',{})
        if not isinstance(additive,Mapping):
            raise TypeError('Joint additive losses must be a named mapping.')
        for name,row in additive.items():
            if not isinstance(name,str) or not name or not isinstance(row,Mapping):
                raise ValueError('Joint additive losses require named numerator/count rows.')
            value=row.get('numerator');count=row.get('denominator')
            if (not torch.is_tensor(value) or value.numel()!=1 or not value.requires_grad
                    or type(count) is not int or count<1 or not torch.isfinite(value.detach())):
                raise ValueError('Joint additive loss requires a finite differentiable numerator and positive case count.')
            gradients=torch.autograd.grad(value,self.parameters,retain_graph=True,allow_unused=True)
            current=self.additive_gradients.setdefault(name,{})
            for parameter,gradient in zip(self.parameters,gradients):
                if gradient is not None:
                    if parameter in current:current[parameter].add_(gradient.detach())
                    else:current[parameter]=gradient.detach().clone()
            self.additive_values.setdefault(name,[]).append(value.detach().reshape(()))
            self.additive_denominators[name]=self.additive_denominators.get(name,0)+count
        for name,value in result.get('telemetry',{}).items():
            if torch.is_tensor(value):value=float(value.detach().cpu())
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Invalid joint telemetry.')
            if name=='joint_dense_router_step':name='joint_dense_router_microbatches'
            if name=='joint_exact_expectation':name='joint_exact_expectation_microbatches'
            self.telemetry[name]=self.telemetry.get(name,0.)+value

    def apply(self):
        self.losses={};self.work=dict(self.telemetry);self.controller_residuals={}
        if not self.enabled:return
        duals=self.core.joint_constraint_duals()
        if set(duals)!=self.constraint_kinds:raise ValueError('Joint controller requires serialized typed dual states.')
        for typed,values in self.residuals.items():
            if not self.denominators[typed]:
                self.controller_residuals[typed]=torch.stack(self.controller[typed]).sum()*0
                continue
            denominator=self.denominators[typed];residual=torch.stack(values).sum()/denominator;dual=duals[typed]
            penalty=dual.penalty_residual(residual)
            coefficient=(dual.dual_lambda.to(residual)*(residual>0).to(residual.dtype)+dual.zeta.to(residual)*residual.relu()).detach()
            for parameter,gradient in self.gradients[typed].items():
                contribution=gradient*(coefficient/denominator)
                if parameter.grad is None:parameter.grad=contribution
                else:parameter.grad.add_(contribution)
            self.losses[f'joint_work_penalty_{typed}']=float(penalty.detach().cpu())
            self.work[f'joint_constraint_residual_{typed}']=float(residual.cpu())
            self.work[f'joint_case_requests_{typed}']=denominator
            self.controller_residuals[typed]=torch.stack(self.controller[typed]).sum()/denominator
        additive_transform = getattr(self.core, 'joint_apply_additive_gradients', None)
        if callable(additive_transform):
            # Optional research capability: receives the completed native task
            # and work gradient before adding bounded structural gradients.
            additive_transform(self)
            self.gradients.clear()
            self.additive_gradients.clear()
            return
        for name,values in self.additive_values.items():
            denominator=self.additive_denominators[name]
            for parameter,gradient in self.additive_gradients[name].items():
                contribution=gradient/denominator
                if parameter.grad is None:parameter.grad=contribution
                else:parameter.grad.add_(contribution)
            self.losses[name]=float((torch.stack(values).sum()/denominator).cpu())
            self.work[name+'_case_requests']=denominator
        self.gradients.clear()
        self.additive_gradients.clear()

    def after_update(self):
        if self.capable:self.core.joint_after_update(self.epoch,self.update_index,self.controller_residuals)
        return self.losses,self.work
