"""Continuous work controller: observed exact support and labelled smooth proxy."""
from __future__ import annotations
import torch
from torch import nn
from ..native_retrieval.budget_state import BudgetDualState
from .primitives import support_surrogate


class ContinuousBudgetSchedule:
    schema='continuous_joint_work_v1'
    @staticmethod
    def state_at_epoch(epoch,update_index=0):
        age=int(epoch)-3000
        if age<0:raise ValueError('Continuous trajectory begins at e3000.')
        if age==0:tm,te=1.,1.
        elif age<=200:
            p=(age-1)/199;tm,te=.98-.03*p,.98-.08*p
        elif age<=1000:
            p=(age-201)/799;tm,te=.95-.10*p,.90-.30*p
        else:tm,te=.85,.60
        delta=10**(-2-2*min(max(age-1,0)/999,1))
        return dict(schema=ContinuousBudgetSchedule.schema,absolute_epoch=int(epoch),sparse_age=age,
            target_M=tm,target_E=te,delta=delta,pressure=min(max(age-1,0)/199,1),
            constraints_enabled=age>0,deterministic=True,dense_router_step=False,
            stochastic_expectation_exact=False)


class ContinuousBudgetState(nn.Module):
    def __init__(self):
        super().__init__()
        self.duals=nn.ModuleDict({k:BudgetDualState(0.,rng_identity='round10_continuous_'+k) for k in ('M','E')})
        self.register_buffer('epoch',torch.tensor(3000,dtype=torch.long))
        self.register_buffer('macro_index',torch.tensor(0,dtype=torch.long))
        self._epoch_value=3000;self._macro_index_value=0
    def set_clock(self,epoch,update_index=0):
        self._epoch_value=int(epoch);self._macro_index_value=int(update_index)
        self.epoch.fill_(epoch);self.macro_index.fill_(update_index)
        return self.schedule()
    def schedule(self):
        return ContinuousBudgetSchedule.state_at_epoch(self._epoch_value,self._macro_index_value)
    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)
        self._epoch_value=int(self.epoch.detach().cpu())
        self._macro_index_value=int(self.macro_index.detach().cpu())
    def get_extra_state(self):return {'schema':ContinuousBudgetSchedule.schema}
    def set_extra_state(self,state):
        if state!={'schema':ContinuousBudgetSchedule.schema}:raise ValueError('Continuous work schema differs.')


def compact_work(gates,local_envelope,records,qmask,delta):
    """[B,2,8]: proxy, exact, possible, local floor, tiny-positive bins."""
    m=records.module_count
    rows=[]
    for sl,local in ((slice(0,m),local_envelope),(slice(m,None),torch.zeros_like(gates[...,m:]))):
        g=gates[...,sl]
        eligible=records.eligible[:,None,sl]&qmask[...,None]
        floor=(local>0)&eligible
        positive=(g.detach()>0)&eligible
        # Compulsory local rows count as one in both objective/controller.
        proxy=torch.where(floor,torch.ones_like(g),support_surrogate(g,delta))*eligible
        rows.append(torch.stack((proxy.sum((1,2)),positive.sum((1,2)),eligible.sum((1,2)),floor.sum((1,2)),
            *[((g.detach()<cut)&positive).sum((1,2)) for cut in (1e-4,1e-3,1e-2,.1)]),-1))
    return torch.stack(rows,1)


def microbatch_result(model):
    state=model.joint_budget.schedule();cases={}
    for keys,work in model._joint_entries:
        for key,row in zip(keys,work):cases[key]=cases.get(key,0)+row
    zero=next(model.router.parameters()).sum()*0
    constraints={};telemetry={}
    for index,kind in enumerate(('M','E')):
        residuals=[];controllers=[]
        for row in cases.values():
            proxy,actual,possible,floor=row[index,:4]
            removable=possible-floor
            target=floor+state['target_'+kind]*removable
            residuals.append(torch.where(removable>0,(proxy-target)/possible.clamp_min(1),proxy*0)*state['pressure'])
            controllers.append(torch.where(removable>0,(actual-target)/possible.clamp_min(1),actual*0).detach())
        constraints[kind]=dict(residual_numerator=sum(residuals,zero),denominator=len(residuals),
            controller_numerator=sum(controllers,zero.detach()))
        telemetry['continuous_'+kind+'_actual_rows']=sum((r[index,1].detach() for r in cases.values()),zero.detach())
        telemetry['continuous_'+kind+'_proxy_rows']=sum((r[index,0].detach() for r in cases.values()),zero.detach())
    model._joint_entries=[]
    return dict(constraints=constraints,telemetry=telemetry)
