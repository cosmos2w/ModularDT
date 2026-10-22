"""Small fixed-layout allocation interface for the native affine heat law.

K must already be expressed in the requested physical observation units, with
columns in the supplied original module-ID order. This surrogate optimization
does not verify a proposed design against a physical solver.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeatAllocationProposal:
    module_ids: tuple
    heat: np.ndarray | None
    predicted_temperature: np.ndarray | None
    surrogate_feasible: bool
    status: str
    physical_verification: str = 'unverified'


def allocate_minimax_heat(kernel, module_ids, *, lower, upper, total_heat,
                          layout_pressure=None, pressure_limit=None):
    """Minimize the largest represented T=K h under explicit heat constraints.

    Pressure is optional but, when supplied, it needs its existing constraint.
    A pressure-infeasible layout receives no heat proposal because flow in this
    benchmark is heat independent. No feasibility tolerance is added to that
    physical constraint.
    """
    from scipy.optimize import linprog

    k=np.asarray(kernel,dtype=np.float64)
    ids=tuple(module_ids)
    if k.ndim!=2 or not k.shape[0] or k.shape[1]!=len(ids) or not ids or len(set(ids))!=len(ids):
        raise ValueError('K needs nonempty observation rows and unique original module-ID columns.')
    lo=np.broadcast_to(np.asarray(lower,dtype=np.float64),(len(ids),))
    hi=np.broadcast_to(np.asarray(upper,dtype=np.float64),(len(ids),))
    total=float(total_heat)
    if not np.isfinite(k).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.isfinite(total):
        raise ValueError('Kernel and user heat constraints must be finite.')
    if (lo<0).any() or (hi<lo).any() or total<lo.sum() or total>hi.sum():
        raise ValueError('The user bounds and total heat have no feasible allocation.')
    if (layout_pressure is None)!=(pressure_limit is None):
        raise ValueError('Supply layout pressure together with its pressure constraint.')
    if layout_pressure is not None:
        if not np.isfinite([layout_pressure,pressure_limit]).all():
            raise ValueError('Pressure and its constraint must be finite.')
        if layout_pressure>pressure_limit:
            return HeatAllocationProposal(ids,None,None,False,'layout_pressure_infeasible_heat_independent')
    # Epigraph constraints K h <= t remain valid for signed source responses.
    result=linprog(np.r_[np.zeros(len(ids)),1.],
        A_ub=np.c_[k,-np.ones(k.shape[0])],b_ub=np.zeros(k.shape[0]),
        A_eq=np.r_[np.ones(len(ids)),0.][None],b_eq=[total],
        bounds=[*zip(lo,hi),(None,None)],method='highs')
    if not result.success:
        return HeatAllocationProposal(ids,None,None,False,f'optimizer_failed: {result.message}')
    heat=result.x[:-1]
    return HeatAllocationProposal(ids,heat,k@heat,True,'surrogate_minimax_proposal')
