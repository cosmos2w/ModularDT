"""A scheduled response is counted once regardless of ordinary case packing."""
from types import SimpleNamespace

from channelthermal.training.content_hypergraph_task import ThermalContentHypergraphTask


def test_response_first_microbatch_does_not_absorb_later_ordinary_cases():
    batches=[SimpleNamespace(targets=SimpleNamespace(case_ids=('a',),response=object())),
             SimpleNamespace(targets=SimpleNamespace(case_ids=('b','c'),response=None))]
    denominators=ThermalContentHypergraphTask.loss_denominators(None,batches,'joint','A2-D')
    assert denominators['response']==1
    assert all(value==3 for name,value in denominators.items() if name!='response')
