"""Independent dense and packed nonlinear values; no omitted wide rows in packed."""
from __future__ import annotations
import torch
from ..joint_native.readers import raw_scores, _dense_reduce, _flat_messages, TypedRead
from .primitives import guarded_weights


def _empty_selected_values(reader, qstates, qcoords, coordinates, lengths,
                           scale, values, alpha, gate):
    """Preserve dense zero-gradient edges without evaluating omitted values.

    AdamW treats a missing gradient differently from an explicit zero. Empty
    views retain the latter's autograd connections without reading padding or
    constructing Fourier features, messages, or full-width activations.
    """
    inputs = (qstates, qcoords, coordinates, lengths, scale, values, alpha)
    zero = qstates.new_zeros(())
    for value in inputs:
        zero = zero + value.unsqueeze(0)[:0].sum()
    for name in ('query_value', 'relative_value', 'source_scale_projection', 'message_output'):
        layer = getattr(reader, name)
        if layer is not None:
            for parameter in layer.parameters():
                zero = zero + parameter.unsqueeze(0)[:0].sum()
    batch, queries = qstates.shape[:2]
    context = zero.expand(batch, queries, reader.heads, reader.head_width)
    scalar_zero = zero + gate.unsqueeze(0)[:0].sum()
    if reader.thermal_output is not None:
        for parameter in reader.thermal_output.parameters():
            scalar_zero = scalar_zero + parameter.unsqueeze(0)[:0].sum()
    return context, scalar_zero.expand(0), zero.expand(0, reader.width)


def _selected_source_order_reduce(reader,qstates,qcoords,coordinates,lengths,scale,values,indices,alpha,gate,
                                  return_messages=False):
    """Compute positive-g NN rows, then use the parent's source-axis sum order.

    The weighted zero scratch is bounded by this query tile. It never escapes
    the checkpoint and contains no evaluated message for an omitted donor.
    """
    batch,queries,count=alpha.shape[:3]
    if not indices.shape[0]:
        return _empty_selected_values(reader, qstates, qcoords, coordinates, lengths,
                                      scale, values, alpha, gate)
    messages,scalar=_flat_messages(reader,qstates,qcoords,coordinates,lengths,scale,values,indices)
    b,q,n=indices.unbind(-1)
    weighted=messages.reshape(-1,reader.heads,reader.head_width)*alpha[b,q,n,:,None]
    scratch=messages.new_zeros(batch,queries,count,reader.heads,reader.head_width)
    scratch.index_put_((b,q,n),weighted)
    context=scratch.sum(2)
    if reader.thermal_output is not None:scalar=scalar*gate[b,q,n]
    return context,scalar,messages if return_messages else messages.new_empty((0,))


def read_continuous(reader,query,qstates,*,coordinates,lengths,valid,measure,frame_scale,
                    keys,values,gate,scores=None,backend='packed',return_messages=False,frozen_coefficients=None,
                    reference=None):
    batch,queries,width=qstates.shape
    count=coordinates.shape[1]
    qmask=query.query_mask if query.query_mask is not None else torch.ones((batch,queries),device=qstates.device,dtype=torch.bool)
    gate=gate*valid[:,None]*qmask[...,None]
    if frozen_coefficients is None and scores is None:
        scores=raw_scores(reader,query,qstates,coordinates,lengths,frame_scale,keys)
    if frozen_coefficients is None:
        alpha,mass=guarded_weights(scores,gate,measure,valid,qmask,reference=reference)
    else:
        alpha=frozen_coefficients
        if alpha.shape!=(batch,queries,count,reader.heads):raise ValueError('Frozen read coefficients have incorrect native axes.')
    indices=(gate.detach()>0).nonzero(as_tuple=False)
    args=(reader,qstates,query.coordinates.to(qstates),coordinates.to(qstates),lengths.to(qstates),frame_scale,values)
    if backend=='dense':
        context,scalar,messages=_dense_reduce(*args,alpha,gate,return_messages)
        value_rows=batch*queries*count
        response=scalar if reader.thermal_output is not None else None
    elif backend=='packed':
        context,scalar,messages=_selected_source_order_reduce(*args,indices,alpha,gate,return_messages)
        value_rows=indices.shape[0]
        response=None
        if reader.thermal_output is not None:
            # Only scalar original-source coefficients leave this pure region.
            response=qstates.new_zeros(batch,queries,count)
            b,q,n=indices.unbind(-1)
            response=response.index_put((b,q,n),scalar)
        if return_messages:
            grid=qstates.new_zeros(batch,queries,count,width)
            b,q,n=indices.unbind(-1)
            messages=grid.index_put((b,q,n),messages)
    else:
        raise ValueError('Continuous backend must be packed or dense.')
    counts=dict(value_rows=int(value_rows),selected_rows=int(indices.shape[0]),candidate_rows=batch*queries*count,
        score_rows=batch*queries*count,message_projection_rows=int(value_rows),
        scalar_projection_rows=int(value_rows) if reader.thermal_output is not None else 0,
        fourier_entries=int(value_rows)*7*reader.coordinate_dim,hidden_entries=int(value_rows)*width,
        available_present_rows=int((valid[:,None]&qmask[...,None]).sum()),
        reduction_scratch_bytes=(batch*queries*count*width*qstates.element_size()
                                 if backend=='packed' and indices.shape[0] else 0),
        padded_rows=0,invalid_candidate_rows_available=0,dense_value_rows=int(value_rows) if backend=='dense' else 0,
        checkpoint_forward_rows=0,checkpoint_recomputation_rows_available=0)
    return TypedRead(context.reshape(batch,queries,width),alpha,response,
                     messages if return_messages else None,counts)
