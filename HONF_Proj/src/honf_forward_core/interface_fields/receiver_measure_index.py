"""Canonical geometry/role index view; fine physical atoms remain untouched."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReceiverMeasureIndex:
    coordinates: torch.Tensor
    weights: torch.Tensor
    roles: torch.Tensor
    atom_to_block: torch.Tensor
    block_atoms: tuple[tuple[int, ...], ...]


def canonical_receiver_index(coordinates, weights, roles):
    """Group exact coordinate/role ties and retain differentiable block mass.

    Membership is discrete and input-only. Coordinates use mass-weighted
    means so refinement gradients pull back by the declared child masses.
    No source state is coarsened or removed from the native fine reader.
    """
    values = coordinates.detach().cpu().tolist()
    role_values = roles.detach().cpu().tolist()
    blocks = {}
    for atom, (point, role) in enumerate(zip(values, role_values)):
        blocks.setdefault((*point, role), []).append(atom)
    block_atoms = tuple(tuple(blocks[key]) for key in sorted(blocks))
    assignments = [0] * len(values)
    for block, atoms in enumerate(block_atoms):
        for atom in atoms:
            assignments[atom] = block
    atom_to_block = torch.tensor(assignments, dtype=torch.long, device=weights.device)
    representatives = torch.tensor([atoms[0] for atoms in block_atoms], device=weights.device)
    count, dimension = len(block_atoms), coordinates.shape[-1]
    masses = weights.new_zeros(count).scatter_add(0, atom_to_block, weights)
    moment = coordinates.new_zeros(count, dimension).scatter_add(
        0, atom_to_block[:, None].expand(-1, dimension), coordinates * weights[:, None])
    weighted = moment / masses[:, None]
    # Equal stored coordinates stay bit-identical as geometric keys;
    # backward distributes coordinate sensitivity by child measure.
    centers = weighted + (coordinates[representatives] - weighted).detach()
    return ReceiverMeasureIndex(centers, masses, roles[representatives], atom_to_block, block_atoms)


__all__ = ["ReceiverMeasureIndex", "canonical_receiver_index"]
