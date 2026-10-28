"""Input-only Standard panel for reproducible routed-model qualification."""
import json
import h5py
import numpy as np


def select_panel(path, split, limit):
    if split not in ('train', 'dev') or limit > {'train': 32, 'dev': 16}[split] or limit < 1:
        raise ValueError('qualification selects at most 32 TRAIN or 16 DEV cases')
    with h5py.File(path, 'r') as h:
        plan = json.loads(h['provenance/formal_plan_json'].asstr()[()])
        indices = h[f'splits/{split}'][:]
        ids = h['metadata/state_id'].asstr()[:]
        roots = h['metadata/environment_root_id'].asstr()[:]
        groups = {}
        for local, index in enumerate(indices):
            mask = h['inputs/contact_mask'][index]
            records = h['inputs/environment_records'][index, :, 5:8]
            contrast = float(np.max(np.log(records).max(axis=0) - np.log(records).min(axis=0)))
            family = plan['environments'][roots[index]]['family']
            groups.setdefault((int(mask.sum()), family), []).append((contrast, ids[index], local))
    # Repeatedly sample low/high contrast from each count/family stratum.
    for values in groups.values():
        values.sort()
    selected = []
    counts = sorted({count for count, _ in groups})
    families = sorted({family for _, family in groups})
    order = [(count, families[(j + i) % len(families)])
             for j in range(len(families)) for i, count in enumerate(counts)]
    round_index = 0
    while len(selected) < limit and groups:
        for stratum_index, key in enumerate(order):
            if key not in groups:
                continue
            values = groups[key]
            selected.append(values.pop(0 if (round_index + stratum_index) % 2 == 0 else -1))
            if not values:
                del groups[key]
            if len(selected) == limit:
                break
        round_index += 1
    return sorted((int(local), str(state)) for _, state, local in selected)
