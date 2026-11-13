"""Input-only eligible family scheduling for R16-10."""
from __future__ import annotations

import copy
import numpy as np


class EligibleFamilySchedule:
    """Separate deterministic source permutations and resumable presentation counts."""
    def __init__(self, original, new, native_ids, *, seed=0, preserve_order=False):
        native = set(map(str, native_ids))
        def eligible(values):
            values = tuple(map(str, values))
            if preserve_order:
                if len(set(values)) != len(values):
                    raise ValueError('Eligible family IDs must be unique.')
                return tuple(value for value in values if value in native)
            return tuple(sorted(set(values) & native))
        self.families = {
            'original': eligible(original),
            'new': eligible(new),
        }
        if any(not ids for ids in self.families.values()):
            raise ValueError('Both eligible FIT sources must be nonempty.')
        self.order = {}
        for index, (source, ids) in enumerate(self.families.items()):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), index]))
            self.order[source] = tuple(ids[i] for i in rng.permutation(len(ids)))
        self.cursor = {source: 0 for source in self.order}
        self.counts = {source: {family: 0 for family in ids} for source, ids in self.families.items()}
        self.last_epoch = 10
        self.current = ()

    def epoch(self, epoch, counts):
        epoch = int(epoch)
        if epoch <= 10:
            return ()
        if epoch == self.last_epoch:
            return self.current
        if epoch != self.last_epoch + 1:
            raise ValueError('Family sampler must advance exactly one epoch or restore its saved state.')
        result = []
        for source in ('original', 'new'):
            count = int(counts[source])
            if count < 1 or count > len(self.order[source]):
                raise ValueError('Invalid source family count.')
            for _ in range(count):
                family = self.order[source][self.cursor[source] % len(self.order[source])]
                self.cursor[source] += 1
                self.counts[source][family] += 1
                result.append((source, family))
        self.last_epoch, self.current = epoch, tuple(result)
        return self.current

    def state_dict(self):
        return copy.deepcopy({'order': self.order, 'cursor': self.cursor, 'counts': self.counts,
                              'last_epoch': self.last_epoch, 'current': self.current})

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        if {k: tuple(v) for k, v in state['order'].items()} != self.order:
            raise ValueError('Saved eligible family ordering differs.')
        if set(state['cursor']) != set(self.order) or set(state['counts']) != set(self.order):
            raise ValueError('Invalid family sampler state.')
        for source in self.order:
            if set(state['counts'][source]) != set(self.order[source]) or sum(state['counts'][source].values()) != state['cursor'][source]:
                raise ValueError('Family exposure counts disagree with cursor.')
        self.cursor, self.counts = state['cursor'], state['counts']
        self.last_epoch = int(state['last_epoch'])
        self.current = tuple(tuple(row) for row in state['current'])
