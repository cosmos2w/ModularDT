"""Trainer lifecycle checks use a tiny linear toy, never an HONF training run."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pytest
import torch
from honf_runtime.field_training import fit, load_checkpoint, seed_all


@dataclass
class ToyBatch:
    scene: torch.Tensor
    targets: torch.Tensor
    query_weights: torch.Tensor
    def to(self, device):
        return ToyBatch(self.scene.to(device), self.targets.to(device), self.query_weights.to(device))


class ToyTask:
    path = None
    def __init__(self, identity='toy'):
        self.key = identity
    def dataset(self, split, epoch):
        return [(torch.tensor([[float(i), 1.]]), torch.tensor([[2.*i + 1.]])) for i in range(3)]
    def collate(self, rows):
        return ToyBatch(torch.stack([x for x,y in rows]), torch.stack([y for x,y in rows]), torch.ones(len(rows),1))
    def predict(self, model, batch):
        return model(batch.scene)
    def identity(self):
        return {'dataset': self.key}
    def physical_scale(self):
        return np.zeros(1), np.ones(1)


def _config(epochs):
    return {'model': {'name': 'toy'}, 'training': {'epochs': epochs, 'seed': 3, 'batch_size': 2,
            'learning_rate': .01, 'weight_decay': 0., 'gradient_clip_norm': 1., 'progress': False},
            'storage': {'retained_budget_gib': .03, 'minimum_free_gib': .001}}


def _model():
    seed_all(3)
    return torch.nn.Linear(2,1)


def test_complete_epoch_resume_matches_uninterrupted(tmp_path):
    full, split = tmp_path/'full', tmp_path/'split'
    config = _config(2)
    fit(_model(), ToyTask(), config['training'], output=full, config=config)
    first = _config(1)
    fit(_model(), ToyTask(), first['training'], output=split, config=first)
    fit(_model(), ToyTask(), config['training'], output=split, config=config, resume=split/'latest.pt')
    a, b = load_checkpoint(full/'latest.pt'), load_checkpoint(split/'latest.pt')
    assert a['epoch']==b['epoch']==2
    assert [r['train_cases'] for r in b['history'][1:]]==[3,3]
    for k in a['model']:
        torch.testing.assert_close(a['model'][k], b['model'][k], rtol=0, atol=0)
    assert (split/'loss_curve.pdf').read_bytes().startswith(b'%PDF')
    assert (split/'best_field.pt').exists()
    assert not list(split.glob('epoch_*.pt'))


def test_resume_rejects_different_data_without_overwriting_checkpoint(tmp_path):
    run=tmp_path/'owned';config=_config(1)
    fit(_model(),ToyTask(),config['training'],output=run,config=config)
    before=(run/'latest.pt').read_bytes()
    later=_config(2)
    with pytest.raises(ValueError,match='same dataset'):
        fit(_model(),ToyTask('different'),later['training'],output=run,config=later,resume=run/'latest.pt')
    assert (run/'latest.pt').read_bytes()==before
