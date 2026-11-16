"""The optional wall budget clock must round-trip without changing state."""
from channelthermal.training import contextual_joint_task as contextual


def test_optional_clock_restores_absence_and_preserves_existing_elapsed(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(contextual.time, 'monotonic', lambda: clock[0])
    task = contextual.ThermalContextualJointTask.__new__(contextual.ThermalContextualJointTask)
    task.native_only = True
    task.organizer = None
    task.training_samples = task.validation_samples = 0
    task._budget_binding = None
    task._segment_started = 10.0
    task.state = {}
    saved = {'primary_updates': 3, 'training_samples': 7, 'validation_samples': 2}
    task.load_training_state_dict(saved)
    assert task.training_state_dict() == saved

    task.state['training_elapsed_seconds'] = 12.0
    task.load_training_state_dict({**saved, 'training_elapsed_seconds': 8.0})
    assert task.training_state_dict()['training_elapsed_seconds'] == 12.0

    task.bind_wall_clock_budget(output_dir='/tmp', budget_seconds=100)
    clock[0] = 13.0
    task.load_training_state_dict({**saved, 'training_elapsed_seconds': 14.0})
    assert task.training_state_dict()['training_elapsed_seconds'] == 15.0
