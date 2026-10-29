"""Training utilities that depend only on generic HONF tensors."""

from .diagnostics import (
    HONF_DIAGNOSTIC_KEYS,
    compute_code_permutation_equivariance_diagnostics,
    compute_honf_diagnostics,
    organizer_regularization_loss,
)
from .losses import weighted_channel_mse

__all__ = [
    "HONF_DIAGNOSTIC_KEYS",
    "compute_code_permutation_equivariance_diagnostics",
    "compute_honf_diagnostics",
    "organizer_regularization_loss",
    "weighted_channel_mse",
]
