"""Shared full/restricted reader-mode selection and scoped restoration."""
from contextlib import contextmanager


def validate_read_mode(mode):
    if mode not in ('full', 'restricted'):
        raise ValueError("Read execution must be 'full' or 'restricted'.")
    return mode


def resolve_read_mode(core, *, default='restricted'):
    """Resolve a case's default with an enclosing core execution override."""
    return validate_read_mode(getattr(core, '_read_execution_override', None) or default)


@contextmanager
def read_execution(core, mode):
    """Temporarily select member access and restore nested scopes on exit."""
    validate_read_mode(mode)
    previous = core._read_execution_override
    core._read_execution_override = mode
    try:
        yield core
    finally:
        core._read_execution_override = previous
