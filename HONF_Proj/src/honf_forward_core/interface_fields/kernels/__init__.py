"""Optional fused kernels for the case-neutral interface readers.

The package deliberately has no hard Triton dependency.  Importing the
package on a CPU-only installation exposes the availability query and the
readable reference implementation; the fused entry points report an explicit
runtime error when a CUDA/Triton execution is requested but unavailable.
"""

from .qe_triton import (
    TRITON_AVAILABLE,
    fused_qe_reader,
    is_triton_qe_available,
    qe_reader_reference,
)

__all__ = [
    "TRITON_AVAILABLE",
    "fused_qe_reader",
    "is_triton_qe_available",
    "qe_reader_reference",
]
