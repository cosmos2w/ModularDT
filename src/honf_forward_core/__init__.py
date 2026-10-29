"""CORE HONF package.

This package exposes reusable hypergraph neural-field types, organizer,
decoder, and core model classes. Inputs and outputs are generic tensors, so the
package can be reused across domains.
"""

from .model import HONFNeuralField
from .config import BatchData, UnifiedForwardConfig

__all__ = ["BatchData", "HONFNeuralField", "UnifiedForwardConfig"]
