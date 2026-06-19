"""CORE HONF package.

This package exposes reusable hypergraph neural-field types, organizer,
decoder, and core model classes. Inputs and outputs are generic tensors, so the
package can be reused across domains.
"""

from .honf_core import HONFNeuralField, UnifiedHypergraphNeuralField
from .honf_types import BatchData, UnifiedForwardConfig

__all__ = ["BatchData", "HONFNeuralField", "UnifiedForwardConfig", "UnifiedHypergraphNeuralField"]
