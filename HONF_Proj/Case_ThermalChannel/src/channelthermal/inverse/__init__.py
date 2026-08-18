"""ThermalChannel data contracts and physics for hierarchical inverse design."""

from .compact_plan import (
    COMPACT_PLAN_FEATURE_NAMES,
    COMPACT_PLAN_SCHEMA_NAME,
    COMPACT_PLAN_SCHEMA_VERSION,
    extract_compact_plan,
    validate_compact_plan,
)
from .context import CONTEXT_FEATURE_NAMES, load_context, parse_context
from .dataset_builder import CaseBuildRecord, build_inverse_dataset_from_records
from .dataset_io import InverseH5Dataset, validate_inverse_hdf5
from .geometry import canonicalize_design, evaluate_geometry
from .request import make_request_codec
from .topology_set import (
    TopologySetPlan,
    extract_topology_set,
    topology_set_dataset_arrays,
    topology_set_dataset_attributes,
)
from .verifier import FrozenThermalChannelVerifier
from .vocabulary import REQUEST_TYPES

__all__ = [
    "COMPACT_PLAN_FEATURE_NAMES",
    "COMPACT_PLAN_SCHEMA_NAME",
    "COMPACT_PLAN_SCHEMA_VERSION",
    "CONTEXT_FEATURE_NAMES",
    "CaseBuildRecord",
    "FrozenThermalChannelVerifier",
    "InverseH5Dataset",
    "REQUEST_TYPES",
    "TopologySetPlan",
    "canonicalize_design",
    "build_inverse_dataset_from_records",
    "evaluate_geometry",
    "extract_compact_plan",
    "extract_topology_set",
    "load_context",
    "make_request_codec",
    "parse_context",
    "topology_set_dataset_arrays",
    "topology_set_dataset_attributes",
    "validate_compact_plan",
    "validate_inverse_hdf5",
]
