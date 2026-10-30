"""Round13-3 applied-interaction and original-member readout core."""

from .core import InteractionJointOutput, InteractionReusedMemberRoutingCore
from .anchored import (
    AnchoredMemberRoutingCore,
    EAccessDirectory,
    EAccessRow,
    PackedEAccessRows,
    anchored_row_read,
    blend_anchored_contexts,
    compile_e_access_directory,
)
from .detail import DetailAnchoredMemberRoutingCore


def anchored_core_from_config(config):
    """Restore the implementation identified by the saved configuration schema.

    Older anchored checkpoints retain their original parameter inventory and
    read semantics. New detail-preserving checkpoints use the same public
    families, with their own strictly validated saved schema.
    """
    cores = {
        AnchoredMemberRoutingCore.schema: AnchoredMemberRoutingCore,
        DetailAnchoredMemberRoutingCore.schema: DetailAnchoredMemberRoutingCore,
    }
    core = cores.get(config.get("schema"))
    if core is None:
        raise ValueError("Unsupported anchored routing configuration schema.")
    return core.from_config(config)

__all__ = ["InteractionJointOutput", "InteractionReusedMemberRoutingCore", "AnchoredMemberRoutingCore",
           "DetailAnchoredMemberRoutingCore", "anchored_core_from_config",
           "EAccessDirectory", "EAccessRow", "PackedEAccessRows", "anchored_row_read",
           "blend_anchored_contexts", "compile_e_access_directory"]
