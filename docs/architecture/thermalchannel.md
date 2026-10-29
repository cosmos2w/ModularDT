# ThermalChannel ownership

ThermalChannel owns physical feature preparation, Stage-A local coupling,
outside-temperature refinement, physical losses, field metrics, and inverse
integration. `ChannelThermalHONFModel` keeps its public forward and prepared
decode contracts; non-registering support helpers live in `model_support.py`.

The general HONF core has no ThermalChannel imports. The wrapper still performs
the same base organizer, local response, refinement, final organizer, and field
decode sequence used by Runs 1000 and 1401.
