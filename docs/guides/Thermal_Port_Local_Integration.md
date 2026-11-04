# Thermal native port and local integration

The P64 composition prepares a single 64-angle native query union per design. Each full or restricted field decode also supplies the outside-temperature ports for the frozen local operator; the local result supplies interface and material readouts while the direct field remains available separately. Sixteen-angle native observations select every fourth angle within each original module.

The maintained builder is `channelthermal.training.thermal_port_response_task.build_thermal_port_response_task`. It accepts the historical `thermal_port_response_v1` recipe for the C0 and P64 arms and strictly transfers the declared R14 source, matching dataset membership, model configuration and TRAIN normalization. P64's teacher outside temperature is used for supervision; physical input derivatives pass through the frozen local operator. This integration does not register feedback or response-fitting arms.

`load_thermal_port_response_checkpoint(path, device=..., local_checkpoint=...)` returns the native model, frozen local operator and saved checkpoint. The local asset defaults to the path saved in the recipe; callers may explicitly supply its relocated path. The local checkpoint must be the declared epoch-4975 asset, and its weights are frozen. The loader restores the core clock to the source epoch plus the literal saved epoch. A checkpoint and local asset are both required; neither is bundled here.

Historical C0/P64 checkpoints retain their saved recipe and identity. These development assets and their local operator have prior data exposure; their inclusion does not establish physical response accuracy or independent test performance.
